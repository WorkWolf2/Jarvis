"""
Ollama LLM Client - Interface to locally running Ollama models.
Supports streaming, conversation history, model fallback, and retry with backoff.
"""

import json
import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator
from core.logger import get_logger
from core.config_loader import ConfigLoader

logger = get_logger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed. Run: pip install httpx")

# Default timeouts — LLMs on CPU can be very slow
CONNECT_TIMEOUT = 10      # seconds to establish connection
READ_TIMEOUT    = 300     # seconds to wait for response (5 min — enough for any local model)
MAX_RETRIES     = 2       # number of retry attempts on timeout
RETRY_DELAY     = 2.0     # seconds between retries


class OllamaClient:
    """
    Async client for Ollama API.
    Handles model availability, fallback, streaming, and automatic retry on timeout.
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self.base_url = config.get("llm.base_url", "http://localhost:11434")
        self.model = config.get("llm.model", "llama3")
        # read_timeout overrides the old single "timeout" value
        self.read_timeout  = config.get("llm.read_timeout",  READ_TIMEOUT)
        self.connect_timeout = config.get("llm.connect_timeout", CONNECT_TIMEOUT)
        self.temperature   = config.get("llm.temperature", 0.7)
        self.max_tokens    = config.get("llm.max_tokens", 2048)
        self.fallback_models = config.get("llm.fallback_models", [])
        self.max_retries   = config.get("llm.max_retries", MAX_RETRIES)
        self._active_model: Optional[str] = None
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required. Run: pip install httpx")
        if self._client is None:
            # Use separate connect / read timeouts so a slow model doesn't
            # appear as a connection problem.
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=self.connect_timeout,
                    read=self.read_timeout,
                    write=30.0,
                    pool=5.0,
                )
            )
        return self._client

    async def check_connection(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            available_models = [m["name"].split(":")[0] for m in data.get("models", [])]
            available_full = [m["name"] for m in data.get("models", [])]

            logger.debug(f"Available Ollama models: {available_full}")

            # Try primary model
            if self.model in available_models or self.model in available_full:
                self._active_model = self.model
                return True

            # Try fallback models
            for fallback in self.fallback_models:
                if fallback in available_models or fallback in available_full:
                    logger.warning(
                        f"Model '{self.model}' not found, falling back to '{fallback}'"
                    )
                    self._active_model = fallback
                    return True

            # Model not found but Ollama is running
            logger.error(
                f"No suitable model found. Available: {available_models}. "
                f"Pull a model: ollama pull {self.model}"
            )
            # Still set the configured model and let it fail at generation time
            self._active_model = self.model
            return False

        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            return False

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> str:
        """
        Send a chat message and get a response.
        Automatically retries on timeout with exponential backoff.

        Args:
            system_prompt: The system/persona prompt
            user_message: The current user message
            history: Previous messages as [{"role": "user"/"assistant", "content": "..."}]
            stream: Whether to stream the response

        Returns:
            The assistant's response text
        """
        if not self._active_model:
            await self.check_connection()

        model = self._active_model or self.model

        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        last_error: Exception = None
        for attempt in range(self.max_retries + 1):
            try:
                # Re-create client on retry so connection state is fresh
                if attempt > 0:
                    await self._reset_client()
                    delay = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.info(
                        f"Retrying LLM request (attempt {attempt + 1}/{self.max_retries + 1}) "
                        f"after {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

                client = self._get_client()
                logger.debug(f"Calling Ollama: model={model}, messages={len(messages)}, "
                             f"read_timeout={self.read_timeout}s")

                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()

                data = response.json()
                content = data.get("message", {}).get("content", "")

                if not content:
                    logger.warning("Empty response from LLM")
                    return "I couldn't generate a response. Please try again."

                return content.strip()

            except httpx.ReadTimeout as e:
                last_error = e
                logger.warning(
                    f"Ollama ReadTimeout on attempt {attempt + 1} "
                    f"(read_timeout={self.read_timeout}s). "
                    f"{'Retrying...' if attempt < self.max_retries else 'All retries exhausted.'}"
                )
                if attempt >= self.max_retries:
                    break

            except httpx.ConnectError as e:
                last_error = e
                logger.error(f"Cannot connect to Ollama at {self.base_url}. "
                             "Is 'ollama serve' running?")
                break  # No point retrying a connection error

            except Exception as e:
                last_error = e
                logger.error(f"LLM request failed: {e}", exc_info=True)
                break

        # All attempts failed
        error_type = type(last_error).__name__ if last_error else "Unknown"
        if isinstance(last_error, httpx.ReadTimeout):
            return (
                f"The language model took too long to respond (timeout: {self.read_timeout}s). "
                f"Try a smaller/faster model, or increase 'llm.read_timeout' in config.json. "
                f"Current model: {model}"
            )
        elif isinstance(last_error, httpx.ConnectError):
            return (
                "Cannot connect to Ollama. Make sure it is running: "
                "open a terminal and run 'ollama serve'"
            )
        return f"LLM error ({error_type}): {last_error}"

    async def _reset_client(self) -> None:
        """Close and recreate the HTTP client (used between retries)."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Simple single-turn generation (no history), with retry."""
        payload = {
            "model": self._active_model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    await self._reset_client()
                    await asyncio.sleep(RETRY_DELAY * attempt)
                client = self._get_client()
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
            except httpx.ReadTimeout:
                if attempt >= self.max_retries:
                    logger.warning("generate() timed out after all retries")
                    return ""
            except Exception as e:
                logger.error(f"Generate failed: {e}")
                return ""
        return ""

    async def list_models(self) -> List[str]:
        """List all available local models."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            client = self._get_client()
            async with client.stream("POST", "/api/pull", json={"name": model_name}) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        logger.info(f"Pull status: {status}")
                        if status == "success":
                            return True
            return True
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None