"""
Core Orchestrator - Routes user input, calls LLM, parses actions, executes tools.
This is the central nervous system of JARVIS.
"""

import json
import asyncio
from datetime import datetime
from typing import Optional, Any
from pathlib import Path

from core.config_loader import ConfigLoader
from core.logger import get_logger
from core.router import ActionRouter
from llm.ollama_client import OllamaClient
from memory.memory import MemoryManager
from tools.registry import ToolRegistry
from self_improve.logger import InteractionLogger
from self_improve.analyzer import SelfImprovementAnalyzer
from self_improve.apply_patch import PatchManager
from safety.validator import SafetyValidator

logger = get_logger(__name__)


SYSTEM_PROMPT_TEMPLATE = """You are {name}, an advanced local AI assistant and system controller.

You are precise, intelligent, and capable of executing system-level actions.
You have access to tools that let you control the computer and perform tasks.

RESPONSE FORMAT:
- If you need to execute an ACTION, respond with ONLY a valid JSON object:
  {{"type": "action_name", "key": "value", ...}}
- If you need to CHAIN multiple actions, respond with a JSON array:
  [{{"type": "action1", ...}}, {{"type": "action2", ...}}]
- For CONVERSATION only (no action needed), respond with natural language text.
- NEVER mix JSON and natural language in the same response.

AVAILABLE ACTIONS:
{tool_descriptions}

CONTEXT:
{context}

RULES:
1. Only use actions from the AVAILABLE ACTIONS list above.
2. Always validate that required parameters are present before responding with an action.
3. If you cannot complete a request safely, explain why in plain text.
4. Be concise and precise. Avoid unnecessary words.
5. Remember past interactions and learn from them.
6. Always respond in the same language the user is using.
7. In conversational replies, always address the user as "signore".
8. When the user asks for online or current information, use the web_search action and include source references.
"""


class Orchestrator:
    """
    Central orchestrator that coordinates all JARVIS subsystems.
    Manages the lifecycle of user requests from input to output.
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self._initialized = False

        # Subsystems (initialized in initialize())
        self.llm: Optional[OllamaClient] = None
        self.memory: Optional[MemoryManager] = None
        self.tools: Optional[ToolRegistry] = None
        self.router: Optional[ActionRouter] = None
        self.safety: Optional[SafetyValidator] = None
        self.interaction_logger: Optional[InteractionLogger] = None
        self.analyzer: Optional[SelfImprovementAnalyzer] = None
        self.patch_manager: Optional[PatchManager] = None

        self._interaction_count = 0
        self._session_start = datetime.utcnow()

    async def initialize(self) -> None:
        """Initialize all subsystems."""
        logger.info("Initializing JARVIS subsystems...")

        project_root = Path(__file__).parent.parent

        # 1. Safety validator (first, always)
        self.safety = SafetyValidator(self.config)
        logger.info("✓ Safety validator ready")

        # 2. Memory system
        db_path = project_root / self.config.get("memory.db_path", "data/jarvis_memory.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryManager(db_path, self.config)
        await self.memory.initialize()
        logger.info("✓ Memory system ready")

        # 3. Tool registry
        self.tools = ToolRegistry(self.config)
        self.tools.auto_discover(project_root / "tools")
        logger.info(f"✓ Tool registry ready ({len(self.tools.list_tools())} tools loaded)")

        # 4. LLM client
        self.llm = OllamaClient(self.config)
        model_ok = await self.llm.check_connection()
        if not model_ok:
            logger.warning(
                f"Model '{self.config.get('llm.model')}' not available. "
                "Please ensure Ollama is running: ollama serve"
            )
        else:
            logger.info(f"✓ LLM client ready (model: {self.config.get('llm.model')})")

        # 5. Action router
        self.router = ActionRouter(self.tools, self.safety, self.config)
        logger.info("✓ Action router ready")

        # 6. Interaction logger
        log_path = project_root / self.config.get(
            "self_improve.log_path", "logs/interactions.jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.interaction_logger = InteractionLogger(log_path)
        logger.info("✓ Interaction logger ready")

        # 7. Self-improvement analyzer
        self.analyzer = SelfImprovementAnalyzer(
            self.interaction_logger,
            self.llm,
            self.config
        )

        # 8. Patch manager
        patches_dir = project_root / "data" / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        self.patch_manager = PatchManager(patches_dir, self.config)
        logger.info("✓ Self-improvement engine ready")

        self._initialized = True
        logger.info("JARVIS fully initialized ✓")

    async def process(self, user_input: str) -> str:
        """
        Main processing pipeline:
        1. Retrieve memory context
        2. Build prompt
        3. Call LLM
        4. Parse response
        5. Execute action or return text
        6. Log interaction
        """
        if not self._initialized:
            return "System not initialized. Please call initialize() first."

        self._interaction_count += 1
        interaction_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._interaction_count}"

        logger.debug(f"Processing input [{interaction_id}]: {user_input[:100]}")

        try:
            # 1. Get conversation context from memory
            context = await self.memory.get_context(max_messages=self.config.get(
                "memory.max_context_messages", 20
            ))

            # 2. Build system prompt
            system_prompt = self._build_system_prompt(context)

            # 3. Get conversation history for LLM
            history = await self.memory.get_recent_messages(
                n=self.config.get("llm.history_messages", 6)
            )

            # 4. Call LLM
            raw_response = await self.llm.chat(
                system_prompt=system_prompt,
                user_message=user_input,
                history=history
            )

            logger.debug(f"LLM raw response: {raw_response[:200]}")

            # 5. Parse response (JSON action or plain text)
            parsed = self._parse_response(raw_response)

            # 6. Execute if action
            final_response = raw_response
            action_result = None
            execution_success = True

            if parsed is not None:
                action_result, execution_success = await self.router.route(parsed, user_input)
                final_response = action_result

            # 7. Store in memory
            await self.memory.add_message("user", user_input)
            await self.memory.add_message("assistant", final_response)

            # 8. Log interaction
            await self.interaction_logger.log({
                "id": interaction_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "llm_response": raw_response,
                "parsed_action": parsed if isinstance(parsed, dict) else str(parsed),
                "final_response": final_response,
                "success": execution_success,
                "model": self.config.get("llm.model")
            })

            # 9. Check if analysis should run
            await self._maybe_run_analysis()

            return final_response

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            error_msg = f"I encountered an error processing your request: {str(e)}"

            await self.interaction_logger.log({
                "id": interaction_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "error": str(e),
                "success": False
            })

            return error_msg

    def _parse_response(self, response: str) -> Optional[Any]:
        """
        Try to parse LLM response as JSON action(s).
        Returns parsed JSON or None if plain text.
        """
        response = response.strip()

        # Try to extract JSON if mixed with text
        if response.startswith("{") or response.startswith("["):
            try:
                parsed = json.loads(response)
                # Validate basic structure
                if isinstance(parsed, dict) and "type" in parsed:
                    return parsed
                elif isinstance(parsed, list) and all(
                    isinstance(a, dict) and "type" in a for a in parsed
                ):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try to extract JSON block from response
        import re
        json_pattern = r'\{[^{}]*"type"[^{}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        # Plain text response
        return None

    def _build_system_prompt(self, context: str) -> str:
        """Build the system prompt with tool descriptions and context."""
        tool_descriptions = self.tools.get_descriptions_for_prompt()
        name = self.config.get("assistant.name", "JARVIS")

        return SYSTEM_PROMPT_TEMPLATE.format(
            name=name,
            tool_descriptions=tool_descriptions,
            context=context or "No previous context."
        )

    async def startup_message(self) -> None:
        """Print startup greeting."""
        name = self.config.get("assistant.name", "JARVIS")
        model = self.config.get("llm.model", "unknown")
        tool_count = len(self.tools.list_tools()) if self.tools else 0
        print(f"{name}: All systems online. Model: {model}. "
              f"{tool_count} tools loaded. How can I assist you?")

    async def print_status(self) -> None:
        """Print current system status."""
        name = self.config.get("assistant.name", "JARVIS")
        uptime = datetime.utcnow() - self._session_start

        memory_stats = await self.memory.get_stats()
        tool_list = self.tools.list_tools()
        pending_patches = self.patch_manager.list_pending()

        print(f"""
{name} System Status
{'='*40}
Uptime:          {str(uptime).split('.')[0]}
Interactions:    {self._interaction_count}
Model:           {self.config.get('llm.model')}
Safety:          {'ENABLED' if self.config.get('safety.enabled') else 'DISABLED'}

Memory:
  Conversations: {memory_stats.get('conversation_count', 0)}
  Messages:      {memory_stats.get('message_count', 0)}

Tools ({len(tool_list)} loaded):
  {', '.join(tool_list)}

Pending Patches: {len(pending_patches)}
{'='*40}
        """)

    async def run_analysis(self) -> None:
        """Trigger self-improvement analysis."""
        print("\nRunning self-improvement analysis...")
        patches = await self.analyzer.analyze()

        if not patches:
            print("No improvements suggested at this time.")
            return

        print(f"\nGenerated {len(patches)} improvement proposals:")
        for i, patch in enumerate(patches, 1):
            print(f"\n[{i}] {patch.get('id', 'unknown')}")
            print(f"    Type: {patch.get('type', 'unknown')}")
            print(f"    Description: {patch.get('description', 'N/A')}")

        saved = self.patch_manager.save_patches(patches)
        print(f"\n{len(saved)} patches saved. Use 'approve patch <id>' to apply.")

    async def approve_patch(self, patch_id: str) -> None:
        """Approve and apply a specific patch."""
        result = await self.patch_manager.apply_patch(patch_id)
        if result.success:
            print(f"Patch {patch_id} applied successfully: {result.message}")
        else:
            print(f"Patch {patch_id} failed: {result.message}")

    async def list_pending_patches(self) -> None:
        """List all pending improvement patches."""
        patches = self.patch_manager.list_pending()
        if not patches:
            print("No pending patches.")
            return

        print(f"\nPending patches ({len(patches)}):")
        for patch in patches:
            print(f"  ID: {patch.get('id')}")
            print(f"  Type: {patch.get('type')}")
            print(f"  Description: {patch.get('description', 'N/A')}")
            print()

    async def show_memory_stats(self) -> None:
        """Show memory statistics."""
        stats = await self.memory.get_stats()
        print(f"\nMemory Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    async def _maybe_run_analysis(self) -> None:
        """Check if it's time to run analysis."""
        if not self.config.get("self_improve.enabled", True):
            return

        min_interactions = self.config.get(
            "self_improve.min_interactions_before_analysis", 10
        )
        interval = self.config.get("self_improve.analysis_interval_minutes", 60)

        if self._interaction_count % max(min_interactions, 1) == 0:
            logger.info("Scheduling background self-improvement analysis...")
            asyncio.create_task(self._background_analysis())

    async def _background_analysis(self) -> None:
        """Run analysis in background without blocking."""
        try:
            patches = await self.analyzer.analyze()
            if patches:
                self.patch_manager.save_patches(patches)
                logger.info(f"Background analysis generated {len(patches)} patches")
        except Exception as e:
            logger.error(f"Background analysis failed: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown all subsystems."""
        logger.info("Shutting down JARVIS...")
        if self.memory:
            await self.memory.close()
        logger.info("JARVIS shutdown complete.")