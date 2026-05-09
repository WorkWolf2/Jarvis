"""
Web Tools - HTTP requests and web search (local/offline when possible).
These tools require network access and are disabled in fully offline mode.
"""

import asyncio
import json
from typing import ClassVar, Dict, Any, Optional
from urllib.parse import urlencode

from tools.base_tool import BaseTool, ToolResult
from core.logger import get_logger

logger = get_logger(__name__)


class HttpGetTool(BaseTool):
    """Make an HTTP GET request to a URL."""
    name: ClassVar[str] = "http_get"
    description: ClassVar[str] = "Make an HTTP GET request and return the response body"
    requires_confirmation: ClassVar[bool] = True
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to request"
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers"
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 10)"
            }
        }
    }

    # Allowed URL schemes and blocked hosts
    BLOCKED_HOSTS = {
        "localhost", "127.0.0.1", "0.0.0.0",
        "169.254.169.254",  # AWS metadata
        "::1"
    }

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        url = action["url"]
        headers = action.get("headers", {})
        timeout = float(action.get("timeout", 10))

        # Safety: block internal/metadata URLs
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.hostname in self.BLOCKED_HOSTS:
            return ToolResult.fail(f"Blocked: cannot request internal host '{parsed.hostname}'")

        if parsed.scheme not in ("http", "https"):
            return ToolResult.fail(f"Only http/https URLs allowed, got: {parsed.scheme}")

        try:
            import httpx
        except ImportError:
            return ToolResult.fail("httpx not installed. Run: pip install httpx")

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)

            # Limit response size
            content = response.text[:5000]
            if len(response.text) > 5000:
                content += "\n... [truncated at 5000 chars]"

            return ToolResult.ok(
                f"HTTP {response.status_code} from {url}:\n{content}",
                data={"status": response.status_code, "body": content}
            )

        except Exception as e:
            return ToolResult.fail(f"HTTP request failed: {str(e)}")


class WebSearchTool(BaseTool):
    """
    Search the web using a configured local or API search engine.
    Defaults to DuckDuckGo Lite (no API key needed).
    """
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Search the web for information using DuckDuckGo"
    requires_confirmation: ClassVar[bool] = False
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)"
            }
        }
    }

    DDG_URL = "https://html.duckduckgo.com/html/"

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        query = action["query"]
        max_results = min(int(action.get("max_results", 5)), 10)

        try:
            import httpx
        except ImportError:
            return ToolResult.fail("httpx not installed. Run: pip install httpx")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; JARVIS/1.0)",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = urlencode({"q": query, "kl": "wt-wt"})

            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                response = await client.post(
                    self.DDG_URL,
                    content=data,
                    headers=headers
                )

            # Simple HTML parsing without dependencies
            results = self._parse_ddg_results(response.text, max_results)

            if not results:
                return ToolResult.ok(f"No results found for: {query}")

            output = f"Search results for '{query}':\n\n"
            for i, r in enumerate(results, 1):
                snippet = r["snippet"] or "No snippet available."
                output += f"[{i}] {r['title']} - {snippet}\n"

            output += "\nSources:\n"
            for i, r in enumerate(results, 1):
                output += f"[{i}] {r['url']}\n"

            return ToolResult.ok(output.strip(), data=results)

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolResult.fail(f"Search failed: {str(e)}")

    def _parse_ddg_results(self, html: str, max_results: int) -> list:
        """Very simple HTML result extraction without BeautifulSoup."""
        import re
        results = []

        # Extract result blocks
        result_pattern = r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'

        urls = re.findall(result_pattern, html, re.DOTALL)
        snippets = re.findall(snippet_pattern, html, re.DOTALL)

        for i, (url, title) in enumerate(urls[:max_results]):
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
            if url and clean_title:
                results.append({
                    "title": clean_title,
                    "url": url,
                    "snippet": snippet[:200]
                })

        return results


class GetTimeTool(BaseTool):
    """Get the current date and time."""
    name: ClassVar[str] = "get_time"
    description: ClassVar[str] = "Get the current date and time"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name (e.g. 'Europe/Rome', default: local)"
            },
            "format": {
                "type": "string",
                "description": "Format string (default: readable format)"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        from datetime import datetime
        import time

        tz_name = action.get("timezone", "")
        fmt = action.get("format", "%A, %B %d %Y at %H:%M:%S")

        try:
            if tz_name:
                try:
                    from zoneinfo import ZoneInfo
                    tz = ZoneInfo(tz_name)
                    now = datetime.now(tz)
                except Exception:
                    now = datetime.now()
                    tz_name = "local"
            else:
                now = datetime.now()
                tz_name = "local"

            formatted = now.strftime(fmt)
            return ToolResult.ok(
                f"Current time ({tz_name}): {formatted}",
                data={"timestamp": now.isoformat(), "timezone": tz_name}
            )
        except Exception as e:
            return ToolResult.fail(f"Failed to get time: {str(e)}")


class CalculateTool(BaseTool):
    """Safely evaluate a mathematical expression."""
    name: ClassVar[str] = "calculate"
    description: ClassVar[str] = "Evaluate a safe mathematical expression"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["expression"],
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g. '2 + 2 * 10')"
            }
        }
    }

    # Safe builtins for eval
    SAFE_GLOBALS = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "len": len,
    }

    async def execute(self, action: dict) -> ToolResult:
        import math
        import re

        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        expr = action["expression"].strip()

        # Safety: only allow safe characters
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,\%\^a-z_]+$', expr, re.IGNORECASE):
            return ToolResult.fail(
                "Expression contains unsafe characters. "
                "Only numbers, basic operators (+,-,*,/,**,%), and parentheses allowed."
            )

        # Add math functions to safe globals
        safe_globals = {**self.SAFE_GLOBALS}
        for name in dir(math):
            if not name.startswith("_"):
                safe_globals[name] = getattr(math, name)

        try:
            # Replace ^ with ** for convenience
            expr_safe = expr.replace("^", "**")
            result = eval(expr_safe, safe_globals, {})  # noqa: S307
            return ToolResult.ok(
                f"{expr} = {result}",
                data={"expression": expr, "result": result}
            )
        except ZeroDivisionError:
            return ToolResult.fail("Division by zero")
        except Exception as e:
            return ToolResult.fail(f"Calculation error: {str(e)}")