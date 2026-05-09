"""
Tool Registry - Dynamic plugin system for JARVIS tools.
Auto-discovers and registers tools from the tools directory.
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Optional, List, Type

from tools.base_tool import BaseTool
from core.logger import get_logger
from core.config_loader import ConfigLoader

logger = get_logger(__name__)


class ToolRegistry:
    """
    Dynamic tool registry that auto-discovers and manages tool plugins.

    Tools are Python classes that:
    1. Inherit from BaseTool
    2. Define a unique `name` class attribute
    3. Implement async `execute(action: dict) -> ToolResult`
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self._tools: Dict[str, BaseTool] = {}
        self._disabled = set(config.get("tools.disabled_tools", []))

    def register(self, tool_class: Type[BaseTool]) -> bool:
        """Register a tool class."""
        try:
            if not tool_class.name:
                logger.warning(f"Tool {tool_class.__name__} has no name, skipping")
                return False

            if tool_class.name in self._disabled:
                logger.debug(f"Tool '{tool_class.name}' is disabled, skipping")
                return False

            instance = tool_class()
            self._tools[tool_class.name] = instance
            logger.debug(f"Registered tool: {tool_class.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register tool {tool_class}: {e}")
            return False

    def auto_discover(self, tools_dir: Path) -> int:
        """
        Auto-discover and register all tools in the given directory.
        Returns number of tools loaded.
        """
        loaded = 0
        tools_dir = Path(tools_dir)

        if not tools_dir.exists():
            logger.warning(f"Tools directory not found: {tools_dir}")
            return 0

        for py_file in tools_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "base_tool.py":
                continue

            module_name = f"tools.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and obj.name  # Has a name
                    ):
                        if self.register(obj):
                            loaded += 1

            except Exception as e:
                logger.error(f"Failed to load tool module {module_name}: {e}")

        logger.info(f"Auto-discovered {loaded} tools from {tools_dir}")
        return loaded

    def get_tool(self, action_type: str) -> Optional[BaseTool]:
        """Get a tool by action type."""
        return self._tools.get(action_type)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[dict]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_descriptions_for_prompt(self) -> str:
        """Format tool descriptions for the system prompt."""
        if not self._tools:
            return "No tools available."

        lines = []
        for name, tool in self._tools.items():
            params = tool.parameters.get("properties", {})
            param_str = ", ".join(
                f'"{k}": {v.get("type", "any")}' for k, v in params.items()
            )
            lines.append(f'- {{"type": "{name}", {param_str}}} — {tool.description}')

        return "\n".join(lines)

    def disable_tool(self, name: str) -> None:
        """Disable a tool at runtime."""
        self._disabled.add(name)
        self._tools.pop(name, None)

    def enable_tool(self, name: str) -> None:
        """Re-enable a disabled tool."""
        self._disabled.discard(name)

    def get_tool_stats(self) -> Dict[str, dict]:
        """Get metadata for all tools."""
        return {
            name: {
                "description": tool.description,
                "requires_confirmation": tool.requires_confirmation,
                "is_destructive": tool.is_destructive,
            }
            for name, tool in self._tools.items()
        }