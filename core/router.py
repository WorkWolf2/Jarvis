"""
Action Router - Routes parsed JSON actions to the correct tool.
Handles action validation, chaining, and error recovery.
"""

import json
from typing import Any, Optional, Tuple, List
from core.logger import get_logger
from core.config_loader import ConfigLoader
from tools.registry import ToolRegistry
from safety.validator import SafetyValidator

logger = get_logger(__name__)


class ActionRouter:
    """
    Routes parsed JSON action objects to their corresponding tools.
    Supports single actions, chained actions (arrays), and error recovery.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        safety: SafetyValidator,
        config: ConfigLoader
    ) -> None:
        self.tools = tool_registry
        self.safety = safety
        self.config = config

    async def route(
        self,
        action: Any,
        user_context: str = ""
    ) -> Tuple[str, bool]:
        """
        Route an action (or list of actions) to the appropriate tool.

        Returns:
            Tuple[str, bool]: (response_message, success)
        """
        try:
            # Handle action chains (array of actions)
            if isinstance(action, list):
                return await self._route_chain(action, user_context)

            # Single action
            if isinstance(action, dict):
                return await self._route_single(action, user_context)

            return f"Invalid action format: {type(action)}", False

        except Exception as e:
            logger.error(f"Router error: {e}", exc_info=True)
            return f"Action routing failed: {str(e)}", False

    async def _route_single(
        self,
        action: dict,
        user_context: str = ""
    ) -> Tuple[str, bool]:
        """Route a single action."""
        action_type = action.get("type", "")

        if not action_type:
            return "Action missing 'type' field.", False

        logger.info(f"Routing action: {action_type}")

        # Safety validation
        if self.config.get("safety.enabled", True):
            is_safe, reason = self.safety.validate_action(action)
            if not is_safe:
                logger.warning(f"Action blocked by safety: {reason}")
                return f"Action blocked: {reason}", False

        # Find tool
        tool = self.tools.get_tool(action_type)
        if tool is None:
            # Try to find closest match
            available = self.tools.list_tools()
            suggestion = self._suggest_tool(action_type, available)
            msg = f"Unknown action '{action_type}'."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            return msg, False

        # Execute tool
        try:
            logger.debug(f"Executing tool: {action_type} with params: {action}")
            result = await tool.execute(action)
            logger.info(f"Tool {action_type} executed successfully")
            return result.message, result.success

        except Exception as e:
            logger.error(f"Tool execution error ({action_type}): {e}", exc_info=True)
            return f"Tool '{action_type}' failed: {str(e)}", False

    async def _route_chain(
        self,
        actions: List[dict],
        user_context: str = ""
    ) -> Tuple[str, bool]:
        """Route a chain of actions sequentially."""
        if len(actions) > 10:
            return "Action chain too long (max 10 actions).", False

        results = []
        all_success = True

        for i, action in enumerate(actions, 1):
            logger.info(f"Executing chain action {i}/{len(actions)}: {action.get('type')}")
            message, success = await self._route_single(action, user_context)
            results.append(f"Step {i} ({action.get('type', '?')}): {message}")

            if not success:
                all_success = False
                results.append(f"Chain halted at step {i} due to failure.")
                break

        return "\n".join(results), all_success

    def _suggest_tool(self, action_type: str, available: List[str]) -> Optional[str]:
        """Simple Levenshtein-based suggestion for unknown actions."""
        if not available:
            return None

        best_match = None
        best_distance = float("inf")

        for tool_name in available:
            distance = self._levenshtein(action_type.lower(), tool_name.lower())
            if distance < best_distance:
                best_distance = distance
                best_match = tool_name

        # Only suggest if reasonably close
        threshold = max(3, len(action_type) // 3)
        if best_distance <= threshold:
            return best_match
        return None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return ActionRouter._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]