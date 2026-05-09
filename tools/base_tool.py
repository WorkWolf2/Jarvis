"""
Base classes for JARVIS tools.
All tools must inherit from BaseTool and implement execute().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, ClassVar


@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, message: str, data: Any = None) -> "ToolResult":
        return cls(success=True, message=message, data=data)

    @classmethod
    def fail(cls, message: str, error: Optional[str] = None) -> "ToolResult":
        return cls(success=False, message=message, error=error)


class BaseTool(ABC):
    """
    Abstract base class for all JARVIS tools.

    Each tool must define:
    - name: str - The action type that triggers this tool
    - description: str - Human-readable description
    - parameters: dict - JSON Schema for the action parameters

    And implement:
    - execute(action: dict) -> ToolResult
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[Dict[str, Any]] = {}
    requires_confirmation: ClassVar[bool] = False
    is_destructive: ClassVar[bool] = False

    @abstractmethod
    async def execute(self, action: dict) -> ToolResult:
        """Execute the tool action."""
        ...

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's action."""
        return {
            "type": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation,
            "is_destructive": self.is_destructive,
        }

    def validate_params(self, action: dict) -> Optional[str]:
        """
        Validate action parameters against schema.
        Returns error message if invalid, None if valid.
        """
        required = self.parameters.get("required", [])
        for param in required:
            if param not in action:
                return f"Missing required parameter: '{param}'"
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"