"""
Safety Validator - Validates all LLM-generated actions before execution.
Implements whitelist-based safety with configurable rules.
"""

import re
from typing import Tuple, Optional, List
from core.logger import get_logger
from core.config_loader import ConfigLoader

logger = get_logger(__name__)


class SafetyValidator:
    """
    Validates all actions before they are executed.
    Provides multiple layers of safety checks.
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self.enabled = config.get("safety.enabled", True)
        self.whitelist_mode = config.get("safety.whitelist_mode", True)
        self.allowed_apps = set(
            app.lower() for app in config.get("safety.allowed_apps", [])
        )
        self.blocked_commands = [
            cmd.lower() for cmd in config.get("safety.blocked_commands", [])
        ]
        self.allowed_extensions = set(config.get("safety.allowed_file_extensions", []))
        self.restricted_paths = config.get("safety.restricted_paths", [])
        self.max_command_length = config.get("safety.max_command_length", 500)

    def validate_action(self, action: dict) -> Tuple[bool, str]:
        """
        Validate an action dict.

        Returns:
            Tuple[bool, str]: (is_safe, reason_if_not_safe)
        """
        if not self.enabled:
            return True, ""

        action_type = action.get("type", "").lower()

        # Type-specific validation
        validators = {
            "open_app": self._validate_open_app,
            "write_file": self._validate_file_write,
            "read_file": self._validate_file_read,
            "run_script": self._validate_run_script,
            "system_command": self._validate_system_command,
            "delete_file": self._validate_file_delete,
        }

        validator = validators.get(action_type)
        if validator:
            return validator(action)

        # For unknown action types, allow (tool registry handles "not found")
        return True, ""

    def _validate_open_app(self, action: dict) -> Tuple[bool, str]:
        target = action.get("target", "").lower().strip()

        if not target:
            return False, "No target application specified"

        if len(target) > 100:
            return False, "Target name too long"

        # Check for path traversal or shell injection in app name
        if any(c in target for c in [".", "/", "\\", ";", "&", "|", "`", "$", ">"]):
            return False, f"Invalid characters in application name: '{target}'"

        if self.whitelist_mode and self.allowed_apps:
            # Allow if app name contains any whitelist entry
            if not any(allowed in target for allowed in self.allowed_apps):
                return False, (
                    f"Application '{target}' is not in the allowed list. "
                    f"Allowed: {', '.join(sorted(self.allowed_apps))}"
                )

        return True, ""

    def _validate_file_write(self, action: dict) -> Tuple[bool, str]:
        path = action.get("path", "")

        # Check restricted paths
        for restricted in self.restricted_paths:
            if path.startswith(restricted):
                return False, f"Writing to restricted path not allowed: {restricted}"

        # Check for path traversal
        if ".." in path:
            return False, "Path traversal (..) not allowed"

        # Check file extension
        import os
        ext = os.path.splitext(path)[1].lower()
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return False, f"Writing to '{ext}' files not allowed"

        return True, ""

    def _validate_file_read(self, action: dict) -> Tuple[bool, str]:
        path = action.get("path", "")

        # Check restricted paths
        for restricted in self.restricted_paths:
            if path.startswith(restricted):
                return False, f"Reading from restricted path not allowed: {restricted}"

        # Check for path traversal
        if ".." in path:
            return False, "Path traversal (..) not allowed"

        return True, ""

    def _validate_run_script(self, action: dict) -> Tuple[bool, str]:
        script_name = action.get("script_name", "")

        # Prevent path traversal
        if "/" in script_name or "\\" in script_name or ".." in script_name:
            return False, "Script name cannot contain path separators"

        # Only allow specific extensions
        allowed_script_exts = {".sh", ".py", ".bat"}
        import os
        ext = os.path.splitext(script_name)[1].lower()
        if ext not in allowed_script_exts:
            return False, f"Script type '{ext}' not allowed. Use .sh, .py, or .bat"

        return True, ""

    def _validate_system_command(self, action: dict) -> Tuple[bool, str]:
        """System commands are blocked unless explicitly allowed."""
        command = action.get("command", "").lower()

        if len(command) > self.max_command_length:
            return False, f"Command too long (max {self.max_command_length} chars)"

        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked in command:
                return False, f"Blocked command pattern detected: '{blocked}'"

        # In whitelist mode, system_command is completely blocked unless specifically allowed
        if self.whitelist_mode:
            return False, (
                "Direct system commands are not allowed in whitelist mode. "
                "Use specific tools like open_app, run_script instead."
            )

        return True, ""

    def _validate_file_delete(self, action: dict) -> Tuple[bool, str]:
        """File deletion is always blocked for safety."""
        return False, (
            "File deletion is not allowed for safety. "
            "Please delete files manually."
        )

    def validate_llm_output(self, output: str) -> Tuple[bool, str]:
        """
        Validate raw LLM output before parsing.
        Checks for obvious prompt injection or dangerous patterns.
        """
        if not output:
            return True, ""

        # Check for prompt injection patterns
        injection_patterns = [
            r"ignore previous instructions",
            r"disregard all previous",
            r"you are now",
            r"new instructions:",
            r"system override",
            r"jailbreak",
        ]

        output_lower = output.lower()
        for pattern in injection_patterns:
            if re.search(pattern, output_lower):
                logger.warning(f"Potential prompt injection detected: {pattern}")
                return False, f"Suspicious pattern detected in output: '{pattern}'"

        return True, ""

    def sanitize_string(self, value: str) -> str:
        """Remove potentially dangerous characters from strings."""
        # Remove shell special characters
        dangerous = set(";&|`$><\\")
        return "".join(c for c in value if c not in dangerous)