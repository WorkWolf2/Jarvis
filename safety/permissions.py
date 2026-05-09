"""
Permissions System - Fine-grained access control for JARVIS operations.
Defines permission levels and checks for each capability.
"""

from enum import Enum
from typing import Set, Dict, Optional
from dataclasses import dataclass, field
from core.logger import get_logger

logger = get_logger(__name__)


class PermissionLevel(Enum):
    """Permission levels from most to least restrictive."""
    NONE = 0       # Blocked completely
    READ = 1       # Read-only operations
    WRITE = 2      # Write operations
    EXECUTE = 3    # Execute operations
    ADMIN = 4      # Administrative operations (requires explicit confirmation)


@dataclass
class PermissionProfile:
    """Defines what a given context/user can do."""
    name: str
    file_read: PermissionLevel = PermissionLevel.READ
    file_write: PermissionLevel = PermissionLevel.WRITE
    file_delete: PermissionLevel = PermissionLevel.NONE
    app_open: PermissionLevel = PermissionLevel.EXECUTE
    script_run: PermissionLevel = PermissionLevel.NONE
    system_command: PermissionLevel = PermissionLevel.NONE
    network_access: PermissionLevel = PermissionLevel.READ
    self_modify: PermissionLevel = PermissionLevel.NONE
    allowed_paths: Set[str] = field(default_factory=set)
    denied_paths: Set[str] = field(default_factory=lambda: {"/etc", "/sys", "/proc", "/boot"})


# Default permission profiles
DEFAULT_PROFILE = PermissionProfile(
    name="default",
    file_read=PermissionLevel.READ,
    file_write=PermissionLevel.WRITE,
    file_delete=PermissionLevel.NONE,
    app_open=PermissionLevel.EXECUTE,
    script_run=PermissionLevel.NONE,
    system_command=PermissionLevel.NONE,
    self_modify=PermissionLevel.NONE,
)

RESTRICTED_PROFILE = PermissionProfile(
    name="restricted",
    file_read=PermissionLevel.READ,
    file_write=PermissionLevel.NONE,
    file_delete=PermissionLevel.NONE,
    app_open=PermissionLevel.NONE,
    script_run=PermissionLevel.NONE,
    system_command=PermissionLevel.NONE,
    self_modify=PermissionLevel.NONE,
)

POWER_USER_PROFILE = PermissionProfile(
    name="power_user",
    file_read=PermissionLevel.READ,
    file_write=PermissionLevel.WRITE,
    file_delete=PermissionLevel.WRITE,
    app_open=PermissionLevel.EXECUTE,
    script_run=PermissionLevel.EXECUTE,
    system_command=PermissionLevel.NONE,
    self_modify=PermissionLevel.NONE,
)


class PermissionManager:
    """Manages permission profiles and checks."""

    ACTION_TO_PERMISSION = {
        "read_file": "file_read",
        "write_file": "file_write",
        "delete_file": "file_delete",
        "open_app": "app_open",
        "run_script": "script_run",
        "system_command": "system_command",
    }

    def __init__(self, profile: PermissionProfile = DEFAULT_PROFILE) -> None:
        self._profile = profile
        self._session_grants: Dict[str, PermissionLevel] = {}

    def check(self, action_type: str, required_level: PermissionLevel = PermissionLevel.EXECUTE) -> bool:
        """Check if an action is permitted."""
        perm_field = self.ACTION_TO_PERMISSION.get(action_type)
        if not perm_field:
            return True  # Unknown action types not restricted here

        # Check session-level grants first
        if action_type in self._session_grants:
            return self._session_grants[action_type].value >= required_level.value

        # Check profile
        profile_level = getattr(self._profile, perm_field, PermissionLevel.NONE)
        return profile_level.value >= required_level.value

    def grant_temporary(self, action_type: str, level: PermissionLevel) -> None:
        """Grant temporary elevated permission for this session."""
        logger.warning(f"Temporary permission grant: {action_type} -> {level.name}")
        self._session_grants[action_type] = level

    def revoke(self, action_type: str) -> None:
        """Revoke a temporary permission grant."""
        self._session_grants.pop(action_type, None)

    @property
    def profile_name(self) -> str:
        return self._profile.name