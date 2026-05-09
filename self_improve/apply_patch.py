"""
Patch Applier - Safely applies approved improvement patches.
All patches require explicit approval and are validated before application.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.logger import get_logger
from core.config_loader import ConfigLoader
from self_improve.patch_generator import PatchGenerator

logger = get_logger(__name__)


@dataclass
class PatchResult:
    """Result of a patch application attempt."""
    success: bool
    message: str
    patch_id: str
    rollback_available: bool = False


class PatchManager:
    """
    Manages the lifecycle of improvement patches:
    1. Save pending patches to disk
    2. Review and approve patches
    3. Apply approved patches with rollback support
    4. Track patch history
    """

    def __init__(self, patches_dir: Path, config: ConfigLoader) -> None:
        self.patches_dir = Path(patches_dir)
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._project_root = patches_dir.parent.parent
        self.generator = PatchGenerator(self._project_root)

        # Subdirectories
        (self.patches_dir / "pending").mkdir(exist_ok=True)
        (self.patches_dir / "applied").mkdir(exist_ok=True)
        (self.patches_dir / "rejected").mkdir(exist_ok=True)
        (self.patches_dir / "backups").mkdir(exist_ok=True)

    def save_patches(self, proposals: List[Dict[str, Any]]) -> List[str]:
        """Save proposals as pending patches. Returns list of saved IDs."""
        saved = []
        for proposal in proposals:
            patch_id = proposal.get("id", f"patch_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")

            # Generate concrete patch from proposal
            concrete = self.generator.generate(proposal)
            if concrete:
                proposal["concrete_patch"] = concrete

            patch_path = self.patches_dir / "pending" / f"{patch_id}.json"
            try:
                patch_path.write_text(json.dumps(proposal, indent=2))
                saved.append(patch_id)
                logger.info(f"Saved pending patch: {patch_id}")
            except Exception as e:
                logger.error(f"Failed to save patch {patch_id}: {e}")

        return saved

    def list_pending(self) -> List[Dict[str, Any]]:
        """List all pending patches."""
        pending = []
        for patch_file in (self.patches_dir / "pending").glob("*.json"):
            try:
                data = json.loads(patch_file.read_text())
                pending.append(data)
            except Exception as e:
                logger.error(f"Failed to read patch {patch_file}: {e}")
        return pending

    def list_applied(self) -> List[Dict[str, Any]]:
        """List all applied patches."""
        applied = []
        for patch_file in (self.patches_dir / "applied").glob("*.json"):
            try:
                applied.append(json.loads(patch_file.read_text()))
            except Exception:
                pass
        return applied

    async def apply_patch(self, patch_id: str) -> PatchResult:
        """
        Apply a specific patch by ID.
        Includes validation, backup, and rollback support.
        """
        patch_path = self.patches_dir / "pending" / f"{patch_id}.json"

        if not patch_path.exists():
            return PatchResult(
                success=False,
                message=f"Patch not found: {patch_id}",
                patch_id=patch_id
            )

        try:
            proposal = json.loads(patch_path.read_text())
            concrete = proposal.get("concrete_patch")

            if not concrete:
                # Generate patch if not already generated
                concrete = self.generator.generate(proposal)

            if not concrete:
                # Mark as informational patch
                self._move_patch(patch_id, "applied", {"applied_at": datetime.utcnow().isoformat()})
                return PatchResult(
                    success=True,
                    message=f"Informational patch {patch_id} marked as reviewed",
                    patch_id=patch_id
                )

            # Validate the concrete patch
            is_valid, reason = self._validate_patch(concrete)
            if not is_valid:
                return PatchResult(
                    success=False,
                    message=f"Patch validation failed: {reason}",
                    patch_id=patch_id
                )

            # Apply based on type
            result = await self._apply_concrete(concrete, patch_id)

            if result.success:
                self._move_patch(patch_id, "applied", {
                    "applied_at": datetime.utcnow().isoformat(),
                    "result": result.message
                })
                logger.info(f"Patch applied successfully: {patch_id}")
            else:
                logger.warning(f"Patch application failed: {patch_id} - {result.message}")

            return result

        except Exception as e:
            logger.error(f"Error applying patch {patch_id}: {e}", exc_info=True)
            return PatchResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                patch_id=patch_id
            )

    def reject_patch(self, patch_id: str, reason: str = "") -> bool:
        """Reject a patch."""
        return self._move_patch(patch_id, "rejected", {
            "rejected_at": datetime.utcnow().isoformat(),
            "reason": reason
        })

    def _validate_patch(self, concrete: dict) -> tuple:
        """Validate a concrete patch before applying."""
        patch_type = concrete.get("patch_type", "")

        if patch_type == "config_change":
            if not concrete.get("key"):
                return False, "Config patch missing 'key'"
            if "new_value" not in concrete:
                return False, "Config patch missing 'new_value'"

        elif patch_type == "new_tool_stub":
            file_path = concrete.get("file", "")
            if not file_path.startswith("tools/"):
                return False, "Tool stubs must be in the tools/ directory"
            if ".." in file_path:
                return False, "Path traversal not allowed"

        elif patch_type in ("prompt_improvement", "memory_improvement"):
            # These require manual review
            if not concrete.get("requires_manual_review"):
                return False, "Prompt changes must be marked as requiring manual review"

        return True, ""

    async def _apply_concrete(self, concrete: dict, patch_id: str) -> PatchResult:
        """Apply a concrete patch."""
        patch_type = concrete.get("patch_type", "")

        if patch_type == "config_change":
            return self._apply_config_change(concrete, patch_id)

        elif patch_type == "new_tool_stub":
            return self._create_tool_stub(concrete, patch_id)

        elif patch_type in ("prompt_improvement", "memory_improvement"):
            # Can't auto-apply these - just mark as acknowledged
            return PatchResult(
                success=True,
                message=(
                    f"Informational patch acknowledged. "
                    f"Manual action required: {concrete.get('description', '')}"
                ),
                patch_id=patch_id,
                rollback_available=False
            )

        return PatchResult(
            success=False,
            message=f"Unknown patch type: {patch_type}",
            patch_id=patch_id
        )

    def _apply_config_change(self, concrete: dict, patch_id: str) -> PatchResult:
        """Apply a configuration change."""
        config_path = self._project_root / "config" / "config.json"

        try:
            # Backup
            backup_path = self.patches_dir / "backups" / f"{patch_id}_config.json"
            shutil.copy2(config_path, backup_path)

            # Load and modify
            config = json.loads(config_path.read_text())

            # Navigate to key using dot notation
            key_path = concrete["key"].split(".")
            current = config
            for part in key_path[:-1]:
                current = current.setdefault(part, {})

            old_value = current.get(key_path[-1])
            current[key_path[-1]] = concrete["new_value"]

            # Write back
            config_path.write_text(json.dumps(config, indent=2))

            return PatchResult(
                success=True,
                message=f"Config updated: {concrete['key']} = {concrete['new_value']} (was: {old_value})",
                patch_id=patch_id,
                rollback_available=True
            )

        except Exception as e:
            return PatchResult(
                success=False,
                message=f"Config change failed: {str(e)}",
                patch_id=patch_id
            )

    def _create_tool_stub(self, concrete: dict, patch_id: str) -> PatchResult:
        """Create a new tool stub file."""
        file_path = self._project_root / concrete["file"]

        if file_path.exists():
            return PatchResult(
                success=False,
                message=f"Tool file already exists: {file_path.name}",
                patch_id=patch_id
            )

        try:
            file_path.write_text(concrete["content"])
            return PatchResult(
                success=True,
                message=f"Tool stub created: {file_path.name}. Review and implement before use.",
                patch_id=patch_id,
                rollback_available=True
            )
        except Exception as e:
            return PatchResult(
                success=False,
                message=f"Failed to create tool stub: {str(e)}",
                patch_id=patch_id
            )

    def _move_patch(self, patch_id: str, destination: str, extra_data: dict = None) -> bool:
        """Move a patch file to a different directory."""
        src = self.patches_dir / "pending" / f"{patch_id}.json"
        dst = self.patches_dir / destination / f"{patch_id}.json"

        try:
            if src.exists():
                data = json.loads(src.read_text())
                if extra_data:
                    data.update(extra_data)
                dst.write_text(json.dumps(data, indent=2))
                src.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to move patch {patch_id}: {e}")
            return False