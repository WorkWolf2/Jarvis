"""
Interaction Logger - Records all interactions for self-improvement analysis.
Uses JSONL format for easy streaming and analysis.
"""

import json
import asyncio
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.logger import get_logger

logger = get_logger(__name__)


class InteractionLogger:
    """
    Logs user interactions to a JSONL file for later analysis.
    Thread-safe using async file I/O.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, record: Dict[str, Any]) -> None:
        """Append an interaction record to the log file."""
        # Ensure timestamp
        if "timestamp" not in record:
            record["timestamp"] = datetime.utcnow().isoformat()

        try:
            async with self._lock:
                async with aiofiles.open(self.log_path, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    async def read_recent(self, n: int = 100) -> List[Dict[str, Any]]:
        """Read the N most recent interaction records."""
        if not self.log_path.exists():
            return []

        records = []
        try:
            async with aiofiles.open(self.log_path, "r", encoding="utf-8") as f:
                lines = await f.readlines()

            # Parse last N lines
            for line in reversed(lines[-n:]):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            return list(reversed(records))
        except Exception as e:
            logger.error(f"Failed to read interaction log: {e}")
            return []

    async def read_all(self) -> List[Dict[str, Any]]:
        """Read all interaction records."""
        if not self.log_path.exists():
            return []

        records = []
        try:
            async with aiofiles.open(self.log_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"Failed to read log: {e}")

        return records

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged interactions."""
        records = await self.read_all()
        if not records:
            return {"total": 0}

        failures = [r for r in records if not r.get("success", True)]
        action_types = {}
        for r in records:
            action = r.get("parsed_action", {})
            if isinstance(action, dict):
                t = action.get("type", "unknown")
                action_types[t] = action_types.get(t, 0) + 1

        return {
            "total": len(records),
            "failures": len(failures),
            "failure_rate": len(failures) / len(records) if records else 0,
            "action_distribution": action_types,
            "date_range": {
                "first": records[0].get("timestamp") if records else None,
                "last": records[-1].get("timestamp") if records else None,
            }
        }

    async def get_failures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent failure records."""
        records = await self.read_all()
        failures = [r for r in records if not r.get("success", True)]
        return failures[-limit:]

    async def get_by_action_type(self, action_type: str) -> List[Dict[str, Any]]:
        """Get all interactions with a specific action type."""
        records = await self.read_all()
        return [
            r for r in records
            if isinstance(r.get("parsed_action"), dict)
            and r["parsed_action"].get("type") == action_type
        ]

    async def rotate(self, max_lines: int = 10000) -> None:
        """Rotate the log file if it gets too large."""
        if not self.log_path.exists():
            return

        try:
            async with aiofiles.open(self.log_path, "r") as f:
                lines = await f.readlines()

            if len(lines) > max_lines:
                # Keep last max_lines
                keep = lines[-max_lines:]
                backup_path = self.log_path.with_suffix(".bak.jsonl")
                async with aiofiles.open(backup_path, "w") as f:
                    await f.writelines(lines[:-max_lines])

                async with aiofiles.open(self.log_path, "w") as f:
                    await f.writelines(keep)

                logger.info(f"Log rotated: {len(lines) - max_lines} records archived")

        except Exception as e:
            logger.error(f"Log rotation failed: {e}")