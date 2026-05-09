"""
System Tools - OS-level operations like opening applications and running scripts.
All system commands are validated against the safety whitelist.
"""

import asyncio
import platform
import subprocess
import shutil
from typing import ClassVar, Dict, Any

from tools.base_tool import BaseTool, ToolResult
from core.logger import get_logger

logger = get_logger(__name__)

SYSTEM = platform.system()  # "Linux", "Darwin", "Windows"


class OpenAppTool(BaseTool):
    """Open an application by name."""
    name: ClassVar[str] = "open_app"
    description: ClassVar[str] = "Open an application by name (e.g. chrome, vscode, terminal)"
    requires_confirmation: ClassVar[bool] = False
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["target"],
        "properties": {
            "target": {
                "type": "string",
                "description": "Application name to open"
            }
        }
    }

    # Cross-platform app name mappings
    APP_MAP = {
        "Linux": {
            "chrome": ["google-chrome", "chromium-browser", "chromium"],
            "firefox": ["firefox"],
            "vscode": ["code"],
            "terminal": ["gnome-terminal", "xterm", "konsole"],
            "files": ["nautilus", "dolphin", "thunar"],
            "calculator": ["gnome-calculator", "kcalc"],
            "text": ["gedit", "mousepad", "kate"],
        },
        "Darwin": {
            "chrome": ["open", "-a", "Google Chrome"],
            "firefox": ["open", "-a", "Firefox"],
            "vscode": ["open", "-a", "Visual Studio Code"],
            "terminal": ["open", "-a", "Terminal"],
            "finder": ["open", "-a", "Finder"],
            "calculator": ["open", "-a", "Calculator"],
            "safari": ["open", "-a", "Safari"],
            "calendar": ["open", "-a", "Calendar"],
            "notes": ["open", "-a", "Notes"],
        },
        "Windows": {
            "chrome": ["start", "chrome"],
            "firefox": ["start", "firefox"],
            "notepad": ["notepad"],
            "calculator": ["calc"],
            "explorer": ["explorer"],
            "terminal": ["cmd"],
            "vscode": ["code"],
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        target = action["target"].lower().strip()

        cmd = self._resolve_command(target)
        if cmd is None:
            return ToolResult.fail(
                f"Application '{target}' not found or not supported on {SYSTEM}."
            )

        try:
            if SYSTEM == "Windows":
                subprocess.Popen(cmd, shell=True)
            else:
                # Try each possible command until one works
                launched = False
                if isinstance(cmd[0], list):
                    # Multiple options
                    for option in cmd:
                        if shutil.which(option[0]):
                            subprocess.Popen(option)
                            launched = True
                            break
                else:
                    subprocess.Popen(cmd)
                    launched = True

                if not launched:
                    return ToolResult.fail(f"Could not find executable for '{target}'")

            logger.info(f"Opened application: {target}")
            return ToolResult.ok(f"Opening {target}...")

        except Exception as e:
            logger.error(f"Failed to open {target}: {e}")
            return ToolResult.fail(f"Failed to open {target}: {str(e)}")

    def _resolve_command(self, target: str):
        """Resolve app name to OS command."""
        os_map = self.APP_MAP.get(SYSTEM, {})

        if target in os_map:
            return os_map[target]

        # Try directly with common launchers
        if SYSTEM == "Darwin":
            return ["open", "-a", target.title()]
        elif SYSTEM == "Linux":
            if shutil.which(target):
                return [target]
        elif SYSTEM == "Windows":
            return ["start", target]

        return None


class SayTool(BaseTool):
    """Output text as a response (used for explicit speech/message actions)."""
    name: ClassVar[str] = "say"
    description: ClassVar[str] = "Output text as a spoken or written response"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to say or display"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        text = action["text"]
        return ToolResult.ok(text)


class RunScriptTool(BaseTool):
    """Run a pre-approved script by name (not arbitrary commands)."""
    name: ClassVar[str] = "run_script"
    description: ClassVar[str] = "Run a named script from the scripts directory"
    requires_confirmation: ClassVar[bool] = True
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["script_name"],
        "properties": {
            "script_name": {
                "type": "string",
                "description": "Name of the script to run (without path)"
            },
            "args": {
                "type": "array",
                "description": "Optional arguments for the script"
            }
        }
    }

    SCRIPTS_DIR = "data/scripts"

    async def execute(self, action: dict) -> ToolResult:
        import os
        from pathlib import Path

        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        script_name = action["script_name"]
        args = action.get("args", [])

        # Sanitize script name (no path traversal)
        script_name = Path(script_name).name
        if not script_name.endswith((".sh", ".py", ".bat")):
            return ToolResult.fail("Only .sh, .py, and .bat scripts are allowed")

        script_path = Path(self.SCRIPTS_DIR) / script_name
        if not script_path.exists():
            return ToolResult.fail(f"Script not found: {script_name}")

        try:
            cmd = ["python" if script_name.endswith(".py") else "bash", str(script_path)] + [str(a) for a in args]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)

            if result.returncode == 0:
                output = stdout.decode().strip()
                return ToolResult.ok(f"Script completed: {output[:500]}")
            else:
                error_out = stderr.decode().strip()
                return ToolResult.fail(f"Script failed: {error_out[:500]}")

        except asyncio.TimeoutError:
            return ToolResult.fail("Script timed out (30s limit)")
        except Exception as e:
            return ToolResult.fail(f"Script execution failed: {str(e)}")


class SystemInfoTool(BaseTool):
    """Get system information."""
    name: ClassVar[str] = "system_info"
    description: ClassVar[str] = "Get current system information (CPU, memory, disk usage)"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["cpu", "memory", "disk", "all"],
                "description": "Type of info to retrieve"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        info_type = action.get("type", "all")

        try:
            import psutil
        except ImportError:
            return ToolResult.fail("psutil not installed. Run: pip install psutil")

        info_parts = []

        if info_type in ("cpu", "all"):
            cpu_pct = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            info_parts.append(f"CPU: {cpu_pct}% ({cpu_count} cores)")

        if info_type in ("memory", "all"):
            mem = psutil.virtual_memory()
            info_parts.append(
                f"RAM: {mem.percent}% used "
                f"({mem.used // 1024**3}GB / {mem.total // 1024**3}GB)"
            )

        if info_type in ("disk", "all"):
            disk = psutil.disk_usage("/")
            info_parts.append(
                f"Disk: {disk.percent}% used "
                f"({disk.used // 1024**3}GB / {disk.total // 1024**3}GB)"
            )

        if not info_parts:
            return ToolResult.fail("No system info available")

        return ToolResult.ok("\n".join(info_parts))


class SetReminderTool(BaseTool):
    """Set a simple reminder (in-memory, proof of concept)."""
    name: ClassVar[str] = "set_reminder"
    description: ClassVar[str] = "Set a reminder with a message and delay in minutes"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["message", "minutes"],
        "properties": {
            "message": {
                "type": "string",
                "description": "Reminder message"
            },
            "minutes": {
                "type": "number",
                "description": "Minutes until reminder"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        message = action["message"]
        minutes = float(action.get("minutes", 5))

        if minutes <= 0 or minutes > 1440:  # Max 24 hours
            return ToolResult.fail("Minutes must be between 0 and 1440")

        async def _remind():
            await asyncio.sleep(minutes * 60)
            print(f"\n⏰ REMINDER: {message}\n")

        asyncio.create_task(_remind())
        return ToolResult.ok(f"Reminder set for {minutes} minutes: '{message}'")