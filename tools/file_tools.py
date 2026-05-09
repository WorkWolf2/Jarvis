"""
File Tools - Safe file system operations with path validation.
All file operations are restricted to allowed extensions and paths.
"""

import json
import csv
from pathlib import Path
from typing import ClassVar, Dict, Any, Optional
from datetime import datetime

from tools.base_tool import BaseTool, ToolResult
from core.logger import get_logger

logger = get_logger(__name__)

# Allowed file extensions for read/write
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".json", ".yaml", ".yml",
    ".csv", ".py", ".js", ".ts", ".html",
    ".css", ".sh", ".log", ".conf", ".ini"
}

# Safe default directory (home/documents)
DEFAULT_BASE_DIR = Path.home() / "jarvis_files"


class ReadFileTool(BaseTool):
    """Read the contents of a file."""
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Read the contents of a text file"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read"
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to read (default 5000)"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        path_str = action["path"]
        max_chars = int(action.get("max_chars", 5000))

        path = self._safe_path(path_str)
        if path is None:
            return ToolResult.fail(f"Path not allowed: {path_str}")

        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")

        if not path.is_file():
            return ToolResult.fail(f"Not a file: {path}")

        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return ToolResult.fail(
                f"File type '{ext}' not allowed. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n... [truncated at {max_chars} chars]"
            return ToolResult.ok(f"Contents of {path.name}:\n{content}", data=content)
        except Exception as e:
            return ToolResult.fail(f"Failed to read file: {str(e)}")


class WriteFileTool(BaseTool):
    """Write content to a file."""
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "Write text content to a file (creates or overwrites)"
    requires_confirmation: ClassVar[bool] = True
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["path", "content"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to write to"
            },
            "content": {
                "type": "string",
                "description": "Content to write"
            },
            "append": {
                "type": "boolean",
                "description": "Append instead of overwrite (default false)"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        path_str = action["path"]
        content = action["content"]
        append = bool(action.get("append", False))

        path = self._safe_path(path_str)
        if path is None:
            return ToolResult.fail(f"Path not allowed: {path_str}")

        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS and ext != "":
            return ToolResult.fail(f"File type '{ext}' not allowed for writing")

        try:
            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            path.write_text(content, encoding="utf-8") if not append else open(path, "a").write(content)
            action_word = "Appended to" if append else "Written"
            return ToolResult.ok(
                f"{action_word} {path.name} ({len(content)} chars)",
                data=str(path)
            )
        except Exception as e:
            return ToolResult.fail(f"Failed to write file: {str(e)}")


class ListDirectoryTool(BaseTool):
    """List contents of a directory."""
    name: ClassVar[str] = "list_directory"
    description: ClassVar[str] = "List files and folders in a directory"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (default: home directory)"
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter results (e.g. '*.py')"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        path_str = action.get("path", str(Path.home()))
        pattern = action.get("pattern", "*")

        path = Path(path_str).expanduser().resolve()

        # Basic safety: don't list system directories
        restricted = [Path("/etc"), Path("/sys"), Path("/proc"), Path("/boot")]
        for r in restricted:
            if str(path).startswith(str(r)):
                return ToolResult.fail(f"Directory listing restricted: {path}")

        if not path.exists() or not path.is_dir():
            return ToolResult.fail(f"Directory not found: {path}")

        try:
            items = sorted(path.glob(pattern))[:100]  # Limit to 100 items
            lines = []
            for item in items:
                prefix = "📁 " if item.is_dir() else "📄 "
                size = ""
                if item.is_file():
                    try:
                        size = f" ({item.stat().st_size:,} bytes)"
                    except Exception:
                        pass
                lines.append(f"{prefix}{item.name}{size}")

            result = f"Contents of {path}:\n" + "\n".join(lines)
            if len(items) == 100:
                result += "\n... (showing first 100 items)"

            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to list directory: {str(e)}")


class CreateNoteTool(BaseTool):
    """Create or append to a Jarvis notes file."""
    name: ClassVar[str] = "create_note"
    description: ClassVar[str] = "Create or append a note with a title and content"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["title", "content"],
        "properties": {
            "title": {
                "type": "string",
                "description": "Note title"
            },
            "content": {
                "type": "string",
                "description": "Note content"
            }
        }
    }

    NOTES_FILE = Path.home() / "jarvis_files" / "notes.md"

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        title = action["title"]
        content = action["content"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        note_entry = f"\n## {title}\n*{timestamp}*\n\n{content}\n\n---\n"

        try:
            self.NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.NOTES_FILE, "a", encoding="utf-8") as f:
                f.write(note_entry)
            return ToolResult.ok(f"Note '{title}' saved to {self.NOTES_FILE}")
        except Exception as e:
            return ToolResult.fail(f"Failed to save note: {str(e)}")


class SearchFilesTool(BaseTool):
    """Search for files by name or content."""
    name: ClassVar[str] = "search_files"
    description: ClassVar[str] = "Search for files by name or content within a safe directory"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (filename or content)"
            },
            "search_in": {
                "type": "string",
                "description": "Directory to search in (default: jarvis_files)"
            },
            "by_content": {
                "type": "boolean",
                "description": "Search file contents instead of names"
            }
        }
    }

    async def execute(self, action: dict) -> ToolResult:
        error = self.validate_params(action)
        if error:
            return ToolResult.fail(error)

        query = action["query"].lower()
        search_in = action.get("search_in", str(Path.home() / "jarvis_files"))
        by_content = bool(action.get("by_content", False))

        base = Path(search_in).expanduser().resolve()
        if not base.exists():
            return ToolResult.fail(f"Search directory not found: {base}")

        matches = []
        try:
            for file in base.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue

                if by_content:
                    try:
                        text = file.read_text(encoding="utf-8", errors="ignore").lower()
                        if query in text:
                            matches.append(str(file))
                    except Exception:
                        pass
                else:
                    if query in file.name.lower():
                        matches.append(str(file))

                if len(matches) >= 20:
                    break

            if not matches:
                return ToolResult.ok(f"No files found matching '{query}'")

            result = f"Found {len(matches)} files matching '{query}':\n"
            result += "\n".join(f"  - {m}" for m in matches)
            return ToolResult.ok(result, data=matches)

        except Exception as e:
            return ToolResult.fail(f"Search failed: {str(e)}")


def _safe_path(self, path_str: str) -> Optional[Path]:
    """Validate and return a safe path."""
    try:
        path = Path(path_str).expanduser().resolve()

        # Restrict to home or jarvis_files by default
        home = Path.home().resolve()
        if not str(path).startswith(str(home)):
            return None

        return path
    except Exception:
        return None


# Monkey-patch _safe_path onto all file tools
for _cls in [ReadFileTool, WriteFileTool, ListDirectoryTool, CreateNoteTool, SearchFilesTool]:
    _cls._safe_path = _safe_path