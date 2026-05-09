"""
Memory Manager - Persistent storage for conversations, preferences, and tool history.
Uses SQLite as the primary backend with optional vector storage.
"""

import json
import asyncio
import aiosqlite
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from core.logger import get_logger
from core.config_loader import ConfigLoader

logger = get_logger(__name__)


class MemoryManager:
    """
    Manages persistent memory for JARVIS.

    Stores:
    - Conversation history
    - User preferences
    - Tool execution history
    - System notes and learned facts
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS preferences (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS tool_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action_type TEXT NOT NULL,
        parameters TEXT NOT NULL,
        result TEXT,
        success INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS learned_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact TEXT NOT NULL,
        source TEXT,
        confidence REAL DEFAULT 1.0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS session_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        summary TEXT NOT NULL,
        message_count INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
    CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
    CREATE INDEX IF NOT EXISTS idx_tool_history_type ON tool_history(action_type);
    """

    def __init__(self, db_path: Path, config: ConfigLoader) -> None:
        self.db_path = db_path
        self.config = config
        self._db: Optional[aiosqlite.Connection] = None
        self._session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    async def initialize(self) -> None:
        """Initialize the database."""
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(self.SCHEMA)
        await self._db.commit()
        logger.debug(f"Memory database initialized: {self.db_path}")

    async def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        """Add a message to conversation history."""
        await self._db.execute(
            "INSERT INTO conversations (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (self._session_id, role, content, json.dumps(metadata or {}))
        )
        await self._db.commit()

        # Auto-summarize if conversation is too long
        threshold = self.config.get("memory.conversation_summary_threshold", 50)
        count = await self._count_session_messages()
        if count > threshold:
            await self._auto_summarize()

    async def get_recent_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """Get the N most recent messages for LLM context."""
        async with self._db.execute(
            """
            SELECT role, content FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (self._session_id, n)
        ) as cursor:
            rows = await cursor.fetchall()

        # Return in chronological order (oldest first)
        messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        return messages

    async def get_context(self, max_messages: int = 20) -> str:
        """Build a context string for the system prompt."""
        # Get recent messages summary
        messages = await self.get_recent_messages(max_messages)

        # Get learned facts
        facts = await self.get_learned_facts()

        # Get user preferences
        prefs = await self.get_all_preferences()

        context_parts = []

        if prefs:
            pref_str = ", ".join(f"{k}={v}" for k, v in list(prefs.items())[:5])
            context_parts.append(f"User preferences: {pref_str}")

        if facts:
            fact_str = "; ".join(f["fact"] for f in facts[:5])
            context_parts.append(f"Known facts: {fact_str}")

        # Get session summary if available
        summary = await self._get_latest_summary()
        if summary:
            context_parts.append(f"Previous session summary: {summary}")

        return "\n".join(context_parts) if context_parts else ""

    async def get_learned_facts(self, limit: int = 20) -> List[Dict]:
        """Get learned facts."""
        async with self._db.execute(
            "SELECT fact, source, confidence FROM learned_facts ORDER BY confidence DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def add_learned_fact(self, fact: str, source: str = "interaction", confidence: float = 1.0) -> None:
        """Store a learned fact."""
        await self._db.execute(
            "INSERT INTO learned_facts (fact, source, confidence) VALUES (?, ?, ?)",
            (fact, source, confidence)
        )
        await self._db.commit()

    async def set_preference(self, key: str, value: str) -> None:
        """Set a user preference."""
        await self._db.execute(
            "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
            (key, value)
        )
        await self._db.commit()

    async def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a user preference."""
        async with self._db.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        return row["value"] if row else default

    async def get_all_preferences(self) -> Dict[str, str]:
        """Get all user preferences."""
        async with self._db.execute("SELECT key, value FROM preferences") as cursor:
            rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}

    async def log_tool_execution(
        self,
        action_type: str,
        parameters: dict,
        result: str,
        success: bool
    ) -> None:
        """Log a tool execution to history."""
        await self._db.execute(
            """
            INSERT INTO tool_history (action_type, parameters, result, success)
            VALUES (?, ?, ?, ?)
            """,
            (action_type, json.dumps(parameters), result, int(success))
        )
        await self._db.commit()

    async def get_tool_history(
        self,
        action_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get tool execution history."""
        if action_type:
            query = "SELECT * FROM tool_history WHERE action_type = ? ORDER BY timestamp DESC LIMIT ?"
            params = (action_type, limit)
        else:
            query = "SELECT * FROM tool_history ORDER BY timestamp DESC LIMIT ?"
            params = (limit,)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        async with self._db.execute("SELECT COUNT(*) as count FROM conversations") as cursor:
            conv_count = (await cursor.fetchone())["count"]

        async with self._db.execute(
            "SELECT COUNT(DISTINCT session_id) as count FROM conversations"
        ) as cursor:
            session_count = (await cursor.fetchone())["count"]

        async with self._db.execute("SELECT COUNT(*) as count FROM tool_history") as cursor:
            tool_count = (await cursor.fetchone())["count"]

        async with self._db.execute("SELECT COUNT(*) as count FROM learned_facts") as cursor:
            fact_count = (await cursor.fetchone())["count"]

        return {
            "message_count": conv_count,
            "conversation_count": session_count,
            "tool_executions": tool_count,
            "learned_facts": fact_count,
            "db_path": str(self.db_path)
        }

    async def search_history(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple text search in conversation history."""
        async with self._db.execute(
            """
            SELECT role, content, timestamp FROM conversations
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{query}%", limit)
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def _count_session_messages(self) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) as count FROM conversations WHERE session_id = ?",
            (self._session_id,)
        ) as cursor:
            return (await cursor.fetchone())["count"]

    async def _auto_summarize(self) -> None:
        """Summarize old messages to reduce context size."""
        messages = await self.get_recent_messages(n=30)
        if not messages:
            return

        # Create a simple extractive summary
        summary = f"Session summary ({len(messages)} messages): "
        # Take first and last few messages as summary points
        key_msgs = messages[:3] + messages[-3:]
        points = [f"{m['role']}: {m['content'][:100]}" for m in key_msgs]
        summary += " | ".join(points)

        await self._db.execute(
            "INSERT INTO session_summaries (session_id, summary, message_count) VALUES (?, ?, ?)",
            (self._session_id, summary, len(messages))
        )

        # Delete old messages from this session (keep last 20)
        await self._db.execute(
            """
            DELETE FROM conversations
            WHERE session_id = ? AND id NOT IN (
                SELECT id FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 20
            )
            """,
            (self._session_id, self._session_id)
        )
        await self._db.commit()
        logger.debug("Auto-summarized conversation history")

    async def _get_latest_summary(self) -> Optional[str]:
        """Get the latest session summary."""
        async with self._db.execute(
            "SELECT summary FROM session_summaries ORDER BY created_at DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        return row["summary"] if row else None

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None