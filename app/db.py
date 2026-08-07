from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _column_exists(
        connection: sqlite3.Connection, table: str, column: str
    ) -> bool:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row["name"] == column for row in rows)

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
                    default_endpoint_id TEXT NOT NULL,
                    default_model_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    endpoint_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    previous_response_id TEXT,
                    previous_endpoint_id TEXT,
                    previous_model_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    provider_files_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS project_files (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    provider_files_json TEXT NOT NULL DEFAULT '{}',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                        ON DELETE CASCADE
                );
                """
            )

            if not self._column_exists(connection, "conversations", "project_id"):
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN project_id TEXT"
                )

            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_project
                    ON conversations(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_attachments_conversation
                    ON attachments(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_project_files_project
                    ON project_files(project_id, created_at);
                """
            )

    @staticmethod
    def _decode_json_row(row: sqlite3.Row | None, field: str) -> dict[str, Any]:
        if row is None:
            raise KeyError
        item = dict(row)
        item[field.removesuffix("_json")] = json.loads(item.pop(field) or "{}")
        return item

    def reconcile_provider_ids(
        self,
        *,
        valid_models: dict[str, set[str]],
        default_endpoint_id: str,
        default_model_id: str,
    ) -> dict[str, int]:
        """Remap stored provider IDs that are no longer in the configuration."""
        updated_projects = 0
        updated_conversations = 0

        def is_valid(endpoint_id: str, model_id: str) -> bool:
            return model_id in valid_models.get(endpoint_id, set())

        with self._lock, self._connect() as connection:
            projects = connection.execute(
                "SELECT id, default_endpoint_id, default_model_id FROM projects"
            ).fetchall()
            for project in projects:
                if is_valid(
                    project["default_endpoint_id"], project["default_model_id"]
                ):
                    continue
                connection.execute(
                    """
                    UPDATE projects
                    SET default_endpoint_id = ?, default_model_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        default_endpoint_id,
                        default_model_id,
                        time.time(),
                        project["id"],
                    ),
                )
                updated_projects += 1

            conversations = connection.execute(
                "SELECT id, endpoint_id, model_id FROM conversations"
            ).fetchall()
            for conversation in conversations:
                if is_valid(conversation["endpoint_id"], conversation["model_id"]):
                    continue
                connection.execute(
                    """
                    UPDATE conversations
                    SET endpoint_id = ?, model_id = ?,
                        previous_response_id = NULL,
                        previous_endpoint_id = NULL,
                        previous_model_id = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        default_endpoint_id,
                        default_model_id,
                        time.time(),
                        conversation["id"],
                    ),
                )
                updated_conversations += 1

        return {
            "projects": updated_projects,
            "conversations": updated_conversations,
        }

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def create_project(
        self,
        *,
        name: str,
        instructions: str,
        default_endpoint_id: str,
        default_model_id: str,
    ) -> dict[str, Any]:
        project_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects (
                    id, name, instructions, default_endpoint_id,
                    default_model_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    name,
                    instructions,
                    default_endpoint_id,
                    default_model_id,
                    now,
                    now,
                ),
            )
        return self.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    p.*,
                    (
                        SELECT COUNT(*)
                        FROM conversations c
                        WHERE c.project_id = p.id
                    ) AS conversation_count,
                    (
                        SELECT COUNT(*)
                        FROM project_files pf
                        WHERE pf.project_id = p.id
                    ) AS file_count,
                    (
                        SELECT COUNT(*)
                        FROM project_files pf
                        WHERE pf.project_id = p.id AND pf.is_active = 1
                    ) AS active_file_count
                FROM projects p
                ORDER BY p.updated_at DESC, p.name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    p.*,
                    (
                        SELECT COUNT(*)
                        FROM conversations c
                        WHERE c.project_id = p.id
                    ) AS conversation_count,
                    (
                        SELECT COUNT(*)
                        FROM project_files pf
                        WHERE pf.project_id = p.id
                    ) AS file_count,
                    (
                        SELECT COUNT(*)
                        FROM project_files pf
                        WHERE pf.project_id = p.id AND pf.is_active = 1
                    ) AS active_file_count
                FROM projects p
                WHERE p.id = ?
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        return dict(row)

    def update_project(self, project_id: str, **values: Any) -> None:
        allowed = {
            "name",
            "instructions",
            "default_endpoint_id",
            "default_model_id",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        parameters = list(updates.values()) + [project_id]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE projects SET {assignments} WHERE id = ?",
                parameters,
            )
            if cursor.rowcount == 0:
                raise KeyError(project_id)

    def delete_project(self, project_id: str) -> None:
        with self._lock, self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(project_id)
            connection.execute(
                "UPDATE conversations SET project_id = NULL WHERE project_id = ?",
                (project_id,),
            )
            connection.execute(
                "DELETE FROM projects WHERE id = ?",
                (project_id,),
            )

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        endpoint_id: str,
        model_id: str,
        title: str = "New chat",
        project_id: str | None = None,
    ) -> dict[str, Any]:
        if project_id is not None:
            self.get_project(project_id)

        now = time.time()
        conversation_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, title, endpoint_id, model_id, project_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    endpoint_id,
                    model_id,
                    project_id,
                    now,
                    now,
                ),
            )
        return self.get_conversation(conversation_id)

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.endpoint_id,
                    c.model_id,
                    c.project_id,
                    c.created_at,
                    c.updated_at,
                    p.name AS project_name
                FROM conversations c
                LEFT JOIN projects p ON p.id = c.project_id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.*, p.name AS project_name
                FROM conversations c
                LEFT JOIN projects p ON p.id = c.project_id
                WHERE c.id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return dict(row)

    def update_conversation(self, conversation_id: str, **values: Any) -> None:
        allowed = {
            "title",
            "endpoint_id",
            "model_id",
            "project_id",
            "previous_response_id",
            "previous_endpoint_id",
            "previous_model_id",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if "project_id" in updates and updates["project_id"] is not None:
            self.get_project(str(updates["project_id"]))
        if not updates:
            return
        updates["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        parameters = list(updates.values()) + [conversation_id]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE conversations SET {assignments} WHERE id = ?",
                parameters,
            )
            if cursor.rowcount == 0:
                raise KeyError(conversation_id)

    def clear_response_link(self, conversation_id: str) -> None:
        self.update_conversation(
            conversation_id,
            previous_response_id=None,
            previous_endpoint_id=None,
            previous_model_id=None,
        )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            if cursor.rowcount == 0:
                raise KeyError(conversation_id)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        message_id = str(uuid.uuid4())
        now = created_at if created_at is not None else time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    json.dumps(metadata or {}),
                    now,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (time.time(), conversation_id),
            )
        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }

    def get_message(self, message_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT rowid AS sequence_id, *
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        if row is None:
            raise KeyError(message_id)
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        return item

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT rowid AS sequence_id, id, conversation_id, role, content,
                       metadata_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY rowid ASC
                """,
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            result.append(item)
        return result

    def list_messages_through(
        self, conversation_id: str, message_id: str
    ) -> list[dict[str, Any]]:
        target = self.get_message(message_id)
        if target["conversation_id"] != conversation_id:
            raise KeyError(message_id)

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT rowid AS sequence_id, id, conversation_id, role, content,
                       metadata_json, created_at
                FROM messages
                WHERE conversation_id = ? AND rowid <= ?
                ORDER BY rowid ASC
                """,
                (conversation_id, target["sequence_id"]),
            ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            result.append(item)
        return result

    def truncate_messages_from(
        self, conversation_id: str, message_id: str
    ) -> int:
        target = self.get_message(message_id)
        if target["conversation_id"] != conversation_id:
            raise KeyError(message_id)

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ? AND rowid >= ?
                """,
                (conversation_id, target["sequence_id"]),
            )
            connection.execute(
                """
                UPDATE conversations
                SET previous_response_id = NULL,
                    previous_endpoint_id = NULL,
                    previous_model_id = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (time.time(), conversation_id),
            )
        return int(cursor.rowcount)

    # ------------------------------------------------------------------
    # Conversation attachments
    # ------------------------------------------------------------------

    def create_attachment(
        self,
        conversation_id: str,
        original_name: str,
        stored_name: str,
        content_type: str | None,
        size_bytes: int,
        provider_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        attachment_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO attachments (
                    id, conversation_id, original_name, stored_name,
                    content_type, size_bytes, provider_files_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment_id,
                    conversation_id,
                    original_name,
                    stored_name,
                    content_type,
                    size_bytes,
                    json.dumps(provider_files or {}),
                    now,
                ),
            )
        return self.get_attachment(attachment_id)

    def get_attachment(self, attachment_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM attachments WHERE id = ?",
                (attachment_id,),
            ).fetchone()
        if row is None:
            raise KeyError(attachment_id)
        item = dict(row)
        item["provider_files"] = json.loads(item.pop("provider_files_json") or "{}")
        return item

    def list_attachments(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM attachments
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["provider_files"] = json.loads(
                item.pop("provider_files_json") or "{}"
            )
            result.append(item)
        return result

    def set_provider_file(
        self, attachment_id: str, endpoint_id: str, provider_file_id: str
    ) -> None:
        attachment = self.get_attachment(attachment_id)
        provider_files = attachment["provider_files"]
        provider_files[endpoint_id] = provider_file_id
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE attachments
                SET provider_files_json = ?
                WHERE id = ?
                """,
                (json.dumps(provider_files), attachment_id),
            )

    # ------------------------------------------------------------------
    # Project files
    # ------------------------------------------------------------------

    def create_project_file(
        self,
        *,
        project_id: str,
        original_name: str,
        stored_name: str,
        content_type: str | None,
        size_bytes: int,
        is_active: bool = True,
        provider_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        file_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO project_files (
                    id, project_id, original_name, stored_name, content_type,
                    size_bytes, provider_files_json, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    project_id,
                    original_name,
                    stored_name,
                    content_type,
                    size_bytes,
                    json.dumps(provider_files or {}),
                    int(is_active),
                    now,
                ),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
        return self.get_project_file(file_id)

    def get_project_file(self, file_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM project_files WHERE id = ?",
                (file_id,),
            ).fetchone()
        if row is None:
            raise KeyError(file_id)
        item = dict(row)
        item["provider_files"] = json.loads(item.pop("provider_files_json") or "{}")
        item["is_active"] = bool(item["is_active"])
        return item

    def list_project_files(
        self, project_id: str, *, active_only: bool = False
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM project_files
            WHERE project_id = ?
        """
        parameters: list[Any] = [project_id]
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY created_at ASC"

        with self._lock, self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            item["provider_files"] = json.loads(
                item.pop("provider_files_json") or "{}"
            )
            item["is_active"] = bool(item["is_active"])
            result.append(item)
        return result

    def update_project_file(
        self, file_id: str, *, is_active: bool
    ) -> dict[str, Any]:
        file_row = self.get_project_file(file_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE project_files SET is_active = ? WHERE id = ?",
                (int(is_active), file_id),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (time.time(), file_row["project_id"]),
            )
        return self.get_project_file(file_id)

    def set_project_provider_file(
        self, file_id: str, endpoint_id: str, provider_file_id: str
    ) -> None:
        file_row = self.get_project_file(file_id)
        provider_files = file_row["provider_files"]
        provider_files[endpoint_id] = provider_file_id
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE project_files
                SET provider_files_json = ?
                WHERE id = ?
                """,
                (json.dumps(provider_files), file_id),
            )

    def delete_project_file(self, file_id: str) -> dict[str, Any]:
        file_row = self.get_project_file(file_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM project_files WHERE id = ?",
                (file_id,),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (time.time(), file_row["project_id"]),
            )
        return file_row
