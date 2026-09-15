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
                    last_read_at REAL,
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

                CREATE TABLE IF NOT EXISTS generated_files (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    source_endpoint_id TEXT,
                    source_container_id TEXT,
                    source_file_id TEXT,
                    provider_files_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(message_id) REFERENCES messages(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_message_id TEXT,
                    assistant_message_id TEXT,
                    status TEXT NOT NULL,
                    status_label TEXT NOT NULL DEFAULT 'Queued',
                    error TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(user_message_id) REFERENCES messages(id)
                        ON DELETE SET NULL,
                    FOREIGN KEY(assistant_message_id) REFERENCES messages(id)
                        ON DELETE SET NULL
                );
                """
            )

            if not self._column_exists(connection, "conversations", "project_id"):
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN project_id TEXT"
                )

            if not self._column_exists(connection, "conversations", "last_read_at"):
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN last_read_at REAL"
                )
                connection.execute(
                    "UPDATE conversations SET last_read_at = updated_at"
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
                CREATE INDEX IF NOT EXISTS idx_generated_files_conversation
                    ON generated_files(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_generated_files_message
                    ON generated_files(message_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_generation_jobs_conversation
                    ON generation_jobs(conversation_id, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_generation_jobs_one_active
                    ON generation_jobs(conversation_id)
                    WHERE status IN ('queued', 'running');
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
                    last_read_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    endpoint_id,
                    model_id,
                    project_id,
                    now,
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
                    c.last_read_at,
                    p.name AS project_name,
                    EXISTS (
                        SELECT 1
                        FROM messages unread_message
                        WHERE unread_message.conversation_id = c.id
                          AND unread_message.role = 'assistant'
                          AND unread_message.created_at > COALESCE(
                              c.last_read_at, c.created_at
                          )
                    ) AS unread,
                    active_job.id AS generation_id,
                    active_job.status AS generation_status,
                    active_job.status_label AS generation_status_label
                FROM conversations c
                LEFT JOIN projects p ON p.id = c.project_id
                LEFT JOIN generation_jobs active_job
                  ON active_job.id = (
                      SELECT candidate.id
                      FROM generation_jobs candidate
                      WHERE candidate.conversation_id = c.id
                        AND candidate.status IN ('queued', 'running')
                      ORDER BY candidate.created_at DESC
                      LIMIT 1
                  )
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["unread"] = bool(item["unread"])
            result.append(item)
        return result

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

    def mark_conversation_read(self, conversation_id: str) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET last_read_at = ? WHERE id = ?",
                (time.time(), conversation_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(conversation_id)

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
        message_id: str | None = None,
        generated_files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        message_id = message_id or str(uuid.uuid4())
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
            for file_row in generated_files or []:
                connection.execute(
                    """
                    INSERT INTO generated_files (
                        id, conversation_id, message_id, original_name,
                        stored_name, content_type, size_bytes,
                        source_endpoint_id, source_container_id,
                        source_file_id, provider_files_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_row["id"],
                        conversation_id,
                        message_id,
                        file_row["original_name"],
                        file_row["stored_name"],
                        file_row.get("content_type"),
                        int(file_row["size_bytes"]),
                        file_row.get("source_endpoint_id"),
                        file_row.get("source_container_id"),
                        file_row.get("source_file_id"),
                        json.dumps(file_row.get("provider_files") or {}),
                        float(file_row.get("created_at") or now),
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

    def update_message_metadata(
        self, message_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE messages SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata), message_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(message_id)
        return self.get_message(message_id)

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
    # Generated files retained from Code Interpreter
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_generated_file(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["provider_files"] = json.loads(
            item.pop("provider_files_json") or "{}"
        )
        return item

    def get_generated_file(self, file_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generated_files WHERE id = ?",
                (file_id,),
            ).fetchone()
        if row is None:
            raise KeyError(file_id)
        return self._decode_generated_file(row)

    def list_generated_files(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generated_files
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [self._decode_generated_file(row) for row in rows]

    def list_message_generated_files(
        self, message_id: str
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generated_files
                WHERE message_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (message_id,),
            ).fetchall()
        return [self._decode_generated_file(row) for row in rows]

    def list_generated_files_from_message(
        self, conversation_id: str, message_id: str
    ) -> list[dict[str, Any]]:
        target = self.get_message(message_id)
        if target["conversation_id"] != conversation_id:
            raise KeyError(message_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT generated.*
                FROM generated_files generated
                JOIN messages message ON message.id = generated.message_id
                WHERE generated.conversation_id = ? AND message.rowid >= ?
                ORDER BY generated.created_at ASC, generated.rowid ASC
                """,
                (conversation_id, target["sequence_id"]),
            ).fetchall()
        return [self._decode_generated_file(row) for row in rows]

    def set_generated_provider_file(
        self, file_id: str, endpoint_id: str, provider_file_id: str
    ) -> None:
        file_row = self.get_generated_file(file_id)
        provider_files = file_row["provider_files"]
        provider_files[endpoint_id] = provider_file_id
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE generated_files
                SET provider_files_json = ?
                WHERE id = ?
                """,
                (json.dumps(provider_files), file_id),
            )

    # ------------------------------------------------------------------
    # Server-owned response generation jobs
    # ------------------------------------------------------------------

    def create_generation_job(
        self, conversation_id: str, user_message_id: str
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_jobs (
                    id, conversation_id, user_message_id, status,
                    status_label, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', 'Queued', ?, ?)
                """,
                (job_id, conversation_id, user_message_id, now, now),
            )
        return self.get_generation_job(job_id)

    def get_generation_job(self, job_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def get_active_generation_job(
        self, conversation_id: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM generation_jobs
                WHERE conversation_id = ?
                  AND status IN ('queued', 'running')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_generation_job(self, job_id: str, **values: Any) -> None:
        allowed = {
            "assistant_message_id",
            "status",
            "status_label",
            "error",
            "started_at",
            "completed_at",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        parameters = list(updates.values()) + [job_id]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE generation_jobs SET {assignments} WHERE id = ?",
                parameters,
            )
            if cursor.rowcount == 0:
                raise KeyError(job_id)

    def fail_interrupted_generation_jobs(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generation_jobs
                WHERE status IN ('queued', 'running')
                """
            ).fetchall()
            connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'failed',
                    status_label = 'Interrupted',
                    error = 'The application restarted before this response finished.',
                    updated_at = ?,
                    completed_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (now, now),
            )
        return [dict(row) for row in rows]

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
