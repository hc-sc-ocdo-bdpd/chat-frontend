from __future__ import annotations

from pathlib import Path

from app.db import Database


def test_projects_migrate_and_persist(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    project = database.create_project(
        name="Evalanche",
        instructions="Review benchmark code carefully.",
        default_endpoint_id="azure-main",
        default_model_id="gpt-sol",
    )

    conversation = database.create_conversation(
        "azure-main",
        "gpt-sol",
        project_id=project["id"],
    )
    projects = database.list_projects()

    assert projects[0]["conversation_count"] == 1
    assert database.get_conversation(conversation["id"])["project_id"] == project["id"]


def test_project_files_can_be_toggled(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    project = database.create_project(
        name="Files",
        instructions="",
        default_endpoint_id="azure-main",
        default_model_id="gpt-sol",
    )
    file_row = database.create_project_file(
        project_id=project["id"],
        original_name="README.md",
        stored_name="stored_readme.md",
        content_type="text/markdown",
        size_bytes=123,
    )

    assert database.list_project_files(project["id"], active_only=True)
    database.update_project_file(file_row["id"], is_active=False)
    assert database.list_project_files(project["id"], active_only=True) == []


def test_truncate_messages_clears_response_link(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    conversation = database.create_conversation("azure-main", "gpt-sol")
    first = database.add_message(conversation["id"], "user", "First")
    database.add_message(conversation["id"], "assistant", "Answer")
    second = database.add_message(conversation["id"], "user", "Second")
    database.add_message(conversation["id"], "assistant", "Second answer")
    database.update_conversation(
        conversation["id"],
        previous_response_id="resp_123",
        previous_endpoint_id="azure-main",
        previous_model_id="gpt-sol",
    )

    deleted = database.truncate_messages_from(conversation["id"], second["id"])
    remaining = database.list_messages(conversation["id"])
    updated = database.get_conversation(conversation["id"])

    assert deleted == 2
    assert [(message["role"], message["content"]) for message in remaining] == [
        ("user", "First"),
        ("assistant", "Answer"),
    ]
    assert updated["previous_response_id"] is None


def test_deleting_project_keeps_chats(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    project = database.create_project(
        name="Keep chats",
        instructions="",
        default_endpoint_id="azure-main",
        default_model_id="gpt-sol",
    )
    conversation = database.create_conversation(
        "azure-main",
        "gpt-sol",
        project_id=project["id"],
    )

    database.delete_project(project["id"])

    assert database.get_conversation(conversation["id"])["project_id"] is None



def test_legacy_database_gets_project_column(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE conversations (
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
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE TABLE attachments (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            content_type TEXT,
            size_bytes INTEGER NOT NULL,
            provider_files_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()

    database = Database(path)
    conversation = database.create_conversation("azure-main", "gpt-sol")

    assert "project_id" in database.get_conversation(conversation["id"])


def test_reconcile_provider_ids_updates_legacy_values(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    project = database.create_project(
        name="Legacy project",
        instructions="",
        default_endpoint_id="old-endpoint",
        default_model_id="old-model",
    )
    conversation = database.create_conversation(
        "old-endpoint",
        "old-model",
        project_id=project["id"],
    )
    database.update_conversation(
        conversation["id"],
        previous_response_id="resp_old",
        previous_endpoint_id="old-endpoint",
        previous_model_id="old-model",
    )

    result = database.reconcile_provider_ids(
        valid_models={"azure-openai": {"default"}},
        default_endpoint_id="azure-openai",
        default_model_id="default",
    )

    updated_project = database.get_project(project["id"])
    updated_conversation = database.get_conversation(conversation["id"])
    assert result == {"projects": 1, "conversations": 1}
    assert updated_project["default_endpoint_id"] == "azure-openai"
    assert updated_project["default_model_id"] == "default"
    assert updated_conversation["endpoint_id"] == "azure-openai"
    assert updated_conversation["model_id"] == "default"
    assert updated_conversation["previous_response_id"] is None
