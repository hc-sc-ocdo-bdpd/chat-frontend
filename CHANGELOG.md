# Changelog

## Unreleased

### Added

- Local Python launcher for Windows, macOS, and Linux
- Automatic `.venv` creation and dependency installation
- Automatic `.env` loading outside Docker
- Dockerized FastAPI chat interface for Azure Responses API deployments
- Streaming responses, reasoning summaries, and tool activity
- Code Interpreter, web research, projects, persistent files, and message workflows
- Markdown, code, citation, hyperlink, and math rendering
- Generic environment-based Azure configuration
- Startup validation for required Azure settings

### Security

- Secrets remain in an ignored local `.env` file
- The default Docker port binds only to localhost

## 2026-08-07, automatic multi-model configuration

### Changed

- Local Python now supports Python 3.10 and newer.
- The endpoint selector was removed from the interface.
- Project settings now select only a default model.
- All model deployments use one shared Azure base URL and API key.

### Added

- Automatic discovery of `AZURE_OPENAI_DEPLOYMENT_1`, `_2`, `_3`, and later entries.
- Optional `AZURE_OPENAI_LABEL_<n>` display names.
- Optional per-model reasoning, verbosity, tool, and token-limit overrides.
- Backward-compatible support for the previous unnumbered deployment variable.

### Migration

- Existing chats and projects whose stored model is no longer configured are
  remapped to the first configured deployment at startup.
