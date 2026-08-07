from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_DEPLOYMENT_PATTERN = re.compile(r"^AZURE_OPENAI_DEPLOYMENT_(\d+)$")
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ModelConfig:
    id: str
    label: str
    deployment: str
    reasoning_efforts: list[str]
    default_reasoning_effort: str
    verbosity_options: list[str]
    default_verbosity: str
    supports_code_interpreter: bool
    default_code_interpreter: bool
    supports_web_search: bool
    default_web_search: bool
    default_research_depth: str
    default_max_output_tokens: int


@dataclass(frozen=True)
class EndpointConfig:
    id: str
    name: str
    base_url: str
    api_key_env: str
    models: dict[str, ModelConfig]

    @property
    def api_key(self) -> str:
        value = os.getenv(self.api_key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"API key environment variable {self.api_key_env} is not configured. "
                "Set it in .env and restart the app."
            )
        return value


@dataclass(frozen=True)
class AppConfig:
    title: str
    default_endpoint: str
    default_model: str
    default_instructions: str
    endpoints: dict[str, EndpointConfig]

    @property
    def models(self) -> dict[str, ModelConfig]:
        return self.endpoints[self.default_endpoint].models


def _parse_bool(value: str | None, default: bool, *, name: str) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false")


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    result: list[str] = []
    seen: set[str] = set()
    for item in value.split(","):
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result or list(default)


def _parse_positive_int(value: str | None, default: int, *, name: str) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _discover_deployments(environment: Mapping[str, str]) -> list[tuple[int, str]]:
    discovered: list[tuple[int, str]] = []
    for name, raw_value in environment.items():
        match = _DEPLOYMENT_PATTERN.fullmatch(name)
        if not match:
            continue
        deployment = str(raw_value).strip()
        if deployment:
            discovered.append((int(match.group(1)), deployment))

    discovered.sort(key=lambda item: item[0])

    # Backward compatibility for existing local installations. New setups and
    # the public .env.example use the numbered convention only.
    if not discovered:
        legacy = str(environment.get("AZURE_OPENAI_DEPLOYMENT", "")).strip()
        if legacy:
            discovered.append((1, legacy))

    return discovered


def _required_connection_values(environment: Mapping[str, str]) -> tuple[str, str]:
    base_url = str(environment.get("AZURE_OPENAI_BASE_URL", "")).strip()
    api_key = str(environment.get("AZURE_OPENAI_API_KEY", "")).strip()

    missing: list[str] = []
    if not base_url:
        missing.append("AZURE_OPENAI_BASE_URL")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not _discover_deployments(environment):
        missing.append("AZURE_OPENAI_DEPLOYMENT_1")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in the required Azure "
            "values before starting the app."
        )

    if not base_url.startswith("https://"):
        raise ValueError("AZURE_OPENAI_BASE_URL must use https://")
    if not base_url.rstrip("/").endswith("/openai/v1"):
        raise ValueError("AZURE_OPENAI_BASE_URL must end with /openai/v1/")

    return base_url.rstrip("/") + "/", api_key


def _model_from_environment(
    *,
    index: int,
    deployment: str,
    defaults: Mapping[str, Any],
    environment: Mapping[str, str],
) -> ModelConfig:
    suffix = str(index)
    label = str(
        environment.get(f"AZURE_OPENAI_LABEL_{suffix}", "")
    ).strip() or deployment

    reasoning_efforts = _parse_csv(
        environment.get(f"AZURE_OPENAI_REASONING_EFFORTS_{suffix}"),
        [str(item) for item in defaults.get("reasoning_efforts", ["auto"])],
    )
    default_reasoning_effort = str(
        environment.get(
            f"AZURE_OPENAI_DEFAULT_REASONING_EFFORT_{suffix}",
            defaults.get("default_reasoning_effort", "auto"),
        )
    ).strip()
    if default_reasoning_effort not in reasoning_efforts:
        raise ValueError(
            f"AZURE_OPENAI_DEFAULT_REASONING_EFFORT_{suffix} must be one of "
            + ", ".join(reasoning_efforts)
        )

    verbosity_options = _parse_csv(
        environment.get(f"AZURE_OPENAI_VERBOSITY_OPTIONS_{suffix}"),
        [str(item) for item in defaults.get("verbosity_options", ["medium"])],
    )
    default_verbosity = str(
        environment.get(
            f"AZURE_OPENAI_DEFAULT_VERBOSITY_{suffix}",
            defaults.get("default_verbosity", "medium"),
        )
    ).strip()
    if default_verbosity not in verbosity_options:
        raise ValueError(
            f"AZURE_OPENAI_DEFAULT_VERBOSITY_{suffix} must be one of "
            + ", ".join(verbosity_options)
        )

    supports_code_interpreter = _parse_bool(
        environment.get(f"AZURE_OPENAI_CODE_INTERPRETER_{suffix}"),
        bool(defaults.get("supports_code_interpreter", True)),
        name=f"AZURE_OPENAI_CODE_INTERPRETER_{suffix}",
    )
    supports_web_search = _parse_bool(
        environment.get(f"AZURE_OPENAI_WEB_SEARCH_{suffix}"),
        bool(defaults.get("supports_web_search", True)),
        name=f"AZURE_OPENAI_WEB_SEARCH_{suffix}",
    )

    return ModelConfig(
        # The deployment name is the stable identifier. Numbered environment
        # suffixes control display order only, so reordering them does not break
        # saved conversations or projects.
        id=deployment,
        label=label,
        deployment=deployment,
        reasoning_efforts=reasoning_efforts,
        default_reasoning_effort=default_reasoning_effort,
        verbosity_options=verbosity_options,
        default_verbosity=default_verbosity,
        supports_code_interpreter=supports_code_interpreter,
        default_code_interpreter=(
            supports_code_interpreter
            and bool(defaults.get("default_code_interpreter", True))
        ),
        supports_web_search=supports_web_search,
        default_web_search=(
            supports_web_search and bool(defaults.get("default_web_search", False))
        ),
        default_research_depth="thorough",
        default_max_output_tokens=_parse_positive_int(
            environment.get(f"AZURE_OPENAI_MAX_OUTPUT_TOKENS_{suffix}"),
            int(defaults.get("default_max_output_tokens", 16384)),
            name=f"AZURE_OPENAI_MAX_OUTPUT_TOKENS_{suffix}",
        ),
    )


def load_config(
    path: str | Path,
    environment: Mapping[str, str] | None = None,
) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    env = os.environ if environment is None else environment
    base_url, _ = _required_connection_values(env)
    deployments = _discover_deployments(env)

    app_raw = raw.get("app", {})
    connection_raw = raw.get("connection", {})
    defaults = raw.get("model_defaults", {})

    model_map: dict[str, ModelConfig] = {}
    for index, deployment in deployments:
        if deployment in model_map:
            raise ValueError(
                f"Duplicate Azure deployment name in .env: {deployment}"
            )
        model = _model_from_environment(
            index=index,
            deployment=deployment,
            defaults=defaults,
            environment=env,
        )
        model_map[model.id] = model

    endpoint_id = str(connection_raw.get("id", "azure-openai"))
    endpoint = EndpointConfig(
        id=endpoint_id,
        name=str(connection_raw.get("name", "Azure OpenAI")),
        base_url=base_url,
        api_key_env=str(connection_raw.get("api_key_env", "AZURE_OPENAI_API_KEY")),
        models=model_map,
    )

    default_model = next(iter(model_map))
    return AppConfig(
        title=str(app_raw.get("title", "Foundry Chat")),
        default_endpoint=endpoint.id,
        default_model=default_model,
        default_instructions=str(app_raw.get("default_instructions", "")),
        endpoints={endpoint.id: endpoint},
    )
