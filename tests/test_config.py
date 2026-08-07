from pathlib import Path

import pytest

from app.config import load_config


def clear_deployment_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(__import__("os").environ):
        if name == "AZURE_OPENAI_DEPLOYMENT" or name.startswith(
            "AZURE_OPENAI_DEPLOYMENT_"
        ):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_LABEL_"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_REASONING_EFFORTS_"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_DEFAULT_REASONING_EFFORT_"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_VERBOSITY_OPTIONS_"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_DEFAULT_VERBOSITY_"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_CODE_INTERPRETER_"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_WEB_SEARCH_"):
            monkeypatch.delenv(name, raising=False)
        if name.startswith("AZURE_OPENAI_MAX_OUTPUT_TOKENS_"):
            monkeypatch.delenv(name, raising=False)


def configure_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_deployment_environment(monkeypatch)
    monkeypatch.setenv(
        "AZURE_OPENAI_BASE_URL",
        "https://example-resource.openai.azure.com/openai/v1/",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_1", "test-sol")
    monkeypatch.setenv("AZURE_OPENAI_LABEL_1", "Test Sol")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_2", "test-mini")


def test_config_discovers_multiple_models(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_required_environment(monkeypatch)

    config = load_config(Path("config/models.yaml"))

    assert config.default_endpoint == "azure-openai"
    assert config.default_model == "test-sol"
    endpoint = config.endpoints["azure-openai"]
    assert endpoint.base_url == (
        "https://example-resource.openai.azure.com/openai/v1/"
    )
    assert list(endpoint.models) == ["test-sol", "test-mini"]
    assert endpoint.models["test-sol"].label == "Test Sol"
    assert endpoint.models["test-mini"].label == "test-mini"
    assert endpoint.models["test-sol"].supports_code_interpreter is True
    assert endpoint.models["test-sol"].supports_web_search is True
    assert endpoint.models["test-sol"].default_web_search is False
    assert endpoint.models["test-sol"].default_research_depth == "thorough"
    assert "high" in endpoint.models["test-sol"].reasoning_efforts


def test_numeric_suffix_controls_order_but_deployment_is_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_required_environment(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_1")
    monkeypatch.delenv("AZURE_OPENAI_LABEL_1")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_5", "fifth-model")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_2", "second-model")

    config = load_config(Path("config/models.yaml"))

    assert list(config.models) == ["second-model", "fifth-model"]
    assert config.models["fifth-model"].id == "fifth-model"


def test_per_model_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_required_environment(monkeypatch)
    monkeypatch.setenv(
        "AZURE_OPENAI_REASONING_EFFORTS_2", "auto,low,medium,high"
    )
    monkeypatch.setenv("AZURE_OPENAI_DEFAULT_REASONING_EFFORT_2", "low")
    monkeypatch.setenv("AZURE_OPENAI_VERBOSITY_OPTIONS_2", "low,medium")
    monkeypatch.setenv("AZURE_OPENAI_DEFAULT_VERBOSITY_2", "low")
    monkeypatch.setenv("AZURE_OPENAI_CODE_INTERPRETER_2", "false")
    monkeypatch.setenv("AZURE_OPENAI_WEB_SEARCH_2", "false")
    monkeypatch.setenv("AZURE_OPENAI_MAX_OUTPUT_TOKENS_2", "4096")

    model = load_config(Path("config/models.yaml")).models["test-mini"]

    assert model.reasoning_efforts == ["auto", "low", "medium", "high"]
    assert model.default_reasoning_effort == "low"
    assert model.verbosity_options == ["low", "medium"]
    assert model.default_verbosity == "low"
    assert model.supports_code_interpreter is False
    assert model.default_code_interpreter is False
    assert model.supports_web_search is False
    assert model.default_web_search is False
    assert model.default_max_output_tokens == 4096


def test_duplicate_deployments_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_required_environment(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_2", "test-sol")

    with pytest.raises(ValueError, match="Duplicate Azure deployment"):
        load_config(Path("config/models.yaml"))


def test_legacy_unnumbered_deployment_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_deployment_environment(monkeypatch)
    monkeypatch.setenv(
        "AZURE_OPENAI_BASE_URL",
        "https://example-resource.openai.azure.com/openai/v1/",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "legacy-model")

    config = load_config(Path("config/models.yaml"))

    assert config.default_model == "legacy-model"
    assert list(config.models) == ["legacy-model"]


def test_missing_environment_error_lists_every_required_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_deployment_environment(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        load_config(Path("config/models.yaml"))

    message = str(exc_info.value)
    assert "AZURE_OPENAI_BASE_URL" in message
    assert "AZURE_OPENAI_API_KEY" in message
    assert "AZURE_OPENAI_DEPLOYMENT_1" in message
    assert "Copy .env.example to .env" in message


def test_base_url_must_target_openai_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_required_environment(monkeypatch)
    monkeypatch.setenv(
        "AZURE_OPENAI_BASE_URL",
        "https://example-resource.openai.azure.com/",
    )

    with pytest.raises(ValueError, match="must end with /openai/v1/"):
        load_config(Path("config/models.yaml"))
