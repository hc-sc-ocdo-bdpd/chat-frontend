from pathlib import Path


def test_public_configuration_uses_numbered_deployments() -> None:
    environment = Path(".env.example").read_text(encoding="utf-8")
    configuration = Path("config/models.yaml").read_text(encoding="utf-8")

    assert "AZURE_OPENAI_BASE_URL=" in environment
    assert "AZURE_OPENAI_API_KEY=" in environment
    assert "AZURE_OPENAI_DEPLOYMENT_1=" in environment
    assert "AZURE_OPENAI_DEPLOYMENT_2=" in environment
    assert "AZURE_OPENAI_LABEL_1=" in environment
    assert "deployment:" not in configuration
    assert "connection:" in configuration
    assert "model_defaults:" in configuration


def test_required_environment_values_are_blank_in_example() -> None:
    values = {}
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value

    assert values["AZURE_OPENAI_BASE_URL"] == ""
    assert values["AZURE_OPENAI_API_KEY"] == ""
    assert values["AZURE_OPENAI_DEPLOYMENT_1"] == ""


def test_readme_documents_automatic_model_discovery() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "AZURE_OPENAI_DEPLOYMENT_1" in readme
    assert "AZURE_OPENAI_DEPLOYMENT_2" in readme
    assert "No YAML editing is required to add a model" in readme
    assert "Python 3.10 or newer" in readme
