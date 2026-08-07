from pathlib import Path


def test_local_launchers_are_present() -> None:
    assert Path("run-local.cmd").is_file()
    assert Path("run-local.sh").is_file()
    assert Path("scripts/local_launcher.py").is_file()


def test_local_runtime_dependencies_are_declared() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "python-dotenv" in requirements


def test_main_loads_dotenv_and_uses_repo_relative_defaults() -> None:
    source = Path("app/main.py").read_text(encoding="utf-8")
    assert 'load_dotenv(PROJECT_ROOT / ".env", override=False)' in source
    assert 'str(PROJECT_ROOT / "data")' in source
    assert 'str(PROJECT_ROOT / "config" / "models.yaml")' in source
    assert 'os.getenv("APP_PORT", "3000")' in source
    assert 'os.getenv("APP_HOST", "127.0.0.1")' in source


def test_docker_overrides_local_runtime_defaults() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "APP_HOST: 0.0.0.0" in compose
    assert "APP_PORT: 8000" in compose
    assert '"127.0.0.1:3000:8000"' in compose


def test_env_example_only_contains_required_azure_values() -> None:
    keys = []
    for raw_line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0])

    assert keys == [
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_1",
    ]


def test_readme_documents_both_launch_methods() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docker compose up --build" in readme
    assert "run-local.cmd" in readme
    assert "./run-local.sh" in readme


def test_local_launcher_supports_python_310() -> None:
    launcher = Path("scripts/local_launcher.py").read_text(encoding="utf-8")
    assert "sys.version_info < (3, 10)" in launcher
    assert "Python 3.10 or newer" in launcher


def test_wrapper_scripts_support_python_310() -> None:
    windows = Path("run-local.cmd").read_text(encoding="utf-8")
    shell = Path("run-local.sh").read_text(encoding="utf-8")
    assert "(3, 10)" in windows
    assert "Python 3.10 or newer" in windows
    assert "(3, 10)" in shell
    assert "python3.10" in shell
