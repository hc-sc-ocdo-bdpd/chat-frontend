from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_MARKER = VENV_DIR / ".requirements.sha256"
REQUIRED_AZURE_VALUES = (
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def ensure_environment_file() -> bool:
    if not ENV_FILE.exists():
        shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        print("Created .env from .env.example.")
        print(
            "Fill in the Azure connection values and at least "
            "AZURE_OPENAI_DEPLOYMENT_1, then run this command again."
        )
        return False

    values = parse_env_file(ENV_FILE)
    missing = [name for name in REQUIRED_AZURE_VALUES if not values.get(name)]
    has_deployment = any(
        name.startswith("AZURE_OPENAI_DEPLOYMENT_") and value.strip()
        for name, value in values.items()
    ) or bool(values.get("AZURE_OPENAI_DEPLOYMENT", "").strip())
    if not has_deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT_1")

    if missing:
        print("The following required values are blank in .env:")
        for name in missing:
            print(f"  - {name}")
        print("Fill them in, then run this command again.")
        return False

    return True


def requirements_hash() -> str:
    digest = hashlib.sha256()
    digest.update(REQUIREMENTS_FILE.read_bytes())
    digest.update(b"foundry-chat-local-launcher-v2")
    return digest.hexdigest()


def ensure_virtual_environment() -> Path:
    python = venv_python()
    if not python.exists():
        print("Creating local Python environment in .venv...")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)

    expected_hash = requirements_hash()
    installed_hash = (
        REQUIREMENTS_MARKER.read_text(encoding="utf-8").strip()
        if REQUIREMENTS_MARKER.exists()
        else ""
    )

    if installed_hash != expected_hash:
        print("Installing Python dependencies...")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
            cwd=PROJECT_ROOT,
            check=True,
        )
        REQUIREMENTS_MARKER.write_text(expected_hash, encoding="utf-8")

    return python


def main() -> int:
    if sys.version_info < (3, 10):
        print(
            "Python 3.10 or newer is required. "
            f"Current version: {sys.version.split()[0]}"
        )
        return 1

    if not ensure_environment_file():
        return 1

    python = ensure_virtual_environment()
    runtime_env = os.environ.copy()
    runtime_env.setdefault("APP_HOST", "127.0.0.1")
    runtime_env.setdefault("APP_PORT", "3000")
    runtime_env.setdefault("APP_DATA_DIR", str(PROJECT_ROOT / "data"))
    runtime_env.setdefault(
        "APP_CONFIG",
        str(PROJECT_ROOT / "config" / "models.yaml"),
    )

    print()
    print("Starting Foundry Chat at http://localhost:3000")
    print("Press Ctrl+C to stop.")
    print()

    try:
        return subprocess.call(
            [str(python), "-m", "app.main"],
            cwd=PROJECT_ROOT,
            env=runtime_env,
        )
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
