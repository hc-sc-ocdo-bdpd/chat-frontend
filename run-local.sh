#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        exec "$candidate" scripts/local_launcher.py "$@"
    fi
done

echo "Python 3.10 or newer was not found."
echo "Install Python, then run ./run-local.sh again."
exit 1
