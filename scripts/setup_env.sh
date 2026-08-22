#!/usr/bin/env bash
# One-command setup. Run from the repo root:  bash scripts/setup_env.sh
set -euo pipefail

echo "==> Python environment"
if command -v py >/dev/null 2>&1; then PY=py
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python
fi

$PY -m venv .venv
# shellcheck disable=SC1091
if [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "==> Frontend"
( cd frontend && npm install )

echo "==> Config"
[ -f .env ] || { cp .env.example .env; echo "created .env - FILL IN YOUR KEYS"; }
[ -f frontend/.env.local ] || cp frontend/.env.local.example frontend/.env.local

cat <<'MSG'

Setup complete.

  Backend:   uvicorn backend.main:app --reload
  Frontend:  cd frontend && npm run dev

Then open http://localhost:8000/health and http://localhost:3000
MSG
