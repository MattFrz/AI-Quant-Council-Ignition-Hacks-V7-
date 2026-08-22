#!/usr/bin/env bash
# Build the C++ extension and drop it where Python can import it. Step 4.6.
#
#   bash scripts/build_cpp.sh
#
# A failure here is NOT fatal: quant/backtest/slippage.py falls back to the
# pure-Python participation model, so the demo still runs. See step 4.7.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if   [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
  elif [ -x ".venv/bin/python" ];        then PY=".venv/bin/python"
  else PY="python"; fi
fi

echo "==> building aqc_exec with $PY"
"$PY" -m pip install --quiet pybind11 setuptools wheel || true

if ! "$PY" cpp/bindings/setup.py build_ext --inplace; then
  echo
  echo "BUILD FAILED - this is survivable."
  echo "  The Python slippage model still works; you just lose the order-book"
  echo "  execution numbers. Check you have a C++17 compiler:"
  echo "    Windows: winget install --id Microsoft.VisualStudio.2022.BuildTools \\"
  echo "               --override \"--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\""
  echo "    Linux:   sudo apt install build-essential"
  echo "    macOS:   xcode-select --install"
  exit 1
fi

echo
echo "==> verifying import"
"$PY" -c "import aqc_exec; print('aqc_exec', aqc_exec.__version__, 'OK')" || exit 1
echo "done"
