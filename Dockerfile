# Backend image for a hosted deployment (Render, Fly, Cloud Run).
#
# The C++ execution extension is deliberately NOT built here. It is optional at
# runtime - quant/backtest/slippage.py falls back to the analytic model and logs
# that it did - and building it would add a compiler toolchain to the image for
# a code path the API does not currently take.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the whole stack.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY quant/ ./quant/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Where the persistent disk gets mounted. `data_cache_dir` is joined onto the
# project root, and pathlib lets an absolute right-hand side win, so an
# absolute value here relocates the whole cache onto the disk.
ENV DATA_CACHE_DIR=/var/data/cache

EXPOSE 8000

# Fetch the data bundle if the disk is empty, then serve. One worker on
# purpose: job state and the SSE event stream live in process memory, so a
# second worker would answer /api/research/{id} for jobs it has never heard of.
CMD ["sh", "-c", "python scripts/fetch_data.py && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
