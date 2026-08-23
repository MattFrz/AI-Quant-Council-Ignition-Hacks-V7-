# Backend image for a hosted deployment (Render, Fly, Cloud Run).
#
# Built in two stages so the compiler needed for the C++ extension does not ship
# in the running image: the builder compiles aqc_exec, the runtime gets the
# resulting .so and nothing else from it.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pybind11 setuptools wheel

COPY cpp/ ./cpp/
RUN cd cpp/bindings && python setup.py build_ext --inplace \
    && mkdir -p /build/out && cp aqc_exec*.so /build/out/


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

# The compiled extension, importable as `aqc_exec` from the working directory.
# It stays optional: quant/backtest/slippage.py falls back to the analytic model
# if it is absent, and /api/execution answers 503 rather than pretending.
COPY --from=builder /build/out/ ./

# Where the persistent disk gets mounted. `data_cache_dir` is joined onto the
# project root, and pathlib lets an absolute right-hand side win, so an
# absolute value here relocates the whole cache onto the disk.
ENV DATA_CACHE_DIR=/var/data/cache

EXPOSE 8000

# Fetch the data bundle if the disk is empty, then serve. One worker on
# purpose: job state and the SSE event stream live in process memory, so a
# second worker would answer /api/research/{id} for jobs it has never heard of.
CMD ["sh", "-c", "python scripts/fetch_data.py && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
