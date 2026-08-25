# API

FastAPI, mounted under `/api`. Interactive docs run at `/docs` when the server
is up.

Every response shape is a Pydantic model in `backend/api/schemas.py`, and the
domain objects come from `data/schemas/`. The TypeScript mirror in
`frontend/lib/types.ts` is byte-identical by hand, so a field renamed on one
side has to be renamed on the other in the same commit.

## Running research

### `POST /api/research`

Starts a run. Returns immediately: it either replays a cached result or spawns a
background thread.

```json
{ "thesis": "Analyze AMD and whether the market is underpricing its AI exposure." }
```

```json
{
  "job_id": "20f16a77e774",
  "status": "done",
  "stream_url": "/api/research/stream/20f16a77e774",
  "from_cache": true
}
```

`from_cache` matters to the caller: a cached run has already finished, so its
event stream arrives all at once.

Optional fields: `as_of`, `max_candidates`, `universe_size`.

Two refusals are deliberate rather than errors:

| Status | Meaning |
|---|---|
| 409 | `PUBLIC_DEMO_MODE` is on and the thesis is not warmed |
| 429 | The instance has hit `LIVE_RUNS_PER_HOUR` or `LIVE_RUNS_PER_DAY` |

Both carry an explanation in `detail`. See [deploy.md](deploy.md).

### `GET /api/research/stream/{job_id}`

Server-sent events, one per pipeline step. Thirteen steps, each reporting when
it starts and when it finishes.

```
data: {"step_id": "scan_universe", "label": "Scanned universe",
       "status": "done", "detail": "504 scanned, 499 in universe"}
```

The stream closes after `final_recommendation`. Behind a tunnel or proxy, force
HTTP/2: QUIC drops long-lived SSE connections partway through.

### `GET /api/research/{job_id}`

The finished result: `top_idea`, `runners_up`, `scan`, `llm_cost_usd`.

Job state lives in process memory, so a restart forgets it. The cache does not,
which is why a warmed thesis survives a redeploy and a job id does not.

## Quant

| Route | Does |
|---|---|
| `POST /api/scan` | Thesis to universe funnel and ranked candidates |
| `POST /api/thesis/decompose` | Thesis to structured screening criteria |
| `POST /api/backtest` | Backtest a set of weights |
| `GET /api/backtest/scoreboard` | Strategy comparison |
| `GET /api/backtest/weights` | Current model weights |
| `POST /api/risk` | Portfolio risk metrics |
| `POST /api/risk/sized-book` | Risk after position sizing |
| `POST /api/risk/tail` | Tail risk |
| `POST /api/portfolio` | Size a book from validated ideas |

`POST /api/portfolio` is worth reading in full. It returns `positions` and
`excluded`, and the interesting half is `excluded`: a name that cleared research
and survived the backtest can still be refused here, with the constraint that
refused it quoted verbatim.

```json
{ "ticker": "VRT", "reasons": ["risk_band=high requires confidence >= 0.75, got 0.67"] }
```

## Execution

### `POST /api/execution`

Runs the C++ order book. Builds a book, then executes the same order as a market
order, a sliced order and a passive limit, and returns what each cost.

```json
{ "shares": 2500, "side": "BUY", "slices": 4 }
```

Response carries the book depth, mid, spread, capacity within 5bps, and one
outcome per mode. Returns 503 if the extension is not built, rather than
substituting a Python approximation and calling it the same thing.

### `GET /api/execution/status`

Whether `aqc_exec` is importable. Does not raise.

## Health

`GET /health` returns status and whether offline mode is set. This is the
liveness check a host should point at.

## CORS

Allowed origins come from `FRONTEND_ORIGIN` plus `EXTRA_CORS_ORIGINS`, which is
comma-separated. Origins carry no path, so a trailing slash makes the entry fail
to match and every browser request gets blocked while curl keeps working.
