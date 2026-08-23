# Deployment

How to put this online permanently, so nothing depends on a laptop being open
or a tunnel staying up.

Two pieces, deployed separately:

| Piece | Host | Why |
|---|---|---|
| FastAPI backend | Render, Docker, with a disk | Needs a persistent 250 MB data directory and 70-second streaming requests. Neither works on serverless. |
| Next.js frontend | Vercel | Free, and it is their framework. Render static hosting works too if you would rather have one dashboard. |

---

## Why not one host for both

The backend cannot run on Vercel. Serverless functions cap the bundle far below
the 154 MB FAISS index, have no persistent disk to put it on, and time out long
before a cold pipeline run finishes streaming. Those are limits of the platform,
not settings to raise.

The frontend runs anywhere. Vercel is suggested because it costs nothing and
needs no configuration, which leaves the whole Render budget for the part that
has to stay warm.

---

## 1. Package the data

The server does not need the full 1.4 GB cache. It needs the index, the price
panel, the profiles and the warmed results, which come to about 250 MB.

```bash
python scripts/package_data.py --list     # check what it will include
python scripts/package_data.py            # writes dist/aqc-data.tar.gz
```

Upload `dist/aqc-data.tar.gz` as a **GitHub release asset** (2 GB limit per
file, free, and it does not bloat the repo). Copy the asset's download URL.

Do not commit the bundle. GitHub rejects files over 100 MB, and `filings.faiss`
alone is 154 MB.

## 2. Deploy the backend

From the Render dashboard: **New > Blueprint**, point it at this repo. It reads
[`render.yaml`](../render.yaml) and prompts for the secrets.

| Variable | Value |
|---|---|
| `LLM_API_KEY` | your OpenAI key |
| `DATA_BUNDLE_URL` | the release asset URL from step 1 |
| `SEC_USER_AGENT` | `Your Name your@email.com` |
| `EXTRA_CORS_ORIGINS` | the frontend URL, once it exists |

On first boot `scripts/fetch_data.py` downloads the bundle onto the disk and
unpacks it. Later boots find it already there and skip straight to serving.

Confirm it came up:

```bash
curl https://YOUR-SERVICE.onrender.com/health
```

## 3. Deploy the frontend

On Vercel: import the repo, set the **root directory** to `frontend`, and add
one environment variable.

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_API_BASE` | `https://YOUR-SERVICE.onrender.com` |
| `NEXT_PUBLIC_USE_FIXTURE` | `false` |

`NEXT_PUBLIC_USE_FIXTURE` matters. Anything other than the literal string
`false` puts the app in fixture mode, where it replays a canned result and never
calls the backend at all - which looks like it works right up until someone
reads the numbers.

Then go back to Render and put the Vercel URL in `EXTRA_CORS_ORIGINS`. Without
it the browser blocks every request and the UI reports "Failed to fetch".

---

## Two settings that decide how a public instance behaves

### `PUBLIC_DEMO_MODE`

On by default in `render.yaml`. The API answers theses that are already warmed
and refuses anything else with a 409 and an explanation.

This exists because a public URL with a live LLM key behind it bills you for
every visitor who types something new. Off, each novel thesis is a real run:
roughly 70 seconds and a few cents, with no upper bound on how many.

Turn it off for a private instance where you want live research.

### `PINNED_AS_OF`

The cache key includes the date a run treats as "today". Left unset, that
resolves to the actual current date, so **every warmed entry goes cold at
midnight** and the next visitor pays for a full live run.

Pinning it to the date the cache was warmed keeps every entry valid
indefinitely, and makes results reproducible - which is the honest behaviour
for a system whose price data stops on a known day.

If you re-seed on newer data, re-warm and move this date to match.

---

## Re-warming a deployed instance

The warmed results live on the disk, so refreshing them means rebuilding the
bundle and redeploying:

```bash
python scripts/warm_all.py --force
python scripts/package_data.py
```

Upload the new bundle, update `DATA_BUNDLE_URL`, and redeploy with
`fetch_data.py --force` (or wipe the disk) so it pulls the new copy rather than
finding the old index and skipping.

---

## What is deliberately not deployed

**The C++ execution simulator.** `aqc_exec` is optional at runtime -
[`slippage.py`](../quant/backtest/slippage.py) falls back to the analytic model
and logs that it did - so the image skips a compiler toolchain for a path the
API does not currently take.

**The seed and index-build pipelines.** Building the index needs the 690 MB
embedding cache and a lot of API calls. That is a local operation whose output
ships as the bundle.
