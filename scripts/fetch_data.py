"""Populate the data directory on first boot, then get out of the way.

Render gives the service a persistent disk but starts it empty. This downloads
the bundle produced by `scripts/package_data.py` and unpacks it, once. On every
later boot it finds the index already there and exits immediately, so restarts
stay fast.

    python scripts/fetch_data.py            # no-op if the index is present
    python scripts/fetch_data.py --force    # re-download over what is there

Set DATA_BUNDLE_URL to the release asset. With it unset this exits cleanly, so
a local machine that already has its own data never has to think about it.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402

#: Presence of this file means the bundle already landed.
SENTINEL = "index/filings.faiss"


def download(url: str, dest: Path) -> None:
    print(f"fetch_data: downloading {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - operator-supplied URL
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        step = 25_000_000  # progress line every ~25 MB, enough for a boot log
        next_mark = step
        with open(dest, "wb") as fh:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                read += len(chunk)
                if read >= next_mark:
                    pct = f" ({read/total:.0%})" if total else ""
                    print(f"fetch_data:   {read/1e6:.0f} MB{pct}", flush=True)
                    next_mark += step
    print(f"fetch_data: downloaded {read/1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the runtime data bundle")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = settings.cache_path
    base.mkdir(parents=True, exist_ok=True)

    if (base / SENTINEL).exists() and not args.force:
        print(f"fetch_data: data already present at {base}, nothing to do")
        return 0

    url = os.environ.get("DATA_BUNDLE_URL", "").strip()
    if not url:
        print("fetch_data: DATA_BUNDLE_URL not set, skipping")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        try:
            download(url, archive)
        except Exception as exc:  # noqa: BLE001 - a failed fetch must not block boot
            # Starting without data is still better than not starting: /health
            # answers, the logs say why, and a redeploy can retry. Crashing here
            # would take the whole service down over a transient network error.
            print(f"fetch_data: FAILED ({exc}). Server will start without data.")
            return 0

        print(f"fetch_data: extracting into {base}")
        with tarfile.open(archive, "r:gz") as tar:
            # Refuse paths that escape the target directory. The bundle is ours,
            # but an archive extractor that trusts its input is a habit worth
            # not forming.
            for member in tar.getmembers():
                target = (base / member.name).resolve()
                if not str(target).startswith(str(base.resolve())):
                    raise RuntimeError(f"unsafe path in bundle: {member.name}")
            tar.extractall(base)  # noqa: S202 - members validated above

    print(f"fetch_data: ready at {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
