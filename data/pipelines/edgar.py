from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from datetime import date

from data.sources.sec_edgar import EdgarClient
from backend.config import settings  # exposes settings.DATA_CACHE_DIR


@dataclass
class Filing:
    accession: str
    form_type: str
    filed_date: date
    url: str
    ticker: str
    cik: str


def _cache_path(ticker: str, accession: str) -> Path:
    cache_dir = Path(settings.DATA_CACHE_DIR) / "filings" / ticker.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{accession}.txt"


def pull_filings_for_tickers(
    tickers: list[str],
    forms: list[str],
    client: EdgarClient,
    since: date | None = None,
) -> dict[str, list[Filing]]:
    """
    For each ticker, pulls filing metadata + raw document text, caching to disk.
    Returns {ticker: [Filing, ...]}. Safe to re-run — cached filings are skipped.
    """
    results: dict[str, list[Filing]] = {}

    for ticker in tickers:
        filing_records = client.list_recent_filings(ticker, forms=forms, since=since)
        filings: list[Filing] = []

        for record in filing_records:
            cache_file = _cache_path(ticker, record["accession"])
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(record['cik'])}/{record['accession'].replace('-', '')}/"
                f"{record['primary_doc']}"
            )

            if not cache_file.exists():
                text = client.get_filing_document(
                    record["cik"], record["accession"], record["primary_doc"]
                )
                cache_file.write_text(text, encoding="utf-8")

            filings.append(Filing(
                accession=record["accession"],
                form_type=record["form_type"],
                filed_date=record["filed_date"],
                url=url,
                ticker=ticker,
                cik=record["cik"],
            ))

        results[ticker] = filings

    return results


def load_cached_filing_text(ticker: str, accession: str) -> str:
    cache_file = _cache_path(ticker, accession)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cached filing for {ticker}/{accession} — run pull_filings_for_tickers first."
        )
    return cache_file.read_text(encoding="utf-8")


if __name__ == "__main__":
    # quick manual smoke test
    import os
    client = EdgarClient(user_agent=os.environ["SEC_USER_AGENT"])
    out = pull_filings_for_tickers(["AAPL"], forms=["10-K", "10-Q"], client=client)
    for ticker, filings in out.items():
        print(ticker, [f.accession for f in filings])