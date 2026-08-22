from __future__ import annotations
from datetime import date
from pathlib import Path

from data.sources.sec_edgar import EdgarClient
from data.schemas.filing import Filing
from backend.config import settings


def _cache_path(ticker: str, accession_no: str) -> Path:
    cache_dir = Path(settings.data_cache_dir) / "filings" / ticker.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{accession_no}.txt"


def pull_filings_for_tickers(
    tickers: list[str],
    forms: list[str],
    client: EdgarClient,
    since: date | None = None,
) -> dict[str, list[Filing]]:
    """
    For each ticker, pulls filing metadata + cleaned document text, caching
    to disk. Returns {ticker: [Filing, ...]} using the REAL Filing schema
    (data/schemas/filing.py) - not a local dataclass.

    `since` is not optional in practice - leaving it None pulls EVERY
    filing EDGAR has for that ticker/form combination, which for 8-Ks in
    particular means decades of material-event filings. Callers should
    pass a real cutoff (build_index.py now does).

    Text is cleaned (HTML/iXBRL stripped) by EdgarClient.get_filing_document
    before it's written to cache, so what lands in filings/{TICKER}/*.txt
    is plain prose, not markup.

    KNOWN GOTCHA: EDGAR sometimes returns amended form types like "10-K/A"
    which won't match FormType's fixed enum values and will raise a
    pydantic ValidationError here. Not handled yet - if you hit this,
    either extend FormType or filter these out before constructing Filing.
    """
    results: dict[str, list[Filing]] = {}

    for ticker in tickers:
        filing_records = client.list_recent_filings(ticker, forms=forms, since=since)
        filings: list[Filing] = []

        for record in filing_records:
            accession_no = record["accession"]
            cache_file = _cache_path(ticker, accession_no)
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(record['cik'])}/{accession_no.replace('-', '')}/"
                f"{record['primary_doc']}"
            )

            if not cache_file.exists():
                text = client.get_filing_document(
                    record["cik"], accession_no, record["primary_doc"]
                )
                cache_file.write_text(text, encoding="utf-8")

            filings.append(Filing(
                accession_no=accession_no,
                ticker=ticker,
                cik=record["cik"],
                form_type=record["form_type"],
                filed_date=record["filed_date"],
                url=url,
            ))

        results[ticker] = filings

    return results


def load_cached_filing_text(ticker: str, accession_no: str) -> str:
    cache_file = _cache_path(ticker, accession_no)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cached filing for {ticker}/{accession_no} - run pull_filings_for_tickers first."
        )
    return cache_file.read_text(encoding="utf-8")


if __name__ == "__main__":
    import os
    from datetime import timedelta
    client = EdgarClient(user_agent=os.environ["SEC_USER_AGENT"])
    out = pull_filings_for_tickers(
        ["AAPL"], forms=["10-K", "10-Q"], client=client,
        since=date.today() - timedelta(days=730),
    )
    for ticker, filings in out.items():
        print(ticker, [f.accession_no for f in filings])