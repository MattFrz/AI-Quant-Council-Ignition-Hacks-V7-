from __future__ import annotations
import time
import requests
from datetime import date

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q={query}&forms={forms}"
EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
TICKER_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

MAX_REQUESTS_PER_SECOND = 8  # stay under SEC's 10 req/sec limit with margin


class EdgarClient:
    def __init__(self, user_agent: str):
        """
        user_agent MUST be a real name + email, e.g.
        'John Doe john@example.com' — EDGAR blocks generic/missing agents.
        """
        if "@" not in user_agent:
            raise ValueError(
                "SEC_USER_AGENT must include a real email address, e.g. "
                "'Your Name your.email@domain.com'"
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request_time = 0.0
        self._ticker_to_cik: dict[str, str] | None = None

    def _rate_limited_get(self, url: str, params: dict | None = None) -> requests.Response:
        min_interval = 1.0 / MAX_REQUESTS_PER_SECOND
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        response = self.session.get(url, params=params, timeout=15)
        self._last_request_time = time.monotonic()
        response.raise_for_status()
        return response

    def _load_ticker_cik_map(self) -> dict[str, str]:
        if self._ticker_to_cik is None:
            response = self._rate_limited_get(TICKER_CIK_MAP_URL)
            raw = response.json()
            # raw is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            self._ticker_to_cik = {
                entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
                for entry in raw.values()
            }
        return self._ticker_to_cik

    def ticker_to_cik(self, ticker: str) -> str:
        mapping = self._load_ticker_cik_map()
        cik = mapping.get(ticker.upper())
        if cik is None:
            raise KeyError(f"No CIK found for ticker {ticker!r}")
        return cik

    def get_submissions(self, ticker: str) -> dict:
        cik = self.ticker_to_cik(ticker)
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        return self._rate_limited_get(url).json()

    def full_text_search(self, query: str, forms: list[str] | None = None) -> list[dict]:
        forms_param = ",".join(forms) if forms else ""
        url = EDGAR_FULLTEXT_SEARCH_URL.format(query=query, forms=forms_param)
        response = self._rate_limited_get(url)
        return response.json().get("hits", {}).get("hits", [])

    def get_filing_document(self, cik: str, accession: str, doc_filename: str) -> str:
        """
        accession must be passed WITHOUT dashes for the archive path
        (EDGAR submissions.json gives it with dashes, e.g. 0000320193-24-000123).
        """
        accession_nodash = accession.replace("-", "")
        url = f"{EDGAR_ARCHIVES_BASE}/{int(cik)}/{accession_nodash}/{doc_filename}"
        return self._rate_limited_get(url).text

    def list_recent_filings(
        self, ticker: str, forms: list[str], since: date | None = None
    ) -> list[dict]:
        """
        Returns a list of dicts: {accession, form_type, filed_date, primary_doc, cik}
        filtered to the requested form types (and optionally a since-date).
        """
        data = self.get_submissions(ticker)
        cik = data["cik"]
        recent = data["filings"]["recent"]

        results = []
        for i in range(len(recent["accessionNumber"])):
            form_type = recent["form"][i]
            if form_type not in forms:
                continue
            filed_date = date.fromisoformat(recent["filingDate"][i])
            if since and filed_date < since:
                continue
            results.append({
                "accession": recent["accessionNumber"][i],
                "form_type": form_type,
                "filed_date": filed_date,
                "primary_doc": recent["primaryDocument"][i],
                "cik": cik,
            })
        return results