"""Factor base class and the in-memory panel every factor reads. Step B1.

Two things live here and the split between them is the point.

`Panel` is the only thing a Factor ever sees. It is built once from whatever
Matt's A3 pipeline writes to disk. Factors never open files, never touch
parquet, never know a path. If the on-disk format turns out different from what
we assumed, one loader function changes and every factor below keeps working.

`Factor` enforces the as-of rule in the base class instead of trusting each
subclass to slice correctly. `compute()` truncates the panel to what was
knowable at `as_of` and hands `_compute()` a window whose last row IS the
decision date. A careless subclass cannot leak future data, because the future
is not in the object it was given.

That is deliberately stricter than the plan requires. Leakage is silent: it
does not raise, it just inflates every number downstream until a judge asks
how you handled it. Cheaper to make it structurally impossible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date as Date
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from data.schemas.signal import FactorCategory

DateLike = Union[str, Date, pd.Timestamp]


@dataclass(frozen=True)
class Panel:
    """Aligned market data for one universe over one date range.

    Wide frames (index=date, columns=ticker) because this model is
    cross-sectional: nearly every operation is "rank these tickers against each
    other on this date", which is one line on a wide frame and a groupby on a
    long one.

    Fundamentals stay long, because they are sparse and irregular — a company
    reports four times a year, on its own schedule, and the as-of join keys on
    `report_date`.
    """

    adj_close: pd.DataFrame
    volume: pd.DataFrame
    close: Optional[pd.DataFrame] = None
    securities: Optional[pd.DataFrame] = None     # index=ticker: sector, market_cap, ...
    fundamentals: Optional[pd.DataFrame] = None   # long, must carry ticker + report_date
    universe: Optional[pd.DataFrame] = None       # bool, date x ticker membership
    adv: Optional[pd.DataFrame] = None            # rolling avg dollar volume, from A3

    # ---- validation --------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.adj_close.index, pd.DatetimeIndex):
            raise TypeError(
                "Panel.adj_close must be indexed by a DatetimeIndex (dates down "
                f"the rows, tickers across the columns); got {type(self.adj_close.index).__name__}."
            )
        if not self.adj_close.index.is_monotonic_increasing:
            raise ValueError(
                "Panel dates must be sorted ascending — as_of() slices positionally "
                "and would silently return the wrong window otherwise."
            )
        if self.adj_close.index.has_duplicates:
            raise ValueError("Panel has duplicate dates; de-duplicate before constructing.")
        if not self.adj_close.index.equals(self.volume.index):
            raise ValueError("adj_close and volume must share an identical date index.")
        if not self.adj_close.columns.equals(self.volume.columns):
            raise ValueError("adj_close and volume must share an identical ticker column set.")
        if self.fundamentals is not None:
            missing = {"ticker", "report_date"} - set(self.fundamentals.columns)
            if missing:
                raise ValueError(
                    f"Panel.fundamentals is missing {sorted(missing)}. report_date is not "
                    "optional — it is the field the whole as-of join keys on."
                )

    # ---- shape -------------------------------------------------------------

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    @property
    def tickers(self) -> pd.Index:
        return self.adj_close.columns

    @property
    def n_days(self) -> int:
        return len(self.adj_close.index)

    @property
    def n_tickers(self) -> int:
        return len(self.adj_close.columns)

    # ---- the as-of rule ----------------------------------------------------

    def as_of(self, as_of: DateLike, lag_days: int = 0) -> "Panel":
        """Everything knowable at `as_of`, minus `lag_days` trading days.

        `lag_days` is TRADING days, not calendar days — it counts rows, because
        rows are the only thing that exist here. Matt's A8 guard uses the same
        definition; if it ever doesn't, this is the line to argue about.
        """
        if lag_days < 0:
            raise ValueError(f"lag_days must be >= 0, got {lag_days}")

        ts = pd.Timestamp(as_of)
        pos = int(self.dates.searchsorted(ts, side="right"))  # rows dated <= as_of
        cut = max(pos - lag_days, 0)

        kept_dates = self.dates[:cut]
        effective = kept_dates[-1] if len(kept_dates) else None

        funds = None
        if self.fundamentals is not None:
            if effective is None:
                funds = self.fundamentals.iloc[:0]
            else:
                reported = pd.to_datetime(self.fundamentals["report_date"])
                funds = self.fundamentals.loc[reported <= effective]

        return Panel(
            adj_close=self.adj_close.iloc[:cut],
            volume=self.volume.iloc[:cut],
            close=None if self.close is None else self.close.iloc[:cut],
            securities=self.securities,
            fundamentals=funds,
            universe=None if self.universe is None else self.universe.iloc[:cut],
            adv=None if self.adv is None else self.adv.iloc[:cut],
        )

    # ---- derived views -----------------------------------------------------

    def returns(self, log: bool = False) -> pd.DataFrame:
        px = self.adj_close.where(self.adj_close > 0)
        return np.log(px).diff() if log else px.pct_change()

    def dollar_volume(self) -> pd.DataFrame:
        px = self.close if self.close is not None else self.adj_close
        return px * self.volume

    def members(self, as_of: Optional[DateLike] = None) -> pd.Index:
        """Tickers in the universe on a date. Ranking against today's survivors
        instead of the names that were actually listed is survivorship bias."""
        if self.universe is None:
            return self.tickers
        if as_of is None:
            row = self.universe.iloc[-1]
        else:
            pos = int(self.universe.index.searchsorted(pd.Timestamp(as_of), side="right")) - 1
            if pos < 0:
                return self.tickers[:0]
            row = self.universe.iloc[pos]
        return self.tickers[row.astype(bool).values]


    # ---- adapters ----------------------------------------------------------

    @classmethod
    def from_wide(
        cls,
        prices,
        securities: Optional[pd.DataFrame] = None,
        fundamentals: Optional[pd.DataFrame] = None,
        universe: Optional[pd.DataFrame] = None,
    ) -> "Panel":
        """Build a Panel from Matt's A3 `load_wide()` result.

        The parquet on disk is long format; `load_wide()` does the reshape and
        hands back an object carrying `.close` (adjusted), `.raw_close`,
        `.volume`, `.adv` and `.returns`. This is the seam the whole lane was
        designed around — the on-disk layout can change again tomorrow and only
        this method moves.

        Duck-typed on purpose: anything exposing `.close` and `.volume` works,
        which is what lets the tests exercise it before A3 ships.
        """
        def grab(name: str, required: bool = False) -> Optional[pd.DataFrame]:
            frame = getattr(prices, name, None)
            if frame is None:
                if required:
                    raise AttributeError(
                        f"load_wide() result has no `.{name}` — Panel.from_wide needs "
                        "at least `.close` (adjusted) and `.volume`."
                    )
                return None
            frame = pd.DataFrame(frame).copy()
            frame.index = pd.to_datetime(frame.index)
            return frame.sort_index()

        adj_close = grab("close", required=True)
        volume = grab("volume", required=True)
        raw_close = grab("raw_close")
        adv = grab("adv")

        # Align to the intersection rather than trusting the upstream promise.
        # A single ticker present in one frame and absent from another produces
        # NaN columns that quietly poison the cross-section.
        common = adj_close.columns.intersection(volume.columns)
        dates = adj_close.index.intersection(volume.index)
        for extra in (raw_close, adv):
            if extra is not None:
                common = common.intersection(extra.columns)
                dates = dates.intersection(extra.index)

        def cut(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            return None if frame is None else frame.loc[dates, common]

        return cls(
            adj_close=cut(adj_close),
            volume=cut(volume),
            close=cut(raw_close),
            adv=cut(adv),
            securities=securities,
            fundamentals=fundamentals,
            universe=None if universe is None else universe.loc[dates, common],
        )

    def __repr__(self) -> str:
        span = f"{self.dates[0].date()}..{self.dates[-1].date()}" if self.n_days else "empty"
        return f"Panel({self.n_days} days x {self.n_tickers} tickers, {span})"


class Factor(ABC):
    """One number per ticker, as of one date.

    Subclasses implement `_compute(window)`, where `window` is already truncated
    to what was knowable. The last row of `window` is the decision date.

    `min_lag_days` declares how stale this factor's inputs must be. It defaults
    to 1: a signal for date t is built from data through t-1, so the decision
    never uses a bar it could not have observed before acting on it.
    """

    name: str = "unnamed_factor"
    category: FactorCategory = FactorCategory.MOMENTUM
    min_lag_days: int = 1
    required_history: int = 0

    def compute(self, panel: Panel, as_of: DateLike) -> pd.Series:
        window = panel.as_of(as_of, self.min_lag_days)

        if window.n_days < max(self.required_history, 1):
            return pd.Series(np.nan, index=panel.tickers, name=self.name, dtype=float)

        values = self._compute(window)
        if not isinstance(values, pd.Series):
            raise TypeError(
                f"{type(self).__name__}._compute must return a pandas Series indexed "
                f"by ticker; got {type(values).__name__}."
            )
        return values.reindex(panel.tickers).astype(float).rename(self.name)

    @abstractmethod
    def _compute(self, window: Panel) -> pd.Series:
        """Raw factor value per ticker. `window` contains no future data."""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "min_lag_days": self.min_lag_days,
            "required_history": self.required_history,
            "class": type(self).__name__,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, lag={self.min_lag_days})"


def load_panel(**kwargs) -> Panel:
    """The one line that swaps synthetic data for Matt's real cache.

    Imported lazily so `quant/` carries no import-time dependency on the data
    pipelines — the factors stay testable without a populated cache.
    """
    from data.pipelines.prices import load_wide  # noqa: WPS433

    return Panel.from_wide(load_wide(), **kwargs)
