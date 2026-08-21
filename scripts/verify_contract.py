"""Phase 1 gate check. Run:  python scripts/verify_contract.py

Loads the fixture through the pydantic models. If this passes, the contract is
real and the four lanes can split up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.schemas.trade_idea import TradeIdea  # noqa: E402


def main() -> int:
    fixture = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "sample_trade_idea.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    raw.pop("_README", None)

    idea = TradeIdea.model_validate(raw)

    checks = [
        ("fixture validates as TradeIdea", True),
        ("has a clickable audit trail", idea.has_audit_trail()),
        ("catalysts present", len(idea.catalysts) > 0),
        ("alpha contributions sum to composite", idea.alpha is not None and idea.alpha.check_sums(1e-9)),
        ("backtest train/test do not overlap", idea.backtest is not None and idea.backtest.window.is_clean()),
        ("every catalyst predates the decision date",
         all(c.is_known_at(idea.as_of) for c in idea.catalysts)),
        ("risk metrics attached", idea.risk is not None),
    ]

    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    print()
    if ok:
        print(f"Contract verified: {idea.ticker} {idea.side.value}, "
              f"alpha {idea.alpha_score}/10, Sharpe {idea.backtest.sharpe}")
        return 0
    print("Contract check FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
