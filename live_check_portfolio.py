"""Live check for the Portfolio agent (REAL market provider and REAL LLM, so it uses network and tokens).

Run from the project root:  python live_check_portfolio.py
Prints the computed metrics table, the warnings, the plain-language summary and the token cost.
The sample numbers below (quantities, costs, the fund's expense ratio) are made up for illustration.
"""
from src.agents.portfolio import PortfolioAgent
from src.core.usage_tracker import get_usage_registry
from live_checks.transcript import run_with_transcript

SAMPLE_PORTFOLIO = [
    {"ticker": "RELIANCE.NS", "quantity": 10, "avg_cost": 2400.0, "asset_class": "equity"},
    {"ticker": "TCS.NS", "quantity": 5, "avg_cost": 3500.0, "asset_class": "equity"},
    {"ticker": "NIFTYBEES.NS", "quantity": 100, "avg_cost": 240.0, "instrument_type": "etf",
     "asset_class": "equity", "expense_ratio_pct": 0.05},               # index ETF, still equity exposure
    {"ticker": "AAPL", "quantity": 4, "avg_cost": 170.0, "asset_class": "equity"},   # priced in USD
    {"ticker": "HDFCBANK.NS", "quantity": 8, "asset_class": "equity"},                # no avg_cost: no P&L
]


def main() -> None:
    agent = PortfolioAgent()
    result = agent.run({"turn_id": "live", "portfolio": SAMPLE_PORTFOLIO})
    print(f"status: {result['status']}")
    data = result["data"] or {}
    metrics = data.get("metrics")
    if metrics:
        print("\n=== Positions ===")
        print(f"{'ticker':14s}{'ccy':5s}{'price':>12s}{'value':>14s}{'weight':>8s}{'P&L %':>9s}  source")
        for p in metrics["positions"]:
            pnl = "-" if p["pnl_pct"] is None else f"{p['pnl_pct']:+.1f}"
            flag = " (not live)" if p["price_stale"] else ""
            print(f"{p['ticker']:14s}{p['currency']:5s}{p['price']:>12,.2f}{p['value']:>14,.2f}"
                  f"{p['weight'] * 100:>7.1f}%{pnl:>9s}  {p['price_source']}{flag}")
        for cm in metrics["currencies"].values():
            print(f"\n=== {cm['currency']} portfolio ===")
            print(f"  total value:       {cm['total_value']:,.2f} {cm['currency']}")
            print(f"  largest position:  {cm['largest_ticker']} {cm['largest_weight'] * 100:.1f}%"
                  f" (top 3: {cm['top3_weight'] * 100:.1f}%)")
            print("  by asset class:    " + ", ".join(
                f"{k} {v * 100:.1f}%" for k, v in cm["allocation_by_asset_class"].items()))
            print(f"  diversification:   {cm['diversification_score']} / 100 ({cm['diversification_label']}),"
                  f" HHI {cm['hhi']:.3f}")
            print(f"  risk label:        {cm['risk_label']} (equity share {cm['equity_share'] * 100:.1f}%)")
            ratio = cm["expense_ratio_weighted_pct"]
            print("  expense ratio:     " + ("not provided" if ratio is None
                  else f"{ratio:.2f}% ({cm['expense_ratio_coverage']})"))
    print("\n=== Warnings ===")
    for warning in data.get("warnings", []) or ["none"]:
        print(f"  - {warning}")
    print("\n=== Summary ===")
    print(result["answer"])

    print("\n=== Token cost ===")
    registry = get_usage_registry()
    for role, t in sorted(registry.totals().items()):
        cost = "unpriced model" if t["cost"] is None else f"${t['cost']:.6f}"
        print(f"  {role:20s} in={t['input_tokens']:>6} out={t['output_tokens']:>6} cost={cost}")


if __name__ == "__main__":
    run_with_transcript(main, __file__)
