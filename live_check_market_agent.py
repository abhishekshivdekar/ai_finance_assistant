"""Live check for the Market agent (REAL market provider and REAL LLM, so it uses network and tokens).

Run from the project root:  python live_check_market_agent.py
Prints the metrics table, the comparison, the warnings, the plain-language summary and the token cost.
"""
from src.agents.market import MarketAgent
from src.core.usage_tracker import get_usage_registry
from live_checks.transcript import run_with_transcript

TICKERS = ["RELIANCE.NS", "TCS.NS", "^NSEI", "AAPL"]    # three in INR, one in USD: each is reported in its own currency
PERIOD = "6mo"
# one fixed width per column and a space between every column (text columns left, number columns right)
ROW = "{:<13} {:<4} {:>12} {:>8} {:>10} {:>8} {:>11} {:>11} {:>11} {:<10} {}"


def fmt(value: float | None, suffix: str = "") -> str:
    """A number with one decimal, or '-' when the metric was not available."""
    return "-" if value is None else f"{value:.1f}{suffix}"


def main() -> None:
    agent = MarketAgent()
    result = agent.run({"turn_id": "live", "market_params": {"tickers": TICKERS, "period": PERIOD}})
    print(f"status: {result['status']}")
    data = result["data"] or {}
    metrics = data.get("metrics")
    if metrics:
        print(f"\n=== Metrics ({metrics['period']}) ===")
        print(ROW.format("ticker", "ccy", "price", "day %", "return %", "vol %", "drawdown %",
                         "vs SMA20 %", "vs SMA50 %", "trend", "source"))
        for t in metrics["tickers"]:
            vs = {a["window"]: a["price_vs_pct"] for a in t["averages"]}
            flag = " (not live)" if t["price_stale"] else ""
            print(ROW.format(
                t["ticker"], t["currency"], f"{t['latest_price']:,.2f}", fmt(t["day_change_pct"]),
                fmt(t["period_return_pct"]), fmt(t["volatility_pct"]), fmt(t["max_drawdown_pct"]),
                fmt(vs.get(20)), fmt(vs.get(50)), t["trend"] or "-", f"{t['price_source']}{flag}"))
        print("\n=== 52-week range ===")
        for t in metrics["tickers"]:
            r = t["range_52w"]
            text = "not available" if r is None else (
                f"high {r['high']:,.2f} ({r['pct_below_high']:.1f}% below), low {r['low']:,.2f} ({r['pct_above_low']:.1f}% above)")
            print(f"  {t['ticker']:13s}{text}")
    if data.get("comparison"):
        print("\n=== Comparison ===")
        for row in data["comparison"]:
            print(f"  {row['ticker']:13s}{row['currency']:5s}return {fmt(row['period_return_pct'], '%'):>8s}"
                  f"  volatility {fmt(row['volatility_pct'], '%'):>7s}  drawdown {fmt(row['max_drawdown_pct'], '%'):>7s}"
                  f"  {row['trend'] or '-'}")
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
