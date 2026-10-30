"""Live check for the market data layer (uses the real network)."""
import time

from src.data.market import get_market_provider
from live_checks.transcript import run_with_transcript


def show(label, result):
    print(f"{label:26s} source={result.source:12s} currency={result.currency} "
          f"stale={result.is_stale} error={result.error}")
    return result


def main() -> None:
    provider = get_market_provider()
    for ticker in ("RELIANCE.NS", "TCS.NS", "^NSEI", "AAPL"):
        r = show(f"quote {ticker}", provider.get_quote(ticker))
        if r.data:
            print("   ", r.data)
    start = time.perf_counter()
    show("quote TCS.NS (again)", provider.get_quote("TCS.NS"))
    print(f"    second call took {time.perf_counter() - start:.3f}s")
    h = show("history RELIANCE.NS 1mo", provider.get_history("RELIANCE.NS", "1mo"))
    if h.data:
        print("    rows:", len(h.data), "last:", h.data[-1])
    n = show("news Apple stock", provider.get_news("Apple stock"))
    if n.data:
        print("    first headline:", n.data[0])
    show("bad ticker '???'", provider.get_quote("???"))
    show("unknown ticker ZZZZQ", provider.get_quote("ZZZZQ"))


if __name__ == "__main__":
    run_with_transcript(main, __file__)
