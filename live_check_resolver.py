"""Live check for the ticker lookup: RAW Yahoo Search candidates. NO accept/ask rule is applied here.

Uses the network (Yahoo search and Yahoo price history through yfinance). It uses NO language model, so the cost is $0.
Run from the project root:  python live_check_resolver.py     (VS Code: open this file, press F5 or "Run Python File")

For each name it prints what Yahoo's search returned, untouched: symbol, name, exchange code, display exchange and quote
type, plus the time the search took, and whether the market provider returns recent prices for the symbol (the
price check the resolver will use). This tells us which exchange codes and quote types really come back, so that
`ticker_lookup.same_market_groups` and `allowed_quote_types` in config.yaml can be confirmed. The output is saved to
docs/live_checks/live_check_resolver-<date>.txt.
"""
import time

from live_checks.transcript import run_with_transcript
from src.data.market.factory import get_market_provider
from src.data.tickers.names import clean_query
from src.data.tickers.resolver import provider_validator
from src.data.tickers.strategies import YahooSearchStrategy
from src.utils.config import get_config

NAMES = [
    "taiwan semiconductors",
    "Taiwan Semiconductor Manufacturing",
    "Suzlon",
    "Tata Consultancy Services",
    "Infosys",
    "Reliance Industries",
    "HDFC Bank",
    "State Bank of India",
    "Alphabet",
    "Meta Platforms",
    "Nifty 50",
    "qzxv wplk jjrt zzqy",
]


def main() -> None:
    cfg = get_config()["ticker_lookup"]
    search = YahooSearchStrategy(cfg["timeout_seconds"], cfg["max_results"])
    check = provider_validator(get_market_provider(), cfg["validation_period"])
    print("RAW candidates from Yahoo Search. No accept/ask rule is applied. No language model is used (cost $0).")
    print(f"Search timeout {cfg['timeout_seconds']}s, up to {cfg['max_results']} results, price check = {cfg['validation_period']} history.")
    for name in NAMES:
        query = clean_query(name, cfg["max_query_chars"])
        print(f"\n=== {name!r} (sent as {query!r}) ===")
        started = time.monotonic()
        try:
            found = search.find(query)
            error = None
        except Exception as exc:                       # a live check shows failures instead of hiding them
            found, error = [], f"{type(exc).__name__}: {exc}"
        print(f"search took {time.monotonic() - started:.2f}s; {len(found)} candidates" + (f"; ERROR {error}" if error else ""))
        for candidate in found:
            verdict = check(candidate.symbol)
            print(f"  {candidate.symbol:<14} | {candidate.name[:44]:<44} | exchange={candidate.exchange or '-':<5} "
                  f"| display={candidate.exchange_name or '-':<10} | type={candidate.quote_type or '-':<9} "
                  f"| prices={'OK ' + str(verdict.currency) if verdict.ok else 'NOT AVAILABLE'}")


if __name__ == "__main__":
    run_with_transcript(main, __file__)
