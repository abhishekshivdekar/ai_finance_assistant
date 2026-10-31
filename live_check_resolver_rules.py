"""Live check for the ticker resolver RULES: the REAL resolver (real Yahoo search + real price check) on 16 names.

Uses the network (Yahoo search and Yahoo price history through yfinance). It uses NO language model, so the cost is $0.
Run from the project root:  python live_check_resolver_rules.py     (VS Code: open this file, press F5 or "Run Python File")

For each name it prints what the resolver decided: status, chosen symbol, exchange, currency, the disclosure note, the
options when the answer is ambiguous, the message when the answer needs the user, and the elapsed time. The script
does not interpret the results. The output is saved to docs/live_checks/live_check_resolver_rules-<date>.txt.
(live_check_resolver.py prints the RAW Yahoo candidates instead, with no rules applied.)
"""
import time

from live_checks.transcript import run_with_transcript
from src.data.market.factory import get_market_provider
from src.data.tickers.resolver import build_resolver

NAMES = [
    "infosys",
    "suzlon",
    "tata consultancy services",
    "reliance industries",
    "hdfc bank",
    "state bank of india",
    "tsmc",
    "taiwan semiconductors",
    "Taiwan Semiconductor Manufacturing Co Ltd TPE: 2330",
    "nvidia corp",
    "berkshire hathaway",
    "alphabet inc",
    "meta platforms",
    "tata",
    "zomato",
    "qzxv wplk",
]


def main() -> None:
    resolver = build_resolver(get_market_provider())
    print("REAL resolver: real Yahoo search and real price checks. No language model is used (cost $0). Results are printed, not interpreted.")
    for name in NAMES:
        started = time.monotonic()
        result = resolver.resolve(name)
        elapsed = time.monotonic() - started
        print(f"\n=== {name!r} ===")
        print(f"status: {result.status}   source: {result.source or '-'}   elapsed: {elapsed:.2f}s")
        print(f"chosen symbol: {', '.join(result.symbols) or '-'}")
        for candidate in result.candidates:
            print(f"  {'chosen' if result.symbols else 'option'}: {candidate.symbol} | {candidate.name} | exchange={candidate.exchange or '-'} "
                  f"({candidate.exchange_name or '-'}) | type={candidate.quote_type or '-'} | currency={candidate.currency or 'not checked'}")
        print(f"note: {result.note or '-'}")
        print(f"message: {result.message or '-'}")


if __name__ == "__main__":
    run_with_transcript(main, __file__)
