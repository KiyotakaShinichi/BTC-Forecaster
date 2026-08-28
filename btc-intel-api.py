from __future__ import annotations

import argparse

import uvicorn

from market_intelligence.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local BTC market-intelligence research service")
    parser.add_argument("--db", default="btc-intelligence.duckdb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    uvicorn.run(create_app(args.db), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
