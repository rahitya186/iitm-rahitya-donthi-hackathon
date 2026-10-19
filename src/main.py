"""RiskPulse command line: one entry point for everything.

python src/main.py run                       fetch 3 live sources -> NLP risk signals -> Module A rebalance
python src/main.py run --loop 15             keep running every 15 minutes (real-time mode)
python src/main.py run --offline             replay the saved snapshot in data/raw (no internet needed)
python src/main.py run --offline --reset     rebuild every output in data/ from the snapshot, from scratch
python src/main.py dashboard                 Streamlit dashboard  ->  http://localhost:8501
python src/main.py api                       REST API             ->  http://localhost:8000/docs
python src/main.py evaluate                  benchmark the NLP models against naive baselines
"""

import argparse
import contextlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

# Windows consoles default to cp1252: never crash on a non-ASCII headline in the logs
with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cmd_run(args: argparse.Namespace) -> None:
    import engine
    import rebalancer

    if args.reset:
        engine.reset()
        rebalancer.PARAMS_FILE.unlink(missing_ok=True)
    while True:
        mode = "offline" if args.offline else "live"
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] NLP Risk Engine cycle ({mode})")
        engine.run_cycle(offline=args.offline)
        rebalancer.update(now=rebalancer.data_now() if args.offline else None)
        top = rebalancer.current_weights().head(3)
        print("  top weights: " + ", ".join(f"{r.ticker} {r.weight:.1%}" for r in top.itertuples()))
        if not args.loop:
            break
        print(f"  next cycle in {args.loop:g} min (Ctrl+C to stop)")
        time.sleep(args.loop * 60)


def cmd_dashboard(args: argparse.Namespace) -> None:
    print(f"Dashboard: http://localhost:{args.port}  (Ctrl+C to stop)")
    command = [sys.executable, "-m", "streamlit", "run", str(SRC / "dashboard.py"), "--server.port", str(args.port)]
    subprocess.call(command, cwd=SRC.parent)  # run from the repo root so .streamlit/config.toml is used


def cmd_api(args: argparse.Namespace) -> None:
    import uvicorn

    import api

    print(f"API docs: http://localhost:{args.port}/docs  (Ctrl+C to stop)")
    uvicorn.run(api.app, host="127.0.0.1", port=args.port)


def cmd_evaluate(args: argparse.Namespace) -> None:
    import evaluate

    evaluate.main(with_finbert=args.with_finbert)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RiskPulse: AI/NLP Risk Engine + Module A index rebalancer")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the engine and the rebalancer")
    run.add_argument("--offline", action="store_true", help="use the data/raw snapshots instead of the live APIs")
    run.add_argument("--loop", type=float, default=0, metavar="MINUTES", help="repeat every N minutes")
    run.add_argument("--reset", action="store_true", help="clear all outputs and rebuild them from scratch")
    run.set_defaults(func=cmd_run)

    dashboard = sub.add_parser("dashboard", help="start the Streamlit dashboard")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(func=cmd_dashboard)

    api = sub.add_parser("api", help="start the REST API")
    api.add_argument("--port", type=int, default=8000)
    api.set_defaults(func=cmd_api)

    evaluate = sub.add_parser("evaluate", help="benchmark the sentiment and event models")
    evaluate.add_argument("--with-finbert", action="store_true", help="also benchmark ProsusAI/finbert")
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
