"""AI/NLP Risk Engine: ingest -> analyse -> publish structured risk signals.

The published output is the interface downstream modules subscribe to:
    data/signals.jsonl   append-only signal stream (one JSON object per line)
    data/signals.csv     the same signals as a flat table
    REST API             python src/main.py api  ->  http://localhost:8000/docs
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

import config
import ingestion
import nlp

COLUMNS = [
    "signal_id",
    "doc_id",
    "timestamp",
    "ingested_at",
    "source",
    "source_type",
    "ticker",
    "company",
    "headline",
    "text",
    "url",
    "sentiment_score",
    "sentiment_label",
    "sentiment_confidence",
    "event_type",
    "event_confidence",
    "event_keywords",
    "impact_score",
    "user_sentiment",
    "model",
]

_MODEL = None


def get_model() -> nlp.SentimentModel:
    """Load the sentiment model once per process."""
    global _MODEL
    if _MODEL is None:
        _MODEL = nlp.SentimentModel()
    return _MODEL


# --- Signal stream (publish / subscribe) ------------------------------------------------------------------


def _read_lines() -> list[dict]:
    if not config.SIGNALS_FILE.exists():
        return []
    lines = config.SIGNALS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def read_signals(since: str | None = None) -> pd.DataFrame:
    """Subscribe: every signal the engine has published, optionally only those newer than `since`.

    `since` is any timestamp pandas understands ("2026-10-10", "2026-10-10T12:00:00Z"); times without a
    timezone are read as UTC. An invalid value raises ValueError.
    """
    df = pd.DataFrame(_read_lines(), columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if since:
        cutoff = pd.Timestamp(since)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        df = df[df["timestamp"] > cutoff]
    return df.sort_values("timestamp").reset_index(drop=True)


def publish(signals: list[dict]) -> None:
    """Append signals to the JSON-lines stream and refresh the CSV copy."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config.SIGNALS_FILE.open("a", encoding="utf-8") as f:
        for signal in signals:
            f.write(json.dumps(signal, ensure_ascii=False) + "\n")
    export_csv()


def export_csv() -> None:
    df = read_signals()
    df["event_keywords"] = df["event_keywords"].apply(lambda k: "; ".join(k) if isinstance(k, list) else k)
    df.to_csv(config.SIGNALS_CSV, index=False)


def reset() -> None:
    """Forget all published signals and the index history (the raw snapshots in data/raw are kept)."""
    outputs = (
        config.SIGNALS_FILE,
        config.SIGNALS_CSV,
        config.WEIGHTS_FILE,
        config.REBALANCE_LOG,
        config.PROCESSED_FILE,
        config.LAST_RUN_FILE,
    )
    for path in outputs:
        path.unlink(missing_ok=True)


# --- One engine cycle -------------------------------------------------------------------------------------


def run_cycle(offline: bool = False, log=print) -> dict:
    """Ingest the three sources, analyse every new document and publish the signals."""
    started = time.time()
    docs, status = ingestion.ingest_all(offline=offline)
    for source, state in status.items():
        note = f"  ({state['error']})" if state.get("error") else ""
        log(f"  {ingestion.SOURCES[source]:42s} {state['mode']:18s} {state['documents']:5d} docs{note}")

    seen = set()
    if config.PROCESSED_FILE.exists():
        seen = set(config.PROCESSED_FILE.read_text(encoding="utf-8").split())
    new_docs = list({d["doc_id"]: d for d in docs if d["doc_id"] not in seen}.values())

    model = get_model()
    if model.error:
        log(f"  note: {model.error}")
    nlp_started = time.time()
    signals = nlp.analyze(new_docs, model)
    nlp_seconds = time.time() - nlp_started

    publish(signals)
    with config.PROCESSED_FILE.open("a", encoding="utf-8") as f:
        f.writelines(d["doc_id"] + "\n" for d in new_docs)

    report = {
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": status,
        "documents_fetched": len(docs),
        "new_documents": len(new_docs),
        "new_signals": len(signals),
        "model": model.name,
        "nlp_seconds": round(nlp_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
    }
    config.LAST_RUN_FILE.write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(
        f"  {len(new_docs)} new documents -> {len(signals)} signals with {model.name} "
        f"in {nlp_seconds:.1f}s (cycle {report['total_seconds']}s)"
    )
    return report


def last_run() -> dict:
    """Status report of the most recent engine cycle (empty if the engine has not run yet)."""
    if not config.LAST_RUN_FILE.exists():
        return {}
    return json.loads(config.LAST_RUN_FILE.read_text(encoding="utf-8"))


# --- Company view -----------------------------------------------------------------------------------------


def _top_event(events: pd.Series) -> str:
    specific = events[events != "General"]
    return (specific if len(specific) else events).mode().iloc[0]


def company_summary(df: pd.DataFrame | None = None, hours: int = config.LOOKBACK_HOURS) -> pd.DataFrame:
    """Latest risk picture per company (and MARKET): impact-weighted sentiment, top event, max impact."""
    df = read_signals() if df is None else df
    if df.empty:
        return pd.DataFrame()
    df = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(hours=hours)].copy()
    df["w"] = df["impact_score"].astype(float)
    rows = []
    for ticker, group in df.groupby("ticker"):
        top = group.loc[group["impact_score"].idxmax()]
        weighted = (group["sentiment_score"] * group["w"]).sum() / group["w"].sum()
        rows.append(
            {
                "ticker": ticker,
                "company": group["company"].iloc[0],
                "signals": len(group),
                "sentiment": round(float(weighted), 3),
                "positive": int((group["sentiment_label"] == "positive").sum()),
                "negative": int((group["sentiment_label"] == "negative").sum()),
                "top_event": _top_event(group["event_type"]),
                "max_impact": int(group["impact_score"].max()),
                "top_headline": top["headline"],
                "last_signal": group["timestamp"].max().strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
    return pd.DataFrame(rows).sort_values("sentiment", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one cycle of the NLP Risk Engine")
    parser.add_argument("--offline", action="store_true", help="use the snapshots in data/raw instead of live APIs")
    run_cycle(offline=parser.parse_args().offline)
