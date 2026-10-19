"""REST API: how downstream applications consume the Risk Engine's structured signals.

Start:  python src/main.py api   ->  interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import config
import engine
import nlp
import rebalancer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.get_model()  # load the sentiment model at startup, not on the first /analyze request
    yield


app = FastAPI(
    title="RiskPulse - AI/NLP Risk Engine API",
    version="1.0",
    description=(
        "Structured risk signals (sentiment score, event classification, impact score) extracted from "
        "financial news, global news (GDELT) and social media (StockTwits)."
    ),
    lifespan=lifespan,
)


class TextIn(BaseModel):
    text: str
    source_type: Literal["news", "global_news", "social"] = "news"


def _records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records", date_format="iso")) if not df.empty else []


def _read_signals(since: str | None) -> pd.DataFrame:
    try:
        return engine.read_signals(since)
    except ValueError:
        raise HTTPException(400, f"Invalid 'since' timestamp: {since!r} (example: 2026-10-10T12:00:00Z)") from None


@app.get("/health")
def health():
    """Status and signal counts per source."""
    df = engine.read_signals()
    return {
        "status": "ok",
        "signals": len(df),
        "latest_signal": df["timestamp"].max().isoformat() if not df.empty else None,
        "by_source": df["source"].value_counts().to_dict() if not df.empty else {},
    }


@app.get("/signals")
def signals(
    ticker: str | None = None,
    event_type: str | None = None,
    source: str | None = None,
    min_impact: int = Query(1, ge=1, le=10),
    since: str | None = None,
    limit: int = Query(100, ge=1, le=5000),
):
    """Structured signals, newest first. Filter by company, event type, source, impact or time."""
    df = _read_signals(since)
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    if event_type:
        df = df[df["event_type"].str.lower() == event_type.lower()]
    if source:
        df = df[df["source"] == source]
    df = df[df["impact_score"] >= min_impact]
    return _records(df.sort_values("timestamp", ascending=False).head(limit))


@app.get("/companies")
def companies():
    """Latest risk picture per company (and MARKET for macro / geopolitical news)."""
    return _records(engine.company_summary())


@app.get("/companies/{ticker}")
def company(ticker: str, limit: int = Query(20, ge=1, le=500)):
    """One company's risk summary and its latest signals."""
    summary = engine.company_summary()
    row = summary[summary["ticker"] == ticker.upper()] if not summary.empty else summary
    if row.empty:
        raise HTTPException(404, f"No signals for {ticker}")
    df = engine.read_signals()
    latest = df[df["ticker"] == ticker.upper()].sort_values("timestamp", ascending=False).head(limit)
    return {"summary": _records(row)[0], "latest_signals": _records(latest)}


@app.get("/events")
def events(
    min_impact: int = Query(config.HIGH_IMPACT_THRESHOLD, ge=1, le=10),
    event_type: str | None = None,
    limit: int = Query(50, ge=1, le=1000),
):
    """High-impact events (default impact >= 7), e.g. the trigger feed for a stress-testing module."""
    return signals(event_type=event_type, min_impact=min_impact, limit=limit)


@app.post("/analyze")
def analyze(body: TextIn):
    """Run the NLP engine on any text: sentiment score, event type, impact score and companies."""
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    return nlp.analyze_text(body.text, engine.get_model(), body.source_type)


@app.get("/index/weights")
def index_weights():
    """Module A output: the current weights of the sentiment-tilted mock index."""
    return _records(rebalancer.current_weights())


@app.get("/index/history")
def index_history(ticker: str | None = None):
    """Module A output: the weight history (one row per rebalance and stock)."""
    df = rebalancer.load_history()
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    return _records(df)
