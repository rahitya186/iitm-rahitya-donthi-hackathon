"""Shared test fixtures: a temporary data folder and a fast, offline sentiment model."""

from datetime import datetime, timedelta, timezone

import pytest

import config
import engine
import nlp
import rebalancer
from ingestion import make_doc

T0 = datetime(2026, 10, 10, tzinfo=timezone.utc)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point every output file at a temporary folder, so tests never touch data/."""
    paths = {
        "DATA_DIR": tmp_path,
        "RAW_DIR": tmp_path / "raw",
        "SIGNALS_FILE": tmp_path / "signals.jsonl",
        "SIGNALS_CSV": tmp_path / "signals.csv",
        "WEIGHTS_FILE": tmp_path / "weights_history.csv",
        "REBALANCE_LOG": tmp_path / "rebalance_log.csv",
        "LAST_RUN_FILE": tmp_path / "last_run.json",
        "PROCESSED_FILE": tmp_path / "processed_doc_ids.txt",
    }
    for name, path in paths.items():
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(rebalancer, "PARAMS_FILE", tmp_path / "module_a_params.json")
    return tmp_path


@pytest.fixture
def vader(monkeypatch):
    """The VADER fallback: no download, deterministic, fast."""
    model = nlp.SentimentModel("vader")
    monkeypatch.setattr(engine, "_MODEL", model)
    return model


def news(uid: str, hours: float, title: str, tickers=()) -> dict:
    """A news document published `hours` after T0."""
    return make_doc(
        "yahoo_finance", "news", uid, T0 + timedelta(hours=hours), title, title, "https://example.com", tickers
    )


@pytest.fixture
def published(data_dir, vader):
    """A small signal stream: good news for Apple, bad news for Boeing, one macro story."""
    docs = [news(f"aapl{i}", i, "Apple shares soar to a record high after strong earnings", ["AAPL"]) for i in range(3)]
    docs += [
        news(f"ba{i}", i, "Boeing shares plunge as Moody's cuts its credit rating to junk", ["BA"]) for i in range(3)
    ]
    docs.append(news("fed", 2, "Fed raises interest rates as inflation surges"))
    signals = nlp.analyze(docs, vader)
    engine.publish(signals)
    return signals
