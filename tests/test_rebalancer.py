"""Module A: position limits, tilt direction and an end-to-end rebalance on a small signal stream."""

import pandas as pd
import pytest

import config
import rebalancer
from conftest import T0

EQUAL = 1 / len(config.TICKERS)


def test_apply_bounds_respects_limits_and_sums_to_one():
    raw = pd.Series([5.0, 1.0, 1.0, 0.01, 0.01] + [1.0] * 10, index=config.TICKERS)
    weights = rebalancer.apply_bounds(raw, 0.02, 0.12)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.02 - 1e-9
    assert weights.max() <= 0.12 + 1e-9


def test_positive_sentiment_raises_weight_and_negative_lowers_it():
    scores = pd.DataFrame({"sentiment": 0.0, "confidence": 0.0}, index=config.TICKERS)
    scores.loc["AAPL"] = [0.8, 0.9]
    scores.loc["BA"] = [-0.8, 0.9]
    weights = rebalancer.target_weights(scores, rebalancer.Params())
    assert weights["AAPL"] > EQUAL > weights["BA"]
    assert weights.sum() == pytest.approx(1.0)


def test_update_rebalances_from_the_signal_stream(published):
    history = rebalancer.update(now=pd.Timestamp(T0) + pd.Timedelta(hours=6))

    totals = history.groupby("timestamp")["weight"].sum()
    assert totals.between(1 - 1e-5, 1 + 1e-5).all()  # every rebalance is fully invested
    assert history["weight"].between(config.MIN_WEIGHT - 1e-6, config.MAX_WEIGHT + 1e-6).all()

    latest = rebalancer.current_weights(history).set_index("ticker")["weight"]
    assert latest["AAPL"] > EQUAL > latest["BA"]
    assert config.WEIGHTS_FILE.exists()
    assert config.REBALANCE_LOG.exists()


def test_signals_published_after_t_are_not_used(published):
    """No look-ahead: a rebalance before the first signal stays at equal weight."""
    history = rebalancer.update(now=pd.Timestamp(T0) - pd.Timedelta(hours=1))
    assert history["weight"].round(6).eq(round(EQUAL, 6)).all()
