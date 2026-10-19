"""Module A - Tactical Index Rebalancing.

Subscribes to the Sentiment Score published by the NLP Risk Engine (data/signals.jsonl) and tilts a mock index
of 15 S&P 100 stocks away from equal weight: positive sentiment -> weight up, negative sentiment -> weight down.

For every rebalance time t and stock i:
  1. take the stock's signals published in the LOOKBACK_HOURS before t
  2. weight each signal by recency, impact and source:
        w_j = 0.5 ** (age_hours / HALF_LIFE) * impact_j * source_weight_j        (social posts count 0.6x)
  3. stock sentiment   S_i = sum(w_j * s_j) / sum(w_j)                           (in -1 .. 1)
     confidence        c_i = E_i / (E_i + K),  E_i = sum of recency weights       (thin evidence -> small tilt)
  4. target weight     w_i = 1/N * (1 + TILT * c_i * S_i), clipped to [MIN_WEIGHT, MAX_WEIGHT], sum = 100 %
  5. turnover control  new weight = old + SMOOTHING * (target - old)              (half-way each rebalance)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd

import config
import engine

PARAMS_FILE = config.DATA_DIR / "module_a_params.json"  # settings changed from the dashboard
LOG_THRESHOLD = 0.0025  # log weight moves of 0.25 percentage points or more


@dataclass
class Params:
    """Module A settings; defaults come from src/config.py."""

    tilt: float = config.TILT
    half_life: float = config.HALF_LIFE_HOURS
    lookback: float = config.LOOKBACK_HOURS
    min_weight: float = config.MIN_WEIGHT
    max_weight: float = config.MAX_WEIGHT
    evidence_k: float = config.EVIDENCE_K
    smoothing: float = config.SMOOTHING


def load_params() -> Params:
    if PARAMS_FILE.exists():
        saved = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        return Params(**{k: v for k, v in saved.items() if k in Params.__dataclass_fields__})
    return Params()


def save_params(p: Params) -> None:
    PARAMS_FILE.write_text(json.dumps(asdict(p), indent=1), encoding="utf-8")


# --- Core maths -------------------------------------------------------------------------------------------


def stock_scores(signals: pd.DataFrame, t: pd.Timestamp, p: Params) -> pd.DataFrame:
    """Recency-, impact- and source-weighted sentiment per stock, using only signals known at time t."""
    in_window = (signals["timestamp"] <= t) & (signals["timestamp"] > t - pd.Timedelta(hours=p.lookback))
    window = signals[in_window & signals["ticker"].isin(config.TICKERS)]
    age_hours = (t - window["timestamp"]).dt.total_seconds() / 3600
    decay = 0.5 ** (age_hours / p.half_life)
    source_weight = window["source_type"].map(config.SOURCE_WEIGHT).fillna(1.0)
    w = decay * window["impact_score"].astype(float) * source_weight
    frame = pd.DataFrame(
        {
            "ticker": window["ticker"],
            "decay": decay,
            "w": w,
            "ws": w * window["sentiment_score"],
            "headline": window["headline"],
        }
    )
    rows = []
    for ticker in config.TICKERS:
        g = frame[frame["ticker"] == ticker]
        if g.empty or g["w"].sum() == 0:
            rows.append(
                {
                    "ticker": ticker,
                    "sentiment": 0.0,
                    "confidence": 0.0,
                    "n_signals": 0,
                    "driver_up": "",
                    "driver_down": "",
                }
            )
            continue
        evidence = g["decay"].sum()
        rows.append(
            {
                "ticker": ticker,
                "sentiment": g["ws"].sum() / g["w"].sum(),
                "confidence": evidence / (evidence + p.evidence_k),
                "n_signals": len(g),
                # the strongest positive / negative signal explains an up / down move in the rebalance log
                "driver_up": g.loc[g["ws"].idxmax(), "headline"] if (g["ws"] > 0).any() else "",
                "driver_down": g.loc[g["ws"].idxmin(), "headline"] if (g["ws"] < 0).any() else "",
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def apply_bounds(w: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip to the position limits and renormalise to 100 % (iteratively, so the limits still hold)."""
    w = w / w.sum()
    for _ in range(100):
        w = w.clip(lo, hi)
        gap = 1 - w.sum()
        if abs(gap) < 1e-12:
            break
        free = (w > lo + 1e-12) if gap < 0 else (w < hi - 1e-12)
        w[free] += gap * w[free] / w[free].sum()
    return w


def target_weights(scores: pd.DataFrame, p: Params) -> pd.Series:
    """Equal weight tilted by confidence x sentiment, within the position limits."""
    base = 1 / len(scores)
    raw = base * (1 + p.tilt * scores["confidence"] * scores["sentiment"])
    return apply_bounds(raw.clip(lower=1e-6), p.min_weight, p.max_weight)


# --- History ----------------------------------------------------------------------------------------------


def _read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _write_csv(df: pd.DataFrame, path) -> None:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(path, index=False)


def load_history() -> pd.DataFrame:
    """Index weights per rebalance and stock."""
    if not config.WEIGHTS_FILE.exists():
        return pd.DataFrame(columns=["timestamp", "ticker", "weight", "sentiment", "confidence", "n_signals"])
    return _read_csv(config.WEIGHTS_FILE)


def load_log() -> pd.DataFrame:
    """Every weight move of LOG_THRESHOLD or more, with the headline that drove it."""
    if not config.REBALANCE_LOG.exists():
        return pd.DataFrame()
    return _read_csv(config.REBALANCE_LOG)


def _schedule(history: pd.DataFrame, now: pd.Timestamp, p: Params) -> list[pd.Timestamp]:
    """Hourly grid since the last rebalance (or over the look-back window on the first run), plus 'now'."""
    start = (now - pd.Timedelta(hours=p.lookback)).floor("h") if history.empty else history["timestamp"].max()
    grid = pd.date_range(start.ceil("h"), now.floor("h"), freq=pd.Timedelta(hours=config.STEP_HOURS))
    steps = [t for t in grid if t > start]
    if (not steps or now - steps[-1] >= pd.Timedelta(minutes=1)) and now > start:
        steps.append(now)
    return steps


def update(now=None, rebuild: bool = False, params: Params | None = None, log=print) -> pd.DataFrame:
    """Consume the signal stream and append the new rebalances to the weight history.

    now      rebalance up to this time (default: the current time; offline replays pass data_now())
    rebuild  recompute the whole history, e.g. after the settings changed
    params   settings to use and save (default: the saved settings, else src/config.py)
    """
    p = params or load_params()
    if params is not None:
        save_params(params)
    signals = engine.read_signals()
    previous = load_history()
    if now is None and rebuild and not previous.empty:
        now = previous["timestamp"].max()  # recompute the same time window with the new settings
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    history = pd.DataFrame() if rebuild else previous
    moves = pd.DataFrame() if rebuild else load_log()
    steps = _schedule(history, now, p)

    equal = 1 / len(config.TICKERS)
    if history.empty:
        prev = pd.Series(equal, index=config.TICKERS)
    else:
        last = history[history["timestamp"] == history["timestamp"].max()]
        prev = last.set_index("ticker")["weight"].reindex(config.TICKERS).fillna(equal)

    rows, log_rows = [], []
    for t in steps:
        scores = stock_scores(signals, t, p)
        target = target_weights(scores, p)
        w = prev + p.smoothing * (target - prev)  # partial move = turnover control
        w = w / w.sum()
        for ticker in config.TICKERS:
            s = scores.loc[ticker]
            rows.append(
                {
                    "timestamp": t,
                    "ticker": ticker,
                    "weight": round(float(w[ticker]), 6),
                    "sentiment": round(float(s["sentiment"]), 4),
                    "confidence": round(float(s["confidence"]), 3),
                    "n_signals": int(s["n_signals"]),
                }
            )
            change = w[ticker] - prev[ticker]
            if abs(change) >= LOG_THRESHOLD:
                driver = s["driver_up"] if change > 0 else s["driver_down"]
                log_rows.append(
                    {
                        "timestamp": t,
                        "ticker": ticker,
                        "old_weight": round(float(prev[ticker]), 4),
                        "new_weight": round(float(w[ticker]), 4),
                        "change_pp": round(float(change) * 100, 2),
                        "sentiment": round(float(s["sentiment"]), 3),
                        "n_signals": int(s["n_signals"]),
                        "driver": driver or "older signals fading (decay)",
                    }
                )
        prev = w

    if rows:
        new = pd.DataFrame(rows)
        history = new if history.empty else pd.concat([history, new], ignore_index=True)
    if log_rows:
        new_moves = pd.DataFrame(log_rows)
        moves = new_moves if moves.empty else pd.concat([moves, new_moves], ignore_index=True)
    if not history.empty:
        _write_csv(history, config.WEIGHTS_FILE)
    if not moves.empty:
        _write_csv(moves, config.REBALANCE_LOG)
    elif rebuild:
        config.REBALANCE_LOG.unlink(missing_ok=True)  # no moves under the new settings: drop the stale log
    log(f"  Module A: {len(steps)} rebalance step(s), {len(log_rows)} weight moves >= 0.25 pp")
    return history


def data_now() -> pd.Timestamp | None:
    """Clock for offline replays: the newest signal, so the 72 h window covers the saved data, not today."""
    signals = engine.read_signals()
    return None if signals.empty else signals["timestamp"].max()


def current_weights(history: pd.DataFrame | None = None) -> pd.DataFrame:
    """The latest rebalance, with company names, sectors and the tilt versus equal weight."""
    history = load_history() if history is None else history
    if history.empty:
        return pd.DataFrame()
    last = history[history["timestamp"] == history["timestamp"].max()].copy()
    last["company"] = last["ticker"].map(lambda t: config.UNIVERSE[t]["name"])
    last["sector"] = last["ticker"].map(lambda t: config.UNIVERSE[t]["sector"])
    last["vs_equal_pp"] = (last["weight"] - 1 / len(config.TICKERS)) * 100
    return last.sort_values("weight", ascending=False).reset_index(drop=True)


def turnover(history: pd.DataFrame | None = None) -> pd.Series:
    """One-way turnover per rebalance: 0.5 * sum |w_t - w_t-1|."""
    history = load_history() if history is None else history
    if history.empty:
        return pd.Series(dtype=float)
    wide = history.pivot_table(index="timestamp", columns="ticker", values="weight")
    return (wide.diff().abs().sum(axis=1) / 2).iloc[1:]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module A: tactical index rebalancer")
    parser.add_argument("--rebuild", action="store_true", help="recompute the whole history from the signal stream")
    update(rebuild=parser.parse_args().rebuild)
    print(current_weights()[["ticker", "weight", "sentiment", "n_signals"]].to_string(index=False))
