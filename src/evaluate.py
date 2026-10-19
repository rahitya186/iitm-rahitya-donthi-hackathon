"""Evaluation: is the NLP engine better than a naive approach?

  1. Sentiment, 3 classes, on 2,388 labeled finance tweets     data/eval/finance_tweets_valid.csv
     (huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment, validation split, MIT licence)
  2. Direction vs. StockTwits authors' own Bullish/Bearish tags  data/raw/stocktwits.json
  3. Event classification on hand-labeled real headlines        data/eval/event_labels.csv
     (split "dev": 70 headlines used while writing the keyword rules; split "test": 60 held-out headlines,
      labeled before the classifier was run on them. Only the test split is reported.)

Writes data/eval/results.json.   Run:  python src/main.py evaluate  [--with-finbert]
"""

from __future__ import annotations

import json
import time

import pandas as pd

import config
import ingestion
import nlp

TWEETS_URL = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_valid.csv"
TWEETS_FILE = config.EVAL_DIR / "finance_tweets_valid.csv"
EVENTS_FILE = config.EVAL_DIR / "event_labels.csv"
RESULTS_FILE = config.EVAL_DIR / "results.json"
TWEET_LABELS = {0: "negative", 1: "positive", 2: "neutral"}


# --- Metrics ----------------------------------------------------------------------------------------------


def accuracy(gold: list, pred: list) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def macro_f1(gold: list, pred: list) -> float:
    """Unweighted mean of the per-class F1 scores (every class counts equally)."""
    scores = []
    for c in sorted(set(gold)):
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


# --- Benchmarks -------------------------------------------------------------------------------------------


def _models(with_finbert: bool) -> dict:
    models = {
        "Naive baseline: generic VADER lexicon": nlp.VaderSentiment(finance_lexicon=False),
        "Fallback: VADER + finance lexicon": nlp.VaderSentiment(finance_lexicon=True),
    }
    engine_model = nlp.SentimentModel()
    if engine_model.name != "vader-finance":
        models[f"Engine: {engine_model.name}"] = engine_model
    if with_finbert:
        models["Comparison: ProsusAI/finbert"] = nlp.SentimentModel("ProsusAI/finbert")
    return models


def eval_tweets(models: dict) -> list[dict]:
    """3-class sentiment on labeled finance tweets (downloaded once if the file is missing)."""
    if not TWEETS_FILE.exists():
        config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
        pd.read_csv(TWEETS_URL).to_csv(TWEETS_FILE, index=False)
    df = pd.read_csv(TWEETS_FILE)
    gold = df["label"].map(TWEET_LABELS).tolist()
    texts = [nlp.sentiment_text({"source_type": "social", "text": t, "title": t}) for t in df["text"]]
    always_neutral = ["neutral"] * len(gold)
    rows = [
        {
            "model": "Majority class (always neutral)",
            "accuracy": accuracy(gold, always_neutral),
            "macro_f1": macro_f1(gold, always_neutral),
            "docs_per_second": None,
        }
    ]
    for name, model in models.items():
        started = time.time()
        pred = [p["label"] for p in model.predict(texts)]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy(gold, pred),
                "macro_f1": macro_f1(gold, pred),
                "docs_per_second": round(len(texts) / (time.time() - started), 1),
            }
        )
    return rows


def eval_stocktwits(models: dict) -> list[dict]:
    """Sentiment direction vs. the Bullish/Bearish tag the post's author chose."""
    docs = [
        d for d in ingestion.load_snapshot("stocktwits") if d["meta"].get("user_sentiment") in ("Bullish", "Bearish")
    ]
    if not docs:
        return []
    gold = [d["meta"]["user_sentiment"] for d in docs]
    texts = [nlp.sentiment_text(d) for d in docs]
    rows = []
    for name, model in models.items():
        pred = ["Bullish" if p["score"] > 0 else "Bearish" if p["score"] < 0 else "none" for p in model.predict(texts)]
        recalls = [
            sum(g == c and p == c for g, p in zip(gold, pred, strict=True)) / max(1, gold.count(c))
            for c in ("Bullish", "Bearish")
        ]
        rows.append(
            {
                "model": name,
                "n": len(docs),
                "bullish_share": round(gold.count("Bullish") / len(gold), 3),
                "accuracy": accuracy(gold, pred),
                "balanced_accuracy": sum(recalls) / 2,
            }
        )
    return rows


def eval_events(split: str = "test") -> dict:
    """Event classification vs. hand labels, compared with always predicting the most common label."""
    df = pd.read_csv(EVENTS_FILE)
    df = df[df["split"] == split]
    gold = df["label"].tolist()
    pred = [nlp.classify_event(h)[0] for h in df["headline"]]
    majority = max(set(gold), key=gold.count)
    errors = [
        {"headline": h, "label": g, "predicted": p}
        for h, g, p in zip(df["headline"], gold, pred, strict=True)
        if g != p
    ]
    return {
        "split": split,
        "n": len(df),
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "baseline": f"always '{majority}'",
        "baseline_accuracy": accuracy(gold, [majority] * len(gold)),
        "baseline_macro_f1": macro_f1(gold, [majority] * len(gold)),
        "errors": errors,
    }


def main(with_finbert: bool = False) -> dict:
    models = _models(with_finbert)
    results = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sentiment_finance_tweets": eval_tweets(models),
        "sentiment_vs_stocktwits_user_tags": eval_stocktwits(models),
        "event_classification": eval_events(),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=1), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\n1) Sentiment on 2,388 labeled finance tweets (3 classes)")
    print(pd.DataFrame(results["sentiment_finance_tweets"]).round(3).to_string(index=False))
    if results["sentiment_vs_stocktwits_user_tags"]:
        print("\n2) Direction vs StockTwits users' own Bullish/Bearish tags")
        print(pd.DataFrame(results["sentiment_vs_stocktwits_user_tags"]).round(3).to_string(index=False))
    ev = results["event_classification"]
    print(
        f"\n3) Event classification on {ev['n']} held-out hand-labeled headlines: "
        f"accuracy {ev['accuracy']:.1%}, macro-F1 {ev['macro_f1']:.2f}  "
        f"(baseline {ev['baseline']}: {ev['baseline_accuracy']:.1%}, macro-F1 {ev['baseline_macro_f1']:.2f})"
    )
    print(f"\nSaved {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()
