"""NLP layer of the Risk Engine: entity linking, sentiment, event classification and impact scoring.

Output of analyze(): one structured signal per (document, company) with
  sentiment_score  float in [-1, 1]   (transformer: P(positive) - P(negative))
  event_type       categorical label  (Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, ...)
  impact_score     int in [1, 10]     (predicted severity of the market impact)
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone

import config
from ingestion import clean_text

# --- Entity linking ---------------------------------------------------------------------------------------


def _build_entity_patterns() -> dict:
    """One compiled regex per ticker: company aliases, $cashtag, "(KO)", "NYSE:KO" and safe bare tickers."""
    patterns = {}
    for ticker, info in config.UNIVERSE.items():
        alias = "|".join(re.escape(a) for a in info["aliases"])
        parts = [
            rf"(?<![A-Za-z0-9])(?:{alias})(?![A-Za-z0-9])",  # company names (case-sensitive)
            rf"(?i:\${ticker})\b",  # $AAPL cashtag
            rf"\({ticker}\)",  # "Coca-Cola (KO)"
            rf"(?:NYSE|NASDAQ|Nasdaq)\s*:\s*{ticker}\b",  # "NYSE:KO"
        ]
        if ticker in config.BARE_TICKERS:
            parts.append(rf"(?<![A-Za-z0-9$]){ticker}(?![A-Za-z0-9])")  # bare upper-case ticker
        patterns[ticker] = re.compile("|".join(parts))
    return patterns


ENTITY_PATTERNS = _build_entity_patterns()


def link_entities(text: str) -> list[str]:
    """Return the universe tickers mentioned in a piece of text."""
    return [t for t, p in ENTITY_PATTERNS.items() if p.search(text or "")]


# --- Sentiment --------------------------------------------------------------------------------------------

# fmt: off
FINANCE_LEXICON = {  # VADER scale (-4..4): finance words that generic VADER misses or misreads
    "beat": 2.0, "beats": 2.0, "surge": 2.5, "surges": 2.5, "soar": 2.8, "soars": 2.8, "jump": 1.8,
    "jumps": 1.8, "rally": 2.2, "rallies": 2.2, "gain": 1.6, "gains": 1.6, "rise": 1.3, "rises": 1.3,
    "record": 1.5, "upgrade": 2.0, "upgraded": 2.0, "upgrades": 2.0, "outperform": 2.0, "bullish": 2.8,
    "growth": 1.5, "profit": 1.5, "strong": 1.8, "raises": 1.2, "breakthrough": 2.5, "approval": 2.0,
    "approved": 2.0, "buyback": 1.5, "moon": 2.0, "undervalued": 1.5, "higher": 1.0, "up": 0.8,
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "plunge": -3.0, "plunges": -3.0, "plummet": -3.0,
    "slump": -2.5, "tumble": -2.5, "tumbles": -2.5, "drop": -1.8, "drops": -1.8, "fall": -1.6,
    "falls": -1.6, "fell": -1.6, "decline": -1.6, "declines": -1.6, "slide": -1.6, "slides": -1.6,
    "sink": -2.0, "sinks": -2.0, "loss": -2.0, "losses": -2.0, "downgrade": -2.0, "downgraded": -2.0,
    "downgrades": -2.0, "underperform": -2.0, "bearish": -2.8, "lawsuit": -2.0, "probe": -1.5,
    "investigation": -1.5, "recall": -2.0, "layoffs": -2.2, "cuts": -1.2, "bankruptcy": -3.5,
    "default": -3.0, "fraud": -3.5, "fined": -2.0, "sanctions": -2.0, "tariff": -1.5, "tariffs": -1.5,
    "recession": -2.5, "weak": -1.8, "warning": -1.8, "warns": -1.8, "crash": -3.2, "selloff": -2.5,
    "sell-off": -2.5, "dump": -2.0, "overvalued": -1.5, "halt": -1.8, "delay": -1.5, "delays": -1.5,
    "shortage": -1.8, "outage": -2.0, "lower": -1.0, "down": -0.8, "puts": -0.6, "short": -0.8,
}
# fmt: on


class TransformerSentiment:
    """Financial-news sentiment transformer (default: DistilRoBERTa fine-tuned on Financial PhraseBank)."""

    def __init__(self, name=config.SENTIMENT_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        # set explicitly: only used when decoding (we never decode), and it silences a transformers 4.44 warning
        self.tokenizer = AutoTokenizer.from_pretrained(name, clean_up_tokenization_spaces=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labels = {int(i): lab.lower() for i, lab in self.model.config.id2label.items()}
        self.pos = next(i for i, lab in labels.items() if lab.startswith("pos"))
        self.neg = next(i for i, lab in labels.items() if lab.startswith("neg"))
        self.labels = labels

    def predict(self, texts, batch_size=32):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[i : i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt"
                )
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).tolist()
                for p in probs:
                    best = max(range(len(p)), key=p.__getitem__)
                    out.append({"score": p[self.pos] - p[self.neg], "label": self.labels[best], "confidence": p[best]})
        return out


class VaderSentiment:
    """Rule-based fallback (no download needed): VADER + a small finance lexicon."""

    def __init__(self, finance_lexicon=True):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.analyzer = SentimentIntensityAnalyzer()
        if finance_lexicon:
            self.analyzer.lexicon.update(FINANCE_LEXICON)

    def predict(self, texts):
        out = []
        for t in texts:
            c = self.analyzer.polarity_scores(t)["compound"]
            label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
            out.append({"score": c, "label": label, "confidence": abs(c)})
        return out


class SentimentModel:
    """Transformer model when torch/transformers are available, otherwise the VADER-finance fallback.

    backend: "vader" forces the light fallback; any other value is a Hugging Face model id
    (default config.SENTIMENT_MODEL). Can also be set with the env variable RISKPULSE_MODEL.
    """

    def __init__(self, backend: str | None = None):
        backend = backend or os.getenv("RISKPULSE_MODEL") or config.SENTIMENT_MODEL
        self.error = None
        self._impl = None
        if backend.lower() != "vader":
            try:
                self._impl = TransformerSentiment(backend)
                self.name = config.MODEL_ALIASES.get(backend, backend)
            except Exception as exc:  # no torch installed, no internet for the first download, ...
                reason = f"{exc.__class__.__name__}: {str(exc)[:120]}"
                self.error = f"Transformer model unavailable ({reason}); using VADER"
        if self._impl is None:
            self._impl, self.name = VaderSentiment(), "vader-finance"

    def predict(self, texts: list[str]) -> list[dict]:
        return self._impl.predict([t if t.strip() else "." for t in texts]) if texts else []


def sentiment_text(doc: dict) -> str:
    """Headlines carry the sentiment of news; social posts are short, so use the whole body."""
    if doc["source_type"] == "social":
        text = re.sub(r"https?://\S+", " ", doc["text"])
        text = re.sub(r"\$([A-Za-z]{1,6})\b", r"\1", text)  # $AAPL -> AAPL
        return re.sub(r"\s+", " ", text).strip()[:600]
    return doc["title"]


# --- Event classification ---------------------------------------------------------------------------------

# (pattern, weight) per event type. The summed weights of the matching keywords decide the label;
# ties are broken by the PRIORITY order.
EVENT_RULES = {
    "Credit Event": [
        (r"\bdefault(s|ed|ing)?\b", 1.5),
        (r"\bbankrupt(cy|cies)?\b", 2),
        (r"chapter 11", 2),
        (r"\binsolven", 2),
        (r"credit (rating|outlook|watch)", 2),
        (r"\bjunk\b", 1.5),
        (r"\b(moody'?s|fitch)\b", 1.5),
        (r"\b(rating|ratings) (cut|downgrade)", 2),
        (r"\bdowngrade[sd]?\b.{0,40}\b(credit|debt|bonds?)\b", 2),
        (r"debt restructur", 2),
        (r"missed .{0,20}payment", 2),
        (r"credit (spread|risk|crunch|crisis|losses)", 1.5),
        (r"liquidity (crisis|crunch|squeeze)", 1.5),
        (r"\bbail-?out\b", 1.5),
        (r"\bcovenant", 1),
        (r"\b(debt|bond) (sale|offering|issuance)\b", 1),
        (r"\bwrite-?downs?\b", 1),
    ],
    "Geopolitical": [
        (r"(?<!price )(?<!bidding )(?<!talent )(?<!culture )(?<!neighbor )(?<!streaming )\bwars?\b", 1.5),
        (r"\bsanction(s|ed)?\b", 1.5),
        (r"\btariffs?\b", 1.5),
        (r"trade (war|tension|dispute|deal|talks|truce)", 1.5),
        (r"export (control|ban|curb|restriction)s?", 1.5),
        (r"\bembargo", 1.5),
        (r"\bmilitary\b", 1.2),
        (r"\binvasion\b", 1.5),
        (r"\bmissiles?\b", 1.5),
        (r"\bgeopolitic", 1.5),
        (r"\bconflict\b", 1),
        (r"\bnato\b", 1),
        (r"\bterror", 1.5),
        (r"\b(tensions?|summit|diplomat\w*|retaliat\w*|ceasefire|troops|airstrikes?|foreign minister)\b", 1.0),
        (r"\bnuclear (weapons?|tests?|talks|deal|program\w*)\b", 1.5),
        (
            r"\b(china|chinese|beijing|taiwan|russia|putin|ukraine|iran|israel|gaza|middle east|"
            r"north korea|kremlin|india)\b",
            0.4,
        ),
        (r"\b(election|white house|trump administration|lawmakers|congress)\b", 0.4),
    ],
    "Macroeconomic": [
        (r"\binflation\b", 1.2),
        (r"(?-i:\b(CPI|PPI|GDP|PMI|FOMC|ECB)\b)", 1.2),
        (r"interest rates?", 1.2),
        (r"\brate (cut|hike|decision)s?\b", 1.2),
        (r"(?-i:\bFed\b)", 1.2),
        (r"federal reserve", 1.2),
        (r"\bpowell\b", 1),
        (r"\brecession", 1.2),
        (r"\bunemployment|jobs report|payrolls|jobless claims", 1.2),
        (r"(treasury|bond) yields?|10-year", 1),
        (r"central bank", 1),
        (r"monetary policy", 1.2),
        (r"\beconom(y|ic|ies)\b", 0.7),
        (r"consumer (spending|confidence|sentiment)", 1),
        (r"oil prices?", 1),
        (r"government shutdown", 1.2),
        (r"\b(deficit|stimulus|soft landing)\b", 0.8),
        (r"repo rate|basis points|(?-i:\b(RBI|BoE|BoJ|bps)\b)", 1.2),
        (r"\bcrude\b", 0.8),
    ],
    "Merger/Acquisition": [
        (r"\bacquir(e|es|ed|ing)\b", 1.5),
        (r"\bacquisitions?\b", 1.5),
        (r"\bmerg(e|er|ers|ing)\b", 1.5),
        (r"\btakeover", 1.5),
        (r"\bbuyout", 1.5),
        (r"\bdeal to (buy|acquire)", 1.5),
        (r"\bbid for\b", 1.2),
        (r"\bstake in\b", 1),
        (r"\bdivest", 1.3),
        (r"\bspin-?off|spins? off", 1.3),
        (r"\bjoint venture\b", 1),
        (r"\bsell(s|ing)? .{0,30}\b(unit|division|business|stake)\b", 1.2),
        (r"\b(explor|consider|weigh|reopen|resum)\w* .{0,20}\b(sale|bid|merger)\b", 1.2),
        (r"\bipo\b", 1.0),
    ],
    "Product Launch": [
        (r"\blaunch(es|ed|ing)?\b", 1.2),
        (r"\bunveil(s|ed|ing)?\b", 1.3),
        (r"\bintroduc(e|es|ed|ing)\b", 1),
        (r"\bdebut(s|ed)?\b", 1.2),
        (r"\brollout|rolls? out\b", 1.1),
        (r"\bnew (product|model|chip|phone|iphone|feature|device|service|drug|ai|vehicle|car)s?\b", 1),
        (r"\b(pre-?orders?|release date)\b", 1.2),
        (r"fda (approv|clear)|approval .{0,20}(fda|drug)", 1.2),
        (r"\b(iphone|ipad|gpu|chatbot|robotaxi|cybercab|blackwell|rubin|gemini|copilot|llama)\b", 0.6),
        (r"phase (1|2|3|i{1,3})\b|clinical trial|trial (data|results)|long-term data", 1.0),
    ],
    "Earnings": [
        (r"\bearnings\b", 1.3),
        (r"quarterly (results|profit|revenue|report)", 1.3),
        (r"\brevenues?\b", 0.9),
        (r"\bprofits?\b", 0.9),
        (r"(?-i:\bEPS\b)", 1.2),
        (r"\bguidance\b", 1.2),
        (r"\b(beats?|miss(es|ed)?|tops?|topped)\b.{0,15}\b(estimates|expectations|forecasts?)", 1.5),
        (r"(?-i:\bQ[1-4]\b)", 0.8),
        (r"\boutlook\b", 0.7),
        (r"\bdividends?\b", 0.8),
        (r"\bbuybacks?|share repurchase", 0.8),
        (r"\b(sales|margins?)\b", 0.6),
    ],
    "Legal/Regulatory": [
        (r"\blawsuits?\b", 1.4),
        (r"\bsue[sd]?\b|\bsuing\b", 1.4),
        (r"(?-i:\b(SEC|DOJ|FTC|FAA|FDA|EU)\b)", 0.8),
        (r"justice department", 1.2),
        (r"\bantitrust\b", 1.5),
        (r"\bregulat(or|ors|ory|ion|ions)\b", 1),
        (r"\bfined?\b.{0,20}\$|\bfined\b", 1.2),
        (r"\bprobe[sd]?\b", 1.3),
        (r"\binvestigat", 1.2),
        (r"\bsettle(ment|s|d)?\b", 1),
        (r"\bcourt\b|\bjudge\b|\bruling\b|\bverdict\b", 1.1),
        (r"\brecalls?\b|\brecalled\b", 1.3),
        (r"\bban(s|ned)?\b", 0.8),
        (r"\bpenalt", 1),
        (r"pleads? guilty|\bindict\w*|\bcharged with\b|\bjury\b|\bscrutiny\b", 1.2),
    ],
    "Analyst Rating": [
        (r"\bupgrade[sd]?\b", 1.2),
        (r"\bdowngrade[sd]?\b", 1.2),
        (r"price target|(?-i:\bPT\b)", 1.2),
        (r"\b(outperform|underperform|overweight|underweight|equal-?weight|market perform)\b", 1.2),
        (r"\b(buy|sell|hold|neutral) rating\b", 1.2),
        (r"\breiterat", 1),
        (r"initiat(es|ed) coverage", 1.2),
        (r"\banalysts?\b", 0.6),
        (r"turns? (bullish|bearish)", 1.2),
        (r"research reports?", 1.2),
        (r"\b(cuts?|raises?|lifts?|trims?|tweaks?|boosts?|hikes?)\b.{0,25}\btargets?\b", 1.2),
        (r"(bull|bear) of the day|\bzacks rank\b", 1.0),
    ],
}
PRIORITY = [
    "Credit Event",
    "Geopolitical",
    "Merger/Acquisition",
    "Legal/Regulatory",
    "Macroeconomic",
    "Product Launch",
    "Earnings",
    "Analyst Rating",
]
EVENT_TYPES = PRIORITY + ["General"]
_COMPILED = {e: [(re.compile(p, re.I), w) for p, w in rules] for e, rules in EVENT_RULES.items()}


def classify_event(text: str) -> tuple[str, float, list[str], float]:
    """Return (event_type, confidence 0..1, matched keywords, evidence score = summed keyword weights)."""
    scores, hits = {}, {}
    for event, rules in _COMPILED.items():
        for rx, w in rules:
            m = rx.search(text)
            if m:
                scores[event] = scores.get(event, 0) + w
                hits.setdefault(event, []).append(m.group(0).lower())
    if not scores:
        return "General", 0.0, [], 0.0
    best = max(PRIORITY, key=lambda e: (scores.get(e, 0), -PRIORITY.index(e)))
    return best, round(scores[best] / sum(scores.values()), 2), hits[best][:5], round(scores[best], 2)


# --- Impact score -----------------------------------------------------------------------------------------

BASE_IMPACT = {
    "Credit Event": 6,
    "Geopolitical": 5,
    "Macroeconomic": 5,
    "Merger/Acquisition": 5,
    "Legal/Regulatory": 4,
    "Earnings": 3,
    "Analyst Rating": 3,
    "Product Launch": 3,
    "General": 1,
}
SOURCE_ADJ = {"news": 0.5, "global_news": 0.5, "social": -1.0}
SOCIAL_MAX_IMPACT = 7  # a single social post never counts as an 8-10 severity event
INTENSITY = re.compile(
    r"\b(plung\w*|plummet\w*|soar\w*|surg\w*|crash\w*|collaps\w*|tumbl\w*|skyrocket\w*|record|halt\w*|"
    r"bankrupt\w*|default\w*|war|invasion|sanction\w*|emergency|crisis|scandal|fraud|recall\w*|layoffs?|"
    r"billion|trillion|biggest|worst|historic|massive|shock\w*|slash\w*)\b",
    re.I,
)
MARKET_EVENTS = {"Macroeconomic", "Geopolitical", "Credit Event"}
MARKET_MIN_EVIDENCE = 1.0  # a market-wide signal needs one strong keyword; a country name is not enough


def impact_score(event, sentiment, text, source_type, meta=None, market_wide=False) -> int:
    """Transparent severity model (1-10): event base + tone strength + intensity words + reach.

    base(event type: Credit 6 ... General 1) + 2*|sentiment| + up to 2 intensity words ("plunge", "default",
    "billion", ...) + source (news +0.5, social -1) + syndication coverage (up to +1.5) + viral post (+0.5)
    + market-wide (+0.5); social posts are capped at 7.
    """
    meta = meta or {}
    s = BASE_IMPACT[event] + 2 * abs(sentiment)
    s += min(2, len(INTENSITY.findall(text)))
    s += SOURCE_ADJ.get(source_type, 0)
    s += min(1.5, math.log2(max(1, meta.get("coverage", 1))) / 2)  # syndicated by many outlets
    if source_type == "social" and ((meta.get("likes") or 0) >= 10 or (meta.get("followers") or 0) >= 5000):
        s += 0.5  # viral / influential post
    if market_wide:
        s += 0.5
    cap = SOCIAL_MAX_IMPACT if source_type == "social" else 10
    return int(min(cap, max(1, math.floor(s + 0.5))))


# --- From documents to signals ----------------------------------------------------------------------------


def resolve_tickers(doc: dict, text: str) -> list[str]:
    """The universe companies a document is about."""
    if doc["source"] == "stocktwits":  # $cashtags are explicit
        hints = [t for t in doc.get("tickers_hint", []) if t in config.UNIVERSE]
        return sorted(set(hints) | set(link_entities(text)))
    # news: the company must be named in the HEADLINE. Yahoo's per-ticker feeds also carry loosely related
    # stories (Pfizer news in the JNJ feed) and summaries name companies in passing (JPMorgan as a lender),
    # so neither the feed tag nor a mention in the summary is trusted.
    return link_entities(doc["title"])


def analyze(docs: list[dict], model: SentimentModel) -> list[dict]:
    """Turn raw documents into structured, machine-readable risk signals."""
    if not docs:
        return []
    preds = model.predict([sentiment_text(d) for d in docs])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signals = []
    for doc, pred in zip(docs, preds, strict=True):
        text = clean_text(doc["text"])
        event, event_conf, keywords, evidence = classify_event(f"{doc['title']} {text}")
        tickers = resolve_tickers(doc, text)
        market_wide = False
        if not tickers:
            if event not in MARKET_EVENTS or evidence < MARKET_MIN_EVIDENCE:
                continue  # not about our universe or the market
            tickers, market_wide = [config.MARKET], True  # macro / geopolitical / credit news
        score = round(float(pred["score"]), 4)
        impact = impact_score(event, score, text, doc["source_type"], doc["meta"], market_wide)
        for ticker in tickers:
            signals.append(
                {
                    "signal_id": f"{doc['doc_id']}|{ticker}",
                    "doc_id": doc["doc_id"],
                    "timestamp": doc["published_at"],
                    "ingested_at": now,
                    "source": doc["source"],
                    "source_type": doc["source_type"],
                    "ticker": ticker,
                    "company": config.UNIVERSE.get(ticker, {}).get("name", "Market-wide"),
                    "headline": doc["title"][:200],
                    "text": text[:400],
                    "url": doc["url"],
                    "sentiment_score": score,
                    "sentiment_label": pred["label"],
                    "sentiment_confidence": round(float(pred["confidence"]), 3),
                    "event_type": event,
                    "event_confidence": event_conf,
                    "event_keywords": keywords,
                    "impact_score": impact,
                    "user_sentiment": doc["meta"].get("user_sentiment"),
                    "model": model.name,
                }
            )
    return signals


def analyze_text(text: str, model: SentimentModel, source_type: str = "news") -> dict:
    """Analyse an arbitrary piece of text (used by the API and the dashboard playground)."""
    doc = {
        "doc_id": "adhoc",
        "source": "user",
        "source_type": source_type,
        "title": text,
        "text": text,
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": "",
        "tickers_hint": [],
        "meta": {},
    }
    pred = model.predict([sentiment_text(doc)])[0]
    event, conf, keywords, evidence = classify_event(text)
    tickers = link_entities(text)
    market_wide = not tickers and event in MARKET_EVENTS and evidence >= MARKET_MIN_EVIDENCE
    return {
        "companies": [{"ticker": t, "name": config.UNIVERSE[t]["name"]} for t in tickers]
        or ([{"ticker": config.MARKET, "name": "Market-wide"}] if market_wide else []),
        "sentiment_score": round(float(pred["score"]), 4),
        "sentiment_label": pred["label"],
        "event_type": event,
        "event_confidence": conf,
        "event_keywords": keywords,
        "impact_score": impact_score(event, pred["score"], text, source_type, {}, market_wide),
        "model": model.name,
    }
