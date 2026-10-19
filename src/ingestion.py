"""Data ingestion: three free, independent sources normalised to one document schema.

    source          what it provides                                source_type
    -------------   ---------------------------------------------   -----------
    yahoo_finance   company news headlines (Yahoo Finance RSS)      news
    gdelt           global news: companies, macro, geopolitics      global_news
    stocktwits      social-media posts tagged with $cashtags        social

Every live fetch is merged into data/raw/<source>.json, a rolling 7-day snapshot. If a source is down or
rate-limited, the engine falls back to that snapshot, so the pipeline and the demo keep working.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

import config

HEADERS = {"User-Agent": config.USER_AGENT, "Accept": "application/json, application/xml, text/xml, */*"}

SOURCES = {
    "yahoo_finance": "Yahoo Finance RSS (company news)",
    "gdelt": "GDELT DOC 2.0 API (global news & events)",
    "stocktwits": "StockTwits (social media)",
}


# --- Helpers ----------------------------------------------------------------------------------------------


def _sha(text: str) -> str:
    """Short, stable id for a URL or feed GUID."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _iso(dt: datetime) -> str:
    """UTC timestamp in the one format used everywhere, e.g. 2026-10-10T12:35:00Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(text: str | None) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_doc(source, source_type, uid, published, title, text, url, tickers_hint=None, meta=None) -> dict:
    """The common document schema that every connector produces."""
    return {
        "doc_id": f"{source}:{uid}",
        "source": source,
        "source_type": source_type,
        "published_at": _iso(published),
        "title": title,
        "text": text,
        "url": url,
        "tickers_hint": sorted(set(tickers_hint or [])),
        "meta": meta or {},
    }


def _get(url: str, params: dict | None = None) -> requests.Response:
    response = requests.get(url, params=params, headers=HEADERS, timeout=config.HTTP_TIMEOUT)
    response.raise_for_status()
    return response


def _fetch_each(fetch_one, tickers, workers: int) -> list[dict]:
    """Run a per-ticker fetcher in parallel. A failing ticker is skipped instead of stopping the others."""

    def safe(ticker):
        try:
            return fetch_one(ticker)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(safe, tickers))
    return [doc for batch in batches for doc in batch]


def _merge_same_doc(docs: list[dict]) -> list[dict]:
    """One article can appear in several tickers' feeds: keep one copy and merge the ticker hints."""
    merged = {}
    for doc in docs:
        if doc["doc_id"] in merged:
            known = merged[doc["doc_id"]]
            known["tickers_hint"] = sorted(set(known["tickers_hint"]) | set(doc["tickers_hint"]))
        else:
            merged[doc["doc_id"]] = doc
    return list(merged.values())


# --- Source 1: Yahoo Finance RSS --------------------------------------------------------------------------


def _rss_date(value: str | None) -> datetime:
    """Parse an RSS pubDate; use the current time if it is missing or malformed."""
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def fetch_yahoo_finance(tickers=config.TICKERS) -> list[dict]:
    """Company headlines and their RSS summaries, one Yahoo Finance feed per ticker."""

    def fetch_one(ticker: str) -> list[dict]:
        response = _get(config.YAHOO_RSS_URL, {"s": ticker, "region": "US", "lang": "en-US"})
        docs = []
        for item in ET.fromstring(response.content).iter("item"):
            title = clean_text(item.findtext("title"))
            if not title:
                continue
            summary = clean_text(item.findtext("description"))
            link = (item.findtext("link") or "").strip()
            uid = _sha(item.findtext("guid") or link or title)
            text = f"{title}. {summary}" if summary else title
            published = _rss_date(item.findtext("pubDate"))
            docs.append(make_doc("yahoo_finance", "news", uid, published, title, text, link, [ticker]))
        return docs

    docs = _fetch_each(fetch_one, tickers, workers=5)
    if not docs:
        raise RuntimeError("Yahoo Finance RSS returned no items")
    return _merge_same_doc(docs)


# --- Source 2: GDELT DOC 2.0 API --------------------------------------------------------------------------


def fetch_gdelt(queries=config.GDELT_QUERIES, hours=config.LOOKBACK_HOURS) -> list[dict]:
    """Global news from GDELT: one query for our companies, one for macro and geopolitical events."""
    docs, errors = [], []
    for i, (name, query) in enumerate(queries.items()):
        if i:
            time.sleep(6)  # GDELT allows at most one request every 5 seconds
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 250,
            "timespan": f"{hours}h",
            "sort": "datedesc",
        }
        try:
            articles = _gdelt_articles(_get(config.GDELT_URL, params))
        except Exception as exc:  # keep the results of the other query
            errors.append(f"{name}: {exc}")
            continue
        docs += parse_gdelt_articles(articles, name)
    if not docs:
        raise RuntimeError("; ".join(errors) or "GDELT returned no articles")
    return _collapse_syndicated(docs)


def _gdelt_articles(response: requests.Response) -> list[dict]:
    try:
        return json.loads(response.text).get("articles", [])
    except json.JSONDecodeError:
        # GDELT answers rate-limited requests with a plain-text message instead of JSON
        raise RuntimeError(response.text[:120]) from None


def parse_gdelt_articles(articles: list[dict], query_name: str) -> list[dict]:
    """Convert GDELT article records into documents."""
    docs = []
    for article in articles:
        title = _tidy_title(clean_text(article.get("title")))
        if not title:
            continue
        published = datetime.strptime(article["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        meta = {"query": query_name, "domain": article.get("domain"), "country": article.get("sourcecountry")}
        url = article["url"]
        docs.append(make_doc("gdelt", "global_news", _sha(url), published, title, title, url, meta=meta))
    return docs


def _tidy_title(title: str) -> str:
    """Undo GDELT's tokenised titles ("rate to 5 . 50 %", "H - 1B") and drop the appended site name."""
    title = re.sub(r"(\d) ([.,]) (\d)", r"\1\2\3", title)
    title = re.sub(r" ([,.:;?!%)'’])", r"\1", title)
    title = re.sub(r"([($]) ", r"\1", title)
    title = re.sub(r"(['’]) (s|t|re|ve|ll|d)\b", r"\1\2", title)
    # Drop a trailing " - Publisher Name" (then " | Site"): words such as "Economic Times" in a site name
    # would otherwise fool the event classifier.
    for separators in ("-|", "|"):
        match = re.search(rf"\s[{separators}]\s([^-|]{{2,45}})$", title)
        if match:
            words = match.group(1).split()
            if len(words) <= 6 and all(w[:1].isupper() or w[:1].isdigit() for w in words):
                title = title[: match.start()]
    return re.sub(r"\b(\w{1,4}) - (\w)", r"\1-\2", title).strip()  # "H - 1B" -> "H-1B", "low - key"


def _collapse_syndicated(docs: list[dict]) -> list[dict]:
    """The same wire story appears on many sites: keep one copy and count the outlets as its coverage."""
    groups = {}
    for doc in sorted(docs, key=lambda d: d["published_at"]):
        key = re.sub(r"[^a-z0-9]", "", doc["title"].lower())[:80]
        if key in groups:
            groups[key]["meta"]["coverage"] += 1
        else:
            doc["meta"]["coverage"] = 1
            groups[key] = doc
    return list(groups.values())


# --- Source 3: StockTwits ---------------------------------------------------------------------------------

_SPAM = re.compile(
    r"link in bio|discord|telegram|whatsapp|free trial|\d+[- ]day trial|subscribe|giveaway|join (my|our)|"
    r"promo code|dm me|signals? group|% off|€|alerts? (service|room)",
    re.I,
)


def _is_spam(body: str, n_symbols: int) -> bool:
    """Too short, tags too many tickers, or advertises something."""
    words = re.sub(r"https?://\S+|\$[A-Za-z.]+|@\w+", " ", body).split()
    return len(words) < 4 or n_symbols > 6 or bool(_SPAM.search(body))


def fetch_stocktwits(tickers=config.TICKERS) -> list[dict]:
    """The latest posts of each ticker's StockTwits stream, without spam and copy-paste campaigns."""

    def fetch_one(ticker: str) -> list[dict]:
        data = _get(config.STOCKTWITS_URL.format(ticker=ticker)).json()
        docs = []
        for message in data.get("messages", []):
            body = clean_text(message.get("body"))
            symbols = [s.get("symbol", "").upper() for s in message.get("symbols", [])]
            if _is_spam(body, len(symbols)):
                continue
            published = datetime.strptime(message["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            title = re.sub(r"\s+", " ", re.sub(r"https?://\S+", " ", body)).strip()[:140]
            meta = {
                "user_sentiment": ((message.get("entities") or {}).get("sentiment") or {}).get("basic"),
                "likes": (message.get("likes") or {}).get("total", 0),
                "followers": (message.get("user") or {}).get("followers", 0),
            }
            url = f"https://stocktwits.com/message/{message['id']}"
            docs.append(
                make_doc("stocktwits", "social", str(message["id"]), published, title, body, url, symbols, meta)
            )
        return docs

    docs = _fetch_each(fetch_one, tickers, workers=3)  # gentle on the free API
    if not docs:
        raise RuntimeError("StockTwits returned no messages")
    return drop_copy_paste(_merge_same_doc(docs))


def _body_key(doc: dict) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"https?://\S+|\$[a-z.]+", "", doc["text"].lower()))[:120]


def drop_copy_paste(docs: list[dict]) -> list[dict]:
    """Promo campaigns post the same text under different $cashtags: drop every copy of such a text."""
    counts = Counter(_body_key(d) for d in docs)
    return [d for d in docs if counts[_body_key(d)] == 1 or len(_body_key(d)) < 15]


# --- Snapshots and the ingestion entry point --------------------------------------------------------------

FETCHERS = {"yahoo_finance": fetch_yahoo_finance, "gdelt": fetch_gdelt, "stocktwits": fetch_stocktwits}


def snapshot_path(source: str):
    return config.RAW_DIR / f"{source}.json"


def load_snapshot(source: str) -> list[dict]:
    """Documents saved for a source (empty list if there is no snapshot yet)."""
    path = snapshot_path(source)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("documents", [])


def save_snapshot(source: str, docs: list[dict]) -> None:
    """Merge new documents into the rolling snapshot: dedupe by doc_id, keep the last SNAPSHOT_KEEP_DAYS."""
    by_id = {d["doc_id"]: d for d in load_snapshot(source)}
    by_id.update({d["doc_id"]: d for d in docs})
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(days=config.SNAPSHOT_KEEP_DAYS))
    kept = sorted((d for d in by_id.values() if d["published_at"] >= cutoff), key=lambda d: d["published_at"])
    payload = {
        "source": source,
        "description": SOURCES[source],
        "updated_at": _iso(datetime.now(timezone.utc)),
        "documents": kept,
    }
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path(source).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def ingest_all(offline: bool = False) -> tuple[list[dict], dict]:
    """Fetch the three sources in parallel. Returns (documents, status per source)."""

    def run(source: str):
        if offline:
            docs = load_snapshot(source)
            return source, docs, {"mode": "offline snapshot", "documents": len(docs), "error": None}
        started = time.time()
        try:
            docs = FETCHERS[source]()
            save_snapshot(source, docs)
            seconds = round(time.time() - started, 1)
            return source, docs, {"mode": "live", "documents": len(docs), "error": None, "seconds": seconds}
        except Exception as exc:
            docs = load_snapshot(source)
            reason = str(exc).split(" for url")[0][:160]  # keep the message, drop the long URL
            return source, docs, {"mode": "snapshot fallback", "documents": len(docs), "error": reason}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, SOURCES))
    docs = [doc for _, batch, _ in results for doc in batch]
    status = {source: state for source, _, state in results}
    return docs, status
