"""NLP layer: entity linking, event classification, impact score and the signal schema."""

import pytest

import config
import ingestion
import nlp
from conftest import news


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Coca-Cola (KO) shares rise after earnings", ["KO"]),
        ("$TSLA is going to the moon", ["TSLA"]),
        ("NYSE:KO hits a record", ["KO"]),
        ("Meta Platforms unveils new glasses", ["META"]),
        ("Nvidia and Microsoft lead the rally", ["MSFT", "NVDA"]),
        ("She earned a BA degree in economics", []),  # a bare "BA" is too ambiguous
        ("I baked an apple pie", []),  # aliases are case-sensitive
    ],
)
def test_link_entities(text, expected):
    assert sorted(nlp.link_entities(text)) == expected


@pytest.mark.parametrize(
    ("headline", "event"),
    [
        ("Moody's downgrades Boeing to junk as cash burn deepens", "Credit Event"),
        ("US imposes sweeping new tariffs on Chinese electronics", "Geopolitical"),
        ("Fed signals more rate cuts as inflation cools", "Macroeconomic"),
        ("JPMorgan agrees to acquire fintech lender in $2 billion deal", "Merger/Acquisition"),
        ("Nvidia unveils next-generation AI chip", "Product Launch"),
        ("Apple quarterly revenue beats estimates", "Earnings"),
        ("EU opens antitrust probe into Meta", "Legal/Regulatory"),
        ("Morgan Stanley downgrades Tesla and cuts its price target", "Analyst Rating"),
        ("Walmart stock is trending today", "General"),
    ],
)
def test_classify_event(headline, event):
    label, confidence, keywords, _ = nlp.classify_event(headline)
    assert label == event
    assert 0 <= confidence <= 1
    assert bool(keywords) == (event != "General")


def test_impact_score_stays_in_range_and_caps_social_posts():
    loud = "plunge crash default billion"
    for event in nlp.EVENT_TYPES:
        for sentiment in (-1.0, 0.0, 1.0):
            assert 1 <= nlp.impact_score(event, sentiment, loud, "news", {"coverage": 64}, True) <= 10
            assert 1 <= nlp.impact_score(event, sentiment, loud, "social", {"likes": 99}) <= nlp.SOCIAL_MAX_IMPACT
    assert nlp.impact_score("Credit Event", -1.0, "bankruptcy", "news") > nlp.impact_score("General", 0.0, "", "news")


def test_analyze_produces_the_required_fields(vader):
    docs = [
        news("credit", 0, "Moody's downgrades Boeing to junk", ["BA"]),
        news("macro", 0, "Fed raises interest rates as inflation surges"),
        news("noise", 0, "Local bakery wins a regional award"),
    ]
    signals = {s["doc_id"]: s for s in nlp.analyze(docs, vader)}

    assert signals["yahoo_finance:credit"]["ticker"] == "BA"
    assert signals["yahoo_finance:credit"]["event_type"] == "Credit Event"
    assert signals["yahoo_finance:credit"]["sentiment_score"] < 0
    assert signals["yahoo_finance:macro"]["ticker"] == config.MARKET  # macro news without a company
    assert "yahoo_finance:noise" not in signals  # about neither our companies nor the market
    for signal in signals.values():
        assert -1 <= signal["sentiment_score"] <= 1
        assert 1 <= signal["impact_score"] <= 10
        assert signal["event_type"] in nlp.EVENT_TYPES


def test_ingestion_helpers():
    raw = "RBI increases repo rate to 5 . 50 %: How does it impact a consumer ?"
    assert ingestion._tidy_title(raw) == "RBI increases repo rate to 5.50%: How does it impact a consumer?"
    assert (
        ingestion._tidy_title("India Slams Vance H - 1B Remarks - The Economic Times")
        == "India Slams Vance H-1B Remarks"
    )
    assert ingestion._is_spam("FRIDAY'S RESULT LINK IN BIO, DONT MISS NEXT WEEK $SPY", 1)
    assert not ingestion._is_spam("$AAPL Monday is launch day to begin orders for the dual iPhone", 1)
    assert ingestion._rss_date("Fri, 09 Oct 2026 21:12:58 +0000").day == 9
    assert ingestion._rss_date("not a date").tzinfo is not None  # malformed dates fall back to now
