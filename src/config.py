"""Central configuration: file paths, the mock index universe, data sources and model settings."""

from pathlib import Path

# --- Paths ------------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # raw documents per source (also the offline snapshot)
EVAL_DIR = DATA_DIR / "eval"  # labeled data used by src/evaluate.py
SIGNALS_FILE = DATA_DIR / "signals.jsonl"  # engine output: append-only signal stream
SIGNALS_CSV = DATA_DIR / "signals.csv"  # the same signals as a flat table
WEIGHTS_FILE = DATA_DIR / "weights_history.csv"  # Module A output: index weights over time
REBALANCE_LOG = DATA_DIR / "rebalance_log.csv"  # Module A output: why the weights moved
LAST_RUN_FILE = DATA_DIR / "last_run.json"  # status of the most recent engine cycle
PROCESSED_FILE = DATA_DIR / "processed_doc_ids.txt"  # every document already analysed

# --- Mock index: 15 S&P 100 stocks ------------------------------------------------------------------------

COMPANIES = {  # ticker: (company name, sector)
    "AAPL": ("Apple", "Info Tech"),
    "MSFT": ("Microsoft", "Info Tech"),
    "NVDA": ("Nvidia", "Info Tech"),
    "AMZN": ("Amazon", "Cons. Discr."),
    "GOOGL": ("Alphabet", "Comm. Services"),
    "META": ("Meta Platforms", "Comm. Services"),
    "TSLA": ("Tesla", "Cons. Discr."),
    "JPM": ("JPMorgan Chase", "Financials"),
    "BAC": ("Bank of America", "Financials"),
    "XOM": ("Exxon Mobil", "Energy"),
    "JNJ": ("Johnson & Johnson", "Health Care"),
    "LLY": ("Eli Lilly", "Health Care"),
    "WMT": ("Walmart", "Cons. Staples"),
    "KO": ("Coca-Cola", "Cons. Staples"),
    "BA": ("Boeing", "Industrials"),
}
ALIASES = {  # how each company appears in free text (case-sensitive, used by the entity linker)
    "AAPL": ["Apple", "iPhone", "iPad", "Tim Cook"],
    "MSFT": ["Microsoft", "Azure", "Satya Nadella"],
    "NVDA": ["Nvidia", "NVIDIA", "Jensen Huang"],
    "AMZN": ["Amazon", "AWS", "Andy Jassy"],
    "GOOGL": ["Alphabet", "Google", "YouTube", "Sundar Pichai", "GOOG"],
    "META": ["Meta Platforms", "Meta", "Facebook", "Instagram", "WhatsApp", "Zuckerberg"],
    "TSLA": ["Tesla"],
    "JPM": ["JPMorgan", "JP Morgan", "J.P. Morgan", "Jamie Dimon"],
    "BAC": ["Bank of America", "BofA"],
    "XOM": ["Exxon", "ExxonMobil"],
    "JNJ": ["Johnson & Johnson", "J&J"],
    "LLY": ["Eli Lilly", "Lilly", "Mounjaro", "Zepbound"],
    "WMT": ["Walmart"],
    "KO": ["Coca-Cola", "Coca Cola", "Coke"],
    "BA": ["Boeing"],
}
UNIVERSE = {
    ticker: {"name": name, "sector": sector, "aliases": ALIASES[ticker]} for ticker, (name, sector) in COMPANIES.items()
}
TICKERS = list(UNIVERSE)

# Tickers that are safe to match as a bare upper-case word. KO and BA are too ambiguous on their own,
# so they are only matched as "$KO", "(KO)" or "NYSE:KO".
BARE_TICKERS = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "TSLA",
    "JPM",
    "BAC",
    "XOM",
    "JNJ",
    "LLY",
    "WMT",
}
MARKET = "MARKET"  # pseudo-ticker for market-wide (macro / geopolitical / credit) signals

# --- Data sources -----------------------------------------------------------------------------------------

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RiskPulse/1.0 (student hackathon prototype)"
HTTP_TIMEOUT = 20  # seconds per request
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
GDELT_QUERIES = {  # GDELT allows about one request every 5 s, so all companies share one OR-query
    "companies": (
        '(Apple OR Microsoft OR Nvidia OR Amazon OR Alphabet OR Google OR "Meta Platforms" OR Tesla '
        'OR JPMorgan OR "Bank of America" OR Exxon OR "Johnson & Johnson" OR "Eli Lilly" OR Walmart '
        'OR "Coca-Cola" OR Boeing) sourcelang:english'
    ),
    "macro": (
        '("Federal Reserve" OR inflation OR "interest rates" OR tariffs OR sanctions OR recession '
        'OR "trade war" OR geopolitical OR "oil prices") sourcelang:english'
    ),
}
SNAPSHOT_KEEP_DAYS = 7  # raw snapshots keep a rolling 7-day window of documents

# --- NLP --------------------------------------------------------------------------------------------------

# DistilRoBERTa fine-tuned on financial news. It beat ProsusAI/finbert on our benchmark
# (macro-F1 0.71 vs 0.66) and is ~1.9x faster on CPU. Override with the environment variable
# RISKPULSE_MODEL: "vader" forces the light fallback, any other value is a Hugging Face model id.
SENTIMENT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
MODEL_ALIASES = {SENTIMENT_MODEL: "distilroberta-financial", "ProsusAI/finbert": "finbert"}
HIGH_IMPACT_THRESHOLD = 7  # impact score from which a signal counts as a high-impact event

# --- Module A: tactical index rebalancer ------------------------------------------------------------------

LOOKBACK_HOURS = 72  # signals older than this are ignored; also the first back-fill window
STEP_HOURS = 1  # rebalancing grid of the back-fill (hourly)
HALF_LIFE_HOURS = 24  # a signal's influence halves every 24 hours
TILT = 1.0  # at full confidence, sentiment +1 doubles a raw weight and -1 sends it to MIN_WEIGHT
SOURCE_WEIGHT = {"news": 1.0, "global_news": 1.0, "social": 0.6}  # social chatter is noisier
MIN_WEIGHT = 0.02  # position limits per stock
MAX_WEIGHT = 0.12
EVIDENCE_K = 2.0  # shrinkage: a stock needs ~2 fresh signals before it gets half of the tilt
SMOOTHING = 0.5  # each rebalance closes 50 % of the gap to the target weights (turnover control)
