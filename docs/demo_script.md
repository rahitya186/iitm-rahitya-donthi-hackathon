# RiskPulse - demo run sheet (5 minutes)

Fits the case study's 5-minute live demo; for the recorded video (up to 10 minutes) follow the same order and
add the optional code tour at the end.

## Before you start (once, 5 minutes earlier)

Open two terminals in the repo folder and activate the environment in both: `.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` on macOS / Linux.

1. Terminal 1: `python src/main.py run` warms the model cache and pulls fresh data (check that the sources say
   `live`). Demoing days after the data was collected? Run `python src/main.py run --reset` instead, so the
   72-hour window starts from today's news.
2. Terminal 2: `python src/main.py api` (leave it running) and open http://localhost:8000/docs in a browser tab.
3. Terminal 1: `python src/main.py dashboard` and open http://localhost:8501 in a browser tab.
4. Weak Wi-Fi? Switch on **Offline mode** in the dashboard sidebar: everything replays from `data/raw`.

Ctrl+C in a terminal stops the dashboard or the API.

## The run

| Time | Show | Say |
|---|---|---|
| 0:00-0:30 | Slide 2 (problem & approach) | "Markets react to text - news, global events, social posts. RiskPulse turns that text into structured risk signals and uses them to rebalance an index in near real time." |
| 0:30-1:00 | Terminal: `python src/main.py run` | "One command: three live sources - Yahoo Finance news, GDELT global news, StockTwits - fetched in parallel, new documents analysed, signals published, index rebalanced." Point at the per-source lines and the signal count. |
| 1:00-1:40 | Dashboard -> KPI row and heatmap | "This is Module A: 15 S&P 100 stocks, rebalanced every hour. Blue is overweight, red is underweight versus equal weight." Point at a row that changes colour (e.g. JPM turning red on 10 Oct). |
| 1:40-2:10 | Weight path: pick JPM and BA; then the rebalance log | "Every move has a reason: JPM was cut after 'Jamie Dimon warns the next credit crisis could be severe' - a Credit Event with impact 9." |
| 2:10-2:50 | Tab **Risk signals** -> filter Event type = Credit Event, Geopolitical; Min impact = 7 | "This is the engine's output: sentiment score, event type, impact score per company, with the keywords that triggered the label. Market-wide news becomes MARKET signals." |
| 2:50-3:30 | Tab **Analyze any text** -> type a fresh headline, e.g. *"Moody's downgrades Boeing to junk as cash burn deepens"* | "Live NLP on any text: negative, Credit Event, impact 8, linked to Boeing." Try *"Nvidia unveils next-generation AI chip, shares jump 6%"* for the positive case. |
| 3:30-3:50 | Sidebar -> **Fetch latest data & rebalance** | "A new cycle: new signals, and a new rebalance appears at the right edge of the heatmap." |
| 3:50-4:20 | API tab: `GET /events` (min_impact 7) -> Execute; `POST /analyze` | "Any downstream app - for example a stress-testing module - consumes the same signals over this REST API, or from the JSON-lines stream." |
| 4:20-5:00 | Slide 5 (results) | "Against naive baselines: sentiment accuracy 75.6 % vs 49.4 % for a generic lexicon, event accuracy 76.7 % vs 66.7 % on held-out headlines, and ~30 seconds per cycle instead of hours of reading. Every weight change is explainable." |

## Optional for the video: 60-second code tour

`src/main.py` (one entry point) -> `ingestion.py` (3 connectors + snapshot fallback) -> `nlp.py` (entity linking,
sentiment model, event taxonomy, impact score) -> `engine.py` (publish the stream) -> `rebalancer.py` (Module A
formula) -> `api.py` -> `dashboard.py`. Then `python src/main.py evaluate` for the benchmark table.

## Likely jury questions - short answers

- **Why DistilRoBERTa and not FinBERT?** We benchmarked both on 2,388 labeled finance tweets the models never saw:
  macro-F1 0.71 vs 0.66, and ~1.9x faster on CPU.
- **Why rules for events and impact?** No labeled event data was available; rules are transparent (keywords
  returned), fast, and measured on a held-out set. A fine-tuned classifier is the next step.
- **How do you stop one tweet moving the index?** Social posts count 0.6x, impact caps at 7 for a single post,
  confidence shrinkage needs ~2 fresh signals for half the tilt, and turnover control closes only half the gap.
- **What if an API is down?** Each source falls back to its last snapshot; the dashboard shows which sources were live.
- **Look-ahead bias?** A rebalance at time *t* only uses signals published before *t*.
