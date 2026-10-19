"""RiskPulse dashboard: NLP Risk Engine signals + Module A (tactical index rebalancing).

Start:  python src/main.py dashboard   (or, from the repo root: streamlit run src/dashboard.py)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # Streamlit runs this file directly

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import engine
import ingestion
import nlp
import rebalancer

st.set_page_config(page_title="RiskPulse", page_icon="📈", layout="wide")

# --- Palette: light and dark steps, colour-blind safe ----------------------------------------------------

PALETTES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=["#2a78d6", "#eb6834", "#1baf7a"],
        pos="#2a78d6",
        neg="#e34948",
        mid="#f0efec",
        deemph="#d3d1ca",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=["#3987e5", "#d95926", "#199e70"],
        pos="#3987e5",
        neg="#e66767",
        mid="#383835",
        deemph="#4a4a46",
    ),
}
try:
    _theme = st.context.theme.type or "light"
except Exception:
    _theme = "light"
P = PALETTES["dark" if _theme == "dark" else "light"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
N = len(config.TICKERS)
EQUAL = 1 / N


def style(fig, height=360, legend=False):
    """Shared chart styling: transparent background, recessive grid and axes, theme-aware text."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=30 if legend else 8, b=8),
        showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=P["ink2"]),
        hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["axis"], font=dict(family=FONT, color=P["ink"])),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", title=None, font=dict(color=P["ink2"])),
    )
    fig.update_xaxes(
        showgrid=False, linecolor=P["axis"], tickfont=dict(color=P["muted"]), zeroline=False, automargin=True
    )
    fig.update_yaxes(
        gridcolor=P["grid"],
        gridwidth=1,
        linecolor=P["axis"],
        tickfont=dict(color=P["muted"]),
        zeroline=False,
        automargin=True,
    )
    return fig


def padded(values, pad=0.28):
    """Axis range with room for the value labels at the bar tips."""
    lo, hi = min(0.0, float(min(values))), max(0.0, float(max(values)))
    span = (hi - lo) or 1.0
    return [lo - (pad * span if lo < 0 else 0), hi + (pad * span if hi > 0 else 0)]


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


# --- Data: cached, reloaded when the files change ---------------------------------------------------------


def _mtime(path):
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_signals(mtime):
    return engine.read_signals()


@st.cache_data(show_spinner=False)
def load_history(mtime):
    return rebalancer.load_history()


@st.cache_data(show_spinner=False)
def load_log(mtime):
    return rebalancer.load_log()


@st.cache_resource(show_spinner="Loading the sentiment model (first run downloads ~330 MB)...")
def get_model():
    return engine.get_model()


def run_engine(offline: bool):
    """One engine cycle + rebalance, with progress shown in the page."""
    with st.status("Running the NLP Risk Engine...", expanded=True) as status:
        status.write("Fetching Yahoo Finance news, GDELT global events and StockTwits posts...")
        get_model()
        engine.run_cycle(offline=offline, log=status.write)
        rebalancer.update(now=rebalancer.data_now() if offline else None, log=status.write)
        status.update(
            label="Engine cycle complete - signals published, index rebalanced", state="complete", expanded=False
        )


# --- Sidebar: controls ------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Engine controls")
    offline = st.toggle(
        "Offline mode", value=False, help="Replay the saved snapshot in data/raw instead of calling the live APIs."
    )
    if st.button("Fetch latest data & rebalance", type="primary", icon=":material/refresh:", width="stretch"):
        run_engine(offline)
        st.rerun()
    auto = st.toggle(
        "Auto-refresh every 15 min",
        value=False,
        help="Runs an engine cycle every 15 minutes while this page is open (real-time mode).",
    )
    if auto:

        @st.fragment(run_every=15 * 60)
        def _auto_refresh():
            last_run_at = st.session_state.get("last_auto_run", 0)
            if time.time() - last_run_at >= 14 * 60:
                st.session_state["last_auto_run"] = time.time()
                if last_run_at:  # the first tick only arms the timer
                    run_engine(offline)
                    st.rerun()
            elapsed = time.time() - st.session_state["last_auto_run"]
            st.caption(f"Next automatic cycle in ~{max(1, int((15 * 60 - elapsed) / 60))} min")

        _auto_refresh()

    st.divider()
    st.subheader("Module A settings")
    params = rebalancer.load_params()
    with st.form("params"):
        tilt = st.slider(
            "Tilt strength",
            0.0,
            2.0,
            float(params.tilt),
            0.1,
            help="Sentiment +1 multiplies a stock's raw weight by (1 + tilt); -1 by (1 - tilt).",
        )
        half_life = st.slider("Signal half-life (hours)", 6, 72, int(params.half_life), 6)
        lo, hi = st.slider("Position limits (%)", 0.0, 20.0, (params.min_weight * 100, params.max_weight * 100), 0.5)
        smooth = st.slider(
            "Speed (share of the gap closed per rebalance)",
            0.1,
            1.0,
            float(params.smoothing),
            0.1,
            help="1.0 jumps straight to the target weights; 0.5 moves half-way each time (less turnover).",
        )
        if st.form_submit_button("Apply & recompute history", width="stretch"):
            rebalancer.update(
                rebuild=True,
                params=rebalancer.Params(
                    tilt=tilt,
                    half_life=half_life,
                    lookback=params.lookback,
                    min_weight=lo / 100,
                    max_weight=hi / 100,
                    evidence_k=params.evidence_k,
                    smoothing=smooth,
                ),
            )
            st.rerun()

    st.divider()
    st.caption("REST API for downstream apps: `python src/main.py api` then open http://localhost:8000/docs")

# --- Load -------------------------------------------------------------------------------------------------

signals = load_signals(_mtime(config.SIGNALS_FILE))
history = load_history(_mtime(config.WEIGHTS_FILE))
moves = load_log(_mtime(config.REBALANCE_LOG))
last = engine.last_run()

st.title("RiskPulse")
st.caption(
    "AI/NLP Risk Engine turning financial news, global events and social media into structured risk "
    "signals, and Module A: a tactical rebalancer for a 15-stock S&P 100 mock index."
)

if signals.empty or history.empty:
    st.info("No signals yet. Click **Fetch latest data & rebalance** in the sidebar (or run `python src/main.py run`).")
    st.stop()

current = rebalancer.current_weights(history)
turn = rebalancer.turnover(history)
summary = engine.company_summary(signals)
sources = last.get("sources", {})
delivering = sum(1 for s in sources.values() if s.get("documents"))
modes = {s.get("mode") for s in sources.values()}
source_note = "live" if modes == {"live"} else "offline" if modes == {"offline snapshot"} else "mixed"

# --- KPI row ----------------------------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Signals published", f"{len(signals):,}", f"+{last.get('new_signals', 0)} last cycle" if last else None, border=True
)
k2.metric(
    "Data sources",
    f"{delivering}/3 {source_note}" if sources else "-",
    border=True,
    help="Sources that delivered documents in the last cycle. 'mixed' = at least one API was down or "
    "rate-limited and its saved snapshot was used instead (see the Sources & API tab).",
)
top, bottom = current.iloc[0], current.iloc[-1]
k3.metric("Most overweight", top["ticker"], f"{top['vs_equal_pp']:+.2f} pp vs equal", border=True)
k4.metric("Most underweight", bottom["ticker"], f"{bottom['vs_equal_pp']:+.2f} pp vs equal", border=True)
k5.metric(
    "Avg turnover",
    f"{turn.mean():.2%}" if len(turn) else "-",
    help="Average one-way turnover per rebalance: half the sum of absolute weight changes.",
    border=True,
)

tab_index, tab_signals, tab_try, tab_sources = st.tabs(
    ["Index weights (Module A)", "Risk signals (engine output)", "Analyze any text", "Sources & API"]
)

# --- Tab 1: Module A --------------------------------------------------------------------------------------

with tab_index:
    st.subheader("How the index weights changed over time")
    st.caption(
        f"{history['timestamp'].nunique()} rebalances from {history['timestamp'].min():%d %b %H:%M} to "
        f"{history['timestamp'].max():%d %b %H:%M} UTC. Blue = overweight, red = underweight vs equal "
        f"weight ({EQUAL:.2%})."
    )
    order = current["ticker"].tolist()
    wide = history.pivot_table(index="ticker", columns="timestamp", values="weight").reindex(order)
    sent = history.pivot_table(index="ticker", columns="timestamp", values="sentiment").reindex(order)
    nsig = history.pivot_table(index="ticker", columns="timestamp", values="n_signals").reindex(order)
    z = (wide - EQUAL) * 100
    m = max(0.5, float(np.nanmax(np.abs(z.values))))
    custom = np.dstack([wide.values * 100, sent.values, nsig.values])
    heat = go.Figure(
        go.Heatmap(
            z=z.values,
            x=z.columns,
            y=order,
            customdata=custom,
            zmid=0,
            zmin=-m,
            zmax=m,
            xgap=1,
            ygap=2,
            colorscale=[[0, P["neg"]], [0.5, P["mid"]], [1, P["pos"]]],
            colorbar=dict(
                title=dict(text="pp vs equal", side="right"),
                thickness=10,
                outlinewidth=0,
                tickfont=dict(color=P["muted"]),
            ),
            hovertemplate="<b>%{customdata[0]:.2f}%</b> weight · %{z:+.2f} pp<br>%{y} · %{x|%d %b %H:%M} UTC"
            "<br>sentiment %{customdata[1]:+.2f} from %{customdata[2]:.0f} signals<extra></extra>",
        )
    )
    heat.update_yaxes(autorange="reversed", showgrid=False)
    show(style(heat, height=430))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Weight path of selected stocks**")
        if "slots" not in st.session_state:  # colour follows the stock, not its position in the list
            st.session_state["slots"] = {}
        default = [order[0], order[-1]]
        picked = st.multiselect(
            "Highlight up to 3 stocks", order, default=default, max_selections=3, label_visibility="collapsed"
        )
        slots = {t: s for t, s in st.session_state["slots"].items() if t in picked}
        for t in picked:
            if t not in slots:
                slots[t] = min(set(range(3)) - set(slots.values()))
        st.session_state["slots"] = slots
        lines = go.Figure()
        for t in order:
            if t in picked:
                continue
            h = history[history["ticker"] == t]
            lines.add_trace(
                go.Scatter(
                    x=h["timestamp"],
                    y=h["weight"] * 100,
                    mode="lines",
                    hoverinfo="skip",
                    line=dict(color=P["deemph"], width=1),
                    showlegend=False,
                )
            )
        lines.add_hline(
            y=EQUAL * 100,
            line=dict(color=P["muted"], width=1),
            annotation=dict(text=f"equal weight {EQUAL:.2%}", font=dict(color=P["muted"], size=11)),
            annotation_position="bottom left",
        )
        for t in picked:
            h = history[history["ticker"] == t]
            color = P["series"][slots[t]]
            lines.add_trace(
                go.Scatter(
                    x=h["timestamp"],
                    y=h["weight"] * 100,
                    name=t,
                    mode="lines",
                    line=dict(color=color, width=2, shape="hv"),
                    hovertemplate=f"<b>%{{y:.2f}}%</b> {t}<extra></extra>",
                )
            )
            lines.add_trace(
                go.Scatter(
                    x=[h["timestamp"].iloc[-1]],
                    y=[h["weight"].iloc[-1] * 100],
                    mode="markers",
                    marker=dict(size=9, color=color, line=dict(color=P["surface"], width=2)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        lines.update_layout(hovermode="x unified")
        lines.update_yaxes(ticksuffix="%")
        show(style(lines, height=330, legend=True))
    with right:
        st.markdown("**Current tilt vs equal weight**")
        cur = current.sort_values("vs_equal_pp")
        bars = go.Figure(
            go.Bar(
                x=cur["vs_equal_pp"],
                y=cur["ticker"],
                orientation="h",
                width=0.62,
                marker=dict(color=[P["pos"] if v >= 0 else P["neg"] for v in cur["vs_equal_pp"]], cornerradius=4),
                text=[f"{v:+.2f}" for v in cur["vs_equal_pp"]],
                textposition="outside",
                textfont=dict(color=P["ink2"], size=11),
                cliponaxis=False,
                customdata=np.stack([cur["weight"] * 100, cur["sentiment"], cur["n_signals"]], axis=1),
                hovertemplate="<b>%{x:+.2f} pp</b> · %{y}<br>weight %{customdata[0]:.2f}%<br>"
                "sentiment %{customdata[1]:+.2f} from %{customdata[2]} signals<extra></extra>",
            )
        )
        bars.update_xaxes(
            ticksuffix=" pp",
            showgrid=True,
            gridcolor=P["grid"],
            zeroline=True,
            zerolinecolor=P["axis"],
            zerolinewidth=1,
            range=padded(cur["vs_equal_pp"]),
        )
        bars.update_yaxes(showgrid=False)
        show(style(bars, height=380))

    st.markdown("**Current index composition**")
    st.dataframe(
        current[["ticker", "company", "sector", "weight", "vs_equal_pp", "sentiment", "confidence", "n_signals"]],
        hide_index=True,
        width="stretch",
        column_config={
            "weight": st.column_config.ProgressColumn(
                "weight", format="percent", min_value=0, max_value=float(params.max_weight)
            ),
            "vs_equal_pp": st.column_config.NumberColumn("vs equal (pp)", format="%+.2f"),
            "sentiment": st.column_config.NumberColumn("sentiment", format="%+.3f"),
            "confidence": st.column_config.NumberColumn("confidence", format="%.2f"),
            "n_signals": st.column_config.NumberColumn("signals (72h)"),
        },
    )

    st.markdown("**Rebalance log** - every weight move of 0.25 pp or more, with the headline that drove it")
    if moves.empty:
        st.caption("No moves logged yet.")
    else:
        st.dataframe(
            moves.sort_values("timestamp", ascending=False).head(200),
            hide_index=True,
            width="stretch",
            column_config={
                "timestamp": st.column_config.DatetimeColumn("time (UTC)", format="DD MMM HH:mm"),
                "old_weight": st.column_config.NumberColumn("from", format="percent"),
                "new_weight": st.column_config.NumberColumn("to", format="percent"),
                "change_pp": st.column_config.NumberColumn("change (pp)", format="%+.2f"),
            },
        )

    with st.expander("How Module A rebalances"):
        st.markdown(f"""
For every rebalance time *t* and stock *i* (only signals published before *t* are used):

1. Signal weight = recency x impact x source:
   `0.5 ** (age_hours / {params.half_life:g}) x impact_score x source_weight` (news 1.0, social 0.6).
   Signals older than {params.lookback:g} h are ignored.
2. Stock sentiment `S` = weighted average of the engine's `sentiment_score`;
   confidence `c = E / (E + {params.evidence_k:g})`, where `E` is the summed recency weight,
   so one stray tweet cannot move the index.
3. Target weight = `1/{N} x (1 + {params.tilt:g} x c x S)`,
   clipped to [{params.min_weight:.0%}, {params.max_weight:.0%}] and renormalised to 100 %.
   **Positive sentiment raises a stock's weight, negative sentiment lowers it.**
4. Turnover control: each rebalance closes {params.smoothing:.0%} of the gap between current and target weights.
5. The back-fill runs hourly over the last {params.lookback:g} h; every live engine cycle adds a new rebalance.
""")

# --- Tab 2: engine output ---------------------------------------------------------------------------------

with tab_signals:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    tick_opts = sorted(signals["ticker"].unique())
    sel_t = f1.multiselect("Company", tick_opts, placeholder="All companies")
    sel_s = f2.multiselect("Source", sorted(signals["source"].unique()), placeholder="All sources")
    sel_e = f3.multiselect("Event type", nlp.EVENT_TYPES, placeholder="All event types")
    min_imp = f4.number_input("Min impact", 1, 10, 1)
    view = signals
    if sel_t:
        view = view[view["ticker"].isin(sel_t)]
    if sel_s:
        view = view[view["source"].isin(sel_s)]
    if sel_e:
        view = view[view["event_type"].isin(sel_e)]
    view = view[view["impact_score"] >= min_imp]
    st.caption(f"{len(view):,} of {len(signals):,} signals match the filters")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Signals by event type**")
        ev = view["event_type"].value_counts().reindex(nlp.EVENT_TYPES).dropna().sort_values()
        fig = go.Figure(
            go.Bar(
                x=ev.values,
                y=ev.index,
                orientation="h",
                width=0.62,
                marker=dict(color=P["series"][0], cornerradius=4),
                text=ev.values.astype(int),
                textposition="outside",
                textfont=dict(color=P["ink2"], size=11),
                cliponaxis=False,
                hovertemplate="<b>%{x}</b> signals · %{y}<extra></extra>",
            )
        )
        fig.update_xaxes(showgrid=True, gridcolor=P["grid"], range=padded(ev.values, 0.15) if len(ev) else None)
        fig.update_yaxes(showgrid=False)
        show(style(fig, height=330))
    with c2:
        st.markdown("**Impact-weighted sentiment by company (last 72 h)**")
        sm = engine.company_summary(view).sort_values("sentiment") if not view.empty else pd.DataFrame()
        if sm.empty:
            st.caption("No signals for this filter.")
        else:
            fig = go.Figure(
                go.Bar(
                    x=sm["sentiment"],
                    y=sm["ticker"],
                    orientation="h",
                    width=0.62,
                    marker=dict(color=[P["pos"] if v >= 0 else P["neg"] for v in sm["sentiment"]], cornerradius=4),
                    customdata=np.stack([sm["signals"], sm["top_event"], sm["max_impact"]], axis=1),
                    hovertemplate="<b>%{x:+.2f}</b> · %{y}<br>%{customdata[0]} signals · top event %{customdata[1]}"
                    "<br>max impact %{customdata[2]}/10<extra></extra>",
                )
            )
            fig.update_xaxes(
                range=[-1, 1],
                showgrid=True,
                gridcolor=P["grid"],
                zeroline=True,
                zerolinecolor=P["axis"],
                zerolinewidth=1,
            )
            fig.update_yaxes(showgrid=False)
            show(style(fig, height=330))

    alerts = view[view["impact_score"] >= config.HIGH_IMPACT_THRESHOLD].sort_values("timestamp", ascending=False)
    st.markdown(f"**High-impact alerts** (impact >= {config.HIGH_IMPACT_THRESHOLD}) - {len(alerts)} signals")
    st.dataframe(
        alerts[["timestamp", "impact_score", "event_type", "ticker", "sentiment_score", "headline", "source"]].head(
            100
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "timestamp": st.column_config.DatetimeColumn("time (UTC)", format="DD MMM HH:mm"),
            "impact_score": st.column_config.NumberColumn("impact", format="%d / 10"),
            "sentiment_score": st.column_config.NumberColumn("sentiment", format="%+.2f"),
        },
    )

    st.markdown("**All structured signals** (newest first)")
    cols = [
        "timestamp",
        "source",
        "ticker",
        "headline",
        "sentiment_score",
        "sentiment_label",
        "event_type",
        "event_keywords",
        "impact_score",
        "url",
    ]
    st.dataframe(
        view.sort_values("timestamp", ascending=False)[cols],
        hide_index=True,
        width="stretch",
        height=380,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("time (UTC)", format="DD MMM HH:mm"),
            "sentiment_score": st.column_config.NumberColumn("sentiment", format="%+.2f"),
            "impact_score": st.column_config.NumberColumn("impact"),
            "url": st.column_config.LinkColumn("link", display_text="open"),
        },
    )
    st.download_button(
        "Download filtered signals (CSV)", view.to_csv(index=False).encode("utf-8"), "signals_filtered.csv", "text/csv"
    )

# --- Tab 3: live NLP --------------------------------------------------------------------------------------

with tab_try:
    st.subheader("Run the NLP engine on any headline or post")
    examples = [
        "Moody's downgrades Boeing to junk as cash burn deepens",
        "Nvidia unveils next-generation AI chip, shares jump 6%",
        "US imposes sweeping new tariffs on Chinese electronics",
        "JPMorgan agrees to acquire fintech lender in $2 billion deal",
        "Fed signals more rate cuts as inflation cools faster than expected",
        "$TSLA this is going to the moon, earnings will crush it 🚀",
    ]
    ex = st.selectbox("Example", examples, index=0)
    text = st.text_area("Text", ex, height=90)
    stype = st.radio("Source type", ["news", "global_news", "social"], horizontal=True)
    if st.button("Analyze", type="primary", icon=":material/psychology:"):
        res = nlp.analyze_text(text, get_model(), stype)
        a, b, c, d = st.columns(4)
        a.metric("Sentiment score", res["sentiment_label"], f"{res['sentiment_score']:+.2f}", border=True)
        b.metric(
            "Event type",
            res["event_type"],
            border=True,
            help="Matched keywords: " + (", ".join(res["event_keywords"]) or "none"),
        )
        c.metric("Impact score", f"{res['impact_score']} / 10", border=True)
        d.metric("Companies", ", ".join(x["ticker"] for x in res["companies"]) or "none", border=True)
        st.caption("Event keywords: " + (", ".join(res["event_keywords"]) or "none"))
        st.json(res)

# --- Tab 4: sources & API ---------------------------------------------------------------------------------

with tab_sources:
    st.subheader("Data sources")
    rows = []
    for src, desc in ingestion.SOURCES.items():
        snap = ingestion.load_snapshot(src)
        cycle = sources.get(src, {})
        rows.append(
            {
                "source": desc,
                "documents in snapshot": len(snap),
                "newest document": max((d["published_at"] for d in snap), default="-"),
                "signals published": int((signals["source"] == src).sum()),
                "last cycle": cycle.get("mode", "-"),
                "note": cycle.get("error") or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Last engine cycle: {last.get('finished_at', '-')} UTC · model: {last.get('model', '-')} · "
        f"NLP time {last.get('nlp_seconds', '-')} s for {last.get('new_documents', '-')} new documents."
    )
    st.subheader("Signal API for downstream applications")
    st.markdown("""
Start it with `python src/main.py api`, then open **http://localhost:8000/docs** for interactive docs.

| Endpoint | What it returns |
|---|---|
| `GET /signals?ticker=NVDA&event_type=Geopolitical&min_impact=7` | structured signals, newest first |
| `GET /companies` · `GET /companies/{ticker}` | per-company sentiment, top event, max impact, latest signals |
| `GET /events?min_impact=7` | high-impact events (trigger feed for e.g. a stress-testing module) |
| `POST /analyze` `{"text": "...", "source_type": "news"}` | run the NLP engine on any text |
| `GET /index/weights` · `GET /index/history` | Module A output |

The same signals are written to `data/signals.jsonl` (append-only stream) and `data/signals.csv`.
""")
