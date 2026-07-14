from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dashboard.data import (  # noqa: E402
    build_apples_to_apples_rows,
    build_calibration_summary,
    build_derived_run_history,
    build_family_accuracy_summary,
    build_family_results,
    build_live_latest_snapshots,
    build_match_catalog,
    build_metrics_summary,
    build_reliability_curve_data,
    build_roc_curve_data,
    build_version_summary,
    feature_log_signature,
    file_mtimes,
    find_feature_snapshot_row,
    load_dashboard_data,
    latest_pipeline_run_id,
)


st.set_page_config(
    page_title="Tennis Ops Dashboard",
    page_icon="🎾",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data_cached(_mtimes):
    return load_dashboard_data()


@st.cache_data(show_spinner=False)
def load_feature_snapshot_cached(feature_snapshot_id: str, _signature):
    return find_feature_snapshot_row(feature_snapshot_id)


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def pick_latest_timestamp(*series_list: pd.Series) -> pd.Timestamp | None:
    values = []
    for series in series_list:
        if series is None or series.empty:
            continue
        values.append(series.max())
    values = [v for v in values if pd.notna(v)]
    return max(values) if values else None


def fmt(value):
    if value is None or pd.isna(value):
        return "n/a"
    return value


def pp(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.1%}"


def signed_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.{digits}f}"


def render_dashboard_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e7ebf0;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] p {
            color: #475569;
            font-size: 0.82rem;
        }
        .insight-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #dfe7ef;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            min-height: 132px;
        }
        .insight-card h4 {
            margin: 0 0 0.45rem 0;
            color: #0f172a;
            font-size: 0.95rem;
        }
        .insight-card .big {
            color: #111827;
            font-size: 1.35rem;
            font-weight: 650;
            line-height: 1.2;
        }
        .insight-card .muted {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


FEATURE_OVERRIDES = {
    "Rank_Diff": ("Rank difference", "Player 1 rank minus Player 2 rank; lower ATP ranks are better."),
    "Rank_Points_Diff": ("Rank points difference", "Player 1 ranking points minus Player 2 ranking points."),
    "Rank_Ratio": ("Rank ratio", "Relative rank comparison between the two players."),
    "Avg_Rank": ("Average rank", "Average ATP rank of the two players."),
    "Avg_Rank_Points": ("Average rank points", "Average ranking points of the two players."),
    "Age_Diff": ("Age difference", "Player 1 age minus Player 2 age."),
    "Height_Diff": ("Height difference", "Player 1 height minus Player 2 height."),
    "Avg_Age": ("Average age", "Average age of the two players."),
    "Avg_Height": ("Average height", "Average height of the two players."),
    "draw_size": ("Draw size", "Tournament draw size used for round and tournament context."),
    "H2H_Total_Matches": ("Head-to-head matches", "Prior meetings between these two players."),
    "H2H_P1_Wins": ("P1 head-to-head wins", "Prior head-to-head wins by Player 1."),
    "H2H_P2_Wins": ("P2 head-to-head wins", "Prior head-to-head wins by Player 2."),
    "H2H_P1_WinRate": ("P1 head-to-head win rate", "Player 1 win rate in prior meetings."),
    "H2H_Recent_P1_Advantage": ("Recent H2H advantage", "Player 1 advantage over 50% in the most recent head-to-head meetings."),
    "Clay_Season": ("Clay season", "Flag for clay-season timing."),
    "Grass_Season": ("Grass season", "Flag for grass-season timing."),
    "Indoor_Season": ("Indoor season", "Flag for the fall/winter indoor-season window."),
    "Surface_Transition_Flag": ("Surface transition", "Whether the current surface differs from the player's recent surface context."),
}


def feature_side(feature: str) -> str:
    if feature.startswith("P1_") or feature.startswith("Player1_"):
        return "Player 1"
    if feature.startswith("P2_") or feature.startswith("Player2_"):
        return "Player 2"
    if feature.startswith("H2H_"):
        return "Head-to-head"
    if feature.startswith(("Surface_", "Level_", "Round_", "Handedness_Matchup_")):
        return "Context / flag"
    return "Match context"


def feature_group(feature: str) -> str:
    text = feature.lower()
    if "rank" in text:
        return "Ranking"
    if "surface" in text:
        return "Surface"
    if "winrate" in text or "form" in text or "winstreak" in text:
        return "Form"
    if "matches" in text or "sets" in text or "days_since" in text or "rust" in text:
        return "Activity / fatigue"
    if "h2h" in text:
        return "Head-to-head"
    if "age" in text or "height" in text:
        return "Physical"
    if "country" in text or "hand" in text:
        return "Player profile"
    if "round" in text or "level" in text or "draw" in text:
        return "Tournament context"
    if "season" in text:
        return "Seasonality"
    return "Other"


def explain_feature(feature: str) -> tuple[str, str]:
    if feature in FEATURE_OVERRIDES:
        return FEATURE_OVERRIDES[feature]

    label = feature
    for prefix, replacement in [
        ("P1_", "P1 "),
        ("P2_", "P2 "),
        ("Player1_", "P1 "),
        ("Player2_", "P2 "),
        ("H2H_", "H2H "),
        ("Surface_", "Surface: "),
        ("Level_", "Level: "),
        ("Round_", "Round: "),
        ("Handedness_Matchup_", "Handedness matchup: "),
    ]:
        if label.startswith(prefix):
            label = replacement + label[len(prefix):]
            break
    label = label.replace("_", " ")

    text = feature.lower()
    if "form_trend_30d" in text:
        desc = "Exponentially weighted win form over the last 30 days; recent matches count more."
    elif "winrate_last10_120d" in text:
        desc = "Win rate over the most recent 10 matches inside the last 120 days."
    elif "winstreak_current" in text:
        desc = "Current winning streak before this match."
    elif "surface_winrate_90d" in text:
        desc = "Win rate on this surface over the previous 90 days."
    elif "surface_matches_30d" in text:
        desc = "Matches on this surface in the previous 30 days."
    elif "surface_matches_90d" in text:
        desc = "Matches on this surface in the previous 90 days."
    elif "surface_experience" in text:
        desc = "Career match count on this surface before this match."
    elif "level_winrate_career" in text:
        desc = "Career win rate at this tournament level before this match."
    elif "level_matches_career" in text:
        desc = "Career match count at this tournament level before this match."
    elif "round_winrate_career" in text:
        desc = "Career win rate in this round before this match."
    elif "finals_winrate" in text:
        desc = "Career finals win rate before this match."
    elif "semifinals_winrate" in text:
        desc = "Career semifinals win rate before this match."
    elif "bigmatch_winrate" in text:
        desc = "Career win rate in Grand Slam and Masters-level matches."
    elif "rank_change_30d" in text:
        desc = "Ranking movement over roughly the previous 30 days."
    elif "rank_change_90d" in text:
        desc = "Ranking movement over roughly the previous 90 days."
    elif "rank_volatility_90d" in text:
        desc = "Standard deviation of recent ranks over a 90-day window."
    elif "rank_momentum" in text:
        desc = "Difference in recent ranking momentum between the players."
    elif "days_since_last" in text:
        desc = "Days since the player's last prior tournament week."
    elif "rust_flag" in text:
        desc = "Flag for more than 21 days since the player's last prior tournament week."
    elif "matches_14d" in text:
        desc = "Matches played in the previous 14 days."
    elif "matches_30d" in text:
        desc = "Matches played in the previous 30 days."
    elif "sets_14d" in text:
        desc = "Estimated sets played in the previous 14 days."
    elif "vs_lefty_winrate" in text:
        desc = "Career or historical win rate against left-handed opponents."
    elif "peak_age" in text:
        desc = "Flag for whether the player is in the modeled peak-age range."
    elif feature.startswith(("Surface_", "Level_", "Round_", "P1_Country_", "P2_Country_", "P1_Hand_", "P2_Hand_", "Handedness_Matchup_")):
        desc = "One-hot context/profile flag; active when the value is 1."
    else:
        desc = "Model input feature from the production feature snapshot."
    return label, desc


def format_feature_value(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return str(value)
    if numeric in (0, 1):
        return str(int(numeric))
    if abs(numeric) < 1:
        return f"{numeric:.3f}"
    if abs(numeric) < 100:
        return f"{numeric:.2f}"
    return f"{numeric:,.0f}"


def build_feature_display_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df
    rows = []
    for row in feature_df.itertuples(index=False):
        label, desc = explain_feature(str(row.feature))
        rows.append(
            {
                "side": feature_side(str(row.feature)),
                "group": feature_group(str(row.feature)),
                "label": label,
                "feature": row.feature,
                "value": row.value,
                "formatted": format_feature_value(row.value),
                "description": desc,
            }
        )
    return pd.DataFrame(rows)


def apply_history_filters(df: pd.DataFrame, *, trust_mode: str, surfaces, levels, rounds, statuses, versions, start_date, end_date) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if trust_mode == "Decision-grade only":
        out = out[out["decision_grade"]]
    if surfaces:
        out = out[out["surface"].isin(surfaces)]
    if levels:
        out = out[out["level"].isin(levels)]
    if rounds:
        out = out[out["round"].isin(rounds)]
    if statuses:
        out = out[out["record_status"].isin(statuses)]
    if versions:
        out = out[out["effective_model_version"].isin(versions)]
    out = out[out["effective_logged_at"].between(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1))]
    return out


def apply_live_filters(df: pd.DataFrame, *, trust_mode: str, surfaces, levels, rounds, versions, start_date, end_date) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if trust_mode == "Decision-grade only":
        out = out[out["decision_grade"]]
    if surfaces:
        out = out[out["surface"].isin(surfaces)]
    if levels:
        out = out[out["level"].isin(levels)]
    if rounds:
        out = out[out["round"].isin(rounds)]
    if versions:
        out = out[out["model_version"].isin(versions)]
    out = out[out["logged_at"].between(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1))]
    return out


def render_summary_metrics(prediction_log: pd.DataFrame, live_latest: pd.DataFrame, family_summary: pd.DataFrame, latest_data_ts):
    settled = prediction_log[prediction_log["is_settled"]]
    decision_grade = prediction_log[prediction_log["decision_grade"]]

    xgb_row = family_summary[family_summary["family"] == "XGB"] if "family" in family_summary.columns else pd.DataFrame()
    nn_row = family_summary[family_summary["family"] == "NN"] if "family" in family_summary.columns else pd.DataFrame()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Settled matches", f"{len(settled):,}")
    metric_cols[1].metric("Pending live rows", f"{len(live_latest):,}")
    metric_cols[2].metric("Decision-grade rows", f"{len(decision_grade):,}")
    metric_cols[3].metric(
        "XGB accuracy",
        pct(xgb_row["accuracy"].iloc[0] if not xgb_row.empty else None),
        delta=pct(xgb_row["edge_vs_market"].iloc[0]) if not xgb_row.empty else None,
    )
    metric_cols[4].metric(
        "NN accuracy",
        pct(nn_row["accuracy"].iloc[0] if not nn_row.empty else None),
        delta=pct(nn_row["edge_vs_market"].iloc[0]) if not nn_row.empty else None,
    )

    freshness = latest_data_ts.strftime("%Y-%m-%d %H:%M:%S") if latest_data_ts is not None else "n/a"
    st.caption(f"Latest dashboard source update: {freshness}")


def render_insight_strip(prediction_log: pd.DataFrame, live_latest: pd.DataFrame, run_history: pd.DataFrame):
    cards = st.columns(4)

    if not live_latest.empty and "largest_player_market_edge" in live_latest.columns:
        top_edge = live_latest.sort_values("largest_player_market_edge", ascending=False).head(1)
    else:
        top_edge = pd.DataFrame()
    if top_edge.empty:
        edge_title = "No live edge"
        edge_big = "n/a"
        edge_note = "No current snapshots match the active filters."
    else:
        row = top_edge.iloc[0]
        edge_title = "Largest live model-market gap"
        edge_big = pp(row.get("largest_player_market_edge"))
        edge_note = f"{row.get('match_label', 'n/a')} | consensus lean: {row.get('consensus_edge_player', 'n/a')}"

    settled = prediction_log[prediction_log["is_settled"]] if not prediction_log.empty and "is_settled" in prediction_log else pd.DataFrame()
    recent = settled.sort_values("settled_at", ascending=False).head(50) if "settled_at" in settled.columns else settled.tail(50)
    required = {"actual_winner", "xgb_p1_prob", "market_p1_prob"}
    recent_common = recent.dropna(subset=list(required)) if required.issubset(recent.columns) else pd.DataFrame()
    if not recent_common.empty:
        winner = pd.to_numeric(recent_common["actual_winner"], errors="coerce")
        recent_common = recent_common[winner.isin([1, 2])].copy()
        winner = pd.to_numeric(recent_common["actual_winner"], errors="coerce")
        xgb_correct = (
            ((recent_common["xgb_p1_prob"] >= 0.5) & winner.eq(1))
            | ((recent_common["xgb_p1_prob"] < 0.5) & winner.eq(2))
        )
        market_correct = (
            ((recent_common["market_p1_prob"] >= 0.5) & winner.eq(1))
            | ((recent_common["market_p1_prob"] < 0.5) & winner.eq(2))
        )
        xgb_recent = xgb_correct.mean()
        market_recent = market_correct.mean()
        model_title = "Recent XGB vs market"
        model_big = pp(xgb_recent - market_recent)
        model_note = f"{pct(xgb_recent)} model accuracy vs {pct(market_recent)} market across {len(recent_common):,} common settled rows"
    else:
        model_title = "Recent XGB vs market"
        model_big = "n/a"
        model_note = "Needs settled rows with XGB and market correctness."

    if not prediction_log.empty and "latest_feature_snapshot_id" in prediction_log.columns:
        snapshot_rate = prediction_log["latest_feature_snapshot_id"].fillna("").astype(str).str.len().gt(0).mean()
        lineage_big = pct(snapshot_rate)
        lineage_note = "Rows carrying a feature snapshot ID; the ledger separately verifies referential availability."
    else:
        lineage_big = "n/a"
        lineage_note = "Feature snapshot lineage is unavailable in the active view."

    if not run_history.empty and "started_at" in run_history.columns:
        latest_run = run_history.sort_values("started_at", ascending=False).head(1).iloc[0]
        run_big = str(latest_run.get("status", "unknown"))
        run_note = f"{latest_run.get('run_id', 'n/a')} | predictions: {fmt(latest_run.get('prediction_rows_success', 'n/a'))}"
    else:
        run_big = "n/a"
        run_note = "No audit run history found."

    payload = [
        (edge_title, edge_big, edge_note),
        (model_title, model_big, model_note),
        ("Feature lineage coverage", lineage_big, lineage_note),
        ("Latest nightly run", run_big, run_note),
    ]
    for col, (title, big, note) in zip(cards, payload):
        col.markdown(
            f"""
            <div class="insight-card">
              <h4>{escape(str(title))}</h4>
              <div class="big">{escape(str(big))}</div>
              <div class="muted">{escape(str(note))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_apples_to_apples_section(prediction_log: pd.DataFrame):
    apples = build_apples_to_apples_rows(prediction_log)
    st.markdown("### Apples-to-Apples Evaluation")
    if apples.empty:
        st.info("No common settled cohort is available where NN, XGB, RF, and market all have valid probabilities.")
        return

    st.caption(
        f"Common settled cohort: {len(apples):,} matches where NN, XGB, RF, and market all have usable probabilities."
    )

    metrics = build_metrics_summary(apples)
    if not metrics.empty:
        display = metrics.copy()
        for col in ["accuracy", "auc"]:
            display[col] = display[col].map(pct)
        for col in ["brier", "log_loss", "ece"]:
            display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
        display["avg_confidence"] = display["avg_confidence"].map(pct)
        st.dataframe(display, use_container_width=True, hide_index=True)

    chart_cols = st.columns(2)

    roc = build_roc_curve_data(apples)
    if not roc.empty:
        roc_fig = px.line(
            roc,
            x="fpr",
            y="tpr",
            color="family",
            title="ROC Curves on Common Settled Cohort",
            hover_data={"threshold": ":.3f"},
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "#888"},
                name="Random",
            )
        )
        chart_cols[0].plotly_chart(roc_fig, use_container_width=True)

    reliability = build_reliability_curve_data(apples)
    if not reliability.empty:
        reliability_fig = go.Figure()
        families = reliability["family"].dropna().unique()
        for family in families:
            sub = reliability[reliability["family"] == family].sort_values("bin_mid")
            reliability_fig.add_trace(
                go.Scatter(
                    x=sub["avg_predicted"],
                    y=sub["actual_rate"],
                    mode="lines+markers",
                    name=family,
                    text=sub["matches"],
                    hovertemplate="Predicted=%{x:.1%}<br>Actual=%{y:.1%}<br>Matches=%{text}<extra>" + family + "</extra>",
                )
            )
        reliability_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "#888"},
                name="Ideal",
            )
        )
        reliability_fig.update_layout(
            title="Reliability / Calibration Curves",
            xaxis_title="Average Predicted P1 Probability",
            yaxis_title="Actual P1 Win Rate",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )
        chart_cols[1].plotly_chart(reliability_fig, use_container_width=True)


def render_overview_tab(prediction_log: pd.DataFrame, family_results: pd.DataFrame, family_summary: pd.DataFrame, version_summary: pd.DataFrame):
    st.subheader("Performance Overview")

    if family_summary.empty:
        st.info("No settled predictions are available for the current filter set.")
        return

    chart_cols = st.columns([1, 1])

    summary_chart = px.bar(
        family_summary,
        x="family",
        y="accuracy",
        color="family",
        text=family_summary["accuracy"].map(pct),
        hover_data={"matches": True, "market_accuracy": ":.3f", "edge_vs_market": ":.3f", "versions": True},
        title="Settled Accuracy by Family",
    )
    summary_chart.update_layout(showlegend=False, yaxis_tickformat=".0%")
    chart_cols[0].plotly_chart(summary_chart, use_container_width=True)

    surface_summary = (
        family_results[family_results["family"] != "Market"]
        .groupby(["family", "surface"], dropna=False)
        .agg(matches=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )
    if not surface_summary.empty:
        surface_chart = px.bar(
            surface_summary,
            x="surface",
            y="accuracy",
            color="family",
            barmode="group",
            text=surface_summary["matches"],
            hover_data={"matches": True, "accuracy": ":.3f"},
            title="Accuracy by Surface",
        )
        surface_chart.update_layout(yaxis_tickformat=".0%")
        chart_cols[1].plotly_chart(surface_chart, use_container_width=True)

    timeline = family_results.sort_values(["family", "settled_at", "effective_logged_at"]).copy()
    timeline["match_number"] = timeline.groupby("family").cumcount() + 1
    timeline["cumulative_accuracy"] = timeline.groupby("family")["correct"].expanding().mean().reset_index(level=0, drop=True)
    time_chart = px.line(
        timeline,
        x="match_number",
        y="cumulative_accuracy",
        color="family",
        hover_data={"match_label": True, "version": True},
        title="Cumulative Accuracy Over Time",
    )
    time_chart.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(time_chart, use_container_width=True)

    if not version_summary.empty:
        version_chart = px.bar(
            version_summary,
            x="version",
            y="accuracy",
            color="family",
            barmode="group",
            text=version_summary["matches"],
            hover_data={"matches": True, "accuracy": ":.3f"},
            title="Accuracy by Model Version",
        )
        version_chart.update_layout(yaxis_tickformat=".0%", xaxis_title="")
        st.plotly_chart(version_chart, use_container_width=True)

    summary_table = family_summary.copy()
    summary_table["accuracy"] = summary_table["accuracy"].map(pct)
    summary_table["market_accuracy"] = summary_table["market_accuracy"].map(pct)
    summary_table["edge_vs_market"] = summary_table["edge_vs_market"].map(pct)
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    render_apples_to_apples_section(prediction_log)


def render_live_slate_tab(live_latest: pd.DataFrame, prediction_log: pd.DataFrame,
                          prediction_snapshots: pd.DataFrame, odds_history: pd.DataFrame,
                          run_id: str = "", run_status: str = ""):
    st.subheader("Live Slate")
    if run_id:
        st.caption(f"Pinned pipeline run: `{run_id}` · status: `{run_status or 'unknown'}`")
    if live_latest.empty:
        st.info("The pinned latest run has no current snapshots matching the active filters.")
        return

    lens_options = {
        "Consensus": "consensus",
        "NN": "nn",
        "XGB": "xgb",
        "RF": "rf",
    }
    lens_label = st.selectbox("Model lens", options=list(lens_options.keys()), key="live_slate_model_lens")
    lens = lens_options[lens_label]
    prob_prefix = "model" if lens == "nn" else lens
    p1_prob_col = f"{prob_prefix}_p1_prob"
    p2_prob_col = f"{prob_prefix}_p2_prob"
    p1_edge_col = f"{lens}_p1_market_edge"
    p2_edge_col = f"{lens}_p2_market_edge"
    p1_lift_col = f"{lens}_p1_market_lift"
    p2_lift_col = f"{lens}_p2_market_lift"

    summary_cols = st.columns(4)
    summary_cols[0].metric("Live matches", f"{len(live_latest):,}")
    summary_cols[1].metric("Highest disagreement", pct(live_latest["probability_range"].max()))
    summary_cols[2].metric(f"Avg {lens_label} P1 edge", pp(live_latest[p1_edge_col].mean() if p1_edge_col in live_latest else None))
    summary_cols[3].metric(f"Avg {lens_label} P2 edge", pp(live_latest[p2_edge_col].mean() if p2_edge_col in live_latest else None))

    disagreement = live_latest.sort_values("probability_range", ascending=False).copy()
    chart_cols = st.columns([1.1, 0.9])
    if p1_prob_col in disagreement.columns:
        chart_data = disagreement.copy()
        chart_data["market_p1_prob"] = pd.to_numeric(
            chart_data["market_p1_prob"], errors="coerce"
        )
        chart_data[p1_prob_col] = pd.to_numeric(
            chart_data[p1_prob_col], errors="coerce"
        )
        chart_data["_marker_size"] = pd.to_numeric(
            chart_data["probability_range"], errors="coerce"
        ).fillna(0.0).clip(lower=0.01)
        chart_data = chart_data.dropna(subset=["market_p1_prob", p1_prob_col])
        if chart_data.empty:
            chart_cols[0].info(
                f"No finite {lens_label} and market probability pairs are available."
            )
        else:
            disagreement_chart = px.scatter(
                chart_data,
                x="market_p1_prob",
                y=p1_prob_col,
                color="surface",
                size="_marker_size",
                hover_name="match_label",
                hover_data={
                    "tournament": True,
                    "round": True,
                    "xgb_p1_prob": ":.3f",
                    "rf_p1_prob": ":.3f",
                    "consensus_p1_prob": ":.3f" if "consensus_p1_prob" in chart_data.columns else False,
                    "probability_range": ":.3f",
                    "_marker_size": False,
                },
                title=f"{lens_label} vs Market on Current Slate",
                labels={"market_p1_prob": "Market P1 Prob", p1_prob_col: f"{lens_label} P1 Prob"},
            )
            disagreement_chart.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line={"dash": "dash", "color": "#94a3b8"},
                    name="No edge",
                )
            )
            chart_cols[0].plotly_chart(disagreement_chart, use_container_width=True)

    if p1_edge_col in disagreement.columns:
        top_edges = disagreement.copy()
        top_edges[p1_edge_col] = pd.to_numeric(
            top_edges[p1_edge_col], errors="coerce"
        )
        top_edges = top_edges.dropna(subset=[p1_edge_col])
        top_edges["abs_edge"] = top_edges[p1_edge_col].abs()
        top_edges = top_edges.sort_values("abs_edge", ascending=False).head(12)
        if top_edges.empty:
            chart_cols[1].info(f"No finite {lens_label} market edges are available.")
        else:
            edge_chart = px.bar(
                top_edges.sort_values(p1_edge_col),
                x=p1_edge_col,
                y="match_label",
                color=p1_edge_col,
                color_continuous_scale="RdBu",
                title=f"Largest {lens_label} P1 edges",
                hover_data={"tournament": True, "round": True, "market_p1_prob": ":.3f", p1_prob_col: ":.3f"},
                labels={p1_edge_col: "Probability point edge", "match_label": ""},
            )
            edge_chart.update_layout(xaxis_tickformat="+.0%", coloraxis_showscale=False)
            chart_cols[1].plotly_chart(edge_chart, use_container_width=True)

    table_cols = [
        "logged_at",
        "match_date",
        "match_start_time",
        "tournament",
        "round",
        "surface",
        "p1",
        "p2",
        p1_prob_col,
        "market_p1_prob",
        p1_edge_col,
        p1_lift_col,
        p2_prob_col,
        "market_p2_prob",
        p2_edge_col,
        p2_lift_col,
        "probability_range",
        "consensus_edge_player",
        "run_id",
        "match_uid",
    ]
    table_cols = [col for col in table_cols if col in disagreement.columns]
    slate_table = disagreement[table_cols].copy()
    rename_map = {
        p1_prob_col: f"{lens_label} P1",
        p2_prob_col: f"{lens_label} P2",
        "market_p1_prob": "Market P1",
        "market_p2_prob": "Market P2",
        p1_edge_col: "P1 edge",
        p2_edge_col: "P2 edge",
        p1_lift_col: "P1 lift",
        p2_lift_col: "P2 lift",
        "probability_range": "Model spread",
        "consensus_edge_player": "Consensus lean",
    }
    for col in [p1_prob_col, p2_prob_col, "market_p1_prob", "market_p2_prob", "probability_range", p1_lift_col, p2_lift_col]:
        if col in slate_table.columns:
            slate_table[col] = slate_table[col].map(pct)
    for col in [p1_edge_col, p2_edge_col]:
        if col in slate_table.columns:
            slate_table[col] = slate_table[col].map(pp)
    slate_table = slate_table.rename(columns=rename_map)
    st.dataframe(slate_table, use_container_width=True, hide_index=True)

    labels = {
        row.match_uid: f"{row.match_date.date() if pd.notna(row.match_date) else 'n/a'} | {row.tournament} | {row.match_label}"
        for row in disagreement.itertuples(index=False)
        if getattr(row, "match_uid", None)
    }
    if labels:
        selected_uid = st.selectbox(
            "Inspect live matchup",
            options=list(labels.keys()),
            format_func=lambda uid: labels.get(uid, uid),
            key="live_match_select",
        )
        render_match_detail(
            selected_uid, prediction_log, prediction_snapshots, odds_history,
            chart_key="live_match_probability_timeline",
        )


def render_feature_snapshot_panel(feature_snapshot_id: str):
    st.markdown("**Feature Snapshot**")
    if not feature_snapshot_id:
        st.caption("No exact feature snapshot is attached to this row.")
        return

    row = load_feature_snapshot_cached(feature_snapshot_id, feature_log_signature())
    if not row:
        st.warning("Feature snapshot id was present, but no matching row was found in `production/logs/features_*.csv`.")
        return

    metadata_columns = [
        "run_id",
        "run_started_at",
        "match_uid",
        "feature_snapshot_id",
        "player1_raw",
        "player2_raw",
        "event",
        "timestamp",
        "match_time",
        "status",
        "meta_level_input",
        "meta_surface_input",
        "meta_round_input",
        "meta_match_date",
        "meta_defaulted_features",
        "meta_draw_input",
        "meta_resolver_source",
        "_source_file",
    ]
    metadata = {
        key: row.get(key)
        for key in metadata_columns
        if key in row and not (pd.isna(row.get(key)) if not isinstance(row.get(key), str) else False)
    }

    feature_items = []
    for key, value in row.items():
        if key.startswith("_") or key in metadata_columns or key == "match_id":
            continue
        if pd.isna(value):
            continue
        feature_items.append({"feature": key, "value": value})

    feature_df = pd.DataFrame(feature_items)
    if feature_df.empty:
        st.caption("No feature values were found in the snapshot row.")
        return
    display_features = build_feature_display_df(feature_df)

    st.caption(f"Snapshot id: `{feature_snapshot_id}`")
    if metadata:
        with st.expander("Snapshot metadata", expanded=False):
            meta_df = pd.DataFrame(metadata.items(), columns=["field", "value"])
            st.dataframe(meta_df, use_container_width=True, hide_index=True)

    numeric_values = pd.to_numeric(display_features["value"], errors="coerce")
    display_features["abs_value"] = numeric_values.abs()
    active_flags = display_features[display_features["value"].isin([1, 1.0, True])].copy().sort_values(["group", "label"])
    nonzero = display_features[display_features["value"] != 0].copy().sort_values("abs_value", ascending=False)
    continuous = display_features[
        numeric_values.notna() & ~display_features["value"].isin([0, 0.0, 1, 1.0, True, False])
    ].copy().sort_values("abs_value", ascending=False)
    display_columns = ["side", "group", "label", "formatted", "description", "feature"]

    overview_cols = st.columns(4)
    overview_cols[0].metric("Feature values", f"{len(display_features):,}")
    overview_cols[1].metric("Active flags", f"{len(active_flags):,}")
    overview_cols[2].metric("Non-zero", f"{len(nonzero):,}")
    overview_cols[3].metric("Largest absolute value", format_feature_value(continuous["value"].iloc[0]) if not continuous.empty else "n/a")

    tabs = st.tabs(["Key Numeric", "Player 1", "Player 2", "Active Flags", "Full Vector"])
    with tabs[0]:
        if continuous.empty:
            st.caption("No continuous numeric feature values were found.")
        else:
            st.dataframe(continuous[display_columns].head(30), use_container_width=True, hide_index=True)
    with tabs[1]:
        p1_features = display_features[display_features["side"] == "Player 1"].sort_values(["group", "label"])
        if p1_features.empty:
            st.caption("No Player 1 features were found.")
        else:
            st.dataframe(p1_features[display_columns], use_container_width=True, hide_index=True)
    with tabs[2]:
        p2_features = display_features[display_features["side"] == "Player 2"].sort_values(["group", "label"])
        if p2_features.empty:
            st.caption("No Player 2 features were found.")
        else:
            st.dataframe(p2_features[display_columns], use_container_width=True, hide_index=True)
    with tabs[3]:
        if active_flags.empty:
            st.caption("No active one-hot flags for this snapshot.")
        else:
            st.dataframe(active_flags[display_columns], use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(
            display_features.sort_values(["side", "group", "label"])[display_columns],
            use_container_width=True,
            hide_index=True,
        )


def render_match_detail(
    selected_uid: str,
    prediction_log: pd.DataFrame,
    prediction_snapshots: pd.DataFrame,
    odds_history: pd.DataFrame,
    *,
    chart_key: str,
):
    if not selected_uid:
        st.info("Select a match to inspect.")
        return

    snap = prediction_snapshots[prediction_snapshots["match_uid"] == selected_uid].sort_values("logged_at").copy()
    odds = odds_history[odds_history["match_uid"] == selected_uid].sort_values("logged_at").copy()
    log_row = prediction_log[prediction_log["match_uid"] == selected_uid].sort_values("effective_logged_at").tail(1)

    if not log_row.empty:
        row = log_row.iloc[0]
        st.markdown(f"### {fmt(row.get('match_label'))}")
        st.caption(f"{fmt(row.get('tournament'))} | {fmt(row.get('surface'))} | {fmt(row.get('level'))} | {fmt(row.get('round'))}")
        top_cols = st.columns(6)
        top_cols[0].metric("Status", str(row.get("record_status", "unknown")))
        top_cols[1].metric("Decision-grade", "yes" if bool(row.get("decision_grade", False)) else "no")
        top_cols[2].metric("NN version", str(row.get("effective_nn_model_version", "n/a")))
        top_cols[3].metric("Settled", "yes" if bool(row.get("is_settled", False)) else "no")
        top_cols[4].metric("XGB version", str(row.get("effective_xgb_model_version", "n/a")))
        top_cols[5].metric("RF version", str(row.get("effective_rf_model_version", "n/a")))

        market_p1 = row.get("market_p1_prob")
        market_p2 = row.get("market_p2_prob")
        probability_rows = []
        for family, p1_col, p2_col in [
            ("NN", "model_p1_prob", "model_p2_prob"),
            ("XGB", "xgb_p1_prob", "xgb_p2_prob"),
            ("RF", "rf_p1_prob", "rf_p2_prob"),
            ("Market", "market_p1_prob", "market_p2_prob"),
        ]:
            p1_prob = row.get(p1_col)
            p2_prob = row.get(p2_col)
            probability_rows.append(
                {
                    "source": family,
                    "P1 probability": pct(p1_prob),
                    "P1 edge": pp(p1_prob - market_p1) if family != "Market" and pd.notna(p1_prob) and pd.notna(market_p1) else "baseline",
                    "P2 probability": pct(p2_prob),
                    "P2 edge": pp(p2_prob - market_p2) if family != "Market" and pd.notna(p2_prob) and pd.notna(market_p2) else "baseline",
                }
            )
        st.dataframe(pd.DataFrame(probability_rows), use_container_width=True, hide_index=True)

    probability_fig = go.Figure()
    if not snap.empty:
        probability_fig.add_trace(go.Scatter(x=snap["logged_at"], y=snap["model_p1_prob"], mode="lines+markers", name="NN"))
        probability_fig.add_trace(go.Scatter(x=snap["logged_at"], y=snap["xgb_p1_prob"], mode="lines+markers", name="XGB"))
        probability_fig.add_trace(go.Scatter(x=snap["logged_at"], y=snap["rf_p1_prob"], mode="lines+markers", name="RF"))
    if not odds.empty:
        probability_fig.add_trace(go.Scatter(x=odds["logged_at"], y=odds["market_p1_prob"], mode="lines+markers", name="Market"))
    probability_fig.update_layout(title="Probability Timeline", yaxis_tickformat=".0%")
    st.plotly_chart(probability_fig, use_container_width=True, key=chart_key)

    details_cols = st.columns(2)
    with details_cols[0]:
        st.markdown("**Prediction snapshots**")
        if snap.empty:
            st.info("No prediction snapshots recorded for this match.")
        else:
            st.dataframe(
                snap[
                    [
                        "logged_at",
                        "run_id",
                        "model_p1_prob",
                        "xgb_p1_prob",
                        "rf_p1_prob",
                        "market_p1_prob",
                        "odds_scraped_at",
                        "match_start_time",
                        "model_version",
                        "xgb_model_version",
                        "rf_model_version",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    with details_cols[1]:
        st.markdown("**Odds history**")
        if odds.empty:
            st.info("No odds history snapshots recorded for this match.")
        else:
            st.dataframe(
                odds[
                    [
                        "logged_at",
                        "odds_scraped_at",
                        "match_start_time",
                        "market_p1_prob",
                        "p1_odds_american",
                        "p2_odds_american",
                        "spread_handicap",
                        "total_games",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    lower_cols = st.columns([1, 1])
    with lower_cols[0]:
        if not log_row.empty:
            row = log_row.iloc[0]
            st.markdown("**Operational / model row**")
            summary = pd.DataFrame(
                [
                    ("match_uid", row.get("match_uid")),
                    ("prediction_uid", row.get("latest_prediction_uid") or row.get("prediction_uid")),
                    ("feature_snapshot_id", row.get("latest_feature_snapshot_id") or row.get("feature_snapshot_id")),
                    ("model_version", row.get("effective_model_version")),
                    ("nn_probability_source", row.get("effective_nn_probability_source")),
                    ("logging_quality", row.get("logging_quality")),
                    ("rescore_quality", row.get("rescore_quality")),
                    ("record_status", row.get("record_status")),
                ],
                columns=["field", "value"],
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
            with st.expander("Full operational row"):
                st.dataframe(log_row.T, use_container_width=True)
    with lower_cols[1]:
        feature_snapshot_id = ""
        if not log_row.empty:
            row = log_row.iloc[0]
            feature_snapshot_id = row.get("latest_feature_snapshot_id") or row.get("feature_snapshot_id") or ""
        render_feature_snapshot_panel(str(feature_snapshot_id))


def render_prediction_log_tab(prediction_log: pd.DataFrame, prediction_snapshots: pd.DataFrame, odds_history: pd.DataFrame):
    st.subheader("Prediction Log Browser")
    if prediction_log.empty:
        st.info("No prediction log rows are available.")
        return

    filter_cols = st.columns([1.2, 1.2, 1.4, 1])
    tournaments = filter_cols[0].multiselect(
        "Tournaments",
        options=sorted(x for x in prediction_log["tournament"].dropna().astype(str).unique() if x),
    )
    player_query = filter_cols[1].text_input("Player search", placeholder="Rublev, Fils, Shelton...")
    settlement_state = filter_cols[2].selectbox("Settlement state", ["All", "Settled only", "Pending only"])
    max_rows = filter_cols[3].slider("Rows shown", min_value=10, max_value=500, value=100, step=10)

    filtered = prediction_log.copy()
    if tournaments:
        filtered = filtered[filtered["tournament"].isin(tournaments)]
    if player_query.strip():
        mask = filtered["match_label"].fillna("").str.contains(player_query, case=False, na=False)
        filtered = filtered[mask]
    if settlement_state == "Settled only":
        filtered = filtered[filtered["is_settled"]]
    elif settlement_state == "Pending only":
        filtered = filtered[~filtered["is_settled"]]

    filtered = filtered.sort_values(["effective_logged_at", "effective_match_date"], ascending=[False, False]).copy()
    for label, p1_col, p2_col in [
        ("nn", "model_p1_prob", "model_p2_prob"),
        ("xgb", "xgb_p1_prob", "xgb_p2_prob"),
        ("rf", "rf_p1_prob", "rf_p2_prob"),
    ]:
        if p1_col in filtered.columns and "market_p1_prob" in filtered.columns:
            filtered[f"{label}_p1_edge"] = filtered[p1_col] - filtered["market_p1_prob"]
        if p2_col in filtered.columns and "market_p2_prob" in filtered.columns:
            filtered[f"{label}_p2_edge"] = filtered[p2_col] - filtered["market_p2_prob"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(filtered):,}")
    metric_cols[1].metric("Settled", f"{int(filtered['is_settled'].sum()):,}" if "is_settled" in filtered.columns else "0")
    metric_cols[2].metric("Decision-grade", f"{int(filtered['decision_grade'].sum()):,}" if "decision_grade" in filtered.columns else "0")
    metric_cols[3].metric("Exact feature snapshots", f"{int(filtered['latest_feature_snapshot_id'].fillna('').astype(bool).sum()):,}" if "latest_feature_snapshot_id" in filtered.columns else "0")

    table = filtered[
        [
            "effective_logged_at",
            "effective_match_date",
            "tournament",
            "round",
            "surface",
            "level",
            "match_label",
            "model_p1_prob",
            "xgb_p1_prob",
            "rf_p1_prob",
            "market_p1_prob",
            "nn_p1_edge",
            "xgb_p1_edge",
            "rf_p1_edge",
            "effective_model_version",
            "effective_xgb_model_version",
            "effective_rf_model_version",
            "effective_nn_probability_source",
            "record_status",
            "logging_quality",
            "rescore_quality",
        ]
    ].head(max_rows).copy()
    for col in ["model_p1_prob", "xgb_p1_prob", "rf_p1_prob", "market_p1_prob"]:
        if col in table.columns:
            table[col] = table[col].map(pct)
    for col in ["nn_p1_edge", "xgb_p1_edge", "rf_p1_edge"]:
        if col in table.columns:
            table[col] = table[col].map(pp)
    table = table.rename(
        columns={
            "nn_p1_edge": "NN P1 edge",
            "xgb_p1_edge": "XGB P1 edge",
            "rf_p1_edge": "RF P1 edge",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    if filtered.empty:
        return

    labels = {
        row.match_uid: f"{fmt(row.effective_match_date).date() if pd.notna(row.effective_match_date) else 'n/a'} | {row.tournament} | {row.round} | {row.match_label}"
        for row in filtered.head(max_rows).itertuples(index=False)
    }
    selected_uid = st.selectbox(
        "Inspect a logged match",
        options=list(labels.keys()),
        format_func=lambda uid: labels.get(uid, uid),
        key="prediction_log_match_select",
    )
    render_match_detail(
        selected_uid, prediction_log, prediction_snapshots, odds_history,
        chart_key="prediction_log_probability_timeline",
    )


def render_match_explorer_tab(prediction_log: pd.DataFrame, prediction_snapshots: pd.DataFrame, odds_history: pd.DataFrame):
    st.subheader("Match Explorer")
    catalog = build_match_catalog(prediction_log, prediction_snapshots)
    if catalog.empty:
        st.info("No match lineage is available yet.")
        return

    labels = {
        row.match_uid: f"{row.event_label} [{row.match_uid}]"
        for row in catalog.itertuples(index=False)
    }
    default_uid = catalog.iloc[0]["match_uid"]
    selected_uid = st.selectbox(
        "Select a match",
        options=list(labels.keys()),
        format_func=lambda uid: labels.get(uid, uid),
        index=0 if default_uid in labels else None,
        key="generic_match_select",
    )
    render_match_detail(
        selected_uid, prediction_log, prediction_snapshots, odds_history,
        chart_key="match_explorer_probability_timeline",
    )


def render_bets_tab(all_bets: pd.DataFrame, betting_sessions: pd.DataFrame):
    st.subheader("Bets and Bankroll")
    if all_bets.empty:
        st.info("No bet-tracking rows were found.")
        return

    settled = all_bets[all_bets["status"] == "settled"].copy()
    pending = all_bets[all_bets["status"] == "pending"].copy()
    total_profit = settled["actual_profit"].fillna(0).sum() if not settled.empty else 0.0
    total_staked = settled["stake"].fillna(0).sum() if not settled.empty else 0.0
    roi = (total_profit / total_staked) if total_staked else None
    win_rate = (settled["outcome"] == "win").mean() if not settled.empty else None

    bet_cols = st.columns(4)
    bet_cols[0].metric("Settled bets", f"{len(settled):,}")
    bet_cols[1].metric("Pending bets", f"{len(pending):,}")
    bet_cols[2].metric("Win rate", pct(win_rate))
    bet_cols[3].metric("Realized ROI", pct(roi))

    if not settled.empty:
        settled = settled.sort_values("settled_timestamp").copy()
        settled["cumulative_profit"] = settled["actual_profit"].fillna(0).cumsum()
        pnl_chart = px.line(
            settled,
            x="settled_timestamp",
            y="cumulative_profit",
            title="Cumulative Realized P/L",
            hover_data={"match": True, "model_version": True, "actual_profit": ":.2f"},
        )
        st.plotly_chart(pnl_chart, use_container_width=True)

        outcome_summary = (
            settled.groupby(["model_version", "outcome"], dropna=False)
            .size()
            .reset_index(name="bets")
            .sort_values("bets", ascending=False)
        )
        outcome_chart = px.bar(
            outcome_summary,
            x="model_version",
            y="bets",
            color="outcome",
            barmode="group",
            title="Settled Bets by Model Version",
        )
        st.plotly_chart(outcome_chart, use_container_width=True)

    if not betting_sessions.empty:
        with st.expander("Betting sessions"):
            st.dataframe(betting_sessions.sort_values("start_time", ascending=False), use_container_width=True, hide_index=True)

    with st.expander("Raw bet log"):
        st.dataframe(all_bets.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)


def render_ops_tab(run_history: pd.DataFrame, prediction_snapshots: pd.DataFrame, odds_history: pd.DataFrame, skipped_live_matches: pd.DataFrame, settlement_audit: pd.DataFrame):
    st.subheader("Ops and Audit")

    if run_history.empty:
        st.info("Audit run history has not populated yet, so this section is using a derived fallback from snapshots and odds history.")
        run_history = build_derived_run_history(prediction_snapshots, odds_history)
    else:
        run_history = run_history.copy()
        run_history["run_source"] = "audit"

    if not run_history.empty:
        runs_table = run_history.sort_values("started_at", ascending=False).copy()
        st.dataframe(runs_table, use_container_width=True, hide_index=True)

        if "prediction_rows_success" in runs_table.columns or "odds_rows_fetched" in runs_table.columns:
            melted = runs_table.melt(
                id_vars=["run_id"],
                value_vars=[c for c in ["prediction_rows_success", "odds_rows_fetched"] if c in runs_table.columns],
                var_name="metric",
                value_name="count",
            )
            melted = melted.dropna(subset=["count"])
            if not melted.empty:
                run_chart = px.bar(
                    melted,
                    x="run_id",
                    y="count",
                    color="metric",
                    barmode="group",
                    title="Per-run snapshot counts",
                )
                st.plotly_chart(run_chart, use_container_width=True)

    audit_cols = st.columns(2)
    with audit_cols[0]:
        st.markdown("**Skipped live matches**")
        if skipped_live_matches.empty:
            st.caption("No skipped-match audit rows recorded yet.")
        else:
            skip_summary = (
                skipped_live_matches.groupby("skip_reason_code", dropna=False)
                .size()
                .reset_index(name="matches")
                .sort_values("matches", ascending=False)
            )
            skip_chart = px.bar(skip_summary, x="skip_reason_code", y="matches", title="Skip reasons")
            st.plotly_chart(skip_chart, use_container_width=True)
            st.dataframe(skipped_live_matches.sort_values("logged_at", ascending=False), use_container_width=True, hide_index=True)

    with audit_cols[1]:
        st.markdown("**Settlement audit**")
        if settlement_audit.empty:
            st.caption("No settlement audit rows recorded yet.")
        else:
            settle_summary = (
                settlement_audit.groupby("outcome_code", dropna=False)
                .size()
                .reset_index(name="attempts")
                .sort_values("attempts", ascending=False)
            )
            settle_chart = px.bar(settle_summary, x="outcome_code", y="attempts", title="Settlement outcomes")
            st.plotly_chart(settle_chart, use_container_width=True)
            st.dataframe(settlement_audit.sort_values("logged_at", ascending=False), use_container_width=True, hide_index=True)


def main():
    render_dashboard_css()
    st.title("Tennis Betting Operations Dashboard")
    st.caption("Live predictions, model-vs-market performance, odds movement, version lineage, and operations audit in one place.")

    data = load_data_cached(file_mtimes())
    prediction_log = data["prediction_log"]
    prediction_snapshots = data["prediction_snapshots"]
    odds_history = data["odds_history"]

    if prediction_log.empty and prediction_snapshots.empty:
        st.error("No production logs were found. The dashboard needs prediction_log.csv and/or prediction_snapshots.csv.")
        return

    history_for_filters = prediction_log if not prediction_log.empty else prediction_snapshots.rename(columns={"logged_at": "effective_logged_at"})
    min_dt = history_for_filters["effective_logged_at"].dropna().min()
    max_dt = history_for_filters["effective_logged_at"].dropna().max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        min_dt = pd.Timestamp.today().normalize()
        max_dt = min_dt

    st.sidebar.header("Filters")
    trust_mode = st.sidebar.radio(
        "History quality",
        options=["All history", "Decision-grade only"],
        help="Decision-grade uses snapshot_v2 rows with exact feature snapshots. All history also includes legacy backfilled rows.",
    )
    refresh_label = st.sidebar.selectbox("Auto-refresh", options=["Off", "15s", "30s", "60s", "5m"], index=2)
    refresh_map = {"Off": None, "15s": "15s", "30s": "30s", "60s": "60s", "5m": "5m"}

    date_range = st.sidebar.date_input(
        "Prediction log window",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = max_dt.date()

    surfaces = st.sidebar.multiselect(
        "Surfaces",
        options=sorted(x for x in prediction_log.get("surface", pd.Series(dtype=str)).dropna().astype(str).unique() if x),
    )
    levels = st.sidebar.multiselect(
        "Levels",
        options=sorted(x for x in prediction_log.get("level", pd.Series(dtype=str)).dropna().astype(str).unique() if x),
    )
    rounds = st.sidebar.multiselect(
        "Rounds",
        options=sorted(x for x in prediction_log.get("round", pd.Series(dtype=str)).dropna().astype(str).unique() if x),
    )
    statuses = st.sidebar.multiselect(
        "Record status",
        options=sorted(x for x in prediction_log.get("record_status", pd.Series(dtype=str)).dropna().astype(str).unique() if x),
    )
    versions = st.sidebar.multiselect(
        "Primary model versions",
        options=sorted(x for x in prediction_log.get("effective_model_version", pd.Series(dtype=str)).dropna().astype(str).unique() if x),
    )

    @st.fragment(run_every=refresh_map[refresh_label])
    def render_dashboard():
        fresh = load_data_cached(file_mtimes())
        fixed_run_id = latest_pipeline_run_id(
            fresh["run_history"], fresh["prediction_snapshots"]
        )
        fixed_run_status = ""
        if fixed_run_id and not fresh["run_history"].empty:
            run_ids = fresh["run_history"].get(
                "run_id", pd.Series("", index=fresh["run_history"].index)
            ).fillna("").astype(str)
            run_row = fresh["run_history"][run_ids == fixed_run_id]
            if not run_row.empty:
                fixed_run_status = str(run_row.iloc[-1].get("status", ""))
        run_snapshots = fresh["prediction_snapshots"]
        if fixed_run_id and "run_id" in run_snapshots.columns:
            run_snapshots = run_snapshots[
                run_snapshots["run_id"].fillna("").astype(str) == fixed_run_id
            ].copy()
        filtered_log = apply_history_filters(
            fresh["prediction_log"],
            trust_mode=trust_mode,
            surfaces=surfaces,
            levels=levels,
            rounds=rounds,
            statuses=statuses,
            versions=versions,
            start_date=start_date,
            end_date=end_date,
        )
        filtered_snaps = apply_live_filters(
            run_snapshots,
            trust_mode=trust_mode,
            surfaces=surfaces,
            levels=levels,
            rounds=rounds,
            versions=versions,
            start_date=start_date,
            end_date=end_date,
        )
        live_latest = build_live_latest_snapshots(filtered_snaps, run_id=fixed_run_id)
        family_results = build_family_results(filtered_log)
        family_summary = build_family_accuracy_summary(filtered_log)
        version_summary = build_version_summary(family_results)
        latest_data_ts = pick_latest_timestamp(
            filtered_log.get("effective_logged_at"),
            filtered_snaps.get("logged_at"),
            fresh["odds_history"].get("logged_at"),
        )

        render_summary_metrics(filtered_log, live_latest, family_summary, latest_data_ts)
        render_insight_strip(filtered_log, live_latest, fresh["run_history"])

        tab_overview, tab_log, tab_live, tab_match, tab_bets, tab_ops = st.tabs(
            ["Overview", "Prediction Log", "Live Slate", "Match Explorer", "Bets", "Ops & Audit"]
        )

        with tab_overview:
            render_overview_tab(filtered_log, family_results, family_summary, version_summary)

            calibration = build_calibration_summary(family_results)
            if not calibration.empty:
                calibration_chart = px.line(
                    calibration,
                    x="confidence_bin",
                    y="win_rate",
                    color="family",
                    markers=True,
                    hover_data={"matches": True, "avg_confidence": ":.3f"},
                    title="Confidence vs Realized Win Rate",
                )
                calibration_chart.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(calibration_chart, use_container_width=True)

        with tab_log:
            render_prediction_log_tab(filtered_log, fresh["prediction_snapshots"], fresh["odds_history"])

        with tab_live:
            render_live_slate_tab(
                live_latest, filtered_log, fresh["prediction_snapshots"],
                fresh["odds_history"], run_id=fixed_run_id,
                run_status=fixed_run_status,
            )

        with tab_match:
            render_match_explorer_tab(filtered_log, fresh["prediction_snapshots"], fresh["odds_history"])

        with tab_bets:
            render_bets_tab(fresh["all_bets"], fresh["betting_sessions"])

        with tab_ops:
            render_ops_tab(
                fresh["run_history"],
                fresh["prediction_snapshots"],
                fresh["odds_history"],
                fresh["skipped_live_matches"],
                fresh["settlement_audit"],
            )

    render_dashboard()


if __name__ == "__main__":
    main()
