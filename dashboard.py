"""Privacy-minded classroom observation dashboard demo.

Run with: streamlit run dashboard.py
"""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


st.set_page_config(
    page_title="Classroom Signals | Observation Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones so it doesn't cover the page
)

REQUIRED_COLUMNS = {
    "date",
    "learner_id",
    "engagement_pct",
    "movement_events",
    "fidgeting_events",
    "looking_away_events",
    "participation_pct",
    "observation_minutes",
}

INK = "#1B2F6B"
GREEN_PEN = "#1F8A70"
SKY = "#7FA6D6"
PENCIL = "#5B6780"
RULE = "#CFE0F2"


def make_demo_data() -> pd.DataFrame:
    """Create clearly synthetic, repeatable sample observations."""
    rng = np.random.default_rng(27)
    rows: list[dict] = []
    today = date.today()
    learner_ids = [f"L-{number:03d}" for number in range(1, 13)]
    learner_baselines = rng.normal(68, 10, len(learner_ids))

    for day_offset in range(27, -1, -1):
        observation_date = today - timedelta(days=day_offset)
        if observation_date.weekday() >= 5:
            continue
        for index, learner_id in enumerate(learner_ids):
            daily_shift = 5 * np.sin((27 - day_offset) / 5)
            engagement = float(np.clip(learner_baselines[index] + daily_shift + rng.normal(0, 9), 18, 98))
            minutes = 35
            movement = int(np.clip(rng.poisson(13 + (70 - engagement) * 0.23), 1, 55))
            fidgeting = int(np.clip(rng.poisson(7 + (70 - engagement) * 0.12), 0, 35))
            looking_away = int(np.clip(rng.poisson(10 + (70 - engagement) * 0.18), 0, 40))
            participation = float(np.clip(engagement + rng.normal(1, 11), 10, 100))
            rate = movement / minutes
            activity_level = "Low" if rate < 0.34 else "Moderate" if rate < 0.68 else "High"
            rows.append(
                {
                    "date": observation_date,
                    "learner_id": learner_id,
                    "engagement_pct": round(engagement, 1),
                    "movement_events": movement,
                    "fidgeting_events": fidgeting,
                    "looking_away_events": looking_away,
                    "participation_pct": round(participation, 1),
                    "observation_minutes": minutes,
                    "activity_level": activity_level,
                }
            )
    return pd.DataFrame(rows)


def load_uploaded_data(upload) -> tuple[pd.DataFrame | None, str | None]:
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:  # show a helpful message for malformed uploads
        return None, f"Could not read this CSV: {exc}"

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        return None, "Missing required columns: " + ", ".join(missing)

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    numeric_columns = [
        "engagement_pct",
        "movement_events",
        "fidgeting_events",
        "looking_away_events",
        "participation_pct",
        "observation_minutes",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["learner_id"] = frame["learner_id"].astype("string").str.strip()
    frame = frame.dropna(subset=["date", "learner_id", *numeric_columns])
    frame = frame[frame["learner_id"] != ""]

    if frame.empty:
        return None, "No usable rows found. Check that dates, learner IDs, and numeric values are filled in."

    if "activity_level" not in frame.columns:
        rate = frame["movement_events"] / frame["observation_minutes"].replace(0, np.nan)
        frame["activity_level"] = np.select(
            [rate < 0.34, rate < 0.68], ["Low", "Moderate"], default="High"
        )
        frame.loc[rate.isna(), "activity_level"] = "Unrated"
    return frame.sort_values("date"), None


def style_chart(fig, height: int = 310):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=18, b=8),
        dragmode=False,  # drag-to-zoom captures touches and stops the page scrolling on phones
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Andika, system-ui, sans-serif", color=PENCIL, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor=RULE),
        yaxis=dict(gridcolor=RULE, zeroline=False),
    )
    return fig


def pct(value) -> str:
    return f"{value:.0f}%" if pd.notna(value) else "—"


def pct_share(share: float) -> str:
    return f"{share * 100:.0f}%"


def sparkline(values: list[float]) -> str:
    """Draw a small engagement line on a fixed 0-100 scale, so learners stay comparable."""
    width, height = 120, 24
    label = f"Engagement from {values[0]:.0f}% to {values[-1]:.0f}%"
    ys = [height - 2 - min(max(value, 0), 100) / 100 * (height - 4) for value in values]
    if len(values) == 1:
        mark = f'<circle cx="{width / 2}" cy="{ys[0]:.1f}" r="2.5" fill="currentColor"></circle>'
    else:
        step = width / (len(values) - 1)
        points = " ".join(f"{index * step:.1f},{y:.1f}" for index, y in enumerate(ys))
        mark = f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="1.5" vector-effect="non-scaling-stroke"></polyline>'
    return f'<svg role="img" aria-label="{label}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">{mark}</svg>'


def register_html(frame: pd.DataFrame, review_ids: set[str]) -> str:
    """Build the learner register. Rows stay in ID order: this view never ranks children."""
    rows = []
    for learner_id, group in frame.sort_values("date").groupby(frame["learner_id"].astype(str), sort=True):
        safe_id = escape(learner_id)
        if learner_id in review_ids:
            who = f'<mark>{safe_id}</mark><span class="flag">Review</span>'
        else:
            who = safe_id
        rows.append(
            f'<tr><th scope="row">{who}</th>'
            f'<td class="num">{pct(group["engagement_pct"].mean())}</td>'
            f'<td class="num">{pct(group["participation_pct"].mean())}</td>'
            f'<td class="trend">{sparkline(group["engagement_pct"].tolist())}</td>'
            f'<td class="num obs">{len(group):,}</td></tr>'
        )
    head = (
        '<tr><th scope="col">Learner</th><th scope="col" class="num">Engagement</th>'
        '<th scope="col" class="num">Participation</th><th scope="col" class="trend"><span class="wide">Engagement trend</span><span class="narrow">Trend</span></th>'
        '<th scope="col" class="num obs">Observations</th></tr>'
    )
    return f'<table class="register"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def make_pdf_report(frame: pd.DataFrame, start: date, end: date, learner: str) -> bytes:
    """Build a concise, non-diagnostic PDF summary for the selected data."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Classroom Observation Summary",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Note",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#64748B"),
        )
    )
    title_style = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor(INK),
        fontSize=21,
        leading=26,
        alignment=0,
        spaceAfter=7,
    )
    story = [
        Paragraph("Classroom Observation Summary", title_style),
        Paragraph(f"Period: {start:%d %b %Y} – {end:%d %b %Y} · Scope: {escape(learner)}", styles["BodyText"]),
        Spacer(1, 14),
    ]

    overview = [
        ["Observations", "Learners", "Avg. engagement", "Avg. participation"],
        [
            f"{len(frame):,}",
            f"{frame['learner_id'].nunique():,}",
            pct(frame["engagement_pct"].mean()),
            pct(frame["participation_pct"].mean()),
        ],
    ]
    overview_table = Table(overview, colWidths=[1.25 * inch] * 4)
    overview_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK)),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor(INK)),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8E2E8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8E2E8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend([overview_table, Spacer(1, 18), Paragraph("Learner summaries", styles["Heading2"])])

    summary = (
        frame.groupby("learner_id", as_index=False)
        .agg(
            observations=("date", "count"),
            engagement=("engagement_pct", "mean"),
            participation=("participation_pct", "mean"),
            movement=("movement_events", "mean"),
        )
        .sort_values("learner_id")
    )
    table_data = [["Learner ID", "Observations", "Engagement", "Participation", "Movement / obs."]]
    for row in summary.itertuples(index=False):
        table_data.append(
            [
                escape(str(row.learner_id)),
                f"{int(row.observations)}",
                pct(row.engagement),
                pct(row.participation),
                f"{row.movement:.1f}",
            ]
        )
    detail_table = Table(table_data, colWidths=[1.35 * inch, 1.0 * inch, 1.0 * inch, 1.05 * inch, 1.2 * inch], repeatRows=1)
    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8E2E8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([detail_table, Spacer(1, 16), Paragraph("Interpretation", styles["Heading2"])])
    story.append(
        Paragraph(
            "These are descriptive summaries of recorded observations. They are intended to support educator reflection and follow-up, not to label, rank, or diagnose any learner. Interpret results in classroom context and in line with school consent and data-retention practices.",
            styles["Note"],
        )
    )
    document.build(story)
    return buffer.getvalue()


st.markdown(
    """
    <style>
      /* The page: white paper with the red margin line of an exercise book. */
      .block-container {
        max-width: 1180px;
        /* Streamlit sizes this to one screen; it must grow so the margin line runs the full page. */
        height: auto !important;
        flex: 1 0 auto;
        min-height: 100dvh;
        padding: 3.75rem 2.5rem 2.5rem 6rem;
        background: linear-gradient(#C8363F, #C8363F) 4.5rem 0 / 2px 100% no-repeat;
      }
      [data-testid="stMain"] h1 { font-size: 2.75rem; font-weight: 700; line-height: 1.15; padding: 0; }
      [data-testid="stMain"] :is(h2, h3) { font-size: 1.5rem; font-weight: 700; padding: .5rem 0 .25rem; }
      [data-testid="stMain"] p.scope { color: #5B6780; font-size: 1.05rem; margin: 0 0 1rem; }
      [data-testid="stMain"] p.page-note { color: #5B6780; font-size: .9rem; line-height: 1.5; max-width: 70ch; margin: .75rem 0 1.5rem; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #5B6780 !important; opacity: 1 !important; }
      [data-testid="stMain"] code { color: #1B2F6B; background: #EAF1F9; }
      /* Both dates need the room the calendar icon takes; clicking the field still opens the calendar. */
      [data-testid="stSidebar"] [data-testid="stDateInput"] input { font-size: .875rem; padding-right: 0; }
      [data-testid="stSidebar"] [data-testid="stDateInput"] svg { display: none; }
      .register .narrow { display: none; }
      :focus-visible { outline: 2px solid #1B2F6B; outline-offset: 2px; }

      /* Totals sit in the page as a written line, not as cards. */
      [data-testid="stMetricLabel"] { color: #5B6780; }
      [data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 700; font-variant-numeric: tabular-nums; }

      /* The register: every learner on a ruled line, in ID order. */
      [data-testid="stMain"] table.register { display: table; width: 100%; border-collapse: collapse; border: 0; margin: .25rem 0 0; font-variant-numeric: tabular-nums; }
      [data-testid="stMain"] table.register tr { border: 0; background: transparent; }
      [data-testid="stMain"] table.register :is(th, td) { border: 0; border-bottom: 1px solid #CFE0F2; padding: 0 .75rem; height: 2.5rem; background: transparent; color: #1B2F6B; text-align: left; vertical-align: middle; font-size: 1rem; white-space: nowrap; }
      [data-testid="stMain"] table.register thead th { border-bottom: 2px solid #1B2F6B; font-weight: 700; font-size: .9rem; }
      [data-testid="stMain"] table.register tbody th { font-weight: 700; }
      [data-testid="stMain"] table.register .num { text-align: right; }
      [data-testid="stMain"] table.register td.trend { width: 40%; color: #1B2F6B; }
      .register svg { display: block; width: 100%; max-width: 280px; height: 24px; }
      .register mark { background: #FFF1A8; color: #1B2F6B; padding: .1rem .35rem; margin-left: -.35rem; }
      .register .flag { display: inline-block; margin-left: .5rem; font-size: .8rem; font-weight: 400; }

      .stDownloadButton button { background: #1B2F6B; color: white; border: 0; font-weight: 700; }
      .stDownloadButton button p { color: white; }

      /* Touch devices: no tap flash, larger targets, no iOS zoom on input focus. */
      html { -webkit-tap-highlight-color: transparent; }
      button, [role="tab"] { touch-action: manipulation; }
      .stDownloadButton button:active { background: #12214D; }
      @media (hover: none) {
        .modebar-container { display: none !important; }
      }
      @media (pointer: coarse) {
        input, textarea, select { font-size: 16px !important; }
        [role="tab"], .stDownloadButton button { min-height: 44px; }
      }

      /* Phone layout */
      @media (max-width: 640px) {
        .block-container { padding: 3.75rem .875rem 2rem 2rem; background-position: 1.125rem 0; }
        [data-testid="stMain"] h1 { font-size: 2.25rem; }
        [data-testid="stMain"] :is(h2, h3) { font-size: 1.3rem; }
        [data-testid="stMain"] p.scope { font-size: .95rem; }
        [role="tablist"] { gap: .75rem; }
        [class*="st-key-kpis"] [data-testid="stHorizontalBlock"] { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
        [class*="st-key-kpis"] [data-testid="stColumn"] { width: auto; min-width: 0; }
        [class*="st-key-kpis"] [data-testid="stColumn"]:last-child:nth-child(odd) { grid-column: 1 / -1; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important; }
        [data-testid="stMain"] table.register :is(th, td) { padding: 0 .25rem; font-size: .9rem; }
        .register .wide { display: none; }
        .register .narrow { display: inline; }
        [data-testid="stMain"] table.register { table-layout: fixed; }
        [data-testid="stMain"] table.register tbody th { white-space: normal; overflow-wrap: anywhere; }
        [data-testid="stMain"] table.register thead th { font-size: .72rem; white-space: normal; line-height: 1.2; }
        [data-testid="stMain"] table.register thead th:nth-child(1) { width: 20%; }
        [data-testid="stMain"] table.register thead th:nth-child(2) { width: 28%; }
        [data-testid="stMain"] table.register thead th:nth-child(3) { width: 30%; }
        [data-testid="stMain"] table.register thead th:nth-child(4) { width: auto; }
        [data-testid="stMain"] table.register .obs { display: none; }
        .register .flag { display: block; margin-left: 0; font-size: .7rem; line-height: 1; }
        [data-testid="stElementContainer"]:has(.stDownloadButton), .stDownloadButton, .stDownloadButton button { width: 100% !important; }
      }
      /* Smallest phones: keep the column headings on one line. */
      @media (max-width: 360px) {
        [data-testid="stMain"] table.register thead th { font-size: .68rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Classroom Signals")
    uploaded = st.file_uploader("Upload an observation CSV", type=["csv"], help="The dashboard expects the columns listed under the About tab.")
    if uploaded is None:
        data = make_demo_data()
        st.info("Showing synthetic demo data. No real learner information is included.")
    else:
        data, error = load_uploaded_data(uploaded)
        if error:
            st.error(error)
            st.stop()
        st.success(f"Loaded {len(data):,} usable observations.")

    minimum_date = min(data["date"])
    maximum_date = max(data["date"])
    selected_dates = st.date_input(
        "Observation period",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    learner_options = ["All learners", *sorted(data["learner_id"].dropna().astype(str).unique())]
    selected_learner = st.selectbox("Learner", learner_options)
    st.caption("Use pseudonymous IDs. This tool supports observation and discussion; it does not diagnose or rank children.")

# While a range is being picked, the date input briefly returns only its first date.
if isinstance(selected_dates, (tuple, list)):
    start_date, end_date = selected_dates[0], selected_dates[-1]
else:
    start_date = end_date = selected_dates

filtered = data[(data["date"] >= start_date) & (data["date"] <= end_date)].copy()
if selected_learner != "All learners":
    filtered = filtered[filtered["learner_id"].astype(str) == selected_learner]

scope = "all learners" if selected_learner == "All learners" else escape(selected_learner)
st.title("Class register")
st.markdown(f'<p class="scope">{start_date:%d %b %Y} to {end_date:%d %b %Y}, {scope}</p>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("No observations match this learner and period. Change the filters in the side panel.")
    st.stop()

avg_engagement = filtered["engagement_pct"].mean()
avg_participation = filtered["participation_pct"].mean()
learner_count = filtered["learner_id"].nunique()
observation_count = len(filtered)

metric_columns = st.container(key="kpis_overview").columns(4)
metric_columns[0].metric("Average engagement", pct(avg_engagement))
metric_columns[1].metric("Participation", pct(avg_participation))
metric_columns[2].metric("Observations", f"{observation_count:,}")
metric_columns[3].metric("Learners included", f"{learner_count:,}")

overview_tab, learner_tab, reports_tab, data_tab, privacy_tab = st.tabs(
    ["Register", "Learner", "Reports", "Data", "About"]
)

with overview_tab:
    per_learner = (
        filtered.sort_values("date")
        .groupby("learner_id", group_keys=False)
        .tail(3)
        .groupby("learner_id")
        .agg(Recent_observations=("engagement_pct", "size"), Recent_engagement=("engagement_pct", "mean"))
        .reset_index()
    )
    review = per_learner[(per_learner["Recent_observations"] >= 3) & (per_learner["Recent_engagement"] < 45)]
    st.markdown(register_html(filtered, set(review["learner_id"].astype(str))), unsafe_allow_html=True)
    if review.empty:
        review_note = "No learner is marked for review in this selection. Keep using classroom context and your own judgment."
    else:
        review_note = "Marked for review: the last three observations average below 45% engagement. This is a prompt for context, never a conclusion about a learner."
    st.markdown(f'<p class="page-note">{review_note}</p>', unsafe_allow_html=True)

    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Engagement over time")
        trend = filtered.groupby("date", as_index=False).agg(
            Engagement=("engagement_pct", "mean"), Participation=("participation_pct", "mean")
        )
        trend_long = trend.melt(id_vars="date", var_name="Measure", value_name="Percent")
        fig = px.line(
            trend_long,
            x="date",
            y="Percent",
            color="Measure",
            markers=True,
            color_discrete_map={"Engagement": INK, "Participation": GREEN_PEN},
            labels={"date": "Date", "Percent": "Average", "Measure": ""},
        )
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
    with right:
        st.subheader("Observed activity mix")
        activity = filtered["activity_level"].value_counts().rename_axis("Level").reset_index(name="Observations")
        shares = activity["Observations"] / activity["Observations"].sum()
        activity["Label"] = activity["Level"].astype(str) + " " + shares.map(pct_share)
        level_colors = {"Low": RULE, "Moderate": SKY, "High": INK, "Unrated": "#AAB5C2"}
        order = [name for name in level_colors if name in activity["Level"].values]
        activity = activity.set_index("Level").loc[order].reset_index()
        fig = px.bar(
            activity,
            x="Observations",
            y=[""] * len(activity),
            color="Label",
            orientation="h",
            color_discrete_sequence=[level_colors[name] for name in order],
            labels={"Label": ""},
        )
        style_chart(fig, 150)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(bargap=0.25)
        st.plotly_chart(fig, use_container_width=True)

with learner_tab:
    st.subheader("Learner detail")
    if selected_learner == "All learners":
        learner_for_detail = st.selectbox(
            "Choose a learner ID to inspect", sorted(filtered["learner_id"].astype(str).unique()), key="detail_learner"
        )
    else:
        learner_for_detail = selected_learner
        st.caption(f"Showing pseudonymous learner ID {learner_for_detail}.")

    learner_data = filtered[filtered["learner_id"].astype(str) == learner_for_detail].sort_values("date")
    c1, c2, c3 = st.container(key="kpis_learner").columns(3)
    c1.metric("Average engagement", pct(learner_data["engagement_pct"].mean()))
    c2.metric("Average participation", pct(learner_data["participation_pct"].mean()))
    c3.metric("Observations", f"{len(learner_data):,}")
    learner_long = learner_data.melt(
        id_vars="date", value_vars=["engagement_pct", "participation_pct"], var_name="Measure", value_name="Percent"
    )
    learner_long["Measure"] = learner_long["Measure"].map(
        {"engagement_pct": "Engagement", "participation_pct": "Participation"}
    )
    fig = px.line(
        learner_long,
        x="date",
        y="Percent",
        color="Measure",
        markers=True,
        color_discrete_map={"Engagement": INK, "Participation": GREEN_PEN},
        labels={"date": "Date", "Percent": "Observed score", "Measure": ""},
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(style_chart(fig, 360), use_container_width=True)

    signals = learner_data[["date", "movement_events", "fidgeting_events", "looking_away_events"]].melt(
        id_vars="date", var_name="Observation", value_name="Events"
    )
    signals["Observation"] = signals["Observation"].map(
        {"movement_events": "Movement", "fidgeting_events": "Fidgeting", "looking_away_events": "Looking away"}
    )
    fig = px.bar(signals, x="date", y="Events", color="Observation", barmode="group", color_discrete_sequence=[INK, GREEN_PEN, SKY], labels={"date": "Date"})
    fig.update_layout(legend_title_text="")
    st.plotly_chart(style_chart(fig, 300), use_container_width=True)

with reports_tab:
    st.subheader("Teacher-ready summary")
    st.markdown("Create a concise report for the selected date range and learner filter. The report contains descriptive measures and a clear non-diagnostic interpretation note.")
    st.download_button(
        "Download observation report (PDF)",
        data=make_pdf_report(filtered, start_date, end_date, selected_learner),
        file_name="classroom_observation_summary.pdf",
        mime="application/pdf",
    )
    st.caption("The PDF includes an overall summary and a per-learner table. Use pseudonymous IDs and follow school consent and retention practices.")

with data_tab:
    st.subheader("Filtered observations")
    st.caption("Use learner IDs only. Avoid adding names or other identifying details to the CSV.")
    st.dataframe(
        filtered.sort_values(["date", "learner_id"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
    )
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered observations (CSV)",
        data=csv_bytes,
        file_name="classroom_observations.csv",
        mime="text/csv",
    )
    summary = filtered.groupby("learner_id", as_index=False).agg(
        observations=("date", "count"),
        average_engagement_pct=("engagement_pct", "mean"),
        average_participation_pct=("participation_pct", "mean"),
        average_movement_events=("movement_events", "mean"),
        average_fidgeting_events=("fidgeting_events", "mean"),
        average_looking_away_events=("looking_away_events", "mean"),
    )
    st.download_button(
        "Download learner summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="learner_summary.csv",
        mime="text/csv",
    )

with privacy_tab:
    a, b = st.columns(2)
    with a:
        st.markdown("### Privacy first")
        st.markdown("This demo uses pseudonymous learner IDs. The app code does not save uploaded CSV files to disk or send them to an external analytics service. If deployed, the hosting environment still processes the upload, so follow your school’s consent, access, and retention rules.")
    with b:
        st.markdown("### Observation, not diagnosis")
        st.markdown("Movement, engagement, and attention-related measures can be affected by the activity, environment, and many other factors. The dashboard offers descriptive summaries only; it should not be used to label, rank, or diagnose a child.")
    st.markdown("### CSV format")
    st.markdown("Required columns: `date`, `learner_id`, `engagement_pct`, `movement_events`, `fidgeting_events`, `looking_away_events`, `participation_pct`, `observation_minutes`. Dates should be parseable (for example `2026-11-09`); percentage fields should be from 0 to 100. `activity_level` is optional and is derived from movement events per observation minute when omitted.")
    st.markdown("### Review rule used in this demo")
    st.markdown("A learner is marked **Review** in the register when at least three observations are present in the selected period and the mean engagement percentage across the most recent three observations is below 45%. This example threshold is illustrative and should be reviewed with educators before any real use.")
    st.markdown("### Project stack")
    st.markdown("Built with Python, Streamlit, pandas, and Plotly. The reference materials also mention OpenCV, YOLO, ByteTrack, MediaPipe Pose, and federated learning; those video and model pipelines are outside this dashboard demo and are not implemented here.")

st.markdown('<p class="page-note">Demo interface. Shows synthetic data until you upload a CSV.</p>', unsafe_allow_html=True)
