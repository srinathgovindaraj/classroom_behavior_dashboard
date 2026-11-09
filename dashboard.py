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
    initial_sidebar_state="expanded",
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

NAVY = "#152B45"
TEAL = "#168C83"
MUTED = "#728198"


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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#42536A", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor="#E7ECF1"),
        yaxis=dict(gridcolor="#EEF2F5", zeroline=False),
    )
    return fig


def pct(value) -> str:
    return f"{value:.0f}%" if pd.notna(value) else "—"


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
        textColor=colors.HexColor("#152B45"),
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
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3F2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#176F68")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#152B45")),
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
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#152B45")),
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
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #F5F7FA; color: #192B42; }
      [data-testid="stSidebar"] { background: #152B45; }
      [data-testid="stSidebar"] * { color: #F5F8FC; }
      [data-testid="stSidebar"] [data-baseweb="select"] * { color: #192B42; }
      [data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: #203B59; border-color: #4A6682; }
      .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1440px; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; }
      h1 { color: #152B45; font-size: 2.15rem; }
      h2, h3 { color: #203B59; }
      [data-testid="stMetric"] { background: white; border: 1px solid #E8EDF2; padding: 18px 20px; border-radius: 14px; box-shadow: 0 2px 8px rgba(21,43,69,.035); }
      [data-testid="stMetricLabel"] { color: #728198; font-weight: 600; }
      [data-testid="stMetricValue"] { color: #152B45; font-family: 'Manrope', sans-serif; }
      div[data-testid="stTabs"] button { font-weight: 600; }
      .eyebrow { text-transform: uppercase; color: #168C83; font-weight: 700; font-size: .75rem; letter-spacing: .12em; }
      .subtle { color: #728198; font-size: .96rem; }
      .info-card { background: white; border: 1px solid #E8EDF2; padding: 20px 22px; border-radius: 14px; height: 100%; }
      .info-card p { color: #62738A; margin-bottom: 0; }
      .signal-card { background: #F1F8F7; border: 1px solid #D5EBE7; border-radius: 12px; padding: 15px 17px; margin: 8px 0; }
      .signal-card strong { color: #176F68; }
      .stDownloadButton button { background: #168C83; color: white; border: 0; border-radius: 9px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📘 Classroom Signals")
    st.caption("Observation dashboard")
    st.markdown("---")
    st.markdown("### Data source")
    uploaded = st.file_uploader("Upload an observation CSV", type=["csv"], help="The dashboard expects the columns listed under Data & privacy.")
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
    st.markdown("---")
    st.caption("Use pseudonymous IDs. This tool supports observation and discussion; it does not diagnose or rank children.")

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

filtered = data[(data["date"] >= start_date) & (data["date"] <= end_date)].copy()
if selected_learner != "All learners":
    filtered = filtered[filtered["learner_id"].astype(str) == selected_learner]

st.markdown('<div class="eyebrow">Classroom observation</div>', unsafe_allow_html=True)
st.title("Behavior & engagement overview")
st.markdown('<p class="subtle">A supportive view of observed patterns over time, designed to help educators decide what to discuss or observe next.</p>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("There are no observations for the selected learner and date range. Adjust the filters to continue.")
    st.stop()

avg_engagement = filtered["engagement_pct"].mean()
avg_participation = filtered["participation_pct"].mean()
learner_count = filtered["learner_id"].nunique()
observation_count = len(filtered)

metric_columns = st.columns(4)
metric_columns[0].metric("Average engagement", pct(avg_engagement))
metric_columns[1].metric("Participation", pct(avg_participation))
metric_columns[2].metric("Observations", f"{observation_count:,}")
metric_columns[3].metric("Learners included", f"{learner_count:,}")

overview_tab, learner_tab, reports_tab, data_tab, privacy_tab = st.tabs(
    ["Overview", "Learner detail", "Reports", "Observation data", "Data & privacy"]
)

with overview_tab:
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
            color_discrete_map={"Engagement": TEAL, "Participation": "#E5A84B"},
            labels={"date": "Date", "Percent": "Average", "Measure": ""},
        )
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
    with right:
        st.subheader("Observed activity mix")
        activity = filtered["activity_level"].value_counts().rename_axis("Activity level").reset_index(name="Observations")
        order = [name for name in ["Low", "Moderate", "High", "Unrated"] if name in activity["Activity level"].values]
        fig = px.pie(
            activity,
            names="Activity level",
            values="Observations",
            hole=0.72,
            color="Activity level",
            category_orders={"Activity level": order},
            color_discrete_map={"Low": "#75B9AE", "Moderate": "#E8B65D", "High": "#D8806A", "Unrated": "#AAB5C2"},
        )
        fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="white", width=3)))
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08, x=0))
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)

    st.subheader("Patterns to review")
    st.caption("These simple flags summarize recent observations. They are prompts for context, never conclusions about a learner.")
    per_learner = (
        filtered.sort_values("date")
        .groupby("learner_id", group_keys=False)
        .tail(3)
        .groupby("learner_id")
        .agg(Recent_observations=("engagement_pct", "size"), Recent_engagement=("engagement_pct", "mean"), Recent_participation=("participation_pct", "mean"))
        .reset_index()
    )
    review = per_learner[(per_learner["Recent_observations"] >= 3) & (per_learner["Recent_engagement"] < 45)]
    if review.empty:
        st.success("No learner met the demo review rule in this selection. Keep using classroom context and teacher judgment.")
    else:
        cols = st.columns(min(3, len(review)))
        for index, row in enumerate(review.itertuples(index=False)):
            with cols[index % len(cols)]:
                st.markdown(
                    f'<div class="signal-card"><strong>{row.learner_id}</strong><br><span>Average engagement below 45% across the last {row.Recent_observations} observations in this selection.</span></div>',
                    unsafe_allow_html=True,
                )

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
    c1, c2, c3 = st.columns(3)
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
        color_discrete_map={"Engagement": TEAL, "Participation": "#E5A84B"},
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
    fig = px.bar(signals, x="date", y="Events", color="Observation", barmode="group", color_discrete_sequence=[TEAL, "#E5A84B", "#7296C2"])
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
    st.subheader("Data handling")
    a, b = st.columns(2)
    with a:
        st.markdown(
            '<div class="info-card"><h3>Privacy first</h3><p>This demo uses pseudonymous learner IDs. The app code does not save uploaded CSV files to disk or send them to an external analytics service. If deployed, the hosting environment still processes the upload, so follow your school’s consent, access, and retention rules.</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="info-card"><h3>Observation, not diagnosis</h3><p>Movement, engagement, and attention-related measures can be affected by the activity, environment, and many other factors. The dashboard offers descriptive summaries only; it should not be used to label, rank, or diagnose a child.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### CSV format")
    st.markdown("Required columns: `date`, `learner_id`, `engagement_pct`, `movement_events`, `fidgeting_events`, `looking_away_events`, `participation_pct`, `observation_minutes`. Dates should be parseable (for example `2026-11-09`); percentage fields should be from 0 to 100. `activity_level` is optional and is derived from movement events per observation minute when omitted.")
    st.markdown("### Review rule used in this demo")
    st.markdown("A learner appears under **Patterns to review** when at least three observations are present in the selected period and the mean engagement percentage across the most recent three observations is below 45%. This example threshold is illustrative and should be reviewed with educators before any real use.")
    st.markdown("### Project stack")
    st.markdown("Python · Streamlit · pandas · Plotly. The reference materials also mention OpenCV, YOLO, ByteTrack, MediaPipe Pose, and federated learning; those video and model pipelines are outside this dashboard demo and are not implemented here.")

st.markdown("<br><div style='color:#8A97A7;font-size:.8rem'>Classroom Signals · Demo interface · Synthetic data by default</div>", unsafe_allow_html=True)
