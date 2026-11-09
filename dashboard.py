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

CAPTURE_COLUMNS = {
    "timestamp",
    "learner_id",
    "engagement_pct",
    "movement_events",
    "fidgeting_events",
    "looking_away_events",
    "observation_minutes",
}

# Below this, the camera probably couldn't see a face, so the number says little about the child.
LOW_ENGAGEMENT_CHECK = 5

INK = "#1A1A1A"
ORANGE = "#E8501F"
LIGHT_ORANGE = "#F7B79E"
GRAY = "#C9C9C9"
PENCIL = "#6B6B6B"
RULE = "#E0E0E0"


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


def make_demo_timeline(data: pd.DataFrame) -> pd.DataFrame:
    """Create clearly synthetic per-minute measurements for the latest demo day."""
    rng = np.random.default_rng(41)
    latest = data["date"].max()
    rows: list[dict] = []
    for record in data[data["date"] == latest].itertuples(index=False):
        for minute in range(35):
            # A settled start, a restless middle, and a calmer finish.
            restless = np.exp(-(((minute - 20) / 7) ** 2))
            rows.append(
                {
                    "timestamp": pd.Timestamp(latest) + pd.Timedelta(hours=9, minutes=minute),
                    "date": latest,
                    "learner_id": record.learner_id,
                    "engagement_pct": float(np.clip(record.engagement_pct - 25 * restless + rng.normal(0, 6), 0, 100)),
                    "movement_events": int(rng.poisson(0.2 + 1.3 * restless)),
                    "fidgeting_events": int(rng.poisson(0.4 + 1.0 * restless)),
                    "looking_away_events": int(rng.poisson(0.2 + 0.8 * restless)),
                    "observation_minutes": 1.0,
                }
            )
    return pd.DataFrame(rows)


def with_activity_level(frame: pd.DataFrame) -> pd.DataFrame:
    rate = frame["movement_events"] / frame["observation_minutes"].replace(0, np.nan)
    frame["activity_level"] = np.select([rate < 0.34, rate < 0.68], ["Low", "Moderate"], default="High")
    frame.loc[rate.isna(), "activity_level"] = "Unrated"
    return frame


def load_capture_file(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """Read per-minute measurements from capture.py and summarize them by day."""
    missing = sorted(CAPTURE_COLUMNS - set(frame.columns))
    if missing:
        return None, None, "Missing required columns: " + ", ".join(missing)

    timeline = frame.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
    numeric_columns = sorted(CAPTURE_COLUMNS - {"timestamp", "learner_id"})
    for column in numeric_columns:
        timeline[column] = pd.to_numeric(timeline[column], errors="coerce")
    timeline["learner_id"] = timeline["learner_id"].astype("string").str.strip()
    timeline = timeline.dropna(subset=["timestamp", "learner_id", *numeric_columns])
    timeline = timeline[timeline["learner_id"] != ""]
    if timeline.empty:
        return None, None, "No usable rows found. Check that times, learner IDs, and numeric values are filled in."

    timeline["date"] = timeline["timestamp"].dt.date
    daily = timeline.groupby(["date", "learner_id"], as_index=False).agg(
        engagement_pct=("engagement_pct", "mean"),
        movement_events=("movement_events", "sum"),
        fidgeting_events=("fidgeting_events", "sum"),
        looking_away_events=("looking_away_events", "sum"),
        observation_minutes=("observation_minutes", "sum"),
    )
    daily["participation_pct"] = np.nan  # a camera can't measure participation
    return with_activity_level(daily).sort_values("date"), timeline.sort_values("timestamp"), None


def count_label(count: float, noun: str) -> str:
    return f"{count:.0f} {noun}" + ("" if round(count) == 1 else "s")


def time_window(start: pd.Timestamp, minutes: int) -> str:
    return f"{start:%H:%M} to {start + pd.Timedelta(minutes=minutes):%H:%M}"


def caregiver_summary(day: pd.DataFrame, buckets: pd.DataFrame, minutes: int, other_days: pd.DataFrame) -> list[str]:
    """Explain one learner's day in plain language, comparing them only with themselves."""
    learner = day["learner_id"].iloc[0]
    observed = day["observation_minutes"].sum()
    first, last = day["timestamp"].min(), day["timestamp"].max() + pd.Timedelta(minutes=1)
    average = day["engagement_pct"].mean()
    lines = [f"{learner} was observed for about {observed:.0f} minutes on {first:%d %b %Y}, from {first:%H:%M} to {last:%H:%M}."]

    high = buckets.loc[buckets["engagement_pct"].idxmax()]
    low = buckets.loc[buckets["engagement_pct"].idxmin()]
    if average < LOW_ENGAGEMENT_CHECK:
        lines.append("Estimated engagement can't be read reliably for this session. See the note above.")
    elif high["engagement_pct"] - low["engagement_pct"] < 1:
        lines.append(f"Estimated engagement averaged **{average:.0f}%** and stayed about the same throughout.")
    else:
        lines.append(
            f"Estimated engagement averaged **{average:.0f}%**."
            f" It was highest from {time_window(high['timestamp'], minutes)} ({high['engagement_pct']:.0f}%)"
            f" and lowest from {time_window(low['timestamp'], minutes)} ({low['engagement_pct']:.0f}%)."
        )

    busiest = buckets.loc[buckets["movement_events"].idxmax()]
    if busiest["movement_events"] > 0:
        lines.append(f"Most movement was from {time_window(busiest['timestamp'], minutes)}, with {count_label(busiest['movement_events'], 'movement event')}.")
    else:
        lines.append("No large movements were recorded.")

    away = buckets.loc[buckets["looking_away_events"].idxmax()]
    if away["looking_away_events"] > 0:
        lines.append(f"Looking away happened most from {time_window(away['timestamp'], minutes)}, with {count_label(away['looking_away_events'], 'event')}.")

    if other_days["date"].nunique() >= 2 and average >= LOW_ENGAGEMENT_CHECK:
        usual = other_days.groupby("date")["engagement_pct"].mean().mean()
        difference = average - usual
        if abs(difference) < 5:
            lines.append(f"This is close to their own average of {usual:.0f}% on other observed days.")
        else:
            direction = "above" if difference > 0 else "below"
            lines.append(f"This is about {abs(difference):.0f} points {direction} their own average of {usual:.0f}% on other observed days.")

    if busiest["movement_events"] > 0:
        lines.append(f"A useful question for the teacher: what was the class doing from {time_window(busiest['timestamp'], minutes)}?")
    return lines


def load_uploaded_data(upload) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:  # show a helpful message for malformed uploads
        return None, None, f"Could not read this CSV: {exc}"

    if "timestamp" in frame.columns:
        return load_capture_file(frame)

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        return None, None, "Missing required columns: " + ", ".join(missing)

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
        return None, None, "No usable rows found. Check that dates, learner IDs, and numeric values are filled in."

    if "activity_level" not in frame.columns:
        frame = with_activity_level(frame)
    return frame.sort_values("date"), None, None


def style_chart(fig, height: int = 310):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=18, b=8),
        dragmode=False,  # drag-to-zoom captures touches and stops the page scrolling on phones
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", color=PENCIL, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor=RULE),
        yaxis=dict(gridcolor=RULE, griddash="dash", zeroline=False),
        barcornerradius=6,
    )
    return fig


def hatch_bars(fig, name: str):
    """Give one bar series diagonal orange stripes, so it reads without relying on color alone."""
    fig.for_each_trace(
        lambda trace: trace.update(marker_pattern=dict(shape="/", fgcolor=ORANGE, bgcolor=LIGHT_ORANGE, solidity=0.45))
        if trace.name.startswith(name)
        else None
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
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F5F5")),
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
      /* The page: white rounded panels floating on a light gray canvas. */
      .block-container {
        max-width: 1240px;
        /* Streamlit sizes this to one screen; it must grow with the content. */
        height: auto !important;
        flex: 1 0 auto;
        min-height: 100dvh;
        padding: 3.5rem 2rem 2.5rem;
      }
      [data-testid="stHeader"] { background: #EFEFEF; }
      [data-testid="stMain"] h1 { font-size: 2.5rem; font-weight: 500; letter-spacing: -.02em; line-height: 1.15; padding: 0; }
      [data-testid="stMain"] :is(h2, h3) { font-size: 1.15rem; font-weight: 600; letter-spacing: -.01em; padding: .5rem 0 .25rem; }
      [data-testid="stSidebar"] h2 { font-size: 1.25rem; font-weight: 600; letter-spacing: -.01em; }
      [data-testid="stSidebar"] { border: 0; border-radius: 0 24px 24px 0; }
      [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: #FFF1EA; }
      [data-testid="stAlertContentInfo"] * { color: #9A3412; }
      [data-testid="stMain"] p.scope { color: #6B6B6B; font-size: 1rem; margin: 0 0 1rem; }
      [data-testid="stMain"] p.page-note { color: #6B6B6B; font-size: .875rem; line-height: 1.55; max-width: 70ch; margin: .75rem 0 1.5rem; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #6B6B6B !important; opacity: 1 !important; }
      [data-testid="stMain"] code { color: #1A1A1A; background: #F2F2F2; }
      /* Both dates need the room the calendar icon takes; clicking the field still opens the calendar. */
      [data-testid="stSidebar"] [data-testid="stDateInput"] input { font-size: .875rem; padding-right: 0; }
      [data-testid="stSidebar"] [data-testid="stDateInput"] svg { display: none; }
      .register .narrow { display: none; }
      :focus-visible { outline: 2px solid #E8501F; outline-offset: 2px; }

      /* Totals: white tiles, with one orange tile leading the row. */
      [data-testid="stMetric"] { background: #FFFFFF; border-radius: 18px; padding: 1rem 1.15rem; }
      [data-testid="stMetricLabel"] { color: #6B6B6B; }
      [data-testid="stMetricValue"] { color: #1A1A1A; font-size: 2rem; font-weight: 600; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
      .st-key-kpis_overview [data-testid="stColumn"]:first-child [data-testid="stMetric"] { background: linear-gradient(160deg, #D9480F, #EF6A33); }
      .st-key-kpis_overview [data-testid="stColumn"]:first-child [data-testid="stMetric"] * { color: #FFFFFF; }
      [role="tabpanel"] [data-testid="stMetric"] { background: #F5F5F5; }

      /* The menu: a white pill bar with a black pill on the open tab. */
      [data-testid="stTabs"] [role="tablist"] { background: #FFFFFF; border-radius: 999px; padding: .35rem; gap: .25rem; width: fit-content; max-width: 100%; }
      [data-testid="stTabs"] [role="tab"] { border-radius: 999px; padding: .5rem 1.1rem; height: auto; }
      [data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: #1A1A1A; }
      [data-testid="stTabs"] [role="tab"][aria-selected="true"] * { color: #FFFFFF; }
      [data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
      /* No focus ring after a click or tap; keyboard users still get a clear neutral one. */
      [data-testid="stTabs"] [role="tab"]:focus, [data-testid="stTabs"] [role="tab"]:focus-visible { outline: none; box-shadow: none; }
      [data-testid="stTabs"] [role="tab"][data-focus-visible] { outline: 2px solid #1A1A1A; outline-offset: 2px; }
      @media (hover: hover) and (pointer: fine) {
        [data-testid="stTabs"] [role="tab"][aria-selected="false"]:hover { background: #F2F2F2; }
        [data-testid="stTabs"] [role="tab"][aria-selected="false"]:hover * { color: #1A1A1A; }
      }

      /* Each tab opens as one large white panel; charts sit on a gray inset inside it. */
      [role="tabpanel"] { background: #FFFFFF; border-radius: 24px; padding: 1.5rem; margin-top: .75rem; }
      [role="tabpanel"] [data-testid="stPlotlyChart"] { background: #F7F7F7; border-radius: 16px; padding: .5rem; }

      /* The register: every learner on its own row, in ID order. */
      [data-testid="stMain"] table.register { display: table; width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #E6E6E6; border-radius: 16px; margin: 0; font-variant-numeric: tabular-nums; }
      [data-testid="stMain"] table.register tr { border: 0; background: transparent; }
      [data-testid="stMain"] table.register :is(th, td) { border: 0; border-bottom: 1px solid #EFEFEF; padding: 0 .9rem; height: 2.75rem; background: transparent; color: #1A1A1A; text-align: left; vertical-align: middle; font-size: .95rem; white-space: nowrap; }
      [data-testid="stMain"] table.register tbody tr:last-child :is(th, td) { border-bottom: 0; }
      [data-testid="stMain"] table.register thead th { background: #F5F5F5; color: #6B6B6B; font-weight: 500; font-size: .85rem; }
      [data-testid="stMain"] table.register thead th:first-child { border-top-left-radius: 15px; }
      [data-testid="stMain"] table.register thead th:last-child { border-top-right-radius: 15px; }
      [data-testid="stMain"] table.register tbody th { font-weight: 500; }
      [data-testid="stMain"] table.register .num { text-align: right; }
      [data-testid="stMain"] table.register td.trend { width: 40%; color: #E8501F; }
      .register svg { display: block; width: 100%; max-width: 280px; height: 24px; }
      .register mark { background: #FFE9B0; color: #1A1A1A; border-radius: 999px; padding: .15rem .55rem; margin-left: -.55rem; }
      .register .flag { display: inline-block; margin-left: .5rem; font-size: .8rem; font-weight: 400; }

      .stDownloadButton button { background: #1A1A1A; color: white; border: 0; border-radius: 999px; font-weight: 600; padding-inline: 1.25rem; }
      .stDownloadButton button p { color: white; }

      /* Touch devices: no tap flash, larger targets, no iOS zoom on input focus. */
      html { -webkit-tap-highlight-color: transparent; }
      button, [role="tab"] { touch-action: manipulation; }
      .stDownloadButton button:active { background: #3A3A3A; }
      @media (hover: none) {
        .modebar-container { display: none !important; }
      }
      @media (pointer: coarse) {
        input, textarea, select { font-size: 16px !important; }
        [role="tab"], .stDownloadButton button { min-height: 44px; }
      }

      /* Phone layout */
      @media (max-width: 640px) {
        .block-container { padding: 3.25rem .75rem 2rem; }
        [data-testid="stMetric"] { padding: .75rem .9rem; border-radius: 16px; }
        [data-testid="stMain"] h1 { font-size: 2rem; }
        [data-testid="stMain"] :is(h2, h3) { font-size: 1.1rem; }
        [data-testid="stMain"] p.scope { font-size: .9rem; }
        [data-testid="stTabs"] [role="tablist"] { gap: 0; padding: .25rem; }
        [data-testid="stTabs"] [role="tab"] { padding: .4rem .5rem; }
        [data-testid="stTabs"] [role="tab"] p { font-size: .8rem; }
        [role="tabpanel"] { padding: .875rem .75rem; border-radius: 20px; }
        [role="tabpanel"] [data-testid="stPlotlyChart"] { padding: .25rem; }
        [data-testid="stMain"] table.register thead th:nth-child(4) { border-top-right-radius: 15px; }
        [class*="st-key-kpis"] [data-testid="stHorizontalBlock"] { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
        [class*="st-key-kpis"] [data-testid="stColumn"] { width: auto; min-width: 0; }
        [class*="st-key-kpis"] [data-testid="stColumn"]:last-child:nth-child(odd) { grid-column: 1 / -1; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important; }
        [data-testid="stMain"] table.register :is(th, td) { padding: 0 .3rem; font-size: .875rem; }
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
        .register mark { display: inline-block; max-width: 100%; margin-left: 0; padding: .1rem .25rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; vertical-align: bottom; }
        .register .flag { display: block; margin-left: 0; margin-top: .15rem; font-size: .7rem; line-height: 1; }
        [data-testid="stElementContainer"]:has(.stDownloadButton), .stDownloadButton, .stDownloadButton button { width: 100% !important; }
      }
      /* Smallest phones: keep the column headings on one line. */
      @media (max-width: 360px) {
        [role="tabpanel"] { padding: .75rem .5rem; }
        [data-testid="stMain"] table.register thead th { font-size: .66rem; }
        [data-testid="stMain"] table.register thead th:nth-child(1) { width: 21%; }
        [data-testid="stMain"] table.register thead th:nth-child(2) { width: 28%; }
        [data-testid="stMain"] table.register thead th:nth-child(3) { width: 30%; }
        [data-testid="stMain"] h1 { font-size: 1.85rem; }
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
        timeline = make_demo_timeline(data)
        st.info("Showing synthetic demo data. No real learner information is included.")
    else:
        data, timeline, error = load_uploaded_data(uploaded)
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
hero = st.container(key="hero")
hero.title("Class register")
hero.markdown(f'<p class="scope">{start_date:%d %b %Y} to {end_date:%d %b %Y}, {scope}</p>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("No observations match this learner and period. Change the filters in the side panel.")
    st.stop()

avg_engagement = filtered["engagement_pct"].mean()
avg_participation = filtered["participation_pct"].mean()
learner_count = filtered["learner_id"].nunique()
observation_count = len(filtered)

metric_columns = hero.container(key="kpis_overview").columns(4)
metric_columns[0].metric("Engagement", pct(avg_engagement))
metric_columns[1].metric("Participation", pct(avg_participation))
metric_columns[2].metric("Observations", f"{observation_count:,}")
metric_columns[3].metric("Learners", f"{learner_count:,}")

overview_tab, timeline_tab, learner_tab, reports_tab, data_tab, privacy_tab = st.tabs(
    ["Register", "Timeline", "Learner", "Reports", "Data", "About"]
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
    # Near-zero engagement usually means the camera couldn't see a face, so it isn't marked for review.
    review = per_learner[
        (per_learner["Recent_observations"] >= 3)
        & (per_learner["Recent_engagement"] < 45)
        & (per_learner["Recent_engagement"] >= LOW_ENGAGEMENT_CHECK)
    ]
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
        trend_long = trend.melt(id_vars="date", var_name="Measure", value_name="Percent").dropna(subset=["Percent"])
        fig = px.line(
            trend_long,
            x="date",
            y="Percent",
            color="Measure",
            markers=True,
            color_discrete_map={"Engagement": ORANGE, "Participation": INK},
            labels={"date": "Date", "Percent": "Average", "Measure": ""},
        )
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
    with right:
        st.subheader("Observed activity mix")
        activity = filtered["activity_level"].value_counts().rename_axis("Level").reset_index(name="Observations")
        shares = activity["Observations"] / activity["Observations"].sum()
        activity["Label"] = activity["Level"].astype(str) + " " + shares.map(pct_share)
        level_colors = {"Low": GRAY, "Moderate": ORANGE, "High": INK, "Unrated": "#E6E6E6"}
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
        hatch_bars(fig, "Moderate")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(bargap=0.25)
        st.plotly_chart(fig, use_container_width=True)

with timeline_tab:
    st.subheader("Behavior through the day")
    scoped = None
    if timeline is None:
        st.info("This file has daily totals only. To see behavior through the day, upload a file made by capture.py.")
    else:
        scoped = timeline[(timeline["date"] >= start_date) & (timeline["date"] <= end_date)]
        if selected_learner != "All learners":
            scoped = scoped[scoped["learner_id"].astype(str) == selected_learner]
        if scoped.empty:
            st.info("No camera measurements match this learner and period. Change the filters in the side panel.")
            scoped = None

    if scoped is not None:
        if uploaded is None:
            st.caption("Demo timeline made from synthetic data, not from a camera.")
        learner_col, day_col, group_col = st.columns(3)
        if selected_learner == "All learners":
            timeline_learner = learner_col.selectbox("Learner", sorted(scoped["learner_id"].astype(str).unique()), key="timeline_learner")
        else:
            timeline_learner = selected_learner
            learner_col.markdown(f"**Learner**  \n{timeline_learner}")
        one_learner = scoped[scoped["learner_id"].astype(str) == timeline_learner]
        days = sorted(one_learner["date"].unique(), reverse=True)
        chosen_day = day_col.selectbox("Day", days, format_func=lambda day: f"{day:%d %b %Y}", key="timeline_day")
        minutes = group_col.selectbox("Group by", [5, 10, 15], format_func=lambda value: f"Every {value} minutes", key="timeline_group")

        day_rows = one_learner[one_learner["date"] == chosen_day]
        buckets = (
            day_rows.assign(timestamp=day_rows["timestamp"].dt.floor(f"{minutes}min"))
            .groupby("timestamp", as_index=False)
            .agg(
                engagement_pct=("engagement_pct", "mean"),
                movement_events=("movement_events", "sum"),
                fidgeting_events=("fidgeting_events", "sum"),
                looking_away_events=("looking_away_events", "sum"),
            )
        )

        t1, t2, t3, t4 = st.container(key="kpis_timeline").columns(4)
        t1.metric("Engagement", pct(day_rows["engagement_pct"].mean()))
        t2.metric("Movement", f"{day_rows['movement_events'].sum():.0f}")
        t3.metric("Fidgeting", f"{day_rows['fidgeting_events'].sum():.0f}")
        t4.metric("Looking away", f"{day_rows['looking_away_events'].sum():.0f}")

        if day_rows["engagement_pct"].mean() < LOW_ENGAGEMENT_CHECK:
            st.warning(
                "The camera rarely saw a forward-facing face in this session. This usually means the camera position, distance, "
                "or lighting needs fixing, or the seat was empty. Please don't read this as low engagement."
            )

        st.subheader("What this shows")
        other_days = one_learner[one_learner["date"] != chosen_day]
        st.markdown("\n".join(f"- {line}" for line in caregiver_summary(day_rows, buckets, minutes, other_days)))
        st.markdown(
            '<p class="page-note">These are rough camera estimates to support a conversation, not a diagnosis or a judgment. '
            "A high count is not automatically a problem: moving can help a child focus, and looking away can mean thinking, reading, or talking with a classmate.</p>",
            unsafe_allow_html=True,
        )

        st.subheader("Estimated engagement")
        fig = px.line(buckets, x="timestamp", y="engagement_pct", markers=True, color_discrete_sequence=[ORANGE], labels={"timestamp": "Time", "engagement_pct": "Estimated engagement"})
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        fig.update_xaxes(tickformat="%H:%M")
        st.plotly_chart(style_chart(fig, 280), use_container_width=True)

        st.subheader("Events")
        events = buckets.melt(
            id_vars="timestamp",
            value_vars=["movement_events", "fidgeting_events", "looking_away_events"],
            var_name="Event",
            value_name="Count",
        )
        events["Event"] = events["Event"].map({"movement_events": "Movement", "fidgeting_events": "Fidgeting", "looking_away_events": "Looking away"})
        fig = px.bar(events, x="timestamp", y="Count", color="Event", color_discrete_sequence=[INK, ORANGE, GRAY], labels={"timestamp": "Time", "Event": ""})
        hatch_bars(fig, "Fidgeting")
        fig.update_xaxes(tickformat="%H:%M")
        st.plotly_chart(style_chart(fig, 300), use_container_width=True)

        with st.expander("How each measure is worked out"):
            st.markdown(
                "- **Estimated engagement:** the share of time a forward-facing face was visible in the seat area, without large movement. "
                "It can't tell whether a child is actually paying attention.\n"
                "- **Movement:** a burst of large motion in the seat area, such as standing up or leaning across.\n"
                "- **Fidgeting:** a burst of smaller motion in the seat area.\n"
                "- **Looking away:** no forward-facing face for at least 3 seconds. Reading, writing, or poor lighting can also cause this.\n"
                "- **Not measured:** participation, mood, learning, or any medical condition."
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
    c1, c2, c3 = st.container(key="kpis_learner").columns(3)
    c1.metric("Engagement", pct(learner_data["engagement_pct"].mean()))
    c2.metric("Participation", pct(learner_data["participation_pct"].mean()))
    c3.metric("Observations", f"{len(learner_data):,}")
    learner_long = learner_data.melt(
        id_vars="date", value_vars=["engagement_pct", "participation_pct"], var_name="Measure", value_name="Percent"
    )
    learner_long = learner_long.dropna(subset=["Percent"])
    learner_long["Measure"] = learner_long["Measure"].map(
        {"engagement_pct": "Engagement", "participation_pct": "Participation"}
    )
    fig = px.line(
        learner_long,
        x="date",
        y="Percent",
        color="Measure",
        markers=True,
        color_discrete_map={"Engagement": ORANGE, "Participation": INK},
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
    fig = px.bar(signals, x="date", y="Events", color="Observation", barmode="group", color_discrete_sequence=[INK, ORANGE, GRAY], labels={"date": "Date"})
    hatch_bars(fig, "Fidgeting")
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
    st.markdown("A learner is marked **Review** in the register when at least three observations are present in the selected period and the mean engagement percentage across the most recent three observations is below 45%. An average below 5% isn't marked, because it usually means the camera couldn't see a face. These example thresholds are illustrative and should be reviewed with educators before any real use.")
    st.markdown("### Project stack")
    st.markdown("Built with Python, Streamlit, pandas, Plotly, and OpenCV. The reference materials also mention YOLO, ByteTrack, MediaPipe Pose, and federated learning; those are not implemented here.")
    st.markdown("### Camera measurements")
    st.markdown("`capture.py` runs on a local computer and measures only the chosen seat area for one learner with caregiver consent and school approval. It saves per-minute numbers and never saves video, images, or faces. It has no face recognition and does not detect or diagnose ADHD or any other condition. Upload its CSV here to see the Timeline tab.")

if uploaded is None:
    st.markdown('<p class="page-note">Demo interface. Shows synthetic data until you upload a CSV.</p>', unsafe_allow_html=True)
