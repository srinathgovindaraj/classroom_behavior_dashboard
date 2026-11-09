# Classroom Signals dashboard

A Streamlit dashboard starter for the classroom observation project shown in the reference pages. It includes a synthetic demo dataset, date and learner filters, engagement summaries, activity charts, pseudonymous learner detail, CSV and PDF report exports, and a CSV upload workflow.

The dashboard is descriptive and non-diagnostic. It does not implement video capture, YOLO detection, ByteTrack, pose estimation, federated learning, or clinical classification. Those can be connected as separate data pipelines once their requirements and consent process are defined.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard.py
```

`dashboard.py` is the dashboard. `app.py` wraps it in `st.App` as an ASGI entrypoint for hosts such as Vercel; you can run that directly with `uvicorn app:app`.

The first launch shows synthetic demo data. Upload a CSV to use your own observations. The app code processes the uploaded data in the active Streamlit session and does not write it to disk; a hosted deployment still processes data on its server.

## CSV columns

Required: `date`, `learner_id`, `engagement_pct`, `movement_events`, `fidgeting_events`, `looking_away_events`, `participation_pct`, `observation_minutes`.

`activity_level` is optional. If omitted, the dashboard derives Low, Moderate, or High from movement events per observation minute. Use pseudonymous IDs, and do not upload names or identifying details.

The example review flag (three recent observations with average engagement below 45%) is illustrative only. Educators should set and interpret any review criteria in context.
