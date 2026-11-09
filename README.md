# Classroom Signals dashboard

A Streamlit dashboard starter for the classroom observation project shown in the reference pages. It includes a synthetic demo dataset, date and learner filters, engagement summaries, activity charts, pseudonymous learner detail, CSV and PDF report exports, and a CSV upload workflow.

The dashboard is descriptive and non-diagnostic. It does not implement YOLO detection, ByteTrack, pose estimation, federated learning, or clinical classification.

## Camera measurements

`capture.py` uses OpenCV to measure one learner's seat area from a camera or video file, and writes per-minute numbers that the dashboard's Timeline tab explains for caregivers.

Before using it with a real child, get written caregiver consent and school approval, tell the people in the room, and check the privacy law that applies to you. The camera's view can include classmates, so aim it to show as little of them as possible.

```bash
pip install -r requirements-capture.txt
python capture.py --source 0 --learner-id L-001 --consent-confirmed
```

Drag a box around the seat when the window opens. Press Ctrl+C to stop, or set `--minutes`. Results are added to `observations/L-001.csv`, which git ignores. Upload that file to the dashboard. Use `--source path/to/video.mp4 --start 2026-11-09T09:00` for a recorded video.

What it does and doesn't do:

- It saves numbers only, never video, images, or faces. Nothing outside the seat area is analyzed.
- It has no face recognition, so the seat area must contain only the consenting learner.
- It does not detect or diagnose ADHD or any other condition.
- It can't measure participation, so that column is blank for camera data.
- Its thresholds are starting values and haven't been validated in real classrooms. Compare results with what a teacher saw, and adjust `--fidgeting-motion` and `--movement-motion` for each camera position.

Run its tests with `python -m unittest discover tests`.

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
