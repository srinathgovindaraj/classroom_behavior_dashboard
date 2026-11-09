"""ASGI entrypoint for hosts that expect a top-level `app`, such as Vercel.

The dashboard itself lives in dashboard.py.
Run locally with: streamlit run dashboard.py
"""

from pathlib import Path

import streamlit as st

app = st.App(str(Path(__file__).with_name("dashboard.py")))
