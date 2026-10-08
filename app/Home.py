"""Landing page: what this is, the six pages, and one note about synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app._components import note, page_link_or_label, set_page

set_page("Home")

st.title("Which sellers should we back, and which are we overlooking?")
st.markdown(
    "This dashboard helps a marketplace decide, seller by seller, whether to keep, warn, "
    "suspend, feature or promote them. It also finds good sellers who are being overlooked, "
    "and explains every recommendation in plain words."
)

PAGES = (
    ("pages/1_Intelligence.py", "Intelligence", "What is happening in the marketplace."),
    ("pages/2_Design.py", "Design", "How likely an order is to arrive late, and why."),
    ("pages/3_Choice.py", "Choice", "Which established sellers are best overall."),
    ("pages/4_Discovery.py", "Discovery", "Good sellers who are being overlooked."),
    ("pages/5_Implementation.py", "Implementation",
     "What to do about one seller, and the reason."),
    ("pages/6_DSS_Architecture.py", "DSS Architecture", "How the system fits together."),
)
for target, label, blurb in PAGES:
    left, right = st.columns([1, 3])
    with left:
        page_link_or_label(target, label)
    right.markdown(blurb)

note("The clickstream and call data are synthetic: Olist has no real web or call-centre data.")
