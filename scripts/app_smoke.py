"""Headless smoke test of every Streamlit page (no browser).

Runs each page under `streamlit.testing.v1.AppTest`, reports how long its first load took, the
metrics it shows and any exception, and exits 1 if any page raised. Part of `.\\tasks.ps1 verify`.

    python scripts/app_smoke.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = (
    ("Home", "app/Home.py"),
    ("Intelligence", "app/pages/1_Intelligence.py"),
    ("Design", "app/pages/2_Design.py"),
    ("Choice", "app/pages/3_Choice.py"),
    ("Discovery", "app/pages/4_Discovery.py"),
    ("Implementation", "app/pages/5_Implementation.py"),
    ("DSS Architecture", "app/pages/6_DSS_Architecture.py"),
)


def _quiet_streamlit() -> None:
    """Silence Streamlit's "missing ScriptRunContext" warnings (no browser is attached)."""
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("streamlit"):
            logging.getLogger(name).setLevel(logging.ERROR)


def main() -> int:
    """Run every page once; print a table; return 1 if any page raised."""
    failed = 0
    sys.stdout.write(f"{'page':<18}{'first load (s)':>15}  {'metrics':>8}  status\n")
    for name, relative in PAGES:
        _quiet_streamlit()
        started = time.perf_counter()
        app = AppTest.from_file(str(ROOT / relative), default_timeout=180).run()
        seconds = time.perf_counter() - started
        problems = [str(e.value) for e in app.exception]
        failed += bool(problems)
        status = "OK" if not problems else "EXCEPTION " + "; ".join(problems)[:120]
        sys.stdout.write(f"{name:<18}{seconds:>15.2f}  {len(app.metric):>8}  {status}\n")
        for metric in app.metric:
            sys.stdout.write(f"    {metric.label}: {metric.value}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
