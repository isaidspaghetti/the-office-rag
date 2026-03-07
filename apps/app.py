from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run apps/app.py` by importing the repo-root entrypoint.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app as root_app  # noqa: E402

root_app.main()
