"""CLI: python scripts/run_ui.py"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    app = ROOT / "frontend" / "Overview.py"
    host = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app),
            "--server.port",
            os.getenv("STREAMLIT_SERVER_PORT", "8501"),
            "--server.address",
            host,
        ],
        check=True,
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()
