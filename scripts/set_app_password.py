"""
Set the password of the Ajuste section of the app. Only the hash is written (config/seguridad.json), never the text.

A password in a local app prevents accidental use; it is not real security: whoever has the files of the app can get around it.

Usage:
    python scripts/set_app_password.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.app.paths import default_paths
from src.app.tools import set_password

if __name__ == "__main__":
    set_password(default_paths())
