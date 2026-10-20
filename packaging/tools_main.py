"""Entry point of detour-tools.exe (console): version, self-test and password."""

import sys

from src.app.tools import main

if __name__ == "__main__":
    sys.exit(main())
