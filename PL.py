#!/usr/bin/env python3
"""Premier League Win Prediction - Quick Start Script.

Usage:
    python PL.py

Or use the CLI:
    python -m premier_league.cli --train
"""

import sys
from pathlib import Path

# Add project root to path so premier_league package is discoverable
project_root = str(Path(__file__).parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from premier_league.cli import main

if __name__ == "__main__":
    main()