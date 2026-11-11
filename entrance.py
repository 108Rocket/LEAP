#!/usr/bin/env python3
"""Run guided SDD Geolife Hausdorff reproduction; Porto remains optional."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "01_LEAP"))

from leap.launch.wizard import main

if __name__ == "__main__":
    raise SystemExit(main())
