"""Put the parent calibration dir on sys.path so these viz/ scripts can import the core
modules (calibrate, cali_configs, calibration_eval, paper_viz). Import this first."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
