"""Locate the native Java adapter without changing the author implementation."""
from pathlib import Path

DATASET_ADAPTER = Path(__file__).with_name('NativeBench.java')
