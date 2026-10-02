"""Benchmark runner script for SparDA research models."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dense import main as dense_main


def main():
    # Currently dispatches to dense baseline benchmark; will support --backend sparse/forecast in Phase 3/4
    dense_main()


if __name__ == "__main__":
    main()
