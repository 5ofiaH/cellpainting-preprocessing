#!/usr/bin/env python
"""Run Levels 3 → 5 (+ metrics and plots) from an existing Level 2.5.

Thin wrapper so the pipeline runs from a fresh clone without installing:
    python run_pipeline.py --help
Equivalent to: python -m cp_preprocess.run_pipeline
"""
from cp_preprocess.run_pipeline import main

if __name__ == "__main__":
    main()
