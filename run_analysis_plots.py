#!/usr/bin/env python
"""Regenerate plots from an existing run's output folder.

Thin wrapper so the pipeline runs from a fresh clone without installing:
    python run_analysis_plots.py --help
Equivalent to: python -m cp_preprocess.run_analysis_plots
"""
from cp_preprocess.run_analysis_plots import main

if __name__ == "__main__":
    main()
