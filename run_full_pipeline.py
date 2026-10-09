#!/usr/bin/env python
"""Build Level 2.5 from the image database, then run Levels 3 → 5.

Thin wrapper so the pipeline runs from a fresh clone without installing:
    python run_full_pipeline.py --help
Equivalent to: python -m cp_preprocess.run_full_pipeline
"""
from cp_preprocess.run_full_pipeline import main

if __name__ == "__main__":
    main()
