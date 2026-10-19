#!/usr/bin/env python3
"""Repo-level entry point for the isolated reproduction pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'reproduction'))
from pipeline import cli

if __name__ == '__main__':
    raise SystemExit(cli())
