#!/bin/bash
# Run from agentic/scalar_experiments/
set -euo pipefail
poetry run python scripts/aggregate_conclusions.py
