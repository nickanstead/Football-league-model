#!/usr/bin/env python3
"""
Rebuilds the points / goal-difference / previous-season-position logistic
regression model from real historical English top-flight results
(seanelvidge/England-football-results), fetches the current Premier League
table from football-data.org, and writes live-table.html with each club's
modelled chance of winning the league, finishing top 4, and being relegated.

Run by .github/workflows/update-predictions.yml on a schedule.
Requires env var FOOTBALL_DATA_TOKEN (a free key from football-data.org).
"""
import csv, re, json, math, os, subprocess, sys, tempfile
from datetime import datetime, timezone
from collections import defaultdict
import urllib.request
import numpy as np

SCENARIOS = {
    "title": lambda n: 1,
    "top4":  lambda n: 4 if n >= 4 else None,
    "releg": lambda n: (n - 3) if n >= 5 else None,
}

PLACEHOLDER_NORM = 1.15  # "worse than bottom of the table" - used for promoted/unmatched teams

# A team currently in the Premier League can be named differently by the live API
# (e.g. "Man City") than by the historical archive (e.g. "Manchester City"). This
# covers every club that has appeared in the Premier League plus recent
# Championship arrivals; anything not listed here safely falls back to the
# promoted-team placeholder rather than crashing or silently mismatching.
NAME_ALIASES = {
    "arsenal": "Arsenal", "aston villa": "Aston Villa", "bournemouth": "AFC Bournemouth",
    "afc bournemouth": "AFC Bournemouth", "brentford":
