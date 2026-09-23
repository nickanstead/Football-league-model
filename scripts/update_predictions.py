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
    "afc bournemouth": "AFC Bournemouth", "brentford": "Brentford",
    "brighton": "Brighton & Hove Albion", "brighton hove albion": "Brighton & Hove Albion",
    "brighton and hove albion": "Brighton & Hove Albion", "burnley": "Burnley",
    "chelsea": "Chelsea", "crystal palace": "Crystal Palace", "everton": "Everton",
    "fulham": "Fulham", "leeds": "Leeds United", "leeds united": "Leeds United",
    "leicester": "Leicester City", "leicester city": "Leicester City",
    "liverpool": "Liverpool", "man city": "Manchester City", "manchester city": "Manchester City",
    "man united": "Manchester United", "man utd": "Manchester United",
    "manchester united": "Manchester United", "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United", "nottingham forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest", "norwich": "Norwich City", "norwich city": "Norwich City",
    "southampton": "Southampton", "sunderland": "Sunderland",
    "tottenham": "Tottenham Hotspur", "tottenham hotspur": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur", "watford": "Watford", "west brom": "West Bromwich Albion",
    "west bromwich albion": "West Bromwich Albion", "west ham": "West Ham United",
    "west ham united": "West Ham United", "wolves": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers", "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town", "luton": "Luton Town", "luton town": "Luton Town",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "hull": "Hull City", "hull city": "Hull City", "coventry": "Coventry City",
    "coventry city": "Coventry City",
}


def normalize_team_name(name):
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def match_archive_name(live_name):
    key = normalize_team_name(live_name)
    return NAME_ALIASES.get(key)


# ---------- Step 1: load the England archive and compute final standings per season ----------

def start_year(season):
    m = re.match(r"(\d{4})", season)
    return int(m.group(1)) if m else None


def load_seasons(repo_dir):
    by_season = defaultdict(list)
    csv_path = os.path.join(repo_dir, "EnglandLeagueResults.csv")
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("Tier") != "1":
                continue
            try:
                d = datetime.strptime(r["Date"].strip(), "%Y-%m-%d")
            except Exception:
                continue
            try:
                g1 = int(r["hGoal"]); g2 = int(r["aGoal"])
            except Exception:
                continue
            t1 = r["HomeTeam"].strip(); t2 = r["AwayTeam"].strip()
            season = r["Season"].strip()
            by_season[season].append((d, t1, t2, g1, g2))

    season_rows = {}
    for season, rows in by_season.items():
        # A season only enters training (or counts as anyone's "previous season") once
        # it's actually finished - checked relative to how many matches a complete
        # double round-robin needs for however many teams played that season, since a
        # genuinely complete season had far fewer teams (and matches) a century ago
        # than it does now. Without this, a currently in-progress season gets treated
        # as if its partial table were the real final one.
        teams = set()
        for (d, t1, t2, g1, g2) in rows:
            teams.add(t1); teams.add(t2)
        n_teams = len(teams)
        expected_matches = n_teams * (n_teams - 1)
        if n_teams < 6 or len(rows) < expected_matches * 0.95:
            continue
        rows.sort(key=lambda x: x[0])
        season_rows[season] = rows

    seasons_sorted = sorted(season_rows.keys(), key=start_year)
    return seasons_sorted, season_rows


def compute_final_standings(seasons_sorted, season_rows):
    final_rank_by_season = {}
    n_teams_by_season = {}
    match_count_by_season = {}
    for season in seasons_sorted:
        rows = season_rows[season]
        win_pts = 3 if start_year(season) >= 1981 else 2
        pts = defaultdict(int); gd = defaultdict(int)
        for (d, t1, t2, g1, g2) in rows:
            if g1 > g2: pts[t1] += win_pts
            elif g2 > g1: pts[t2] += win_pts
            else: pts[t1] += 1; pts[t2] += 1
            gd[t1] += g1 - g2; gd[t2] += g2 - g1
        order = sorted(pts.keys(), key=lambda t: (-pts[t], -gd[t]))
        final_rank_by_season[season] = {t: i + 1 for i, t in enumerate(order)}
        n_teams_by_season[season] = len(order)
        match_count_by_season[season] = len(rows)
    return final_rank_by_season, n_teams_by_season, match_count_by_season


def make_prev_norm_lookup(seasons_sorted, final_rank_by_season, n_teams_by_season):
    def prev_norm_rank(season_idx, team):
        if season_idx == 0:
            return PLACEHOLDER_NORM
        prev_season = seasons_sorted[season_idx - 1]
        if start_year(seasons_sorted[season_idx]) - start_year(prev_season) != 1:
            return PLACEHOLDER_NORM
        prev_ranks = final_rank_by_season[prev_season]
        n_prev = n_teams_by_season[prev_season]
        if team not in prev_ranks:
            return PLACEHOLDER_NORM
        return (prev_ranks[team] - 1) / max(1, (n_prev - 1))
    return prev_norm_rank


# ---------- Step 2: build row-level training observations ----------

MAX_K = 19  # covers every possible finishing position below 1st for a 20-team league

def build_training_rows(seasons_sorted, season_rows, final_rank_by_season, n_teams_by_season, prev_norm_rank):
    rows_by_K = {k: [] for k in range(1, MAX_K + 1)}
    for season_idx, season in enumerate(seasons_sorted):
        rows = season_rows[season]
        win_pts = 3 if start_year(season) >= 1981 else 2
        total_matches = len(rows)
        N = n_teams_by_season[season]
        if N < 6:
            continue
        final_rank = final_rank_by_season[season]

        points = defaultdict(int); gd = defaultdict(int); played = defaultdict(int)
        snapshot_every = max(1, total_matches // 20)
        prev_norm = {t: prev_norm_rank(season_idx, t) for t in final_rank}

        for idx, (d, t1, t2, g1, g2) in enumerate(rows):
            if g1 > g2: points[t1] += win_pts
            elif g2 > g1: points[t2] += win_pts
            else: points[t1] += 1; points[t2] += 1
            gd[t1] += g1 - g2; gd[t2] += g2 - g1
            played[t1] += 1; played[t2] += 1

            if idx % snapshot_every != 0:
                continue
            week_frac = (idx + 1) / total_matches
            teams_active = [t for t in points if played[t] >= 3]
            if len(teams_active) < 6:
                continue
            ranked = sorted(teams_active, key=lambda t: (-points[t], -gd[t]))

            for K in range(1, min(MAX_K, N - 1) + 1):
                if K > len(ranked):
                    continue
                boundary = ranked[K - 1]
                b_pts, b_gd, b_played = points[boundary], gd[boundary], played[boundary]
                for cand in ranked:
                    if cand == boundary:
                        continue
                    pgap = max(-20, min(20, points[cand] - b_pts))
                    ggap = max(-20, min(20, gd[cand] - b_gd))
                    cand_prev = prev_norm[cand]
                    games_diff = max(-6, min(6, b_played - played[cand]))  # + = candidate has games in hand
                    outcome = 1 if final_rank[cand] <= K else 0
                    rows_by_K[K].append((pgap, ggap, cand_prev, games_diff, week_frac, outcome))
    return rows_by_K


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def make_features(pgap, ggap, cand_prev, games_diff, wf):
    pg = pgap / 20.0; gg = ggap / 20.0; gih = games_diff / 6.0
    return np.column_stack([np.ones_like(pg), pg, wf, pg * wf, gg, gg * wf, cand_prev, cand_prev * wf, gih, gih * wf])


def fit_logreg(rows, epochs=3000, lr=0.3):
    arr = np.array(rows)
    X = make_features(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])
    y = arr[:, 5]
    b = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = sigmoid(X @ b)
        grad = X.T @ (p - y) / len(y)
        b -= lr * grad
    return b


def predict(b, pgap, ggap, cand_prev, games_diff, wf, games_left):
    # Hard mathematical ceiling based on points alone: a gap bigger than 3x games
    # remaining cannot be closed, whatever goal difference or history says. Note
    # games_left here should already include any games in hand (see call site) -
    # that's exactly how a game in hand can turn "impossible" into "still alive".
    if games_left <= 0:
        if pgap > 0: return 1.0
        if pgap < 0: return 0.0
        return 0.5
    max_swing = 3 * games_left
    if abs(pgap) > max_swing:
        return 1.0 if pgap > 0 else 0.0
    x = np.array([1.0, pgap / 20.0, wf, (pgap / 20.0) * wf, ggap / 20.0, (ggap / 20.0) * wf,
                  cand_prev, cand_prev * wf, games_diff / 6.0, (games_diff / 6.0) * wf])
    return float(sigmoid(x @ b))


# ---------- Step 3: fetch the live Premier League table ----------

def fetch_live_table(token):
    req = urllib.request.Request(
        "https://api.football-data.org/v4/competitions/PL/standings",
        headers={"X-Auth-Token": token},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    total_table = next(s["table"] for s in data["standings"] if s["type"] == "TOTAL")
    table = []
    for row in total_table:
        table.append({
            "position": row["position"],
            "name": row["team"].get("shortName") or row["team"]["name"],
            "played": row["playedGames"],
            "gd": row["goalDifference"],
            "points": row["points"],
        })
    return sorted(table, key=lambda r: r["position"])


# ---------- Step 4: attach each live team's previous-season position ----------

def attach_previous_season(table, seasons_sorted, season_rows, final_rank_by_season, n_teams_by_season, match_count_by_season):
    # seasons_sorted now only contains genuinely complete seasons (see load_seasons),
    # so the most recent entry is always a real finished season - never the one
    # currently in progress.
    if not seasons_sorted:
        for t in table:
            t["prev_norm"] = PLACEHOLDER_NORM
        return table
    prev_season = seasons_sorted[-1]
    prev_ranks = final_rank_by_season[prev_season]
    n_prev = n_teams_by_season[prev_season]
    print(f"Using {prev_season} as the previous season for position lookups", file=sys.stderr)

    unmatched = []
    for t in table:
        archive_name = match_archive_name(t["name"])
        if archive_name and archive_name in prev_ranks:
            t["prev_norm"] = (prev_ranks[archive_name] - 1) / max(1, (n_prev - 1))
        else:
            t["prev_norm"] = PLACEHOLDER_NORM
            unmatched.append(t["name"])
    if unmatched:
        print(f"No previous-season match for: {', '.join(unmatched)} (treated as promoted)", file=sys.stderr)
    return table


# ---------- Step 5: compute predictions and render the page ----------

def compute_predictions(table, models_by_K, season_len=38, conf=0.80):
    N = len(table)

    results = []
    for team in table:
        games_left = max(0, season_len - team["played"])
        week_frac = min(1.0, team["played"] / season_len)

        # Build the full CDF over final position: cdf[K] = P(finish at or above K).
        # Each K uses its own independently-fitted model and its own real boundary
        # team (whoever actually sits at position K in the live table right now).
        cdf = [0.0] * (N + 1)
        for K in range(1, N):
            if K > MAX_K:
                cdf[K] = cdf[K - 1]  # no model beyond MAX_K; carry forward rather than guess
                continue
            boundary = table[K - 1]
            gih = boundary["played"] - team["played"]
            p = predict(models_by_K[K], team["points"] - boundary["points"], team["gd"] - boundary["gd"],
                        team["prev_norm"], gih, week_frac, games_left)
            cdf[K] = p
        cdf[N] = 1.0
        # Independently-fit models can occasionally produce a tiny non-monotonic
        # step (P(<=5th) coming out below P(<=4th), which can't really happen) -
        # enforce it rather than trust every model in isolation.
        for K in range(1, N + 1):
            cdf[K] = max(cdf[K], cdf[K - 1])

        title_pct = cdf[1] * 100
        top4_pct = cdf[4] * 100 if N >= 4 else cdf[N] * 100
        safety_K = N - 3 if N >= 5 else N
        releg_pct = (1 - cdf[safety_K]) * 100

        lo_target, hi_target = (1 - conf) / 2, 1 - (1 - conf) / 2
        median = next(K for K in range(1, N + 1) if cdf[K] >= 0.5)
        range_lo = next(K for K in range(1, N + 1) if cdf[K] >= lo_target)
        range_hi = next(K for K in range(1, N + 1) if cdf[K] >= hi_target)

        results.append({**team, "title_pct": title_pct, "top4_pct": top4_pct, "releg_pct": releg_pct,
                         "median_pos": median, "range_lo": range_lo, "range_hi": range_hi})
    return results


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Premier League Predictions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0E1B2A; --surface:#16283D; --surface-2:#1C3350; --line:#2A415F;
    --text:#EDEFEF; --text-dim:#9FB1C4; --amber:#F2A93B; --teal:#4FA8A0; --red:#E0654F;
    font-family:'Inter',system-ui,sans-serif;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:var(--bg); color:var(--text); }}
  body {{ padding:24px 16px 40px; max-width:720px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .updated {{ font-size:12.5px; color:var(--text-dim); margin:0 0 20px; line-height:1.5; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ padding:9px 6px; text-align:right; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }}
  th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  th {{ color:var(--text-dim); font-weight:600; font-size:11.5px; text-transform:uppercase; letter-spacing:0.03em; }}
  td.title {{ color:var(--amber); font-weight:600; }}
  td.releg {{ color:var(--red); font-weight:600; }}
  .note {{ font-size:12.5px; color:var(--text-dim); line-height:1.6; margin-top:22px; }}
  a {{ color:var(--teal); }}
</style>
</head>
<body>
  <h1>Premier League — modelled title, top-4 &amp; relegation odds</h1>
  <p class="updated">Last updated {updated} UTC. Model: points, goal difference, previous-season finishing position &amp; games in hand, {n_obs:,} historical observations. Source: {source_label}.</p>
  <table>
    <thead><tr><th>#</th><th>Team</th><th>Pld</th><th>GD</th><th>Pts</th><th>Title</th><th>Top&nbsp;4</th><th>Releg.</th><th>Median</th><th>80%&nbsp;range</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <p class="note">
    Assumes a standard bottom-3 relegation zone. "Previous season" means each team's actual finishing position last time they were in the top flight;
    newly promoted or unrecognised teams are treated as if they finished below the bottom of the previous table. Percentages are a historical-comparison
    model, not a live bookmaker forecast — see the <a href="index.html">interactive version</a> for the full methodology and uncertainty ranges.
  </p>
  <p class="note">
    "Median" and "80% range" come from stitching together 19 separately-fitted models (one per possible finishing position) into a single probability
    curve over final position, then reading off the middle 80% of it. Unlike the title/top-4/relegation percentages, this hasn't been through the same
    held-out historical validation — treat it as a genuine extrapolation of the same method, not an equally-tested one. Positions with no real competitive
    stakes attached (mid-table finishes nobody is actually chasing) have thinner, noisier historical support than the title, Europe, or relegation cutoffs do.
  </p>
  <p class="note">
    Regenerated automatically on a schedule via GitHub Actions.
  </p>
</body>
</html>
"""

ROW_TEMPLATE = ("      <tr><td>{position}</td><td>{name}</td><td>{played}</td><td>{gd:+d}</td><td>{points}</td>"
                 "<td class=\"title\">{title_pct:.1f}%</td><td>{top4_pct:.1f}%</td><td class=\"releg\">{releg_pct:.1f}%</td>"
                 "<td>{median_pos}</td><td>{range_lo}\u2013{range_hi}</td></tr>")


def render_page(results, n_obs, source_label):
    rows = "\n".join(ROW_TEMPLATE.format(**r) for r in results)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return PAGE_TEMPLATE.format(rows=rows, updated=updated, n_obs=n_obs, source_label=source_label)


def main():
    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        print("FOOTBALL_DATA_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = os.path.join(tmp, "England-football-results")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/seanelvidge/England-football-results.git", repo_dir],
            check=True,
        )
        seasons_sorted, season_rows = load_seasons(repo_dir)
        final_rank_by_season, n_teams_by_season, match_count_by_season = compute_final_standings(
            seasons_sorted, season_rows
        )
        prev_norm_rank = make_prev_norm_lookup(seasons_sorted, final_rank_by_season, n_teams_by_season)
        rows_by_K = build_training_rows(
            seasons_sorted, season_rows, final_rank_by_season, n_teams_by_season, prev_norm_rank
        )
        models_by_K = {K: fit_logreg(rows_by_K[K]) for K in rows_by_K}
        n_obs = len(rows_by_K[1])

        table = fetch_live_table(token)
        table = attach_previous_season(
            table, seasons_sorted, season_rows, final_rank_by_season, n_teams_by_season, match_count_by_season
        )

    results = compute_predictions(table, models_by_K)
    html = render_page(
        results, n_obs,
        source_label="England, full top-flight history (1888–2024), seanelvidge/England-football-results",
    )

    with open("live-table.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote live-table.html", file=sys.stderr)


if __name__ == "__main__":
    main()
