#!/usr/bin/env python3
"""
Rebuilds the points/goal-difference logistic regression model from real historical
top-flight results (footballcsv/cache.footballdata), fetches the current Premier
League table from football-data.org, and writes live-table.html with each club's
modelled chance of winning the league, finishing top 4, and being relegated.

Run by .github/workflows/update-predictions.yml on a schedule.
Requires env var FOOTBALL_DATA_TOKEN (a free key from football-data.org).
"""
import csv, glob, re, json, math, os, subprocess, sys, tempfile
from datetime import datetime, timezone
from collections import defaultdict
import urllib.request

SCENARIOS = {
    "title": lambda n: 1,
    "top4":  lambda n: 4 if n >= 4 else None,
    "releg": lambda n: (n - 3) if n >= 5 else None,
}

# ---------- Step 1: rebuild the historical model from footballcsv/cache.footballdata ----------

def process_season(rows, win_pts, bins_out):
    total_matches = len(rows)
    if total_matches < 40:
        return
    final_points = defaultdict(int)
    final_gd = defaultdict(int)
    for (d, t1, t2, g1, g2) in rows:
        if g1 > g2: final_points[t1] += win_pts
        elif g2 > g1: final_points[t2] += win_pts
        else:
            final_points[t1] += 1; final_points[t2] += 1
        final_gd[t1] += g1 - g2
        final_gd[t2] += g2 - g1

    N = len(final_points)
    if N < 6:
        return
    final_order = sorted(final_points.keys(), key=lambda t: (-final_points[t], -final_gd[t]))
    final_rank = {t: i + 1 for i, t in enumerate(final_order)}

    points = defaultdict(int)
    gd = defaultdict(int)
    played = defaultdict(int)
    snapshot_every = max(1, total_matches // 20)

    for idx, (d, t1, t2, g1, g2) in enumerate(rows):
        if g1 > g2: points[t1] += win_pts
        elif g2 > g1: points[t2] += win_pts
        else:
            points[t1] += 1; points[t2] += 1
        gd[t1] += g1 - g2
        gd[t2] += g2 - g1
        played[t1] += 1; played[t2] += 1

        if idx % snapshot_every != 0:
            continue
        week_frac = (idx + 1) / total_matches
        decile = min(9, int(week_frac * 10))

        teams_active = [t for t in points if played[t] >= 3]
        if len(teams_active) < 6:
            continue
        ranked = sorted(teams_active, key=lambda t: (-points[t], -gd[t]))

        for scen, kfn in SCENARIOS.items():
            K = kfn(N)
            if K is None or K < 1 or K > len(ranked):
                continue
            boundary_team = ranked[K - 1]
            boundary_pts = points[boundary_team]
            boundary_gd = gd[boundary_team]
            for cand in ranked:
                if cand == boundary_team:
                    continue
                raw_pgap = points[cand] - boundary_pts
                pgap = max(-20, min(20, int(round(raw_pgap / 2.0)) * 2))
                raw_ggap = gd[cand] - boundary_gd
                ggap = max(-20, min(20, int(round(raw_ggap / 4.0)) * 4))
                good = 1 if final_rank[cand] <= K else 0
                key = (pgap, ggap, decile)
                bins_out[scen][key][0] += good
                bins_out[scen][key][1] += 1


def parse_score_ft(ft):
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", ft or "")
    if not m: return None
    return int(m.group(1)), int(m.group(2))


def parse_date_dow(s):
    try: return datetime.strptime(s.strip(), "%a %b %d %Y")
    except Exception: return None


def build_england_source_bins(repo_dir):
    bins_out = {k: defaultdict(lambda: [0, 0]) for k in SCENARIOS}
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

    seasons_used = 0
    for season, rows in by_season.items():
        rows.sort(key=lambda x: x[0])
        m = re.match(r"(\d{4})", season)
        start_year = int(m.group(1)) if m else 2000
        # English football moved from 2 points for a win to 3 points from the 1981-82 season.
        win_pts = 3 if start_year >= 1981 else 2
        process_season(rows, win_pts, bins_out)
        seasons_used += 1
    print(f"Processed {seasons_used} England top-flight seasons (1888-present)", file=sys.stderr)
    return bins_out


def sigmoid(z):
    if z < -35: return 0.0
    if z > 35: return 1.0
    return 1 / (1 + math.exp(-z))


def fit_model(bins):
    b = [0.0] * 6
    lr = 0.5
    n_total = sum(v[1] for v in bins.values()) or 1
    rows = [(pg, gg, d, good, tot) for (pg, gg, d), (good, tot) in bins.items()]
    for _ in range(2200):
        grad = [0.0] * 6
        for pgap, ggap, decile, good, tot in rows:
            pg = pgap / 20.0; gg = ggap / 20.0; wf = (decile + 0.5) / 10.0
            x = [1.0, pg, wf, pg * wf, gg, gg * wf]
            z = sum(b[k] * x[k] for k in range(6))
            pred = sigmoid(z)
            err = pred * tot - good
            for k in range(6): grad[k] += err * x[k]
        for k in range(6): b[k] -= lr * grad[k] / n_total
    return b


def predict(b, pgap, ggap, wf, games_left):
    # Hard mathematical ceiling based on points alone: a gap bigger than 3x games
    # remaining cannot be closed, whatever goal difference says.
    if games_left <= 0:
        if pgap > 0: return 1.0
        if pgap < 0: return 0.0
        return 0.5
    max_swing = 3 * games_left
    if abs(pgap) > max_swing:
        return 1.0 if pgap > 0 else 0.0
    pg = pgap / 20.0; gg = ggap / 20.0
    z = b[0] + b[1] * pg + b[2] * wf + b[3] * pg * wf + b[4] * gg + b[5] * gg * wf
    return sigmoid(z)


# ---------- Step 2: fetch the live Premier League table ----------

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


# ---------- Step 3: compute predictions and render the page ----------

def compute_predictions(table, models, season_len=38):
    leader = table[0]
    fourth = table[3]
    safety = table[16]  # 17th place = last safe spot, assuming a standard 3-team drop zone

    results = []
    for team in table:
        games_left = max(0, season_len - team["played"])
        week_frac = min(1.0, team["played"] / season_len)

        p_title = predict(models["title"], team["points"] - leader["points"],
                           team["gd"] - leader["gd"], week_frac, games_left)
        p_top4 = predict(models["top4"], team["points"] - fourth["points"],
                          team["gd"] - fourth["gd"], week_frac, games_left)
        p_safe = predict(models["releg"], team["points"] - safety["points"],
                          team["gd"] - safety["gd"], week_frac, games_left)

        results.append({**team, "title_pct": p_title * 100, "top4_pct": p_top4 * 100,
                         "releg_pct": (1 - p_safe) * 100})
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
  .updated {{ font-size:12.5px; color:var(--text-dim); margin:0 0 20px; }}
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
  <p class="updated">Last updated {updated} UTC. Model: real historical points &amp; goal-difference gaps, {n_obs:,} observations. Source: {source_label}.</p>
  <table>
    <thead><tr><th>#</th><th>Team</th><th>Pld</th><th>GD</th><th>Pts</th><th>Title</th><th>Top&nbsp;4</th><th>Releg.</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <p class="note">
    Assumes a standard bottom-3 relegation zone. Percentages are a historical-comparison model, not a live bookmaker forecast — see the
    <a href="index.html">interactive version</a> for the full methodology, uncertainty ranges, and other data-source options.
    Regenerated automatically on a schedule via GitHub Actions.
  </p>
</body>
</html>
"""

ROW_TEMPLATE = "      <tr><td>{position}</td><td>{name}</td><td>{played}</td><td>{gd:+d}</td><td>{points}</td><td class=\"title\">{title_pct:.1f}%</td><td>{top4_pct:.1f}%</td><td class=\"releg\">{releg_pct:.1f}%</td></tr>"


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
        bins = build_england_source_bins(repo_dir)

    n_obs = sum(v[1] for v in bins["title"].values())
    models = {scen: fit_model(bins[scen]) for scen in SCENARIOS}

    table = fetch_live_table(token)
    results = compute_predictions(table, models)
    html = render_page(results, n_obs, source_label="England, full top-flight history (1888–2024), seanelvidge/England-football-results")

    with open("live-table.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote live-table.html", file=sys.stderr)


if __name__ == "__main__":
    main()
