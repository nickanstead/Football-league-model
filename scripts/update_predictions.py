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


def predict(b, pgap, ggap, cand_prev, games_diff, wf, games_left, win_pts=3):
    # Hard mathematical ceiling based on points alone: a gap bigger than
    # (points-per-win x games remaining) cannot be closed, whatever goal difference
    # or history says. Pre-1981 seasons awarded 2 points for a win, not 3 - using a
    # hardcoded 3 here regardless of era quietly made the ceiling too loose for
    # those seasons (requiring a bigger gap than was really needed before
    # declaring something mathematically closed out). Note games_left here should
    # already include any games in hand (see call site) - that's exactly how a
    # game in hand can turn "impossible" into "still alive".
    if games_left <= 0:
        if pgap > 0: return 1.0
        if pgap < 0: return 0.0
        return 0.5
    max_swing = win_pts * games_left
    if abs(pgap) > max_swing:
        return 1.0 if pgap > 0 else 0.0
    # Clip the same way training does (see build_training_rows): without this, a
    # lopsided small old-era league can produce gaps far outside anything the
    # model was ever fitted on, and the raw linear score saturates the sigmoid to
    # a false-certain 0% or 100% rather than genuinely reflecting the evidence.
    pgap = max(-20, min(20, pgap))
    ggap = max(-20, min(20, ggap))
    games_diff = max(-6, min(6, games_diff))
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


def fetch_season_matches(token):
    """Every finished match of the current PL season so far, grouped by matchday
    (in chronological matchday order) so a trajectory can be replayed from them."""
    req = urllib.request.Request(
        "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED",
        headers={"X-Auth-Token": token},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    by_matchday = defaultdict(list)
    for m in data.get("matches", []):
        md = m.get("matchday")
        ft = m.get("score", {}).get("fullTime", {})
        hg, ag = ft.get("home"), ft.get("away")
        if md is None or hg is None or ag is None:
            continue  # unplayed, abandoned, or an oddity without a normal round number
        home = m["homeTeam"].get("shortName") or m["homeTeam"]["name"]
        away = m["awayTeam"].get("shortName") or m["awayTeam"]["name"]
        by_matchday[md].append((home, away, hg, ag))

    return sorted(by_matchday.items())  # [(matchday, [(home, away, hg, ag), ...]), ...]


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

def compute_predictions(table, models_by_K, season_len=38, conf=0.80, win_pts=3, n_releg_spots=None, normalize=True):
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
            if boundary is team:
                # Team IS the Kth-place team: comparing it to itself would always read as a
                # dead-even gap of zero, no matter how big its real cushion over the chasing
                # pack is (training data explicitly excludes this self-comparison case for
                # the same reason). Compare against whoever's immediately chasing from K+1
                # instead, so the real margin actually shows up.
                boundary = table[K]
            gih = boundary["played"] - team["played"]
            p = predict(models_by_K[K], team["points"] - boundary["points"], team["gd"] - boundary["gd"],
                        team["prev_norm"], gih, week_frac, games_left, win_pts=win_pts)
            cdf[K] = p
        cdf[N] = 1.0
        # Independently-fit models can occasionally produce a tiny non-monotonic
        # step (P(<=5th) coming out below P(<=4th), which can't really happen) -
        # enforce it rather than trust every model in isolation.
        for K in range(1, N + 1):
            cdf[K] = max(cdf[K], cdf[K - 1])

        title_pct = cdf[1] * 100
        top4_pct = cdf[4] * 100 if N >= 4 else cdf[N] * 100
        if n_releg_spots is not None:
            safety_K = max(0, N - n_releg_spots)
        else:
            safety_K = N - 3 if N >= 5 else N  # standard modern-PL assumption
        releg_pct = (1 - cdf[safety_K]) * 100

        lo_target, hi_target = (1 - conf) / 2, 1 - (1 - conf) / 2
        median = next(K for K in range(1, N + 1) if cdf[K] >= 0.5)
        range_lo = next(K for K in range(1, N + 1) if cdf[K] >= lo_target)
        range_hi = next(K for K in range(1, N + 1) if cdf[K] >= hi_target)

        results.append({**team, "title_pct": title_pct, "top4_pct": top4_pct, "releg_pct": releg_pct,
                         "median_pos": median, "range_lo": range_lo, "range_hi": range_hi})

    if not normalize:
        # Used when scanning history for genuine precedents (compute_historical_highlights):
        # we want the model's raw estimate for that moment, not the display-only rescale
        # below. That rescale is a single factor applied across the whole column, and on a
        # small/noisy historical snapshot where the raw column total is far from its target,
        # the same factor that sensibly tidies up a well-calibrated live table can instead
        # inflate a middling raw probability (e.g. a genuine 29%) all the way to the 100%
        # cap - manufacturing a false "impossible" precedent rather than finding a real one.
        return results

    # Each team's percentage comes from its own independent model (a separate yes/no
    # question - "will THIS team specifically finish 1st / top 4 / bottom 3" - rather
    # than one joint model that distributes a fixed 100% across every team). Nothing
    # ties them together, so they don't naturally sum to the number of real spots on
    # offer (100% for the title, 400% across four Europe spots, etc).
    #
    # Rescaling title_pct and top4_pct separately would be wrong: each team's own
    # title_pct <= top4_pct is only guaranteed because both come from the same
    # monotonic per-team CDF, and two different correction factors applied to two
    # numbers that used to be ordered can flip that order (a team "more likely to
    # win the league than finish top 4", which cannot really happen).
    #
    # So instead we normalize the non-negative CDF *slices* - "chance of exactly
    # 1st" and "chance of finishing 2nd-4th specifically" - and add the rescaled
    # slices back together. Since both slices are always >= 0, title_pct can never
    # end up above top4_pct again, whatever each slice's own correction factor is.
    n_top4_spots = min(4, N)
    n_releg_spots = max(0, N - safety_K)

    for r in results:
        r["_slice_title"] = r["title_pct"]
        r["_slice_2to4"] = max(0.0, r["top4_pct"] - r["title_pct"])

    def rescale_factor(key, target):
        total = sum(r[key] for r in results)
        return (target / total) if total > 0 else 1.0

    f_title = rescale_factor("_slice_title", 100.0)
    f_2to4 = rescale_factor("_slice_2to4", max(0.0, n_top4_spots - 1) * 100.0)
    f_releg = rescale_factor("releg_pct", n_releg_spots * 100.0)

    for r in results:
        r["title_pct"] = r["_slice_title"] * f_title
        r["top4_pct"] = min(100.0, r["title_pct"] + r["_slice_2to4"] * f_2to4)
        r["releg_pct"] = min(100.0, r["releg_pct"] * f_releg)
        del r["_slice_title"], r["_slice_2to4"]

    return results


def compute_relegated_by_season(seasons_sorted, season_rows):
    """How many teams were really relegated each season, and which ones - derived
    directly from the data (who played top-flight this season but not the next)
    rather than assumed from modern rules. This correctly varies across eras: 2
    down for most of the 20th century, 3 from 1973/74 on, 4 in one-off
    contraction seasons like 1994/95, and simply "not re-elected" before
    automatic relegation existed at all (1888-1898) - which this method still
    captures correctly, since it only asks "were they still in the top flight
    next year", not "were they relegated via today's specific mechanism"."""
    teams_by_season = {
        season: {t1 for (d, t1, t2, g1, g2) in rows} | {t2 for (d, t1, t2, g1, g2) in rows}
        for season, rows in season_rows.items()
    }
    relegated_by_season = {}
    for i, season in enumerate(seasons_sorted[:-1]):
        next_season = seasons_sorted[i + 1]
        if start_year(next_season) - start_year(season) != 1:
            continue  # a gap (wartime suspension etc.) - no real relegation relationship
        relegated_by_season[season] = teams_by_season[season] - teams_by_season[next_season]
    return relegated_by_season


# ---------- Step 5a-ii: find historical precedents at this matchday's equivalent point ----------

def compute_historical_highlights(seasons_sorted, season_rows, final_rank_by_season, n_teams_by_season,
                                   prev_norm_rank, models_by_K, current_matchday, season_len=38,
                                   min_start_year=1888):
    """For every historical season, replays matches up to the point proportionally
    equivalent to 'current_matchday' of a modern 38-game season (a 42-game season
    from an older, bigger-division era gets compared at the same % of its own
    season, not the same raw match count), re-runs the same model on that
    snapshot, and looks for the most striking precedents: the eventual champion
    who had the lowest title chance at this stage, the team given the best odds
    who then blew it, etc. Uses the real points-per-win (2 before 1981, 3 since)
    and the real number of relegation spots for that specific season (derived
    from who actually wasn't in the top flight the following year) rather than
    assuming today's rules applied throughout history."""
    target_frac = min(1.0, current_matchday / season_len)
    relegated_by_season = compute_relegated_by_season(seasons_sorted, season_rows)

    champion_at_stage = []       # (season, team, title_pct) - eventual champions
    non_champion_at_stage = []   # (season, team, title_pct) - everyone who did NOT win
    top4_at_stage = []           # (season, team, top4_pct) - eventual top-4 finishers
    non_top4_at_stage = []       # (season, team, top4_pct) - everyone who missed top 4
    survivor_at_stage = []       # (season, team, releg_pct) - teams that stayed up
    relegated_at_stage = []      # (season, team, releg_pct) - teams that actually went down

    for season_idx, season in enumerate(seasons_sorted):
        if start_year(season) < min_start_year:
            continue
        if season not in relegated_by_season:
            continue  # last season in the archive, or right before a wartime gap - no reliable "did they survive" answer
        rows = season_rows[season]
        N = n_teams_by_season[season]
        if N < 6:
            continue
        win_pts = 3 if start_year(season) >= 1981 else 2
        final_rank = final_rank_by_season[season]
        total_matches = len(rows)
        prev_norm = {t: prev_norm_rank(season_idx, t) for t in final_rank}

        points = defaultdict(int); gd = defaultdict(int); played = defaultdict(int)
        target_idx = None
        for idx, (d, t1, t2, g1, g2) in enumerate(rows):
            if g1 > g2: points[t1] += win_pts
            elif g2 > g1: points[t2] += win_pts
            else: points[t1] += 1; points[t2] += 1
            gd[t1] += g1 - g2; gd[t2] += g2 - g1
            played[t1] += 1; played[t2] += 1
            if (idx + 1) / total_matches >= target_frac:
                target_idx = idx
                break
        if target_idx is None:
            continue

        relegated_teams = relegated_by_season[season]
        n_releg_spots = len(relegated_teams)
        if n_releg_spots < 1 or n_releg_spots > 6:
            continue  # a restructuring year with an unusual number of relegation spots - not a fair comparison point

        ranked = sorted(points.keys(), key=lambda t: (-points[t], -gd[t]))
        snapshot = [{"name": t, "played": played[t], "gd": gd[t], "points": points[t],
                     "prev_norm": prev_norm.get(t, PLACEHOLDER_NORM)} for t in ranked]
        results = compute_predictions(snapshot, models_by_K, season_len=(N - 1) * 2,
                                       win_pts=win_pts, n_releg_spots=n_releg_spots, normalize=False)
        by_name = {r["name"]: r for r in results}

        champion = min(final_rank, key=lambda t: final_rank[t])
        top4_teams = {t for t, rk in final_rank.items() if rk <= 4}

        for t, r in by_name.items():
            (champion_at_stage if t == champion else non_champion_at_stage).append((season, t, r["title_pct"]))
            (top4_at_stage if t in top4_teams else non_top4_at_stage).append((season, t, r["top4_pct"]))
            (relegated_at_stage if t in relegated_teams else survivor_at_stage).append((season, t, r["releg_pct"]))

    def safe_min(records):
        return min(records, key=lambda x: x[2]) if records else None

    def safe_max(records):
        return max(records, key=lambda x: x[2]) if records else None

    return {
        "matchday": current_matchday,
        "lowest_title_champion": safe_min(champion_at_stage),
        "highest_title_bottler": safe_max(non_champion_at_stage),
        "lowest_top4_qualifier": safe_min(top4_at_stage),
        "highest_top4_miss": safe_max(non_top4_at_stage),
        "great_escape": safe_max(survivor_at_stage),
        "shock_relegation": safe_min(relegated_at_stage),
    }


def render_highlights_html(highlights):
    if not highlights:
        return ""
    md = highlights["matchday"]

    def line(key, template):
        rec = highlights.get(key)
        if not rec:
            return None
        season, team, pct = rec
        return template.format(team=team, season=season, pct=pct, md=md)

    lines = [
        line("lowest_title_champion",
             "The lowest title chance ever given to an eventual champion at this stage (matchday {md}) "
             "was just <em>{pct:.1f}%</em>, for <em>{team}</em> in {season} \u2014 and they still won it."),
        line("highest_title_bottler",
             "The highest title chance ever given to a team that did not go on to win the league "
             "was <em>{pct:.1f}%</em>, for <em>{team}</em> in {season}."),
        line("lowest_top4_qualifier",
             "The lowest top-4 chance ever given to a team that still finished in the top four was "
             "<em>{pct:.1f}%</em>, for <em>{team}</em> in {season}."),
        line("highest_top4_miss",
             "The highest top-4 chance ever given to a team that then missed out was <em>{pct:.1f}%</em>, "
             "for <em>{team}</em> in {season}."),
        line("great_escape",
             "The biggest great escape: <em>{team}</em> were given just a <em>{pct:.1f}%</em> chance of staying up "
             "at this stage in {season} \u2014 and survived anyway."),
        line("shock_relegation",
             "The biggest shock relegation: <em>{team}</em> had only a <em>{pct:.1f}%</em> relegation risk "
             "at this stage in {season}, and still went down."),
    ]
    lines = [l for l in lines if l]
    if not lines:
        return ""
    items = "\n".join(f"      <li>{l}</li>" for l in lines)
    return f"""  <div id="highlights">
    <strong>At matchday {md}, historically&hellip;</strong>
    <ul>
{items}
    </ul>
    <p class="sub">Based on every complete English top-flight season since 1888/89, compared at the equivalent % of the season played.</p>
  </div>"""


# ---------- Step 5b: replay this season's matches to build each team's trajectory ----------

def compute_trajectories(matchdays, prev_norm_by_team, models_by_K, season_len=38):
    """matchdays: [(matchday_num, [(home, away, hg, ag), ...]), ...] in chronological order.
    Returns {team_name: [{"matchday":, "played":, "title_pct":, "top4_pct":, "releg_pct":}, ...]}.
    Reuses compute_predictions on each matchday's snapshot standings, so the trajectory is
    computed with exactly the same model and the same self-comparison fix as the final table."""
    points = defaultdict(int)
    gd = defaultdict(int)
    played = defaultdict(int)
    trajectory = defaultdict(list)

    for md_num, matches in matchdays:
        for (home, away, hg, ag) in matches:
            if hg > ag:
                points[home] += 3
            elif ag > hg:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1
            gd[home] += hg - ag
            gd[away] += ag - hg
            played[home] += 1
            played[away] += 1

        ranked = sorted(points.keys(), key=lambda t: (-points[t], -gd[t]))
        snapshot = [
            {"name": t, "played": played[t], "gd": gd[t], "points": points[t],
             "prev_norm": prev_norm_by_team.get(t, PLACEHOLDER_NORM)}
            for t in ranked
        ]
        for r in compute_predictions(snapshot, models_by_K, season_len=season_len):
            trajectory[r["name"]].append({
                "matchday": md_num,
                "played": r["played"],
                "title_pct": round(r["title_pct"], 1),
                "top4_pct": round(r["top4_pct"], 1),
                "releg_pct": round(r["releg_pct"], 1),
            })
    return trajectory


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
  tbody tr {{ cursor:pointer; }}
  tbody tr:active, tbody tr.open {{ background:var(--surface-2); }}
  #trajectory {{ display:none; margin-top:18px; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px; }}
  #trajectory.visible {{ display:block; }}
  #trajectory h2 {{ font-size:15px; margin:0 0 2px; }}
  #trajectory .sub {{ font-size:12px; color:var(--text-dim); margin:0 0 12px; }}
  #trajectory .legend {{ display:flex; gap:14px; font-size:11.5px; color:var(--text-dim); margin-top:8px; flex-wrap:wrap; }}
  #trajectory .legend span {{ display:flex; align-items:center; gap:5px; }}
  #trajectory .swatch {{ width:9px; height:9px; border-radius:2px; display:inline-block; }}
  #trajectory .chart-wrap {{ position:relative; width:100%; height:220px; }}
  #trajectory .close-btn {{ float:right; background:none; border:none; color:var(--text-dim); font-size:13px; cursor:pointer; padding:2px 6px; }}
  #highlights {{ margin-top:20px; padding:14px 16px; background:var(--surface); border-left:4px solid var(--amber); border-radius:6px; }}
  #highlights strong {{ display:block; font-size:14.5px; color:var(--text); margin-bottom:8px; }}
  #highlights ul {{ margin:0; padding-left:18px; }}
  #highlights li {{ font-size:14.5px; font-weight:600; line-height:1.55; margin-bottom:6px; color:var(--text); }}
  #highlights li em {{ font-style:normal; color:var(--amber); }}
  #highlights .sub {{ margin-top:10px; font-size:11px; font-weight:400; color:var(--text-dim); }}
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

  <div id="trajectory">
    <button class="close-btn" onclick="closeTrajectory()">✕ close</button>
    <h2 id="traj-title">Team</h2>
    <p class="sub" id="traj-sub"></p>
    <div class="chart-wrap"><canvas id="traj-canvas"></canvas></div>
    <div class="legend">
      <span><span class="swatch" style="background:#F2A93B;"></span>Title</span>
      <span><span class="swatch" style="background:#4FA8A0;"></span>Top&nbsp;4</span>
      <span><span class="swatch" style="background:#E0654F;"></span>Relegation</span>
    </div>
  </div>

{highlights_html}

  <p class="note">
    Tap any team's row to see how their modelled odds have moved matchday-by-matchday this season so far.
  </p>
  <p class="note">
    Assumes a standard bottom-3 relegation zone. "Previous season" means each team's actual finishing position last time they were in the top flight;
    newly promoted or unrecognised teams are treated as if they finished below the bottom of the previous table. Percentages are a historical-comparison
    model, not a live bookmaker forecast — see the <a href="index.html">interactive version</a> for the full methodology and uncertainty ranges.
    Each team's raw percentage comes from its own independent model, so each column is rescaled to sum to the number of real spots on offer
    (100% for the title, 400% across four Europe spots, 300% across three relegation spots) rather than left to add up to whatever they happen to.
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

ROW_TEMPLATE = ("      <tr data-team=\"{name_attr}\"><td>{position}</td><td>{name}</td><td>{played}</td><td>{gd:+d}</td><td>{points}</td>"
                 "<td class=\"title\">{title_pct:.1f}%</td><td>{top4_pct:.1f}%</td><td class=\"releg\">{releg_pct:.1f}%</td>"
                 "<td>{median_pos}</td><td>{range_lo}\u2013{range_hi}</td></tr>")

TRAJECTORY_SCRIPT = """<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const TRAJECTORIES = {trajectories_json};
let trajChart = null;

function closeTrajectory() {{
  document.getElementById('trajectory').classList.remove('visible');
  document.querySelectorAll('tbody tr.open').forEach(r => r.classList.remove('open'));
}}

function showTrajectory(team, row) {{
  const data = TRAJECTORIES[team];
  const box = document.getElementById('trajectory');
  if (!data || !data.length) {{
    document.getElementById('traj-title').textContent = team;
    document.getElementById('traj-sub').textContent = "No match-by-match data yet this season.";
    if (trajChart) {{ trajChart.destroy(); trajChart = null; }}
    box.classList.add('visible');
    return;
  }}
  document.querySelectorAll('tbody tr.open').forEach(r => r.classList.remove('open'));
  if (row) row.classList.add('open');

  document.getElementById('traj-title').textContent = team;
  document.getElementById('traj-sub').textContent =
    'Modelled odds after each matchday this season (' + data.length + ' played so far)';

  const labels = data.map(d => 'MD' + d.matchday);
  const ctx = document.getElementById('traj-canvas').getContext('2d');
  if (trajChart) trajChart.destroy();
  trajChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [
        {{ label: 'Title', data: data.map(d => d.title_pct), borderColor: '#F2A93B', backgroundColor: '#F2A93B', borderWidth: 2, pointRadius: 2, tension: 0.2 }},
        {{ label: 'Top 4', data: data.map(d => d.top4_pct), borderColor: '#4FA8A0', backgroundColor: '#4FA8A0', borderWidth: 2, pointRadius: 2, tension: 0.2 }},
        {{ label: 'Relegation', data: data.map(d => d.releg_pct), borderColor: '#E0654F', backgroundColor: '#E0654F', borderWidth: 2, pointRadius: 2, tension: 0.2 }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#9FB1C4', font: {{ size: 10 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }}, grid: {{ color: '#2A415F' }} }},
        y: {{ min: 0, max: 100, ticks: {{ color: '#9FB1C4', font: {{ size: 11 }}, callback: v => v + '%' }}, grid: {{ color: '#2A415F' }} }}
      }}
    }}
  }});
  box.classList.add('visible');
  box.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}

document.querySelectorAll('tbody tr').forEach(row => {{
  row.addEventListener('click', () => showTrajectory(row.dataset.team, row));
}});
</script>"""


def render_page(results, n_obs, source_label, trajectories=None, highlights_html=""):
    rows = "\n".join(
        ROW_TEMPLATE.format(**r, name_attr=r["name"].replace('"', "&quot;").replace("&", "&amp;"))
        for r in results
    )
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    page = PAGE_TEMPLATE.format(rows=rows, updated=updated, n_obs=n_obs, source_label=source_label,
                                 highlights_html=highlights_html or "")
    trajectories_json = json.dumps(trajectories or {}, separators=(",", ":"))
    script = TRAJECTORY_SCRIPT.format(trajectories_json=trajectories_json)
    return page.replace("</body>", script + "\n</body>")


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

    prev_norm_by_team = {t["name"]: t["prev_norm"] for t in table}
    matchdays = []
    try:
        matchdays = fetch_season_matches(token)
        trajectories = compute_trajectories(matchdays, prev_norm_by_team, models_by_K)
    except Exception as e:
        # The trajectory feature is a bonus on top of the main table - if the matches
        # endpoint fails or rate-limits, still publish the table rather than crashing.
        print(f"Could not build trajectories: {e}", file=sys.stderr)
        trajectories = {}

    highlights_html = ""
    if matchdays:
        try:
            current_matchday = matchdays[-1][0]
            highlights = compute_historical_highlights(
                seasons_sorted, season_rows, final_rank_by_season, n_teams_by_season,
                prev_norm_rank, models_by_K, current_matchday,
            )
            highlights_html = render_highlights_html(highlights)
        except Exception as e:
            print(f"Could not build historical highlights: {e}", file=sys.stderr)

    html = render_page(
        results, n_obs,
        source_label="England, full top-flight history (1888–2024), seanelvidge/England-football-results",
        trajectories=trajectories,
        highlights_html=highlights_html,
    )

    with open("live-table.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote live-table.html", file=sys.stderr)


if __name__ == "__main__":
    main()
