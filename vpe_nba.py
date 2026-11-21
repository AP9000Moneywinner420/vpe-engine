"""Verified Prop Engine - NBA, single-file in-chat edition (v1.1).
Data: free ESPN box scores + schedules that sportsdataverse publishes on GitHub (parquet files, no API key).

  python vpe_nba.py board  [--date 2026-10-20] [--team NY,PHI] [--player "A;B"] [--injuries "Name:Out[:TEAM]"] [--games games.csv]
  python vpe_nba.py slate  --src lines/points.csv [--date ...] [--top 30]     # who to pull DraftKings pages for
  python vpe_nba.py lines  --src pts=lines/points.csv --src fg3m=lines/threes.csv --out props.csv   # best-of-market rows
  python vpe_nba.py check  --props lines/dk.csv [--games games.csv] [--injuries "..."] [--bankroll 50] [--out bets.csv]
  python vpe_nba.py log    --placed placed.csv --log nba_pick_log.csv
  python vpe_nba.py grade  --log nba_pick_log.csv --date 2026-10-20 [--actuals actuals.csv]
  python vpe_nba.py backtest --start 2026-01-05 --end 2026-04-12 [--step 2] [--odds game_odds.csv]
  python vpe_nba.py styles [--team DEN,GS] [--stat pts]          # team categories, player archetypes, matchup table
"""
from scipy import stats as st
import io
import itertools
import math
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

SPORT = "NBA"
VERSION = "NBA v2.36.0"

# ==================== data ====================

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
URLS = {"box": BASE + "/espn_nba_player_boxscores/player_box_{s}.parquet",
        "sched": BASE + "/espn_nba_schedules/nba_schedule_{s}.parquet"}
CACHE = os.path.join(os.path.expanduser("~"), ".vpe_nba_cache")

def _download(url, path, max_age_hours):
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nba-prop-engine/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(path + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(path + ".tmp", path)
    except urllib.error.HTTPError as e:
        if os.path.exists(path):
            print(f"  ! download failed ({e}); using cached {os.path.basename(path)}")
        elif e.code == 404:
            raise FileNotFoundError(url)
        else:
            raise
    except Exception as e:
        if os.path.exists(path):
            print(f"  ! download failed ({e}); using cached {os.path.basename(path)}")
        else:
            raise
    return path

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "oreb", "dreb", "fga", "fta", "fg3a", "fgm", "pf"]
RAW = {"points": "pts", "rebounds": "reb", "assists": "ast", "three_point_field_goals_made": "fg3m", "steals": "stl",
       "blocks": "blk", "turnovers": "tov", "offensive_rebounds": "oreb", "defensive_rebounds": "dreb",
       "field_goals_attempted": "fga", "free_throws_attempted": "fta", "three_point_field_goals_attempted": "fg3a",
       "field_goals_made": "fgm", "fouls": "pf"}
POS = {"G": "G", "PG": "G", "SG": "G", "F": "F", "SF": "F", "PF": "F", "C": "C"}
NBA_TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM",
             "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"}
TEAM_ALIASES = {"GSW": "GS", "NYK": "NY", "SAS": "SA", "NOP": "NO", "NOR": "NO", "UTA": "UTAH", "WAS": "WSH",
                "PHO": "PHX", "BRK": "BKN", "BRO": "BKN", "CHO": "CHA", "CHH": "CHA", "LAK": "LAL"}
NICK = {"hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA", "bulls": "CHI", "cavaliers": "CLE",
        "cavs": "CLE", "mavericks": "DAL", "mavs": "DAL", "nuggets": "DEN", "pistons": "DET", "warriors": "GS",
        "rockets": "HOU", "pacers": "IND", "clippers": "LAC", "lakers": "LAL", "grizzlies": "MEM", "heat": "MIA",
        "bucks": "MIL", "timberwolves": "MIN", "wolves": "MIN", "pelicans": "NO", "knicks": "NY", "thunder": "OKC",
        "magic": "ORL", "76ers": "PHI", "sixers": "PHI", "suns": "PHX", "trail blazers": "POR", "blazers": "POR",
        "spurs": "SA", "kings": "SAC", "raptors": "TOR", "jazz": "UTAH", "wizards": "WSH"}

def norm_team(x):
    s = str(x).strip()
    u = s.upper()
    if u in NBA_TEAMS:
        return u
    if u in TEAM_ALIASES:
        return TEAM_ALIASES[u]
    low = s.lower()
    for k, v in NICK.items():
        if low == k or low.endswith(" " + k):
            return v
    return None

def parse_matchup(m):
    s = str(m).strip()
    if " @ " in s:
        a, b = s.split(" @ ", 1)
        return norm_team(a), norm_team(b), False
    parts = re.split(r"\s+vs\.?\s+", s, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return norm_team(parts[0]), norm_team(parts[1]), True
    return None, None, None

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")

def norm_name(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace(".", "").replace(",", " ").replace("'", "").replace("’", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

def season_for(date):
    d = pd.Timestamp(date)
    return d.year + 1 if d.month >= 8 else d.year

def today_et():
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()

def load_box(season, max_age_hours=6):
    """Regular-season player box scores for the season ending in `season` (2026 = 2025-26). Empty if not posted."""
    try:
        p = _download(URLS["box"].format(s=season), os.path.join(CACHE, f"box_{season}.parquet"), max_age_hours)
    except FileNotFoundError:
        return pd.DataFrame()
    b = pd.read_parquet(p)
    b = b[(b["season_type"] == 2) & b["team_abbreviation"].isin(NBA_TEAMS) & b["athlete_id"].notna()].copy()
    b = b.rename(columns=RAW)
    b["date"] = pd.to_datetime(b["game_date"]).dt.normalize()
    b["min"] = pd.to_numeric(b["minutes"], errors="coerce").fillna(0.0)
    b["pid"] = b["athlete_id"].astype("int64")
    b["name"] = b["athlete_display_name"]
    b["_key"] = b["name"].map(norm_name)
    b["team"] = b["team_abbreviation"]
    b["opp"] = b["opponent_team_abbreviation"]
    b["pos"] = b["athlete_position_abbreviation"].map(POS).fillna("F")
    b["played"] = (~b["did_not_play"].fillna(False).astype(bool)) & (b["min"] > 0)
    b["home"] = b["home_away"].eq("home")
    b["margin"] = pd.to_numeric(b["team_score"], errors="coerce") - pd.to_numeric(b["opponent_team_score"], errors="coerce")
    b["starter"] = b["starter"].fillna(False).astype(bool)
    for c in STATS:
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0)
    keep = ["game_id", "date", "pid", "name", "_key", "team", "opp", "pos", "home", "starter", "played", "min",
            "margin", "reason"] + STATS
    return b[keep].sort_values(["date", "game_id"]).reset_index(drop=True)

def load_schedule(season, max_age_hours=6):
    try:
        p = _download(URLS["sched"].format(s=season), os.path.join(CACHE, f"sched_{season}.parquet"), max_age_hours)
    except FileNotFoundError:
        return pd.DataFrame(columns=["game_id", "date", "tip", "home", "away"])
    s = pd.read_parquet(p)
    s = s[s["season_type"] == 2]
    return pd.DataFrame({"game_id": s["game_id"].astype(str), "date": pd.to_datetime(s["game_date"]).dt.normalize(),
                         "tip": pd.to_datetime(s["game_date_time"]).dt.tz_convert("America/New_York").dt.tz_localize(None),
                         "home": s["home_abbreviation"].map(norm_team), "away": s["away_abbreviation"].map(norm_team)})

# ==================== pricing ====================

def american_to_decimal(a):
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / abs(a))

def implied_prob(a):
    return 1 / american_to_decimal(a)

def decimal_to_american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))

def no_vig(over_odds, under_odds):
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    return po / (po + pu), pu / (po + pu)

def ev_per_unit(prob, american):
    return prob * (american_to_decimal(american) - 1) - (1 - prob)

def kelly_fraction(prob, american):
    b = american_to_decimal(american) - 1
    return max((b * prob - (1 - prob)) / b, 0.0)

def parse_odds(x):
    s = str(x).strip().lower()
    if s in ("even", "ev", "evens", "+100", "100"):
        return 100.0
    try:
        return float(s)
    except ValueError:
        return np.nan

# ==================== markets ====================

MARKETS = {"pts": ["pts"], "reb": ["reb"], "ast": ["ast"], "fg3m": ["fg3m"], "stl": ["stl"], "blk": ["blk"],
           "tov": ["tov"], "pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"],
           "ra": ["reb", "ast"], "stocks": ["stl", "blk"]}
# v2.30 (Adrian 11/18): moneylines, spreads and totals are props too. Rows in bets_all.csv: player = the team
# ("BOS") for spread / ml, "AWAY@HOME" for total; side "Over" on a spread / ml row means THAT TEAM covers / wins;
# line = the spread as shown (-4.5), 0 for a moneyline, the total for a total.
GAME_MK = {"spread", "ml", "total"}
CONT = {"pts", "pra", "pr", "pa"}  # normal approximation; the rest are counts (negative binomial)
_MK = {"points": "pts", "pts": "pts", "rebounds": "reb", "reb": "reb", "totalrebounds": "reb", "assists": "ast",
       "ast": "ast", "threes": "fg3m", "3pointers": "fg3m", "3pointersmade": "fg3m", "threepointersmade": "fg3m",
       "threepointers": "fg3m", "3pm": "fg3m", "3ptm": "fg3m", "fg3m": "fg3m", "3s": "fg3m", "madethrees": "fg3m",
       "steals": "stl", "stl": "stl", "blocks": "blk", "blk": "blk", "turnovers": "tov", "tov": "tov",
       "pra": "pra", "pts+reb+ast": "pra", "points+rebounds+assists": "pra", "pointsreboundsassists": "pra",
       "points+rebounds++assists": "pra", "pr": "pr", "pts+reb": "pr", "points+rebounds": "pr", "pointsrebounds": "pr",
       "pa": "pa", "pts+ast": "pa", "points+assists": "pa", "pointsassists": "pa", "ra": "ra", "reb+ast": "ra",
       "ast+reb": "ra", "rebounds+assists": "ra", "assists+rebounds": "ra", "reboundsassists": "ra",
       "stocks": "stocks", "stl+blk": "stocks", "blk+stl": "stocks", "steals+blocks": "stocks", "blocks+steals": "stocks"}

_MK.update({"spread": "spread", "pointspread": "spread", "handicap": "spread", "ats": "spread", "ml": "ml",
            "moneyline": "ml", "total": "total", "gametotal": "total", "totalpoints": "total", "overunder": "total",
            "over/under": "total", "o/u": "total"})

def canon_market(m):
    s = str(m).strip().lower().replace(" and ", "+").replace("&", "+")
    s = re.sub(r"[\s_,\-]", "", s)
    return _MK.get(s)

# ==================== model ====================

def player_table(b, decay_min, decay_rate):
    cols = ["n", "min", "eff_min", "start_rate", "last_date", "name", "_key", "team", "pos"] + ["r_" + s for s in STATS]
    p = b[b["played"]] if len(b) else b
    if p is None or len(p) == 0:
        return pd.DataFrame(columns=cols)
    p = p.sort_values(["date", "game_id"])
    age = p.groupby("pid").cumcount(ascending=False).to_numpy()
    wm, wr = decay_min ** age, decay_rate ** age
    mins = p["min"].to_numpy()
    p = p.assign(_wm=wm, _wmm=wm * mins, _wrm=wr * mins, **{"_s_" + s: wr * p[s].to_numpy() for s in STATS})
    g = p.groupby("pid")
    t = pd.DataFrame({"n": g.size(), "min": g["_wmm"].sum() / g["_wm"].sum(), "eff_min": g["_wrm"].sum(),
                      "start_rate": g["starter"].mean(), "last_date": g["date"].max()})
    sums = g[["_s_" + s for s in STATS]].sum()
    for s in STATS:
        t["r_" + s] = sums["_s_" + s] / t["eff_min"]
    last = p.groupby("pid").tail(1).set_index("pid")[["name", "_key", "team", "pos"]]
    return t.join(last)

def team_games(b):
    p = b[b["played"]]
    t = p.groupby(["game_id", "date", "team", "opp"], as_index=False)[STATS].sum()
    t["poss"] = t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]
    o = t[["game_id", "team", "poss"]].rename(columns={"team": "opp", "poss": "opp_poss"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["game_poss"] = (t["poss"] + t["opp_poss"].fillna(t["poss"])) / 2
    return t

def dvp_table(b, tg, k):
    """What each defense allows per minute to each position, per possession (pace removed), vs league; shrunk."""
    p = b[b["played"]]
    g = p.groupby(["opp", "pos"])
    allowed = g[STATS].sum().div(g["min"].sum(), axis=0)
    lgp = p.groupby("pos")[STATS].sum().div(p.groupby("pos")["min"].sum(), axis=0)
    lg_rows = lgp.reindex(allowed.index.get_level_values("pos")).to_numpy()
    ratio = allowed / lg_rows
    lg_pace = tg["game_poss"].mean()
    pace = tg.groupby("team")["game_poss"].mean() / lg_pace
    ratio = ratio.div(pace.reindex(allowed.index.get_level_values("opp")).to_numpy(), axis=0)
    n = tg.groupby("team")["game_id"].nunique().reindex(allowed.index.get_level_values("opp")).to_numpy()
    f = 1 + (ratio - 1).mul(n / (n + k), axis=0)
    return f, tg.groupby("team")["game_id"].nunique()

def prep_prev(prev_box, cfg):
    m = cfg["model"]
    if prev_box is None or len(prev_box) == 0:
        return None
    tg = team_games(prev_box)
    dvp, _ = dvp_table(prev_box, tg, m["dvp_shrink_games"])
    p = prev_box[prev_box["played"]]
    players = player_table(prev_box, m["prev_decay"], m["prev_decay"])
    sty = fit_styles(prev_box, players, tg, cfg)
    df_, _ = def_features(tg)
    rows = matchup_rows(prev_box, players, dvp, sty["P"].train, sty["D"].train, cfg, w=m["styles"]["prev_weight"])
    return {"players": players, "tg": tg, "dvp": dvp, "styles": sty, "style_rows": rows, "def_feat": df_,
            "off_feat": off_features(tg),
            "allow": tg.groupby("opp")["pts"].mean(),
            "pace": tg.groupby("team")["game_poss"].mean(), "lg_pace": tg["game_poss"].mean(),
            "pos_min": p.groupby("pos")["min"].sum(), "pos_sum": p.groupby("pos")[STATS].sum(),
            "team_tot": tg.groupby("team")[STATS].mean(), "lg_team_tot": tg[STATS].mean()}

# ==================== styles: player archetypes x team defensive styles ====================
# Players are grouped into archetypes by HOW they produce (usage, 3-point share, free-throw rate, passing, boards,
# blocks); defenses into styles by WHAT they give up (threes allowed, free throws allowed, turnovers forced, offensive
# boards allowed, rim protection, pace). Groups are fitted on last season (fixed all season, so labels don't jump) and
# this season's players/teams are assigned to the nearest group. Then: how does each archetype do vs each defensive
# style, home and away, compared with what the rest of the model already expects (player rate x defense-vs-position)?
# Pooling ~7 teams per style and dozens of players per archetype gives samples big enough to trust; one team's
# "road vs zone teams" split is mostly noise on its own.
ARCH_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]

def arch_features(t):
    fga = t["r_fga"].astype(float).clip(lower=1e-4)
    reb = t["r_reb"].astype(float).clip(lower=1e-4)
    return pd.DataFrame({"usage": (t["r_fga"] + 0.44 * t["r_fta"] + t["r_tov"]).astype(float) * 36,
                         "three": t["r_fg3a"].astype(float) / fga, "ftr": t["r_fta"].astype(float) / fga,
                         "ast": t["r_ast"].astype(float) * 36, "reb": reb * 36,
                         "oreb": t["r_oreb"].astype(float) / reb, "blk": t["r_blk"].astype(float) * 36}, index=t.index)

def def_features(tg):
    """Per defense: what offenses do against it (from team-game rows; `opp` is the defense)."""
    g = tg.groupby("opp")
    s = g[["fga", "fg3a", "fta", "tov", "oreb", "poss", "fgm", "ast"]].sum()
    own = tg.groupby("team")[["dreb", "blk"]].sum()
    out = pd.DataFrame({"three": s["fg3a"] / s["fga"], "ftr": s["fta"] / s["fga"], "tov": s["tov"] / s["poss"],
                        "oreb": s["oreb"] / (s["oreb"] + own["dreb"].reindex(s.index)),
                        "rim": own["blk"].reindex(s.index) / s["fga"], "ast": s["ast"] / s["fgm"].clip(lower=1),
                        "pace": g["game_poss"].mean()})
    return out, g["game_id"].nunique()

def off_features(tg):
    g = tg.groupby("team")
    s = g[["fga", "fg3a", "fta", "oreb", "dreb", "ast", "fgm"]].sum()
    return pd.DataFrame({"three": s["fg3a"] / s["fga"], "ftr": s["fta"] / s["fga"], "ast": s["ast"] / s["fgm"].clip(lower=1),
                         "oreb": s["oreb"] / (s["oreb"] + s["dreb"]), "pace": g["game_poss"].mean()})

def _kmeans(X, k, n_init=12, iters=200):
    best = None
    for seed in range(n_init):
        rng = np.random.default_rng(seed)
        C = X[[rng.integers(len(X))]]
        for _ in range(1, k):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1).min(1)
            C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            nc = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(nc, C):
                break
            C = nc
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    # stable order: sort groups by their first feature so labels don't depend on the random start
    order = np.argsort(best[1][:, 0])
    remap = np.empty_like(order); remap[order] = np.arange(len(order))
    return best[1][order], remap[best[2]]

class Styler:
    """Fitted groups + names. fit on a feature table; assign() maps new rows to the nearest group."""
    def __init__(self, F, k, namer, w=None):
        self.cols = list(F.columns)
        self.mu, self.sd = F.mean(), F.std().replace(0, 1)
        X = ((F - self.mu) / self.sd).to_numpy()
        self.w = np.ones(len(self.cols)) if w is None else np.asarray([w.get(c, 1.0) for c in self.cols])
        C, lab = _kmeans(X * self.w, k)
        self.C = C
        self.names = namer(pd.DataFrame(C / self.w, columns=self.cols))
        self.train = pd.Series([self.names[j] for j in lab], index=F.index)
    def assign(self, F):
        F = F[self.cols].replace([np.inf, -np.inf], np.nan).dropna()
        if F.empty:
            return pd.Series(dtype=object)
        X = ((F - self.mu) / self.sd).to_numpy() * self.w
        lab = ((X[:, None, :] - self.C[None]) ** 2).sum(-1).argmin(1)
        return pd.Series([self.names[j] for j in lab], index=F.index)

def _greedy_names(Z, labels):
    """Give each group the label that fits it best, each label used once (best fits picked first)."""
    sc = np.array([[f(z) for f in labels.values()] for _, z in Z.iterrows()])
    names, used, out = list(labels), set(), [None] * len(Z)
    for _ in range(len(Z)):
        sc_ = sc.copy()
        for j, n in enumerate(names):
            if n in used:
                sc_[:, j] = -1e9
        for i in range(len(Z)):
            if out[i] is not None:
                sc_[i, :] = -1e9
        i, j = np.unravel_index(np.argmax(sc_), sc_.shape)
        out[i] = names[j]
        used.add(names[j])
    return out

_bal = lambda z: -float(np.abs(z).sum()) / 2
PLAYER_LABELS = {
    "Primary creator": lambda z: z["usage"] + z["ast"], "Scorer": lambda z: z["usage"] - 0.5 * z["ast"] + 0.3 * z["ftr"],
    "Table-setter": lambda z: z["ast"] - 0.5 * z["usage"], "Spot-up shooter": lambda z: z["three"] - 0.7 * z["usage"],
    "Slasher": lambda z: z["ftr"] - z["three"] - 0.3 * z["reb"], "Rim-running big": lambda z: z["reb"] + z["blk"] + z["oreb"] - z["three"] - z["usage"],
    "Stretch big": lambda z: z["reb"] + z["three"] + 0.5 * z["blk"], "Playmaking big": lambda z: z["reb"] + z["ast"],
    "Post scorer big": lambda z: z["reb"] + z["usage"] - z["three"] - z["ast"],
    "Point forward": lambda z: z["usage"] + z["ast"] + z["reb"] - z["three"], "Paint big": lambda z: z["reb"] + z["ftr"] - z["three"] - z["ast"], "3-and-D wing": lambda z: z["three"] - z["ast"] + 0.2 * z["reb"],
    "Role wing": _bal}
DEF_LABELS = {
    "Pressure / gamble": lambda z: z["tov"] + 0.5 * z["ftr"], "Run-off-the-line": lambda z: -z["three"] - 0.3 * z["rim"],
    "Drop / protect paint": lambda z: z["rim"] + z["three"], "Leaky on the glass": lambda z: z["oreb"],
    "Up-tempo": lambda z: z["pace"], "Foul-prone": lambda z: z["ftr"] - z["tov"], "Balanced": _bal}
OFF_LABELS = {
    "Pace & space": lambda z: z["three"] + z["pace"], "Half-court threes": lambda z: z["three"] - z["pace"],
    "Ball movement": lambda z: z["ast"], "Paint & glass": lambda z: z["oreb"] + z["ftr"] - z["three"],
    "Iso-heavy": lambda z: -z["ast"], "Slow grind": lambda z: -z["pace"], "Balanced": _bal}

def name_players(Z):
    return _greedy_names(Z, PLAYER_LABELS)

def name_defenses(Z):
    return _greedy_names(Z, DEF_LABELS)

def name_offenses(Z):
    return _greedy_names(Z, OFF_LABELS)

def fit_styles(box, ptab, tg, cfg):
    """Fit player archetypes and team styles on one (finished) season."""
    sc = cfg["model"]["styles"]
    pl = ptab[(ptab["n"] >= 15) & (ptab["min"] >= 12)]
    P = Styler(arch_features(pl).replace([np.inf, -np.inf], np.nan).dropna(), sc["k_players"], name_players)
    D = Styler(def_features(tg)[0], sc["k_def"], name_defenses)
    O = Styler(off_features(tg), sc["k_off"], name_offenses)
    return {"P": P, "D": D, "O": O}

def matchup_rows(box, rates, dvp, arch, dstyle, cfg, w=1.0):
    """One row per player-game: actual vs what the model's base expects (minutes x his rate x defense-vs-position x
    home/away), tagged with his archetype, the opponent's defensive style and home/away."""
    sit = cfg["model"]["situational"]
    p = box[box["played"] & (box["min"] >= 10)]
    p = p[p["pid"].isin(rates.index) & p["pid"].isin(arch.index) & p["opp"].isin(dstyle.index)]
    if p.empty:
        return pd.DataFrame()
    hf = np.where(p["home"], sit["home_rate"], sit["away_rate"])
    out = pd.DataFrame({"arch": arch.reindex(p["pid"]).to_numpy(), "dstyle": dstyle.reindex(p["opp"]).to_numpy(),
                        "home": p["home"].to_numpy(), "w": w})
    idx = pd.MultiIndex.from_arrays([p["opp"], p["pos"]])
    for s in ARCH_STATS:
        dv = dvp[s].reindex(idx).fillna(1.0).to_numpy() if len(dvp) else 1.0
        out["e_" + s] = p["min"].to_numpy() * rates["r_" + s].reindex(p["pid"]).to_numpy() * dv * hf
        out["a_" + s] = p[s].to_numpy()
    return out

def matchup_factors(rows, cfg):
    """archetype x defensive style (x home/away) multipliers, shrunk twice: cell -> (archetype, style) -> 1.0."""
    sc = cfg["model"]["styles"]
    k, kh, cap = sc["shrink_games"], sc["home_shrink_games"], sc["cap"]
    if rows is None or rows.empty:
        return {}
    r = rows.copy()
    out = {}
    for s in ARCH_STATS:
        r["_e"], r["_a"] = r["e_" + s] * r["w"], r["a_" + s] * r["w"]
        g2 = r.groupby(["arch", "dstyle"]).agg(e=("_e", "sum"), a=("_a", "sum"), n=("w", "sum"))
        g2 = g2[g2["e"] > 0]
        f2 = 1 + (g2["a"] / g2["e"] - 1) * g2["n"] / (g2["n"] + k)
        g3 = r.groupby(["arch", "dstyle", "home"]).agg(e=("_e", "sum"), a=("_a", "sum"), n=("w", "sum"))
        g3 = g3[g3["e"] > 0]
        base = f2.reindex(pd.MultiIndex.from_arrays([g3.index.get_level_values(0), g3.index.get_level_values(1)])).fillna(1.0).to_numpy()
        # home split: how this cell's home (or road) games differ from the cell as a whole
        g2a = g2.reindex(pd.MultiIndex.from_arrays([g3.index.get_level_values(0), g3.index.get_level_values(1)]))
        cell_ratio = (g2a["a"] / g2a["e"]).to_numpy()
        rel = (g3["a"] / g3["e"]).to_numpy() / np.where(cell_ratio > 0, cell_ratio, 1.0)
        f3 = base * (1 + (rel - 1) * g3["n"].to_numpy() / (g3["n"].to_numpy() + kh))
        out[s] = pd.Series(np.clip(f3, 1 - cap, 1 + cap), index=g3.index)
    return out

class State:
    """Everything known before tip on `date`: player baselines, team pace, defense-vs-position, team totals."""
    def __init__(self, cur_box, prev_box, date, cfg, prev_cache=None, prev2_box=None):
        m = cfg["model"]
        self.cfg, self.date = cfg, pd.Timestamp(date)
        cur = cur_box[cur_box["date"] < self.date] if len(cur_box) else cur_box
        self.cur = cur
        self.pv = prev_cache if prev_cache is not None else prep_prev(prev_box, cfg)
        pv = self.pv
        ct = player_table(cur, m["decay_min"], m["decay_rate"])
        pt = pv["players"] if pv else pd.DataFrame(columns=ct.columns)
        # players who missed all of last season (injury): fall back to the season before, at half the prior weight
        if prev2_box is not None and len(prev2_box):
            p2 = player_table(prev2_box, m["prev_decay"], m["prev_decay"])
            p2 = p2[~p2.index.isin(pt.index)]
            if len(p2):
                p2 = p2.assign(eff_min=p2["eff_min"] * 0.5)
                pt = pd.concat([pt, p2]) if len(pt) else p2
        # league per-minute rates by position (this season + last)
        pos_min = pv["pos_min"].copy() if pv else pd.Series(dtype=float)
        pos_sum = pv["pos_sum"].copy() if pv else pd.DataFrame(columns=STATS)
        cp = cur[cur["played"]] if len(cur) else cur
        if len(cp):
            pos_min = pos_min.add(cp.groupby("pos")["min"].sum(), fill_value=0)
            pos_sum = pos_sum.add(cp.groupby("pos")[STATS].sum(), fill_value=0)
        self.lg = pos_sum.div(pos_min, axis=0)  # index pos, cols STATS
        # blend players: this season (recency-weighted) + last season as a prior
        ids = ct.index.union(pt.index)
        c, p = ct.reindex(ids), pt.reindex(ids)
        num_cols = ["n", "min", "eff_min", "start_rate"] + ["r_" + s for s in STATS]
        c[num_cols] = c[num_cols].apply(pd.to_numeric, errors="coerce")
        p[num_cols] = p[num_cols].apply(pd.to_numeric, errors="coerce")
        c["last_date"] = pd.to_datetime(c["last_date"])
        p["last_date"] = pd.to_datetime(p["last_date"])
        out = pd.DataFrame(index=ids)
        for col in ("name", "_key", "pos", "team"):
            out[col] = c[col].where(c[col].notna(), p[col])
        out["prev_team"] = p["team"]
        out["n_cur"], out["n_prev"] = c["n"].fillna(0), p["n"].fillna(0)
        hc, hp = c["n"].notna(), p["n"].notna()
        k = m["prior_min_games"]
        out["min"] = np.where(hc & hp, (c["n"] * c["min"] + k * p["min"]) / (c["n"] + k),
                              np.where(hc, c["min"], p["min"]))
        ec = c["eff_min"].fillna(0).astype(float)
        kp = np.minimum(p["eff_min"].fillna(0).astype(float), m["prior_rate_minutes"])
        ks = m["pos_shrink_minutes"]
        for s in STATS:
            lg = out["pos"].map(self.lg[s]).astype(float)
            num = ec * c["r_" + s].fillna(0) + kp * p["r_" + s].fillna(0)
            out["r_" + s] = (num + ks * lg) / (ec + kp + ks)
        out["eff_min"] = ec + kp
        out["start_rate"] = c["start_rate"].where(c["start_rate"].notna(), p["start_rate"])
        out["last_date"] = c["last_date"].where(c["last_date"].notna(), p["last_date"])
        self.players = out[out["min"].notna()]
        # team pace and totals (this season, prior season as a shrunk prior)
        self.tg = team_games(cur) if len(cp) else None
        lg_pace_p = pv["lg_pace"] if pv else 99.0
        n_cur = self.tg.groupby("team")["game_id"].nunique() if self.tg is not None else pd.Series(dtype=float)
        self.lg_pace = self.tg["game_poss"].mean() if self.tg is not None and len(self.tg) > 60 else lg_pace_p
        kpace, tr = m["pace_prior_games"], m["prior_trust"]
        pace = {}
        for t in NBA_TEAMS:
            pp = pv["pace"].get(t, lg_pace_p) if pv else lg_pace_p
            prior = self.lg_pace + (pp - lg_pace_p) * tr
            n = n_cur.get(t, 0)
            pc = self.tg[self.tg["team"] == t]["game_poss"].mean() if n else 0.0
            pace[t] = (n * pc + kpace * prior) / (n + kpace)
        self.pace = pd.Series(pace)
        # team production per game (what the usage model has to add up to): this season, recency-weighted,
        # shrunk toward the league average team
        self.lg_team_tot = pv["lg_team_tot"] if pv else self.tg[STATS].mean()
        self.n_team = n_cur
        tot = {}
        kt = m["team_total_prior_games"]
        for t in NBA_TEAMS:
            g = self.tg[self.tg["team"] == t].sort_values("date") if self.tg is not None else None
            if g is not None and len(g):
                w = 0.95 ** np.arange(len(g) - 1, -1, -1)
                own = (g[STATS].mul(w, axis=0)).sum() / w.sum()
                tot[t] = (w.sum() * own + kt * self.lg_team_tot) / (w.sum() + kt)
            else:
                tot[t] = self.lg_team_tot
        self.team_tot = pd.DataFrame(tot).T
        # availability: share of the team's last 15 games each player played (counted from his first game there)
        self.avail = {}
        if len(cp):
            for t, g in cp.groupby("team"):
                games = g.drop_duplicates("game_id").sort_values("date")[["game_id", "date"]].tail(15)
                first = g.groupby("pid")["date"].min()
                pl = g[g["game_id"].isin(games["game_id"])].groupby("pid").size()
                for pid, fd in first.items():
                    avail_games = (games["date"] >= fd).sum()
                    if avail_games:
                        self.avail[(t, pid)] = pl.get(pid, 0) / avail_games
        # defense vs position: this season shrunk, last season (half trust) as prior
        kd, cap = m["dvp_prior_games"], m["dvp_cap"]
        prior = 1 + (pv["dvp"] - 1) * tr if pv else None
        if self.tg is not None and len(cp):
            fc, nc = dvp_table(cur, self.tg, m["dvp_shrink_games"])
            nn = nc.reindex(fc.index.get_level_values("opp")).fillna(0).to_numpy()
            pr = prior.reindex(fc.index).fillna(1.0) if prior is not None else 1.0
            dvp = fc.mul(nn, axis=0).add(pr * kd if prior is not None else kd).div(nn + kd, axis=0)
        else:
            dvp = prior if prior is not None else pd.DataFrame()
        self.dvp = dvp.clip(cap[0], cap[1]) if len(dvp) else dvp
        # points each defense allows per game (for the game-total check), this season shrunk toward last season
        lg_pts = float(self.lg_team_tot["pts"])
        prior_allow = (lg_pts + (pv["allow"] - lg_pts) * tr) if pv and "allow" in pv else pd.Series(lg_pts, index=sorted(NBA_TEAMS))
        if self.tg is not None and len(self.tg):
            ca = self.tg.groupby("opp")["pts"].agg(["mean", "size"]).reindex(prior_allow.index)
            nn_ = ca["size"].fillna(0)
            self.allow = (nn_ * ca["mean"].fillna(0) + kt * prior_allow) / (nn_ + kt)
        else:
            self.allow = prior_allow
        self.lg_pts = lg_pts
        # ---- styles: archetypes, defensive/offensive styles, archetype x style factors ----
        self.arch, self.dstyle, self.ostyle, self.mf = pd.Series(dtype=object), pd.Series(dtype=object), pd.Series(dtype=object), {}
        self.pz, self.dz = pd.DataFrame(), pd.DataFrame()
        sty = pv.get("styles") if pv else None
        if sty:
            ms = m["styles"]
            self.arch = sty["P"].assign(arch_features(self.players[self.players["eff_min"] >= ms["min_minutes"]]))
            dprev, oprev = pv["def_feat"], pv["off_feat"]
            dmix, omix = dprev, oprev
            if self.tg is not None and len(self.tg) > 30:
                kk = ms["team_prior_games"]
                dc, nd = def_features(self.tg)
                oc = off_features(self.tg)
                no = self.tg.groupby("team")["game_id"].nunique()
                wd = (nd.reindex(dprev.index).fillna(0) / (nd.reindex(dprev.index).fillna(0) + kk)).to_numpy()[:, None]
                wo = (no.reindex(oprev.index).fillna(0) / (no.reindex(oprev.index).fillna(0) + kk)).to_numpy()[:, None]
                dmix = dprev * (1 - wd) + dc.reindex(dprev.index).fillna(dprev) * wd
                omix = oprev * (1 - wo) + oc.reindex(oprev.index).fillna(oprev) * wo
            self.dstyle, self.ostyle = sty["D"].assign(dmix), sty["O"].assign(omix)
            # continuous traits in z-scores (vs last season's rotation players / teams) for trait x trait matchups
            pf_ = arch_features(self.players).replace([np.inf, -np.inf], np.nan)
            self.pz = (pf_ - sty["P"].mu) / sty["P"].sd
            self.dz = (dmix - sty["D"].mu) / sty["D"].sd
            rows = [pv["style_rows"]]
            if len(cp):
                rows.append(matchup_rows(cur, player_table(cur, 1.0, 1.0), self.dvp, self.arch, self.dstyle, cfg, 1.0))
            self.mf = matchup_factors(pd.concat([r for r in rows if len(r)]), cfg)
        # ---- per-player extras: recent role, foul rate, how streaky he is (dispersion vs the fitted variance) ----
        ex = pd.DataFrame(index=self.players.index)
        if len(cp):
            srt = cp.sort_values(["date", "game_id"])
            ex["start_l2"] = srt.groupby("pid").tail(2).groupby("pid")["starter"].mean()
            ex["min_l3"] = srt.groupby("pid").tail(3).groupby("pid")["min"].mean()
            l10 = srt.groupby("pid").tail(10).groupby("pid")["min"]
            ex["min_sd10"], ex["min_lo10"], ex["min_hi10"], ex["min_m10"] = l10.std(), l10.min(), l10.max(), l10.mean()
        allg = pd.concat([x for x in (cp, prev_box[prev_box["played"]] if prev_box is not None and len(prev_box) else None)
                          if x is not None and len(x)]) if (len(cp) or (prev_box is not None and len(prev_box))) else None
        if allg is not None:
            allg = allg[allg["pid"].isin(self.players.index) & (allg["min"] >= 10)]
            for s_ in ("pts", "reb", "ast", "fg3m"):
                mu_ = allg["min"].to_numpy() * self.players["r_" + s_].reindex(allg["pid"]).to_numpy()
                a_, b_ = m["var"][s_]
                v_ = np.maximum(a_ * mu_ + b_ * mu_ ** 2, 0.05)
                t_ = pd.DataFrame({"pid": allg["pid"].to_numpy(), "z2": (allg[s_].to_numpy() - mu_) ** 2 / v_}).groupby("pid")["z2"].agg(["sum", "count"])
                ex["disp_" + s_] = t_["sum"] / t_["count"]
                ex["disp_n"] = t_["count"]
            # minutes when he starts vs comes off the bench (this season + last): what a news role change moves him to
            sb_ = allg.groupby(["pid", "starter"])["min"].agg(["mean", "count"]).unstack("starter")
            for flag_, nm_ in ((True, "start"), (False, "bench")):
                if ("mean", flag_) in sb_.columns:
                    ex[nm_ + "_min"] = sb_[("mean", flag_)]
                    ex[nm_ + "_n"] = sb_[("count", flag_)]
        self.extra = ex

    def find(self, name):
        key = norm_name(name)
        P = self.players
        hit = P[P["_key"] == key]
        if hit.empty:
            parts = key.split()
            if len(parts) >= 2:
                cand = P[P["_key"].str.endswith(" " + parts[-1]) & P["_key"].str.startswith(parts[0][:1])]
                if len(cand) == 1:
                    hit = cand
        if hit.empty:
            return None
        return hit.sort_values("last_date").index[-1]

POS_SIM = {("G", "G"): 1.0, ("F", "F"): 1.0, ("C", "C"): 1.0, ("G", "F"): 0.5, ("F", "G"): 0.5,
           ("F", "C"): 0.6, ("C", "F"): 0.6, ("G", "C"): 0.1, ("C", "G"): 0.1}
USE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "oreb", "dreb", "fga", "fta", "fg3a"]

def without_split(state, pid, team, out_pid):
    """Player's real minutes and per-minute rates in games his team played without `out_pid` (this + last season)."""
    rows = []
    for box, w in ((state.cur, 1.0), (state.pv_box if hasattr(state, "pv_box") else None, 0.6)):
        if box is None or len(box) == 0:
            continue
        tb = box[(box["team"] == team) & box["played"]]
        o_games = set(tb[tb["pid"] == out_pid]["game_id"])
        if not o_games:
            continue
        mine = tb[(tb["pid"] == pid) & ~tb["game_id"].isin(o_games)]
        if len(mine):
            rows.append(mine.assign(_w=w))
    if not rows:
        return None
    r = pd.concat(rows)
    wmin = (r["_w"] * r["min"]).sum()
    if wmin <= 0:
        return None
    out = {"n": len(r), "neff": r["_w"].sum(), "min": (r["_w"] * r["min"]).sum() / r["_w"].sum()}
    for s in USE_STATS:
        out["r_" + s] = (r["_w"] * r[s]).sum() / wmin
    return out

def team_adjust(state, sl, outs, rosters, cfg, use_split=True, sure=None):
    """USAGE MODEL. Team totals have to add up: ~240.7 minutes and a team-sized stat line per game.
    1) An out player's minutes go first to teammates at his position (weighted by their role), scaled by how often he
       has actually been playing lately (a long-term absence is already in everyone's averages -> no double count).
    2) Minutes of everyone expected to play are normalized toward 240.7 (healthy roster -> slightly fewer each).
    3) Each stat (points, shots, assists, rebounds...) produced by the active roster is scaled toward the team's
       normal per-game production: stars out -> everyone's rates go up; a roster stacked with scorers -> down.
    Partial rosters (no games yet this season) leave room for unlisted bench minutes and are only trimmed down."""
    m = cfg["model"]
    P = state.players
    sl = sl.copy()
    sl["dmin"] = 0.0
    for s in USE_STATS:
        sl["u_" + s] = 1.0
    sl["usage_note"] = ""
    for c in ("ws_n", "ws_min", "ws_r_pts", "ws_r_reb", "ws_r_ast", "ws_r_fg3m", "m2"):
        sl[c] = np.nan
    out_ids = set(outs["pid"]) if outs is not None and len(outs) else set()
    for team in sl["team"].dropna().unique():
        roster = [p for p in dict.fromkeys(rosters.get(team, []) + list(sl.index[sl["team"] == team]))
                  if p in P.index and p not in out_ids]
        R = P.loc[roster, ["name", "pos", "min"] + ["r_" + s for s in USE_STATS]].copy()
        if R.empty:
            continue
        partial = state.n_team.get(team, 0) < 3
        av = pd.Series({p: state.avail.get((team, p), np.nan) for p in R.index})
        # who will play: today's slate players yes; other healthy roster players at their recent play rate
        surely = sl.index if sure is None else pd.Index(list(sure))
        R["p"] = np.where(R.index.isin(surely), 1.0, av.fillna(m["unknown_play_rate"]).clip(0, 1))
        ot = outs[(outs["team"] == team) & (outs["min"] >= m["usage_min_out_minutes"])] if len(out_ids) else []
        dmin = pd.Series(0.0, index=R.index)
        names = []
        for _, o in (ot.iterrows() if len(ot) else []):
            # early season (<3 team games): the roster is only who is listed today, so an out player's minutes are
            # not in it to begin with -> no explicit transfer (the 240-minute fill below handles it, capped)
            a_o = 0.0 if partial else state.avail.get((team, o["pid"]), 0.0)
            w = R["pos"].map(lambda q: POS_SIM.get((o["pos"], q), 0.3)) * R["min"] * R["p"]
            if w.sum() > 0 and a_o > 0:
                dmin += o["min"] * a_o * w / w.sum()
            names.append(f"{o['name']}" + ("" if partial or a_o >= 0.8 else f" (played {a_o:.0%} lately)"))
        m1 = R["min"] + dmin
        tot_min = (R["p"] * m1).sum()
        bench = float(np.clip(m["team_minutes"] - tot_min, 0, m["bench_allowance_max"])) if partial else 0.0
        scale = (m["team_minutes"] / (tot_min + bench)) ** m["min_norm_alpha"] if tot_min > 0 else 1.0
        if partial:
            scale = min(scale, m["partial_scale_cap"])
        if m.get("norm_mode", "scale") == "bench" and scale < 1 and tot_min > 0:
            # too many minutes on the roster: take the excess mostly from bench minutes, not from the starters
            # (early season the prior over-credits bench roles; proportional scaling cut starters ~2-4 min)
            cut = tot_min * (1 - scale)
            w = (m1 * np.clip(m.get("bench_pivot", 40) - m1, 1, None)).clip(lower=0)
            denom = float((R["p"] * w).sum())
            m2 = np.minimum((m1 - (cut * w / denom if denom > 0 else 0)).clip(lower=0), m["max_minutes"])
        else:
            m2 = np.minimum(m1 * scale, m["max_minutes"])
        mult = {}
        for s in USE_STATS:
            beta = m["usage_beta"].get(s, m["usage_beta"]["default"]) * (m["partial_beta_mult"] if partial else 1.0)
            prod = (R["p"] * m2 * R["r_" + s]).sum() + bench * state.lg.loc["F", s] * m["usage_rep_factor"]
            target = state.team_tot.loc[team, s]
            mult[s] = (target / prod) ** beta if prod > 0 else 1.0
            mult[s] = float(np.clip(mult[s], 1 / m["usage_mult_cap"], m["usage_mult_cap"]))
        big = ot.sort_values("min").iloc[-1] if len(ot) else None
        for pid in sl.index[sl["team"] == team]:
            if pid not in R.index:
                continue
            sl.at[pid, "dmin"] = m2[pid] - P.at[pid, "min"]
            sl.at[pid, "m2"] = m2[pid]
            for s in USE_STATS:
                sl.at[pid, "u_" + s] = mult[s]
            why = ("team-total trim: this roster's rates add up to more than the team normally produces" if mult["pts"] < 0.995
                   else "missing production shared out" if mult["pts"] > 1.005 else "")
            note = (f"OUT: {', '.join(names)} | " if names else "") + \
                   f"{m2[pid] - P.at[pid, 'min']:+.1f} min, usage x{mult['fga']:.2f}, pts x{mult['pts']:.2f}" + (f" ({why})" if why else "")
            if use_split and big is not None:
                ws = without_split(state, pid, team, big.name)
                if ws:
                    for c in ("n", "min", "r_pts", "r_reb", "r_ast", "r_fg3m"):
                        sl.at[pid, "ws_" + c] = ws[c]
                if ws and ws["n"] >= m["wo_min_games"]:
                    note += f" | real games w/o {big['name']}: {ws['n']} ({ws['min']:.0f} min, {ws['r_pts'] * 36:.1f} pts/36)"
            sl.at[pid, "usage_note"] = note
    return sl

def blowout_factor(spread, starter, cfg):
    """Expected minutes vs an average game given the team's spread (negative = favored). Margin ~ N(-spread, 12)."""
    b = cfg["model"]["blowout"]
    if spread is None or pd.isna(spread):
        return 1.0
    edges = np.array(b["edges"], dtype=float)
    f = np.array(b["starter"] if starter else b["bench"], dtype=float)
    xs = np.linspace(-45, 45, 181)
    def expect(mu):
        w = st.norm.pdf(xs, mu, b["margin_sd"])
        idx = np.clip(np.searchsorted(edges, np.abs(xs), side="right") - 1, 0, len(f) - 1)
        return float(np.sum(w * f[idx]) / np.sum(w))
    return expect(-float(spread)) / expect(0.0)

def project(state, sl, outs, rosters, cfg, flags=None, sure=None, limited=None, news=None):
    """sl: DataFrame indexed by pid with team, opp, home, b2b, spread. Returns means and variances per market."""
    m = cfg["model"]
    flags = flags or {}
    P = state.players
    sl = sl.join(P[["name", "pos", "min", "n_cur", "n_prev", "prev_team", "start_rate"]], how="inner")
    # new team (offseason move or fresh trade): effect fades over his first 10 games with the new team (see by_role)
    cur_team = P["team"].reindex(sl.index)
    games_with = np.where(cur_team != sl["team"], 0, np.where(sl["prev_team"].notna() & (sl["prev_team"] != sl["team"]), sl["n_cur"], 99))
    nt = m["new_team"]
    sl["games_new_team"] = games_with
    sl["new_team"] = games_with < len(nt["min"])
    idx = np.minimum(games_with, len(nt["min"]) - 1).astype(int)
    # role matters (measured 2024-25 + 2025-26, vs same-team players of the same role): starters (28+ min) keep their
    # minutes on a new team but score ~10% less per minute early; 20-28 min players lose ~10% of minutes; bench more.
    tier = np.where(sl["min"] >= 28, "starter", np.where(sl["min"] >= 20, "rotation", "bench"))
    mins_tab = {k: np.array(v["min"]) for k, v in nt["by_role"].items()}
    rate_tab = {k: np.array(v["rate"]) for k, v in nt["by_role"].items()}
    sl["nt_min"] = [mins_tab[t][i] if n else 1.0 for t, i, n in zip(tier, idx, sl["new_team"])]
    sl["nt_rate"] = [rate_tab[t][i] if n else 1.0 for t, i, n in zip(tier, idx, sl["new_team"])]
    sl = team_adjust(state, sl, outs if flags.get("usage", True) else None, rosters, cfg,
                     use_split=flags.get("split", True), sure=sure) if flags.get("team", True) else \
        sl.assign(dmin=0.0, usage_note="", **{"u_" + s: 1.0 for s in USE_STATS})
    sit = m["situational"]
    stren = sit["strength"]
    mins = sl["min"] + sl["dmin"]
    sl["xm_usage"] = np.asarray(mins, dtype=float)  # v2.15: minutes chain kept for `explain`
    # recent role: last 3 games' minutes pull the projection 40% of the way (backtest 2025-26: minutes MAE 5.00 -> 4.69)
    if flags.get("role", True):
        ml3 = state.extra["min_l3"].reindex(sl.index) if "min_l3" in state.extra else pd.Series(np.nan, index=sl.index)
        mins = mins + m["role_weight"] * (ml3.fillna(sl["min"]) - sl["min"])
    sl["xm_role"] = np.asarray(mins, dtype=float)
    # real games without tonight's biggest absence: blend his actual minutes in those games (weight n / (n + 32))
    if flags.get("wo", True) and "ws_n" in sl:
        n_ = sl["ws_n"].fillna(0).to_numpy()
        ok_ = n_ >= m["wo_min_games"]
        k_ = m["wo_min_k"]
        mins = pd.Series(np.where(ok_, (n_ * sl["ws_min"].fillna(0).to_numpy() + k_ * mins.to_numpy()) / (n_ + k_), mins.to_numpy()), index=sl.index)
    sl["xm_wo"] = np.asarray(mins, dtype=float)
    b2b = sl["b2b"].fillna(False).astype(bool)
    if flags.get("b2b", True):
        mins = mins * np.where(b2b, 1 + (sit["b2b_min"] - 1) * stren, 1.0)
    sl["xm_b2b"] = np.asarray(mins, dtype=float)
    if flags.get("blowout", True):
        bf = [blowout_factor(sp, mn >= 26, cfg) for sp, mn in zip(sl["spread"], sl["min"])]
        mins = mins * (1 + (np.array(bf) - 1) * stren)
    sl["xm_blow"] = np.asarray(mins, dtype=float)
    # minutes restriction: listed as Limited (news says "minutes restriction"), or back from a long absence
    ret = m["return"]
    lim = sl.index.isin(list(limited or []))
    last_ = pd.to_datetime(P["last_date"].reindex(sl.index))
    if state.tg is not None and len(state.tg):
        td_ = state.tg[["team", "date"]].drop_duplicates()
        miss_ = np.array([int(((td_["team"] == t) & (td_["date"] > d)).sum()) if pd.notna(d) else 0 for t, d in zip(sl["team"], last_)])
    else:
        miss_ = np.zeros(len(sl), dtype=int)
    use_ret = flags.get("return", True)
    rmult = np.ones(len(sl))
    if use_ret:
        for lo_, f_ in zip(ret["missed"], ret["mult"]):
            rmult = np.where(miss_ >= lo_, f_, rmult)
    back = use_ret & (miss_ >= ret["hold_missed"])
    sl["limited"] = lim
    sl["returning"] = back
    sl["missed"] = miss_
    mins = mins * np.where(lim, np.minimum(m["restricted_min_mult"], rmult), rmult)
    sl["xm_restrict"] = np.asarray(mins, dtype=float)
    # news roles (v2.7, from the news sweep): Starting / Bench move minutes 60% of the way to his own minutes in starts /
    # bench games (3+ such games, else +15% / -20%); "Cap N" = a reported minutes limit. Only hard facts from news.
    sl["news_note"] = ""
    if news:
        ex_ = state.extra
        mv_ = mins.copy()
        for pid_, (kind_, val_) in news.items():
            if pid_ not in mv_.index:
                continue
            cur_ = float(mv_[pid_])
            if kind_ == "cap":
                new_ = min(cur_, float(val_))
            else:
                col_ = "start" if kind_ == "start" else "bench"
                n_ = float(ex_[col_ + "_n"].get(pid_, 0) or 0) if col_ + "_n" in ex_ else 0.0
                tgt_ = float(ex_[col_ + "_min"].get(pid_)) if n_ >= 3 else cur_ * (1.15 if kind_ == "start" else 0.8)
                new_ = 0.4 * cur_ + 0.6 * tgt_
                new_ = max(cur_, new_) if kind_ == "start" else min(cur_, new_)
            mv_[pid_] = new_
            sl.loc[pid_, "news_note"] = f"news: {'starting' if kind_ == 'start' else 'bench' if kind_ == 'bench' else f'minutes cap {val_:g}'} ({cur_:.0f}->{new_:.0f} min)"
        mins = mv_
    sl["xm_news"] = np.asarray(mins, dtype=float)
    sl["proj_min"] = np.minimum(mins * sl["nt_min"], m["max_minutes"])
    sl["xm_nt"] = sl["proj_min"].to_numpy(dtype=float)
    # v2.13 bench pull: a bench player (started < half his games) projected ABOVE his last 3 games' minutes plays less
    # than projected (2024-25 and 2025-26: projected 5+ min over his last 3 -> played ~3 min less, stats ~4% under the
    # model, Overs hit ~4 pts less often). Pull 20% of that gap back (fit on one season, tested on the other, both ways;
    # 11/2 Isaiah Collier: bench guard, last season's 26 mpg prior kept his minutes up, he played 12).
    bp = m.get("bench_l3_pull", 0.0)
    if flags.get("bench_pull", True) and bp and "min_l3" in state.extra:
        l3 = state.extra["min_l3"].reindex(sl.index).to_numpy(dtype=float)
        sr = pd.to_numeric(sl["start_rate"], errors="coerce").fillna(1.0).to_numpy()
        gap = np.where(np.isfinite(l3) & (sr < 0.5), np.maximum(sl["proj_min"].to_numpy() - l3, 0.0), 0.0)
        sl["proj_min"] = sl["proj_min"].to_numpy() - bp * gap
        sl["bench_trim"] = np.round(bp * gap, 1)
    pace = (state.pace.reindex(sl["opp"]).to_numpy() / state.lg_pace) if flags.get("pace", True) else 1.0
    pace_f = np.power(pace, m["pace_elasticity"])
    home_f = np.where(sl["home"].fillna(False).astype(bool), sit["home_rate"], sit["away_rate"]) if flags.get("home", True) else 1.0
    home_f = 1 + (np.asarray(home_f) - 1) * stren
    b2b_r = np.where(b2b, 1 + (sit["b2b_rate"] - 1) * stren, 1.0) if flags.get("b2b", True) else 1.0
    # styles: his archetype vs the opponent's defensive style (home/away)
    homeb = sl["home"].fillna(False).astype(bool).to_numpy()
    sl["arch"] = state.arch.reindex(sl.index).to_numpy()
    sl["opp_style"] = state.dstyle.reindex(sl["opp"]).to_numpy()
    sl["team_style"] = state.ostyle.reindex(sl["team"]).to_numpy()
    for c_ in state.pz.columns:
        sl["pz_" + c_] = state.pz[c_].reindex(sl.index).to_numpy()
    for c_ in state.dz.columns:
        sl["dz_" + c_] = state.dz[c_].reindex(sl["opp"]).to_numpy()
    sstr = m["styles"]["strength"]
    # game total: the market's implied team points vs what the base model expects (team scoring x opponent allowed)
    tcfg = m["total"]
    tot = pd.to_numeric(sl["total"], errors="coerce") if "total" in sl else pd.Series(np.nan, index=sl.index)
    spr = pd.to_numeric(sl["spread"], errors="coerce")
    implied = ((tot - spr) / 2).to_numpy()
    base_pts = state.team_tot["pts"].reindex(sl["team"]).to_numpy() * \
        state.allow.reindex(sl["opp"]).to_numpy() / state.lg_pts * np.asarray(home_f, dtype=float)
    env = np.clip(np.where(np.isfinite(implied) & (base_pts > 0), implied / base_pts, 1.0), 1 - tcfg["cap"], 1 + tcfg["cap"])
    sl["env"] = env
    sl["xbase_pts"] = base_pts
    sl["implied_pts"] = implied
    sl["days_off"] = (state.date - pd.to_datetime(P["last_date"].reindex(sl.index))).dt.days.to_numpy()
    sl = sl.join(state.extra, how="left")
    sl["r_pf"] = P["r_pf"].reindex(sl.index).to_numpy() if "r_pf" in P else np.nan
    for s in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
        if flags.get("arch", True) and s in state.mf and s in m["styles"]["apply"]:
            key = pd.MultiIndex.from_arrays([sl["arch"], sl["opp_style"], homeb])
            af = state.mf[s].reindex(key).fillna(1.0).to_numpy()
            af = 1 + (af - 1) * sstr
        else:
            af = np.ones(len(sl))
        # v2.24: a LOW total cuts rebounds by less (reb_low 0.5 below 1.0, 1.25 above). Pace is already applied by pace_f, so a
        # low total mostly means bad shooting expected = more misses to rebound (11/16 GS@NO: total 223.5, game 209 on 33%
        # shooting, Missi 14 reb vs U6.5). Backtest 2025-26 (8,275 reb projections with DK totals): totals <=0.94 of normal
        # had rebounds +0.29 above the projection and Overs hit 55.5% vs 50.1% said; low side 0.5 fits best overall and
        # was never worse than 1.25 in either half of the season.
        _el = tcfg["elasticity"].get(s, 0.0)
        _el = np.where(env < 1, tcfg["elasticity"].get(s + "_low", _el), _el)
        ef = np.power(env, _el) if flags.get("total", True) else 1.0
        sl["af_" + s] = af
        if flags.get("dvp", True) and len(state.dvp):
            idx = pd.MultiIndex.from_arrays([sl["opp"], sl["pos"]])
            # v2.36 (research desk 11/20, re-tested in the engine): made threes are rated by the 3-point ATTEMPTS a defense
            # allows to the position, not the makes. The 3P% a defense "allows" is mostly luck and does not carry over;
            # attempts allowed do. Walk-forward on 34,007 player-games (2024-25, 2025-26, 2026-27): the attempts factor had
            # lower error in all three seasons (pooled 3.9 SE), and it is ~110% real vs ~69% for the makes factor.
            _dc = "fg3a" if s == "fg3m" and "fg3a" in state.dvp.columns else s
            dv = state.dvp[_dc].reindex(idx).fillna(1.0).to_numpy()
        else:
            dv = 1.0
        rate = P["r_" + s].reindex(sl.index).to_numpy() * sl["u_" + s].to_numpy()
        sl["mu_" + s] = sl["proj_min"].to_numpy() * rate * dv * pace_f * home_f * b2b_r * sl["nt_rate"].to_numpy() * af * ef
        sl["xraw_" + s] = P["r_" + s].reindex(sl.index).to_numpy()
        sl["ef_" + s] = np.ones(len(sl)) * ef
        sl["dvp_" + s] = dv
    sl["pace_f"] = pace_f
    sl["xh_f"] = np.ones(len(sl)) * home_f
    sl["xb2b_r"] = np.ones(len(sl)) * b2b_r
    for mk, comps in MARKETS.items():
        if len(comps) > 1:
            sl["mu_" + mk] = sum(sl["mu_" + c] for c in comps)
    sdm = np.where(sl["new_team"], m["new_team_sd_mult"], np.where(sl["n_cur"] < 5, m["early_sd_mult"], 1.0))
    for mk in MARKETS:
        a, b = m["var"][mk]
        mu = sl["mu_" + mk].clip(lower=0.01)
        sl["var_" + mk] = (a * mu + b * mu ** 2) * sdm ** 2
    return sl

# ---------- probability calibration ----------
# Raw P(over) from the distribution -> calibrated chance. Middle (35-65%) = the old flat 0.8 shrink toward 50%; tails
# (<25%, >75% raw) = isotonic fit on walk-forward backtests (the flat shrink was too timid at the low end and too
# strong at the high end); linear blend in between. Map in CONFIG["model"]["prob_map"] (x = raw, y = calibrated).
def cal_prob(p):
    m = CONFIG["model"].get("prob_map")
    if m:
        return np.interp(p, m["x"], m["y"]) if np.ndim(p) else float(np.interp(p, m["x"], m["y"]))
    sh = CONFIG["model"]["prob_calibration_shrink"]
    return 0.5 + (np.asarray(p) - 0.5) * sh if np.ndim(p) else 0.5 + (p - 0.5) * sh

def prob_over(mu, var, market, line):
    if market in GAME_MK:   # v2.30: mu = this team's expected margin (spread / ml) or the expected total
        thr = -float(line) if market == "spread" else (0.0 if market == "ml" else float(line))
        return float(st.norm.sf(thr, mu, math.sqrt(max(var, 1e-6))))
    if mu <= 0:
        return 0.0
    if market in CONT:
        return float(st.norm.sf(math.floor(line) + 0.5, mu, math.sqrt(max(var, 1e-6))))
    k = math.floor(line)
    if var <= mu * 1.02:
        return float(st.poisson.sf(k, mu))
    n = mu * mu / (var - mu)
    return float(st.nbinom.sf(k, n, n / (n + mu)))

# ==================== slate building ====================

def build_slate(state, schedule, date, players, teams_override, games=None):
    """players: list of pids; teams_override: pid -> team (from today's lines / injury list)."""
    day = schedule[schedule["date"] == pd.Timestamp(date)] if len(schedule) else schedule
    rows = []
    gl = {}
    if games is not None and len(games):
        for _, g in games.iterrows():
            gl[(g["away"], g["home"])] = g
    prev_day = pd.Timestamp(date) - pd.Timedelta(days=1)
    played_prev = set(schedule[schedule["date"] == prev_day][["home", "away"]].stack()) if len(schedule) else set()
    for pid in players:
        team = teams_override.get(pid) or state.players.at[pid, "team"]
        g = day[(day["home"] == team) | (day["away"] == team)]
        if g.empty:
            continue
        g = g.iloc[0]
        home = g["home"] == team
        opp = g["away"] if home else g["home"]
        spread = total = np.nan
        gg = gl.get((g["away"], g["home"]))
        if gg is not None:
            hs = float(gg["home_spread"]) if pd.notna(gg.get("home_spread")) else np.nan
            spread = hs if home else -hs
            total = float(gg["total"]) if pd.notna(gg.get("total")) else np.nan
        rows.append({"pid": pid, "team": team, "opp": opp, "home": home, "b2b": team in played_prev,
                     "spread": spread, "total": total, "tip": g["tip"], "game_id": g["game_id"]})
    return pd.DataFrame(rows).set_index("pid") if rows else pd.DataFrame()

def team_rosters(state, overrides, date):
    """Who is on each team now: today's lines/injury list first, then this season's games, then last season."""
    P = state.players
    ros = {}
    recent = P["last_date"] >= pd.Timestamp(date) - pd.Timedelta(days=30)
    known = pd.Series(overrides) if overrides else pd.Series(dtype=object)
    for t in NBA_TEAMS:
        k = list(known[known == t].index) if len(known) else []
        cur_team = P[(P["team"] == t) & recent & ~P.index.isin(known.index)].index.tolist()
        if len(k) >= 6 or (len(k) and not cur_team):
            ros[t] = k + [p for p in cur_team if p not in k]
        else:
            fallback = P[(P["team"] == t) & ~P.index.isin(known.index) & (P["min"] >= 8)].index.tolist() if not cur_team else cur_team
            ros[t] = k + [p for p in fallback if p not in k]
    return ros

def parse_injuries(text):
    out = []
    for part in (text or "").split(";"):
        bits = [x.strip() for x in part.split(":")]
        if len(bits) >= 2 and bits[0]:
            out.append({"player": bits[0], "status": bits[1].title(), "team": norm_team(bits[2]) if len(bits) > 2 else None})
    return pd.DataFrame(out, columns=["player", "status", "team"])

def build_outs(state, inj, overrides):
    """Out/Doubtful players with a real role -> rows used by the usage model."""
    if inj is None or inj.empty:
        return pd.DataFrame(columns=["pid", "team", "name", "pos", "min"])
    P = state.players
    rows = []
    for _, r in inj[inj["status"].isin(["Out", "Doubtful", "Ir", "O", "Suspended"])].iterrows():
        pid = state.find(r["player"])
        if pid is None:
            continue
        team = r["team"] or overrides.get(pid) or P.at[pid, "team"]
        rows.append({"pid": pid, "team": team, "name": P.at[pid, "name"], "pos": P.at[pid, "pos"],
                     "min": P.at[pid, "min"], **{"r_" + s: P.at[pid, "r_" + s] for s in USE_STATS}})
    o = pd.DataFrame(rows)
    return o.set_index("pid", drop=False) if len(o) else pd.DataFrame(columns=["pid", "team", "name", "pos", "min"])

def load_games_file(path):
    if not path or not os.path.exists(path):
        return None
    g = _read_loose_csv(path)
    g["away"] = g["away"].map(norm_team)
    g["home"] = g["home"].map(norm_team)
    for c in ("home_spread", "total"):
        g[c] = pd.to_numeric(g[c].astype(str).str.replace("+", "", regex=False).replace({"PK": "0", "pk": "0"}), errors="coerce")
    return g

# ==================== selection ====================

def guard_games(games, sched, date):
    """Date guard for games.csv (spreads/totals): keep only matchups on D's schedule that haven't tipped. Odds pages
    often serve another day's lines; a wrong-night spread silently moves blowout risk and the game total."""
    if games is None or not len(games) or sched is None or not len(sched):
        return games
    day = sched[sched["date"] == pd.Timestamp(date)]
    tonight = {frozenset((h, w)) for h, w in zip(day["home"], day["away"])}
    ok = pd.Series([frozenset((h, w)) in tonight for h, w in zip(games["home"], games["away"])], index=games.index)
    if (~ok).any():
        bad = ", ".join(f"{w} @ {h}" for h, w in zip(games.loc[~ok, "home"], games.loc[~ok, "away"]))
        print(f"  ! DATE GUARD games.csv: dropped {int((~ok).sum())} of {len(games)} games not on the {pd.Timestamp(date).date()} "
              f"schedule ({bad})" + (" -> spreads page is on another day: re-fetch" if not ok.any() else ""))
    return games[ok]

def _tier(edge, prob, cfg):
    for t in ("tier_1", "tier_2", "tier_3"):
        c = cfg["tiers"][t]
        if edge >= c["min_edge"] and prob >= c["min_prob"]:
            return t
    return None

def _take_if(mu, var, mk, line, side, prob, shrink, cfg):
    c = cfg["tiers"]["tier_3"]
    be = prob - c["min_edge"]
    min_odds = int(decimal_to_american(1 / be)) if 0.02 < be < 0.98 else None
    be110 = implied_prob(-110)
    limit, L = None, line
    po0 = cal_prob(prob_over(mu, var, mk, line))
    for _ in range(30):
        po = cal_prob(prob_over(mu, var, mk, L))
        po = (1 - prob if side == "Under" else prob) + _mw(mk) * (po - po0)  # blended, moved with the line
        pr = po if side == "Over" else 1 - po
        if pr - be110 >= c["min_edge"] and pr >= c["min_prob"]:
            limit = L
            L = L + 1 if side == "Over" else L - 1
            if L < 0:
                break
        else:
            break
    return min_odds, limit

def market_mean(line, p_over, mk, cfg, sdm=1.0):
    """Mean the market implies: the mu at which this stat's distribution (VPE's fitted variance shape) gives
    P(over line) = the main line's no-vig Over chance."""
    a, b = cfg["model"]["var"][mk]
    f = lambda m: prob_over(m, (a * m + b * m * m) * sdm ** 2, mk, line)
    lo, hi = 0.02, max(3.0 * line + 10.0, 5.0)
    if not (f(lo) <= p_over <= f(hi)):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < p_over else (lo, mid)
    return (lo + hi) / 2


def anchored_market_over(mk, line, main, cfg, sdm=1.0):
    """No-vig market chance of Over `line` for a one-sided price (alt line / milestone), read off the player's
    main two-sided line instead of guessing the vig. main: list of (line, no-vig over prob). None if no anchor."""
    if not main:
        return None
    ml, mp = min(main, key=lambda x: abs(x[0] - line))
    if ml == line:
        return mp
    m = market_mean(ml, mp, mk, cfg, sdm)
    if m is None:
        return None
    a, b = cfg["model"]["var"][mk]
    return prob_over(m, (a * m + b * m * m) * sdm ** 2, mk, line)


# ---------- role risk (v2.14): how sturdy is his role? A LABEL, never part of the math ----------
# Two props can both say 60%: one on a locked-in starter, one on a bench player whose minutes swing 12-24 (11/2 Collier).
# Measured on 2024-25 + 2025-26 (~38k player-games): chance he plays under 70% of his projected minutes, by his minutes
# swing over the last 10 games (CV = std / mean). Same pattern both seasons:
#   starter CV<0.15 ~5% | starter CV 0.15-0.3 ~6-10% | starter CV>=0.3 ~15% | bench CV 0.1-0.2 ~11% | bench CV>=0.2 ~14-21%
#   bench projected 3+ min above his last 3 ~18%. STABLE ~5%, OK ~6-11%, SHAKY ~14-21% (or news / absence / new team).
ROLE = {"stable_cv": 0.15, "bench_cv": 0.20, "starter_cv": 0.30, "min_games": 3, "trim": 0.6}

def _rr(r):
    """' | role STABLE' style tag for printed lines (empty when the run predates v2.14)."""
    v = r.get("role_risk") if hasattr(r, "get") else None
    return f" | role {v}" if isinstance(v, str) and v else ""

def role_risk(p):
    why = []
    g = lambda k, d=np.nan: p.get(k, d)
    f_ = lambda v, d: d if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    sr, sd, m10, l3 = f_(g("start_rate"), 1.0), f_(g("min_sd10"), np.nan), f_(g("min_m10"), np.nan), f_(g("min_l3"), np.nan)
    lo, hi, n = f_(g("min_lo10"), np.nan), f_(g("min_hi10"), np.nan), int(f_(g("n_cur"), 0))
    cv = sd / m10 if np.isfinite(sd) and np.isfinite(m10) and m10 > 0 else np.nan
    bench = sr < 0.5
    if bool(g("limited", False)):
        why.append("Limited (minutes cap / role-loss news)")
    if bool(g("returning", False)):
        why.append("back from an absence")
    if str(g("news_note", "") or "").strip() not in ("", "nan", "None"):
        why.append("news role change")
    if bool(g("new_team", False)):
        why.append("new team")
    if n < ROLE["min_games"]:
        why.append(f"only {n} games this season")
    if np.isfinite(cv) and cv >= (ROLE["bench_cv"] if bench else ROLE["starter_cv"]):
        why.append(f"{'bench' if bench else 'starter'}, minutes swing {lo:.0f}-{hi:.0f}")
    if bench and f_(g("bench_trim"), 0.0) >= ROLE["trim"]:
        why.append("projected above his last 3 games")
    swing = f", swing {lo:.0f}-{hi:.0f}" if np.isfinite(lo) else ""
    if why:
        return "SHAKY", "; ".join(why) + " (~15-20% chance of a minutes collapse)"
    if not bench and np.isfinite(cv) and cv < ROLE["stable_cv"]:
        return "STABLE", f"starter{swing} (~5% chance of a minutes collapse)"
    return "OK", f"{'bench' if bench else 'starter'}{swing} (~6-11%)"

def evaluate(props, proj, injuries, cfg):
    unit_pct, kf = cfg["unit_pct_of_bankroll"], cfg["staking"]["kelly_fraction"]
    profile = cfg["strategy_profiles"][cfg["strategy_profile"]]
    shrink = cfg["model"]["prob_calibration_shrink"]
    status = {norm_name(r["player"]): r["status"] for _, r in injuries.iterrows()} if injuries is not None else {}
    rows, skipped = [], []
    mains = {}
    for _, r in props.iterrows():
        if r["market"] is not None and pd.notna(r["pid"]) and pd.notna(r["over_odds"]) and pd.notna(r["under_odds"]):
            mains.setdefault((r["pid"], r["market"]), []).append((float(r["line"]), no_vig(r["over_odds"], r["under_odds"])[0]))
    smax = cfg["staking"].get("straight_max_odds", 200)
    for _, r in props.iterrows():
        mk, pid = r["market"], r["pid"]
        if mk is None:
            skipped.append((r["player"], r["market_raw"], "unknown market"))
            continue
        if pd.isna(pid):
            skipped.append((r["player"], mk, "not in NBA box-score data (rookie / no games)"))
            continue
        if pid not in proj.index:
            skipped.append((r["player"], mk, "no game today for his team (check team / date)"))
            continue
        p = proj.loc[pid]
        stt = status.get(norm_name(r["player"]))
        if stt in ("Out", "Doubtful", "Ir", "Suspended"):
            skipped.append((r["player"], mk, f"injury: {stt}"))
            continue
        if pd.notna(p.get("tip")) and p["tip"] <= pd.Timestamp.now(tz="America/New_York").tz_localize(None) and not cfg.get("ignore_tip"):
            skipped.append((r["player"], mk, "game already started"))
            continue
        line = float(r["line"])
        mu, var = float(p["mu_" + mk]), float(p["var_" + mk])
        raw_over = prob_over(mu, var, mk, line)
        pm_over = cal_prob(raw_over)
        oo, uo = r["over_odds"], r["under_odds"]
        fair_how = "2way"
        if pd.notna(oo) and pd.notna(uo):
            fair_over, _ = no_vig(oo, uo)
        else:
            # one-sided price (alt line / milestone like "6+ assists +250"): books charge far more than 7% there, so
            # read the market's chance off this player's main two-sided line (market-implied mean) instead
            a_, b_ = cfg["model"]["var"][mk]
            sdm = math.sqrt(max(var, 1e-9) / max(a_ * mu + b_ * mu * mu, 1e-9))
            fair_over = anchored_market_over(mk, line, mains.get((pid, mk)), cfg, sdm)
            fair_how = "anchor"
            if fair_over is None:
                fair_how = "1way"
                one = oo if pd.notna(oo) else uo
                if -200 <= one <= 150:  # near-even single side (split lines): ordinary vig, old estimate is fine
                    fair = one_sided_fair(one)
                    fair_over = fair if pd.notna(oo) else 1 - fair
                else:
                    skipped.append((r["player"], mk, f"one-sided {int(one):+d} with no main line to price it against"))
                    continue
        pb_over = blend_leg(pm_over, fair_over, mk, "Over")  # win chance: learned model/market mix
        xj = _json.dumps(nba_x(p, mk, mu, var, oo, uo, why_prob(pm_over, fair_over, mk, fair_how, raw_over)))
        for side, odds, prob, pm in (("Over", oo, pb_over, pm_over), ("Under", uo, 1 - pb_over, 1 - pm_over)):
            if pd.isna(odds):
                continue
            edge = prob - implied_prob(odds)
            xm = (EST_MARGIN if fair_how == "1way" else 0.0) + MARKET_EXTRA.get(mk, 0.0)  # v2.16 est. market + v2.22 market
            tier = None if _killed(mk, side, odds) else _tier(edge - xm, prob, cfg)  # v2.27 KILL: no unboosted bet at -130 or shorter
            if tier == "tier_3" and not profile["allow_tier3"]:
                tier = None
            if tier and odds > smax:  # long shots: tail of the distribution is where the model is least reliable
                tier = None
            if tier and side == "Over" and (bool(p.get("limited", False)) or bool(p.get("returning", False))):
                tier = None  # minutes restriction risk: never back the Over
            units = 0.0
            if tier:
                units = min(kf * kelly_fraction(prob, odds) / unit_pct, cfg["tiers"][tier]["max_units"])
                if stt in ("Questionable", "Gtd", "Game Time Decision", "Probable") and stt != "Probable":
                    units *= cfg["staking"]["questionable_stake_mult"]
                units = round(units * 4) / 4
                if units < 0.25:
                    tier, units = None, 0.0
            mo, ll = _take_if(mu, var, mk, line, side, prob - xm, shrink, cfg)
            rr_ = role_risk(p)
            rows.append({"player": p["name"], "team": p["team"], "opp": p["opp"], "home": p["home"],
                         "game_id": p["game_id"], "market": mk, "side": side, "line": line, "odds": int(odds),
                         "book": r.get("book", "draftkings"), "mean": round(mu, 2), "var": round(var, 3),
                         "proj_min": round(p["proj_min"], 1), "base_min": round(p["min"], 1),
                         "model_prob": round(pm, 3), "blend_prob": round(prob, 3),
                         "market_prob": round(fair_over if side == "Over" else 1 - fair_over, 3),
                         "breakeven": round(implied_prob(odds), 3), "edge": round(edge, 3),
                         "ev_per_unit": round(ev_per_unit(prob, odds), 3), "tier": tier, "units": units,
                         "injury": stt or "", "usage_note": p["usage_note"], "new_team": bool(p["new_team"]),
                         "dvp": round(float(np.prod([p["dvp_" + c] for c in MARKETS[mk] if "dvp_" + c in p])), 3),
                         "pace_f": round(float(p["pace_f"]), 3), "b2b": bool(p["b2b"]), "spread": p["spread"],
                         "n_cur": int(p["n_cur"]), "min_odds": mo, "line_limit": ll,
                         "arch": p.get("arch", ""), "opp_style": p.get("opp_style", ""),
                         "style_f": round(float(np.prod([p["af_" + c] for c in MARKETS[mk] if "af_" + c in p])), 3),
                         "env": round(float(p.get("env", 1.0)), 3), "limited": bool(p.get("limited", False)),
                         "returning": bool(p.get("returning", False)),
                         "role_risk": rr_[0], "role_why": rr_[1], "x": xj, "mkt_est": fair_how == "1way"})
    ev = pd.DataFrame(rows)
    if len(ev):
        ev["market_gap"] = (ev["model_prob"] - ev["market_prob"]).abs().round(3)
    return ev, pd.DataFrame(skipped, columns=["player", "market", "reason"])

def held_mask(ev, cfg):
    m = cfg["model"]
    return ev["tier"].notna() & ((ev["market_gap"] > m["max_market_gap"]) | (ev["base_min"] < m["min_minutes_to_bet"]))

def used_units(log_path, mask_fn, unit_usd=None):
    """Units already staked (straights + parlays, not legs) in the pick log for this week/night, so a second run
    (e.g. Thursday game, then Sunday slate) shares the cap instead of resetting it.
    v2.23 / v15.24: the log stores DOLLARS (Pick Log unit = $1), so the dollars staked are converted back into VPE
    units (bankroll x unit %) before they are compared with the cap. Uses stake_usd when present, else units x $1."""
    if not log_path or not os.path.exists(log_path):
        return 0.0
    d = pd.read_csv(log_path, dtype=str)
    if d.empty:
        return 0.0
    m = mask_fn(d) & (d.get("bet_type", pd.Series("straight", index=d.index)).fillna("straight") != "leg")
    col = "stake_usd" if "stake_usd" in d.columns else "units"
    usd = pd.to_numeric(d.loc[m, col], errors="coerce")
    if col == "stake_usd":
        usd = usd.fillna(pd.to_numeric(d.loc[m, "units"], errors="coerce") * LOG_UNIT_USD) if "units" in d.columns else usd
    else:
        usd = usd * LOG_UNIT_USD
    usd = float(usd.fillna(0).sum())
    return round(usd / unit_usd, 2) if unit_usd else round(usd, 2)

# ---------- locked prices (v2.11 / v15.12): the user's own recording = the price he gets ----------
# When every line comes from the user's DraftKings / FanDuel recording or screenshots (book = draftkings or fanduel,
# not a best-of-market "a/b" pair), VPE treats those prices as fixed: no take-if conditions, no "bet only if the app
# shows X", no TAKE/SKIP step. The edge is computed once, at the recorded price, and a bet either is a pick or isn't.
LOCK = {"on": False, "books": ""}

def detect_lock(books):
    b = pd.Series(books).astype(str).str.lower().str.replace(" ", "", regex=False)
    # v2.33: an "over book/under book" pair made only of his two books (the odds feed's DK + FD merge) is still his exact price
    parts = {x for v in b.unique() for x in str(v).split("/")}
    on = len(b) > 0 and parts <= {"draftkings", "fanduel"}
    LOCK["on"], LOCK["books"] = bool(on), " + ".join(sorted({BOOK_NAME_L.get(x, x) for x in parts})) if on else ""
    return LOCK["on"]

def lock_from(allp):
    """Read the lock flag that `check` saved into bets_all.csv."""
    v = allp.get("locked") if allp is not None and len(allp) else None
    LOCK["on"] = bool(v is not None and v.astype(str).str.lower().isin(["true", "1"]).any())
    LOCK["books"] = str(allp["locked_books"].dropna().iloc[0]) if LOCK["on"] and "locked_books" in allp and allp["locked_books"].notna().any() else ""
    return LOCK["on"]

BOOK_NAME_L = {"draftkings": "DraftKings", "fanduel": "FanDuel"}

def apply_caps(ev, cfg):
    if ev.empty:
        return ev
    bets = ev[ev["tier"].notna() & ~held_mask(ev, cfg)].copy()
    # v2.11: among bets that clear the edge bar at their price, the strongest prop (win chance) gets the cap room first
    bets["score"] = bets["blend_prob"] + 1e-3 * bets["ev_per_unit"] * bets["units"]
    bets = bets.sort_values("score", ascending=False).drop_duplicates(["player", "market"])
    st_ = cfg["staking"]
    cap_day = cfg["strategy_profiles"][cfg["strategy_profile"]]["max_units_per_day"] - cfg.get("used_units", 0.0)
    kept, pp, pg, total, ng = [], {}, {}, 0.0, {}
    for _, b in bets.iterrows():
        if pp.get(b["player"], 0) >= st_["max_props_per_player"]:
            continue
        if pg.get(b["game_id"], 0) + b["units"] > st_["max_units_per_game"] or total + b["units"] > cap_day:
            continue
        if ng.get(b["game_id"], 0) >= st_.get("max_bets_per_game", 99):  # Sun 10/25: 69% of the money rode on one game
            continue
        ng[b["game_id"]] = ng.get(b["game_id"], 0) + 1
        kept.append(b)
        pp[b["player"]] = pp.get(b["player"], 0) + 1
        pg[b["game_id"]] = pg.get(b["game_id"], 0) + b["units"]
        total += b["units"]
    return pd.DataFrame(kept).drop(columns="score") if kept else bets.iloc[0:0]

def build_parlays(picks, cfg, top_n=3):
    pc = cfg["parlays"]
    if not pc["enabled"] or picks.empty:
        return []
    pool = picks[picks["tier"] == "tier_1"].sort_values("ev_per_unit", ascending=False).head(10)
    out = []
    for n in range(2, pc["max_legs"] + 1):
        for combo in itertools.combinations(pool.to_dict("records"), n):
            if len({c["game_id"] for c in combo}) < n:
                continue
            prob, dec = 1.0, 1.0
            for c in combo:
                prob *= c["model_prob"]
                dec *= american_to_decimal(c["odds"])
            ev = prob * (dec - 1) - (1 - prob)
            if ev > 0:
                out.append({"legs": [f'{c["player"]} {c["side"]} {c["line"]} {c["market"]} ({c["odds"]:+d})' for c in combo],
                            "prob": round(prob, 3), "odds": decimal_to_american(dec), "ev_per_unit": round(ev, 3),
                            "units": pc["units"]})
    out.sort(key=lambda x: x["ev_per_unit"], reverse=True)
    return out[:top_n]

# ==================== config ====================

CONFIG = {
  "bankroll": 50.0, "unit_pct_of_bankroll": 0.02, "strategy_profile": "balanced",
  "strategy_profiles": {"conservative": {"max_units_per_day": 5, "allow_tier3": False},
                        "balanced": {"max_units_per_day": 8, "allow_tier3": True},
                        "aggressive": {"max_units_per_day": 12, "allow_tier3": True}},
  "model": {"decay_min": 0.85, "decay_rate": 0.97, "prev_decay": 0.97,
            "prior_min_games": 1, "prior_rate_minutes": 400, "pos_shrink_minutes": 25,
            "dvp_shrink_games": 15, "dvp_prior_games": 10, "prior_trust": 0.5, "dvp_cap": [0.9, 1.1],
            "pace_prior_games": 10, "pace_elasticity": 1.0,
            "usage_min_out_minutes": 12, "usage_rep_factor": 0.85, "team_minutes": 240.7, "min_norm_alpha": 0.5,
            "bench_allowance_max": 35, "norm_mode": "bench", "bench_pivot": 40, "unknown_play_rate": 0.6, "team_total_prior_games": 8, "partial_beta_mult": 0.5, "partial_scale_cap": 1.0, "usage_mult_cap": 1.3,
            "usage_beta": {"default": 0.5}, "wo_min_games": 3, "max_minutes": 40, "bench_l3_pull": 0.2,
            # measured on 2024-25 + 2025-26 regular seasons (~40k player-games): players who suit up on the 2nd night of
            # a back-to-back show no minutes/per-minute drop (the B2B effect is rest DNPs -> injury report), so 1.0;
            # home +0.7% / away -0.7% per-minute scoring; minutes vs final margin for starters (28+ mpg) and bench.
            "situational": {"strength": 1.0, "b2b_min": 1.0, "b2b_rate": 1.0, "home_rate": 1.007, "away_rate": 0.993},
            "blowout": {"margin_sd": 13.0, "edges": [0, 10, 20, 30], "starter": [1.055, 0.997, 0.907, 0.84],
                        "bench": [1.04, 0.955, 0.98, 1.06]},
            # variance = a*mean + b*mean^2, fitted on ~150k 2025-26 backtest player-game-markets
            "var": {"pts": [3.001, 0.0], "reb": [1.309, 0.0095], "ast": [1.202, 0.0257], "fg3m": [1.163, 0.0],
                    "stl": [1.072, 0.0], "blk": [0.993, 0.1438], "tov": [1.018, 0.0], "pra": [2.997, 0.0],
                    "pr": [3.017, 0.0], "pa": [2.866, 0.0], "ra": [1.568, 0.0], "stocks": [1.125, 0.0174]},
            # new team, by games already played with the new team: [0, 1-4, 5-9] -> minutes and per-minute factors
            "new_team": {"min": [0.80, 0.88, 0.88, 0.88, 0.88, 0.94, 0.94, 0.94, 0.94, 0.94],
                         "by_role": {
                             "starter": {"min": [1.0] * 10, "rate": [0.90] * 5 + [0.95] * 5},
                             "rotation": {"min": [0.88] * 5 + [0.94] * 5, "rate": [0.93] * 5 + [0.97] * 5},
                             "bench": {"min": [0.80, 0.88, 0.88, 0.88, 0.88, 0.94, 0.94, 0.94, 0.94, 0.94],
                                       "rate": [0.90, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}},
            # game total (DK, via ESPN): player stat x (market implied team points / base expectation) ^ elasticity,
            # fitted on 8k player-games per market 2025-26 (stable in both halves), shrunk a little
            "total": {"cap": 0.15, "elasticity": {"pts": 0.85, "reb": 1.25, "reb_low": 0.5, "ast": 0.5, "fg3m": 1.25, "stl": 1.0, "blk": 1.5, "tov": 0.0}},
            # back from an absence: actual vs projected minutes by team games missed (2025-26, relative to everyone else)
            "restricted_min_mult": 0.8, "return": {"missed": [1, 2, 3, 5, 8, 13], "mult": [0.97, 0.92, 0.9, 0.89, 0.82, 0.75], "hold_missed": 3},
            "role_weight": 0.4, "wo_min_k": 32,
            "styles": {"k_players": 8, "k_def": 4, "k_off": 4, "min_minutes": 150, "team_prior_games": 12,
                       "prev_weight": 1.0, "shrink_games": 250, "home_shrink_games": 400, "cap": 0.06, "strength": 1.0,
                       # v2.3 two-season test: assists better in BOTH seasons (MAE t=-3.4 2025-26, -3.2 2024-25);
                       # points helped 2025-26 but HURT 2024-25 (t=+4.2) -> dropped; reb/stl/blk hurt -> display only.
                       # 343 continuous trait x defense pairs (slasher x rim protection, PnR x switching, shooter x
                       # threes allowed, ...) tested on top of defense-vs-position: none beat it in both seasons.
                       "apply": ["ast"]},
            "new_team_sd_mult": 1.05, "early_sd_mult": 1.0,
            "prob_calibration_shrink": 0.8, "max_market_gap": 0.10, "min_minutes_to_bet": 20,
            # v2.3 calibration map (see cal_prob): cross-season test 2024-25 <-> 2025-26 log-loss better both ways
            # (t=+3.6 / +7.8); 0-10% bucket 12% -> 4% predicted (actual 4-5%), 80-90% 78-80% -> 84% (actual 82-85%)
            "prob_map": {"x": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
                         "y": [0.0, 0.0243, 0.0275, 0.0517, 0.0607, 0.12, 0.1216, 0.1236, 0.1271, 0.1305, 0.1339, 0.1407, 0.1485, 0.1563, 0.1648, 0.1735, 0.1824, 0.1919, 0.203, 0.2142, 0.2208, 0.2292, 0.2382, 0.2472, 0.2535, 0.2571, 0.276, 0.2954, 0.3047, 0.3208, 0.3319, 0.3421, 0.3508, 0.3596, 0.3693, 0.38, 0.388, 0.396, 0.404, 0.412, 0.42, 0.428, 0.436, 0.444, 0.452, 0.46, 0.468, 0.476, 0.484, 0.492, 0.5, 0.508, 0.516, 0.524, 0.532, 0.54, 0.548, 0.556, 0.564, 0.572, 0.58, 0.588, 0.596, 0.604, 0.612, 0.62, 0.6288, 0.637, 0.6442, 0.6523, 0.6605, 0.6695, 0.6795, 0.6896, 0.6984, 0.7056, 0.7142, 0.7251, 0.7367, 0.7523, 0.7703, 0.7838, 0.797, 0.8101, 0.8247, 0.8288, 0.8406, 0.8524, 0.8588, 0.8637, 0.8734, 0.8824, 0.9037, 0.9109, 0.9134, 0.9187, 0.9324, 0.9461, 0.9798, 1.0, 1.0]}},
  "tiers": {"tier_1": {"min_edge": 0.04, "min_prob": 0.55, "max_units": 3.0},
            "tier_2": {"min_edge": 0.03, "min_prob": 0.52, "max_units": 2.0},
            "tier_3": {"min_edge": 0.025, "min_prob": 0.30, "max_units": 1.0}},
  "staking": {"straight_max_odds": 200, "kelly_fraction": 0.25, "questionable_stake_mult": 0.5, "max_props_per_player": 1, "max_units_per_game": 4,
              "max_bets_per_game": 2},
  "parlays": {"enabled": False, "max_legs": 3, "units": 0.5},
  # hold on one-sided "X+" prices with no main line to anchor them (placeholder until `vig` measures it: the old flat
  # 7% is kept for near-even prices; long shots carry more hold at every book we've seen -> assumed higher)
  # v2.25: MEASURED on 4,824 graded one-sided NBA lines with no main line (Shadow tab through 11/15): the price implied
  # 52.9%, 47.7% hit -> real hold x0.107 (0.102 at -200..-130, 0.113 at -130..+100, 0.107 at +100..+150). The old 0.07 was
  # an assumption and made VPE's "market chance" ~1.6 pts too high on every ladder-only prop (fake edge, inflated boost EV).
  "one_sided_hold": {"short": 0.105, "mid": 0.105, "long": 0.15, "short_to": -200, "long_from": 150},
}

def _cfg(a):
    import copy
    c = copy.deepcopy(CONFIG)
    if getattr(a, "bankroll", None):
        c["bankroll"] = a.bankroll
    if getattr(a, "profile", None):
        c["strategy_profile"] = a.profile
    return c

# ==================== CLI helpers ====================

def _read_loose_csv(path):
    keep = [l for l in open(path, encoding="utf-8").read().splitlines()
            if l.strip() and not l.strip().startswith("```") and not l.startswith(("MARKET=", "LINKS=", "FILTERS=", "#"))]
    return pd.read_csv(io.StringIO("\n".join(keep)), dtype=str).fillna("")

def _setup(a, cfg):
    date = pd.Timestamp(a.date) if getattr(a, "date", None) else today_et()
    season = season_for(date)
    cur, prev = load_box(season), load_box(season - 1, max_age_hours=24 * 30)
    prev2 = load_box(season - 2, max_age_hours=24 * 30)
    sched = load_schedule(season)
    state = State(cur, prev, date, cfg, prev2_box=prev2)
    state.pv_box = prev
    n_games = cur["game_id"].nunique() if len(cur) else 0
    last = cur["date"].max().date() if len(cur) else "none yet"
    print(f"DATA [{VERSION}]: season {season - 1}-{str(season)[2:]}, slate date {date.date()}, this-season box scores: {n_games} games "
          f"(through {last}); last season loaded as prior; schedule: {len(sched)} games")
    return date, state, sched

def _read_props(paths):
    frames = []
    for path in paths:
        d = _read_loose_csv(path)
        if "book" not in d:
            d["book"] = "draftkings"
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["market_raw"] = d["market"]
    d["market"] = d["market"].map(canon_market)
    d["line"] = pd.to_numeric(d["line"], errors="coerce")
    d["over_odds"] = d["over_odds"].map(parse_odds)
    d["under_odds"] = d["under_odds"].map(parse_odds)
    d = d[d["line"].notna() & (d["over_odds"].notna() | d["under_odds"].notna())]
    if "matchup" in d:
        # the matchup names BOTH teams; which one is his is decided later from his box scores (_prepare).
        # (before v2.2 every player got the AWAY team here -> home players were scored as road players on a
        # "new team" against their own team: ~20% too low, fake Unders)
        mm = d["matchup"].map(parse_matchup)
        d["team"] = [(x[0], x[1]) if x[0] and x[1] else None for x in mm]
    else:
        d["team"] = None
    return d.reset_index(drop=True)

def _prepare(state, sched, date, props_players, injuries, games, cfg, sure_names=None):
    """Map names -> ids, current teams, today's games, outs. Returns (proj, overrides, outs)."""
    overrides = {}
    pids = []
    inj_team = {}
    for _, r in (injuries if injuries is not None else pd.DataFrame()).iterrows():
        if r.get("team"):
            inj_team[norm_name(r["player"])] = r["team"]
    mism = []
    for name, team in props_players:
        pid = state.find(name)
        if pid is None:
            continue
        if isinstance(team, tuple):
            cur = state.players.at[pid, "team"]
            if norm_name(name) in inj_team or cur in team:
                pids.append(pid)
            elif name not in mism:
                mism.append(name)
                print(f"  ! TEAM CHECK: {name} is listed in {team[0]} @ {team[1]} but his last games were for {cur} "
                      f"-> skipped (traded? add '{name}:Active:TEAM' to the injuries string to set his team)")
        else:
            pids.append(pid)
            if team:
                overrides[pid] = team
    for _, r in (injuries if injuries is not None else pd.DataFrame()).iterrows():
        pid = state.find(r["player"])
        if pid is not None and r.get("team"):
            overrides[pid] = r["team"]
    sl = build_slate(state, sched, date, sorted(set(pids)), overrides, games)
    outs = build_outs(state, injuries, overrides)
    rosters = team_rosters(state, overrides, date)
    sure = None if sure_names is None else {p for p in (state.find(n) for n in sure_names) if p is not None}
    lim = set()
    if injuries is not None and len(injuries):
        for _, r in injuries[injuries["status"].isin(["Limited", "Restricted", "Minutes Restriction"])].iterrows():
            pid = state.find(r["player"])
            if pid is not None:
                lim.add(pid)
    news = {}
    if injuries is not None and len(injuries):
        for _, r in injuries.iterrows():
            st_ = str(r["status"]).strip().lower()
            kind_ = "start" if st_ in ("starting", "start", "starter") else "bench" if st_ in ("bench", "reserve", "off bench") else None
            mm_ = re.match(r"(cap|mins|minutes)\s*(\d+)", st_)
            if kind_ is None and not mm_:
                continue
            pid = state.find(r["player"])
            if pid is not None:
                news[pid] = ("cap", float(mm_.group(2))) if mm_ else (kind_, None)
    proj = project(state, sl, outs, rosters, cfg, sure=sure, limited=lim, news=news) if len(sl) else pd.DataFrame()
    if len(proj) and "news_note" in proj and "usage_note" in proj:
        proj["usage_note"] = [(u + ("; " if u and n else "") + n) if isinstance(u, str) else n
                              for u, n in zip(proj["usage_note"].fillna(""), proj["news_note"].fillna(""))]
    return proj, overrides, outs

# ==================== commands ====================

def cmd_board(a):
    cfg = _cfg(a)
    date, state, sched = _setup(a, cfg)
    inj = parse_injuries(a.injuries)
    games = guard_games(load_games_file(a.games), sched, date)
    day = sched[sched["date"] == date]
    teams = set(day["home"]) | set(day["away"])
    if a.team:
        teams &= {norm_team(t) for t in a.team.split(",")}
    P = state.players
    props_players = []
    if a.props:
        pr = _read_props([a.props])
        props_players = list(zip(pr["player"], pr["team"]))
    played_now = P["n_cur"] > 0
    has_games = P["team"].map(lambda x: state.n_team.get(x, 0) > 0)
    pids = [(n, None) for n in P[(P["team"].isin(teams)) & (P["min"] >= 10) & (played_now | ~has_games)]["name"]]
    # board: only players with lines are "sure to play"; everyone else plays at his recent availability rate
    # (before v2.1.4 every roster player counted as playing, which cut each player's minutes ~10-50%)
    proj, _, outs = _prepare(state, sched, date, props_players + pids, inj, games, cfg,
                             sure_names=[n for n, _ in props_players])
    if proj.empty:
        print("No players with a game on this date.")
        return
    proj = proj[proj["team"].isin(teams) & ~proj.index.isin(set(outs["pid"]) if len(outs) else set())]
    if a.player:
        keys = {norm_name(x) for x in a.player.split(";")}
        proj = proj[proj["name"].map(norm_name).isin(keys)]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 80)
    cols = ["name", "team", "opp", "pos", "arch", "opp_style", "proj_min", "mu_pts", "mu_reb", "mu_ast", "mu_fg3m", "mu_pra",
            "dvp_pts", "af_pts", "env", "pace_f", "new_team", "usage_note"]
    print(proj.sort_values(["team", "proj_min"], ascending=[True, False])[cols].round(2).to_string(index=False))

def cmd_styles(a):
    """Team categories (offense + defense style) and player archetypes, plus how each archetype does vs each defense."""
    cfg = _cfg(a)
    date, state, sched = _setup(a, cfg)
    if not len(state.dstyle):
        print("STYLES: last season not loaded -> no groups")
        return
    T = pd.DataFrame({"offense": state.ostyle, "defense": state.dstyle})
    print("=== TEAM STYLES (this season, shrunk toward last season) ===")
    for d, g in T.groupby("defense"):
        print(f"  DEF {d:22s}: " + ", ".join(f"{t} ({o})" for t, o in g["offense"].items()))
    P = state.players
    arch = state.arch
    teams = {norm_team(t) for t in a.team.split(",")} if a.team else None
    print("\n=== PLAYER ARCHETYPES (top players by minutes) ===")
    for n, g in arch.groupby(arch):
        pl = P.loc[g.index]
        if teams:
            pl = pl[pl["team"].isin(teams)]
        top = pl[pl["last_date"] >= date - pd.Timedelta(days=30)].sort_values("min", ascending=False).head(8)
        if len(top):
            print(f"  {n:18s}: " + ", ".join(f"{r['name']} ({r['team']})" for _, r in top.iterrows()))
    stat = a.stat or "pts"
    if stat in state.mf:
        f = state.mf[stat].unstack("home")
        f.columns = ["road" if not c else "home" for c in f.columns]
        print(f"\n=== {stat.upper()}: archetype vs defensive style (x1.03 = 3% more than the base model expects) ===")
        print(f.round(3).to_string())

def cmd_slate(a):
    """Rank today's players (from the points page) by projected minutes; print the per-player pages to pull."""
    cfg = _cfg(a)
    date, state, sched = _setup(a, cfg)
    d = _read_loose_csv(a.src)
    d["team"] = d["matchup"].map(lambda m: parse_matchup(m)[0])
    proj, _, _ = _prepare(state, sched, date, list(zip(d["player"], d["team"])), parse_injuries(a.injuries), None, cfg)
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    rows = []
    for _, r in d.iterrows():
        pid = state.find(r["player"])
        if pid is None or pid not in proj.index:
            rows.append((r["player"], r["matchup"], np.nan, "", r.get("url", ""), "not in data / no game"))
            continue
        p = proj.loc[pid]
        started = pd.notna(p["tip"]) and p["tip"] <= now
        rows.append((r["player"], r["matchup"], round(p["proj_min"], 1), str(p["tip"])[11:16], r.get("url", ""),
                     "STARTED" if started else ""))
    out = pd.DataFrame(rows, columns=["player", "matchup", "proj_min", "tip", "url", "note"])
    ok = out[(out["note"] == "") & (out["proj_min"] >= cfg["model"]["min_minutes_to_bet"] - 2)]
    ok = ok.sort_values("proj_min", ascending=False).head(a.top)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(f"PULL THESE {len(ok)} PLAYER PAGES (DraftKings lines):")
    print(ok.to_string(index=False))
    skip = out[out["note"] != ""]
    if len(skip):
        print("\nnot pulled:", "; ".join(f"{r.player} ({r.note})" for r in skip.itertuples()))

def merge_best(df):
    """Several odds sources for the same prop -> one row per player/market/line with the best Over and best Under
    price found anywhere (line shopping across every source that passed the slate guard)."""
    if df.empty:
        return df
    d = df.copy()
    d["_k"] = d["player"].map(norm_name)
    out = []
    for (_, mk, ln), g in d.groupby(["_k", "market", "line"], sort=False):
        best = {}
        for side, col in (("o", "over_odds"), ("u", "under_odds")):
            cand = [(american_to_decimal(float(v)), v, side_book(b, "Over" if side == "o" else "Under") if "/" in str(b) else str(b))
                    for v, b in zip(g[col], g["book"]) if str(v).strip() not in ("", "nan")]
            best[side] = max(cand) if cand else None
        r = g.iloc[0]
        ob = best["o"][2] if best["o"] else ""
        ub = best["u"][2] if best["u"] else ""
        out.append({"player": r["player"], **({"matchup": r["matchup"]} if "matchup" in g else {}), "market": mk, "line": ln,
                    "over_odds": best["o"][1] if best["o"] else "", "under_odds": best["u"][1] if best["u"] else "",
                    "book": f"{ob}/{ub}" if ob and ub else (ob or ub)})
    m = pd.DataFrame(out, columns=list(df.columns))
    if len(m) < len(df):
        print(f"MERGE: {len(df)} rows from all sources -> {len(m)} props, best Over and best Under price kept for each")
    return m

def cmd_lines(a):
    out = []
    for spec in a.src:
        mk, path = spec.split("=", 1)
        if canon_market(mk) is None:
            print(f"  ! unknown market {mk}")
            continue
        d = _read_loose_csv(path)
        for _, r in d.iterrows():
            ol, oo, ob = r.get("over_line", ""), r.get("over_odds", ""), r.get("over_book", "")
            ul, uo, ub = r.get("under_line", ""), r.get("under_odds", ""), r.get("under_book", "")
            mu = r.get("matchup", "")
            if ol and ul and oo and uo and float(ol) == float(ul):
                out.append((r["player"], mu, mk, ol, oo, uo, f"{ob}/{ub}"))
            else:
                if ol and oo:
                    out.append((r["player"], mu, mk, ol, oo, "", ob))
                if ul and uo:
                    out.append((r["player"], mu, mk, ul, "", uo, ub))
    df = pd.DataFrame(out, columns=["player", "matchup", "market", "line", "over_odds", "under_odds", "book"])
    # slate guard: odds pages sometimes still show another night's games; keep only matchups on date D's schedule
    date = pd.Timestamp(a.date) if getattr(a, "date", None) else today_et()
    day = load_schedule(season_for(date))
    day = day[day["date"] == date] if len(day) else day
    if len(day) == 0:
        print(f"  ! SLATE GUARD: no games on the schedule for {date.date()}; rows kept unchecked")
    else:
        games = {frozenset((h, w)) for h, w in zip(day["home"], day["away"])}
        teams = [parse_matchup(m)[:2] for m in df["matchup"]]
        ok = pd.Series([bool(t[0] and t[1] and frozenset(t) in games) for t in teams], index=df.index)
        for mk, g in df.groupby("market"):
            k, n = int(ok[g.index].sum()), len(g)
            if k < n:
                print(f"  ! SLATE GUARD {mk}: dropped {n - k} of {n} rows not on the {date.date()} schedule"
                      + (" -> page is still on another night (props pages roll over on their own): wait ~5 min, re-fetch" if k == 0 else ""))
        df = df[ok]
        # coverage: how many of tonight's games each market has lines for (stale page = 0)
        cov = {mk: len({frozenset(parse_matchup(m)[:2]) for m in g["matchup"]}) for mk, g in df.groupby("market")}
        srcs = list(dict.fromkeys(s.split("=", 1)[0] for s in a.src))
        print(f"COVERAGE ({len(games)} games on {date.date()}): "
              + " | ".join(f"{mk} {cov.get(canon_market(mk) or mk, 0)}/{len(games)}" for mk in srcs)
              + ("  -> STALE: " + ", ".join(mk for mk in srcs if not cov.get(canon_market(mk) or mk))
                 + " (re-fetch; if still stale, leave out and say so)" if any(not cov.get(canon_market(mk) or mk) for mk in srcs) else ""))
    df = merge_best(df)
    df.to_csv(a.out, index=False)
    print(f"LINES: {len(df)} rows -> {a.out} | " + ", ".join(f"{k} {v}" for k, v in df["market"].value_counts().items()))

def find_middles(props, proj, cfg, max_cost=0.12):
    """MIDDLES (the real hedge): an Over at a low line at one book + an Under at a higher line at another book.
    Lands between the two lines -> both win; anywhere else -> one wins, one loses (small cost). Ranked by EV per $1 a
    side using VPE's projection (shrunk like every pick) for the chance of landing in the window."""
    shrink = cfg["model"]["prob_calibration_shrink"]
    rows = []
    p = props[props["pid"].notna() & props["market"].notna()]
    for (pid, mk), g in p.groupby(["pid", "market"]):
        if pid not in proj.index:
            continue
        ov = g[g["over_odds"].notna()][["line", "over_odds", "book"]]
        un = g[g["under_odds"].notna()][["line", "under_odds", "book"]]
        if ov.empty or un.empty:
            continue
        mu, var = float(proj.at[pid, "mu_" + mk]), float(proj.at[pid, "var_" + mk])
        for _, o in ov.iterrows():
            for _, u in un.iterrows():
                l1, l2 = float(o["line"]), float(u["line"])
                if l2 - l1 < 1:
                    continue
                d1, d2 = american_to_decimal(o["over_odds"]), american_to_decimal(u["under_odds"])
                worst = min(d1 - 2, d2 - 2)  # one side wins, the other loses ($1 each)
                if worst < -max_cost:
                    continue
                p_hi = cal_prob(prob_over(mu, var, mk, l2 - 0.0))   # over the high line
                p_lo = cal_prob(prob_over(mu, var, mk, l1))         # over the low line
                p_mid = max(p_lo - p_hi, 0.0)
                p_below = 1 - p_lo
                ev = p_mid * (d1 + d2 - 2) + p_hi * (d1 - 2) + p_below * (d2 - 2)
                ob = side_book(o["book"], "Over") if "/" in str(o["book"]) else str(o["book"])
                ub = side_book(u["book"], "Under") if "/" in str(u["book"]) else str(u["book"])
                rows.append({"player": proj.at[pid, "name"], "market": mk, "over": f"O{l1:g} {int(o['over_odds']):+d} {ob}",
                             "under": f"U{l2:g} {int(u['under_odds']):+d} {ub}", "window": f"{math.floor(l1) + 1:g}-{math.ceil(l2) - 1:g}",
                             "p_middle": round(p_mid, 3), "worst_case": round(worst, 3), "ev_per_2": round(ev, 3), "mean": round(mu, 1)})
    m = pd.DataFrame(rows)
    return m.sort_values("ev_per_2", ascending=False).drop_duplicates(["player", "market"]) if len(m) else m

def cmd_check(a):
    cfg = _cfg(a)
    set_calib(getattr(a, "calib", None), SPORT)
    if a.ignore_tip:
        cfg["ignore_tip"] = True
    date, state, sched = _setup(a, cfg)
    props = _read_props(a.props)
    if detect_lock(props["book"]):
        print(f"PRICES LOCKED: every line is from your {LOCK['books']} recording/screenshots -> priced as recorded; every pick still shows its take-if (worst price still worth it) in case the price moves before you bet")
    inj = parse_injuries(a.injuries)
    games = guard_games(load_games_file(a.games), sched, date)
    day = sched[sched["date"] == date] if len(sched) else sched
    if len(day):
        tonight = {frozenset((h, w)) for h, w in zip(day["home"], day["away"])}
        have = {frozenset((h, w)) for h, w in zip(games["home"], games["away"])} if games is not None else set()
        n = len(tonight & have)
        print(f"SPREADS: {n}/{len(tonight)} of tonight's games have a spread"
              + ("" if n == len(tonight) else " -> missing games get no blowout adjustment"
                 + (" (spreads page looks stale: re-fetch with ?date=D)" if n == 0 else "")))
    if len(day) and props["team"].map(lambda t: isinstance(t, tuple)).any():
        # date guard on the props file itself (screenshots / recordings go straight to check, not through `lines`)
        ok = props["team"].map(lambda t: not isinstance(t, tuple) or frozenset(t) in tonight)
        if (~ok).any():
            print(f"  ! DATE GUARD props: dropped {int((~ok).sum())} of {len(props)} rows whose matchup isn't on the "
                  f"{date.date()} schedule ({', '.join(sorted({f'{t[0]} @ {t[1]}' for t in props.loc[~ok, 'team']}))})")
        props = props[ok].reset_index(drop=True)
    proj, overrides, outs = _prepare(state, sched, date, list(zip(props["player"], props["team"])), inj, games, cfg)
    props["pid"] = [state.find(n) for n in props["player"]]
    print(f"USAGE MODEL: {len(outs)} players Out/Doubtful with a role -> minutes and usage redistributed"
          + (": " + ", ".join(f"{r['name']} ({r['team']}, {r['min']:.0f} min)" for _, r in outs.iterrows()) if len(outs) else ""))
    cfg["used_units"] = used_units(a.log, lambda d: d.get("date", pd.Series("", index=d.index)).astype(str).str.lstrip("'") == str(date.date()),
                                   unit_usd=cfg["bankroll"] * cfg["unit_pct_of_bankroll"])
    cap = cfg["strategy_profiles"][cfg["strategy_profile"]]["max_units_per_day"]
    print(f"CAP: {cfg['used_units']:g}u already staked on {date.date()} (pick log) -> {max(cap - cfg['used_units'], 0):g}u of {cap}u left tonight")
    ev, skipped = evaluate(props, proj, inj, cfg)
    picks = apply_caps(ev, cfg)
    parlays = build_parlays(picks, cfg)
    unit = cfg["bankroll"] * cfg["unit_pct_of_bankroll"]
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 110)
    print(f"BANKROLL ${cfg['bankroll']:.2f} | 1u = ${unit:.2f} | profile {cfg['strategy_profile']} | projections pure stats; win chance = learned model/market mix (CALIB)")
    print("\n=== BETS (after caps) ===")
    if not picks.empty:
        print(picks[["tier", "player", "team", "opp", "market", "side", "line", "odds", "book", "mean", "proj_min",
                     "model_prob", "blend_prob", "edge", "units", "min_odds", "line_limit", "role_risk"]].to_string(index=False))
        print("\nROLE RISK (label only, not in the math): " + " | ".join(f"{r['player']}: {r['role_risk']} - {r['role_why']}" for _, r in picks.iterrows()))
        print("\n" + picks[["player", "market", "dvp", "pace_f", "b2b", "spread", "new_team", "usage_note"]].to_string(index=False))
    else:
        print("none")
    if True:  # v2.16: take-if on recordings too (prices move before bets go in)
        print("  take-if: DK line no worse than line_limit (Over <=, Under >=) AND DK odds at least min_odds.")
    if not ev.empty:
        held = ev[held_mask(ev, cfg)].sort_values("market_gap", ascending=False).drop_duplicates(["player", "market"])
        print(f"\n=== HELD ({len(held)}): model vs market > {cfg['model']['max_market_gap']:.0%} or < "
              f"{cfg['model']['min_minutes_to_bet']} base minutes - judgment only ===")
        if len(held):
            print(held[["player", "market", "side", "line", "odds", "mean", "base_min", "model_prob", "market_prob",
                        "new_team", "usage_note"]].head(25).to_string(index=False))
        near = ev[ev["tier"].isna() & (ev["edge"] > 0)].sort_values("edge", ascending=False).head(6)
        print("\n=== CLOSEST PASSES ===")
        print(near[["player", "market", "side", "line", "odds", "mean", "model_prob", "edge"]].to_string(index=False) if len(near) else "none")
    print("\n(parlays: built by the parlay command)")
    for p in parlays:
        print(f"  {p['odds']:+d} | prob {p['prob']:.1%} | EV/u {p['ev_per_unit']:+.3f} | {p['units']}u | " + " + ".join(p["legs"]))
    mids = find_middles(props, proj, cfg) if len(proj) else pd.DataFrame()
    print("\n=== MIDDLES (Over low line + Under high line at different books; both win inside the window) ===")
    print(mids.head(8).to_string(index=False) if len(mids) else "none tonight")
    print("\n=== SKIPPED ===")
    print(skipped.drop_duplicates().to_string(index=False) if not skipped.empty else "none")
    if a.out:
        picks.assign(date=str(date.date())).to_csv(a.out, index=False)
        ev = ev.assign(locked=LOCK["on"], locked_books=LOCK["books"])
        ev.assign(date=str(date.date())).to_csv(a.out.replace(".csv", "_all.csv"), index=False)
        # run-stamped copy: a later rerun overwrites bets_all.csv, but `log` still finds bets from this run (10/26: 4 bets
        # from a replaced 6:00 slate were logged "unscored")
        os.makedirs("runs", exist_ok=True)
        ev.assign(date=str(date.date())).to_csv(f"runs/bets_all_{date.date()}_{pd.Timestamp.now(tz='America/New_York'):%H%M%S}.csv", index=False)
        print(f"\nsaved bets -> {a.out} (all lines -> {a.out.replace('.csv', '_all.csv')})")
    write_shadow(ev, str(date.date()), a.shadow_out, getattr(a, "shadow_saved", None))

# ---------- logging: VPE picks vs the user's own picks, profit boosts, parlays ----------
def _odds(x):
    s = str(x).strip().lower().replace(" ", "")
    if s in ("", "nan", "none"):
        return None
    if s in ("even", "ev", "evens"):
        return 100
    try:
        return int(float(s))
    except ValueError:
        return None

def _num(x):
    try:
        v = float(str(x).replace("%", "").strip())
        return 0.0 if math.isnan(v) else v
    except (TypeError, ValueError):
        return 0.0

def game_player(mk, name):
    """v2.30: 'Celtics' -> 'BOS' for spread / ml; 'Knicks @ Celtics' / 'NY vs BOS' -> 'NY@BOS' for a total."""
    s_ = str(name).strip()
    if mk == "total":
        parts = [p.strip() for p in re.split(r"\s*@\s*|\s+at\s+|\s+vs\.?\s+|\s+v\s+", s_) if p.strip()]
        return "@".join(norm_team(p) for p in parts) if len(parts) == 2 else s_
    return norm_team(s_)

def boosted_american(odds, boost_pct):
    """DraftKings profit boost: winnings x (1 + boost%). Returns the effective American odds."""
    if not boost_pct:
        return int(odds)
    return int(decimal_to_american(1 + (american_to_decimal(odds) - 1) * (1 + boost_pct / 100)))

SLIP_MEDAL = {}


def slip_keys(path):
    """(player, market, side) of every bet on the slip the last `check` produced; None if no slip was saved."""
    if not path or not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    kinds = s["kind"] if "kind" in s else ["pick"] * len(s)
    if "medal" in s:   # v2.25: Gold (#1-3) / Silver (#4-6) of that run's leg ranking -> stamped on the logged row
        SLIP_MEDAL.update({(norm_name(p), canon_market(m) or m, str(sd).title()): str(md) for p, m, sd, md in
                           zip(s["player"], s["market"], s["side"], s["medal"]) if str(md) in ("Gold", "Silver")})
    return {(norm_name(p), canon_market(m) or m, str(sd).title()): str(k) for p, m, sd, k in zip(s.get("player", []),
            s.get("market", []), s.get("side", []), kinds)}

def rate_pick(prob, odds, eff_odds, cfg, suggested):
    """VPE's verdict at the price actually taken. Returns (tier at effective price, verdict, edge at base, edge at eff)."""
    eb, ee = prob - implied_prob(odds), prob - implied_prob(eff_odds)
    tb, te = _tier(eb, prob, cfg), _tier(ee, prob, cfg)
    if suggested:
        v = "VPE pick"
    elif tb:
        v = "VPE-grade, not on slip" if suggested is False else "VPE-grade (no slip saved to compare)"
    elif te:
        v = "+EV only with boost"
    elif ee > 0:
        v = "small +EV, below tiers"
    else:
        v = "-EV at price taken"
    return te, v, round(eb, 3), round(ee, 3)

SGP_RATINGS = []  # v2.21: ratings the sgp command saved tonight (sgp_ratings.json), matched to logged SGPs

def build_log_rows(placed, score, cfg, unit, slip, stamp):
    """placed.csv: player,market,side,line,odds,stake[,boost_pct,parlay_id,parlay_odds,note].
    Straight bets: one row each. Parlays: one row per leg (same parlay_id; odds = that leg's odds if shown, stake =
    the parlay stake, parlay_odds = the parlay's total odds). score(player, market, side, line, odds) ->
    ({"prob","team","opp"} or None, reason)."""
    placed = placed.copy()
    for c in ("boost_pct", "parlay_id", "parlay_odds", "note"):
        if c not in placed:
            placed[c] = np.nan
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    tag = "-".join(str(v) for v in stamp.values())
    rows = []

    def leg_or_straight(b, as_leg):
        mk = canon_market(b["market"])
        if mk in GAME_MK:   # v2.30: team markets -> one spelling of the team / matchup, side Over = that team
            b = b.copy()
            b["player"] = game_player(mk, b["player"])
            if mk != "total":
                b["side"] = "Over"
            elif str(b["side"]).strip().lower() in ("o", "u"):
                b["side"] = "Over" if str(b["side"]).strip().lower() == "o" else "Under"
            if mk == "ml" or str(b["line"]).strip() in ("", "nan"):
                b["line"] = "0"
        side, line, odds = str(b["side"]).title(), float(b["line"]), _odds(b["odds"])
        kind = None if slip is None else slip.get((norm_name(b["player"]), mk, side))
        sug = None if slip is None else kind is not None
        s, why = score(b["player"], mk, side, line, odds if odds is not None else -110) if mk else (None, "unknown market")
        note = "" if pd.isna(b["note"]) else str(b["note"])
        md_ = SLIP_MEDAL.get((norm_name(b["player"]), mk, side))
        if md_ and f"[{md_}]" not in note:
            note = (note + f" [{md_}]").strip()
        r = {**stamp, "logged_at": now, "player": b["player"], "market": mk or b["market"], "side": side, "line": line,
             "odds": odds, "book": _book(b), "note": note,
             "suggested": "unknown" if sug is None else ("lean" if kind == "lean" else ("yes" if sug else "no"))}
        if s:
            r.update({"team": s["team"], "opp": s["opp"], "model_prob": round(s["prob"], 3), "blend_prob": round(s["prob"], 3)})
        if as_leg:
            r.update({"bet_type": "leg", "parlay_id": f"{tag}-{b['parlay_id']}", "units": 0.0, "stake_usd": 0.0,
                      "vpe_verdict": "parlay leg" if s else "parlay leg (unscored: " + why + ")"})
            if s and odds is not None:
                r.update({"edge": round(s["prob"] - implied_prob(odds), 3), "tier": _tier(s["prob"] - implied_prob(odds), s["prob"], cfg) or "off_model"})
            return r, (s or {}).get("leg_prob", (s or {}).get("prob")), sug, s
        stake, boost = float(b["stake"]), _num(b["boost_pct"])
        r.update({"bet_type": "straight", "units": round(stake / unit, 2), "stake_usd": stake,
                  "boost_pct": boost or np.nan, "face_usd": _face(b), "bonus_bet": _bonus(b),
                  "rec_units": LEAN["units"] if kind == "lean" else ((s or {}).get("rec_units", 0.0) if sug else 0.0)})
        if odds is None:
            r.update({"tier": "unscored", "source": "unscored: odds unreadable", "vpe_verdict": "unscored"})
            return r, None, sug, s
        eff = boosted_american(odds, boost)
        r["odds_eff"] = eff
        if not s:
            r.update({"tier": "unscored", "source": "unscored: " + why, "vpe_verdict": "unscored"})
        else:
            te, v, eb, ee = rate_pick(s["prob"], odds, eff, cfg, sug)
            if kind == "lean":
                v = "VPE lean (best available, no edge)" if not te else "VPE lean (+EV at the price taken)"
            if s.get("held") and not sug:   # v2.35: rule order step 1, HELD beats any price or boost
                te, v = None, f"HELD: not a VPE play ({s['held']})"
            mo = s.get("min_odds")
            if mo is not None and pd.notna(mo) and american_to_decimal(odds) < american_to_decimal(mo) - 1e-9:
                v = (f"taken below take-if: {int(odds):+d} vs {int(mo):+d} (price moved after the recording)" if LOCK["on"]
                     else f"SKIP rule broken: {int(odds):+d} worse than take-if {int(mo):+d}")
                r["note"] = (r["note"] + " [below take-if]").strip()
            r.update({"tier": te or "off_model", "source": "engine" if te else "off_model", "vpe_verdict": v,
                      "edge_base": eb, "edge": ee, "ev_per_unit": round(ev_per_unit(s["prob"], eff), 3)})
        return r, None, sug, s

    pid = placed["parlay_id"].astype(str).str.strip()
    is_leg = placed["parlay_id"].notna() & (pid != "")
    for _, b in placed[~is_leg].iterrows():
        rows.append(leg_or_straight(b, False)[0])
    for p, legs in placed[is_leg].groupby(pid[is_leg]):
        out = [leg_or_straight(b, True) for _, b in legs.iterrows()]
        rows += [o[0] for o in out]
        b0 = legs.iloc[0]
        podds, stake, boost = _odds(b0["parlay_odds"]), float(b0["stake"]), _num(b0["boost_pct"])
        probs = [o[1] for o in out]
        teams = [frozenset((o[3]["team"], o[3]["opp"])) for o in out if o[3]]
        same_game = len(teams) != len(set(teams))
        if same_game:  # v2.16: legs of a same-game parlay carry the parlay's label (rate-only SGP)
            for o in out:
                o[0]["suggested"] = "sgp"
        # a placed parlay is a VPE pick only if it is EXACTLY a parlay VPE built tonight (same legs, nothing added/removed)
        lk = frozenset((norm_name(o[0]["player"]), o[0]["market"], o[0]["side"]) for o in out)
        vpe_par = next((v for v in VPE_PARLAYS if frozenset((norm_name(x[0]), x[1], x[2]) for x in v["legs"]) == lk), None)
        # v2.23 / v15.24: a BOOST-built parlay placed without its boost (or at the other book) is NOT the VPE pick: its
        # edge was the boost (11/16: unboosted FD copy of a boost build was logged "suggested = yes")
        boost_unused = None
        if vpe_par and _num(vpe_par.get("boost_pct")):
            _vb, _pb = float(_num(vpe_par.get("boost_pct"))), _num(b0["boost_pct"]) if "boost_pct" in b0 else None
            _bk_ok = BOOK_NAME.get(vpe_par.get("book"), str(vpe_par.get("book"))).lower() == _book(b0).lower()
            if not _pb or float(_pb) + 1e-9 < _vb or not _bk_ok:
                boost_unused = (f"VPE legs, boost not used (built for {_vb:g}% boost at "
                                f"{BOOK_NAME.get(vpe_par.get('book'), vpe_par.get('book'))}; placed "
                                + (f"{float(_pb):g}% boost" if _pb else "no boost") + f" at {_book(b0)})")
                vpe_par = None
        r = {**stamp, "logged_at": now, "bet_type": "parlay", "parlay_id": f"{tag}-{p}", "market": "parlay",
             "player": " + ".join(f"{o[0]['player']} {o[0]['side']} {o[0]['line']:g} {o[0]['market']}" for o in out),
             "side": np.nan, "odds": podds, "book": _book(b0), "units": round(stake / unit, 2),
             "stake_usd": stake, "boost_pct": boost or np.nan, "face_usd": _face(b0), "bonus_bet": _bonus(b0),
             "suggested": ("lean" if vpe_par.get("lean") else "yes") if vpe_par else "no",
             "rec_stake_usd": vpe_par["stake_usd"] if vpe_par else 0.0,
             "note": ("same-game legs: model treats them as independent, real odds of all hitting differ. " if same_game else "")
                     + ("" if pd.isna(b0["note"]) else str(b0["note"]))}
        if podds is None:
            r.update({"tier": "unscored", "vpe_verdict": "unscored (parlay odds unreadable)"})
        else:
            eff = boosted_american(podds, boost)
            r["odds_eff"] = eff
            if same_game:
                # v2.12: SGPs are RATE-ONLY; even one VPE rated TAKE is logged apart from VPE picks (11/1: a +146u SGP
                # win was sitting inside "VPE pick" profit)
                r["suggested"], r["rec_stake_usd"] = "sgp", 0.0
                r.update({"tier": "sgp_correlated", "vpe_verdict": "same-game parlay: legs correlated, EV not rated "
                          "(use the sgp command before placing)"})
                # v2.21: if the sgp checker rated these exact legs tonight, log its rating (re-priced at the odds taken)
                pk = frozenset((norm_name(o[0]["player"]), o[0]["market"], o[0]["side"], float(o[0]["line"])) for o in out)
                rt = next((x for x in reversed(SGP_RATINGS)
                           if frozenset((l[0], l[1], l[2], float(l[3])) for l in x["legs"]) == pk), None)
                if rt:
                    ev_t = rt["joint"] * american_to_decimal(eff) - 1
                    vd = "TAKE" if ev_t >= rt.get("min_ev", 0.10) else "SKIP"
                    r.update({"vpe_verdict": f"SGP checker: {vd} at the price taken (joint {rt['joint']:.1%}, fair "
                                             f"{rt['fair']:+d}, EV {ev_t:+.0%} at {int(eff):+d}; rated {rt['verdict']} at {rt['price']:+d})",
                              "ev_per_unit": round(ev_t, 3), "model_prob": round(rt["joint"], 4)})
            elif any(x is None for x in probs):
                r.update({"tier": "unscored", "vpe_verdict": "unscored (a leg has no projection)"})
            else:
                prob = float(np.prod(probs))
                te, v, eb, ee = rate_pick(prob, podds, eff, cfg, r["suggested"] in ("yes", "lean") or (None if r["suggested"] == "unknown" else False))
                if r["suggested"] == "lean":
                    v = "VPE lean parlay (best available, no edge)" if ee <= 0 else "VPE lean parlay (+EV at the price taken)"
                if v == "small +EV, below tiers":
                    v = "+EV parlay" if eb > 0 else "+EV only with boost"
                if boost_unused:
                    v = boost_unused + f" -> {v}"
                r.update({"model_prob": round(prob, 3), "blend_prob": round(prob, 3), "tier": te or "off_model",
                          "source": "engine" if te else "off_model", "vpe_verdict": v, "edge_base": eb, "edge": ee,
                          "ev_per_unit": round(ev_per_unit(prob, eff), 3)})
        rows.append(r)
    # v2.23 / v15.24: stamp the engine version on every row (note) so grading can split picks by engine version
    for r_ in rows:
        n_ = "" if r_.get("note") is None or (isinstance(r_.get("note"), float) and pd.isna(r_.get("note"))) else str(r_["note"])
        r_["note"] = (n_ + f" [{VERSION}]").strip()
    return rows

VPE_PARLAYS = []  # parlays VPE built this run (vpe_parlays.json, written by `parlay`), loaded by `log`

def _face(b):
    v = pd.to_numeric(b.get("face", np.nan) if hasattr(b, "get") else np.nan, errors="coerce")
    return float(v) if pd.notna(v) else np.nan

def _bonus(b):
    v = str(b.get("bonus", "") if hasattr(b, "get") else "").strip().lower()
    return "yes" if v in ("yes", "y", "true", "1", "bonus") else ""

def _book(b):
    """Book from placed.csv (optional `book` column: DraftKings / FanDuel); default DraftKings."""
    v = str(b.get("book", "") if hasattr(b, "get") else "").strip()
    return {"fd": "FanDuel", "fanduel": "FanDuel", "dk": "DraftKings", "draftkings": "DraftKings"}.get(
        v.lower().replace(" ", ""), v if v and v.lower() != "nan" else "DraftKings")

def win_profit(r):
    """Units won on a WIN at the effective (boosted) odds. Bankroll rule: stake_usd = total CHARGED (bet + 3% Bet
    Protect+ fee); face_usd = the bet amount the payout is based on. Bonus bet: stake_usd = the fee only and the bonus
    stake itself is not returned."""
    o = r.get("odds_eff")
    o = r.get("odds") if o is None or pd.isna(o) else o
    if not r["units"] or o is None or pd.isna(o):
        return 0.0
    face, charged = pd.to_numeric(r.get("face_usd"), errors="coerce"), pd.to_numeric(r.get("stake_usd"), errors="coerce")
    if pd.notna(face) and face > 0 and pd.notna(charged) and charged > 0:
        unit = charged / r["units"]
        bonus = str(r.get("bonus_bet", "")).strip().lower() in ("yes", "true", "1")
        won = face * (american_to_decimal(o) - (1 if bonus else 0)) - charged
        return round(won / unit, 3)
    return round(r["units"] * (american_to_decimal(o) - 1), 3)

def settle_parlays(log):
    """Parlay result from its graded legs: any leg LOSS -> LOSS; all WIN -> WIN at the boosted odds;
    a VOID/PUSH leg with the rest winning -> REPRICE (ask the user for DK's actual payout)."""
    if "bet_type" not in log:
        return log
    for i in log[(log["bet_type"] == "parlay") & log["result"].isna()].index:
        res = log.loc[(log["bet_type"] == "leg") & (log["parlay_id"] == log.loc[i, "parlay_id"]), "result"]
        if res.empty or res.isna().any():
            continue
        if (res == "LOSS").any():
            log.loc[i, "result"], log.loc[i, "profit_units"] = "LOSS", -log.loc[i, "units"]
        elif (res == "WIN").all():
            log.loc[i, "result"], log.loc[i, "profit_units"] = "WIN", win_profit(log.loc[i])
        else:
            log.loc[i, "result"] = "REPRICE"
    return log


# ---------- cash-out tracker (v2.5.3 / v15.8.3): what holding to the end would have paid ----------
# A cash-out row keeps result CASHOUT and the profit actually taken. Grading fills held_result / held_profit_units
# (as if held) and cashout_cost = held - taken (positive = cashing out cost money, negative = it saved money).
def settle_cashouts(log, actual_of):
    """actual_of(row) -> the player's stat for a straight bet (None if not available yet)."""
    for c in ("held_result", "held_profit_units", "cashout_cost"):
        if c not in log.columns:
            log[c] = np.nan
    log["held_result"] = log["held_result"].astype(object)
    cash = log[(log["result"].astype(str).str.upper() == "CASHOUT") & log["held_result"].isna()]
    for i in cash.index:
        r = log.loc[i]
        held = None
        if str(r.get("bet_type")) == "parlay":
            legs = log.loc[(log["bet_type"] == "leg") & (log["parlay_id"] == r["parlay_id"]), "result"]
            if legs.empty or legs.isna().any():
                continue
            held = "LOSS" if (legs == "LOSS").any() else ("WIN" if (legs == "WIN").all() else None)
        elif r["market"] in MARKETS:
            act = actual_of(r)
            if act is None:
                continue
            held = "PUSH" if act == float(r["line"]) else ("WIN" if (act > float(r["line"])) == (r["side"] == "Over") else "LOSS")
        if held is None:
            continue
        hp = win_profit(r) if held == "WIN" else (-float(r["units"]) if held == "LOSS" else 0.0)
        taken = float(pd.to_numeric(r["profit_units"], errors="coerce") or 0.0)
        log.loc[i, ["held_result", "held_profit_units", "cashout_cost"]] = [held, round(hp, 2), round(hp - taken, 2)]
    c = log[log["result"].astype(str).str.upper() == "CASHOUT"]
    if len(c):
        k = pd.to_numeric(c["cashout_cost"], errors="coerce")
        print(f"CASH-OUTS: {len(c)} | taken {pd.to_numeric(c['profit_units'], errors='coerce').sum():+.2f} | if held "
              f"{pd.to_numeric(c['held_profit_units'], errors='coerce').sum():+.2f} | cost of cashing out {k.sum():+.2f}"
              f" ({(k > 0).sum()} cost money, {(k < 0).sum()} saved money, {k.isna().sum()} not settled yet)")
    return log

def print_logged(rows):
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(rows).reindex(columns=["bet_type", "player", "market", "side", "line", "odds", "boost_pct",
          "odds_eff", "stake_usd", "suggested", "vpe_verdict", "tier", "model_prob", "edge_base", "edge"]).to_string(index=False))

# ---------- parlay builder v3: market-anchored legs, DraftKings + FanDuel, boosts, final check ----------
# Leg prob = learned model/market mix, 40/60 to start (the raw model disagreed with sharp books by ~14 pts on its "best" legs:
# mostly model error). Each leg is priced at each of the user's books (exact if that book has the best price on that
# side, else best price minus a haircut). A parlay is placed at ONE book, so each parlay is built per book and the
# better one is shown. A profit boost (--boost) is applied to the big parlay at --boost-book.
PARLAY = {"model_weight": 0.4, "side_bias": {},  # replaced every run by the learned weight (CALIB / Calib tab)
          "min_edge": 0.015,         # leg prob - break-even at the user's best book, after haircut
          "min_prob": 0.53, "max_gap": 0.10,          # model vs market gap: bigger = model probably missing news
          "min_odds": -200, "max_odds": 150, "min_minutes": 20, "min_minutes_ratio": 0.85,
          "books": ["draftkings", "fanduel"],
          "haircut": 0.05,           # price from another book -> assume ours pays 0.05 less in decimal odds (~6-8 cents)
          "junk_lines": {"rush_yds": 10.5, "rec_yds": 10.5, "receptions": 1.5, "rush_att": 3.5},  # line below -> skip
          "max_under_share": 0.45,   # v2.25 HARD RULE: Unders <= 45% of a parlay's legs, never under 1 (2-4 legs: 1, 5-6: 2) unless the math is
                                     # overwhelming (under_override_*). Evidence 11/16: logged Unders hit 31% vs 56% needed (16 props), boost-pool
                                     # Unders in the Shadow tab hit 47.0% vs 54.1% said (115 lines), and 7 of 9 legs on the 0-for-4 night were Unders
          "under_override_hit": 1.20,  # ...unless the Under-heavy parlay is >= 20% more likely to hit
          "under_override_ev": 0.10,   # ...AND has >= 10 pts more EV than the best balanced one
          "pool": 14, "max_legs": 7,
          "stake_pct": 0.05,         # default stake per parlay = 5% of bankroll (min $1)
          "targets": [("SAFE ~4x", 4.0), ("BIG ~10x", 10.0)]}
BOOK_NAME = {"draftkings": "DraftKings", "fanduel": "FanDuel"}

def _dec(o):
    return american_to_decimal(o)

def side_book(book, side):
    """odds-site book field is 'overbook/underbook' for two-sided rows; return the book for this side."""
    parts = [p.strip().lower() for p in ("" if pd.isna(book) else str(book)).split("/")]
    if len(parts) == 2:
        return parts[0] if side == "Over" else parts[1]
    return parts[0]


def market_prob_for(allp, player, mk, side, line, odds):
    """No-vig market prob for this exact leg from the run's bets_all.csv; else estimated from the price taken."""
    if allp is not None and len(allp) and "market_prob" in allp:
        h = allp[(allp["player"].map(norm_name) == norm_name(player)) & (allp["market"] == mk)
                 & (allp["side"] == side) & (pd.to_numeric(allp["line"], errors="coerce") == float(line))]
        if len(h) and pd.notna(h.iloc[0]["market_prob"]):
            return float(h.iloc[0]["market_prob"])
    return min(implied_prob(odds) / 1.045, 0.97)

# ---------- best available (v2.9 / v15.11): never a bare "no bets" ----------
# When nothing clears the edge bar, VPE still says which of tonight's lines are the least-bad to play, labeled LEAN.
# Why: a night with no bets teaches the learning loop nothing new about placed bets and CLV, and boosts get used
# either way. Leans are logged as suggested = "lean" (their own group in the Summary), never mixed with VPE picks.
LEAN = {"min_dec": 1.0 + 100.0 / 129.0,  # v2.27 KILL: no lean at -130 or shorter (was -150)
        "_was_min_dec": 1.0 / 0.6,  # v2.20: no lean shorter than -150 (11/16: 5-9, -118.7u, mostly -180 to -200 favourites)
        "min_edge": -0.04,   # leans may be up to 4 pts below break-even (anything worse is listed as a pass)
        "min_prob": 0.50, "straights": 3, "units": 0.5, "parlay_target": 4.0, "parlay_max_legs": 3}

def lean_pool(allp, exclude=()):
    d = parlay_pool(allp, exclude, min_edge=LEAN["min_edge"], min_prob=LEAN["min_prob"])
    if len(d):
        d = d[np.array([not _killed(m_, s_) for m_, s_ in zip(d["market"], d["side"])], dtype=bool)]
    return d[d["best_dec"] >= LEAN["min_dec"] - 1e-9] if len(d) and "best_dec" in d else d

def lean_straights(allp, exclude, unit_usd):
    """Top straight bets by edge (one per game) when the staked slip is empty."""
    d = lean_pool(allp, exclude)
    if d.empty:
        return d
    # v2.20: least-bad by EDGE first (closest to break-even), not by win chance: sorting by win chance picked the
    # heaviest favourites, where the juice costs the most.
    d = d.sort_values(["leg_edge", "leg_prob"], ascending=False).drop_duplicates("game_id").head(LEAN["straights"]).copy()
    d["bet_odds"] = [decimal_to_american(x) for x in d["best_dec"]]
    d["units"] = LEAN["units"]
    d["stake_usd"] = round(LEAN["units"] * unit_usd, 2)
    d["cost_usd"] = (d["leg_edge"] * d["stake_usd"]).round(2)
    return d

def best_available(allp, exclude, n=3):
    """v2.29 (hard rule 3: never a bare "no bets"): when not one line is within the lean limit, the closest lines to
    break-even anyway, one per game, rate-only. KILL-list slices are left out unless nothing else exists."""
    d = parlay_pool(allp, exclude, min_edge=-1.0, min_prob=0.0)
    if d.empty:
        return d.assign(on_kill=[])
    ok = np.array([not _killed(m_, s_) for m_, s_ in zip(d["market"], d["side"])], dtype=bool) \
        & (d["best_dec"] >= LEAN["min_dec"] - 1e-9).to_numpy()
    d = d.assign(on_kill=~ok)
    d = d[ok] if ok.any() else d
    d = d.sort_values(["leg_edge", "leg_prob"], ascending=False).drop_duplicates("game_id").head(n).copy()
    d["bet_odds"] = [decimal_to_american(x) for x in d["best_dec"]]
    return d


def _print_best_available(allp, excl):
    ba = best_available(allp, excl)
    if ba.empty:
        print("  BEST AVAILABLE: no scoreable line tonight (every line is held, injured or outside -200..+150): say so, "
              "and offer a rerun on a fresh recording")
        return []
    print(f"  BEST AVAILABLE (rate-only, no stake suggested): no line is within {-LEAN['min_edge']:.0%} of break-even, so "
          f"these are tonight's closest, one per game:")
    for _, r in ba.iterrows():
        print(f"    {r['player']} ({r['team']} vs {r['opp']}) {r['side']} {r['line']} {r['market']} "
              f"({int(r['bet_odds']):+d} at {r['bet_at']}) | win {r['leg_prob']:.1%} | edge {r['leg_edge']:+.1%}"
              + (" | on the KILL list (nothing else to show)" if r["on_kill"] else ""))
    return list(zip(ba["player"], ba["market"], ba["side"]))


def _under_max(n, share=None):
    """v2.26: most Unders allowed in an n-leg parlay: 45% of the legs, but never less than 1 (Adrian 11/16: 1 of 2 is fine).
    2 legs: 1 · 3: 1 · 4: 1 · 5: 2 · 6: 2."""
    share = PARLAY["max_under_share"] if share is None else share
    return n if share >= 1 else max(1, math.floor(share * n + 1e-9))


def lean_parlay(allp, exclude):
    """Least-bad ~4x parlay (highest EV, one leg per game) when no +EV parlay exists."""
    pool = lean_pool(allp, exclude)
    if pool.empty:
        return None
    cand = pool.sort_values("leg_edge", ascending=False).head(PARLAY["pool"]).to_dict("records")
    best, heavy = None, None
    for bk in PARLAY["books"]:
        col = "dec_" + bk
        for n in range(2, LEAN["parlay_max_legs"] + 1):
            for combo in itertools.combinations(cand, n):
                if len({str(c["game_id"]) for c in combo}) < n:
                    continue
                dec = float(np.prod([c[col] for c in combo]))
                if dec < LEAN["parlay_target"]:
                    continue
                p = float(np.prod([c["leg_prob"] for c in combo]))
                ev = p * dec - 1
                if sum(c["side"] == "Under" for c in combo) > _under_max(n):
                    if heavy is None or ev > heavy[0] + 1e-9:   # over the Under cap: kept only as the last resort
                        heavy = (ev, p, dec, combo, bk)
                    continue
                if best is None or ev > best[0] + 1e-9 or (abs(ev - best[0]) <= 1e-9 and p > best[1]):
                    best = (ev, p, dec, combo, bk)
    if best is None and heavy is not None:
        print(f"  UNDER CAP {PARLAY['max_under_share']:.0%}: no lean parlay fits the cap tonight -> the Under-heavy one is the only play (say so)")
    return best or heavy

def parlay_pool(allp, exclude=(), min_edge=None, min_prob=None):
    """Legs a parlay may use (one per player), ranked by leg prob. #1-3 Gold, #4-6 Silver."""
    d = allp.copy()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    _ts = d[d["odds"].notna()].assign(_l=lambda x: pd.to_numeric(x["line"], errors="coerce")).groupby(["player", "market", "_l"])["side"].nunique()
    _two = set(_ts[_ts >= 2].index)   # lines with BOTH an Over and an Under price = a real two-sided main line
    d["model_prob"] = pd.to_numeric(d["model_prob"], errors="coerce")
    d["market_prob"] = pd.to_numeric(d.get("market_prob"), errors="coerce")
    d = d[d["odds"].between(PARLAY["min_odds"], PARLAY["max_odds"]) & d["model_prob"].notna() & d["market_prob"].notna()]
    if "market_gap" in d:
        d = d[d["market_gap"].fillna(1) <= PARLAY["max_gap"]]
    if "usage_note" in d:
        d = d[~d["usage_note"].fillna("").astype(str).str.contains("backfield shift")]
    if "base_min" in d:
        d = d[d["base_min"] >= PARLAY["min_minutes"]]
    if "proj_min" in d and "base_min" in d:  # NBA: projected minutes cut 15%+ (blowout/usage) -> unstable leg
        d = d[(d["proj_min"] / d["base_min"].replace(0, np.nan)).fillna(0) >= PARLAY["min_minutes_ratio"]]
    if "new_team" in d:  # NBA: new team = role not settled yet
        d = d[~d["new_team"].fillna(False).astype(str).str.lower().isin(["true", "1"])]
    if "usage_factor" in d:  # NFL: usage model cut him (should not happen after the v11 fix; kept as a guard)
        d = d[pd.to_numeric(d["usage_factor"], errors="coerce").fillna(1) >= 0.9]
    if "injury" in d:
        d = d[d["injury"].fillna("").astype(str).str.strip() == ""]
    for mk, lo in PARLAY["junk_lines"].items():
        d = d[~((d["market"] == mk) & (pd.to_numeric(d["line"], errors="coerce") < lo))]
    if exclude:
        d = d[~d["player"].map(norm_name).isin({norm_name(x) for x in exclude})]
    if d.empty:
        return d.assign(rank=[], medal=[])
    d["src_book"] = [side_book(b, s) for b, s in zip(d["book"], d["side"])]
    for b in PARLAY["books"]:
        d["dec_" + b] = [_dec(o) - (0 if sb == b else PARLAY["haircut"]) for o, sb in zip(d["odds"], d["src_book"])]
    decs = d[["dec_" + b for b in PARLAY["books"]]]
    d["bet_at"] = [BOOK_NAME[c[4:]] for c in decs.idxmax(axis=1)]
    d["best_dec"] = decs.max(axis=1)
    d["price_dk"] = [decimal_to_american(x) for x in d["dec_draftkings"]]
    d["price_fd"] = [decimal_to_american(x) for x in d["dec_fanduel"]]
    d["leg_prob"] = np.round(blend_leg(d["model_prob"], d["market_prob"], d["market"], d["side"]), 3)
    # v2.26: pick shrink (see PICK): a picked leg's chance moves a share k toward what the price itself implies
    d["two_sided"] = [k_ in _two for k_ in zip(d["player"], d["market"], pd.to_numeric(d["line"], errors="coerce"))]
    d.loc[d["market"].isin(GAME_MK), "two_sided"] = True   # v2.30: a game line always has both sides priced
    d["leg_prob_raw"] = d["leg_prob"]
    _fair = np.array([_price_fair(o, t, mp) for o, t, mp in zip(d["odds"], d["two_sided"], d["market_prob"])])
    d["leg_prob"] = np.round(d["leg_prob"] - _pick_k() * np.clip(d["leg_prob"] - _fair, 0, None), 3)
    d["leg_edge"] = (d["leg_prob"] - 1 / d["best_dec"]).round(3)
    d["_estm"] = ((d["mkt_est"].astype(str).str.lower().isin(["true", "1"]) * EST_MARGIN) if "mkt_est" in d else 0.0) \
        + d["market"].map(MARKET_EXTRA).fillna(0.0)
    d = d[(d["leg_edge"] - d["_estm"] >= (PARLAY["min_edge"] if min_edge is None else min_edge))
          & (d["leg_prob"] >= (PARLAY["min_prob"] if min_prob is None else min_prob))]
    if min_edge is None:   # v2.27 KILL: an unboosted parlay never uses a leg at -130 or shorter (boost builds may, down to -200)
        d = d[d["best_dec"] > american_to_decimal(KILL["short_odds"]) + 1e-9] if _killed(odds=KILL["short_odds"]) else d
        d = d[np.array([not _killed(m_, s_) for m_, s_ in zip(d["market"], d["side"])], dtype=bool)]
    # take-if: worst odds (at either book) at which this leg still clears min_edge
    # v2.23 / v15.24: at THIS pool's cutoff (boost pool -3%, leans pool its own), so a leg the pool keeps at its price
    # never shows a take_min above that price (was always the regular parlay cutoff -> "take if +105" on a +100 leg it kept)
    _me = PARLAY["min_edge"] if min_edge is None else min_edge
    d["take_min"] = [int(decimal_to_american(1 / min(max(p - _me - m, 0.02), 0.98))) for p, m in zip(d["leg_prob"], d["_estm"])]
    # v2.16: flag a leg whose player is HELD on another market tonight (model vs market gap: usually news)
    gap_ = (pd.to_numeric(allp["model_prob"], errors="coerce") - pd.to_numeric(allp["market_prob"], errors="coerce")).abs()
    hmap = {}
    for _, h in allp[gap_ > CONFIG["model"].get("max_market_gap", 0.15)].iterrows():
        hmap.setdefault(norm_name(h["player"]), set()).add((h["market"], h["side"], float(h["line"])))
    d["held_note"] = ["; ".join(f"HELD {m} {sd2} {l:g}" for m, sd2, l in sorted(hmap.get(norm_name(pl), set()))
                                if (m, sd2, l) != (mk, sd, float(ln)))
                      for pl, mk, sd, ln in zip(d["player"], d["market"], d["side"], d["line"])]
    d = d.sort_values(["leg_prob", "leg_edge"], ascending=False).drop_duplicates("player").reset_index(drop=True)
    d["rank"] = d.index + 1
    d["medal"] = np.where(d["rank"] <= 3, "Gold", np.where(d["rank"] <= 6, "Silver", "pool"))
    return d

def best_parlay(pool, target, book, boost=0.0, under_share=None):
    """Highest hit chance among cross-game combos (one leg per game, <= max Unders) placed at `book`, every leg
    still +EV at that book, paying >= target x stake after any profit boost, and +EV overall."""
    col = "dec_" + book
    ok = pool[pool["leg_prob"] * pool[col] > 1].head(PARLAY["pool"])
    cand, best = ok.to_dict("records"), None
    for n in range(2, PARLAY["max_legs"] + 1):
        for combo in itertools.combinations(cand, n):
            if len({str(c["game_id"]) for c in combo}) < n:
                continue
            if under_share is not None and sum(c["side"] == "Under" for c in combo) > _under_max(n, under_share):
                continue
            dec = float(np.prod([c[col] for c in combo]))
            dec_eff = 1 + (dec - 1) * (1 + boost / 100)
            if dec_eff < target:
                continue
            p = float(np.prod([c["leg_prob"] for c in combo]))
            if p * dec_eff <= 1:
                continue
            if best is None or p > best[0] + 1e-9 or (abs(p - best[0]) <= 1e-9 and dec_eff > best[2]):
                best = (p, dec, dec_eff, combo, book)
    return best

def cmd_parlay(a):
    if getattr(a, "boost_min_leg_edge", None) is not None:  # v2.24 / v15.25
        BOOST_LEG["min_edge"] = float(a.boost_min_leg_edge)
    set_calib(getattr(a, "calib", None), SPORT)
    allp = pd.read_csv(a.all)
    if lock_from(allp):
        print(f"PRICES LOCKED ({LOCK['books']} recording): legs are priced as recorded; take_min / take-if shown in case the price moves before you bet")
    pool = parlay_pool(allp, [x for x in (a.exclude or "").split(";") if x.strip()])
    bank = a.bankroll or CONFIG["bankroll"]
    stake = a.stake if a.stake else max(1.0, round(bank * PARLAY["stake_pct"] * 2) / 2)
    boost, bbook = (a.boost or 0.0), (a.boost_book or "draftkings").lower().replace(" ", "")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    cols = [c for c in ("rank", "medal", "player", "team", "opp", "market", "side", "line", "odds", "src_book",
                        "price_dk", "price_fd", "bet_at", "leg_prob", "model_prob", "market_prob", "leg_edge",
                        "take_min", "line_limit", "role_risk") if c in pool] + (["held_note"] if "held_note" in pool and pool["held_note"].astype(str).str.len().gt(0).any() else [])
    print(f"LEG POOL: {len(pool)} legs qualify | leg prob = {PARLAY['model_weight']:.0%} model + "
          f"{1 - PARLAY['model_weight']:.0%} market | edge >= {PARLAY['min_edge']:.1%} at DK/FD price | "
          f"gap <= {PARLAY['max_gap']:.0%} | odds {PARLAY['min_odds']} to +{PARLAY['max_odds']}")
    print(pool.head(6)[cols].to_string(index=False) if len(pool) else "none")
    print(f"STAKE: ${stake:.2f} per parlay ({PARLAY['stake_pct']:.0%} of ${bank:.0f} bankroll unless --stake given)"
          + (f" | {boost:g}% profit boost -> its own BOOST parlay at {BOOK_NAME.get(bbook, bbook)}" if boost else ""))
    keys, built = [], []
    for name, target in PARLAY["targets"]:
        found, free = [], []
        for bk in PARLAY["books"]:
            bst = 0.0  # boosts get their own parlay
            r = best_parlay(pool, target, bk, bst, PARLAY["max_under_share"])
            if r:
                found.append(r)
            r = best_parlay(pool, target, bk, bst, None)
            if r:
                free.append(r)
        print(f"\n=== {name} PARLAY (pays >= {target:g}x) ===")
        pick = lambda xs: max(xs, key=lambda r: (round(r[0], 4), r[0] * r[2])) if xs else None
        bal, heavy = pick(found), pick(free)
        override = False
        if heavy and sum(c["side"] == "Under" for c in heavy[3]) > _under_max(len(heavy[3])):
            h_ev = heavy[0] * heavy[2] - 1
            if bal:
                b_ev = bal[0] * bal[2] - 1
                override = heavy[0] >= bal[0] * PARLAY["under_override_hit"] and h_ev >= b_ev + PARLAY["under_override_ev"]
                why = (f"Under-heavy option: hit {heavy[0]:.1%} vs {bal[0]:.1%} balanced, EV {h_ev:+.0%} vs {b_ev:+.0%}")
            else:
                override = h_ev >= 0.25
                why = f"no balanced parlay qualifies; Under-heavy option EV {h_ev:+.0%} (needs >= +25%)"
            print(f"  UNDER CAP {PARLAY['max_under_share']:.0%}: {why} -> "
                  + ("OVERRIDE: the math clearly favors the Unders, using it" if override else "kept the balanced parlay"))
        found = [heavy] if override else ([bal] if bal else [])
        if not found:
            print("  not available tonight: no +EV combination at honest win chances (rules not loosened)."
                  " A profit boost is the reliable edge: tell me the % and book")
            continue
        p, dec, dec_eff, combo, bk = max(found, key=lambda r: (round(r[0], 4), r[0] * r[2]))
        book_p = float(np.prod([1 / c["dec_" + bk] for c in combo]))
        bnote = f" incl. {boost:g}% boost" if dec_eff > dec + 1e-9 else ""
        print(f"  PLACE AT {BOOK_NAME[bk].upper()} | {len(combo)} legs | est. odds {decimal_to_american(dec):+d} | "
              f"${stake:.2f} pays ~${stake * dec_eff:.2f}{bnote} | hit chance {p:.1%} vs book {book_p:.1%} | "
              f"EV {p * dec_eff - 1:+.0%} | Unders {sum(c['side'] == 'Under' for c in combo)}")
        print(pd.DataFrame(combo)[cols].to_string(index=False))
        need = 1 + (target - 1) / (1 + boost / 100) if bnote else target
        tk_ = parlay_take_if(p)
        print(f"  -> build it at {BOOK_NAME[bk]}{' as recorded' if LOCK['on'] else ''} | TAKE-IF: parlay odds >= "
              f"{decimal_to_american(tk_):+d} (break-even at VPE's hit chance; ~{target:g}x needs {decimal_to_american(need):+d})"
              f" and every leg at or above its take_min. Price moved? -> `reprice`")
        keys += [(c["player"], c["market"], c["side"]) for c in combo]
        built.append({"name": name, "target": target, "book": bk, "odds": decimal_to_american(dec), "hit": round(p, 4),
                      "take_if": decimal_to_american(parlay_take_if(p)),
                      "stake_usd": stake, "legs": [[c["player"], c["market"], c["side"], float(c["line"])] for c in combo]})
    specs = parse_boosts(getattr(a, "boosts", None)) or ([boost_spec_from_args(a)] if boost else [])
    bps, used = [], []
    for spec in sorted(specs, key=lambda x: -x["pct"]):  # biggest boost picks first; no leg is reused across boosts
        if spec.get("sgp"):
            print(f"\n=== BOOST {spec['label']}: same-game only -> run `sgpbuild --boosts {spec['label']}` (needs --games)"
                  " to search same-game combos; not built here ===")
            continue
        r_ = print_boost_spec(spec, allp, stake, cols, keys, used, a)
        if r_:
            bps.append(r_)
            built.append(r_)
            used += [l[0] for l in r_["legs"]]
    bp = next((b for b in bps if not b.get("lean")), bps[0] if bps else None)
    excl = [x for x in (a.exclude or "").split(";") if x.strip()]
    lean_keys = [tuple(l[:3]) for b in bps if b.get("lean") for l in b["legs"]]
    if not any(b["target"] for b in built):
        lp = lean_parlay(allp, excl)
        print(f"\n=== LEAN - BEST AVAILABLE PARLAY (~{LEAN['parlay_target']:g}x): VPE can't beat these lines tonight ===")
        if lp:
            ev, p, dec, combo, bk = lp
            print(f"  PLACE AT {BOOK_NAME[bk].upper()} | {len(combo)} legs | est. odds {decimal_to_american(dec):+d} | "
                  f"${stake:.2f} pays ~${stake * dec:.2f} | hit chance {p:.1%} | EV {ev:+.0%} (expected cost ~${-ev * stake:.2f})")
            print(pd.DataFrame(combo)[cols].to_string(index=False))
            lean_keys += [(c["player"], c["market"], c["side"]) for c in combo]
            built.append({"name": f"LEAN ~{LEAN['parlay_target']:g}x", "target": 0, "book": bk, "lean": True,
                          "odds": decimal_to_american(dec), "hit": round(p, 4), "ev": round(ev, 3), "stake_usd": stake,
                          "legs": [[c["player"], c["market"], c["side"], float(c["line"])] for c in combo]})
        else:
            print(f"  none: no legs within {-LEAN['min_edge']:.0%} of break-even tonight")
    has_bets, lean_straight = False, set()
    if a.bets and os.path.exists(a.bets):
        bb = pd.read_csv(a.bets)
        has_bets = len(bb[bb["tier"].notna()] if "tier" in bb else bb) > 0
    if not has_bets:
        unit_usd = bank * CONFIG["unit_pct_of_bankroll"]
        ls = lean_straights(allp, excl, unit_usd)
        print(f"\n=== LEAN - BEST AVAILABLE STRAIGHTS: VPE says none of tonight's lines beat the book; "
              f"if you play, these are closest ({LEAN['units']:g}u each, one per game) ===")
        if len(ls):
            for _, r in ls.iterrows():
                print(f"  {r['player']} ({r['team']} vs {r['opp']}) {r['side']} {r['line']} {r['market']} "
                      f"({int(r['bet_odds']):+d} at {r['bet_at']}) | win {r['leg_prob']:.1%} | edge {r['leg_edge']:+.1%} | "
                      f"{r['units']:g}u (${r['stake_usd']:.2f})" + f" | take only at {int(r['bet_odds']):+d} or better" + _rr(r))
            lean_keys += list(zip(ls["player"], ls["market"], ls["side"]))
            lean_straight = set(zip(ls["player"], ls["market"], ls["side"]))
        else:
            _bk = _print_best_available(allp, excl)
            lean_keys += _bk
            lean_straight = set(_bk)
        ls.to_csv(getattr(a, "leans_out", None) or "leans.csv", index=False)
    elif os.path.exists(getattr(a, "leans_out", None) or "leans.csv"):
        os.remove(getattr(a, "leans_out", None) or "leans.csv")
    # ONE-PARLAY RULE (10/21-10/26: parlays were the whole loss, -$15.14 of -$15.89): recommend ONE parlay a night:
    # the BOOST parlay if there is one (+EV at market odds alone), else the biggest VPE could build (~10x), else ~4x.
    if bps:   # v2.27 BOOST RANKER: every boost scored by true EV (after the pick shrink), best first
        print("\nBOOST RANK (true EV after the pick shrink; vig = what the legs' hold costs before the boost pays it back):")
        for i_, b in enumerate(sorted(bps, key=lambda x: -x.get("ev", -9)), 1):
            ev_ = b.get("ev", 0.0)
            verdict = "USE IT" if ev_ >= 0.05 and not b.get("lean") else ("THIN: about break-even" if ev_ > -0.02 and not b.get("lean") else "SAVE IT / lean only")
            print(f"  #{i_} {b['name']} at {BOOK_NAME[b['book']]}: {len(b['legs'])} legs ({b.get('n_two', 0)} on two-sided main lines) | "
                  f"hit {b['hit']:.1%} | vig paid {b.get('vig', 0):.1%} | EV {ev_:+.1%} = ${ev_ * b['stake_usd']:+.2f} on ${b['stake_usd']:.2f} | {verdict}")
    real = [b for b in built if not b.get("lean")]
    real_boosts = [b for b in bps if not b.get("lean")]
    if len(real_boosts) > 1:
        for b in real_boosts:
            b["recommended"] = True
        print(f"\nONE-PARLAY RULE (one per boost): place ONLY these {len(real_boosts)} BOOST parlays, no leg shared: "
              + " | ".join(f"{b['name']} at {BOOK_NAME[b['book']]} {b['odds']:+d} (take-if {b['take_if']:+d})" for b in real_boosts)
              + ". Any other parlay above = reference only.")
    elif built:
        # boost parlay (real, then lean) > biggest real parlay > lean parlay
        rec = (bp if bp and not bp.get("lean") else None) or (max(real, key=lambda b: b["target"]) if real else None) \
            or bp or built[-1]
        rec["recommended"] = True
        tag = (" - LEAN: no edge, the least-bad play with what we're given; logged as a VPE lean"
               if rec.get("lean") else "")
        print(f"\nONE-PARLAY RULE: place ONLY the {rec['name']} parlay at {BOOK_NAME[rec['book']]} "
              f"({rec['odds']:+d}, hit {rec['hit']:.1%}, ${rec['stake_usd']:.2f}{', take-if ' + format(rec['take_if'], '+d') if rec.get('take_if') is not None and not rec.get('lean') else ''}){tag}. Any other parlay above = reference only.")
    else:
        print("\nONE-PARLAY RULE: no parlay tonight (no legs close enough to break-even). Tonight's plays are the "
              "straight bets or the BEST AVAILABLE lines printed above.")
    import json
    json.dump(built, open(getattr(a, "parlays_out", None) or "vpe_parlays.json", "w"), indent=1)
    if len(pool):
        keys += list(zip(pool.head(6)["player"], pool.head(6)["market"], pool.head(6)["side"]))
    if a.bets and os.path.exists(a.bets):
        b = pd.read_csv(a.bets)
        keys += list(zip(b["player"], b["market"], b["side"]))
    # a lean straight stays a lean even if the same prop is also a leg somewhere (it is a -EV straight bet)
    sk = pd.DataFrame(sorted(set(keys) - lean_straight), columns=["player", "market", "side"]).assign(kind="pick")
    lk = pd.DataFrame(sorted((set(lean_keys) - set(keys)) | lean_straight), columns=["player", "market", "side"]).assign(kind="lean")
    sl_ = pd.concat([sk, lk], ignore_index=True)
    md_ = {(r_["player"], r_["market"], r_["side"]): r_["medal"] for _, r_ in pool.head(6).iterrows()} if len(pool) and "medal" in pool else {}
    sl_["medal"] = [md_.get(k_, "") for k_ in zip(sl_["player"], sl_["market"], sl_["side"])]
    sl_.to_csv(a.slip_out, index=False)
    print(f"\nslip of every VPE leg/pick -> {a.slip_out} (use with log --slip and final --slip)")
    print("NOTE: legs treated as independent. Hit chances lean on the market on purpose; the model only tilts them.")


# ---------- learning loop: every scored line is graded the next day ("shadow"), the model weight is learned ----------
# One night of real lines (NBA 10/21/26, 619 line-sides) showed the raw model was overconfident: tiered picks said 63%
# and hit 48%; parlay-grade legs said 59% and hit 48%; the no-vig market alone scored best. So the model's weight is
# not fixed by hand any more: it starts at 40% and is re-fit from every graded line (shrunk toward 40% until
# ~1,500 lines are in), and each market's Over rate is corrected if it keeps landing off what VPE predicted.
CALIB_W = [round(i * 0.1, 1) for i in range(11)]
CALIB_COLS = ["sport", "market", "n", "sum_over", "sum_pm", "sum_pk"] + [f"ll_{int(round(x * 100)):03d}" for x in CALIB_W]
CAL_EXTRA = {}
CALIB = {"prior_w": 0.4, "n0_w": 1500, "n0_bias": 1500, "bias_cap": 0.03, "w_min": 0.15, "w_max": 0.6}
SHADOW_COLS = ["key", "player", "market", "line", "over_odds", "under_odds", "p_model", "p_market", "actual", "over"]


def _values(path):
    import json
    raw = json.load(open(path))
    return raw.get("values", []) if isinstance(raw, dict) else raw


def _calib_rows(path):
    if not path or not os.path.exists(path):
        return []
    vals = _values(path)
    if len(vals) < 2:
        return []
    hdr = [str(h) for h in vals[0]]
    return [dict(zip(hdr, list(r) + [""] * (len(hdr) - len(r)))) for r in vals[1:] if r and str(r[0]).strip()]


def _f(r, c):
    try:
        return float(str(r.get(c, "") or 0).replace(",", ""))
    except ValueError:
        return 0.0


def load_calib(path, sport):
    """Calib tab (get_values JSON) -> (model weight, {market: Over-rate correction}, graded lines)."""
    rows = [r for r in _calib_rows(path) if str(r.get("sport", "")).upper() == sport and _f(r, "n") > 0]
    w, bias = CALIB["prior_w"], {}
    if not rows:
        return w, bias, 0
    n_tot = sum(_f(r, "n") for r in rows)
    ll = [sum(_f(r, f"ll_{int(round(x * 100)):03d}") for r in rows) for x in CALIB_W]
    w_fit = CALIB_W[int(np.argmin(ll))]
    w = (n_tot * w_fit + CALIB["n0_w"] * CALIB["prior_w"]) / (n_tot + CALIB["n0_w"])
    w = float(min(max(w, CALIB["w_min"]), CALIB["w_max"]))
    mk_w = {}
    for r in rows:
        n = _f(r, "n")
        # v2.17: each market's own best mix, shrunk toward the shared weight (1,500 lines = half trust)
        llm = [_f(r, f"ll_{int(round(x * 100)):03d}") for x in CALIB_W]
        wm = (n * CALIB_W[int(np.argmin(llm))] + CALIB["n0_w"] * w) / (n + CALIB["n0_w"])
        wm = float(min(max(wm, CALIB["w_min"]), CALIB["w_max"]))
        mk_w[str(r["market"])] = round(wm, 3)
        pred = (wm * _f(r, "sum_pm") + (1 - wm) * _f(r, "sum_pk")) / n
        b = (_f(r, "sum_over") / n - pred) * n / (n + CALIB["n0_bias"])
        bias[str(r["market"])] = float(np.clip(b, -CALIB["bias_cap"], CALIB["bias_cap"]))
    CAL_EXTRA.update({"mk_w": mk_w, "w_fit": w_fit, "n": int(n_tot)})
    return round(w, 3), bias, int(n_tot)


def set_calib(path, sport):
    CAL_EXTRA.clear()
    w, bias, n = load_calib(path, sport)
    PARLAY["model_weight"], PARLAY["side_bias"] = w, bias
    PARLAY["mk_weight"] = dict(CAL_EXTRA.get("mk_w") or {})
    # v15.18: NFL goes boosts + leans only while the model adds nothing over the market (best mix 0% model on 3,000+
    # graded lines); it lifts itself as soon as the Calib tab shows the model earning weight again
    PARLAY["boost_only"] = (sport == "NFL" and CAL_EXTRA.get("n", 0) >= 3000 and CAL_EXTRA.get("w_fit", 1) <= 0.0)
    b = ", ".join(f"{k} Over {v:+.1%}" for k, v in sorted(bias.items()) if abs(v) >= 0.005) or "none yet"
    src = f"{n} graded lines" if n else "no graded lines yet (prior)"
    mw_ = PARLAY.get("mk_weight") or {}
    mws = ", ".join(f"{k} {v:.0%}" for k, v in sorted(mw_.items()) if abs(v - w) >= 0.03)
    print(f"CALIB: {src} -> win chance = {w:.0%} model + {1 - w:.0%} market"
          + (f" (own mix: {mws})" if mws else "") + f" | Over-rate corrections: {b}")
    if PARLAY.get("boost_only"):
        print("NFL MODE: BOOSTS + LEANS ONLY - on 3,000+ graded lines the most accurate mix is 0% model, so VPE's NFL"
              " numbers add nothing over the market. No straight picks / model parlays; boost parlays and labeled leans only."
              " Lifts itself when the model earns weight again (or pass --all-picks).")
    set_learned(os.path.join(os.path.dirname(path) or ".", "learned_tab.json") if path else None, sport)
    return w, bias


def blend_leg(model_p, market_p, market=None, side=None):
    """VPE win chance = learned mix of model and no-vig market, + the market's learned Over-rate correction."""
    w = _mw(market)
    p = w * model_p + (1 - w) * market_p
    b = PARLAY.get("side_bias") or {}
    if market is not None and b:
        if isinstance(market, pd.Series):
            p = p + pd.Series(market).map(b).fillna(0.0).to_numpy() * np.where(np.asarray(side) == "Over", 1.0, -1.0)
        else:
            p = p + b.get(market, 0.0) * (1.0 if side == "Over" else -1.0)
    if LEARNED["tail"] or LEARNED["gate"]:
        p2 = _learn_adjust(p, market_p, market, side)
        # v2.30: tail fixes and gates were learned from player props; game lines keep their own mix untouched
        if isinstance(market, (pd.Series, np.ndarray)):
            p = np.where(pd.Series(market).isin(GAME_MK).to_numpy(), np.asarray(p, dtype=float), np.asarray(p2, dtype=float))
        else:
            p = p if market in GAME_MK else p2
    if isinstance(p, (pd.Series, np.ndarray)):
        return np.clip(p, 0.01, 0.99)
    return float(min(max(p, 0.01), 0.99))


# ---------- self-learning (v2.4 / v15.7): bounded corrections learned from the graded Shadow tab ----------
# Runs every morning after grading (`learn`). Writes the 'Learned NBA' / 'Learned NFL' tab; check/parlay/final/takeif read it via the
# learned_tab.json saved next to calib_tab.json. Missing file = no change. Two things are learned, both capped:
#  1. tail fix: when VPE says 60%+ for a side, does that side hit that often? correct each band, max +-4 pts, shrunk.
#  2. market gate: per market x side, the lines VPE would bet; if they hit 5+ pts under what VPE said (z <= -2,
#     150+ lines) VPE trusts the model half as much there; 10+ pts under (z <= -3) it defers fully to the market (no bet).
#     A gate lifts itself when the recent window recovers. The core projection math is never touched.
LEARN = {"start": {"NBA": "2026-10-26", "NFL": "2026-W1"}, "window": 8000, "n0_tail": 600, "tail_cap": 0.04,
         "tail_bins": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0], "bet_edge": 0.025, "gate_n": 150, "est_hold": 1.045}
LEARNED = {"tail": None, "gate": {}, "pick": None, "kill": None}
# v2.26 PICK SHRINK (approved 11/16). The legs VPE picks for parlays and boosts are the lines where its chance sits furthest
# above the price, and that selection is mostly noise: 253 graded NBA pool legs said 54.2%, hit 50.2% (NFL 94: 55.9% -> 51.1%),
# almost exactly the chance the PRICE implies once the book's hold is taken out. So a picked leg's chance is pulled a share k
# of the way from VPE's number toward the price-fair chance. k is re-learned every morning from the graded Shadow tab
# (`learn`, row kind "pick"), shrunk by sample size (n0 300 legs); until a learned value exists it is 0.5.
PICK = {"k0": 0.5, "n0": 300, "pool_prob": 0.50, "pool_edge": -0.03, "min_n": 60}


def _pick_k():
    return PICK["k0"] if LEARNED.get("pick") is None else float(LEARNED["pick"])


def _price_fair(odds, two_sided, market_p):
    """What the price itself says the chance is: the no-vig market chance on a two-sided line, else implied / (1 + hold)."""
    if two_sided or odds is None or pd.isna(odds):
        return float(market_p)
    h = CONFIG["one_sided_hold"]["mid"]
    return min(float(market_p), implied_prob(odds) / (1 + h))
LEARNED_COLS = ["sport", "kind", "key", "value", "n", "note", "updated"]


def _key_ord(k):
    k = str(k).lstrip("'")
    m = re.match(r"(\d{4})-W(\d+)$", k)
    return (int(m.group(1)), int(m.group(2))) if m else (int(k[:4]) if k[:4].isdigit() else 0, k)


def _learned_rows(path):
    import json
    if not path or not os.path.exists(path):
        return []
    v = json.load(open(path))
    v = v.get("values", v) if isinstance(v, dict) else v
    if not v or str(v[0][0]) != "sport":
        return []
    return [dict(zip(v[0], list(r) + [""] * (len(v[0]) - len(r)))) for r in v[1:]]


def set_learned(path, sport):
    LEARNED["tail"], LEARNED["gate"], LEARNED["pick"], LEARNED["kill"] = None, {}, None, None
    xs, ds, gates, ax = [], [], {}, []
    for r in _learned_rows(path):
        if str(r.get("sport", "")).upper() != sport:
            continue
        if r["kind"] == "tail":
            xs.append(float(r["key"])); ds.append(float(r["value"]))
        elif r["kind"] == "gate" and str(r["value"]) in ("1", "2"):
            gates[str(r["key"])] = int(r["value"])
        elif r["kind"] == "kill":
            LEARNED["kill"] = (LEARNED["kill"] or set()) | ({str(r["key"])} if str(r["value"]).strip() in ("1", "1.0") else set())
        elif r["kind"] == "pick":
            try:
                LEARNED["pick"] = float(np.clip(float(r["value"]), 0.0, 1.0))
            except (TypeError, ValueError):
                pass
    if xs:
        o = np.argsort(xs)
        # v2.20: the learn step now also fits the 50-55 and 55-60 bands (11/16 review: NBA said 52.5%, hit 49.9% on
        # 2,411 lines, and nothing corrected it). A Learned tab written before that has no row under 60%: keep the old
        # flat-zero anchor there so nothing changes until the next morning learn writes the new bands.
        mid = any(x < LEARN["tail_bins"][2] for x in xs)
        ax, ay = ([0.5], [0.0]) if mid else ([0.5, LEARN["tail_bins"][2]], [0.0, 0.0])
        LEARNED["tail"] = (ax + [xs[i] for i in o], ay + [float(np.clip(ds[i], -LEARN["tail_cap"], LEARN["tail_cap"])) for i in o])
    LEARNED["gate"] = gates
    if path:
        t = "none" if not xs else ", ".join(f"{x:.0%}->{d:+.1%}" for x, d in zip(LEARNED["tail"][0][len(ax):], LEARNED["tail"][1][len(ax):]) if abs(d) >= 0.005) or "none needed"
        g = ", ".join(f"{k} {'HALF' if v == 1 else 'PAUSED'}" for k, v in sorted(gates.items())) or "none"
        print(f"LEARNED: tail fixes {t} | market gates {g} | pick shrink {_pick_k():.0%} | KILL list: {_kl()}"
              + (" (starting value)" if LEARNED.get("pick") is None else " (learned)"))


def _learn_adjust(p, market_p, market=None, side=None):
    """Apply the learned gate and tail fix to blended win chances (array or scalar)."""
    scalar = np.ndim(p) == 0
    p = np.atleast_1d(np.asarray(p, dtype=float)).copy()
    mp = np.broadcast_to(np.atleast_1d(np.asarray(market_p, dtype=float)), p.shape)
    if LEARNED["gate"] and market is not None:
        mk = np.broadcast_to(np.atleast_1d(np.asarray(market, dtype=object)), p.shape)
        sd = np.broadcast_to(np.atleast_1d(np.asarray(side, dtype=object)), p.shape)
        lvl = np.array([LEARNED["gate"].get(f"{m}|{s}", 0) for m, s in zip(mk, sd)])
        p = np.where(lvl == 1, mp + 0.5 * (p - mp), np.where(lvl == 2, mp, p))
    if LEARNED["tail"]:
        c = np.maximum(p, 1 - p)
        d = np.interp(c, *LEARNED["tail"])
        p = np.where(p >= 0.5, p + d, p - d)
    return float(p[0]) if scalar else p


# ---------- situational watch (v2.10): home / away / back-to-back, checked on live results every morning ----------
# Tested 10/29 on 2024-25 + 2025-26 (455k player-game-markets, + DK totals): player-specific and team-specific home/away
# and B2B splits, opponent defense home vs road and opponent-on-B2B all failed (no better than noise); a league-wide B2B
# trim on rebounds/assists was ~1-2% before DK totals and ~0 after (the market already prices tired legs). So nothing is
# applied. Instead every graded live line is split by home / away / 2nd night of a B2B, per market, and VPE's predicted
# Over rate is compared with what hit. REPORT ONLY: a split is flagged for the upgrade chat once n >= 800 and |z| >= 3.
WATCH = {"n_flag": 800, "z_flag": 3.0}

def situational_watch(d, q, today):
    if d.empty:
        return []
    try:
        dates = sorted({str(k)[:10] for k in d["key"]})
        boxes = [load_box(s, max_age_hours=24) for s in sorted({season_for(pd.Timestamp(x)) for x in dates})]
        box = pd.concat([b for b in boxes if len(b)], ignore_index=True)
    except Exception as e:
        print(f"  situational watch: box scores unavailable ({e})")
        return []
    box = box[box["played"]]
    td = set(zip(box["team"], box["date"]))
    ctx = box.drop_duplicates(["_key", "date"]).set_index(["_key", "date"])[["team", "home"]]
    keys = pd.MultiIndex.from_arrays([d["player"].map(norm_name), pd.to_datetime(d["key"].str[:10])])
    c = ctx.reindex(keys)
    home = c["home"].to_numpy()
    b2b = np.array([(t, dt - pd.Timedelta(days=1)) in td if isinstance(t, str) else False
                    for t, dt in zip(c["team"], keys.get_level_values(1))])
    known = pd.notna(c["team"]).to_numpy()
    rows = []
    print("  situational watch (VPE's Over rate said -> hit; report only, nothing applied):")
    for lab, m0 in (("home", known & (home == True)), ("away", known & (home == False)), ("b2b", known & b2b)):
        for mk in sorted(d["market"].unique()):
            m = m0 & (d["market"].to_numpy() == mk)
            n = int(m.sum())
            if n < 50:
                continue
            said, got = float(q[m].mean()), float(d["over"].to_numpy()[m].mean())
            z = (got - said) / max(np.sqrt(said * (1 - said) / n), 1e-9)
            flag = "FLAG for upgrade chat" if n >= WATCH["n_flag"] and abs(z) >= WATCH["z_flag"] else "ok"
            rows.append([SPORT, "watch", f"{lab}|{mk}", round(got - said, 4), n, f"said {said:.1%} hit {got:.1%} z {z:+.1f} {flag}", today])
            if flag != "ok" or lab == "b2b":
                print(f"    {lab:<5} {mk:<6} said {said:.1%} hit {got:.1%}  n={n:<5} z={z:+.1f} {flag}")
    flagged = [r for r in rows if "FLAG" in r[5]]
    print(f"    {len(rows)} splits checked (home {int((known & (home == True)).sum())} / away {int((known & (home == False)).sum())}"
          f" / b2b {int((known & b2b).sum())} lines) -> " + (f"{len(flagged)} FLAGGED" if flagged else "nothing flagged: model holds up"))
    return rows

def cmd_learn(a):
    """Shadow window + Calib tab -> Learned tab (tail fixes + market gates). Prints what changed and why."""
    import json, datetime
    set_calib(a.calib, SPORT)
    LEARNED["tail"], LEARNED["gate"] = None, {}   # learn from the base blend, never from its own fixes
    vals = [r for j in a.json for r in _values(j)]   # several 2000-row reads, oldest first
    rows = [list(r)[:len(SHADOW_COLS)] + [""] * (len(SHADOW_COLS) - len(r)) for r in vals if r and str(r[0]) != "key"]
    start = _key_ord(LEARN["start"][SPORT])
    seen, data = set(), []
    for r in rows:
        k = str(r[0]).lstrip("'")
        if str(r[9]).strip() not in ("0", "1") or _key_ord(k) < start or not str(r[6]).strip() or not str(r[7]).strip():
            continue
        u = (k, r[1], r[2], str(r[3]))
        if u in seen:
            continue
        seen.add(u)
        data.append({"market": r[2], "pm": float(r[6]), "pk": float(r[7]), "over": int(float(r[9])),
                     "oo": str(r[4]).strip(), "uo": str(r[5]).strip(), "key": k, "player": r[1]})
    d = pd.DataFrame(data).tail(LEARN["window"])
    prev = {(r["kind"], str(r["key"])): r for r in _learned_rows(a.learned) if str(r.get("sport", "")).upper() == SPORT}
    today = datetime.date.today().isoformat()
    out = [LEARNED_COLS]
    print(f"LEARN {SPORT}: {len(d)} graded lines since {LEARN['start'][SPORT]} (window {LEARN['window']})")
    if d.empty:
        json.dump(out, open(a.out, "w"))
        print(f"  nothing to learn yet -> {a.out} (header only; write to 'Learned {SPORT}'!A1 after clearing A:G)")
        return
    q = np.asarray(blend_leg(d["pm"].to_numpy(), d["pk"].to_numpy(), d["market"], np.full(len(d), "Over")), dtype=float)
    fav_over = q >= 0.5
    c = np.where(fav_over, q, 1 - q)
    hit = np.where(fav_over, d["over"], 1 - d["over"])
    bins = LEARN["tail_bins"]
    print("  tail check (VPE said -> happened, n, fix):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (c >= lo) & (c < hi)
        n = int(m.sum())
        if n == 0:
            continue
        said, got = float(c[m].mean()), float(hit[m].mean())
        fix = float(np.clip((got - said) * n / (n + LEARN["n0_tail"]), -LEARN["tail_cap"], LEARN["tail_cap"]))
        out.append([SPORT, "tail", round(said, 3), round(fix, 4), n, f"said {said:.1%} hit {got:.1%}", today])
        print(f"    {lo:.0%}-{hi:.0%}: {said:.1%} -> {got:.1%}  n={n:<5} fix {fix:+.1%}")
    print("  market gates (lines VPE would bet):")
    for side in ("Over", "Under"):
        odds = d["oo"] if side == "Over" else d["uo"]
        ok = odds.str.match(r"^[+-]?\d+(\.0)?$")
        p_side = np.where(side == "Over", q, 1 - q)
        # v2.12: recorded "X+" milestone ladders have no Under price (10/26-11/1: 1,225 of 1,357 Shadow lines), so the
        # Under gates never saw a bet. A side with no posted price gets a break-even estimated from the no-vig market
        # chance + a normal 4.5% hold, so every Under VPE would have liked is checked.
        pk_side = d["pk"].to_numpy() if side == "Over" else 1 - d["pk"].to_numpy()
        be = np.clip(pk_side * LEARN["est_hold"], 0.02, 0.97)
        be[ok.to_numpy()] = [implied_prob(float(o)) for o in odds[ok]]
        y = d["over"].to_numpy() if side == "Over" else 1 - d["over"].to_numpy()
        bet = np.isfinite(be) & (p_side - be >= LEARN["bet_edge"])
        for mk in sorted(d["market"].unique()):
            m = bet & (d["market"].to_numpy() == mk)
            n = int(m.sum())
            key = f"{mk}|{side}"
            if n < 20:
                continue
            pred, got = float(p_side[m].mean()), float(y[m].mean())
            z = (got - pred) / max(np.sqrt(pred * (1 - pred) / n), 1e-9)
            lvl = 0
            if n >= LEARN["gate_n"] and got - pred <= -0.10 and z <= -3:
                lvl = 2
            elif n >= LEARN["gate_n"] and got - pred <= -0.05 and z <= -2:
                lvl = 1
            was = int(float(prev.get(("gate", key), {}).get("value", 0) or 0))
            tag = {0: "ok", 1: "HALF trust", 2: "PAUSED"}[lvl] + ("" if lvl == was else f" (was {({0: 'ok', 1: 'HALF', 2: 'PAUSED'})[was]})")
            out.append([SPORT, "gate", key, lvl, n, f"said {pred:.1%} hit {got:.1%} z {z:+.1f}", today])
            if lvl or was or n >= LEARN["gate_n"]:
                print(f"    {key:<22} said {pred:.1%} hit {got:.1%}  n={n:<4} z={z:+.1f} -> {tag}")
    # v2.26: pick shrink. Legs the pool would take (chance >= 50%, edge >= -3% at a real price): how much of VPE's lean
    # above the price-fair chance actually shows up in results?
    ps_, fr_, ys_ = [], [], []
    for side in ("Over", "Under"):
        odds = d["oo"] if side == "Over" else d["uo"]
        oth = d["uo"] if side == "Over" else d["oo"]
        ok = odds.str.match(r"^[+-]?\d+(\.0)?$").to_numpy()
        two = ok & oth.str.match(r"^[+-]?\d+(\.0)?$").to_numpy()
        p_side = np.where(side == "Over", q, 1 - q)
        pk_side = d["pk"].to_numpy() if side == "Over" else 1 - d["pk"].to_numpy()
        y = d["over"].to_numpy() if side == "Over" else 1 - d["over"].to_numpy()
        for i in np.where(ok)[0]:
            o = float(odds.iloc[i])
            if not (PARLAY["min_odds"] <= o <= PARLAY["max_odds"]):
                continue
            if p_side[i] >= PICK["pool_prob"] and p_side[i] - implied_prob(o) >= PICK["pool_edge"]:
                ps_.append(p_side[i]); fr_.append(_price_fair(o, two[i], pk_side[i])); ys_.append(y[i])
    n = len(ps_)
    if n >= PICK["min_n"]:
        said, fair, got = float(np.mean(ps_)), float(np.mean(fr_)), float(np.mean(ys_))
        raw = float(np.clip((said - got) / (said - fair), 0.0, 1.0)) if said - fair > 0.003 else 0.0
        k = round(raw * n / (n + PICK["n0"]), 3)
        was = prev.get(("pick", "shrink"), {}).get("value")
        out.append([SPORT, "pick", "shrink", k, n, f"picked legs said {said:.1%} hit {got:.1%}; price-fair {fair:.1%}; raw {raw:.0%}", today])
        print(f"  pick shrink: {n} pool legs said {said:.1%}, hit {got:.1%}, the price implied {fair:.1%} -> shrink {k:.0%}"
              + (f" (was {float(was):.0%})" if was not in (None, "") else " (first fit)"))
    else:
        print(f"  pick shrink: only {n} graded pool legs (needs {PICK['min_n']}) -> stays at the starting {PICK['k0']:.0%}")
    # v2.31 SIDE WATCH (report only, nothing applied): on two-sided lines, when VPE leans Over vs Under, does the lean show up?
    # 11/18 check: on ~5,000 graded lines there was no Over/Under difference worth a rule (two-sided: Over leans 140, said
    # 49.3% hit 52.9%; Under leans 225, said 51.3% hit 48.9%; both inside 1 standard error), so the win-chance math is unchanged.
    _both = (d["oo"].str.match(r"^[+-]?\d+(\.0)?$") & d["uo"].str.match(r"^[+-]?\d+(\.0)?$")).to_numpy()
    _pk, _ov = d["pk"].to_numpy(), d["over"].to_numpy()
    _sw = []
    for side, m in (("Over", _both & (q - _pk > 0.005)), ("Under", _both & (q - _pk < -0.005))):
        n_ = int(m.sum())
        if n_ < 30:
            _sw.append(f"{side} leans {n_} (too few)")
            continue
        said_ = float((q[m] if side == "Over" else 1 - q[m]).mean()); got_ = float((_ov[m] if side == "Over" else 1 - _ov[m]).mean())
        z_ = (got_ - said_) / max(np.sqrt(said_ * (1 - said_) / n_), 1e-9)
        _sw.append(f"{side} leans {n_}: said {said_:.1%} hit {got_:.1%} (z {z_:+.1f})" + (" FLAG for upgrade chat" if abs(z_) >= 2 and n_ >= 300 else ""))
    print("  side watch (two-sided lines, report only): " + " | ".join(_sw))
    # v2.28: retune every 25 graded bets (KILL list), logged to the 'Tune log' tab
    pk_ = {str(r["key"]) for (kd, _), r in prev.items() if kd == "kill" and str(r["value"]).strip() in ("1", "1.0")}
    if not any(kd == "kill" for (kd, _) in prev):
        pk_ = {"odds:short"}
    kill_, trows = (retune(a.log, getattr(a, "tune_log", None), pk_, today) if getattr(a, "log", None) and os.path.exists(a.log) else (pk_, []))
    for k_ in sorted(kill_ | pk_):
        out.append([SPORT, "kill", k_, 1 if k_ in kill_ else 0, "", "KILL list: no unboosted tier / lean / parlay leg here" if k_ in kill_ else "lifted", today])
    json.dump(trows, open(getattr(a, "tune_out", None) or "tune_new.json", "w"))
    if trows:
        print(f"  Tune log: {len(trows)} rows -> {getattr(a, 'tune_out', None) or 'tune_new.json'} (append to 'Tune log'!A1)")
    out += situational_watch(d, q, today)
    json.dump(out, open(a.out, "w"))
    print(f"Learned table -> {a.out}: clear 'Learned {SPORT}'!A:G, then write it at 'Learned {SPORT}'!A1. Save it as learned_tab.json"
          f" next to calib_tab.json before check/parlay/final/takeif.")


# ---------- NEWS SWEEP (v2.6 / v15.9): who to check in the news before anything is placed ----------
# Every run: one news search per game + one per player VPE would put money on (slip, the parlay, top props, SGP legs).
# What the searches find goes into inj.txt (Out / Doubtful / Questionable / Limited) and `check` is rerun, so a minutes
# restriction mentioned in a presser but missing from the injury page still reaches the projection.
SWEEP = {"top": 10, "per_game_queries": 1}


SB = {"min_bets": 30, "min_bucket": 400, "cal_max": 3.0, "clv_min": 0.50}  # sample-size guardrails for the scoreboard



KILL = {"short_odds": -130}   # v2.27 KILL list (approved 11/17): nothing priced -130 or shorter without a boost. 56 graded props there
# hit 53.6% vs 62.6% needed, 18 tickets returned -32%, and the Shadow tab lost ~17% on thousands of lines shorter than -200.


# v2.28 RETUNE LOOP (approved 11/17). Every 25 newly graded bets, `learn --log ... --tune-log ...` re-checks each slice of the
# graded props (price band, market, side) and moves it on or off the KILL list. Guardrails: a slice is judged only with 30+
# graded props; KILL needs a miss beyond 2 standard errors, LIFT needs the slice back AT OR ABOVE break-even (so one hot or cold week cannot flip
# it); nothing is retuned between 25-bet marks. Every decision is a row in the 'Tune log' tab: what changed, why, before/after.
TUNE = {"every": 25, "min_n": 30, "kill_se": 2.0, "lift_se": 0.0}
TUNE_COLS = ["date", "sport", "graded_bets", "slice", "action", "n", "hit", "needed", "gap_pts", "why"]


def _kl():
    ks = {"odds:short"} if LEARNED.get("kill") is None else LEARNED["kill"]
    return ", ".join(sorted(ks)) or "empty"


def _killed(market=None, side=None, odds=None):
    ks = LEARNED.get("kill")
    ks = {"odds:short"} if ks is None else ks
    if "odds:short" in ks and odds is not None and pd.notna(odds) and float(odds) <= KILL["short_odds"]:
        return True
    return (market is not None and f"market:{market}" in ks) or (side is not None and market not in ("spread", "ml") and f"side:{side}" in ks)


def retune(log_path, tune_path, prev_kill, today):
    """Returns (kill keys now, rows for the Tune log). No rows = fewer than 25 newly graded bets: nothing changes."""
    d = read_log(log_path)
    res = d["result"].astype(str).str.upper()
    bt = d.get("bet_type", pd.Series("straight", index=d.index)).fillna("straight")
    g = d[res.isin(["WIN", "LOSS"])].copy()
    g["w"] = (res[g.index] == "WIN").astype(float)
    n_bets = int((bt[g.index] != "leg").sum())
    last = 0
    if tune_path and os.path.exists(tune_path):
        for r in _values(tune_path):
            if len(r) >= 3 and str(r[1]).upper() == SPORT:
                try:
                    last = max(last, int(float(r[2])))
                except (TypeError, ValueError):
                    pass
    if n_bets - last < TUNE["every"]:
        print(f"  retune: {n_bets - last} newly graded bets since the last retune (every {TUNE['every']}) -> thresholds unchanged")
        return prev_kill, []
    p = g[bt[g.index].isin(["straight", "leg"])].copy()
    p["odds"] = pd.to_numeric(p["odds"], errors="coerce")
    p = p[p["odds"].notna()]
    p["need"] = p["odds"].map(implied_prob)
    boosted = pd.to_numeric(p.get("boost_pct"), errors="coerce").fillna(0) > 0
    slices = {"odds:short": p[(p["odds"] <= KILL["short_odds"]) & ~boosted]}
    for mk, x in p.groupby("market"):
        slices[f"market:{mk}"] = x
    _ou = p[~p["market"].isin(["spread", "ml"])]   # v2.30: a spread / moneyline has no Over or Under side
    for sd, x in _ou.groupby(_ou["side"].astype(str).str.title()):
        if sd in ("Over", "Under"):
            slices[f"side:{sd}"] = x
    kill, rows = set(prev_kill), []
    allg = g[bt[g.index] != "leg"]
    st = pd.to_numeric(allg["stake_usd"], errors="coerce").fillna(pd.to_numeric(allg["units"], errors="coerce"))
    pf = pd.to_numeric(allg["profit_units"], errors="coerce").fillna(0)
    r25 = pf.tail(TUNE["every"]).sum() / max(st.tail(TUNE["every"]).sum(), 1e-9)
    rows.append([today, SPORT, n_bets, "ALL BETS", "checkpoint", n_bets, "", "", "",
                 f"last {TUNE['every']} bets ROI {r25:+.1%}; all {n_bets} bets ROI {pf.sum() / max(st.sum(), 1e-9):+.1%} (${pf.sum():+.2f})"])
    print(f"  retune at {n_bets} graded bets: last {TUNE['every']} ROI {r25:+.1%}")
    for key, x in sorted(slices.items()):
        n = len(x)
        if n < 5:
            continue
        hit, need = float(x["w"].mean()), float(x["need"].mean())
        se = math.sqrt(max(hit * (1 - hit), 1e-9) / n)
        gap, was = hit - need, key in kill
        act = "keep"
        if n < TUNE["min_n"]:
            why = f"only {n} graded props (needs {TUNE['min_n']}): too early, no change"
        elif not was and gap < -TUNE["kill_se"] * se:
            act, why = "KILL", f"hit {hit:.1%} vs {need:.1%} needed, a {abs(gap) / se:.1f}-SE miss on {n} props"
            kill.add(key)
        elif was and gap > -TUNE["lift_se"] * se:
            act, why = "LIFT", f"back at or above break-even ({hit:.1%} vs {need:.1%} on {n} props)"
            kill.discard(key)
        else:
            why = ("stays killed" if was else "inside the margin of error") + f" ({hit:.1%} vs {need:.1%}, {n} props)"
        if act != "keep" or was or n >= TUNE["min_n"]:
            rows.append([today, SPORT, n_bets, key, act, n, round(hit, 3), round(need, 3), round(100 * gap, 1), why])
        if act != "keep":
            print(f"    {act} {key}: {why}")
    return kill, rows


def cmd_newsflash(a):
    """v2.27 NEWS SPEED: after an injury / lineup change and a rerun of `check`, compare the new run with the run before it
    and list the props whose projection moved while the recorded line and price did not: the lines the book has not
    corrected yet. Ranked by the edge it opened; role players and secondary stats are corrected slowest, stars' points fastest."""
    import glob
    new = pd.read_csv(a.all)
    prev_path = a.prev
    if not prev_path:
        runs = sorted(glob.glob(os.path.join(a.runs, "bets_all_*.csv")))
        same = [r for r in runs if os.path.getsize(r) and not pd.read_csv(r).equals(new)]
        prev_path = same[-1] if same else None
    if not prev_path or not os.path.exists(prev_path):
        print("NEWSFLASH: no earlier run to compare with (run `check` before the news, then again after it)")
        return
    old = pd.read_csv(prev_path)
    k = ["player", "market", "side", "line"]
    for d in (new, old):
        for c in ("model_prob", "blend_prob", "edge", "odds", "mean", "line"):
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
    m = new.merge(old[k + ["model_prob", "blend_prob", "edge", "odds", "mean"] + (["injury"] if "injury" in old else [])], on=k, suffixes=("", "_old"))
    inj = []
    if "injury" in new and "injury_old" in m:
        ch = m[m["injury"].fillna("").astype(str) != m["injury_old"].fillna("").astype(str)].drop_duplicates("player")
        inj = [f"{r['player']} ({r['injury_old'] or 'active'} -> {r['injury'] or 'active'})" for _, r in ch.iterrows()]
    m = m[(m["odds"] == m["odds_old"]) & m["model_prob"].notna() & m["model_prob_old"].notna()]
    m["d_model"] = m["model_prob"] - m["model_prob_old"]
    m["d_edge"] = m["edge"] - m["edge_old"]
    f = m[(m["d_model"] >= a.min_move) & (m["edge"] >= a.min_edge) & m["odds"].between(PARLAY["min_odds"], PARLAY["max_odds"])].copy()
    if "injury" in f:
        f = f[f["injury"].fillna("").astype(str).str.strip() == ""]
    star_stats = {"pts", "pra", "pass_yds", "rush_yds"}
    def speed(r):
        role = pd.notna(r.get("base_min")) and float(r["base_min"]) < 28
        if r["market"] in star_stats and not role:
            return 0, "FAST: a star's main stat, books fix it in minutes"
        if role and r["market"] not in star_stats:
            return 2, "SLOW: role player + secondary stat, often open the longest"
        return 1, "MEDIUM"
    sp = [speed(r) for _, r in f.iterrows()]
    f["_sp"], f["speed"] = [x[0] for x in sp], [x[1] for x in sp]
    f = f.sort_values(["d_edge", "edge"], ascending=False).drop_duplicates(["player", "market"]).head(a.top)
    print(f"NEWSFLASH [{VERSION}]: this run vs {os.path.basename(prev_path)} | {len(m)} lines at an unchanged price"
          + (f" | status changes: {'; '.join(inj[:8])}" if inj else ""))
    if f.empty:
        print(f"  nothing moved {a.min_move:.0%}+ in VPE's favour at an unchanged price: the news is not worth a bet on these lines")
        return
    print("  props the news moved that the recorded price has NOT caught up with (bet the FAST ones first, they close first):")
    for _, r in f.sort_values(["_sp", "d_edge"], ascending=[True, False]).iterrows():
        mo = r.get("min_odds")
        print(f"    {r['player']} {r['side']} {r['line']:g} {r['market']} @ {int(r['odds']):+d}: proj {r['mean_old']:.1f} -> {r['mean']:.1f} | "
              f"model {r['model_prob_old']:.1%} -> {r['model_prob']:.1%} | edge {r['edge_old']:+.1%} -> {r['edge']:+.1%}"
              + (f" | take-if >= {int(mo):+d}" if pd.notna(mo) else "") + f" | {r['speed']}"
              + (" | gap over the HOLD limit: here the gap IS the news, confirm the news is real before betting"
                 if pd.notna(r.get("market_gap")) and float(r["market_gap"]) > CONFIG["model"].get("max_market_gap", 0.15) else ""))
    print("  check each price in the app before betting: if it already moved, `reprice` it. Injured players are left out; the slip itself still follows the HOLD rule.")


def cmd_scoreboard(a):
    """v2.25 / v15.26: THE scoreboard, printed after every graded run. Winning is not a hit rate. Four tests:
    1 beat the break-even % of the odds actually taken (after boosts) · 2 positive closing-line value on most bets ·
    3 calibration error under 3 points per bucket (every line VPE scored, from the Shadow tab) · 4 positive ROI by tier.
    Guardrail: a group under 30 graded bets (or a bucket under 400 lines) is shown but never called PASS or FAIL."""
    d = read_log(a.log)
    for c in ("odds", "odds_eff", "stake_usd", "units", "profit_units", "model_prob", "closing_odds", "boost_pct"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    if getattr(a, "since", None) and "date" in d:
        d = d[d["date"].astype(str).str.lstrip("'").str[:10] >= str(a.since)]
    if getattr(a, "since_week", None) and "week" in d:
        d = d[pd.to_numeric(d["week"], errors="coerce") >= int(a.since_week)]
    res = d["result"].astype(str).str.upper()
    d = d.assign(bt=d.get("bet_type", pd.Series("straight", index=d.index)).fillna("straight").replace("", "straight"))
    g = d[res.isin(["WIN", "LOSS"])].copy()
    g["w"] = (res[g.index] == "WIN").astype(float)
    eff = g["odds_eff"].fillna(g["odds"])
    g["be"] = eff.map(lambda o: implied_prob(o) if pd.notna(o) and o != 0 else np.nan)
    g["stake"] = g["stake_usd"].fillna(g["units"])
    sug = g.get("suggested", pd.Series("", index=g.index)).fillna("").astype(str)
    g["sg"] = np.where(sug.str.startswith("yes"), "yes", np.where(sug == "lean", "lean", np.where(sug == "sgp", "sgp", "no")))
    vv = (g.get("vpe_verdict", pd.Series("", index=g.index)).fillna("").astype(str) + " " + g.get("note", pd.Series("", index=g.index)).fillna("").astype(str))
    g["medal"] = np.where(vv.str.contains("Gold"), "Gold", np.where(vv.str.contains("Silver"), "Silver", ""))
    g.loc[vv.str.contains("boost not used"), "sg"] = "no"
    st, pa, lg = g[g["bt"] == "straight"], g[g["bt"] == "parlay"], g[g["bt"] == "leg"]
    staked = g[g["bt"] != "leg"]
    props = g[g["bt"].isin(["straight", "leg"])]
    MB = SB["min_bets"]
    lights = []

    def call(ok, n, need=MB):
        return "too early" if n < need else ("PASS" if ok else "FAIL")

    def be_line(lab, x):
        x = x[x["be"].notna()]
        if not len(x):
            return None
        hit, need = x["w"].mean(), x["be"].mean()
        v = call(hit > need, len(x))
        print(f"    {lab:26s} {len(x):4d}  hit {hit:6.1%}  needed {need:6.1%}  {100 * (hit - need):+6.1f} pts  {v}")
        return v

    def roi_line(lab, x):
        if not len(x) or not x["stake"].sum():
            return None
        roi = x["profit_units"].sum() / x["stake"].sum()
        v = call(roi > 0, len(x))
        print(f"    {lab:26s} {len(x):4d}  staked ${x['stake'].sum():9.2f}  profit ${x['profit_units'].sum():+9.2f}  ROI {roi:+7.1%}  {v}")
        return v

    span = ""
    if "date" in g and len(g):
        dd = g["date"].astype(str).str.lstrip("'").str[:10]
        span = f"{dd.min()} to {dd.max()}"
    elif "week" in g and len(g):
        span = f"weeks {int(pd.to_numeric(g['week']).min())}-{int(pd.to_numeric(g['week']).max())}"
    print(f"VPE SCOREBOARD [{VERSION}] {span} | {len(staked)} graded bets, {len(props)} graded props (straights + legs)")
    print(f"  (a group under {MB} graded bets is 'too early': shown, never called PASS or FAIL)")
    # ---- 1 beat break-even
    print("  1 BEAT BREAK-EVEN: hit rate vs the break-even % of the odds taken (after boosts)")
    v1 = [be_line("VPE straight picks", st[st["sg"] == "yes"]), be_line("VPE parlay legs", lg[lg["sg"] == "yes"]),
          be_line("VPE parlays", pa[pa["sg"] == "yes"])]
    be_line("VPE leans (no edge)", staked[staked["sg"] == "lean"]); be_line("own picks", staked[staked["sg"] == "no"])
    be_line("same-game parlays", staked[staked["sg"] == "sgp"])
    allv = pd.concat([st[st["sg"] == "yes"], lg[lg["sg"] == "yes"]])
    allv = allv[allv["be"].notna()]
    t1 = call(len(allv) and allv["w"].mean() > allv["be"].mean(), len(allv))
    if len(allv):
        print(f"    -> VPE props overall: {len(allv)} graded, hit {allv['w'].mean():.1%} vs {allv['be'].mean():.1%} needed: {t1}")
    lights.append(("beat break-even", t1))
    # ---- 2 CLV
    print("  2 CLV: price taken vs the fair closing price (beating the close = the earliest sign of real edge)")
    pr_all = d[d["bt"].isin(["straight", "leg"]) & d["odds"].notna()]
    c = pr_all[pr_all["closing_odds"].notna()]
    if len(c):
        clv = c["closing_odds"].map(lambda x: 1 / american_to_decimal(x)) - c["odds"].map(implied_prob)
        t2 = call((clv > 0).mean() > SB["clv_min"], len(c))
        print(f"    {len(c)} of {len(pr_all)} priced props have a closing price | beat the close {(clv > 0).mean():.0%} | avg {100 * clv.mean():+.1f} pts: {t2}")
    else:
        t2 = "NO DATA"
        print(f"    0 of {len(pr_all)} priced props have a closing price -> NO DATA (closing capture is not working: see the 'Close log' tab)")
    lights.append(("positive CLV", t2))
    # ---- 3 calibration
    print(f"  3 CALIBRATION: what VPE said vs what happened (target: under {SB['cal_max']:g} pts in every bucket)")
    rows, src = [], None
    for f in (getattr(a, "shadow", None) or []):
        if os.path.exists(f):
            rows += [list(r) for r in _values(f) if r and str(r[0]) != "key"]
    if rows:
        s = pd.DataFrame([r[:11] + [""] * (11 - len(r[:11])) for r in rows], columns=SHADOW_COLS + ["p_vpe"])
        for c_ in ("p_model", "p_market", "over", "p_vpe"):
            s[c_] = pd.to_numeric(s[c_], errors="coerce")
        s = s[s["over"].isin([0, 1]) & s["p_model"].notna() & s["p_market"].notna()].drop_duplicates(["key", "player", "market", "line"])
        if len(s):
            set_calib(getattr(a, "calib", None), SPORT)
            now_p = blend_leg(s["p_model"].to_numpy(), s["p_market"].to_numpy(), s["market"], np.array(["Over"] * len(s)))
            s["said"] = s["p_vpe"].fillna(pd.Series(now_p, index=s.index))
            n_then = int(s["p_vpe"].notna().sum())
            cal = s.assign(win=s["over"])
            src = (f"{len(s)} graded lines from the Shadow tab ({s['key'].nunique()} slates; "
                   + (f"{n_then} at the chance VPE gave that night, " if n_then else "")
                   + f"{len(s) - n_then} re-scored with today's learned mix)")
            edges, need = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], SB["min_bucket"]
    if src is None:
        cal = props[props["model_prob"].notna()].assign(said=lambda x: x["model_prob"], win=lambda x: x["w"])
        src = f"{len(cal)} graded props from the pick log only (pass --shadow for every scored line)"
        edges, need = [0, .4, .5, .55, .6, .65, .7, 1.0], 100
    print(f"    source: {src}")
    worst, judged = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = cal[(cal["said"] > lo) & (cal["said"] <= hi)]
        if not len(x):
            continue
        err = 100 * (x["said"].mean() - x["win"].mean())
        ok = len(x) >= need
        judged += ok
        worst = max(worst, abs(err)) if ok else worst
        print(f"    said {lo:4.0%}-{hi:4.0%}: {len(x):5d} lines  said {x['said'].mean():6.1%}  hit {x['win'].mean():6.1%}  error {err:+5.1f} pts  "
              + ("too few to judge" if not ok else ("ok" if abs(err) < SB["cal_max"] else "OVER the limit")))
    t3 = "too early" if not judged else ("PASS" if worst < SB["cal_max"] else "FAIL")
    print(f"    -> worst judged bucket {worst:.1f} pts: {t3}")
    lights.append(("calibration", t3))
    # ---- 4 ROI by tier
    print("  4 ROI BY TIER (staked bets, $)")
    tv = []
    tier = staked.get("tier", pd.Series("", index=staked.index)).fillna("").astype(str)
    for t_, lab in (("tier_1", "Tier 1"), ("tier_2", "Tier 2"), ("tier_3", "Tier 3")):
        tv.append(roi_line(lab, staked[tier == t_]))
    roi_line("off-model / below tiers", staked[tier.isin(["off_model", "below_tier", ""])])
    roi_line("same-game (rate-only)", staked[tier == "sgp_correlated"])
    tiered = staked[tier.isin(["tier_1", "tier_2", "tier_3"])]
    t4 = call(len(tiered) and tiered["stake"].sum() and tiered["profit_units"].sum() > 0, len(tiered))
    lights.append(("ROI by tier", t4))
    print(f"    -> tiered bets together: {len(tiered)} graded: {t4}")
    # ---- splits
    print("  SPLITS (information; small groups are noise)")
    roi_line("ALL staked bets", staked); roi_line("straights", st); roi_line("parlays", pa)
    for k_, lab in (("yes", "suggested = yes (VPE)"), ("lean", "suggested = lean"), ("no", "not suggested (own)"), ("sgp", "same-game")):
        roi_line(lab, staked[staked["sg"] == k_])
    bst = staked["boost_pct"].fillna(0) > 0
    roi_line("boosted", staked[bst]); roi_line("not boosted", staked[~bst])
    roi_line("VPE-built boost parlays", staked[bst & (staked["sg"] == "yes") & (staked["bt"] == "parlay")])
    for sd in ("Over", "Under"):
        be_line(f"{sd}s (straights + legs)", props[(props["side"].astype(str).str.title() == sd) & ~props["market"].isin(["spread", "ml"])])
    for md in ("Gold", "Silver"):
        be_line(f"{md} legs", props[props["medal"] == md])
    for mk, x in props.groupby("market"):
        if len(x) >= 5:
            be_line(f"{mk}", x)
    npass = sum(v == "PASS" for _, v in lights)
    print("  SCOREBOARD: " + " | ".join(f"{k}: {v}" for k, v in lights) + f"  -> {npass} of 4 green")


# ---------- v2.31: cross-book gap finder ----------
# The book's price has been the best predictor (11/16 audit), so one honest way to an edge is a price at one of his
# two books that is better than what the OTHER book says is fair. For every prop / game line both books post at the
# same line: the other book's no-vig chance for that side minus the break-even of this book's price = the gap.
BOOKGAP = {"take": 0.02, "watch": 0.01}
BOOKGAP_COLS = ["date", "bet", "market", "side", "line", "book", "odds", "other_book", "other_fair", "gap", "vpe_win", "verdict", "result"]

def _bg_props(path):
    d = _read_loose_csv(path)
    d["mk"] = d["market"].map(lambda m: canon_market(m) or str(m))
    d["_k"] = d["player"].map(norm_name)
    d["line"] = pd.to_numeric(d["line"], errors="coerce")
    for c in ("over_odds", "under_odds"):
        d[c] = d[c].map(_odds) if c in d else None
    return d.dropna(subset=["line"])

def _bg_games(path):
    g = _read_loose_csv(path)
    g["home"], g["away"] = g["home"].map(norm_team), g["away"].map(norm_team)
    rows = []
    for _, r in g.iterrows():
        f = lambda c: _odds(r.get(c)) if c in g.columns else None
        n = lambda c: pd.to_numeric(str(r.get(c)).replace("+", "").replace("PK", "0").replace("pk", "0"), errors="coerce") if c in g.columns else np.nan
        hs, tot = n("home_spread"), n("total")
        if pd.notna(hs):
            rows.append({"player": r["home"], "_k": r["home"], "mk": "spread", "line": float(hs), "over_odds": f("spread_home_odds"), "under_odds": f("spread_away_odds"),
                         "lab_o": f"{r['home']} {hs:+g}", "lab_u": f"{r['away']} {-hs:+g}"})
        if pd.notna(tot):
            mu = f"{r['away']}@{r['home']}"
            rows.append({"player": mu, "_k": mu, "mk": "total", "line": float(tot), "over_odds": f("over_odds"), "under_odds": f("under_odds"),
                         "lab_o": f"{mu} Over {tot:g}", "lab_u": f"{mu} Under {tot:g}"})
        rows.append({"player": r["home"], "_k": r["home"], "mk": "ml", "line": 0.0, "over_odds": f("home_ml"), "under_odds": f("away_ml"),
                     "lab_o": f"{r['home']} ML", "lab_u": f"{r['away']} ML"})
    return pd.DataFrame(rows)

def cmd_bookgap(a):
    """Props and game lines posted at BOTH of his books at the same line: where one book's price beats the other book's
    fair (no-vig) chance. TAKE = gap >= 2 pts and VPE's own win chance does not disagree. Rate + suggest; logged like picks."""
    import json
    na, nb = a.book_a.lower(), a.book_b.lower()
    A, B = _bg_props(a.a), _bg_props(a.b)
    for X in (A, B):
        X["lab_o"] = X["player"].astype(str) + " Over " + X["line"].map("{:g}".format) + " " + X["mk"]
        X["lab_u"] = X["player"].astype(str) + " Under " + X["line"].map("{:g}".format) + " " + X["mk"]
    if a.games_a and a.games_b and os.path.exists(a.games_a) and os.path.exists(a.games_b):
        A, B = pd.concat([A, _bg_games(a.games_a)], ignore_index=True), pd.concat([B, _bg_games(a.games_b)], ignore_index=True)
    allp = pd.read_csv(a.all) if a.all and os.path.exists(a.all) else pd.DataFrame()
    def vpe(k, mk, side, line):
        if not len(allp):
            return None, False
        sd = "Over" if mk in ("spread", "ml") else side
        h = allp[(allp["player"].map(norm_name) == norm_name(k)) & (allp["market"] == mk) & (allp["side"] == sd)
                 & ((pd.to_numeric(allp["line"], errors="coerce") - line).abs() < 1e-6)]
        if h.empty:
            return None, bool(held_reason(allp, k, mk, sd, line))   # v2.35: no exact line, but the player can still be HELD
        r = h.iloc[0]
        p = float(r["blend_prob"])
        return p, bool(held_reason(allp, k, mk, sd, line))
    FAIR = {}
    if getattr(a, "fair", None) and os.path.exists(a.fair):
        fr = pd.read_csv(a.fair)
        col = {"draftkings": ("fair_ex_dk", "n_ex_dk"), "fanduel": ("fair_ex_fd", "n_ex_fd")}
        for _, x in fr.iterrows():
            e = {}
            for bkn in (na, nb):
                if bkn in col and pd.notna(x.get(col[bkn][0])) and pd.notna(x.get(col[bkn][1])) and int(x[col[bkn][1]]) >= 2:
                    e[bkn], e["n_" + bkn] = float(x[col[bkn][0]]), int(x[col[bkn][1]])
            if e:
                FAIR[(norm_name(x["player"]), canon_market(x["market"]) or str(x["market"]), float(x["line"]))] = e
    key = ["_k", "mk", "line"]
    M = A.drop_duplicates(key).merge(B.drop_duplicates(key), on=key, suffixes=("_a", "_b"), how="outer" if FAIR else "inner")
    for c_ in ("lab_o", "lab_u", "player"):   # a line only one book posts can still be judged against the other books
        if c_ + "_a" in M and c_ + "_b" in M:
            M[c_ + "_a"] = M[c_ + "_a"].fillna(M[c_ + "_b"])
    rows, arbs = [], []
    date = a.date or str(today_et().date())
    for _, r in M.iterrows():
        oa, ua, ob, ub = r["over_odds_a"], r["under_odds_a"], r["over_odds_b"], r["under_odds_b"]
        ok = lambda x: x is not None and not pd.isna(x)
        fa = implied_prob(oa) / (implied_prob(oa) + implied_prob(ua)) if ok(oa) and ok(ua) else None   # book a's fair P(over side)
        fb = implied_prob(ob) / (implied_prob(ob) + implied_prob(ub)) if ok(ob) and ok(ub) else None
        mk, line = r["mk"], float(r["line"])
        refa, refb = na, nb
        fx = FAIR.get((r["_k"], mk, line))
        if fx:   # v2.33: judge each book against the average no-vig chance of the OTHER books in the feed (2+ books)
            if fx.get(na) is not None:
                fb, refb = fx[na], f"{fx['n_' + na]} other books"      # reference for a bet placed at book a
            if fx.get(nb) is not None:
                fa, refa = fx[nb], f"{fx['n_' + nb]} other books"      # reference for a bet placed at book b
        for side, price, bk, ref, refbk, lab in (("Over", oa, na, fb, refb, r["lab_o_a"]), ("Under", ua, na, None if fb is None else 1 - fb, refb, r["lab_u_a"]),
                                                 ("Over", ob, nb, fa, refa, r["lab_o_a"]), ("Under", ub, nb, None if fa is None else 1 - fa, refa, r["lab_u_a"])):
            if not ok(price) or ref is None or abs(float(price)) < 100:
                continue
            price = int(price)
            gap = ref - implied_prob(price)
            if gap < BOOKGAP["watch"]:
                continue
            # a spread / ml "Under" row is the other team: look it up as that team's own row
            pl_, sd_, ln_ = (r["player_a"] if mk not in GAME_MK else r["_k"]), side, line
            if mk in ("spread", "ml"):   # the "Under" side of a spread / ml row is the other team: store it as that team's own row
                pl_, sd_, ln_ = (lab.split(" ")[0], "Over", (-line if mk == "spread" else 0.0)) if side == "Under" else (r["_k"], "Over", line)
            pv, held = vpe(pl_, mk, sd_, ln_)
            if price <= KILL["short_odds"] and _killed(odds=price):
                verdict = "KILL price (-130 or shorter): boost leg only"
            elif price < PARLAY["min_odds"]:
                verdict = "skip: shorter than -200"
            elif held:
                verdict = "HELD: not a VPE play (" + (held_reason(allp, pl_, mk, sd_, ln_) or "held") + ")"
            elif gap >= BOOKGAP["take"] and (pv is None or pv >= implied_prob(price)):
                verdict = "TAKE" + ("" if pv is not None else " (VPE has no number: book gap only)")
            elif gap >= BOOKGAP["take"]:
                verdict = "WATCH: VPE's own chance is under break-even"
            else:
                verdict = "WATCH"
            rows.append({"date": date, "bet": lab, "market": mk, "side": sd_, "line": ln_, "book": bk, "odds": price, "other_book": refbk,
                         "other_fair": round(ref, 3), "gap": round(gap, 3), "vpe_win": None if pv is None else round(pv, 3), "verdict": verdict, "result": "",
                         "_k": pl_})
        if ok(oa) and ok(ub) and 1 / american_to_decimal(oa) + 1 / american_to_decimal(ub) < 0.999:
            arbs.append(f"{r['lab_o_a']} {int(oa):+d} at {na} + {r['lab_u_a']} {int(ub):+d} at {nb}: both sides together return {1 / (1 / american_to_decimal(oa) + 1 / american_to_decimal(ub)) - 1:+.1%} whatever happens")
        if ok(ob) and ok(ua) and 1 / american_to_decimal(ob) + 1 / american_to_decimal(ua) < 0.999:
            arbs.append(f"{r['lab_o_a']} {int(ob):+d} at {nb} + {r['lab_u_a']} {int(ua):+d} at {na}: both sides together return {1 / (1 / american_to_decimal(ob) + 1 / american_to_decimal(ua)) - 1:+.1%} whatever happens")
    out = pd.DataFrame(rows, columns=BOOKGAP_COLS + ["_k"]).sort_values("gap", ascending=False)
    # different main lines for the same player / stat = a middle chance (two-sided at both books)
    two = lambda X: X[X["over_odds"].notna() & X["under_odds"].notna() & ~X["mk"].isin(["ml"])].drop_duplicates(["_k", "mk"])
    D = two(A).merge(two(B), on=["_k", "mk"], suffixes=("_a", "_b"))
    D = D[(D["line_a"] - D["line_b"]).abs() >= 0.5]
    pd.set_option("display.width", 250)
    print(f"BOOK GAPS [{VERSION}] {na} vs {nb}: {len(M)} lines posted at both books at the same line | {len(out)} where one price beats the other book's fair chance by {BOOKGAP['watch']:.0%}+")
    print(f"  TAKE = gap >= {BOOKGAP['take']:.0%} and VPE's own win chance is at or above break-even. Unproven edge: every row is saved to the 'Book gaps' tab and judged by results.")
    if len(out):
        print(out[BOOKGAP_COLS[1:-1]].head(25).to_string(index=False))
    else:
        print("  none tonight: the two books agree within their vig on every shared line")
    for x in arbs[:8]:
        print("  BOTH-SIDES GAP: " + x)
    if len(D):
        print(f"\n  DIFFERENT LINES ({len(D)}): same player and stat, the books disagree on the line itself (a middle chance; `check` prints MIDDLES when both are in props.csv):")
        for _, r in D.head(12).iterrows():
            print(f"    {r['player_a']} {r['mk']}: {na} {r['line_a']:g} ({int(r['over_odds_a']):+d}/{int(r['under_odds_a']):+d}) vs {nb} {r['line_b']:g} ({int(r['over_odds_b']):+d}/{int(r['under_odds_b']):+d})")
    take = out[out["verdict"].str.startswith("TAKE")] if len(out) else out
    print(f"\nBOOKGAP: {len(M)} shared lines, {len(take)} TAKE, {len(out) - len(take)} watch/skip, {len(arbs)} both-sides gaps, {len(D)} different lines")
    out[BOOKGAP_COLS].to_csv(a.out, index=False)
    json.dump([[("" if (isinstance(v, float) and math.isnan(v)) or v is None else v) for v in r] for r in out[BOOKGAP_COLS].itertuples(index=False, name=None)], open(a.sheet_out, "w"))
    if len(take) and a.slip and os.path.exists(a.slip):   # a TAKE is a VPE suggestion: `log` marks it suggested = yes
        s = pd.read_csv(a.slip)
        add = pd.DataFrame({"player": take["_k"].to_list(), "market": take["market"].to_list(), "side": take["side"].to_list(), "kind": "pick"})
        pd.concat([s, add], ignore_index=True).drop_duplicates(["player", "market", "side"]).to_csv(a.slip, index=False)
    print(f"saved -> {a.out} | rows for the 'Book gaps' tab -> {a.sheet_out}")


def cmd_links(a):
    """v2.33: one-tap bet-slip links (from the odds feed's links.csv) for every bet VPE suggests tonight: straight picks,
    the parlay's legs, leans and BOOK GAP takes. A link opens that book's bet slip with the line loaded."""
    import json
    if not (a.links and os.path.exists(a.links)):
        print("LINKS: no links file (the odds feed was not used this run) -> he opens the bets in the app as usual")
        return
    L = pd.read_csv(a.links)
    L["_k"] = L["bet"].map(norm_name); L["line"] = pd.to_numeric(L["line"], errors="coerce")
    want = []
    for p_ in (a.bets, a.leans):
        if p_ and os.path.exists(p_):
            b = pd.read_csv(p_)
            want += [(r["player"], r["market"], r["side"], float(r["line"]), "pick" if p_ == a.bets else "lean") for _, r in b.iterrows()]
    if a.parlays and os.path.exists(a.parlays):
        for pj in json.load(open(a.parlays)):
            want += [(l[0], l[1], l[2], float(l[3]), pj.get("name", "parlay")) for l in pj.get("legs", [])]
    if a.gaps and os.path.exists(a.gaps):
        g = pd.read_csv(a.gaps)
        g = g[g["verdict"].astype(str).str.startswith("TAKE")]
        want += [(str(r["bet"]).split(" Over ")[0].split(" Under ")[0] if r["market"] not in ("spread", "ml") else str(r["bet"]).split(" ")[0],
                  r["market"], r["side"], float(r["line"]), "book gap") for _, r in g.iterrows()]
    seen, n = set(), 0
    print("BET LINKS (tap to open the bet slip; check the price against the take-if before you place it):")
    for pl, mk, sd, ln, why in want:
        k = (norm_name(pl), mk, sd, ln)
        if k in seen:
            continue
        seen.add(k)
        h = L[(L["_k"] == k[0]) & (L["market"] == mk) & (L["side"] == sd) & ((L["line"] - ln).abs() < 1e-6)]
        lab = f"{pl} {'' if mk in ('spread', 'ml') else sd + ' '}{ln:g} {mk}" if mk != "ml" else f"{pl} ML"
        if h.empty:
            print(f"  {lab} [{why}]: no link for this line (ladder / alt line): open it in the app")
            continue
        n += 1
        print(f"  {lab} [{why}]: " + " | ".join(f"{BOOK_NAME.get(r['book'], r['book'])} {int(r['odds']):+d} {r['url']}" for _, r in h.iterrows()))
    print(f"LINKS: {n} of {len(seen)} suggested bets have a link")


def _sharepdf_short(a):
    """v2.34: a one-page (or two) shareable PDF of tonight's VPE picks: straight picks, the parlay(s) to place, leans and
    book-gap takes, each with price, take-if, win chance and a tappable bet-slip link when the odds feed has one.
    No stakes, no bankroll, nothing personal: safe to send to someone else."""
    import json, subprocess, sys
    try:
        from reportlab.lib.pagesizes import letter
    except ImportError:
        print("SETUP: reportlab missing -> pip install reportlab")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "reportlab", "--break-system-packages"], check=False)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    allp = pd.read_csv(a.all) if os.path.exists(a.all) else pd.DataFrame()
    L = pd.read_csv(a.links) if a.links and os.path.exists(a.links) else pd.DataFrame(columns=["bet", "market", "line", "side", "book", "odds", "url"])
    if len(L):
        L["_k"] = L["bet"].map(norm_name); L["line"] = pd.to_numeric(L["line"], errors="coerce")
    MKN = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "pra": "pts+reb+ast", "pr": "pts+reb", "pa": "pts+ast",
           "ra": "reb+ast", "stl": "steals", "blk": "blocks", "stocks": "steals+blocks", "tov": "turnovers", "spread": "spread", "ml": "moneyline", "total": "total"}
    def label(pl, mk, sd, ln):
        if mk == "ml":
            return f"{pl} moneyline"
        if mk == "spread":
            return f"{pl} {float(ln):+g}"
        return f"{pl} {sd} {float(ln):g} {MKN.get(mk, mk)}"
    def row_of(pl, mk, sd, ln):
        if not len(allp):
            return None
        h = allp[(allp["player"].map(norm_name) == norm_name(pl)) & (allp["market"] == mk) & (allp["side"] == sd)
                 & ((pd.to_numeric(allp["line"], errors="coerce") - float(ln)).abs() < 1e-6)]
        return h.iloc[0] if len(h) else None
    def link(pl, mk, sd, ln, book=None):
        if not len(L):
            return ""
        h = L[(L["_k"] == norm_name(pl)) & (L["market"] == mk) & (L["side"] == sd) & ((L["line"] - float(ln)).abs() < 1e-6)]
        if book:
            h = h[h["book"].astype(str).str.lower() == str(book).lower()]   # a parlay goes on one book: only that book's link
        return " &nbsp; ".join(f'<link href="{r["url"]}" color="#1a56db"><u>Open at {BOOK_NAME.get(r["book"], r["book"])} ({int(r["odds"]):+d})</u></link>' for _, r in h.iterrows())
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, spaceAfter=2, alignment=0)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9, leading=12)
    SM = ParagraphStyle("sm", parent=P, fontSize=7.5, textColor=colors.HexColor("#555555"))
    doc = SimpleDocTemplate(a.out, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title="VPE picks", author="VPE")
    E = [Paragraph(f"VPE picks: {a.title or (str(allp['date'].iloc[0]) if len(allp) and 'date' in allp else '')}", H1),
         Paragraph(f"NBA | prices as of {a.asof or 'the run'} | {VERSION}. Take a bet only at the listed take-if price or better: prices move.", SM)]
    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        return t
    W = [2.5 * inch, 0.75 * inch, 0.8 * inch, 0.75 * inch, 2.5 * inch]
    hdr = [Paragraph(f"<b>{x}</b>", P) for x in ("Bet", "Price", "Take-if", "Win chance", "Link")]
    n_links = n_bets = 0
    def bet_row(pl, mk, sd, ln, book=None, odds=None, takeif=None, prob=None):
        nonlocal n_links, n_bets
        r = row_of(pl, mk, sd, ln)
        odds = odds if odds is not None else (int(r["odds"]) if r is not None and pd.notna(r["odds"]) else None)
        prob = prob if prob is not None else (float(r["blend_prob"]) if r is not None else None)
        takeif = None if takeif == "leg" else (takeif if takeif is not None else (int(r["min_odds"]) if r is not None and pd.notna(r.get("min_odds")) else None))
        lk = link(pl, mk, sd, ln, book)
        n_bets += 1; n_links += 1 if lk else 0
        team = f" ({r['team']} vs {r['opp']})" if r is not None and mk not in GAME_MK else ""
        return [Paragraph(label(pl, mk, sd, ln) + team, P), Paragraph(f"{odds:+d}" if odds is not None else "-", P),
                Paragraph(f"{takeif:+d} or better" if takeif is not None else "see ticket", P), Paragraph(f"{prob:.0%}" if prob is not None else "-", P),
                Paragraph(lk or "ladder / alt line: find it in the app", P)]
    if a.bets and os.path.exists(a.bets):
        b = pd.read_csv(a.bets)
        if len(b):
            E.append(Paragraph("Straight bets", H2))
            E.append(table([hdr] + [bet_row(r["player"], r["market"], r["side"], r["line"]) for _, r in b.iterrows()], W))
    if a.parlays and os.path.exists(a.parlays):
        pj = json.load(open(a.parlays))
        for p_ in [x for x in pj if x.get("recommended")] or pj[:1]:
            tag = " (lean: no edge, the least-bad play)" if p_.get("lean") else ""
            E.append(Paragraph(f"{p_.get('name', 'Parlay')} at {BOOK_NAME.get(p_.get('book'), p_.get('book', ''))}{tag}", H2))
            bits = [f"about {int(p_['odds']):+d}" if p_.get("odds") is not None else "", f"take-if {int(p_['take_if']):+d} before any boost" if p_.get("take_if") is not None else "",
                    f"hit chance {float(p_['hit']):.0%}" if p_.get("hit") is not None else "", f"boost {p_['boost_pct']:g}%" if p_.get("boost_pct") else ""]
            E.append(Paragraph(" | ".join(x for x in bits if x) + ". All legs go on one ticket.", SM))
            E.append(table([hdr] + [bet_row(l[0], l[1], l[2], l[3], book=p_.get("book"), takeif="leg") for l in p_.get("legs", [])], W))
    if a.leans and os.path.exists(a.leans):
        ln_ = pd.read_csv(a.leans)
        if len(ln_):
            E.append(Paragraph("Leans (no edge: the closest lines to break-even)", H2))
            E.append(table([hdr] + [bet_row(r["player"], r["market"], r["side"], r["line"]) for _, r in ln_.iterrows()], W))
    if a.gaps and os.path.exists(a.gaps):
        g = pd.read_csv(a.gaps)
        g = g[g["verdict"].astype(str).str.startswith("TAKE")]
        if len(g):
            E.append(Paragraph("Book gaps (this book's price beats the rest of the market)", H2))
            rows = []
            for _, r in g.iterrows():
                pl = str(r["bet"]).split(" ")[0] if r["market"] in ("spread", "ml") else str(r["bet"]).split(" Over ")[0].split(" Under ")[0]
                rows.append(bet_row(pl, r["market"], r["side"], r["line"], book=r["book"], odds=int(r["odds"]), takeif=int(r["odds"]),
                                    prob=float(r["vpe_win"]) if pd.notna(r.get("vpe_win")) else float(r["other_fair"])))
            E.append(table([hdr] + rows, W))
    E += [Spacer(1, 10), Paragraph("How to read this: win chance is VPE's estimate, mixed from its own projection and the sportsbooks' prices. A link opens that "
                                   "sportsbook's bet slip with the bet loaded; check the price there against the take-if before placing. "
                                   "Information only, not betting advice. No outcome is guaranteed. 21+ and where legal.", SM)]
    doc.build(E)
    print(f"SHAREPDF: {n_bets} bets, {n_links} with a link -> {a.out}")


def held_reason(allp, player, mk, side=None, line=None):
    """v2.35 (rule order step 1): why this player / market is HELD tonight, or "" when it is not. Used by `log`,
    `bookgap` and the day card so a HELD line is never rated as a VPE play anywhere."""
    if allp is None or not len(allp):
        return ""
    m = CONFIG["model"]
    pr = allp[allp["player"].map(norm_name) == norm_name(player)]
    if pr.empty:
        return ""
    d = pr[pr["market"] == mk]
    if len(d) and side is not None:
        ds = d[d["side"] == side]
        d = ds if len(ds) else d
    if len(d) and line is not None:
        ex = d[(pd.to_numeric(d["line"], errors="coerce") - float(line)).abs() < 1e-6]
        d = ex if len(ex) else d.iloc[(pd.to_numeric(d["line"], errors="coerce") - float(line)).abs().argsort()[:1]]
    r = d.iloc[0] if len(d) else pr.iloc[0]
    inj = str(r.get("injury", "") or "").strip()
    if inj.title() in ("Out", "Doubtful"):
        return f"listed {inj.title()}"
    if pd.notna(r.get("base_min")) and float(r["base_min"]) < m["min_minutes_to_bet"]:
        return f"only {float(r['base_min']):.0f} normal minutes (under {m['min_minutes_to_bet']})" + (f", {inj.lower()}" if inj and inj != "nan" else "")
    if len(d) and pd.notna(r.get("market_gap")) and float(r["market_gap"]) > m["max_market_gap"]:
        return f"model vs market gap {float(r['market_gap']) * 100:.0f} pts (over {m['max_market_gap'] * 100:.0f})"
    return ""


def cmd_sharepdf(a):
    """v2.35: the full DAY CARD from one command: every play with its legs, the numbers behind each, the full logic per
    leg (from the same breakdown `explain` prints), bet-slip links, book gaps, shared risk and the run status. Built as
    HTML and printed with the sandbox's Chromium; falls back to the short reportlab list if Chromium is not there.
    It doubles as the run's last check: it re-applies HELD to everything it shows and prints a SHAREPDF: line."""
    import json, html as _h
    set_calib(getattr(a, "calib", None), SPORT)
    cfg = CONFIG
    allp = pd.read_csv(a.all) if os.path.exists(a.all) else pd.DataFrame()
    if not len(allp):
        print("SHAREPDF: no bets_all.csv -> run check first"); return
    lock_from(allp)
    L = pd.read_csv(a.links) if a.links and os.path.exists(a.links) else pd.DataFrame(columns=["bet", "market", "line", "side", "book", "odds", "url"])
    if len(L):
        L["_k"] = L["bet"].map(norm_name); L["line"] = pd.to_numeric(L["line"], errors="coerce")
    e = lambda s_: _h.escape(str(s_))
    MKN = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes", "pra": "pts+reb+ast", "pr": "pts+reb", "pa": "pts+ast",
           "ra": "reb+ast", "stl": "steals", "blk": "blocks", "stocks": "steals+blocks", "tov": "turnovers", "spread": "spread", "ml": "moneyline", "total": "total"}
    am = lambda o: "-" if o is None or pd.isna(o) else f"{int(o):+d}"
    pc = lambda p, d=0: "-" if p is None or pd.isna(p) else f"{float(p) * 100:.{d}f}%"
    def label(pl, mk, sd, ln):
        return f"{pl} ML" if mk == "ml" else (f"{pl} {float(ln):+g}" if mk == "spread" else f"{pl} {sd} {float(ln):g} {MKN.get(mk, mk)}")
    def row_of(pl, mk, sd, ln):
        h = allp[(allp["player"].map(norm_name) == norm_name(pl)) & (allp["market"] == mk) & (allp["side"] == sd)
                 & ((pd.to_numeric(allp["line"], errors="coerce") - float(ln)).abs() < 1e-6)]
        return h.iloc[0] if len(h) else None
    stats = {"links": 0, "nolinks": 0}
    def btn(pl, mk, sd, ln, book):
        bk = str(book or "").split("/")[0].lower()
        h = L[(L["_k"] == norm_name(pl)) & (L["market"] == mk) & (L["side"] == sd) & ((L["line"] - float(ln)).abs() < 1e-6)] if len(L) else L
        hb = h[h["book"].astype(str).str.lower() == bk] if len(h) and bk else h
        if len(hb):
            r_ = hb.iloc[0]; stats["links"] += 1
            return f'<a class="btn" href="{e(r_["url"])}">Open bet slip &middot; {e(BOOK_NAME.get(r_["book"], r_["book"]))} {am(r_["odds"])} &#8599;</a>'
        stats["nolinks"] += 1
        return f'<span class="nolink">No {e(BOOK_NAME.get(bk, bk or "book"))} link for this line: open it in the app</span>'
    def bet_at(r, book=None):
        if book:
            return str(book).lower()
        b = str(r.get("book", "")).split("/")
        return (b[0] if r["side"] == "Over" or len(b) == 1 else b[1]).lower()
    pool = pd.DataFrame()
    try:
        pool = pd.concat([parlay_pool(allp), parlay_pool(allp, (), min_edge=-0.04, min_prob=0.0)]).drop_duplicates(["player", "market", "side", "line"])
    except Exception:
        pass
    def pool_row(pl, mk, sd, ln):
        if not len(pool):
            return None
        h = pool[(pool["player"].map(norm_name) == norm_name(pl)) & (pool["market"] == mk) & (pool["side"] == sd)
                 & ((pd.to_numeric(pool["line"], errors="coerce") - float(ln)).abs() < 1e-6)]
        return h.iloc[0] if len(h) else None
    rr_cls = lambda x_: {"STABLE": "g", "OK": "y", "SHAKY": "r"}.get(str(x_), "y")
    plays, order = [], []   # order = [(pl, mk, sd, ln, book, role chip)]
    pj = json.load(open(a.parlays)) if a.parlays and os.path.exists(a.parlays) else []
    rec = [p for p in pj if p.get("recommended")]
    oth = [p for p in pj if not p.get("recommended")]
    bets = pd.read_csv(a.bets) if a.bets and os.path.exists(a.bets) else pd.DataFrame()
    leans = pd.read_csv(a.leans) if a.leans and os.path.exists(a.leans) else pd.DataFrame()
    gaps = pd.read_csv(a.gaps) if a.gaps and os.path.exists(a.gaps) else pd.DataFrame()
    slip_keys_ = set()
    def leg_rows(p_, role):
        out, mk_prob, unders, flags = [], 1.0, 0, []
        for l in p_.get("legs", []):
            pl, mk, sd, ln = l[0], l[1], l[2], float(l[3])
            r = row_of(pl, mk, sd, ln); q = pool_row(pl, mk, sd, ln)
            order.append((pl, mk, sd, ln, p_.get("book"), role)); slip_keys_.add((norm_name(pl), mk, sd))
            unders += sd == "Under"
            if r is None:
                out.append(f"<tr><td>{e(label(pl, mk, sd, ln))}</td><td colspan=5>not in tonight's scored lines</td></tr>"); continue
            mk_prob *= float(r["market_prob"]) if pd.notna(r.get("market_prob")) else float(r["blend_prob"])
            lp = float(q["leg_prob"]) if q is not None else float(r["blend_prob"])
            tm = am(q["take_min"]) if q is not None and pd.notna(q.get("take_min")) else "-"
            price = am(q["price_" + ("dk" if str(p_.get("book")) == "draftkings" else "fd")]) if q is not None and ("price_dk" in q) else am(r["odds"])
            hr = held_reason(allp, pl, mk, sd, ln)
            oth_held = [x for x in sorted(set(allp[(allp["player"].map(norm_name) == norm_name(pl)) & (pd.to_numeric(allp["market_gap"], errors="coerce") > cfg["model"]["max_market_gap"])]["market"])) if x != mk]
            if hr:
                flags.append(f"{pl}: HELD ({hr})")
            elif oth_held:
                flags.append(f"{pl}: held on his other markets ({', '.join(MKN.get(x, x) for x in oth_held)}): usually news the stats have not seen")
            if str(r.get("role_risk")) == "SHAKY":
                flags.append(f"{pl}: shaky role ({r.get('role_why')})")
            note = f"proj {float(r['mean']):.1f} in {float(r['proj_min']):.0f} min" if mk not in GAME_MK else f"VPE number {float(r['mean']):+.1f}"
            out.append(f"<tr><td><b>{e(pl)}</b> <span class='mut'>({e(r['team'])})</span><br>{e(sd if mk not in ('spread', 'ml') else '')} {e(f'{ln:g}' if mk != 'ml' else '')} {e(MKN.get(mk, mk))} "
                       f"<span class='mut'>vs {e(r['opp'])}</span></td><td class='num'>{price}</td><td class='num'>{pc(lp)}</td>"
                       f"<td><span class='chip {rr_cls(r.get('role_risk'))}'>{e(r.get('role_risk'))}</span></td><td class='num'>{tm}</td>"
                       f"<td class='sm'>{e(note)}<br>{btn(pl, mk, sd, ln, p_.get('book'))}</td></tr>")
        return out, mk_prob, unders, flags
    def parlay_box(p_, role):
        rows, mkp, unders, flags = leg_rows(p_, role)
        n = len(p_.get("legs", []))
        dec = american_to_decimal(p_["odds"]) if p_.get("odds") is not None else None
        boost = float(p_.get("boost_pct") or 0)
        decb = 1 + (dec - 1) * (1 + boost / 100) if dec else None
        hit = float(p_.get("hit") or 0)
        ev = p_.get("ev"); ev = hit * decb - 1 if ev is None and decb else ev
        mev = mkp * decb - 1 if decb else None
        price = (f"<s>{am(p_['odds'])}</s> &rarr; <b>{am(decimal_to_american(decb))}</b>" if boost else f"<b>{am(p_.get('odds'))}</b>")
        tag = "LEAN" if p_.get("lean") else ("THE PLAY" if role == "THE parlay" else "reference")
        cap = _under_max(n) if n else 0
        notes = []
        if boost and dec:
            notes.append(f"Without the boost: EV {(hit * dec - 1) * 100:+.1f}% at {am(p_['odds'])}. Place it only with the {boost:g}% boost, only at {BOOK_NAME.get(p_.get('book'), p_.get('book'))}.")
        notes.append(f"Unders {unders} of {n} legs (cap {cap}).")
        notes += flags
        return (f"<div class='play {'lean' if p_.get('lean') else ''}'><div class='ph'><span class='tag'>{tag}</span> <span class='pt'>{e(p_.get('name', 'Parlay'))} at {e(BOOK_NAME.get(p_.get('book'), p_.get('book', '')))}</span>"
                f"<span class='pp'>{price}</span></div>"
                f"<div class='sub'>Take-if before the boost: <b>{am(p_.get('take_if'))}</b> or better &middot; {n} legs on one ticket</div>"
                f"<div class='boxes'><div class='bx'><div class='bl'>hit chance</div><div class='bv'>{pc(hit, 1)}</div></div>"
                f"<div class='bx'><div class='bl'>market chance</div><div class='bv'>{pc(mkp, 1)}</div></div>"
                f"<div class='bx'><div class='bl'>EV (engine est.)</div><div class='bv {'pos' if (ev or 0) > 0 else 'neg'}'>{(ev or 0) * 100:+.1f}%</div><div class='bl'>market-only {('%+.1f%%' % (mev * 100)) if mev is not None else '-'}</div></div></div>"
                f"<table class='legs'><tr><th>Leg</th><th>Price</th><th>VPE win</th><th>Role</th><th>Take-if</th><th>Note / link</th></tr>{''.join(rows)}</table>"
                f"<div class='note'>{'<br>'.join(e(x) for x in notes)}</div></div>")
    S = []
    if rec:
        S.append("<h2>The play</h2>" + "".join(parlay_box(p_, "THE parlay") for p_ in rec))
    def straight_table(df, role, title):
        if not len(df):
            return ""
        rows = []
        for _, b in df.iterrows():
            pl, mk, sd, ln = b["player"], b["market"], b["side"], float(b["line"])
            r = row_of(pl, mk, sd, ln)
            if r is None:
                continue
            order.append((pl, mk, sd, ln, None, role)); slip_keys_.add((norm_name(pl), mk, sd))
            hr = held_reason(allp, pl, mk, sd, ln)
            why = f"proj {float(r['mean']):.1f} vs line {ln:g} in {float(r['proj_min']):.0f} min" if mk not in GAME_MK else f"VPE number {float(r['mean']):+.1f}"
            edge = float(r["blend_prob"]) - implied_prob(r["odds"])
            rows.append(f"<tr><td><b>{e(label(pl, mk, sd, ln))}</b> <span class='mut'>({e(r['team'])} vs {e(r['opp'])})</span>{'<br><span class=neg>HELD: not a VPE play (' + e(hr) + ')</span>' if hr else ''}</td>"
                        f"<td class='num'>{am(r['odds'])}</td><td class='num'>{am(r.get('min_odds'))}</td>"
                        f"<td class='num'>{pc(r['blend_prob'], 1)} / {pc(implied_prob(r['odds']), 1)} / <span class='{'pos' if edge > 0 else 'neg'}'>{edge * 100:+.1f}</span></td>"
                        f"<td class='sm'>{e(why)} &middot; <span class='chip {rr_cls(r.get('role_risk'))}'>{e(r.get('role_risk'))}</span><br>{btn(pl, mk, sd, ln, bet_at(r))}</td></tr>")
        return (f"<h2>{title}</h2><table class='legs'><tr><th>Bet</th><th>Price</th><th>Take-if</th><th>VPE win / needed / edge</th><th>Why / link</th></tr>{''.join(rows)}</table>") if rows else ""
    S.append(straight_table(bets, "STRAIGHT", "Straight bets"))
    S.append(straight_table(leans, "LEAN", "Leans (no edge: the closest lines to break-even)"))
    if oth:
        S.append("<h2>Other builds (reference only)</h2>" + "".join(parlay_box(p_, "reference") for p_ in oth[:3]))
    # book gaps
    skip_items = [x.strip() for x in (a.skip or "").split(";") if x.strip()]
    if len(gaps):
        rows = []
        for _, g in gaps.head(12).iterrows():
            pl = str(g["bet"]).split(" ")[0] if g["market"] in ("spread", "ml") else str(g["bet"]).split(" Over ")[0].split(" Under ")[0]
            hr = held_reason(allp, pl, g["market"], g["side"], g["line"])
            verdict = f"HELD: not a VPE play ({hr})" if hr else str(g["verdict"])
            opp = any((norm_name(pl), g["market"], o_) in slip_keys_ for o_ in ("Over", "Under") if o_ != g["side"]) and g["market"] not in ("spread", "ml")
            if hr and str(g["verdict"]).startswith("TAKE"):
                skip_items.append(f"{g['bet']} at {g['book']}: the book gap says TAKE, but he is HELD ({hr})")
            rows.append(f"<tr><td>{e(g['bet'])}{' <span class=neg>(opposite side of a bet on the slip)</span>' if opp else ''}</td><td>{e(BOOK_NAME.get(g['book'], g['book']))} {am(g['odds'])}</td>"
                        f"<td class='num'>{pc(g['other_fair'], 1)} <span class='mut'>({e(g['other_book'])})</span></td><td class='num'>{pc(g['vpe_win'], 1) if pd.notna(g.get('vpe_win')) else '-'}</td>"
                        f"<td class='{'pos' if verdict.startswith('TAKE') else ('neg' if verdict.startswith('HELD') else '')}'>{e(verdict)}</td></tr>")
        S.append(f"<h2>Book gaps</h2><table class='legs'><tr><th>Bet</th><th>Price</th><th>Other books' fair</th><th>VPE</th><th>Verdict</th></tr>{''.join(rows)}</table>")
    if skip_items:
        S.append("<div class='skip'><b>SKIP TONIGHT</b><br>" + "<br>".join(e(x) for x in skip_items) + "</div>")
    bl = [p for p in pj if p.get("boost_pct")]
    if bl:
        S.append("<h2>Boosts</h2><table class='legs'><tr><th>Boost</th><th>Book</th><th>Legs</th><th>EV (engine est.)</th><th>Verdict</th></tr>" + "".join(
            f"<tr><td>{e(p_.get('name'))}</td><td>{e(BOOK_NAME.get(p_.get('book'), p_.get('book')))}</td><td class='num'>{len(p_.get('legs', []))}</td><td class='num'>{(p_.get('ev') or 0) * 100:+.1f}%</td>"
            f"<td>{'LEAN: least-bad way to use it' if p_.get('lean') else ('USE IT' if (p_.get('ev') or 0) >= 0.05 else 'THIN: about break-even')}</td></tr>" for p_ in bl) + "</table>")
    # shared risk
    from collections import Counter, defaultdict
    tick = defaultdict(list)
    for pl, mk, sd, ln, bk, role in order:
        tick[norm_name(pl)].append((pl, mk, sd, role))
    sr = [f"<tr><td>{e(v[0][0])}</td><td>{e(', '.join(sorted({MKN.get(m_, m_) + ' ' + s_ for _, m_, s_, _ in v})))}</td><td>{len(v)} tickets / bets"
          + (" &middot; <span class='neg'>opposite sides</span>" if len({s_ for _, _, s_, _ in v}) > 1 and len({m_ for _, m_, _, _ in v}) == 1 else "") + "</td></tr>" for v in tick.values() if len(v) > 1]
    ou = Counter(sd for _, mk, sd, _, _, _ in order if mk not in ("spread", "ml"))
    stc = Counter(MKN.get(mk, mk) for _, mk, _, _, _, _ in order)
    S.append("<h2>Shared risk</h2><table class='legs'><tr><th>Player</th><th>Markets</th><th>Where</th></tr>" + ("".join(sr) or "<tr><td colspan=3>No player appears on more than one ticket.</td></tr>")
             + f"</table><div class='sub'>Over / Under mix: {ou.get('Over', 0)} Overs, {ou.get('Under', 0)} Unders &middot; legs per stat: {e(', '.join(f'{k} {v}' for k, v in stc.most_common()))}</div>")
    # full logic cards
    def bar(lbl, v, cls):
        return f"<div class='br'><span class='bn'>{lbl}</span><span class='bt'><span class='bf {cls}' style='width:{max(min(float(v), 1), 0) * 100:.1f}%'></span></span><span class='bv2'>{pc(v, 1)}</span></div>"
    def chipf(name, f):
        f = float(f)
        return f"<span class='chip {'g' if f > 1.005 else ('r' if f < 0.995 else 'n')}'>{e(name)} &times;{f:.3f}</span>"
    cards, seen = [], set()
    for pl, mk, sd, ln, bk, role in order:
        k = (norm_name(pl), mk, sd, ln)
        if k in seen:
            continue
        seen.add(k)
        r = row_of(pl, mk, sd, ln)
        if r is None:
            continue
        r = r.to_dict()
        try:
            x = json.loads(r["x"])
        except Exception:
            x = {}
        edge = float(r["blend_prob"]) - implied_prob(r["odds"])
        hr = held_reason(allp, pl, mk, sd, ln)
        head = (f"<div class='ch'><span class='cn'>{e(label(pl, mk, sd, ln))}</span> <span class='mut'>{e(r['team'])} vs {e(r['opp'])} &middot; {am(r['odds'])}</span> "
                f"<span class='chip {'g' if edge > 0 else 'r'}'>edge {edge * 100:+.1f}</span> <span class='chip n'>{e(role)}</span>"
                + (f" <span class='chip r'>HELD: not a VPE play</span>" if hr else "") + f"<span class='cb'>{btn(pl, mk, sd, ln, bk or bet_at(r))}</span></div>")
        body = []
        if mk in GAME_MK:
            g = x.get("game", {})
            by = lambda m_: f"{g.get('home') if m_ >= 0 else g.get('away')} by {abs(m_):.1f}"
            steps = [("1 Team ratings", f"{g.get('home')} offense {g.get('off_h', 0):+.1f} / defense {g.get('def_h', 0):+.1f} &middot; {g.get('away')} offense {g.get('off_a', 0):+.1f} / defense {g.get('def_a', 0):+.1f} &middot; home court {g.get('hca', 0):+.1f}"),
                     ("2 Injuries and rest", f"{e(g.get('inj_txt') or 'no rotation player out')} ({g.get('inj', 0):+.1f}) &middot; {e(g.get('rest_txt', 'n/a'))} ({g.get('rest', 0):+.1f})"),
                     ("3 Ratings say", f"{e(by(g.get('m_mod', 0)))}, total {g.get('t_mod', 0):.1f}"), ("4 Book says", f"{e(by(g.get('m_mkt', 0)))}, total {g.get('t_mkt', 0):.1f}"),
                     ("5 Mix", f"{g.get('w', 0) * 100:.0f}% ratings + {(1 - g.get('w', 0)) * 100:.0f}% book &rarr; VPE number {g.get('center', 0):.1f} (games land about {g.get('sd', 0):g} either side)"),
                     ("8 Price and edge", f"{am(r['odds'])} &rarr; needs {pc(implied_prob(r['odds']), 1)} &rarr; edge {edge * 100:+.1f} &rarr; {e(str(r.get('tier')).replace('_', ' ') if isinstance(r.get('tier'), str) else (g.get('block') or 'no tier'))}"),
                     ("9 Stake", f"{float(r['units']):g}u (VPE's size)" if isinstance(r.get("tier"), str) else "none as a straight"),
                     ("10 Risk", e(g.get("track", "")))]
            body.append("<div class='steps'>" + "".join(f"<div class='st'><span class='sn'>{a_}</span> {b_}</div>" for a_, b_ in steps) + "</div>")
            body.append("<div class='bars'>" + bar("ratings", r["model_prob"], "m") + bar("market", r["market_prob"], "k") + bar("VPE win", r["blend_prob"], "v") + bar("needed", implied_prob(r["odds"]), "n") + "</div>")
        else:
            m_, c_, g, p = x.get("m", {}), x.get("c", {}), x.get("g", {}), x.get("p", {})
            names = {"base": "base", "usage": "teammates out", "role": "recent role", "wo": "games without", "b2b": "back-to-back", "blow": "blowout risk",
                     "restrict": "limit", "news": "news role", "nt": "new team", "final": "projected"}
            chain, prev = [], None
            for kk in ("base", "usage", "role", "wo", "b2b", "blow", "restrict", "news", "nt", "final"):
                if kk not in m_ or m_[kk] is None:
                    continue
                v = float(m_[kk])
                if prev is None or kk == "final" or abs(v - prev) >= 0.05:
                    cls = "n" if prev is None or kk == "final" else ("g" if v > prev else "r")
                    chain.append(f"<span class='chip {cls}'>{names[kk]} {v:.1f}</span>")
                prev = v
            body.append(f"<div class='st'><span class='sn'>1 Minutes</span> {' &rarr; '.join(chain)} <span class='mut'>last 3: {c_.get('l3', '-')} &middot; last 10: {c_.get('m10', '-')} &middot; starts {float(c_.get('sr', 0) or 0) * 100:.0f}% &middot; {c_.get('n', '-')} games</span></div>")
            body.append(f"<div class='st'><span class='sn'>2 Game</span> {'home' if g.get('home') else 'away'} &middot; spread {g.get('spread') if g.get('spread') is not None else 'n/a'} &middot; total {g.get('total') if g.get('total') is not None else 'n/a'} &middot; {e(g.get('arch', ''))} vs {e(g.get('opp_style', ''))}</div>")
            chips = []
            for stn, sv in (x.get("s") or {}).items():
                chips.append(f"<span class='chip n'>{e(stn)} {float(sv.get('raw', 0)):.3f}/min</span>")
                for nm, kk in (("usage", "use"), ("defense vs position", "dvp"), ("style", "af"), ("game total", "ef")):
                    if sv.get(kk) is not None:
                        chips.append(chipf(nm, sv[kk]))
            for nm, kk in (("pace", "pace"), ("home/away", "hf"), ("back-to-back", "b2br"), ("new team", "ntr")):
                if g.get(kk) is not None:
                    chips.append(chipf(nm, g[kk]))
            body.append(f"<div class='st'><span class='sn'>3 Rate</span> {' '.join(chips)}</div>")
            mu, sd_ = float(r["mean"]), float(x.get("sd") or math.sqrt(max(float(r["var"]), 1e-6)))
            hi = max(mu + 2.5 * sd_, ln * 1.35, 1.0); pos = lambda v: max(min(v / hi, 1), 0) * 100
            body.append(f"<div class='st'><span class='sn'>4 Projection vs line</span> <b>{mu:.1f}</b> vs <b>{ln:g}</b> <span class='mut'>(&plusmn;{sd_:.1f})</span>"
                        f"<div class='pj'><span class='band' style='left:{pos(mu - sd_):.1f}%;width:{pos(mu + sd_) - pos(mu - sd_):.1f}%'></span>"
                        f"<span class='mk line' style='left:{pos(ln):.1f}%'></span><span class='mk proj' style='left:{pos(mu):.1f}%'></span></div>"
                        f"<span class='mut sm'>dark tick = line, orange tick = projection, band = &plusmn;1 sd</span></div>")
            body.append("<div class='bars'>" + bar("model", r["model_prob"], "m") + bar("market", r["market_prob"], "k") + bar("VPE win", r["blend_prob"], "v") + bar("needed", implied_prob(r["odds"]), "n") + "</div>")
            if p:
                body.append(f"<div class='mono'>5-7 raw {pc(p.get('raw'), 1)} &rarr; calibrated {pc(p.get('cal'), 1)} | market {pc(p.get('fair'), 1)} ({e(p.get('how', ''))}) | mix {float(p.get('w', 0)) * 100:.0f}% model = {pc(p.get('mix'), 1)} | learned fix {float(p.get('learn', 0) or 0) * 100:+.1f} &rarr; {pc(p.get('fin'), 1)}</div>")
            tier = r.get("tier") if isinstance(r.get("tier"), str) else None
            body.append(f"<div class='st'><span class='sn'>8-9 Price</span> {am(r['odds'])} &rarr; needs {pc(implied_prob(r['odds']), 1)} &rarr; edge <span class='{'pos' if edge > 0 else 'neg'}'>{edge * 100:+.1f}</span> &rarr; "
                        f"{e(tier.replace('_', ' ').upper()) if tier else 'no straight tier'}" + (f" &middot; {float(r['units']):g}u (VPE's size) &middot; take-if {am(r.get('min_odds'))}" if tier else "") + "</div>")
            body.append(f"<div class='st'><span class='sn'>10 Risk</span> <span class='chip {rr_cls(r.get('role_risk'))}'>{e(r.get('role_risk'))}</span> {e(r.get('role_why', ''))} &middot; model vs market gap {float(r.get('market_gap') or 0) * 100:.0f} pts"
                        + (f" &middot; <span class='neg'>HELD: {e(hr)}</span>" if hr else "") + "</div>")
        q = pool_row(pl, mk, sd, ln)
        if q is not None and "price_dk" in q:
            body.append(f"<div class='st'><span class='sn'>11 As a parlay leg</span> DraftKings {am(q['price_dk'])} / FanDuel {am(q['price_fd'])} &rarr; best {e(q.get('bet_at', ''))} &middot; leg chance {pc(q['leg_prob'], 1)} &middot; leg edge {float(q['leg_edge']) * 100:+.1f} &middot; take-if {am(q.get('take_min'))}</div>")
        try:
            raw = game_card(r, x, [role], cfg) if mk in GAME_MK else explain_card(r, x, [role], cfg)
        except Exception as ex_:
            raw = f"(engine card unavailable: {ex_})"
        body.append(f"<pre class='raw'>{e(raw)}</pre>")
        cards.append(f"<div class='card'>{head}{''.join(body)}</div>")
    if cards:
        S.append("<h2>Full logic, leg by leg</h2>" + "".join(cards))
    status_txt = open(a.status_file).read() if a.status_file and os.path.exists(a.status_file) else ""
    _vm = re.search(r"RUN STATUS[^\n]*:\s*(GO|PARTIAL|HOLD)\b", status_txt) or re.search(r"->\s*(GO|PARTIAL|HOLD)\b", status_txt)
    verdict = _vm.group(1) if _vm else "n/a"
    if status_txt:
        S.append(f"<h2>Run status</h2><pre class='raw big'>{e(status_txt.strip())}</pre>")
    chips = [f"{p_.get('name')}: {'LEAN' if p_.get('lean') else 'USE IT'}" for p_ in rec] + ([f"{len(bets)} straight" + ("s" if len(bets) != 1 else "")] if len(bets) else ["no straight picks"]) \
        + ([f"{int(gaps['verdict'].astype(str).str.startswith('TAKE').sum())} book-gap takes"] if len(gaps) else []) + [f"skip: {x.split(':')[0]}" for x in skip_items[:2]]
    n_games = allp["game_id"].astype(str).nunique()
    head = (f"<div class='top'><div class='h1'>VPE DAY CARD <span class='acc'>{e(a.title or str(allp['date'].iloc[0]))}</span></div>"
            f"<div class='meta'>{VERSION} &middot; lines: {e(a.source or ('your ' + LOCK['books'] + ' prices' if LOCK['on'] else 'best of market'))}, {e(a.asof or 'time not given')} &middot; {n_games} games &middot; run status: <b>{verdict}</b></div>"
            f"<div class='chips'>{''.join(f'<span class=chip2>{e(c)}</span>' for c in chips)}</div>"
            f"<div class='flow'>{len(allp):,} lines scored &rarr; news sweep &rarr; rules (held, KILL list, price floor, Under cap) &rarr; the plays</div></div>")
    css = """@page{size:Letter;margin:0.33in}*{box-sizing:border-box}body{font-family:Inter,'Helvetica Neue',Arial,sans-serif;font-size:9.6pt;color:#16181d;margin:0;line-height:1.32}
.h1{font-family:'Barlow Condensed','Roboto Condensed','Arial Narrow','DejaVu Sans Condensed',Impact,sans-serif;font-weight:800;font-size:27pt;letter-spacing:.5px;line-height:1}.acc{color:#e8452c}
.meta{color:#555;font-size:8.6pt;margin-top:3px}.chips{margin:6px 0 2px}.chip2{display:inline-block;background:#16181d;color:#fff;border-radius:3px;padding:2px 7px;margin:0 4px 4px 0;font-size:8.4pt;font-weight:600}
.flow{font-size:8.4pt;color:#555;border-top:2px solid #e8452c;padding-top:4px;margin-top:2px}
h2{font-family:'Barlow Condensed','Roboto Condensed','Arial Narrow','DejaVu Sans Condensed',sans-serif;font-size:15pt;font-weight:700;margin:12px 0 5px;text-transform:uppercase;letter-spacing:.4px;break-after:avoid;page-break-after:avoid}
.play,.card,.skip,tr,.st,.bars{break-inside:avoid;page-break-inside:avoid}
.play{border:1.5px solid #16181d;border-radius:6px;padding:8px 10px;margin-bottom:9px}.play.lean{border-style:dashed;border-color:#888}
.ph{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}.tag{background:#e8452c;color:#fff;font-weight:700;font-size:8pt;padding:1px 6px;border-radius:3px}.lean .tag{background:#888}
.pt{font-weight:700;font-size:11.5pt}.pp{margin-left:auto;font-family:'DejaVu Sans Mono',ui-monospace,Menlo,monospace;font-size:12pt}.pp s{color:#999}
.sub{font-size:8.6pt;color:#444;margin:3px 0 6px}.boxes{display:flex;gap:7px;margin-bottom:6px}.bx{flex:1;background:#f3f4f6;border-radius:5px;padding:5px 8px}
.bl{font-size:7.6pt;color:#555;text-transform:uppercase;letter-spacing:.3px}.bv{font-family:'DejaVu Sans Mono',ui-monospace,monospace;font-size:14pt;font-weight:700}
table.legs{width:100%;border-collapse:collapse;table-layout:auto}table.legs th{text-align:left;font-size:7.8pt;text-transform:uppercase;color:#555;border-bottom:1.5px solid #16181d;padding:3px 5px}
table.legs td{padding:4px 5px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}.num{font-family:'DejaVu Sans Mono',ui-monospace,monospace;white-space:nowrap}
.sm{font-size:8.2pt}.mut{color:#666;font-weight:400}.pos{color:#15803d;font-weight:700}.neg{color:#b91c1c;font-weight:700}
.note{background:#fff7ed;border-left:3px solid #e8452c;padding:5px 8px;margin-top:6px;font-size:8.6pt}.skip{background:#fef2f2;border:1.5px solid #b91c1c;border-radius:6px;padding:7px 10px;margin:9px 0;color:#7f1d1d;font-size:9pt}
.chip{display:inline-block;border-radius:3px;padding:0 5px;font-size:8pt;font-weight:600;margin:1px 2px 1px 0;white-space:nowrap}.chip.g{background:#dcfce7;color:#166534}.chip.r{background:#fee2e2;color:#991b1b}.chip.y{background:#fef9c3;color:#854d0e}.chip.n{background:#e5e7eb;color:#374151}
.btn{display:inline-block;background:#e8452c;color:#fff!important;text-decoration:none;font-weight:700;font-size:8.2pt;padding:2px 8px;border-radius:4px;margin-top:2px;white-space:nowrap}
.nolink{display:inline-block;border:1px dashed #999;color:#666;font-size:7.8pt;padding:1px 6px;border-radius:4px;margin-top:2px}
.card{border:1px solid #c9ccd1;border-radius:6px;padding:7px 9px;margin-bottom:8px}.ch{display:flex;align-items:center;gap:6px;flex-wrap:wrap;border-bottom:1px solid #e5e7eb;padding-bottom:4px;margin-bottom:4px}
.cn{font-weight:700;font-size:10.8pt}.cb{margin-left:auto}.st{font-size:8.7pt;margin:2px 0}.sn{font-weight:700;color:#e8452c;margin-right:3px}.steps .st{margin:1px 0}
.pj{position:relative;height:9px;background:#e5e7eb;border-radius:4px;margin:3px 0 1px}.band{position:absolute;top:0;height:9px;background:#fdba74;border-radius:4px}
.mk{position:absolute;top:-3px;width:2.5px;height:15px}.mk.line{background:#16181d}.mk.proj{background:#e8452c}
.bars{margin:4px 0}.br{display:flex;align-items:center;gap:6px;font-size:8.2pt;margin:1px 0}.bn{width:52px;color:#444}.bt{flex:1;height:7px;background:#eef0f3;border-radius:4px;overflow:hidden}
.bf{display:block;height:7px}.bf.m{background:#60a5fa}.bf.k{background:#9ca3af}.bf.v{background:#e8452c}.bf.n{background:#16181d}.bv2{width:44px;text-align:right;font-family:'DejaVu Sans Mono',ui-monospace,monospace}
.mono{font-family:'DejaVu Sans Mono',ui-monospace,monospace;font-size:7.4pt;color:#444;margin:2px 0}
pre.raw{font-family:'DejaVu Sans Mono',ui-monospace,monospace;font-size:5.6pt;line-height:1.25;color:#6b7280;white-space:pre-wrap;overflow-wrap:anywhere;border-top:1px dashed #d1d5db;margin:5px 0 0;padding-top:3px}pre.raw.big{font-size:7.4pt;color:#16181d;break-inside:avoid}
.foot{font-size:7.6pt;color:#666;margin-top:8px;border-top:1px solid #ddd;padding-top:4px}"""
    foot = ("<div class='foot'>Win chances are VPE's estimates (its own projection mixed with the sportsbooks' prices). EV figures are engine estimates. A link opens that sportsbook's bet slip with the bet "
            "loaded: check the price there against the take-if before placing. Information only, not betting advice; no outcome is guaranteed. 21+ and where legal.</div>")
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>VPE day card</title><style>{css}</style></head><body>{head}{''.join(x for x in S if x)}{foot}</body></html>"
    hp = os.path.splitext(a.out)[0] + ".html"
    open(hp, "w").write(doc)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page()
            pg.set_content(doc, wait_until="load")
            pg.pdf(path=a.out, format="Letter", print_background=True, margin={"top": "0.33in", "bottom": "0.33in", "left": "0.33in", "right": "0.33in"})
            b.close()
    except Exception as ex_:
        print(f"SHAREPDF: could not print the day card with Chromium ({type(ex_).__name__}: {str(ex_)[:120]}) -> short PDF instead; the full card is in {hp}")
        return _sharepdf_short(a)
    pages = n_ann = lonely = "?"
    try:
        from pypdf import PdfReader
        rd = PdfReader(a.out)
        pages, n_ann, lonely = len(rd.pages), 0, 0
        heads = {"THE PLAY", "STRAIGHT BETS", "BOOK GAPS", "BOOSTS", "SHARED RISK", "FULL LOGIC, LEG BY LEG", "RUN STATUS", "OTHER BUILDS (REFERENCE ONLY)"}
        for p_ in rd.pages:
            n_ann += sum(1 for an in (p_.get("/Annots") or []) if "/URI" in str(an.get_object().get("/A", {})))
            ls = [t.strip() for t in (p_.extract_text() or "").split("\n") if t.strip()]
            lonely += 1 if ls and ls[-1].upper() in heads else 0
    except Exception:
        pass
    held_shown = doc.count("HELD: not a VPE play")
    ok = (n_ann == stats["links"]) if isinstance(n_ann, int) else True
    print(f"SHAREPDF: day card -> {a.out} | {pages} pages | {len(cards)} logic cards | links {n_ann} working of {stats['links']} expected ({stats['nolinks']} bets without a link) "
          f"| headings alone at a page end: {lonely} | HELD flags shown: {held_shown} | {'OK' if ok and lonely in (0, '?') else 'CHECK: fix before sending'}")


def cmd_linediff(a):
    """v2.24 / v15.25: two recordings of the same night (or a recording vs the closing page): MAIN two-sided lines only
    (ladders are too noisy to compare: 11/16, 736 of 1,180 matched prices 'changed'). Shows real moves, flags jumps that
    look like a misread, and, with --log, how each bet already placed moved since (price taken vs the later fair price)."""
    def main(path):
        p = pd.read_csv(path)
        p["_m"] = p["market"].map(lambda m: canon_market(m) if "canon_market" in globals() else str(m))
        p["_o"], p["_u"] = pd.to_numeric(p["over_odds"], errors="coerce"), pd.to_numeric(p["under_odds"], errors="coerce")
        p["_l"] = pd.to_numeric(p["line"], errors="coerce")
        p = p[p["_o"].notna() & p["_u"].notna() & p["_l"].notna()].copy()
        p["_k"] = p["player"].map(norm_name)
        p["_b"] = p.get("book", pd.Series("", index=p.index)).fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
        p["_f"] = [no_vig(o, u)[0] for o, u in zip(p["_o"], p["_u"])]
        p["_h"] = [implied_prob(o) + implied_prob(u) - 1 for o, u in zip(p["_o"], p["_u"])]
        return p.drop_duplicates(["_k", "_m", "_b"], keep="last")
    A, B = main(a.a), main(a.b)
    m = A.merge(B, on=["_k", "_m", "_b"], suffixes=("_a", "_b"))
    if m.empty:
        print("LINEDIFF: no main two-sided lines in both files (same player, market, book)")
        return
    m["dline"] = m["_l_b"] - m["_l_a"]
    m["dfair"] = m["_f_b"] - m["_f_a"]  # change in the market's Over chance (at each file's own line)
    same = m["dline"].abs() < 1e-9
    def cents(x, y):
        return abs(american_to_decimal(y) - american_to_decimal(x)) * 100
    m["dc_o"] = [cents(x, y) for x, y in zip(m["_o_a"], m["_o_b"])]
    flags = []
    for _, r in m.iterrows():
        why = []
        if not (-0.025 <= r["_h_b"] <= 0.15) and (-0.025 <= r["_h_a"] <= 0.15):  # one book's O/U can't hold <0 or 15%+
            why.append(f"impossible hold {r['_h_b']:+.1%} in B")
        if abs(r["dline"]) >= (3 if r["_m"] in ("pts", "pra", "pass_yds", "rush_yds", "rec_yds", "rush_rec_yds") else 1.5):
            why.append(f"line jumped {r['_l_a']:g} -> {r['_l_b']:g}")
        if abs(r["dline"]) < 1e-9 and abs(r["dfair"]) > 0.12:
            why.append(f"same line, market chance moved {r['dfair']:+.0%}")
        if why:
            flags.append((r["player_a"], r["_m"], "; ".join(why)))
    mv = m[~same | (m["dc_o"] >= 1)]
    print(f"LINEDIFF: {len(m)} main lines in both | {int(same.sum())} same line ({int((same & (m['dc_o'] < 1)).sum())} unchanged,"
          f" {int((same & (m['dc_o'] >= 1)).sum())} price moved) | {int((~same).sum())} line moved | {len(flags)} flagged as likely misread")
    for p_, mk, why in flags[:30]:
        print(f"  FLAG {p_} {mk}: {why} -> check that frame")
    top = m.assign(_ab=m["dfair"].abs()).sort_values("_ab", ascending=False).head(10)
    print("  biggest real moves (Over chance A -> B):")
    for _, r in top.iterrows():
        if (r["player_a"], r["_m"]) in {(f[0], f[1]) for f in flags}:
            continue
        print(f"    {r['player_a']:24s} {r['_m']:8s} {r['_l_a']:g} {int(r['_o_a']):+d}/{int(r['_u_a']):+d} -> {r['_l_b']:g} "
              f"{int(r['_o_b']):+d}/{int(r['_u_b']):+d}  Over {r['_f_a']:.1%} -> {r['_f_b']:.1%}")
    if a.log and os.path.exists(a.log):
        d = read_log(a.log)
        if a.date and "date" in d:
            d = d[d["date"].astype(str).str.lstrip("'") == str(pd.Timestamp(a.date).date())]
        d = d[d.get("bet_type", pd.Series("straight", index=d.index)).fillna("straight").isin(["straight", "leg"])]
        rows = []
        Bk = B.set_index(["_k", "_m"])
        for _, r in d.iterrows():
            k = (norm_name(r["player"]), canon_market(r["market"]) if "canon_market" in globals() else str(r["market"]))
            if k not in Bk.index or pd.isna(r.get("odds")):
                continue
            b = Bk.loc[[k]].iloc[0]
            if abs(float(b["_l"]) - float(r["line"])) > 1e-9:
                _good = (b["_l"] > r["line"]) == (str(r["side"]) == "Under")  # line up helps an Under, hurts an Over
                rows.append(f"    {r['player']} {r['side']} {r['line']:g} {r['market']}: line now {b['_l']:g} -> moved "
                            + ("YOUR way (you beat the later line)" if _good else "against you"))
                continue
            fair = b["_f"] if str(r["side"]) == "Over" else 1 - b["_f"]
            clv = fair - implied_prob(r["odds"])
            rows.append(f"    {r['player']} {r['side']} {r['line']:g} {r['market']} @ {int(r['odds']):+d}: later fair {fair:.1%} vs "
                        f"break-even {implied_prob(r['odds']):.1%} -> CLV {clv:+.1%} ({'beat' if clv > 0 else 'behind'} the later price)")
        print(f"  bets already placed vs B ({len(rows)} matched at B's main line):")
        for x in rows:
            print(x)


def cmd_exposure(a):
    """v2.24 / v15.25: the whole night's tickets together (11/16: 7 of 9 legs Unders, two parlays sharing BKN-PHI and
    CHI-CLE, Demin in 2 bets, 0 of 4 hit). Reads the logged rows for the night/week: shared games and players, Over/Under
    mix, and a 20,000-run simulation of the night with each leg drawn once (a leg in two bets wins or loses for both).
    Legs in the same game are drawn independently, so 'all lose' is a FLOOR when same-game legs lean the same way."""
    d = read_log(a.log)
    if "date" in d.columns and "week" not in d.columns:
        D = str(pd.Timestamp(a.date).date()) if a.date else str(pd.Timestamp.now(tz="America/New_York").date())
        d = d[d["date"].astype(str).str.lstrip("'") == D]; lab = D
    else:
        wk = int(a.week) if a.week else int(pd.to_numeric(d["week"], errors="coerce").max())
        d = d[pd.to_numeric(d["week"], errors="coerce") == wk]; lab = f"week {wk}"
    if d.empty:
        print(f"EXPOSURE {lab}: no logged bets (log them first, then run exposure)")
        return
    bt = d.get("bet_type", pd.Series("straight", index=d.index)).fillna("straight")
    d = d[~d["result"].astype(str).str.upper().isin(["VOID", "PUSH"])] if "result" in d else d
    tickets, legs = [], {}
    def _key(r):
        return (norm_name(r["player"]), str(r["market"]), str(r["side"]), round(float(r["line"]), 1) if pd.notna(r["line"]) else None)
    def _p(r):
        p = r.get("model_prob")
        if pd.notna(p):
            return float(p), False
        o = r.get("odds")
        return (1 / american_to_decimal(o), True) if pd.notna(o) else (0.5, True)
    for _, r in d[bt == "straight"].iterrows():
        k = _key(r); p, est = _p(r); legs.setdefault(k, {"p": p, "est": est, "team": r.get("team"), "opp": r.get("opp")})
        tickets.append({"name": f"{r['player']} {r['side']} {r['line']:g} {r['market']}", "legs": [k],
                        "stake": float(r.get("stake_usd") or r.get("units") or 0), "odds": r.get("odds_eff") if pd.notna(r.get("odds_eff")) else r.get("odds")})
    for pid, g in d[bt == "leg"].groupby("parlay_id"):
        ks = []
        for _, r in g.iterrows():
            k = _key(r); p, est = _p(r); legs.setdefault(k, {"p": p, "est": est, "team": r.get("team"), "opp": r.get("opp")}); ks.append(k)
        pr = d[(bt == "parlay") & (d["parlay_id"] == pid)]
        pr = pr.iloc[0] if len(pr) else None
        tickets.append({"name": f"parlay {pid} ({len(ks)} legs)", "legs": ks,
                        "stake": float(pr.get("stake_usd") or pr.get("units") or 0) if pr is not None else 0.0,
                        "odds": (pr.get("odds_eff") if pd.notna(pr.get("odds_eff")) else pr.get("odds")) if pr is not None else None})
    if not tickets:
        print(f"EXPOSURE {lab}: no tickets found")
        return
    lk = list(legs)
    use = {}
    for t in tickets:
        for k in t["legs"]:
            use.setdefault(k, []).append(t["name"])
    game = lambda k: " vs ".join(sorted([str(legs[k]["team"]), str(legs[k]["opp"])])) if pd.notna(legs[k]["team"]) else "?"
    gm = {}
    for k in lk:
        gm.setdefault(game(k), set()).update(n for n in use[k])
    ov = sum(1 for k in lk if str(k[2]).lower() == "over"); un = sum(1 for k in lk if str(k[2]).lower() == "under")
    stake = sum(t["stake"] for t in tickets)
    print(f"EXPOSURE {lab}: {len(tickets)} tickets, {len(lk)} different legs, ${stake:,.2f} staked | Overs {ov} / Unders {un}"
          + (f"  <- {max(ov, un) / max(ov + un, 1):.0%} one side" if max(ov, un) >= 0.75 * (ov + un) and ov + un >= 4 else ""))
    for k in lk:
        if len(use[k]) > 1:
            print(f"  SHARED LEG  {k[0]} {k[2]} {k[3]:g} {k[1]}: in {len(use[k])} tickets ({'; '.join(use[k])}) -> one miss sinks all of them")
    pl = {}
    for k in lk:
        pl.setdefault(k[0], set()).update(use[k])
    for p_, ts in pl.items():
        if len(ts) > 1 and not any(len(use[k]) > 1 and k[0] == p_ for k in lk):
            print(f"  SHARED PLAYER {p_}: in {len(ts)} tickets on different props")
    nleg = {}
    for k in lk:
        nleg[game(k)] = nleg.get(game(k), 0) + 1
    for g_, ts in sorted(gm.items()):
        if len(ts) > 1 and nleg[g_] > 1:
            print(f"  SHARED GAME {g_}: {len(ts)} tickets ({'; '.join(sorted(ts))}) -> legs in one game move together")
    rng = np.random.default_rng(7)
    N = 20000
    P = np.array([legs[k]["p"] for k in lk])
    W = rng.random((N, len(lk))) < P
    ix = {k: i for i, k in enumerate(lk)}
    prof = np.zeros(N); anyw = np.zeros(N, bool)
    for t in tickets:
        win = W[:, [ix[k] for k in t["legs"]]].all(axis=1)
        dec = american_to_decimal(float(t["odds"])) if t["odds"] is not None and pd.notna(t["odds"]) else 2.0
        prof += np.where(win, t["stake"] * (dec - 1), -t["stake"]); anyw |= win
        t["p"] = float(win.mean())
    print("  tickets (VPE win chance each):")
    for t in tickets:
        print(f"    {t['name'][:60]:60s} ${t['stake']:8.2f}  {t['p']:5.1%}")
    est = [k for k in lk if legs[k]["est"]]
    print(f"  ALL LOSE: {1 - anyw.mean():.1%} (at least one wins {anyw.mean():.1%}) | night in profit {np.mean(prof > 0):.1%}"
          f" | expected ${prof.mean():+,.2f} | worst ${prof.min():+,.2f}, best ${prof.max():+,.2f}")
    print("  note: same-game legs drawn independently, so ALL LOSE is a floor when same-game legs lean the same way"
          + (f"; {len(est)} leg(s) had no VPE chance -> priced at the book's implied chance" if est else ""))


def cmd_sweep(a):
    import json
    allp = pd.read_csv(a.all) if os.path.exists(a.all) else pd.DataFrame()
    if allp.empty:
        print("SWEEP: no scored lines (run check first)")
        return
    for c in ("edge", "odds", "market_gap"):
        allp[c] = pd.to_numeric(allp.get(c), errors="coerce")
    why = {}

    def add(p, reason):
        why.setdefault(p, [])
        if reason not in why[p]:
            why[p].append(reason)
    if a.bets and os.path.exists(a.bets) and os.path.getsize(a.bets) > 0:
        bb = pd.read_csv(a.bets)
        for p in (bb["player"] if "player" in bb else []):
            add(p, "straight bet")
    if a.parlays and os.path.exists(a.parlays):
        for par in json.load(open(a.parlays)):
            for l in par.get("legs", []):
                add(l[0], "parlay leg" + (" (THE parlay)" if par.get("recommended") else ""))
    top = allp[(allp["edge"] >= 0.015) & allp["odds"].between(-200, 200) & (allp["market_gap"].fillna(0) <= CONFIG["model"]["max_market_gap"])]
    for p in top.sort_values("edge", ascending=False)["player"].drop_duplicates().head(a.top):
        add(p, "top prop")
    if a.heavy:  # final check: every player with a qualifying line
        for p in top["player"].drop_duplicates():
            add(p, "qualifying line")
    team_of = allp.drop_duplicates("player").set_index("player")
    games = {}
    for p in why:
        if p not in team_of.index:
            continue
        r = team_of.loc[p]
        g = " vs ".join(sorted([str(r["team"]), str(r["opp"])]))
        games.setdefault(g, []).append(p)
    rank = lambda p: (0 if any(w.startswith(("straight", "parlay")) for w in why[p]) else 1, p)
    n_q = 0
    # v2.24 / v15.25: dated queries (11/16: "is Zion playing tonight" returned a 2025 article; an undated "set to miss 10th
    # straight game" was trusted). Month day year in every query + a freshness rule.
    _td = pd.Timestamp.now(tz="America/New_York")
    _ds = f"{_td.strftime('%B')} {_td.day} {_td.year}"
    _ok = f"{(_td - pd.Timedelta(days=1)).strftime('%b')} {(_td - pd.Timedelta(days=1)).day}-{_td.strftime('%b')} {_td.day}, {_td.year}"
    print(f"NEWS SWEEP ({'HEAVY - final check' if a.heavy else 'every run'}): {len(why)} players in {len(games)} games")
    for g, ps in sorted(games.items()):
        print(f"\n  {g}")
        print(f'    search: "{g.replace(" vs ", " ")} injury news minutes restriction {_ds}"')
        n_q += 1
        for p in sorted(ps, key=rank):
            print(f'    - {p:<26} ({", ".join(why[p])})  search: "{p} status tonight minutes rotation role {_ds}"')
            n_q += 1
    print(f"\nFRESHNESS RULE: use only results dated {_ok} (today or yesterday). A result with an older date, or NO date"
          " at all, is ignored even if it sounds current (\"set to miss his 10th straight game\" with no date = ignore)."
          " Prefer the team's official injury report, beat writers and league sites that show a timestamp.")
    print(f"\n{n_q} searches. Any player reported out / doubtful / questionable / on a minutes limit / coming off the bench"
          " in a new role -> add to inj.txt (Out, Doubtful, Questionable, Limited) and rerun check + parlay + menu."
          "\n  ROLE-LOSS news = Limited (no Overs, minutes x0.8) even with no injury: out of the rotation, a DNP-CD in his last"
          " game, G League assignment, demoted / losing minutes to a teammate, team declined his option or he is being"
          " shopped/waived (11/2: Collier's option declined that day -> 12 garbage-time minutes)."
          " Quote one line in the header: NEWS: N searches, M changes (names) or NEWS: N searches, no changes.")


def shadow_rows(allp, key):
    """One row per line scored this run (Over view): what the model and the market said before the game."""
    out = {}
    for r in allp.to_dict("records"):
        if pd.isna(r.get("model_prob")) or pd.isna(r.get("market_prob")):
            continue
        k = (r["player"], r["market"], float(r["line"]))
        x = out.setdefault(k, {"key": key, "player": r["player"], "market": r["market"], "line": float(r["line"]),
                               "over_odds": "", "under_odds": "", "p_model": None, "p_market": None, "actual": "", "over": "",
                               "p_vpe": None})
        o = r["side"] == "Over"
        x["over_odds" if o else "under_odds"] = int(r["odds"])
        if x["p_model"] is None or o:
            x["p_model"] = round(float(r["model_prob"] if o else 1 - r["model_prob"]), 4)
            x["p_market"] = round(float(r["market_prob"] if o else 1 - r["market_prob"]), 4)
            # v2.25: column K = the Over win chance VPE gave THAT night (learned mix + fixes), so the scoreboard's calibration
            # test compares what was said then with what happened (not a re-score with today's weights)
            bp_ = r.get("blend_prob")
            x["p_vpe"] = "" if bp_ is None or pd.isna(bp_) else round(float(bp_ if o else 1 - bp_), 4)
    return pd.DataFrame(list(out.values()), columns=SHADOW_COLS + ["p_vpe"])


def write_shadow(evald, key, path, saved=None):
    import json
    if evald is None or evald.empty or not path:
        return
    s = shadow_rows(evald, key)
    # v2.23 / v15.24: same-slate rerun -> skip lines already in the Shadow tab (key + player + market + line), so a rerun
    # never saves the same night twice. `saved` = get_values JSON of the tab's last rows (columns A:D).
    if saved and os.path.exists(saved):
        have = set()
        for r in _values(saved):
            if len(r) >= 4:
                try:
                    have.add((str(r[0]).lstrip("'"), norm_name(r[1]), str(r[2]), round(float(r[3]), 1)))
                except (TypeError, ValueError):
                    pass
        dup = [(str(key), norm_name(p), str(m), round(float(l), 1)) in have for p, m, l in zip(s["player"], s["market"], s["line"])]
        n0 = len(s)
        s = s[[not x for x in dup]].reset_index(drop=True)
        print(f"SHADOW DEDUPE: {n0 - len(s)} of {n0} lines already saved for {key} (rerun) -> {len(s)} new")
        for f_ in [path] + [f"{path[:-5] if path.endswith('.json') else path}_{i}.json" for i in range(1, 60)]:
            if os.path.exists(f_):
                os.remove(f_)
        if s.empty:
            json.dump([], open(path, "w"))
            print("SHADOW: 0 new lines (this slate is already in the Shadow tab) -> nothing to append; report 'Shadow: already saved'")
            return
    rows = [[_cell(v) for v in r] for r in s.itertuples(index=False, name=None)]
    json.dump(rows, open(path, "w"))
    # small chunks: one big append is easy to skip or truncate in a chat; 100 rows per call is reliable
    stem, n = path[:-5] if path.endswith(".json") else path, 100
    parts = [rows[i:i + n] for i in range(0, len(rows), n)]
    names = []
    for i, part in enumerate(parts, 1):
        name = f"{stem}_{i}.json"
        json.dump(part, open(name, "w"))
        names.append(name)
    print(f"SHADOW: {len(s)} lines -> {path} (also split into {len(parts)} files of <= {n} rows: {', '.join(names)})\n"
          f"  >>> REQUIRED: read 'Calib'!V1 (= before), append_values each file in order to 'Shadow NBA'!A1, then re-read V1:"
          f" it must equal before + {len(s)}. Report 'Shadow: {len(s)} lines saved' only after that check passes.")


def cmd_shadow(a):
    """Grade the Shadow-tab rows for one night/week and fold them into the Calib tab."""
    import json
    vals = _values(a.json)
    key = str(a.key).lstrip("'")
    value_of = shadow_value_fn(key)
    if value_of is None:
        print(f"No box scores for {key} yet; retry later.")
        return
    rows = [list(r)[:len(SHADOW_COLS)] + [""] * (len(SHADOW_COLS) - len(r)) for r in vals]
    if rows and str(rows[0][0]) == "key":
        rows, first = rows[1:], a.first_row + 1
    else:
        first = a.first_row
    if rows and str(rows[0][0]).lstrip("'") == key and first > 2:
        print(f"  ! WIDEN: row {first} is already {key}; read from an earlier row so the whole night is included")
    seen, new = set(), []
    for r in rows:
        if str(r[0]).lstrip("'") != key or str(r[9]).strip() != "":
            continue
        v = value_of(r[1], r[2])
        if v is None:
            r[8], r[9] = "", "dnp"
            continue
        line = float(r[3])
        r[8] = round(float(v), 1)
        r[9] = "push" if v == line else (1 if v > line else 0)
        u = (r[1], r[2], line)
        if r[9] in (0, 1) and u not in seen and str(r[6]).strip() and str(r[7]).strip():
            seen.add(u)
            new.append((r[2], float(r[6]), float(r[7]), int(r[9])))
    json.dump([[r[8], r[9]] for r in rows], open(a.out, "w"))
    print(f"graded {len(new)} new lines for {key} -> {a.out} (write to columns I:J from row {first})")
    cal = {(str(r.get("sport", "")).upper(), str(r["market"])): r for r in _calib_rows(a.calib)}
    for mk, g in pd.DataFrame(new, columns=["market", "pm", "pk", "y"]).groupby("market"):
        r = cal.setdefault((SPORT, mk), {"sport": SPORT, "market": mk})
        r["n"] = _f(r, "n") + len(g)
        r["sum_over"] = _f(r, "sum_over") + g["y"].sum()
        r["sum_pm"] = round(_f(r, "sum_pm") + g["pm"].sum(), 4)
        r["sum_pk"] = round(_f(r, "sum_pk") + g["pk"].sum(), 4)
        for x in CALIB_W:
            p = np.clip(x * g["pm"] + (1 - x) * g["pk"], 0.01, 0.99)
            c = f"ll_{int(round(x * 100)):03d}"
            r[c] = round(_f(r, c) + float(-(g["y"] * np.log(p) + (1 - g["y"]) * np.log(1 - p)).sum()), 4)
    table = [CALIB_COLS] + [[r.get(c, "") for c in CALIB_COLS] for r in sorted(cal.values(), key=lambda r: (r["sport"], r["market"]))]
    json.dump(table, open(a.calib_out, "w"))
    if new:
        d = pd.DataFrame(new, columns=["market", "pm", "pk", "y"])
        w = PARLAY["model_weight"]
        print(f"  tonight: Over hit {d['y'].mean():.1%} | market said {d['pk'].mean():.1%} | model said {d['pm'].mean():.1%}"
              f" | Brier market {((d['pk'] - d['y']) ** 2).mean():.4f} vs model {((d['pm'] - d['y']) ** 2).mean():.4f}")
    w, bias, n = load_calib_table(table, SPORT)
    print(f"Calib table -> {a.calib_out} (write at 'Calib'!A1). Now: {n} graded lines -> model weight {w:.0%}")


def load_calib_table(table, sport):
    import json, tempfile
    p = os.path.join(tempfile.gettempdir(), "_calib_tmp.json")
    json.dump(table, open(p, "w"))
    return load_calib(p, sport)


# ---------- profit boosts: the one edge that holds even if the model has none ----------
# At market prices a 2-3 leg parlay is +EV with a 20%+ boost and a 4-leg with 25%+; 5 legs needs ~30%. So a boost goes
# on its own short parlay (most likely legs, checked +EV at market-only odds), not on the 10x parlay.
def boost_parlay(pool, book, boost, stake, min_legs=2, max_legs=4, min_dec=1.0, max_win=None, need_market_ev=True):
    col = "dec_" + book
    # low-vig legs first: value at market odds (no-vig chance x this book's price), not the shortest prices
    ranked = pool.assign(_mv=pool["market_prob"] * pool[col] * (1 + pool["market"].map(BOOST_PREF).fillna(0.0) / 6)
                         ).sort_values(["_mv", "leg_prob"], ascending=False)
    # v2.26: boosts are built from TWO-SIDED main lines (both prices recorded = real, lower hold: all two-sided sides in the
    # Shadow tab returned -6.6% vs -9.7% on one-sided ladder rungs). Ladder rungs only when too few main lines exist.
    if "two_sided" in ranked:
        _t = ranked[ranked["two_sided"]]
        if _t["game_id"].astype(str).nunique() >= max(min_legs, 2):
            ranked = _t
        else:
            print(f"  note ({book} {boost:g}% boost): only {len(_t)} two-sided main lines qualify -> ladder rungs allowed too"
                  " (record the main Over/Under lists to get lower-hold legs)")
    ranked = ranked.drop_duplicates("player")
    # both sides in the candidate list (low-vig legs skew Under, and the Unders cap would block every combo)
    ranked = pd.concat([ranked[ranked["side"] == sd].head(PARLAY["pool"] // 2 + 1) for sd in ("Over", "Under")])
    cand, best, allc = ranked.to_dict("records"), None, []
    for n in range(max(min_legs, 2), max_legs + 1):
        for combo in itertools.combinations(cand, n):
            if len({str(c["game_id"]) for c in combo}) < n:
                continue
            uh = sum(c["side"] == "Under" for c in combo) > _under_max(n)
            dec = float(np.prod([c[col] for c in combo]))
            if dec < min_dec:
                continue
            extra = (dec - 1) * stake * boost / 100
            if max_win:
                extra = min(extra, max_win)
            dec_eff = dec + extra / stake
            p = float(np.prod([c["leg_prob"] for c in combo]))
            p_mkt = float(np.prod([c["market_prob"] for c in combo]))
            if need_market_ev and p_mkt * dec_eff < 1:  # must be +EV even if the model adds nothing
                continue
            # least risk for the payout: maximize long-run bankroll growth at a 5% stake (penalizes longshots)
            g = p * math.log(1 + 0.05 * (dec_eff - 1)) + (1 - p) * math.log(0.95)
            if g > 0:  # v2.22: on close calls prefer threes / PRA legs (+3% of the growth score per leg)
                g *= 1 + sum(BOOST_PREF.get(c["market"], 0.0) for c in combo)
            ml = min(float(c.get("leg_edge", 0) or 0) for c in combo)
            if BOOST_LEG["min_edge"] is not None and ml < BOOST_LEG["min_edge"]:
                continue
            allc.append((p * dec_eff - 1, p, p_mkt, dec, dec_eff, combo, g, ml, uh))
    if not allc:
        return None
    # v2.24 / v15.25: close calls go to the STRONGER legs (11/16: the FD 25% builds used legs at +0.1% to +0.4% edge, so
    # the boost was the whole edge). Among combos within 10% of the best growth score, the score gets
    # +BOOST_LEG["w"] x the weakest leg's edge (w 5: a +1% weakest leg = +5%); outside close calls nothing changes.
    def _pick(xs):
        if not xs:
            return None
        gmax = max(x[6] for x in xs)
        close = [x for x in xs if x[6] >= gmax - 0.10 * abs(gmax)]
        return max(close, key=lambda x: x[6] * (1 + BOOST_LEG["w"] * x[7]) if x[6] > 0 else x[6] + BOOST_LEG["w"] * x[7] * 1e-4)
    # v2.25: Under cap (45% of legs) with the same "overwhelming math" override as the regular parlays
    ok_, hv_ = _pick([x for x in allc if not x[8]]), _pick([x for x in allc if x[8]])
    ok_ = ok_ if ok_ and (ok_[6] > 0 or not need_market_ev) else None
    hv_ = hv_ if hv_ and (hv_[6] > 0 or not need_market_ev) else None
    best = ok_
    if hv_:
        mev = lambda x: x[2] * x[4] - 1   # EV at market odds alone
        if ok_ is None:
            over_ = (mev(hv_) >= 0.25) if need_market_ev else True
            why = (f"no combo within the cap qualifies; the Under-heavy one is {mev(hv_):+.0%} at market odds"
                   + (" (needs +25%)" if need_market_ev else " and is the only play"))
        else:
            over_ = hv_[1] >= ok_[1] * PARLAY["under_override_hit"] and mev(hv_) >= mev(ok_) + PARLAY["under_override_ev"]
            why = f"Under-heavy option: hit {hv_[1]:.1%} vs {ok_[1]:.1%}, market-odds EV {mev(hv_):+.0%} vs {mev(ok_):+.0%}"
        if over_ or ok_ is None or hv_[6] > ok_[6]:
            print(f"  UNDER CAP {PARLAY['max_under_share']:.0%} ({book} {boost:g}% boost): {why} -> "
                  + ("OVERRIDE: the math clearly favors the Unders, using it" if over_ else "kept within the cap"))
        if over_:
            best = hv_
    # v2.28 NO-FILLER RULE: a leg stays only if the parlay is worth more with it. If dropping a leg leaves a legal build
    # (still meets the boost's minimum legs / odds) with a higher EV, that smaller build is used instead.
    while best is not None and len(best[5]) > max(min_legs, 2):
        ids = lambda c_: frozenset((x["player"], x["market"], x["side"]) for x in c_)
        subs = {ids(best[5]) - {i_} for i_ in ids(best[5])}
        better = [x for x in allc if ids(x[5]) in subs and x[0] > best[0] + 1e-9 and (not x[8] or best[8])]
        if not better:
            break
        nb = max(better, key=lambda x: x[0])
        gone = next(iter(ids(best[5]) - ids(nb[5])))
        print(f"  NO-FILLER ({book} {boost:g}% boost): dropped {gone[0]} {gone[2]} {gone[1]}: the parlay is worth more without it"
              f" (EV {best[0]:+.1%} -> {nb[0]:+.1%})")
        best = nb
    return best[:6] if best and (best[6] > 0 or not need_market_ev) else None


def print_boost(a, allp, stake, cols, keys):
    boost, bbook = a.boost, (a.boost_book or "draftkings").lower().replace(" ", "")
    pool = parlay_pool(allp, [x for x in (a.exclude or "").split(";") if x.strip()], min_edge=-0.03, min_prob=0.5)
    min_dec = american_to_decimal(a.boost_min_odds) if a.boost_min_odds else 1.0
    r = boost_parlay(pool, bbook, boost, stake, a.boost_min_legs or 2, a.boost_max_legs or 4, min_dec, a.boost_max_win)
    print(f"\n=== BOOST PARLAY ({boost:g}% profit boost at {BOOK_NAME.get(bbook, bbook)}) ===")
    lean = False
    if not r:
        print("  no combination is +EV at market odds with this boost (too small, or min odds/legs too strict)")
        r = boost_parlay(pool, bbook, boost, stake, a.boost_min_legs or 2, a.boost_max_legs or 4, min_dec,
                         a.boost_max_win, need_market_ev=False)
        if not r:
            print("  and no legs qualify even as a lean")
            return
        lean = True
        print("  LEAN - BEST AVAILABLE BOOST: no edge, but if you use this boost tonight, this is the least-bad way")
    ev, p, p_mkt, dec, dec_eff, combo = r
    print(f"  PLACE AT {BOOK_NAME.get(bbook, bbook).upper()} | {len(combo)} legs | est. odds {decimal_to_american(dec):+d} -> "
          f"{decimal_to_american(dec_eff):+d} boosted | ${stake:.2f} pays ~${stake * dec_eff:.2f} | hit chance {p:.1%} "
          f"(market {p_mkt:.1%}) | EV {ev:+.0%} (EV at market odds alone {p_mkt * dec_eff - 1:+.0%}) | "
          f"Unders {sum(c['side'] == 'Under' for c in combo)}")
    print(pd.DataFrame(combo)[cols].to_string(index=False))
    if a.boost_max_win:
        print(f"  (boost winnings capped at ${a.boost_max_win:g}: counted)")
    if not lean:
        keys += [(c["player"], c["market"], c["side"]) for c in combo]
    return {"name": f"BOOST {boost:g}%" + (" (LEAN)" if lean else ""), "target": 0, "book": bbook,
            "boost_pct": boost, "odds": decimal_to_american(dec), "hit": round(p, 4), "lean": lean, "ev": round(ev, 3),
            "stake_usd": stake, "legs": [[c["player"], c["market"], c["side"], float(c["line"])] for c in combo]}



def shadow_value_fn(key):
    """NBA: actual stat for (player, market) on date `key`; None if box scores aren't posted yet."""
    date = pd.Timestamp(key)
    box = load_box(season_for(date), max_age_hours=1)
    box = box[box["date"] == date] if len(box) else box
    if len(box) == 0:
        return None
    idx = {k: g.iloc[0] for k, g in box.groupby("_key")}
    def f(player, market):
        r, mk = idx.get(norm_name(player)), canon_market(market) or market
        if r is None or not bool(r["played"]) or mk not in MARKETS:
            return None
        return float(sum(r[c] for c in MARKETS[mk]))
    return f

_GS = {}

def _game_scores(date_str):
    """v2.30: finished games on a date -> {'BOS': BOS margin, 'NY': NY margin, 'NY@BOS': total points}."""
    if date_str not in _GS:
        out = {}
        try:
            p = _download(URLS["sched"].format(s=season_for(pd.Timestamp(date_str))), os.path.join(CACHE, f"sched_{season_for(pd.Timestamp(date_str))}.parquet"), 1)
            sc = pd.read_parquet(p)
            sc = sc[(pd.to_datetime(sc["game_date"]).dt.normalize().astype(str).str[:10] == date_str) & sc["status_type_completed"].astype(bool)]
            for _, g in sc.iterrows():
                h, w = norm_team(g["home_abbreviation"]), norm_team(g["away_abbreviation"])
                hs, as_ = float(g["home_score"]), float(g["away_score"])
                out[h], out[w], out[f"{w}@{h}"] = hs - as_, as_ - hs, hs + as_
        except Exception as e:
            print(f"  ! game scores unavailable for {date_str}: {e}")
        _GS[date_str] = out
    return _GS[date_str]

def cmd_final(a):
    """Final check ~30-90 min before start: re-fetched lines + latest injuries vs the slip. Flags scratched / downgraded
    players and lines that moved against us, records the current price as the closing price (CLV), writes it into the
    pick log, and rebuilds the parlays without flagged players."""
    slip = pd.read_csv(a.slip)
    allp = pd.read_csv(a.all)
    close = pd.read_csv(a.props, dtype=str).fillna("") if a.props and os.path.exists(a.props) else pd.DataFrame()
    if len(close):
        close["mk"] = close["market"].map(lambda m: canon_market(m) or m)
        close["_key"] = close["player"].map(norm_name)
    status = {}
    for part in (a.injuries or "").split(";"):
        bits = [x.strip() for x in part.split(":")]
        if len(bits) >= 2 and bits[0]:
            status[norm_name(bits[0])] = bits[1].title()
    rows, flagged = [], set()
    for _, s in slip.iterrows():
        key, mk, side = norm_name(s["player"]), canon_market(s["market"]) or s["market"], str(s["side"]).title()
        o = allp[(allp["player"].map(norm_name) == key) & (allp["market"] == mk) & (allp["side"] == side)]
        o_line = float(o.iloc[0]["line"]) if len(o) else np.nan
        o_odds = float(o.iloc[0]["odds"]) if len(o) else np.nan
        st_ = status.get(key, "")
        c_line = c_odds = np.nan
        c_book, flag = "", []
        if st_ in ("Out", "Doubtful", "Questionable", "Ir", "Gtd", "Game Time Decision", "Suspended"):
            flag.append(f"injury: {st_}")
        if len(close) and mk not in GAME_MK:   # v2.30: game lines have no closing props page yet -> never flagged "no line"
            c = close[(close["_key"] == key) & (close["mk"] == mk)]
            col = "over_odds" if side == "Over" else "under_odds"
            c = c[c[col] != ""]
            if c.empty:
                flag.append("no line posted now (scratch/news?)")
            else:
                if pd.notna(o_line) and (c["line"].astype(float) == o_line).any():
                    c = c[c["line"].astype(float) == o_line]
                r = c.iloc[0]
                c_line, c_odds, c_book = float(r["line"]), float(r[col]), side_book(r.get("book", ""), side)
                if pd.notna(o_line) and ((side == "Over" and c_line > o_line) or (side == "Under" and c_line < o_line)):
                    flag.append(f"line moved against: {o_line:g} -> {c_line:g}")
        if any(f.startswith(("injury", "no line", "line moved")) for f in flag):
            flagged.add(s["player"])
        rows.append({"player": s["player"], "market": mk, "side": side, "open_line": o_line, "open_odds": o_odds,
                     "close_line": c_line, "close_odds": c_odds, "close_book": c_book,
                     "clv_pts": round(implied_prob(c_odds) - implied_prob(o_odds), 3)
                     if pd.notna(c_odds) and pd.notna(o_odds) and c_line == o_line else np.nan,
                     "flag": "; ".join(flag) or "ok"})
    out = pd.DataFrame(rows)
    out.to_csv(a.close_out, index=False)
    pd.set_option("display.width", 250)
    print("=== FINAL CHECK (closing prices recorded) ===")
    print(out.to_string(index=False) if len(out) else "slip empty")
    if a.log and os.path.exists(a.log) and a.props and os.path.exists(a.props) and not getattr(a, "no_log", False):
        lg_ = read_log(a.log)
        cpr_ = _read_props([a.props])
        lg_, n_ = fill_closing(lg_, cpr_[cpr_["market"].notna()], str(today_et().date()), _cfg(a))
        lg_.to_csv(a.log, index=False)
        print(f"CLV: closing price written to {n_} placed bets/legs (every bet tonight, not only the slip)")
        clv_report(lg_, str(today_et().date()))
    if a.log and os.path.exists(a.log) and len(out) and not getattr(a, "no_log", False):
        log = read_log(a.log)
        if "closing_odds" not in log:
            log["closing_odds"] = np.nan
        n = 0
        for _, r in out[out["close_odds"].notna()].iterrows():
            m = ((log["player"].map(norm_name) == norm_name(r["player"])) & (log["market"] == r["market"])
                 & (log["side"] == r["side"]) & (pd.to_numeric(log["line"], errors="coerce") == r["close_line"])
                 & log["closing_odds"].isna() & log["result"].isna())
            log.loc[m, "closing_odds"] = r["close_odds"]
            n += int(m.sum())
        log.to_csv(a.log, index=False)
        print(f"\nclosing odds written to {n} rows of {a.log}")
    if flagged:
        print(f"\nFLAGGED -> rebuilding parlays without: {', '.join(sorted(flagged))}")
        a.exclude = ";".join(sorted(flagged | {x for x in (a.exclude or '').split(';') if x.strip()}))
        cmd_parlay(a)
    else:
        print("\nno flags: parlays stand as built.")


LOG_COLS = ["season", "date", "logged_at", "player", "team", "opp", "market", "side", "line", "odds", "book", "tier",
            "units", "stake_usd", "source", "model_prob", "edge", "ev_per_unit", "closing_odds", "actual", "result",
            "profit_units",
            "bet_type", "parlay_id", "suggested", "vpe_verdict", "boost_pct", "odds_eff", "edge_base", "note",
            "face_usd", "bonus_bet", "rec_units", "rec_stake_usd", "held_result", "held_profit_units", "cashout_cost"]
LOG_NUM = ["season", "line", "odds", "units", "stake_usd", "model_prob", "edge", "ev_per_unit", "closing_odds", "actual",
           "profit_units", "boost_pct", "odds_eff", "edge_base", "face_usd", "rec_units", "rec_stake_usd", "held_profit_units", "cashout_cost"]

def read_log(path):
    """Pick log CSV: every column read as text first (a float written into a text column crashed pandas), then the
    numeric ones converted."""
    d = pd.read_csv(path, dtype=object)
    for c in LOG_NUM:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# ---------- Google Sheet sync: the Pick Log sheet is the source of truth ----------
def _cell(v):
    """A pandas value -> a Sheets cell (numbers stay numbers, NaN -> blank, '=' text kept as text)."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return "" if math.isnan(v) else (int(v) if float(v).is_integer() and abs(v) < 1e15 else round(float(v), 4))
    t = str(v)
    if t.lower() == "nan":
        return ""
    # keep dates / timestamps / ids as text so the sheet shows (and returns) exactly what the engine wrote
    return "'" + t if t[:1] in ("=", "+") or re.match(r"^\d{4}-\d{2}-\d{2}", t) else t

def sheet_rows(df, header=False):
    d = df.reindex(columns=LOG_COLS)
    rows = [[_cell(v) for v in r] for r in d.itertuples(index=False, name=None)]
    return ([list(LOG_COLS)] if header else []) + rows

def cmd_sheet(a):
    """pull: Sheet tab values (get_values JSON or a 2D list) -> local log CSV.
    push: local log CSV -> full 2D array (header + all rows) to write at A1 of the tab."""
    import json
    if a.mode == "pull":
        raw = json.load(open(a.json))
        vals = raw.get("values", []) if isinstance(raw, dict) else raw
        if not vals:
            pd.DataFrame(columns=LOG_COLS).to_csv(a.log, index=False)
            print(f"sheet empty -> {a.log} (0 rows)")
            return
        hdr = vals[0]
        body = [list(r) + [""] * (len(hdr) - len(r)) for r in vals[1:]]
        df = pd.DataFrame(body, columns=hdr).replace({"": np.nan}).reindex(columns=LOG_COLS)
        df.to_csv(a.log, index=False)
        print(f"pulled {len(df)} rows -> {a.log}")
    else:
        log = read_log(a.log)
        out = sheet_rows(log, header=True)
        json.dump(out, open(a.out, "w"))
        print(f"{len(out) - 1} rows + header -> {a.out} (write it at A1 of the tab with update_values)")

def _all_runs(path):
    """This run's bets_all.csv first, then every earlier run of the same night (runs/), newest first: a bet from a
    replaced slate is still scored with the projection it was picked from."""
    import glob
    frames = [pd.read_csv(path)] if os.path.exists(path) else []
    d0 = str(frames[0]["date"].iloc[0]) if frames and "date" in frames[0] and len(frames[0]) else None
    for p in sorted(glob.glob(os.path.join(os.path.dirname(path) or ".", "runs", "bets_all_*.csv")), reverse=True):
        if d0 is None or os.path.basename(p).startswith(f"bets_all_{d0}_"):
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    if len(frames) > 1:
        print(f"LOG: projections from {len(frames)} runs tonight (this run first, then earlier ones)")
    return d.drop_duplicates(["player", "market", "side", "line"], keep="first").reset_index(drop=True)

def cmd_log(a):
    set_calib(getattr(a, "calib", None), SPORT)
    """Log bets actually placed (from DraftKings screenshots), including bets VPE did not suggest (profit boosts,
    own picks). Each is re-scored at the line/odds really taken, using the projections saved by the last `check`."""
    cfg = _cfg(a)
    unit = LOG_UNIT_USD  # v2.16: the Pick Log unit is $1, so units == stake_usd (was 2% of --bankroll)
    shrink = cfg["model"]["prob_calibration_shrink"]
    allp = _all_runs(a.all)
    lock_from(allp)

    def score(player, mk, side, line, odds):
        hit = allp[(allp["player"].map(norm_name) == norm_name(player)) & (allp["market"] == mk)] if len(allp) else allp
        if hit.empty:
            return None, ("no game-line rating saved for this game (run games after check)" if mk in GAME_MK
                          else "no projection saved for this player/market (run check first)")
        h = hit.iloc[0]
        if mk in GAME_MK:   # v2.30: a game line, scored like a prop at the line and price actually taken
            ex = hit[(hit["side"] == side) & (pd.to_numeric(hit["line"], errors="coerce") == float(line))]
            e0 = ex.iloc[0] if len(ex) else None
            if e0 is not None:
                pb = float(e0["blend_prob"])
            else:   # an alt spread / alt total: read off the blended centre
                hs_ = hit[hit["side"] == "Over"]
                hh = hs_.iloc[0] if len(hs_) else h
                po = prob_over(float(hh["mean"]), float(hh["var"]), mk, line)
                pb = po if side == "Over" else 1 - po
            return {"prob": pb, "leg_prob": pb, "team": h["team"], "opp": h["opp"],
                    "rec_units": float(e0["units"]) if e0 is not None and pd.notna(e0.get("units")) else 0.0,
                    "min_odds": e0.get("min_odds") if e0 is not None and pd.notna(e0.get("tier")) and str(e0.get("tier")) not in ("", "nan") else None}, ""
        po = cal_prob(prob_over(h["mean"], h["var"], mk, line))
        pr = po if side == "Over" else 1 - po
        mp = market_prob_for(allp, player, mk, side, line, odds)
        pb = blend_leg(pr, mp, mk, side)
        ex = hit[(hit["side"] == side) & (pd.to_numeric(hit["line"], errors="coerce") == float(line))]
        e0 = ex.iloc[0] if len(ex) else None
        return {"held": held_reason(allp, player, mk, side, line), "prob": pb, "leg_prob": pb, "team": h["team"], "opp": h["opp"],
                "rec_units": float(e0["units"]) if e0 is not None and pd.notna(e0.get("units")) else 0.0,
                "min_odds": e0.get("min_odds") if e0 is not None and pd.notna(e0.get("tier")) and str(e0.get("tier")) not in ("", "nan") else None}, ""

    slip = slip_keys(a.slip)
    import json as _j
    pp_ = getattr(a, "parlays", None) or "vpe_parlays.json"
    VPE_PARLAYS[:] = _j.load(open(pp_)) if os.path.exists(pp_) else []
    _sr = getattr(a, "sgp_ratings", None) or "sgp_ratings.json"
    SGP_RATINGS[:] = _j.load(open(_sr)) if os.path.exists(_sr) else []
    stamp = {"season": season_for(today_et()),
             "date": allp["date"].iloc[0] if len(allp) else str(today_et().date())}
    rows = build_log_rows(pd.read_csv(a.placed, dtype=str), score, cfg, unit, slip, stamp)
    if getattr(a, "dry_run", False):
        print_logged(rows)
        print("\nDRY RUN (rate only): nothing logged")
        return
    new = pd.DataFrame(rows).reindex(columns=LOG_COLS)
    if a.log and os.path.exists(a.log):  # v2.16: the same bet logged twice (a rerun of `log`) is skipped, not doubled
        old_ = read_log(a.log)
        kc = [c for c in ("date", "week", "player", "market", "side", "line", "odds", "stake_usd", "bet_type", "parlay_id", "book")
              if c in new and c in old_]
        def _v(v):
            if v is None or (isinstance(v, float) and math.isnan(v)) or str(v) in ("nan", "None", "<NA>"):
                return ""
            try:
                return f"{float(v):g}"
            except (TypeError, ValueError):
                return str(v).strip()
        sig = lambda df: pd.Series(["|".join(_v(v) for v in row) for row in df[kc].itertuples(index=False)], index=df.index)
        dup = sig(new).isin(set(sig(old_))).to_numpy()
        if dup.any():
            print(f"LOG: {int(dup.sum())} row(s) were already in the pick log (same bet logged before) -> skipped")
            rows = [r for r, d_ in zip(rows, dup) if not d_]
            new = new[~dup]
    if os.path.exists(a.log):
        new = pd.concat([read_log(a.log), new], ignore_index=True).reindex(columns=LOG_COLS)
    new.to_csv(a.log, index=False)
    print_logged(rows)
    import json
    json.dump(sheet_rows(pd.DataFrame(rows)), open(a.sheet_out, "w"))
    print(f"sheet rows to append -> {a.sheet_out}")
    if slip is None:
        print(f"  ! no slip found at {a.slip}: run `check --out bets.csv` first so VPE picks can be told apart")
    print(f"\nlogged {len(rows)} rows -> {a.log} (total rows {len(new)})")

def cmd_grade(a):
    cfg = _cfg(a)
    log = read_log(a.log)
    for c in ("actual", "profit_units", "closing_odds", "odds_eff"):
        log[c] = pd.to_numeric(log.get(c), errors="coerce")
    for c in ("bet_type", "parlay_id", "suggested", "vpe_verdict"):
        if c not in log.columns: log[c] = None
    date = pd.Timestamp(a.date) if a.date else None
    if a.actuals:
        act = _read_loose_csv(a.actuals)
        act["_key"] = act["player"].map(norm_name)
        for c in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
            act[c] = pd.to_numeric(act.get(c, 0), errors="coerce").fillna(0)
        box = act.assign(played=True)
        box_all = box
    else:
        season = season_for(date or today_et())
        box = load_box(season, max_age_hours=1)
        box_all = box
        if len(box) and date is not None:
            box = box[box["date"] == date]
    mask = log["result"].isna() & (log["date"].astype(str) == str(date.date()) if date is not None else log["result"].isna())
    if len(box) == 0:
        print("No box scores for that date yet (the free data usually posts the next morning). Retry later or use --actuals.")
        return
    for i in log[mask].index:
        r = log.loc[i]
        mk = r["market"]
        if mk in GAME_MK:   # v2.30: moneylines, spreads and totals grade from the final score
            gs_ = _game_scores(str(r["date"])[:10])
            v_ = gs_.get(str(r["player"]).strip().upper())
            if v_ is None:
                continue
            log.loc[i, "actual"] = v_
            x_ = v_ + float(r["line"]) if mk == "spread" else (v_ if mk == "ml" else v_ - float(r["line"]))
            if mk == "total" and r["side"] == "Under":
                x_ = -x_
            res = "PUSH" if x_ == 0 else ("WIN" if x_ > 0 else "LOSS")
            log.loc[i, "result"] = res
            log.loc[i, "profit_units"] = (win_profit(r) if res == "WIN" else (-r["units"] if res == "LOSS" else 0.0))
            continue
        if mk not in MARKETS:
            continue
        row = box[box["_key"] == norm_name(r["player"])]
        if row.empty or not bool(row["played"].iloc[0]):
            log.loc[i, "result"], log.loc[i, "profit_units"] = "VOID", 0.0
            continue
        actual = float(row[MARKETS[mk]].sum(axis=1).iloc[0])
        log.loc[i, "actual"] = actual
        res = "PUSH" if actual == r["line"] else ("WIN" if (actual > r["line"]) == (r["side"] == "Over") else "LOSS")
        log.loc[i, "result"] = res
        log.loc[i, "profit_units"] = (win_profit(r) if res == "WIN" else (-r["units"] if res == "LOSS" else 0.0))
    log = settle_parlays(log)

    def _actual(r):
        if r["market"] not in MARKETS:
            return None
        b = box_all[box_all["_key"] == norm_name(r["player"])]
        if "date" in b and pd.notna(r.get("date")):
            b = b[b["date"] == pd.Timestamp(str(r["date"]))]
        if b.empty or not bool(b["played"].iloc[0]):
            return None
        return float(b[MARKETS[r["market"]]].sum(axis=1).iloc[0])
    log = settle_cashouts(log, _actual)
    log.to_csv(a.log, index=False)
    pd.set_option("display.width", 250)
    print(log[["date", "player", "market", "side", "line", "odds", "odds_eff", "units", "suggested", "actual", "result",
               "profit_units"]].tail(40).to_string(index=False))
    g = log[log["result"].isin(["WIN", "LOSS", "PUSH"]) & (log["bet_type"].fillna("straight") != "leg")]
    if len(g):
        for col in (None, "suggested", "vpe_verdict", "bet_type", "tier", "market", "side", "source"):
            grp = [("all", g)] if col is None else list(g.groupby(col))
            for k, d in grp:
                w, l = (d["result"] == "WIN").sum(), (d["result"] == "LOSS").sum()
                print(f"{col or 'overall'}={k}: {w}-{l} | units {d['profit_units'].sum():+.2f} on {d['units'].sum():.2f} staked "
                      f"| ROI {d['profit_units'].sum() / max(d['units'].sum(), 1e-9):+.1%} | avg model prob {pd.to_numeric(d['model_prob'], errors='coerce').mean():.3f}")
        c = g[pd.to_numeric(g["closing_odds"], errors="coerce").notna() & pd.to_numeric(g["odds"], errors="coerce").notna()]
        if len(c):
            clv = c["closing_odds"].astype(float).map(implied_prob) - c["odds"].astype(float).map(implied_prob)
            print(f"CLV: {len(c)} bets with a closing price | beat the close {(clv > 0).mean():.0%} | avg {clv.mean():+.3f} prob pts")


# ==================== game lines + same-game parlays (RATE-ONLY) ====================
# Built in the run chat 10/26 (vpe_sgp.py / team_model.py / nba_corr.json), folded in here in v2.3.
# Team ratings: opponent-adjusted offense/defense points, ridge, time-decayed (half-life 60 days), last season as prior.
# Injury-aware (v2.3): each missing rotation player costs his team INJ_BETA x (his game score per minute above
# replacement x his minutes x how often he has been playing lately) points of margin. Walk-forward 2025-26: model
# margin MAE 11.84 -> 11.42 (t=+3.5) and 11.11 -> 10.99 in the other half; totals NOT adjusted (didn't help).
# Honest status: vs DK CLOSING lines the market alone is best (any model weight made MAE worse). vs OPENING lines the
# injury-aware model adds value (open MAE 11.31 -> 11.25 at 30% model) and its gap predicts where the line moves
# (corr 0.47). So a game-line edge exists only BEFORE the line moves -> rate-only until CLV shows it in real bets.
GAME = {"model_w": 0.30, "sd_margin": 12.5, "sd_total": 18.0, "held_margin": 4.0, "held_total": 6.0,
        "inj_beta": 0.42, "repl_gs_per_min": 0.25,
        # v2.32 REST EDGE (Adrian 11/18, tested): a team on the 2nd night of a back-to-back against a rested opponent ran
        # 2.6 pts (se 0.6) under the ratings over 596 such games in 2024-25 + 2025-26 (home team tired -3.4, visitor tired
        # -1.9). A flat 2.0 lowered the margin miss in both seasons when fit on the other. Team-by-team home edges were
        # tested the same day and NOT added: they made the miss worse at every setting and a team's home edge did not carry
        # over (corr 0.05 half to half, 0.00 season to season).
        "rest_pts": 2.0, "half_life": 60, "lam": 2.0, "prior_w": 6.0}
SGP_CORR = {"sdM_market": 12.5, "sdT_market": 18.0, "bins": [0, 5, 10, 15, 20, 100], "stats": ["pts", "reb", "ast", "fg3m"], "starter": {"pts": {"bT": 0.16, "bM": 0.095, "shift": [0.045, 0.092, 0.035, 0.002, -0.164], "res_sd": 1.007}, "reb": {"bT": -0.05, "bM": 0.057, "shift": [0.096, 0.116, 0.02, -0.079, -0.17], "res_sd": 1.015}, "ast": {"bT": 0.113, "bM": 0.098, "shift": [0.004, 0.026, 0.017, 0.02, -0.058], "res_sd": 1.024}, "fg3m": {"bT": 0.105, "bM": 0.095, "shift": [-0.001, 0.02, 0.022, 0.035, -0.061], "res_sd": 0.976}}, "bench": {"pts": {"bT": 0.076, "bM": 0.035, "shift": [-0.044, -0.009, -0.048, -0.038, 0.093], "res_sd": 0.934}, "reb": {"bT": -0.056, "bM": 0.017, "shift": [0.04, 0.006, -0.03, -0.044, 0.016], "res_sd": 0.945}, "ast": {"bT": 0.062, "bM": 0.041, "shift": [-0.041, -0.001, -0.03, -0.038, 0.072], "res_sd": 0.931}, "fg3m": {"bT": 0.061, "bM": 0.043, "shift": [-0.02, -0.006, -0.017, -0.036, 0.052], "res_sd": 0.89}}, "within": [[1.0, 0.333, 0.209, 0.661], [0.333, 1.0, 0.255, 0.153], [0.209, 0.255, 1.0, 0.112], [0.661, 0.153, 0.112, 1.0]], "teammate": [[-0.009, 0.028, 0.049, -0.019], [0.028, 0.004, 0.016, 0.007], [0.049, 0.016, 0.003, 0.035], [-0.019, 0.007, 0.035, -0.014]], "opponent": [[0.012, 0.007, 0.008, 0.006], [0.007, 0.015, 0.006, 0.006], [0.008, 0.006, 0.006, 0.005], [0.006, 0.006, 0.005, 0.001]]}

def _team_games(box):
    b = box[box["played"]]
    return b.groupby(["game_id", "date", "team", "opp", "home"])["pts"].sum().reset_index()

def _fit_ratings(games, asof, half_life=40, lam=4.0, prior=None, prior_w=10.0):
    g = games[games["date"] < asof]
    teams = sorted(set(games["team"]) | set(games["opp"]))
    ix = {t: i for i, t in enumerate(teams)}; n = len(teams)
    w = 0.5 ** ((asof - g["date"]).dt.days.to_numpy() / half_life)
    X = np.zeros((len(g), 2 * n + 2)); X[:, 0] = 1
    X[np.arange(len(g)), 1] = np.where(g["home"].to_numpy(), 1.0, -1.0) / 2
    X[np.arange(len(g)), 2 + g["team"].map(ix).to_numpy()] = 1
    X[np.arange(len(g)), 2 + n + g["opp"].map(ix).to_numpy()] = 1
    y = g["pts"].to_numpy(float); W = np.sqrt(w)[:, None]
    P = np.eye(2 * n + 2) * lam; P[0, 0] = P[1, 1] = 1e-6
    target = np.zeros(2 * n + 2)
    if prior is not None:
        _, _, po, pdf = prior
        for t, i in ix.items():
            target[2 + i] = po.get(t, 0.0); target[2 + n + i] = pdf.get(t, 0.0)
        P[2:, 2:] = np.eye(2 * n) * prior_w
    beta = np.linalg.solve((X * W).T @ (X * W) + P, (X * W).T @ (y * np.sqrt(w)) + P @ target)
    return beta[0], beta[1], {t: beta[2 + i] for t, i in ix.items()}, {t: beta[2 + n + i] for t, i in ix.items()}

def _predict(model, home, away):
    mu, hca, off, deff = model
    ph = mu + hca / 2 + off.get(home, 0) + deff.get(away, 0)
    pa = mu - hca / 2 + off.get(away, 0) + deff.get(home, 0)
    return ph - pa, ph + pa

def team_ratings(state, date):
    prev = getattr(state, "pv_box", None)
    gp = _team_games(prev) if prev is not None and len(prev) else None
    prior = _fit_ratings(gp, gp["date"].max() + pd.Timedelta(days=1), GAME["half_life"], GAME["lam"]) if gp is not None else None
    gc = _team_games(state.cur) if len(state.cur) else None
    if gc is None or not len(gc):
        return prior
    return _fit_ratings(gc, pd.Timestamp(date), GAME["half_life"], GAME["lam"], prior=prior, prior_w=GAME["prior_w"])

def player_value(state, pid, team):
    """Points of margin a player is worth to his team tonight: game score per minute above replacement x minutes x
    how often he has been playing (a long absence is already in the team rating -> small)."""
    P = state.players
    if pid not in P.index:
        return 0.0
    r = P.loc[pid]
    ftm = max(r["r_pts"] - 2 * r["r_fgm"] - r["r_fg3m"], 0)
    gs = (r["r_pts"] + 0.4 * r["r_fgm"] - 0.7 * r["r_fga"] - 0.4 * (r["r_fta"] - ftm) + 0.7 * r["r_oreb"] + 0.3 * r["r_dreb"]
          + r["r_stl"] + 0.7 * r["r_ast"] + 0.7 * r["r_blk"] - 0.4 * r["r_pf"] - r["r_tov"])
    av = state.avail.get((team, pid), 0.6)
    return max(gs - GAME["repl_gs_per_min"], 0) * float(r["min"]) * av

def injury_shift(state, outs):
    """Home-margin shift per game: sum over each team's Out/Doubtful players of inj_beta x value."""
    lost = {}
    if outs is not None and len(outs):
        for _, o in outs.iterrows():
            if o["min"] >= 12:
                lost[o["team"]] = lost.get(o["team"], 0.0) + player_value(state, o["pid"], o["team"])
    return lost

def game_center(model, home, away, mkt_spread, mkt_total, lost=None):
    m_mod, t_mod = _predict(model, home, away)
    lost = lost or {}
    m_mod += GAME["inj_beta"] * (lost.get(away, 0.0) - lost.get(home, 0.0))
    m_mkt = -mkt_spread if mkt_spread is not None and not pd.isna(mkt_spread) else m_mod
    t_mkt = mkt_total if mkt_total is not None and not pd.isna(mkt_total) else t_mod
    w = GAME["model_w"]
    return w * m_mod + (1 - w) * m_mkt, w * t_mod + (1 - w) * t_mkt, m_mod, t_mod, m_mkt, t_mkt

def fit_game_w(hist_path):
    """v2.30: learn how much say the team ratings get on game lines from the graded 'Games NBA' tab. For margin and total
    separately: the model weight (0-60%) with the smallest average miss, shrunk toward the 30% start (100 games = half trust)."""
    out = {"margin": GAME["model_w"], "total": GAME["model_w"], "n": 0}
    try:
        vals = _values(hist_path)
        d = pd.DataFrame([list(r) + [""] * (len(vals[0]) - len(r)) for r in vals[1:]], columns=vals[0])
        for c in ("model_margin", "model_total", "home_spread", "total", "act_margin", "act_total"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        g = d.dropna(subset=["act_margin", "model_margin", "home_spread"])
        grid = np.arange(0, 0.61, 0.05)
        def best(mod, mkt, act):
            return float(grid[int(np.argmin([np.abs(w * mod + (1 - w) * mkt - act).mean() for w in grid]))])
        n = len(g)
        if n >= 20:
            wm = best(g["model_margin"], -g["home_spread"], g["act_margin"])
            gt = g.dropna(subset=["act_total", "model_total", "total"])
            wt = best(gt["model_total"], gt["total"], gt["act_total"]) if len(gt) >= 20 else GAME["model_w"]
            k = n / (n + 100.0)
            out = {"margin": round(k * wm + (1 - k) * GAME["model_w"], 3), "total": round(k * wt + (1 - k) * GAME["model_w"], 3), "n": int(n),
                   "fit_margin": wm, "fit_total": wt,
                   "mae": [round(float(np.abs(g["model_margin"] - g["act_margin"]).mean()), 2), round(float(np.abs(-g["home_spread"] - g["act_margin"]).mean()), 2),
                           round(float(np.abs(gt["model_total"] - gt["act_total"]).mean()), 2) if len(gt) else None,
                           round(float(np.abs(gt["total"] - gt["act_total"]).mean()), 2) if len(gt) else None]}
    except Exception as e:
        print(f"  ! game-line history unreadable ({e}): using the {GAME['model_w']:.0%} start")
    return out


def game_card(r, x, labels, cfg):
    """v2.30: the full logic of a moneyline / spread / total, numbered like a player-prop card."""
    g = x.get("game", {})
    mk, odds = r["market"], int(r["odds"])
    nm = g.get("label", f"{r['player']} {mk}")
    L = [f"=== {nm} ({ {'spread': 'spread', 'ml': 'moneyline', 'total': 'game total'}[mk] }) | {g.get('away')} @ {g.get('home')} | {odds:+d} at {BOOK_NAME.get(str(r.get('book', '')).split('/')[0], r.get('book', ''))} | {', '.join(labels)} ==="]
    L.append(f" 1 Team ratings (points per game vs an average team, opponent-adjusted, recent games weigh more, last season as a prior): "
             f"{g.get('home')} offense {g.get('off_h', 0):+.1f} / defense {g.get('def_h', 0):+.1f} | {g.get('away')} offense {g.get('off_a', 0):+.1f} / defense {g.get('def_a', 0):+.1f} "
             f"(defense: minus = allows fewer) | home court {g.get('hca', 0):+.1f}")
    L.append(f" 2 Injuries: {g.get('inj_txt') or 'no rotation player listed Out / Doubtful'} -> home margin moves {g.get('inj', 0):+.1f}"
             f" | Rest: {g.get('rest_txt', 'n/a')} -> home margin moves {g.get('rest', 0):+.1f} (home court is one league-wide number: team-by-team home edges were tested and did not hold up)")
    by = lambda m: f"{g.get('home') if m >= 0 else g.get('away')} by {abs(m):.1f}"
    L.append(f" 3 Ratings say: {by(g.get('m_mod', 0))}, total {g.get('t_mod', 0):.1f}")
    L.append(f" 4 The book says: {by(g.get('m_mkt', 0))}, total {g.get('t_mkt', 0):.1f}")
    L.append(f" 5 Mix: {g.get('w', 0):.0%} ratings + {1 - g.get('w', 0):.0%} book ({g.get('w_why', '')}) -> VPE's number {g.get('center', 0):+.1f}"
             + (" (total)" if mk == "total" else f" for {r['team']}") + f"; games land about {g.get('sd', 0):g} points either side of it")
    L.append(f" 6 Market chance: {g.get('nv', 0):.1%} (no-vig, {g.get('nv_how', '')})")
    L.append(f" 7 Ratings alone: {float(r['model_prob']):.1%} | win chance (the mix): {float(r['blend_prob']):.1%}")
    be = implied_prob(odds)
    edge = float(r["blend_prob"]) - be
    tier = r.get("tier") if isinstance(r.get("tier"), str) and r.get("tier") not in ("", "nan") else None
    why = g.get("block") or ("no tier: edge short of Tier 3 (+2.5%)" if not tier else "")
    L.append(f" 8 Price {odds:+d} -> break-even {be:.1%} -> edge {edge:+.1%} -> " + (tier.replace("_", " ").upper() if tier else why))
    L.append(f" 9 Stake: " + (f"{float(r['units']):g}u (quarter-Kelly, capped by tier); take-if {int(r['min_odds']):+d} or better" if tier and pd.notna(r.get("min_odds")) else "none (not a straight pick; can still be a parlay / boost leg if the leg rules pass)"))
    L.append(f"10 Risk: ratings vs book gap {abs(float(r['model_prob']) - float(r['market_prob'])):.1%} (over 10% = HELD: usually news the ratings have not seen). "
             f"Track record so far: {g.get('track', 'no graded games yet')}")
    L.append(f"   In one line: {g.get('one', '')}")
    return "\n".join(L)


def cmd_games(a):
    """v2.30: moneylines, spreads and totals scored like props (win chance, edge, tier, take-if, breakdown) and merged
    into bets_all.csv so the parlay / boost / lean builders, `explain`, `log` and `grade` treat them as any other prop."""
    import json
    cfg = _cfg(a)
    date, state, sched = _setup(a, cfg)
    model = team_ratings(state, date)
    mu_, hca_, off_, def_ = model
    g = _read_loose_csv(a.games)
    g["home"], g["away"] = g["home"].map(norm_team), g["away"].map(norm_team)
    for c in g.columns:
        if c not in ("home", "away"):
            g[c] = pd.to_numeric(g[c].astype(str).str.replace("+", "", regex=False).replace({"PK": "0", "pk": "0"}), errors="coerce")
    g = guard_games(g, sched, date)
    inj = parse_injuries(a.injuries)
    outs = build_outs(state, inj, {})
    lost = injury_shift(state, outs)
    W = fit_game_w(a.hist) if getattr(a, "hist", None) and os.path.exists(a.hist) else {"margin": GAME["model_w"], "total": GAME["model_w"], "n": 0}
    json.dump(W, open("games_w.json", "w"))
    _GW.clear(); _GW.update(W)
    track = (f"{W['n']} graded games: ratings miss the margin by {W['mae'][0]} pts vs the book's {W['mae'][1]}; totals {W['mae'][2]} vs {W['mae'][3]}"
             if W.get("mae") else "under 20 graded games")
    w_why = lambda k: (f"learned from {W['n']} graded games, best fit {W.get('fit_' + k, 0):.0%}, shrunk toward the 30% start" if W.get("n") else "the 30% start: no graded history passed in")
    allp = pd.read_csv(a.all) if getattr(a, "all", None) and os.path.exists(a.all) else pd.DataFrame()
    if len(allp):
        allp = allp[~allp["market"].isin(GAME_MK)]
    gid = dict(zip(allp["team"], allp["game_id"])) if len(allp) else {}
    book = (a.book or (allp["book"].astype(str).str.split("/").str[0].mode().iloc[0] if len(allp) else "draftkings")).lower()
    profile = cfg["strategy_profiles"][cfg["strategy_profile"]]
    unit_pct, kf, smax = cfg["unit_pct_of_bankroll"], cfg["staking"]["kelly_fraction"], cfg["staking"].get("straight_max_odds", 200)
    gapmax = cfg["model"]["max_market_gap"]
    rows, log = [], []
    _tg = _team_games(state.cur) if len(state.cur) else None
    _lastg = _tg[_tg["date"] < pd.Timestamp(date)].groupby("team")["date"].max().to_dict() if _tg is not None and len(_tg) else {}
    b2b_ = lambda t: t in _lastg and (pd.Timestamp(date) - _lastg[t]).days == 1
    names = {}
    if outs is not None and len(outs):
        for _, o in outs.iterrows():
            if o["min"] >= 12:
                names.setdefault(o["team"], []).append(f"{o['name']} ({GAME['inj_beta'] * player_value(state, o['pid'], o['team']):.1f} pts)")
    def nv2(o1, o2):
        if o1 is None or pd.isna(o1) or o2 is None or pd.isna(o2):
            return None
        a_, b_ = implied_prob(int(o1)), implied_prob(int(o2))
        return a_ / (a_ + b_)
    for _, r in g.iterrows():
        home, away = r["home"], r["away"]
        hs, tot = r.get("home_spread"), r.get("total")
        m_mod, t_mod = _predict(model, home, away)
        ishift = GAME["inj_beta"] * (lost.get(away, 0.0) - lost.get(home, 0.0))
        m_mod += ishift
        th, ta = b2b_(home), b2b_(away)
        rshift = GAME["rest_pts"] * ((1 if ta and not th else 0) - (1 if th and not ta else 0))
        rest_txt = (f"{away} on the 2nd night of a back-to-back, {home} rested" if rshift > 0 else
                    f"{home} on the 2nd night of a back-to-back, {away} rested" if rshift < 0 else
                    "both on a back-to-back" if th and ta else "no rest mismatch")
        m_mod += rshift
        m_mkt = -hs if pd.notna(hs) else m_mod
        t_mkt = tot if pd.notna(tot) else t_mod
        wm, wt = W["margin"], W["total"]
        cm, ct = wm * m_mod + (1 - wm) * m_mkt, wt * t_mod + (1 - wt) * t_mkt
        inj_note = f"inj {ishift:+.1f}" if (lost.get(home) or lost.get(away)) else ""
        inj_txt = "; ".join(f"{t} without {', '.join(v)}" for t, v in names.items() if t in (home, away))
        sdm, sdt = GAME["sd_margin"], GAME["sd_total"]
        game_id = gid.get(home, gid.get(away, f"{away}@{home}"))
        def leg(mk, team, opp, is_home, side, line, odds, other, label):
            if mk == "total":
                mod_c, mkt_c, cen, sd, w = t_mod, t_mkt, ct, sdt, wt
            else:
                sgn = 1 if is_home else -1
                mod_c, mkt_c, cen, sd, w = sgn * m_mod, sgn * m_mkt, sgn * cm, sdm, wm
            assumed = odds is None or pd.isna(odds)
            odds = -110 if assumed else int(odds)
            thr = -line if mk == "spread" else (0.0 if mk == "ml" else line)
            f = (lambda c: float(st.norm.sf(thr, c, sd))) if side == "Over" else (lambda c: float(st.norm.cdf(thr, c, sd)))
            nv = nv2(odds, other)
            nv_how = "from both prices" if nv is not None else "one price only: the book's own line taken as fair"
            nv = f(mkt_c) if nv is None else nv
            pm = float(min(max(nv + f(mod_c) - f(mkt_c), 0.01), 0.99))
            pb = float(blend_leg(pm, nv, mk, side))
            be = implied_prob(odds)
            edge = pb - be
            gap = abs(pm - nv)
            block = ""
            tier = None
            if gap > gapmax:
                block = f"HELD: ratings vs book gap {gap:.1%} is over {gapmax:.0%} (usually news the ratings have not seen)"
            elif _killed(mk, side, odds):
                block = "KILL list: nothing at -130 or shorter without a boost (or this slice is on the learned KILL list)"
            elif odds > smax:
                block = f"long shot over +{smax}: never a straight pick"
            else:
                tier = _tier(edge, pb, cfg)
                if tier == "tier_3" and not profile["allow_tier3"]:
                    tier, block = None, "Tier 3 off in this profile"
            units = 0.0
            if tier:
                units = round(min(kf * kelly_fraction(pb, odds) / unit_pct, cfg["tiers"][tier]["max_units"]) * 4) / 4
                if units < 0.25:
                    tier, units, block = None, 0.0, "stake rounds below 0.25u"
            mo = int(decimal_to_american(1 / min(max(pb - cfg["tiers"]["tier_3"]["min_edge"], 0.02), 0.98))) if tier else None
            one = (f"ratings {'like' if pm > nv else 'do not like'} this side ({pm:.1%} vs the book's {nv:.1%}); after the mix VPE says {pb:.1%} "
                   f"against {be:.1%} needed at {odds:+d} -> {'a pick' if tier else 'not a straight pick'}")
            x = {"game": {"home": home, "away": away, "label": label, "off_h": off_.get(home, 0), "def_h": def_.get(home, 0),
                          "off_a": off_.get(away, 0), "def_a": def_.get(away, 0), "hca": hca_, "inj": ishift, "inj_txt": inj_txt, "rest": rshift, "rest_txt": rest_txt,
                          "m_mod": m_mod, "t_mod": t_mod, "m_mkt": m_mkt, "t_mkt": t_mkt, "w": w, "w_why": w_why("total" if mk == "total" else "margin"),
                          "center": cen, "sd": sd, "nv": nv, "nv_how": nv_how + ("; price not recorded, -110 assumed" if assumed else ""),
                          "block": block, "track": track, "one": one}}
            rows.append({"player": team if mk != "total" else f"{away}@{home}", "team": team, "opp": opp, "home": is_home, "game_id": game_id,
                         "market": mk, "side": side, "line": line, "odds": odds, "book": book, "mean": round(cen, 3), "var": round(sd * sd, 3),
                         "proj_min": 48.0, "base_min": 48.0, "model_prob": round(pm, 3), "blend_prob": round(pb, 3), "market_prob": round(nv, 3),
                         "breakeven": round(be, 3), "edge": round(edge, 3), "ev_per_unit": round(ev_per_unit(pb, odds), 3), "tier": tier, "units": units,
                         "injury": "", "usage_note": inj_note, "new_team": False, "dvp": 1.0, "pace_f": 1.0, "b2b": False, "spread": hs,
                         "n_cur": 99, "min_odds": mo, "line_limit": None, "arch": "", "opp_style": "", "style_f": 1.0, "env": 1.0,
                         "limited": False, "returning": False, "role_risk": "STABLE", "role_why": "team market: no single player's minutes decide it",
                         "x": json.dumps(x), "mkt_est": False, "market_gap": round(gap, 3), "locked": True, "locked_books": book,
                         "date": str(date.date()), "label": label, "verdict": (tier or block.split(":")[0] or "pass")})
        if pd.notna(hs):
            leg("spread", home, away, True, "Over", float(hs), r.get("spread_home_odds"), r.get("spread_away_odds"), f"{home} {hs:+g}")
            leg("spread", away, home, False, "Over", float(-hs), r.get("spread_away_odds"), r.get("spread_home_odds"), f"{away} {-hs:+g}")
        if pd.notna(tot):
            leg("total", home, away, True, "Over", float(tot), r.get("over_odds"), r.get("under_odds"), f"{away}@{home} Over {tot:g}")
            leg("total", home, away, True, "Under", float(tot), r.get("under_odds"), r.get("over_odds"), f"{away}@{home} Under {tot:g}")
        if pd.notna(r.get("home_ml")) and pd.notna(r.get("away_ml")):
            leg("ml", home, away, True, "Over", 0.0, r.get("home_ml"), r.get("away_ml"), f"{home} ML")
            leg("ml", away, home, False, "Over", 0.0, r.get("away_ml"), r.get("home_ml"), f"{away} ML")
        log.append([str(date.date()), away, home, round(m_mod, 2), round(t_mod, 2),
                    "" if pd.isna(hs) else float(hs), "" if pd.isna(tot) else float(tot), (inj_note + (f" rest {rshift:+.1f}" if rshift else "")).strip(), "", ""])
    out = pd.DataFrame(rows)
    # one straight game-line pick per game (the best edge); the rest stay scored and can still be parlay / boost legs
    if len(out):
        tiered = out[out["tier"].notna()].sort_values("edge", ascending=False)
        drop = tiered[tiered.duplicated("game_id", keep="first")].index
        out.loc[drop, ["tier", "units", "min_odds", "verdict"]] = [None, 0.0, None, "pass (another line in this game is the pick)"]
    pd.set_option("display.width", 250)
    print(f"GAME LINES [{VERSION}] - scored like props | win chance = {1 - W['margin']:.0%} book + {W['margin']:.0%} injury-aware team ratings "
          f"(totals {W['total']:.0%}) | HELD = ratings vs book gap > {gapmax:.0%} | track record: {track}")
    if W.get("mae") and W["mae"][0] > W["mae"][1]:
        print("  HONEST NOTE: the ratings have not beaten the book's lines so far, so their say is small and most game lines "
              "will read as no edge. Say so when showing them. They earn more weight only by results.")
    show = ["label", "odds", "model_prob", "market_prob", "blend_prob", "edge", "verdict", "units", "min_odds", "usage_note"]
    print(out[show].rename(columns={"label": "bet", "model_prob": "ratings", "market_prob": "book", "blend_prob": "win", "usage_note": "injuries"}).to_string(index=False) if len(out) else "no games")
    picks = out[out["tier"].notna()] if len(out) else out
    print(f"\nGAME-LINE PICKS: {len(picks)}" + ("" if len(picks) else " (none clears a tier tonight: say so; they are still in the pool for parlay / boost legs and leans)"))
    for _, p in picks.iterrows():
        print(f"  {p['label']} ({int(p['odds']):+d} at {BOOK_NAME.get(book, book)}) | win {p['blend_prob']:.1%} | edge {p['edge']:+.1%} | {p['tier']} | {p['units']:g}u | take-if {int(p['min_odds']):+d}")
    out.to_csv(a.out, index=False)
    json.dump(log, open(a.log_out, "w"))
    if getattr(a, "all", None) and len(out):
        cols = list(allp.columns) if len(allp) else [c for c in out.columns if c not in ("label", "verdict")]
        merged = pd.concat([allp, out.reindex(columns=cols)], ignore_index=True) if len(allp) else out.reindex(columns=cols)
        merged.to_csv(a.all, index=False)
        bp = getattr(a, "bets", None)
        if bp and os.path.exists(bp):
            b0 = pd.read_csv(bp)
            b0 = b0[~b0["market"].isin(GAME_MK)] if "market" in b0 else b0
            add = picks.reindex(columns=list(b0.columns) if len(b0.columns) else cols)
            pd.concat([b0, add], ignore_index=True).to_csv(bp, index=False)
        print(f"MERGED: {len(out)} game-line rows -> {a.all} ({len(picks)} picks -> {bp or 'no bets file'}). Run parlay / explain / menu AFTER this.")
    print(f"saved -> {a.out} | ratings log rows -> {a.log_out} (append to 'Games NBA'; grade with gamesgrade)")

GAMES_COLS = ["date", "away", "home", "model_margin", "model_total", "home_spread", "total", "injuries", "act_margin", "act_total"]

def cmd_gamesgrade(a):
    """Fill actual margin/total into the 'Games NBA' rows and score model vs market so far (item 11: Sun 10/25 the
    ratings' totals beat the market's 10 vs 15 pts on 5 games - only a real sample can say if that holds)."""
    import json
    vals = _values(a.json)
    hdr, body = vals[0], vals[1:]
    d = pd.DataFrame([list(r) + [""] * (len(hdr) - len(r)) for r in body], columns=hdr, dtype=object)
    sched_all = []
    for s_ in sorted({season_for(pd.Timestamp(x)) for x in d["date"] if x}):
        p = _download(URLS["sched"].format(s=s_), os.path.join(CACHE, f"sched_{s_}.parquet"), 6)
        sc = pd.read_parquet(p)
        sched_all.append(pd.DataFrame({"date": pd.to_datetime(sc["game_date"]).dt.normalize().astype(str).str[:10],
                                       "home": sc["home_abbreviation"].map(norm_team), "away": sc["away_abbreviation"].map(norm_team),
                                       "hs": pd.to_numeric(sc["home_score"], errors="coerce"), "as": pd.to_numeric(sc["away_score"], errors="coerce"),
                                       "done": sc["status_type_completed"].astype(bool)}))
    S = pd.concat(sched_all)
    S = S[S["done"]].set_index(["date", "away", "home"])
    for i, r in d.iterrows():
        k = (str(r["date"])[:10], r["away"], r["home"])
        if k in S.index and str(r.get("act_margin", "")) in ("", "nan"):
            x = S.loc[k]
            x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
            d.at[i, "act_margin"], d.at[i, "act_total"] = float(x["hs"] - x["as"]), float(x["hs"] + x["as"])
    for c in ("model_margin", "model_total", "home_spread", "total", "act_margin", "act_total"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.dropna(subset=["act_margin"])
    json.dump([GAMES_COLS] + [[_cell(v) for v in r] for r in d.reindex(columns=GAMES_COLS).itertuples(index=False, name=None)],
              open(a.out, "w"))
    if len(g):
        mm = (g["act_margin"] + g["home_spread"]).abs().mean()  # market margin = -home_spread
        print(f"GAMES graded {len(g)} | margin MAE: market {(g['act_margin'] + g['home_spread']).abs().mean():.2f} vs ratings "
              f"{(g['act_margin'] - g['model_margin']).abs().mean():.2f} | total MAE: market {(g['act_total'] - g['total']).abs().mean():.2f}"
              f" vs ratings {(g['act_total'] - g['model_total']).abs().mean():.2f}")
        print("  promote game lines only if the ratings beat the market's numbers over 100+ games AND game-line bets beat the close (CLV)")
    print(f"full tab -> {a.out} (update_values at 'Games NBA'!A1)")

# ---------- same-game parlay checker (RATE-ONLY) ----------
# joint = product of each leg's VPE probability x LIFT, where LIFT = P(all hit) / product P(each) in 20,000 simulated
# games: margin and total, blowout minutes, a player's own pts/reb/ast/3PM and teammates/opponents move together the
# way they did in 2025-26 box scores (SGP_CORR). Validated lifts vs data: pts+PRA 1.69 (data 1.82), pts+reb 1.16 (1.21).
SGP_N = 20000

def _sgp_cov(signs):
    W, TM, OP = np.array(SGP_CORR["within"]), np.array(SGP_CORR["teammate"]), np.array(SGP_CORR["opponent"])
    k = len(signs); C = np.zeros((4 * k, 4 * k))
    for i in range(k):
        for j in range(k):
            C[4 * i:4 * i + 4, 4 * j:4 * j + 4] = W if i == j else (TM if signs[i] == signs[j] else OP)
    w, v = np.linalg.eigh((C + C.T) / 2)
    C = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(C)); return C / np.outer(d, d)

def _sgp_ppf(u, mu, var, continuous):
    mu = max(mu, 0.01)
    if continuous:
        return st.norm.ppf(u, mu, math.sqrt(max(var, 1e-6)))
    if var <= mu * 1.02:
        return st.poisson.ppf(u, mu)
    n = mu * mu / (var - mu)
    return st.nbinom.ppf(u, n, n / (n + mu))

def sgp_simulate(legs, proj, home, cm, ct, seed=7):
    rng = np.random.default_rng(seed)
    zM, zT = rng.standard_normal(SGP_N), rng.standard_normal(SGP_N)
    M, T = cm + GAME["sd_margin"] * zM, ct + GAME["sd_total"] * zT
    bins = np.array(SGP_CORR["bins"])
    bidx = np.clip(np.searchsorted(bins, np.abs(M), side="right") - 1, 0, len(bins) - 2)
    pids = list(dict.fromkeys(l["pid"] for l in legs if l["kind"] == "player"))
    sign = [1 if proj.loc[p, "team"] == home else -1 for p in pids]
    vals = {}
    if pids:
        E = rng.multivariate_normal(np.zeros(4 * len(pids)), _sgp_cov(sign), size=SGP_N, method="cholesky")
        for i, p in enumerate(pids):
            row = proj.loc[p]
            role = "starter" if float(row.get("start_rate", 0) or 0) >= 0.5 else "bench"
            for j, c in enumerate(SGP_CORR["stats"]):
                prm = SGP_CORR[role][c]
                L = prm["bT"] * zT + prm["bM"] * sign[i] * zM + np.array(prm["shift"])[bidx] + prm["res_sd"] * E[:, 4 * i + j]
                u = (st.rankdata(L) - 0.5) / SGP_N
                vals[(p, c)] = _sgp_ppf(u, float(row["mu_" + c]), float(row["var_" + c]), c in CONT)
    hits = []
    for l in legs:
        if l["kind"] == "player":
            if any((l["pid"], c) not in vals for c in MARKETS[l["market"]]):
                raise SystemExit(f"{l['label']}: market not simulated (pts/reb/ast/3PM and their combos only)")
            x = sum(vals[(l["pid"], c)] for c in MARKETS[l["market"]])
            thr = math.floor(l["line"]) + 0.5
            hits.append(x > thr if l["side"] == "Over" else x < thr)
        elif l["kind"] == "spread":
            hits.append((M if l["team"] == home else -M) + l["line"] > 0)
        elif l["kind"] == "ml":
            hits.append((M > 0) if l["team"] == home else (M < 0))
        else:
            hits.append(T > l["line"] if l["side"] == "Over" else T < l["line"])
    return np.vstack([np.asarray(h) for h in hits])

def _parse_sgp_leg(s_):
    t = [x.strip() for x in s_.split(";")]
    k = t[1].lower() if len(t) > 1 else ""
    if k == "total":
        tm = norm_team(t[0]) if t[0] else None
        return {"kind": "total", "side": t[2].title(), "line": float(t[3]), "label": f"Game {t[2].title()} {t[3]}",
                "team": tm if tm in NBA_TEAMS else None}
    if k == "spread":
        return {"kind": "spread", "team": norm_team(t[0]), "line": float(t[2]), "label": f"{t[0]} {float(t[2]):+g}"}
    if k == "ml":
        return {"kind": "ml", "team": norm_team(t[0]), "label": f"{t[0]} ML"}
    return {"kind": "player", "name": t[0], "market": canon_market(t[1]), "side": t[2].title(), "line": float(t[3]),
            "odds": t[4] if len(t) > 4 and t[4] else None, "label": f"{t[0]} {t[2].title()} {t[3]} {t[1]}"}

def one_sided_fair(odds):
    """No-vig chance for a price shown on ONE side only ('4+ assists +250'): implied / (1 + hold). The hold on these
    markets is measured by the `vig` command; until it has data, the table in CONFIG is used."""
    h = CONFIG["one_sided_hold"]
    o = float(odds)
    hold = h["long"] if o >= h["long_from"] else (h["short"] if o <= h["short_to"] else h["mid"])
    return min(implied_prob(o) / (1 + hold), 0.97)

def cmd_sgpbuild(a):
    """v2.23: builder for SAME-GAME-only boosts ("fd50:sgp:3leg:+100"). Per game: the best legs from tonight's boost pool
    (edge >= -3%, win >= 50%, not HELD), one simulation of all of them together, then every combo of N legs from different
    players is rated: joint chance = learned-mix leg chances x the simulated correlation lift. The price is ESTIMATED as the
    legs multiplied (books cut SGP prices for correlation) -> always confirm with the app's real SGP price via `sgp`."""
    import itertools as _it
    set_calib(getattr(a, "calib", None), SPORT)
    cfg = _cfg(a)
    specs = [b for b in parse_boosts(a.boosts) if b.get("sgp")] or \
        [{"book": a.boost_book, "pct": float(a.boost or 0), "label": f"{a.boost_book} {a.boost or 0:g}% sgp", "min_legs": a.legs}]
    allp = pd.read_csv(a.all)
    lock_from(allp)
    pool = parlay_pool(allp, [x for x in (a.exclude or "").split(";") if x.strip()], min_edge=-0.03, min_prob=0.5)
    if "market_gap" in pool:
        pool = pool[pool["market_gap"].fillna(0) <= cfg["model"]["max_market_gap"]]
    if pool.empty:
        print("SGPBUILD: no legs in the boost pool tonight (edge >= -3%, win >= 50%, not held)")
        return
    date, state, sched = _setup(a, cfg)
    inj = parse_injuries(a.injuries)
    games = guard_games(load_games_file(a.games), sched, date)
    players = list(dict.fromkeys(pool["player"]))
    proj, _, outs = _prepare(state, sched, date, [(p, None) for p in players], inj, games, cfg)
    inj_sh = injury_shift(state, outs)
    ratings = team_ratings(state, date)
    for spec in specs:
        bk, pct = spec["book"], spec["pct"]
        kmin = int(spec.get("min_legs") or a.legs or 3); kmax = int(spec.get("max_legs") or max(kmin, a.max_legs or kmin))
        mino = spec.get("min_odds", a.min_odds)
        print(f"\n=== SGP BOOST {spec['label']}: {BOOK_NAME.get(bk, bk)} {pct:g}% | {kmin}-{kmax} legs, same game"
              + (f", total odds >= {int(mino):+d}" if mino is not None else "") + " ===")
        found = []
        for gk, g in pool.groupby(pool.apply(lambda r: frozenset((r["team"], r["opp"])), axis=1)):
            g = g.assign(_mv=g["market_prob"] * g["dec_" + bk]).sort_values(["_mv", "leg_prob"], ascending=False)
            g = g.drop_duplicates("player").head(a.per_game)
            legs = []
            for _, r in g.iterrows():
                pid = state.find(r["player"])
                if pid is None or pid not in proj.index or r["market"] not in MARKETS:
                    continue
                legs.append({"kind": "player", "name": r["player"], "market": r["market"], "side": r["side"],
                             "line": float(r["line"]), "pid": pid, "label": f"{r['player']} {r['side']} {r['line']:g} {r['market']}",
                             "p": float(r["leg_prob"]), "pm": float(r["market_prob"]), "dec": float(r["dec_" + bk])})
            if len(legs) < kmin:
                continue
            r0 = proj.loc[legs[0]["pid"]]; home, away = (r0["team"], r0["opp"]) if r0["home"] else (r0["opp"], r0["team"])
            gm = games[(games["home"] == home) & (games["away"] == away)] if games is not None else pd.DataFrame()
            hs = float(gm.iloc[0]["home_spread"]) if len(gm) and pd.notna(gm.iloc[0]["home_spread"]) else None
            tot = float(gm.iloc[0]["total"]) if len(gm) and pd.notna(gm.iloc[0]["total"]) else None
            cm, ct, *_ = game_center(ratings, home, away, hs, tot, inj_sh)
            try:
                H = sgp_simulate(legs, proj, home, cm, ct)
            except SystemExit:
                continue
            for k in range(kmin, min(kmax, len(legs)) + 1):
                for ix in _it.combinations(range(len(legs)), k):
                    Hs = H[list(ix)]
                    ps = Hs.mean(axis=1)
                    lift = Hs.all(axis=0).mean() / max(float(np.prod(ps)), 1e-9)
                    pv = [legs[i]["p"] for i in ix]; pm = [legs[i]["pm"] for i in ix]
                    joint = min(float(np.prod(pv)) * lift, min(pv)); jm = min(float(np.prod(pm)) * lift, min(pm))
                    dec = float(np.prod([legs[i]["dec"] for i in ix]))
                    if mino is not None and decimal_to_american(dec) < mino:
                        continue
                    de = american_to_decimal(boosted_american(decimal_to_american(dec), pct))
                    found.append({"game": f"{away}@{home}", "legs": [legs[i] for i in ix], "joint": joint, "jm": jm,
                                  "lift": lift, "dec": dec, "dec_eff": de, "ev": joint * de - 1, "evm": jm * de - 1})
        if not found:
            print("  no same-game combo qualifies (needs enough pool legs in one game"
                  + (f" and est. odds >= {int(mino):+d}" if mino is not None else "") + ")")
            continue
        for f_ in found:   # v2.25: Under cap (45% of legs): combos within the cap rank first
            f_["uh"] = sum(l["side"] == "Under" for l in f_["legs"]) > _under_max(len(f_["legs"]))
        found.sort(key=lambda x: (not x["uh"], min(x["ev"], x["evm"]), x["ev"]), reverse=True)
        seen = set()
        shown = 0
        for f_ in found:
            if f_["game"] in seen and shown >= 1 and not a.all_games:
                continue
            seen.add(f_["game"]); shown += 1
            tag = "PLACE (confirm price with `sgp`)" if min(f_["ev"], f_["evm"]) > 0 else "LEAN (no edge even boosted)"
            if f_["uh"]:
                tag += " | OVER THE UNDER CAP (45%): only if nothing within the cap is offered"
            print(f"  #{shown} {f_['game']} | est. {decimal_to_american(f_['dec']):+d} -> {decimal_to_american(f_['dec_eff']):+d} boosted"
                  f" | hit {f_['joint']:.1%} (market {f_['jm']:.1%}, lift {f_['lift']:.2f}) | EV {f_['ev']:+.0%}"
                  f" (market-only {f_['evm']:+.0%}) | {tag}")
            for l in f_["legs"]:
                print(f"      {l['label']} ({decimal_to_american(l['dec']):+d}) VPE {l['p']:.1%}")
            fair = decimal_to_american(1 / f_["joint"])
            # v2.24: TAKE-IF = the app price where `sgp` says TAKE (EV >= its +10% bar after the boost), not break-even
            # (11/16: take-if +425 printed, DK gave +440 and `sgp` said +2% -> SKIP). joint x (1 + (d-1)(1+b)) = 1 + min_ev
            # `sgp` reprices each leg with its own learned mix, so re-check there: this is the floor, not a guarantee.
            be = 1 + ((1 + a.min_ev) / f_["joint"] - 1) / (1 + pct / 100)
            f_["take_if"] = decimal_to_american(be)
            print(f"      fair {fair:+d} | TAKE-IF: app SGP price before the {pct:g}% boost >= {decimal_to_american(be):+d}"
                  f" (where `sgp` says TAKE: EV >= {a.min_ev:+.0%})"
                  f" | check: sgp --price <app price> " + " ".join(f'--leg "{l["name"]};{l["market"]};{l["side"]};{l["line"]:g}"' for l in f_["legs"]))
            if shown >= a.top:
                break
        print("  note: est. price = legs multiplied; the app's SGP price is usually LOWER (correlation cut). One per game shown"
              " (--all-games for more). Rate-only until real SGP results prove the correlation model.")


def cmd_sgp(a):
    """Same-game parlay checker (rate only). v2.23: legs priced with the learned model/market mix (same as `reprice`),
    a leg whose model vs market gap is over max_market_gap is HELD (-> SKIP), and legs from several games (SGPx) are
    simulated game by game and the per-game chances multiplied (games are independent of each other)."""
    set_calib(getattr(a, "calib", None), SPORT)
    cfg = _cfg(a)
    date, state, sched = _setup(a, cfg)
    inj = parse_injuries(a.injuries)
    games = guard_games(load_games_file(a.games), sched, date)
    legs = [_parse_sgp_leg(x) for x in a.leg]
    players = [(l["name"], None) for l in legs if l["kind"] == "player"]
    proj, _, outs = _prepare(state, sched, date, players, inj, games, cfg)
    for l in legs:
        if l["kind"] == "player":
            l["pid"] = state.find(l["name"])
            if l["pid"] is None or l["pid"] not in proj.index:
                sys.exit(f"no projection for {l['name']} (not playing today / name mismatch)")
            r = proj.loc[l["pid"]]
            l["game"] = frozenset((r["team"], r["opp"]))
        elif l.get("team"):
            g0 = games[(games["home"] == l["team"]) | (games["away"] == l["team"])] if games is not None else pd.DataFrame()
            if not len(g0):
                sys.exit(f"{l['label']}: {l['team']} has no game today")
            l["game"] = frozenset((g0.iloc[0]["home"], g0.iloc[0]["away"]))
    gks = list(dict.fromkeys(l["game"] for l in legs if l.get("game")))
    for l in legs:
        if not l.get("game"):
            if len(gks) != 1:
                sys.exit(f"{l['label']}: say which game, e.g. \"BOS;total;Over;225.5\" (legs come from {len(gks)} games)")
            l["game"] = gks[0]
    if not gks:
        sys.exit("no game found for these legs")
    gap_max = cfg["model"]["max_market_gap"]
    probs, joint, lifts, held, names = [], 1.0, [], [], []
    inj_sh = injury_shift(state, outs)
    for gk in gks:
        gl = [l for l in legs if l["game"] == gk]
        gp = [l for l in gl if l["kind"] == "player"]
        if gp:
            r0 = proj.loc[gp[0]["pid"]]; home, away = (r0["team"], r0["opp"]) if r0["home"] else (r0["opp"], r0["team"])
        else:
            g0 = games[games["home"].isin(gk) & games["away"].isin(gk)].iloc[0]; home, away = g0["home"], g0["away"]
        gm = games[(games["home"] == home) & (games["away"] == away)] if games is not None else pd.DataFrame()
        hs = float(gm.iloc[0]["home_spread"]) if len(gm) and pd.notna(gm.iloc[0]["home_spread"]) else None
        tot = float(gm.iloc[0]["total"]) if len(gm) and pd.notna(gm.iloc[0]["total"]) else None
        cm, ct, *_ = game_center(team_ratings(state, date), home, away, hs, tot, inj_sh)
        H = sgp_simulate(gl, proj, home, cm, ct)
        p_sim = H.mean(axis=1); lift = H.all(axis=0).mean() / max(np.prod(p_sim), 1e-9)
        gpr = []
        for l, ps in zip(gl, p_sim):
            if l["kind"] == "player":
                r = proj.loc[l["pid"]]
                po = cal_prob(prob_over(float(r["mu_" + l["market"]]), float(r["var_" + l["market"]]), l["market"], l["line"]))
                p = po if l["side"] == "Over" else 1 - po
                l["model_p"] = p
                if l.get("odds"):
                    o = l["odds"]
                    mp = no_vig(*[float(x.replace("+", "")) for x in o.split("/")])[0] if "/" in o else one_sided_fair(o.replace("+", ""))
                    l["mkt_p"], l["w"] = mp, _mw(l["market"])
                    p = blend_leg(p, mp, l["market"], l["side"])
                    if abs(l["model_p"] - mp) > gap_max:
                        held.append(f"{l['label']} (model {l['model_p']:.1%} vs market {mp:.1%}, gap {abs(l['model_p'] - mp):.0%})")
            else:
                p = float(ps)
            l["p"] = p
            gpr.append(p)
        jg = min(float(np.prod(gpr)) * lift, min(gpr))
        joint *= jg; probs += gpr; lifts.append((f"{away}@{home}", lift, jg, cm, ct)); names.append(f"{away}@{home}")
    lift = joint / max(float(np.prod(probs)), 1e-9)
    price = int(float(str(a.price).replace("+", ""))); dec = american_to_decimal(price)
    ev = joint * dec - 1
    verdict = "SKIP" if held else ("TAKE" if ev >= a.min_ev else "SKIP")
    kind = "SGP" if len(gks) == 1 else f"SGPx ({len(gks)} games)"
    print(f"\n{kind} {' + '.join(names)} - RATE ONLY | {SGP_N:,} simulated games per game")
    for gname, lf, jg, cm, ct in lifts:
        print(f"  {gname}: center home margin {cm:+.1f}, total {ct:.1f} | lift {lf:.2f} | this game's legs together {jg:.2%}")
    for l in legs:
        extra = (f" (model {l['model_p']:.1%}, market {l['mkt_p']:.1%}, learned mix {l['w']:.0%} model)"
                 if l.get("mkt_p") is not None else (" (model only: no price given for this leg)" if l["kind"] == "player" else ""))
        print(f"  {l['label']:42s} VPE {l['p']:5.1%}{extra}")
    print(f"  independent product {np.prod(probs):.2%} x correlation lift {lift:.2f} = joint {joint:.2%}")
    print(f"  book price {price:+d} (implies {1 / dec:.2%}) | fair price {decimal_to_american(1 / joint):+d} | EV {ev:+.0%}")
    if held:
        print(f"  HELD leg(s): model vs market gap over {gap_max:.0%} (usually news the model hasn't seen; same rule as the "
              "main slate): " + " | ".join(held))
    print(f"  VERDICT: {verdict} (" + ("a leg is HELD" if held else f"needs EV >= {a.min_ev:+.0%}")
          + "; rate-only - correlations from last season, not yet proven against real SGP results)")
    home, away = names[0].split("@")[1], names[0].split("@")[0]
    # v2.21: save the rating so `log` records it on the SGP row (11/16: it had to be typed into the note by hand)
    import json as _j
    rp = getattr(a, "ratings_out", None) or "sgp_ratings.json"
    saved = _j.load(open(rp)) if os.path.exists(rp) else []
    saved.append({"legs": [[norm_name(l["name"]), l["market"], l["side"], l["line"]] for l in legs if l["kind"] == "player"],
                  "other_legs": [l["label"] for l in legs if l["kind"] != "player"], "game": " + ".join(names),
                  "joint": round(joint, 4), "fair": int(decimal_to_american(1 / joint)), "price": price, "ev": round(ev, 3),
                  "min_ev": a.min_ev, "verdict": verdict, "held": held, "lift": round(float(lift), 3)})
    _j.dump(saved, open(rp, "w"))
    print(f"  saved to {rp}: `log` will record this rating on the SGP if it is placed")
    if lift > 1.05:
        print("  why: legs move together (shared minutes / pace / same-player stats) - the book may or may not price that in")
    elif lift < 0.95:
        print("  why: legs work AGAINST each other (e.g. blowout script vs starter Overs) - worse than it looks")

# ---------- MENU (v2.5): one view of the night's best options ----------
# 1) best straight bets (the staked slip)  2) best cross-game parlay (the ONE-PARLAY RULE pick)
# 3) per game: best props + best same-game "story" parlay (legs that move together, joint chance from the 20k-game sim)
MENU = {"top_overall": 10, "min_edge": 0.015, "props_per_game": 3, "sgp_cands": 6, "sgp_min_dec": 3.0, "sgp_max_legs": 4, "sgp_min_ev": 0.10}


def _story(legs, home):
    ov = [l for l in legs if l["side"] == "Over"]
    teams = {l["team"] for l in legs}
    if len(ov) == len(legs) and len(teams) == 2:
        return "SHOOTOUT: both teams score, every Over hits together"
    if len(ov) == len(legs):
        return "TEAM ROLLS: one team's attack carries all the Overs"
    if not ov:
        return "GRIND / BLOWOUT: low-volume game, Unders together"
    return "MIXED SCRIPT: one side's usage rises as the other's falls"


def cmd_menu(a):
    import json
    set_calib(getattr(a, "calib", None), SPORT)
    cfg = _cfg(a)
    allp = pd.read_csv(a.all)
    lock_from(allp)
    pool = parlay_pool(allp)
    date, state, sched = _setup(a, cfg)
    print("\n" + "=" * 70 + "\nVPE MENU - best options tonight (ranked inside each section)\n" + "=" * 70)
    # 0. overall ranking: every scored prop on the slate with a real edge at its price (not held, not out), best first
    print("\n0) BEST PROPS OVERALL (every game, strongest prop first: win chance, among props with an edge; ★ = on the staked slip)"
          + (f" | prices locked: your {LOCK['books']} recording" if LOCK["on"] else ""))
    on_slip = set()
    if a.bets and os.path.exists(a.bets):
        bb = pd.read_csv(a.bets)
        on_slip = {(r["player"], r["market"], r["side"]) for _, r in bb.iterrows()}
    ov = allp.copy()
    for c in ("edge", "blend_prob", "odds", "market_gap"):
        ov[c] = pd.to_numeric(ov.get(c), errors="coerce")
    ok = (ov["edge"] >= MENU["min_edge"]) & ov["odds"].between(-200, 200) & (ov["market_gap"].fillna(0) <= CONFIG["model"]["max_market_gap"])
    if "base_min" in ov:
        ok &= pd.to_numeric(ov["base_min"], errors="coerce").fillna(99) >= 20
    if "injury" in ov:
        ok &= ~ov["injury"].fillna("").astype(str).str.contains("Out|Doubtful", case=False)
    ov = ov[ok].sort_values(["blend_prob", "edge"], ascending=False).head(MENU["top_overall"])
    if ov.empty:
        print("   none qualify")
    for i, (_, r) in enumerate(ov.iterrows(), 1):
        star = "★" if (r["player"], r["market"], r["side"]) in on_slip else " "
        tk = f" | take if odds >= {int(r['min_odds']):+d}" if pd.notna(r.get("min_odds")) else ""
        print(f"  {i:>2}. {star} {r['player']} ({r['team']} vs {r['opp']}) {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}, {r['book']})"
              f" | win {r['blend_prob']:.1%} | edge {r['edge']:+.1%}{tk}{_rr(r)}")
    # 1. straights
    print("\n1) BEST STRAIGHT BETS (the staked slip; max 2 per game)")
    if a.bets and os.path.exists(a.bets):
        b = pd.read_csv(a.bets)
        b = b[b.get("tier").notna()] if "tier" in b else b
        if len(b):
            for _, r in b.sort_values(["blend_prob", "edge"], ascending=False).iterrows():
                print(f"   {r['player']} ({r['team']}) {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}) vs {r['opp']}"
                      f" | win {r['blend_prob']:.1%} | edge {r['edge']:+.1%} | {r['units']}u{_rr(r)}")
        else:
            print("   none beat the book tonight")
            lf = "leans.csv"
            if os.path.exists(lf) and len(pd.read_csv(lf)):
                print("   LEAN - best available (no edge; if you play, play these):")
                for _, r in pd.read_csv(lf).iterrows():
                    print(f"   ~ {r['player']} ({r['team']}) {r['side']} {r['line']} {r['market']} ({int(r['bet_odds']):+d} at {r['bet_at']})"
                          f" vs {r['opp']} | win {r['leg_prob']:.1%} | edge {r['leg_edge']:+.1%} | {r['units']:g}u{_rr(r)}")
    else:
        print("   (run check first)")
    # 2. cross-game parlay
    print("\n2) BEST CROSS-GAME PARLAY (ONE-PARLAY RULE)")
    pp = a.parlays if a.parlays and os.path.exists(a.parlays) else None
    rec = next((x for x in json.load(open(pp)) if x.get("recommended")), None) if pp else None
    if rec:
        print(f"   {rec['name']} at {BOOK_NAME[rec['book']]} {rec['odds']:+d} | hit {rec['hit']:.1%} | ${rec['stake_usd']:.2f}"
              + (" | LEAN: no edge, least-bad play" if rec.get("lean") else ""))
        for l in rec["legs"]:
            print(f"     - {l[0]} {l[2]} {l[3]} {l[1]}")
    else:
        print("   none tonight (or run parlay first)")
    # 3. per game
    print("\n3) BY GAME: best props + best same-game story parlay (SGP, rate-only until 30 are graded)")
    if pool.empty:
        print("   no legs qualify tonight")
        return
    inj = parse_injuries(a.injuries)
    games = guard_games(load_games_file(a.games), sched, date) if a.games else None
    sim_ok = pool[pool["market"].isin([m for m in MARKETS if all(c in SGP_CORR["stats"] for c in MARKETS[m])])]
    players = [(n, None) for n in sim_ok["player"].unique()]
    proj, _, outs = _prepare(state, sched, date, players, inj, games, cfg) if players else (pd.DataFrame(), {}, {})
    ratings, shift = team_ratings(state, date), injury_shift(state, outs)
    for gid, g in pool.groupby("game_id", sort=False):
        r0 = g.iloc[0]
        home, away = (r0["team"], r0["opp"]) if str(r0["home"]).lower() in ("true", "1") else (r0["opp"], r0["team"])
        print(f"\n  {away} @ {home}")
        top = g.sort_values(["leg_prob", "leg_edge"], ascending=False).head(MENU["props_per_game"])
        for _, r in top.iterrows():
            print(f"    prop: {r['player']} {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}, {r['bet_at']}) | win {r['leg_prob']:.1%}"
                  f" | edge {r['leg_edge']:+.1%}" + f" | take if odds >= {int(r['take_min']):+d}" + _rr(r))
        c = g[g["player"].isin(sim_ok["player"])].head(MENU["sgp_cands"]).to_dict("records")
        legs = []
        for x in c:
            pid = state.find(x["player"])
            if pid is None or pid not in proj.index:
                continue
            legs.append({"kind": "player", "pid": pid, "market": x["market"], "side": x["side"], "line": float(x["line"]),
                         "p": float(x["leg_prob"]), "team": x["team"], "label": f"{x['player']} {x['side']} {x['line']} {x['market']}"})
        if len(legs) < 2:
            print("    SGP: not enough qualifying legs in this game")
            continue
        gm = games[(games["home"] == home) & (games["away"] == away)] if games is not None else pd.DataFrame()
        hs = float(gm.iloc[0]["home_spread"]) if len(gm) and pd.notna(gm.iloc[0]["home_spread"]) else None
        tot = float(gm.iloc[0]["total"]) if len(gm) and pd.notna(gm.iloc[0]["total"]) else None
        cm, ct, *_ = game_center(ratings, home, away, hs, tot, shift)
        H = sgp_simulate(legs, proj, home, cm, ct)
        ps = H.mean(axis=1)
        best = None
        for n in range(2, min(MENU["sgp_max_legs"], len(legs)) + 1):
            for idx in itertools.combinations(range(len(legs)), n):
                lift = H[list(idx)].all(axis=0).mean() / max(float(np.prod(ps[list(idx)])), 1e-9)
                pr = [legs[i]["p"] for i in idx]
                joint = min(float(np.prod(pr)) * lift, min(pr))
                if joint <= 0 or 1 / joint < MENU["sgp_min_dec"]:
                    continue
                if best is None or joint > best[0]:
                    best = (joint, lift, idx)
        if best is None:
            print(f"    SGP: no combo pays {MENU['sgp_min_dec']:.0f}x+ from tonight's legs")
            continue
        joint, lift, idx = best
        sl = [legs[i] for i in idx]
        fair = decimal_to_american(1 / joint)
        need = decimal_to_american((1 + MENU["sgp_min_ev"]) / joint)
        print(f"    SGP ({len(sl)} legs) - {_story(sl, home)}")
        for l in sl:
            print(f"      - {l['label']} (win {l['p']:.1%})")
        print(f"      joint {joint:.1%} (legs move together: x{lift:.2f}) | fair {fair:+d} | "
              f"TAKE only if the app's SGP price is >= {need:+d} (EV +{MENU['sgp_min_ev']:.0%}); a boost counts")


# ---------- CLV capture (v2.8): closing price for every bet actually placed ----------
# Before: `final` only checked VPE's slip and wrote a closing price only when the closing line was EXACTLY the line
# bet, so placed bets (menu props, boost legs, own picks) almost never got one -> CLV was blank on every row.
# Now every unsettled straight bet and parlay leg of the night gets the no-vig closing chance of ITS side at ITS line
# (read off the player's closing main O/U line, same anchoring as alt lines), stored as a fair American price.
# CLV = closing fair chance - break-even of the price taken (> 0 = beat the close).
def fill_closing(log, close_props, date, cfg):
    if close_props is None or close_props.empty:
        return log, 0
    cp = close_props.copy()
    cp["_key"] = cp["player"].map(norm_name)
    mains = {}
    for _, r in cp.iterrows():
        if r["market"] is not None and pd.notna(r["over_odds"]) and pd.notna(r["under_odds"]):
            mains.setdefault((r["_key"], r["market"]), []).append((float(r["line"]), no_vig(r["over_odds"], r["under_odds"])[0]))
    for c in ("closing_odds", "line", "odds"):
        log[c] = pd.to_numeric(log.get(c), errors="coerce")
    bt = log["bet_type"].fillna("straight").astype(str)
    m = (log["date"].astype(str).str.lstrip("'").str[:10] == str(date)) & log["closing_odds"].isna() & bt.isin(["straight", "leg"]) & log["market"].isin(list(MARKETS))
    n = 0
    for i in log[m].index:
        r = log.loc[i]
        main = mains.get((norm_name(r["player"]), r["market"]))
        if not main or pd.isna(r["line"]):
            continue
        po = anchored_market_over(r["market"], float(r["line"]), main, cfg)
        if po is None:
            continue
        p = po if str(r["side"]).title() == "Over" else 1 - po
        p = min(max(p, 0.02), 0.98)
        log.loc[i, "closing_odds"] = decimal_to_american(1 / p)
        n += 1
    return log, n


def clv_report(log, date=None):
    d = log.copy()
    for c in ("closing_odds", "odds"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d[d["closing_odds"].notna() & d["odds"].notna()]
    if date is not None:
        d = d[d["date"].astype(str).str.lstrip("'").str[:10] == str(date)]
    if d.empty:
        print("CLV: no bets with a closing price yet")
        return
    clv = d["closing_odds"].map(lambda x: 1 / american_to_decimal(x)) - d["odds"].map(implied_prob)
    print(f"CLV{' ' + str(date) if date else ''}: {len(d)} bets/legs | beat the close {(clv > 0).mean():.0%} | avg {clv.mean():+.1%}"
          f" (closing chance minus the break-even of the price taken)")
    for (_, r), c in sorted(zip(d.iterrows(), clv), key=lambda x: -x[1]):
        print(f"   {r['player']} {r['side']} {r['line']} {r['market']} @ {int(r['odds']):+d} -> fair close {int(r['closing_odds']):+d}  CLV {c:+.1%}")


def cmd_close(a):
    """Closing prices for tonight's PLACED bets: props_close.csv (main O/U lines, both sides) + the pick log -> closing_odds."""
    cfg = _cfg(a)
    date = str(pd.Timestamp(a.date).date()) if a.date else str(today_et().date())
    log = read_log(a.log)
    close = _read_props(a.props) if a.props else pd.DataFrame()
    if "market" in close:
        close = close[close["market"].notna()]
    snaps = snap_rows(close, date) if len(close) else []
    if getattr(a, "snap_out", None):
        import json as _j
        _j.dump(snaps, open(a.snap_out, "w"))
        print(f"SNAPSHOT: {len(snaps)} closing lines for {date} -> {a.snap_out} (append to the 'Close NBA' tab, every night)")
    if getattr(a, "snap", None):  # closing lines saved earlier (bets logged after the closing run)
        sp = snap_props(a.snap, slate=date)
        close = pd.concat([close, sp], ignore_index=True) if len(sp) else close
        print(f"SNAPSHOT: {len(sp)} saved closing lines for {date} read from {a.snap}")
    for c in ("closing_odds",):
        log[c] = pd.to_numeric(log.get(c), errors="coerce")
    bt_ = log["bet_type"].fillna("straight").astype(str)
    open_ = log[(log["date"].astype(str).str.lstrip("'").str[:10] == str(date)) & log["closing_odds"].isna() & bt_.isin(["straight", "leg"])]
    log, n = fill_closing(log, close, date, cfg)
    if getattr(a, "no_log", False):
        print("--no-log: pick log NOT written (recheck only)")
    else:
        log.to_csv(a.log, index=False)
    print(f"CLOSE {date}: closing price written to {n} bets/legs in {a.log}"
          + ("" if n else " (none matched: the closing file needs each bet player's MAIN O/U line with both prices)"))
    clv_report(log, date)
    miss = sorted(set(open_["market"].astype(str)) - set(close["market"].astype(str))) if len(close) else []
    note = ("no bets logged for this slate yet (the snapshot keeps the closing lines for later)" if open_.empty
            else "ok" if n == len(open_) else f"{len(open_) - n} left blank" + (f" (no closing page for: {', '.join(miss)})" if miss else
                                                                             " (player/line not on the closing pages)"))
    note = close_note(note, len(close), a)
    write_close_status(getattr(a, "status_out", None), "NBA", date, len(open_), n, len(snaps), note)


def cmd_vig(a):
    """Measure the hold on one-sided 'X+' prices (FanDuel milestones) against each player's main two-sided line:
    hold = book implied / fair (fair = main line's no-vig chance carried to the milestone through the implied mean)."""
    props = _read_props(a.props)
    cfg = _cfg(a)
    rows = []
    for (pl, mk), g in props[props["market"].notna()].groupby(["player", "market"]):
        main = [(float(r["line"]), no_vig(r["over_odds"], r["under_odds"])[0]) for _, r in g.iterrows()
                if pd.notna(r["over_odds"]) and pd.notna(r["under_odds"])]
        for _, r in g[g["under_odds"].isna() & g["over_odds"].notna()].iterrows():
            fair = anchored_market_over(mk, float(r["line"]), main, cfg)
            if fair is None or fair <= 0.01:
                continue
            rows.append({"player": pl, "market": mk, "line": r["line"], "odds": int(r["over_odds"]), "book": r.get("book", ""),
                         "fair": round(fair, 4), "implied": round(implied_prob(r["over_odds"]), 4),
                         "hold": round(implied_prob(r["over_odds"]) / fair - 1, 4)})
    d = pd.DataFrame(rows)
    if d.empty:
        print("VIG: no one-sided rows with a main line to compare (need both the main O/U and the X+ ladder)"); return
    d["band"] = pd.cut(d["odds"], [-10000, -200, 150, 100000], labels=["short (<=-200)", "mid", "long (>=+150)"])
    print(d.groupby("band", observed=True)["hold"].agg(["size", "median", "mean"]).round(3).to_string())
    d.to_csv(a.out, index=False)
    print(f"rows -> {a.out} (append to the 'Vig' tab; after ~200 rows per band, set CONFIG one_sided_hold from the medians)")

def cmd_frames(a):
    """App screen recording -> still frames to read (item 2: recordings are now the main price source).
    Extracts one frame every --every seconds, drops near-duplicates (scrolling pauses), writes frames/f_0001.png ...
    Then read the frames and transcribe them into props.csv / games.csv (formats in the skill)."""
    import subprocess, shutil, glob
    os.makedirs(a.out, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg", "--break-system-packages"], check=False)
            import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = os.path.join(a.out, "_raw")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", a.video, "-vf", f"fps=1/{a.every},scale=-2:1280",
                    os.path.join(tmp, "r_%05d.png")], check=True)
    from PIL import Image
    kept, prev = [], None
    for f in sorted(glob.glob(os.path.join(tmp, "r_*.png"))):
        im = Image.open(f).convert("L").resize((90, 160))
        arr = np.asarray(im, dtype=float)
        if prev is None or np.abs(arr - prev).mean() > a.diff:
            kept.append(f); prev = arr
    for i, f in enumerate(kept[:a.max], 1):
        shutil.copy(f, os.path.join(a.out, f"f_{i:04d}.png"))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"FRAMES: {len(kept)} distinct screens kept (of the raw frames) -> {a.out}/f_0001.png ...; "
          f"{'first ' + str(a.max) + ' written' if len(kept) > a.max else 'all written'}")

def cmd_takeif(a):
    """TAKE / SKIP for the price the app shows right now. Straight bets: VPE's min_odds (take-if). Parlay legs: the
    leg's take_min. Anything not tiered tonight = SKIP (not a VPE pick at any price shown)."""
    set_calib(getattr(a, "calib", None), SPORT)
    allp = pd.read_csv(a.all)
    lock_from(allp)
    pool = parlay_pool(allp)
    for spec in a.bet:
        t = [x.strip() for x in spec.split(";")]
        if len(t) < 5:
            print(f"{spec}: need Player;market;Side;line;odds"); continue
        pl, mk, side, line, odds = t[0], canon_market(t[1]) or t[1], t[2].title(), float(t[3]), _odds(t[4])
        k = norm_name(pl)
        h = allp[(allp["player"].map(norm_name) == k) & (allp["market"] == mk) & (allp["side"] == side)
                 & (pd.to_numeric(allp["line"], errors="coerce") == line)]
        lg = pool[(pool["player"].map(norm_name) == k) & (pool["market"] == mk) & (pool["side"] == side)
                  & (pd.to_numeric(pool["line"], errors="coerce") == line)] if len(pool) else pool
        tag = f"{pl} {side} {line:g} {mk} @ {odds:+d}"
        if h.empty:
            print(f"{tag}: SKIP - not in tonight's run (different line? rerun check with this line)"); continue
        r = h.iloc[0]
        if pd.notna(r.get("tier")) and str(r.get("tier")) not in ("", "nan"):
            mo = int(r["min_odds"]) if pd.notna(r.get("min_odds")) else None
            ok = mo is None or american_to_decimal(odds) >= american_to_decimal(mo) - 1e-9
            print(f"{tag}: {'TAKE' if ok else 'SKIP'} straight (take-if {mo:+d}, win chance {r['blend_prob']:.1%})" if mo is not None
                  else f"{tag}: TAKE straight (win chance {r['blend_prob']:.1%})")
        elif len(lg):
            tm_ = int(lg.iloc[0]["take_min"])
            ok = american_to_decimal(odds) >= american_to_decimal(tm_) - 1e-9
            print(f"{tag}: {'TAKE' if ok else 'SKIP'} as a parlay leg only (take_min {tm_:+d}, leg prob {lg.iloc[0]['leg_prob']:.1%})")
        else:
            print(f"{tag}: SKIP - not a VPE pick at any price (win chance {r['blend_prob']:.1%}, needs {implied_prob(odds):.1%} + edge)")

def cmd_status(a):
    """RUN STATUS gate: one block at the top of every slip, same checks in every chat and on every model.
    GO = all clear · PARTIAL = picks allowed, gaps named · HOLD = no picks (say why, fix, rerun)."""
    import json
    D = pd.Timestamp(a.date) if a.date else today_et()
    season = season_for(D)
    sched = load_schedule(season)
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    day = sched[sched["date"] == D] if len(sched) else sched
    left = day[day["tip"] > now] if (len(day) and not a.ignore_tip) else day
    tonight = {frozenset((h, w)) for h, w in zip(left["home"], left["away"])}
    rows, hold, partial = [], [], []

    def add(name, ok, detail, level="partial"):
        rows.append((name, ok, detail))
        if not ok:
            (hold if level == "hold" else partial).append(name)

    add("SLATE", len(tonight) > 0, f"{len(tonight)} games left on {D.date()}" + ("" if len(tonight) else " (nothing to bet)"), "hold")
    # stats: last night's games must be in the box scores
    box = load_box(season)
    last_box = box["date"].max() if len(box) else None
    prev_days = sched[(sched["date"] < D)]["date"] if len(sched) else pd.Series(dtype="datetime64[ns]")
    need = prev_days.max() if len(prev_days) else None
    fresh = need is None or (last_box is not None and last_box >= need)
    add("STATS", fresh, f"box scores through {last_box.date() if last_box is not None else 'none'}"
        + ("" if fresh else f", missing {need.date()} (file rebuilds ~7:30 AM ET)"))
    # lines
    cov, n_rows = {}, 0
    if a.props and os.path.exists(a.props):
        pr = pd.read_csv(a.props, dtype=str).fillna("")
        n_rows = len(pr)
        if "matchup" in pr:
            for mk, g in pr.groupby("market"):
                cov[canon_market(mk) or mk] = len({frozenset(parse_matchup(m)[:2]) for m in g["matchup"]} & tonight)
    games_lined = max(cov.values()) if cov else 0
    add("LINES", games_lined > 0 and games_lined >= len(tonight),
        (" | ".join(f"{k} {v}/{len(tonight)}" for k, v in sorted(cov.items())) or "no props file") + f" ({n_rows} rows)",
        "hold" if games_lined == 0 else "partial")
    # injuries
    inj_ok, inj_n = False, 0
    _ij = getattr(a, "injuries", None)  # v2.23: same flag as check (a list or a file)
    if _ij and not os.path.exists(_ij):
        txt = _ij.strip()
        inj_n = len([x for x in txt.split(";") if x.strip()])
        inj_ok = inj_n > 0 or txt.upper() == "NONE"
    elif (_ij or a.injuries_file) and os.path.exists(_ij or a.injuries_file):
        txt = open(_ij or a.injuries_file).read().strip()
        inj_n = len([x for x in txt.split(";") if x.strip()])
        inj_ok = inj_n > 0 or txt.upper() == "NONE"
    add("INJURIES", inj_ok, f"{inj_n} players listed" if inj_ok else "no injury list built (usage model and Out filter blind)", "hold")
    # spreads
    g = load_games_file(a.games) if a.games else None
    have = {frozenset((h, w)) for h, w in zip(g["home"], g["away"])} if g is not None else set()
    ns = len(have & tonight)
    add("SPREADS", ns == len(tonight) and len(tonight) > 0, f"{ns}/{len(tonight)} games (missing = no blowout adjustment)")
    # calibration loaded
    if a.calib and os.path.exists(a.calib):
        w, _, n = load_calib(a.calib, SPORT)
        add("CALIB", True, f"{n} graded lines, model weight {w:.0%}")
    else:
        add("CALIB", False, "Calib tab not loaded (prior 40% model weight used)")
    # shadow save
    exp = None
    if a.shadow and os.path.exists(a.shadow):
        exp = len(json.load(open(a.shadow)))
    if exp is None:
        add("SHADOW", False, "no shadow_new.json (check not run?)")
    elif a.v1_before is None or a.v1_after is None:
        add("SHADOW", False, f"{exp} lines NOT confirmed saved: pass --v1-before/--v1-after after appending")
    else:
        got = a.v1_after - a.v1_before
        add("SHADOW", got == exp, f"{got}/{exp} lines saved" + ("" if got == exp else (" -> append the missing chunk(s), re-check" if got < exp else " -> too many rows: a chunk was appended twice; delete the duplicate rows")))
    verdict = "HOLD" if hold else ("PARTIAL" if partial else "GO")
    print(f"RUN STATUS [{VERSION}] {D.date()}: {verdict}")
    for name, ok, detail in rows:
        print(f"  {'OK ' if ok else 'XX '} {name:9s} {detail}")
    if hold:
        print(f"  -> HOLD: no picks until fixed: {', '.join(hold)}")
    elif partial:
        print(f"  -> PARTIAL: picks allowed; name these gaps on the slip: {', '.join(partial)}")

def cmd_backtest(a):
    """Walk-forward test on a finished season: project every player-game from data before that day, compare to what
    happened. Reports projection error vs a plain season average and probability calibration at average-based lines."""
    cfg = _cfg(a)
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    season = season_for(start)
    cur, prev = load_box(season), load_box(season - 1, max_age_hours=24 * 30)
    prev2 = load_box(season - 2, max_age_hours=24 * 30)
    pc = prep_prev(prev, cfg)
    dates = sorted(d for d in cur["date"].unique() if start <= d <= end)[::a.step]
    team_dates = cur[cur["played"]].groupby("team")["date"].apply(lambda x: set(x))
    flags = {f: False for f in (a.off or "").split(",") if f}
    odds = None
    if getattr(a, "odds", None):
        odds = pd.read_csv(a.odds, dtype={"game_id": str}).dropna(subset=["total"]).drop_duplicates("game_id").set_index("game_id")
        odds["home_spread"] = pd.to_numeric(odds["home_spread"], errors="coerce")
        odds["total"] = pd.to_numeric(odds["total"], errors="coerce")
    extra_cols = ["arch", "opp_style", "team_style", "home", "env", "implied_pts", "days_off", "missed", "start_l2",
                  "min_l3", "disp_pts", "disp_reb", "disp_ast", "disp_fg3m", "disp_n", "r_pf", "ws_n", "ws_min",
                  "ws_r_pts", "ws_r_reb", "ws_r_ast", "ws_r_fg3m", "m2", "proj_min", "spread", "total", "start_rate",
                  "pz_usage", "pz_three", "pz_ftr", "pz_ast", "pz_reb", "pz_oreb", "pz_blk",
                  "dz_three", "dz_ftr", "dz_tov", "dz_oreb", "dz_rim", "dz_ast", "dz_pace"]
    recs = []
    for d in dates:
        state = State(cur, prev, d, cfg, pc, prev2_box=prev2)
        state.pv_box = prev
        today = cur[cur["date"] == d]
        played = today[today["played"]]
        sl = played.set_index("pid")[["team", "opp", "home", "game_id"]].copy()
        sl = sl[sl.index.isin(state.players.index)]
        sl["b2b"] = [(d - pd.Timedelta(days=1)) in team_dates.get(t, set()) for t in sl["team"]]
        sl["spread"], sl["total"] = np.nan, np.nan
        if odds is not None:
            o = odds.reindex(sl["game_id"].astype(str))
            hs = o["home_spread"].to_numpy()
            sl["spread"] = np.where(sl["home"].to_numpy(), hs, -hs)
            sl["total"] = o["total"].to_numpy()
        P = state.players
        if state.tg is not None:
            cand = P[(P["min"] >= 8) & (P["last_date"] >= d - pd.Timedelta(days=25)) & P["team"].isin(set(today["team"]))]
            out_ids = [p for p in cand.index if p not in set(played["pid"])]
        else:
            out_ids = []
        outs = P.loc[out_ids, ["team", "name", "pos", "min"] + ["r_" + s for s in USE_STATS]].copy()
        outs["pid"] = outs.index
        overrides = dict(zip(played["pid"], played["team"]))
        rosters = team_rosters(state, overrides, d)
        pr = project(state, sl, outs, rosters, cfg, flags)
        act = played.set_index("pid")
        cp = state.cur[state.cur["played"]] if len(state.cur) else state.cur
        if len(cp):
            naive = cp.groupby("pid")[STATS].mean()
        else:
            naive = prev[prev["played"]].groupby("pid")[STATS].mean()
        has_out = pr["usage_note"].ne("")
        for mk, comps in MARKETS.items():
            nv = naive.reindex(pr.index)[comps].sum(axis=1)
            recs.append(pd.DataFrame({"date": d, "pid": pr.index, "market": mk, "mu": pr["mu_" + mk].to_numpy(),
                                      "var": pr["var_" + mk].to_numpy(), "naive": nv.to_numpy(),
                                      "actual": act.reindex(pr.index)[comps].sum(axis=1).to_numpy(),
                                      "min": act.reindex(pr.index)["min"].to_numpy(), "base_min": pr["min"].to_numpy(),
                                      "out": has_out.to_numpy(), "new_team": pr["new_team"].to_numpy(),
                                      "af": pr["af_" + mk].to_numpy() if "af_" + mk in pr else np.nan,
                                      "started": act.reindex(pr.index)["starter"].to_numpy(),
                                      "pf": act.reindex(pr.index)["pf"].to_numpy(),
                                      **{c: pr[c].to_numpy() for c in extra_cols if c in pr}}))
    df = pd.concat(recs, ignore_index=True)
    for c in ("mu", "var", "naive", "actual", "min", "base_min"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["naive"].notna() & df["mu"].notna() & (df["base_min"] >= 15)]
    shrink = cfg["model"]["prob_calibration_shrink"]
    df["line"] = np.floor(df["naive"]) + 0.5
    df["p_raw"] = [prob_over(m, v, mk, l) for m, v, mk, l in zip(df["mu"], df["var"], df["market"], df["line"])]
    df["p"] = cal_prob(df["p_raw"].to_numpy())
    df["over"] = (df["actual"] > df["line"]).astype(float)
    print(f"BACKTEST {start.date()}..{end.date()} every {a.step} day(s), {len(dates)} dates, off={a.off or 'none'}")
    print(f"{'market':7s} {'n':>6s} {'MAE':>6s} {'naive':>6s} {'bias':>6s} {'brier':>6s} {'b_naive':>7s} {'z_sd':>5s}")
    for mk, d in df.groupby("market"):
        mae, mae_n = (d["mu"] - d["actual"]).abs().mean(), (d["naive"] - d["actual"]).abs().mean()
        brier = ((d["p"] - d["over"]) ** 2).mean()
        z = ((d["actual"] - d["mu"]) / np.sqrt(d["var"])).std()
        print(f"{mk:7s} {len(d):6d} {mae:6.3f} {mae_n:6.3f} {(d['mu'] - d['actual']).mean():+6.3f} {brier:6.4f} {0.25:7.4f} {z:5.2f}")
    d = df
    bins = pd.cut(d["p"], [0, .3, .4, .45, .5, .55, .6, .7, 1])
    cal = d.groupby(bins, observed=True).agg(n=("over", "size"), pred=("p", "mean"), actual=("over", "mean"))
    print("\ncalibration (all markets):\n" + cal.round(3).to_string())
    for lab, sub in (("teammate out", d[d["out"]]), ("new team", d[d["new_team"]])):
        if len(sub):
            print(f"{lab}: n={len(sub)} MAE {(sub['mu'] - sub['actual']).abs().mean():.3f} vs naive "
                  f"{(sub['naive'] - sub['actual']).abs().mean():.3f} | brier {((sub['p'] - sub['over']) ** 2).mean():.4f} "
                  f"| pred over {sub['p'].mean():.3f} actual {sub['over'].mean():.3f}")
    if a.save:
        df.to_parquet(a.save)
        print(f"saved -> {a.save}")


# ---------- v2.15 / v15.16: the full logic behind every prop VPE suggests (`explain`) ----------
# `check` stores each prop's whole chain in the "x" column of bets_all.csv: minutes (NBA) or base average (NFL), every
# factor, the distribution, the model's chance, the market's chance, the learned mix, the price, the tier and the stake.
# `explain` prints it as one card per suggested prop (straights, parlay legs, leans). Pure reporting: no number changes.
import json as _json
import warnings as _warnings
_warnings.simplefilter("ignore", pd.errors.PerformanceWarning)  # many small columns in project(); harmless


def _r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if math.isfinite(v) else None


def why_prob(pm_over, fair_over, mk, fair_how, raw_over):
    """The win-chance chain for the Over (the Under is its mirror), exactly as blend_leg computed it."""
    w = _mw(mk)
    mix = w * pm_over + (1 - w) * fair_over
    b = float((PARLAY.get("side_bias") or {}).get(mk, 0.0) or 0.0)
    g = int((LEARNED.get("gate") or {}).get(f"{mk}|Over", 0))
    fin = float(blend_leg(pm_over, fair_over, mk, "Over"))
    return {"raw": _r(raw_over, 4), "cal": _r(pm_over, 4), "fair": _r(fair_over, 4), "how": fair_how, "w": _r(w, 3),
            "mix": _r(mix, 4), "bias": _r(b, 4), "gate": g, "learn": _r(fin - (mix + b), 4), "fin": _r(fin, 4)}


def _pc(v):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{float(v):.1%}"


def _xm(v):
    return "-" if v is None else f"x{float(v):.2f}"


def _tier_text(r, x, cfg):
    prob, odds, side = float(r["blend_prob"]), int(r["odds"]), r["side"]
    edge = prob - implied_prob(odds)
    T = cfg["tiers"]
    rule = " | ".join(f"T{t[-1]} edge>={T[t]['min_edge']:.1%}" + (f" & win>={T[t]['min_prob']:.0%}" if T[t]["min_prob"] > 0 else "")
                      for t in ("tier_1", "tier_2", "tier_3"))
    estm = (EST_MARGIN if (x.get("p") or {}).get("how") == "1way" else 0.0) + MARKET_EXTRA.get(r.get("market"), 0.0)
    nominal = _tier(edge - estm, prob, cfg)
    tier = r.get("tier") if isinstance(r.get("tier"), str) else None
    if tier:
        return f"{tier.replace('_', ' ').upper()} ({rule})", tier
    if nominal:
        why = []
        if side == "Over" and str(r.get("blind_spot", "") or "") not in ("", "nan"):
            why.append(f"blind spot: {r['blind_spot']} -> no Overs")
        if side == "Over" and (str(r.get("limited", "")).lower() in ("true", "1") or str(r.get("returning", "")).lower() in ("true", "1")):
            why.append("minutes restriction / just back from an absence -> no Overs")
        smax = cfg["staking"].get("straight_max_odds")
        if smax and odds > smax:
            why.append(f"long shot over +{smax}")
        if nominal == "tier_3" and not cfg["strategy_profiles"][cfg["strategy_profile"]]["allow_tier3"]:
            why.append("Tier 3 off in this profile")
        return f"clears {nominal.replace('_', ' ')} on edge but blocked: {'; '.join(why) or 'stake rounds below 0.25u'}", None
    need = T["tier_3"]["min_edge"] - (edge - estm)
    return (f"no tier: edge {edge:+.1%} is {need:.1%} short of Tier 3 ({rule})"
            + (f" (needs {estm:+.1%} extra: " + " + ".join(x for x in ("market estimated from one price" if (x.get("p") or {}).get("how") == "1way" else "",
                                                                        "points market" if r.get("market") in MARKET_EXTRA else "") if x) + ")" if estm else "")), None


def _stake_text(r, cfg, tier):
    if not tier:
        return "no stake (no tier)"
    prob, odds = float(r["blend_prob"]), int(r["odds"])
    f = kelly_fraction(prob, odds)
    kf, up = cfg["staking"]["kelly_fraction"], cfg["unit_pct_of_bankroll"]
    raw = kf * f / up
    cap = cfg["tiers"][tier]["max_units"]
    q = str(r.get("injury", "") or "")
    qn = f" x{cfg['staking']['questionable_stake_mult']} ({q})" if q in ("Questionable", "Gtd", "Game Time Decision") else ""
    return (f"full Kelly {f:.1%} of bankroll x {kf:g} (quarter-Kelly) / {up:.0%} per unit = {raw:.2f}u, cap {cap:g}u{qn}, "
            f"rounded to 0.25u = {float(r['units']):g}u")


def _held_text(r, cfg):
    gap = abs(float(r["model_prob"]) - float(r["market_prob"]))
    mg = cfg["model"].get("max_market_gap", 0.15)
    out = [f"model vs market gap {gap:.1%} ({'HELD: over ' + format(mg, '.0%') if gap > mg else 'ok, under ' + format(mg, '.0%')})"]
    if "backfield shift" in str(r.get("usage_note", "")):
        out.append("HELD: backfield shift")
    bm = r.get("base_min")
    if bm is not None and pd.notna(bm) and float(bm) < cfg["model"].get("min_minutes_to_bet", 0):
        out.append(f"HELD: only {float(bm):.0f} base minutes")
    return "; ".join(out)


def _leg_text(r):
    """How the parlay engine prices this leg at DraftKings and FanDuel."""
    sb = side_book(r["book"], r["side"])
    decs = {b: _dec(r["odds"]) - (0 if sb == b else PARLAY["haircut"]) for b in PARLAY["books"]}
    bk = max(decs, key=decs.get)
    lp = float(blend_leg(float(r["model_prob"]), float(r["market_prob"]), r["market"], r["side"]))
    pr = " / ".join(f"{BOOK_NAME[b]} {decimal_to_american(d):+d}{'' if sb == b else ' (est.)'}" for b, d in decs.items())
    return f"leg prob {lp:.1%} (same mix, {r['side']}-side learned fix) | {pr} -> best {BOOK_NAME[bk]} -> leg edge {lp - 1 / decs[bk]:+.1%} (needs {PARLAY['min_edge']:+.1%})"


def _head_nba(r, x):
    L, m, c = [], x["m"], x["c"]
    steps = [("season base", m.get("base")), ("teammates out", m.get("usage")), ("last-3 games pull", m.get("role")),
             ("games without the missing teammate", m.get("wo")), ("back-to-back", m.get("b2b")), ("blowout risk", m.get("blow")),
             ("restriction / back from absence", m.get("restrict")), ("news role", m.get("news")),
             ("new team / 48-min cap", m.get("nt")), ("bench pull", m.get("final"))]
    chain, prev = [], None
    for nm, v in steps:
        if v is None:
            continue
        if prev is None:
            chain.append(f"{v:.1f} {nm}")
        elif abs(v - prev) >= 0.05:
            chain.append(f"{v - prev:+.1f} {nm}")
        prev = v
    L.append("  1 MINUTES    " + " -> ".join(chain) + f" = {m['final']:.1f} projected")
    ctx = []
    if c.get("l3") is not None: ctx.append(f"last 3 games {c['l3']:.1f}")
    if c.get("m10") is not None: ctx.append(f"last 10 {c['m10']:.1f} +/- {c.get('sd10') or 0:.1f}")
    if c.get("sr") is not None: ctx.append(f"started {c['sr']:.0%} of games")
    ctx.append(f"{c.get('n')} games this season")
    if c.get("days") is not None: ctx.append(f"{int(c['days'])} days since last game")
    if c.get("news"): ctx.append(c["news"])
    if c.get("usage"): ctx.append("usage: " + c["usage"])
    L.append("               " + " | ".join(ctx))
    g = x["g"]
    tot = f"implied {g['implied']:.1f} pts vs {g['base']:.1f} expected" if g.get("implied") is not None and g.get("base") else "no game total"
    L.append(f"  2 GAME       {'home' if g.get('home') else 'away'}{' | 2nd night of a back-to-back' if g.get('b2b') else ''} | "
             f"spread {g.get('spread') if g.get('spread') is not None else 'n/a'} | total {g.get('total') if g.get('total') is not None else 'n/a'} ({tot}) | style: {g.get('arch')} vs {g.get('opp_style')} defense")
    parts = []
    for s, v in x["s"].items():
        parts.append(f"  3 {s.upper():<4} RATE {v['raw']:.3f}/min x usage {_xm(v['use'])} | DvP {_xm(v['dvp'])} | pace {_xm(g['pace'])} | "
                     f"home/away {_xm(g['hf'])} | B2B {_xm(g['b2br'])} | new team {_xm(g['ntr'])} | style {_xm(v['af'])} | "
                     f"game total {_xm(v['ef'])}  -> {m['final']:.1f} min x = {v['mu']:.2f}")
    L += parts
    return L


def _head_nfl(r, x):
    L = []
    n, k = x.get("n"), x.get("k")
    pr = (f"2025 prior {x['prior']:.1f}/game ({x['prior_n']} games, worth {k:g} games)" if x.get("prior_n")
          else f"position baseline {x['prior']:.1f} (no 2025 sample, worth {k:g} games)")
    L.append(f"  1 BASE       2026: {n} games, recency-weighted {x['wmean']:.1f} (each older game x{x['decay']:g}; season avg "
             f"{r['season_avg']}, last 3 {r['last3_avg']}) + {pr} = {x['base']:.1f}")
    L.append(f"  2 USAGE      {x['use_add']:+.1f}/game from injured teammates, efficiency {_xm(x['eff'])}"
             + (f" ({r['usage_note']})" if str(r.get("usage_note", "") or "") not in ("", "nan") else " (no change)"))
    env = (f"implied {x['itt']:.1f} pts vs his team's usual {x['itt_avg']:.1f}" if x.get("itt") and x.get("itt_avg") else "no game line")
    sit = str(r.get("situation_note", "") or "")
    sch = str(r.get("scheme_note", "") or "")
    L.append(f"  3 FACTORS    matchup {_xm(r['opp_factor'])} ({r['opp']} vs {r['pos']}s) | game script {_xm(r['env_factor'])} ({env}) | "
             f"situation {_xm(x['sit'])}{' (' + sit + ')' if sit not in ('', 'nan') else ''} | scheme {_xm(r['scheme_factor'])}"
             f"{' (' + sch + ')' if sch not in ('', 'nan') else ''} | mean calibration {_xm(x['mc'])}")
    return L


def explain_card(r, x, labels, cfg):
    side, line, odds = r["side"], float(r["line"]), int(r["odds"])
    flip = (lambda v: v) if side == "Over" else (lambda v: None if v is None else 1 - v)
    sg = 1 if side == "Over" else -1
    book = BOOK_NAME.get(str(r.get("book", "")).lower(), r.get("book", ""))
    L = [f"=== {r['player']} ({r['team']} vs {r['opp']}) - {side} {line:g} {r['market']} {odds:+d} {book} ===",
         "  ROLE: " + " + ".join(labels)]
    L += _head_nba(r, x) if SPORT == "NBA" else _head_nfl(r, x)
    mean, sd = float(r["mean"]), x.get("sd")
    L.append(f"  4 PROJECTION mean {mean:.1f} vs line {line:g} ({mean - line:+.1f})"
             + (f" | median {r['median']}" if "median" in r and pd.notna(r.get("median")) else "")
             + (f" | sd {sd:.1f}" if sd else "") + f" | {x.get('dist')} distribution")
    p = x["p"]
    L.append(f"  5 MODEL      raw P({side} {line:g}) {_pc(flip(p['raw']))} -> calibrated {_pc(flip(p['cal']))} (backtest calibration)")
    fo = lambda v: "-" if v is None else f"{int(v):+d}"
    how = {"2way": f"no-vig of both sides (Over {fo(x.get('oo'))} / Under {fo(x.get('uo'))})", "anchor": "read off his main O/U line (alt/milestone price)",
           "1way": "one-sided price minus estimated vig"}.get(p["how"], p["how"])
    L.append(f"  6 MARKET     no-vig {side} {_pc(flip(p['fair']))} - {how}")
    gate = {0: "", 1: " (gate HALF: model trusted half)", 2: " (gate PAUSED: market only)"}.get(p.get("gate", 0), "")
    L.append(f"  7 WIN CHANCE {p['w']:.0%} model x {_pc(flip(p['cal']))} + {1 - p['w']:.0%} market x {_pc(flip(p['fair']))} = "
             f"{_pc(flip(p['mix']))} | Over-rate fix {sg * (p['bias'] or 0):+.1%} | learned tail/gate {sg * (p['learn'] or 0):+.1%}{gate}"
             f" = {_pc(float(r['blend_prob']))}")
    be = implied_prob(odds)
    tt, tier = _tier_text(r, x, cfg)
    if not tier and any("LEG" in l for l in labels):
        tt += f" - as a parlay leg it only needs {PARLAY['min_edge']:+.1%} (line 11)"
    L.append(f"  8 PRICE      {odds:+d} -> break-even {be:.1%} -> edge {float(r['blend_prob']) - be:+.1%} -> {tt}")
    L.append(f"  9 STAKE      {_stake_text(r, cfg, tier)}")
    risk = [_held_text(r, cfg)]
    if str(r.get("role_risk", "") or "") not in ("", "nan"):
        risk.insert(0, f"role {r['role_risk']} ({r.get('role_why', '')})")
    if str(r.get("injury", "") or "") not in ("", "nan"):
        risk.append(f"injury: {r['injury']}")
    if str(r.get("blind_spot", "") or "") not in ("", "nan"):
        risk.append(f"blind spot: {r['blind_spot']}")
    L.append(f" 10 RISK       " + " | ".join(risk))
    if any("LEG" in l for l in labels):
        L.append(f" 11 PARLAY LEG {_leg_text(r)}")
    L.append(f"  IN ONE LINE: projects {mean:.1f} vs {line:g} -> model {_pc(flip(p['cal']))}, market {_pc(flip(p['fair']))} -> "
             f"VPE {_pc(float(r['blend_prob']))} vs {be:.1%} needed = {float(r['blend_prob']) - be:+.1%}")
    return "\n".join(L)


def cmd_explain(a):
    set_calib(getattr(a, "calib", None), SPORT)
    allp = pd.read_csv(a.all)
    lock_from(allp)
    cfg = CONFIG
    if "x" not in allp:
        print("EXPLAIN: bets_all.csv has no breakdown column (written by an older engine) -> rerun check, then explain")
        return
    want = {}  # (player, market, side, line or None) -> labels

    def add(pl, mk, sd, ln, label):
        want.setdefault((pl, mk, sd, None if ln is None or pd.isna(ln) else float(ln)), []).append(label)
    if a.player:
        mk = canon_market(a.market) if a.market else None
        for nm in [s for s in a.player.split(";") if s.strip()]:
            d = allp[allp["player"].map(norm_name) == norm_name(nm)]
            if mk:
                d = d[d["market"] == mk]
            if a.side:
                d = d[d["side"].str.lower() == a.side.lower()]
            if getattr(a, "line", None):  # v2.23 / v15.24: one rung of a ladder only
                d = d[(pd.to_numeric(d["line"], errors="coerce") - float(a.line)).abs() < 1e-6]
            if d.empty:
                print(f"EXPLAIN: no scored line for {nm}" + (f" {mk}" if mk else ""))
            for _, r in d.iterrows():
                add(r["player"], r["market"], r["side"], r["line"], "asked")
    else:
        if a.bets and os.path.exists(a.bets):
            for _, r in pd.read_csv(a.bets).iterrows():
                add(r["player"], r["market"], r["side"], r["line"], f"STRAIGHT BET {float(r['units']):g}u ({r['tier']})")
        if a.parlays and os.path.exists(a.parlays):
            for pz in _json.load(open(a.parlays)):
                tag = "THE parlay to place" if pz.get("recommended") else "reference only"
                for i, lg in enumerate(pz["legs"], 1):
                    add(lg[0], lg[1], lg[2], lg[3], f"LEG {i} of {pz['name']} parlay at {BOOK_NAME.get(pz['book'], pz['book'])} ({tag})")
        if a.leans and os.path.exists(a.leans):
            ls = pd.read_csv(a.leans)
            for _, r in ls.iterrows():
                add(r["player"], r["market"], r["side"], r["line"], f"LEAN straight {float(r.get('units', 0.5)):g}u (best available, no edge)")
        have = {(k[0], k[1], k[2]) for k in want}
        if a.slip and os.path.exists(a.slip):
            for _, r in pd.read_csv(a.slip).iterrows():
                if (r["player"], r["market"], r["side"]) not in have:
                    add(r["player"], r["market"], r["side"], None, "top of the parlay leg pool (menu pick)" if r.get("kind") != "lean" else "lean leg")
    if not want:
        print("EXPLAIN: nothing suggested in this run (no bets.csv / vpe_parlays.json / leans.csv / slip.csv rows)")
        return
    order = lambda kv: (0 if any(l.startswith("STRAIGHT") for l in kv[1]) else 1 if any("THE parlay" in l for l in kv[1])
                        else 2 if any(l.startswith("LEAN") for l in kv[1]) else 3)
    print(f"EXPLAIN [{VERSION}]: full logic for {len(want)} suggested props (straights, the parlay's legs, leans, other legs)\n")
    for (pl, mk, sd, ln), labels in sorted(want.items(), key=order):
        d = allp[(allp["player"] == pl) & (allp["market"] == mk) & (allp["side"] == sd)]
        if ln is not None:
            d = d[(pd.to_numeric(d["line"], errors="coerce") - ln).abs() < 1e-6]
        if d.empty:
            print(f"=== {pl} {sd} {ln} {mk}: not in bets_all.csv (rerun check) ===\n")
            continue
        r = d.sort_values("blend_prob", ascending=False).iloc[0].to_dict()
        try:
            x = _json.loads(r["x"])
        except Exception:
            print(f"=== {pl} {sd} {mk}: no breakdown stored ===\n")
            continue
        if mk in GAME_MK:
            print(game_card(r, x, labels, cfg) + "\n")
            continue
        print(explain_card(r, x, labels, cfg) + "\n")


def nba_x(p, mk, mu, var, oo, uo, pchain):
    """Everything behind one NBA projection, for `explain` (v2.15)."""
    g = lambda k: _r(p.get(k)) if k in p else None
    fin = _r(p.get("proj_min"), 2)
    return {"m": {"base": _r(p.get("min"), 2), "usage": g("xm_usage"), "role": g("xm_role"), "wo": g("xm_wo"), "b2b": g("xm_b2b"),
                  "blow": g("xm_blow"), "restrict": g("xm_restrict"), "news": g("xm_news"), "nt": g("xm_nt"), "final": fin},
            "c": {"l3": g("min_l3"), "m10": g("min_m10"), "sd10": g("min_sd10"), "sr": g("start_rate"), "n": int(p.get("n_cur", 0) or 0),
                  "days": g("days_off"), "news": str(p.get("news_note", "") or ""), "usage": str(p.get("usage_note", "") or "")},
            "g": {"home": bool(p.get("home")), "b2b": bool(p.get("b2b")), "spread": _r(p.get("spread"), 1), "total": _r(p.get("total"), 1) if "total" in p else None,
                  "implied": g("implied_pts"), "base": g("xbase_pts"), "env": g("env"), "pace": g("pace_f"), "hf": g("xh_f"), "b2br": g("xb2b_r"),
                  "ntr": g("nt_rate"), "arch": str(p.get("arch", "")), "opp_style": str(p.get("opp_style", ""))},
            "s": {c: {"raw": _r(p.get("xraw_" + c), 4), "use": g("u_" + c), "dvp": g("dvp_" + c), "af": g("af_" + c), "ef": g("ef_" + c),
                      "mu": _r(p.get("mu_" + c), 2)} for c in MARKETS[mk]},
            "sd": _r(math.sqrt(max(var, 0)), 2),
            "dist": "normal" if mk in CONT else ("Poisson" if var <= mu * 1.02 else "negative binomial"),
            "oo": None if pd.isna(oo) else int(oo), "uo": None if pd.isna(uo) else int(uo), "p": pchain}



# ---------- v2.16 / v15.17: take-if on recordings, reprice, several boosts in one run, est. market margin ----------
# Lines from a recording are priced as recorded, but bets often go in hours later and the price moves (11/11: three
# boost legs moved 15-35 cents). So every pick, leg and parlay prints a take-if again, and `reprice` re-rates a few
# props (or the parlay they make) at the live price without a full rerun.
MARKET_EXTRA = {"pts": 0.02}  # v2.22: extra edge by market. 11/16 Shadow test: VPE's side on points lost 15.1% (1,068
                              # lines), the worst market by far; threes -1.7%, PRA -2.1%, assists -6.9%, rebounds -8.6%
BOOST_LEG = {"min_edge": None, "w": 5.0}  # v2.24 / v15.25: --boost-min-leg-edge sets a hard floor; w = close-call tilt
BOOST_PREF = {"fg3m": 0.03, "pra": 0.03}  # v2.22: boost parlays lean toward VPE's best markets on close calls
EST_MARGIN = 0.015  # extra edge a pick needs when its market chance comes from ONE price (no main O/U to read it off):
                    # the hold on those is assumed (7%), and ~10% real hold would cost ~1.5 pts of market chance
LOG_UNIT_USD = 1.0  # the Pick Log's unit: $1 (Summary "Unit ($)"), so logged units == stake_usd


def parlay_take_if(p, boost=0.0, p_mkt=None):
    """Worst parlay price (decimal, BEFORE any boost) still worth taking: break-even at VPE's hit chance; for a boost
    parlay also break-even at market-only chances (a boost parlay must pay off even if the model adds nothing)."""
    mult = 1 + (boost or 0.0) / 100
    need = 1 + (1 / max(p, 1e-6) - 1) / mult
    if p_mkt:
        need = max(need, 1 + (1 / max(p_mkt, 1e-6) - 1) / mult)
    return need


def parse_boosts(spec):
    """--boosts "fd30:3leg:+100:win25,fd25,dk30": book + % and optional rules per boost: Nleg = min legs, maxN = max
    legs, +X/-X = min total odds, winN = max extra winnings ($), any = no rules beyond the defaults (rules confirmed)."""
    out = []
    for tok in [t.strip() for t in (spec or "").split(",") if t.strip()]:
        parts = tok.lower().replace(" ", "").split(":")
        m = re.match(r"^(fd|fanduel|dk|draftkings)(\d+(?:\.\d+)?)%?$", parts[0])
        if not m:
            print(f"BOOSTS: can't read '{tok}' (format: fd30, dk25:3leg:+100:win25)")
            continue
        b = {"book": "fanduel" if m.group(1).startswith("f") else "draftkings", "pct": float(m.group(2)),
             "label": tok, "given": len(parts) > 1}
        for q in parts[1:]:
            if re.fullmatch(r"\d+leg?s?", q):
                b["min_legs"] = int(re.match(r"\d+", q).group())
            elif re.fullmatch(r"max\d+", q):
                b["max_legs"] = int(q[3:])
            elif re.fullmatch(r"[+-]\d+", q):
                b["min_odds"] = float(q)
            elif re.fullmatch(r"win\d+(\.\d+)?", q):
                b["max_win"] = float(q[3:])
            elif q == "sgp":  # v2.23: same-game-only boost (FD "3+ leg SGP"): built by `sgpbuild`, not `parlay`
                b["sgp"] = True
            elif q not in ("any", "norules"):
                print(f"BOOSTS: ignored '{q}' in {tok}")
        out.append(b)
    return out


def boost_spec_from_args(a):
    bbook = (a.boost_book or "draftkings").lower().replace(" ", "")
    rules = {"min_legs": getattr(a, "boost_min_legs", None), "max_legs": getattr(a, "boost_max_legs", None),
             "min_odds": getattr(a, "boost_min_odds", None), "max_win": getattr(a, "boost_max_win", None)}
    return {"book": bbook, "pct": float(a.boost), "label": f"{'fd' if bbook.startswith('f') else 'dk'}{a.boost:g}",
            "given": any(v for v in rules.values()), **rules}


def print_boost_spec(spec, allp, stake, cols, keys, used, a):
    """One boost -> its own parlay at its book, never reusing a leg (player) another boost already took."""
    boost, bbook = spec["pct"], spec["book"]
    excl = [x for x in (getattr(a, "exclude", None) or "").split(";") if x.strip()] + list(used)
    pool = parlay_pool(allp, excl, min_edge=-0.03, min_prob=0.5)
    min_dec = american_to_decimal(spec["min_odds"]) if spec.get("min_odds") else 1.0
    lo, hi, mw = int(spec.get("min_legs") or 2), int(spec.get("max_legs") or 4), spec.get("max_win")
    print(f"\n=== BOOST PARLAY {spec['label']}: {boost:g}% profit boost at {BOOK_NAME.get(bbook, bbook)} ===")
    rules = (f"{lo}-{hi} legs" + (f", min total odds {int(spec['min_odds']):+d}" if spec.get("min_odds") else "")
             + (f", extra winnings capped at ${mw:g}" if mw else ""))
    print(f"  rules used: {rules}" + ("" if spec.get("given") else
          "  (DEFAULTS - VPE doesn't know this boost's rules: read min legs / min odds / max win off the boost and pass "
          "them, e.g. fd30:3leg:+100:win25, or :any if it has none)"))
    if used:
        print(f"  not reusing legs from bigger boosts: {', '.join(used)}")
    r = boost_parlay(pool, bbook, boost, stake, lo, hi, min_dec, mw)
    lean = False
    if not r:
        print("  SAVE IT: no combination left is +EV at market odds with this boost"
              + (" without reusing another boost's legs" if used else ""))
        r = boost_parlay(pool, bbook, boost, stake, lo, hi, min_dec, mw, need_market_ev=False)
        if not r:
            print("  and no legs qualify even as a lean")
            return None
        lean = True
        print("  LEAN - BEST AVAILABLE BOOST: no edge; only if you use this boost tonight anyway, this is the least-bad way")
    ev, p, p_mkt, dec, dec_eff, combo = r
    print(f"  PLACE AT {BOOK_NAME.get(bbook, bbook).upper()} | {len(combo)} legs | est. odds {decimal_to_american(dec):+d} -> "
          f"{decimal_to_american(dec_eff):+d} boosted | ${stake:.2f} pays ~${stake * dec_eff:.2f} | hit chance {p:.1%} "
          f"(market {p_mkt:.1%}) | EV {ev:+.0%} (EV at market odds alone {p_mkt * dec_eff - 1:+.0%}) | "
          f"Unders {sum(c['side'] == 'Under' for c in combo)}")
    print(pd.DataFrame(combo)[[c for c in cols if c in combo[0]]].to_string(index=False))
    # where the edge actually comes from: without the boost, and at the other book (11/9: P2 placed at DK, no boost: -3.5%)
    ev0, evm0 = p * dec - 1, p_mkt * dec - 1
    msg = f"  WITHOUT the boost at {BOOK_NAME[bbook]}: EV {ev0:+.1%} (market-only {evm0:+.1%})"
    for ob in [b for b in PARLAY["books"] if b != bbook]:
        dec_o = float(np.prod([c["dec_" + ob] for c in combo]))
        msg += (f" | at {BOOK_NAME[ob]}, no boost: {decimal_to_american(dec_o):+d}, EV {p * dec_o - 1:+.1%} "
                f"(market-only {p_mkt * dec_o - 1:+.1%})")
    print(msg + (f" -> the edge is the boost: place it only WITH the boost at {BOOK_NAME[bbook]}" if min(ev0, evm0) <= 0 else ""))
    tk = parlay_take_if(p, boost, None if lean else p_mkt)
    print(f"  TAKE-IF: {BOOK_NAME[bbook]} parlay odds before the boost >= {decimal_to_american(tk):+d}"
          + (" (below that it's -EV even with the boost)" if not lean else " (VPE break-even; it's a lean)")
          + " and every leg at or above its take_min. Price moved? -> `reprice`")
    est = sum(str(c.get("mkt_est")).lower() in ("true", "1") for c in combo)
    if est:
        print(f"  note: {est} leg(s) have no main O/U line, so their market chance is estimated from one price")
    held = [f"{c['player']}: {c['held_note']}" for c in combo if str(c.get("held_note", "") or "") not in ("", "nan")]
    if held:
        print("  FLAG - player HELD on another market tonight (model vs market gap, usually news): " + " | ".join(held))
    if mw:
        print(f"  (boost winnings capped at ${mw:g}: counted)")
    if not lean:
        keys += [(c["player"], c["market"], c["side"]) for c in combo]
    return {"name": f"BOOST {spec['label']}" + (" (LEAN)" if lean else ""), "target": 0, "book": bbook,
            "boost_pct": boost, "odds": decimal_to_american(dec), "take_if": decimal_to_american(tk), "hit": round(p, 4),
            "lean": lean, "ev": round(ev, 3), "stake_usd": stake,
            "vig": round(1 - float(np.prod([float(c["market_prob"]) * float(c["dec_" + bbook]) for c in combo])), 3),
            "n_two": int(sum(bool(c.get("two_sided")) for c in combo)),
            "legs": [[c["player"], c["market"], c["side"], float(c["line"])] for c in combo]}


def cmd_reprice(a):
    """Live price check without a full rerun: the app's price now for a few props (and the parlay they make), re-rated
    with tonight's projections. The market's chance moves with the price (same hold); the model's chance doesn't."""
    set_calib(getattr(a, "calib", None), SPORT)
    allp = pd.read_csv(a.all)
    lock_from(allp)
    cfg = CONFIG
    legs = []
    for spec in a.bet:
        t = [x.strip() for x in spec.split(";")]
        if len(t) < 5:
            print(f"{spec}: need Player;market;Side;line;odds")
            legs.append(None)
            continue
        pl, mk, side, line, odds = t[0], canon_market(t[1]) or t[1], t[2].title(), float(t[3]), _odds(t[4])
        tag = f"{pl} {side} {line:g} {mk} @ {odds:+d}"
        h = allp[(allp["player"].map(norm_name) == norm_name(pl)) & (allp["market"] == mk) & (allp["side"] == side)]
        hl = h[(pd.to_numeric(h["line"], errors="coerce") - line).abs() < 1e-9]
        if hl.empty:
            lines_ = sorted(pd.to_numeric(h["line"], errors="coerce").dropna().unique())
            print(f"{tag}: not scored at this line tonight" + (f" (scored: {', '.join(f'{x:g}' for x in lines_)})" if lines_ else "")
                  + " -> the LINE moved: rerun check with the new line")
            legs.append(None)
            continue
        r = hl.iloc[0]
        old = int(r["odds"])
        mp_new = float(np.clip(float(r["market_prob"]) * implied_prob(odds) / implied_prob(old), 0.01, 0.99))
        p_new = float(blend_leg(float(r["model_prob"]), mp_new, mk, side))
        estm = (EST_MARGIN if str(r.get("mkt_est")).lower() in ("true", "1") else 0.0) + MARKET_EXTRA.get(mk, 0.0)
        be = implied_prob(odds)
        edge = p_new - be
        tier = _tier(edge - estm, p_new, cfg)
        if tier:
            v = f"TAKE as a straight ({tier.replace('_', ' ')})"
        elif edge - estm >= PARLAY["min_edge"] and p_new >= PARLAY["min_prob"]:
            v = "TAKE as a parlay leg only"
        else:
            v = "SKIP at this price"
        print(f"{tag} (recorded {old:+d}): market {float(r['market_prob']):.1%} -> {mp_new:.1%} | VPE {p_new:.1%} vs "
              f"{be:.1%} needed -> edge {edge:+.1%}{f' (needs {estm:+.1%} more: estimated market / points)' if estm else ''} -> {v}")
        legs.append({"p": p_new, "pm": mp_new, "dec": american_to_decimal(odds), "game": str(r.get("game_id")), "tag": tag})
    if not a.parlay or not legs:
        return
    if any(x is None for x in legs):
        print("PARLAY: a leg isn't scored at its line -> can't rate it (rerun check)")
        return
    p, pm = float(np.prod([x["p"] for x in legs])), float(np.prod([x["pm"] for x in legs]))
    dec = american_to_decimal(_odds(a.parlay_odds)) if a.parlay_odds else float(np.prod([x["dec"] for x in legs]))
    if len({x["game"] for x in legs}) < len(legs):
        print("note: same-game legs -> chances treated as independent; rate it with the sgp command instead")
    boost = a.boost or 0.0
    extra = (dec - 1) * boost / 100
    if a.max_win and a.stake:
        extra = min(extra, a.max_win / a.stake)
    dec_eff = dec + extra
    ev, evm = p * dec_eff - 1, pm * dec_eff - 1
    tk = parlay_take_if(p, boost, pm if boost else None)
    ok = ev > 0 and (not boost or evm >= 0)
    print(f"PARLAY {decimal_to_american(dec):+d}" + (f" -> {decimal_to_american(dec_eff):+d} with the {boost:g}% boost" if boost else "")
          + f" | hit {p:.1%} (market {pm:.1%}) | EV {ev:+.1%} | market-only EV {evm:+.1%} | take-if {decimal_to_american(tk):+d}"
          + f" -> {'TAKE' if ok else 'SKIP'}")



# ---------- v2.17 / v15.18: per-market model weight, closing-line snapshots + status rows ----------
SNAP_COLS = ["date", "week", "player", "matchup", "market", "line", "over_odds", "under_odds", "book", "captured_at"]
CLOSE_STATUS_COLS = ["run_at", "sport", "slate", "open_bets", "priced", "snapshot_rows", "note"]


_GW = {}

def _gw():
    """v2.30: model weight for game lines (margin, total), learned by `games --hist` from the graded 'Games NBA' tab
    and saved to games_w.json next to the run. No file = the 30% start."""
    if not _GW:
        import json
        try:
            _GW.update(json.load(open("games_w.json")))
        except Exception:
            _GW.update({"margin": GAME["model_w"], "total": GAME["model_w"]})
    return _GW

def _mw(market=None):
    """Model weight for this market (v2.17: each market learns its own mix, shrunk toward the shared one)."""
    mw = dict(PARLAY.get("mk_weight") or {})
    g = PARLAY["model_weight"]
    if (isinstance(market, str) and market in GAME_MK) or (isinstance(market, (pd.Series, np.ndarray)) and pd.Series(market).isin(GAME_MK).any()):
        gw_ = _gw()
        mw.update({"spread": gw_["margin"], "ml": gw_["margin"], "total": gw_["total"]})
    if isinstance(market, pd.Series):
        return market.map(mw).fillna(g).astype(float).to_numpy()
    if isinstance(market, np.ndarray):
        return np.array([mw.get(m, g) for m in market], dtype=float)
    return float(mw.get(market, g)) if market is not None else float(g)


def snap_rows(props, slate, week=""):
    """props_close rows -> 'Close <sport>' tab rows (every scored line's closing price, saved whether or not bets were logged)."""
    now = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M")
    out = []
    for _, r in props.iterrows():
        oo, uo = r.get("over_odds"), r.get("under_odds")
        if pd.isna(oo) or pd.isna(uo) or str(oo) == "" or str(uo) == "":
            continue
        out.append([str(slate), str(week), r["player"], r.get("matchup", ""), r["market"], float(r["line"]), int(float(oo)), int(float(uo)),
                    r.get("book", ""), now])
    return out


def snap_props(path, slate=None, week=None):
    """'Close <sport>' tab JSON -> a props frame for one slate (NBA date) or one week (NFL)."""
    vals = _values(path)
    if len(vals) < 2:
        return pd.DataFrame(columns=["player", "matchup", "market", "line", "over_odds", "under_odds", "book"])
    hdr = [str(h) for h in vals[0]]
    d = pd.DataFrame([list(r) + [""] * (len(hdr) - len(r)) for r in vals[1:]], columns=hdr)
    if slate is not None:
        d = d[d["date"].astype(str).str.lstrip("'").str[:10] == str(slate)]
    if week is not None:
        d = d[pd.to_numeric(d["week"], errors="coerce") == int(week)]
    for c in ("line", "over_odds", "under_odds"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[d["over_odds"].notna() & d["under_odds"].notna()].reset_index(drop=True)


def write_close_status(path, sport, slate, open_bets, priced, snap_n, note):
    if not path:
        return
    import json as _j
    now = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M")
    _j.dump([[now, sport, str(slate), int(open_bets), int(priced), int(snap_n), note]], open(path, "w"))
    print(f"CLOSE STATUS: {sport} {slate} | open bets {open_bets} | priced {priced} | snapshot rows {snap_n} | {note}"
          f" -> {path} (append to the 'Close log' tab)")


# ---------- v2.18 / v15.19: cross-check a recording against the free odds pages ----------
# On recording nights the recording stays the price VPE bets on. The free odds pages are pulled too, only to
# (1) catch numbers misread off a video frame, (2) list props the recording missed, (3) measure how far the free
# pages' prices sit from the user's book (the 'Line check' tab: the honest answer to "do I need to record?").
LINECHECK_COLS = ["date", "slate", "player", "market", "line", "side", "rec_book", "rec_odds", "web_odds", "web_book",
                  "gap_pts", "flag", "captured_at"]

def _x_odds(v):
    s = str(v).strip().replace("+", "")
    if s.lower() in ("", "nan", "none"):
        return None
    if s.lower() in ("ev", "even"):
        return 100.0
    try:
        x = float(s)
    except ValueError:
        return None
    return x if abs(x) >= 100 else None

def _x_imp(o):
    return 1.0 / american_to_decimal(o)

def _x_frame(path, web):
    d = _read_loose_csv(path)
    for c in ("player", "market", "line", "over_odds", "under_odds", "book"):
        if c not in d:
            d[c] = ""
    d = d.assign(_k=d["player"].map(norm_name), _m=d["market"].map(canon_market),
                 _l=pd.to_numeric(d["line"], errors="coerce"),
                 _o=d["over_odds"].map(_x_odds), _u=d["under_odds"].map(_x_odds))
    return d[d["_m"].notna() & d["_l"].notna()]

def cmd_crosscheck(a):
    rec, web = _x_frame(a.rec, False), _x_frame(a.web, True)
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        now = _dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = _dt.datetime.utcnow()
    stamp = now.strftime("%Y-%m-%d %H:%M ET")
    slate = a.slate or now.strftime("%Y-%m-%d")
    flags, rows = [], []
    # 1) the recording on its own: a two-sided price whose hold is impossible is a misread number
    for _, r in rec.iterrows():
        if r["_o"] is not None and r["_u"] is not None:
            hold = _x_imp(r["_o"]) + _x_imp(r["_u"]) - 1
            if hold < -0.005 or hold > 0.16:
                flags.append(("RED", r["player"], r["_m"], r["_l"], f"Over {int(r['_o']):+d} / Under {int(r['_u']):+d} = {hold:+.1%} hold"
                              " -> one side is likely misread (real holds run ~4-10%)"))
    # v2.23 / v15.24: FanDuel's SGP tab shows a flat -113 / -113 on every O/U; it is not the straight price. Several
    # flat -113/-113 FD rows in one recording = it was probably recorded from the SGP tab.
    fl = rec[(rec["book"].astype(str).str.lower().str.replace(" ", "", regex=False).isin(["fanduel", "fd"]))
             & (rec["_o"] == -113) & (rec["_u"] == -113)]
    if len(fl):
        lvl_ = "RED" if len(fl) >= 3 else "INFO"
        for _, r in fl.iterrows():
            flags.append((lvl_, r["player"], r["_m"], r["_l"], "FanDuel Over -113 / Under -113: the flat SGP-tab display, "
                          "probably not the straight price -> re-check on the player's prop page"
                          + (f" ({len(fl)} flat -113/-113 FD rows in this recording)" if len(fl) >= 3 else "")))
    if web.empty:
        print(f"CROSSCHECK: no free-page lines to compare (page failed or on another night) | recording checked on its own: "
              f"{sum(f[0] == 'RED' for f in flags)} RED")
    # 2) recording vs the free pages, prop by prop
    wmain = web[web["_o"].notna() & web["_u"].notna()]
    matched = 0
    for (k, m), g in rec.groupby(["_k", "_m"]):
        w = web[(web["_k"] == k) & (web["_m"] == m)]
        if w.empty:
            continue
        rmain = g[g["_o"].notna() & g["_u"].notna()]
        wm = wmain[(wmain["_k"] == k) & (wmain["_m"] == m)]
        name = g["player"].iloc[0]
        if len(rmain) and len(wm):
            rl, wl = float(rmain["_l"].iloc[0]), float(wm["_l"].iloc[0])
            if abs(rl - wl) >= a.line_gap:
                flags.append(("RED", name, m, rl, f"main line {rl:g} in the recording vs {wl:g} on the free page"
                              " -> misread line, or it moved since the recording"))
        for _, r in g.iterrows():
            same = w[w["_l"] == r["_l"]]
            for side, rc, wc in (("Over", "_o", "_o"), ("Under", "_u", "_u")):
                ro = r[rc]
                cand = same[same[wc].notna()]
                if ro is None or cand.empty:
                    continue
                wr = cand.iloc[0]
                wo, wb = wr[wc], side_book(wr["book"], side)
                rb = str(r["book"]).strip().lower()
                gap = _x_imp(ro) - _x_imp(wo)  # + = the recording's price is worse than the free page's
                matched += 1
                flag = ""
                if wb == rb and abs(gap) > a.same_book_gap:
                    flag = "RED"
                    flags.append(("RED", name, m, r["_l"], f"{side} {int(ro):+d} recorded vs {int(wo):+d} at the same book ({wb}) on the free page"
                                  " -> misread, or the price moved"))
                elif wb != rb and gap < -a.better_gap:
                    flag = "RED"
                    flags.append(("RED", name, m, r["_l"], f"{side} {int(ro):+d} recorded beats every book on the free page (best {int(wo):+d} {wb})"
                                  " -> check that number"))
                elif wb != rb and gap > a.better_gap:
                    flag = "INFO"
                    flags.append(("INFO", name, m, r["_l"], f"{side} better elsewhere: {int(wo):+d} at {wb} vs {int(ro):+d} recorded (you bet DK/FD; info only)"))
                rows.append([slate, a.week or "", name, m, float(r["_l"]), side, rb, int(ro), int(wo), wb, round(gap * 100, 1), flag, stamp])
    # 3) props on the free page that the recording doesn't have (not scored tonight: the recording is the price)
    have = set(zip(rec["_k"], rec["_m"]))
    miss = web[[(k, m) not in have for k, m in zip(web["_k"], web["_m"])]].drop_duplicates(["_k", "_m"])
    nred, ninfo = sum(f[0] == "RED" for f in flags), sum(f[0] == "INFO" for f in flags)
    if not web.empty:
        same_book = [r[10] for r in rows if r[6] == r[9]]
        print(f"CROSSCHECK: {matched} prices matched to the free page | {nred} RED | {ninfo} better-price notes | "
              f"{len(miss)} props on the free page not in the recording"
              + (f" | same-book gap: median {pd.Series(same_book).abs().median():.1f} pts" if same_book else ""))
    for lvl in ("RED", "INFO"):
        for f in [f for f in flags if f[0] == lvl][: (40 if lvl == "RED" else 8)]:
            print(f"  {'RED ' if lvl == 'RED' else 'note'} {f[1]} {f[2]} {f[3]:g}: {f[4]}")
    if nred:
        print("  -> for each RED: look at that frame again. Misread -> fix props.csv. Price/line moved -> leave it (the recording is what you bet at; the take-if covers moves).")
    if len(miss):
        by = miss.groupby("_m")["player"].apply(lambda s: ", ".join(list(s)[:6]) + (f" +{len(s) - 6} more" if len(s) > 6 else ""))
        for m, s in by.items():
            print(f"  missing from the recording ({m}): {s}")
        print("  -> not scored tonight (the recording sets the price). Want them? Record that list or screenshot it, then rerun check.")
    if a.out:
        with open(a.out, "w") as fh:
            _json.dump(rows, fh)
        print(f"LINE CHECK: {len(rows)} rows -> {a.out} (append to the 'Line check' tab)")


FETCH_SPORT = "NBA"
# ---------- v2.19 / v15.20: engine-side page download (no web-fetch tool, no permission prompt) ----------
# 11/12-11/14 every unattended closing run failed: the session's web-fetch asked for permission and nobody was there.
# `fetchpages` downloads the odds pages itself through the session's network. It only works when the environment's
# network access allows www.scoresandodds.com; when it doesn't, it says so in plain words (and `close` puts that in
# the 'Close log' note) instead of a vague "blocked or stale?".
ODDS_PAGES = {
    "NBA": {"pts": "https://www.scoresandodds.com/nba/props/points",
            "reb": "https://www.scoresandodds.com/nba/props/rebounds",
            "ast": "https://www.scoresandodds.com/nba/props/assists",
            "fg3m": "https://www.scoresandodds.com/nba/props/3-pointers"},
    "NFL": {"pass_yds": "https://www.scoresandodds.com/nfl/props/passing-yards",
            "rush_yds": "https://www.scoresandodds.com/nfl/props/rushing-yards",
            "rec_yds": "https://www.scoresandodds.com/nfl/props/receiving-yards",
            "receptions": "https://www.scoresandodds.com/nfl/props/receptions"},
}

def _page_text(html):
    import html as _h
    s = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", html)
    s = re.sub(r"(?i)<br\s*/?>|</(div|p|li|tr|td|th|h\d|span|a|section|article)>", "\n", s)
    s = _h.unescape(re.sub(r"<[^>]+>", " ", s))
    lines = [re.sub(r"[ \t ]+", " ", x).strip() for x in s.splitlines()]
    return "\n".join(x for x in lines if x)

def _fetch_one(url, tries=2):
    import urllib.request, urllib.error
    err = ""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (VPE closing-price capture)",
                                                       "Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace"), ""
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} from the site"
        except urllib.error.URLError as e:
            m = str(e.reason)
            err = ("BLOCKED by this session's network (proxy 403): allow www.scoresandodds.com in the cloud environment's "
                   "Network access (Custom > Allowed domains)") if "403" in m or "Tunnel" in m else f"network error: {m}"
            if "BLOCKED" in err:
                break  # a policy block won't change on retry
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(3)
    return None, err

def parse_props_html(html):
    """v2.20: read a ScoresAndOdds props page straight from its HTML (no transcription by hand).
    Each player is one <li class="border"> row: name + matchup, then the best Over and best Under (line, odds, book)."""
    import html as _h
    rows = []
    for blk in re.split(r'<li class="border"', html)[1:]:
        blk = blk.split("</li>", 1)[0] if 'class="table-list-button' not in blk else blk.split('class="table-list-button', 1)[0]
        nm = re.search(r'class="props-name">\s*<a[^>]*>(.*?)</a>', blk, re.S)
        mu = re.search(r'class="props-name">.*?<span class="bold small gray">(.*?)</span>', blk, re.S)
        if not nm:
            continue
        rec = {"player": _h.unescape(re.sub(r"<[^>]+>", "", nm.group(1))).strip(),
               "matchup": _h.unescape(mu.group(1)).strip() if mu else ""}
        for cell in re.findall(r'<div class="game-odds[^"]*">(.*?)</div>\s*<div class="book-logo">(.*?)</div>\s*</div>', blk, re.S):
            ln = re.search(r'class="data-moneyline">\s*([ou])\s*([\d.]+)\s*<', cell[0])
            od = re.search(r'class="data-odds[^"]*">\s*([+-]?\d+|even|EVEN|Even)\s*<', cell[0])
            bk = re.search(r'alt="([^"]+)"', cell[1])
            if not ln or not od:
                continue
            side = "over" if ln.group(1) == "o" else "under"
            if f"{side}_line" in rec:
                continue
            o = od.group(1)
            rec[f"{side}_line"] = ln.group(2)
            rec[f"{side}_odds"] = "100" if o.lower() == "even" else o
            rec[f"{side}_book"] = re.sub(r"\s+", "", bk.group(1).lower()) if bk else ""
        if "over_line" in rec or "under_line" in rec:
            rows.append(rec)
    return rows


MIRROR = "https://raw.githubusercontent.com/AP9000Moneywinner420/vpe-engine/odds-pages/"  # v2.20: GitHub copy, refreshed
MIRROR_MAX_AGE_H = 3.0                                                                      # every 20 min by .github/workflows/odds-pages.yml

def _mirror_one(sport, mk):
    """The same page from the GitHub mirror (GitHub is always reachable from VPE sessions)."""
    import gzip, urllib.request, datetime as _dt
    try:
        global MIRROR
        if "/odds-pages/" in MIRROR:   # v2.25: read by commit id, so a just-refreshed mirror is never served from a 5-minute cache
            try:
                import subprocess
                sha = subprocess.run(["git", "ls-remote", "https://github.com/AP9000Moneywinner420/vpe-engine", "odds-pages"],
                                     capture_output=True, text=True, timeout=30).stdout.split()[0]
                if len(sha) == 40:
                    MIRROR = MIRROR.replace("/odds-pages/", f"/{sha}/")
            except Exception:
                pass
        ts = urllib.request.urlopen(MIRROR + "fetched_at.txt", timeout=30).read().decode().strip()
        age = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)).total_seconds() / 3600
        if age > MIRROR_MAX_AGE_H:
            return None, f"GitHub mirror is {age:.1f} h old (limit {MIRROR_MAX_AGE_H:g} h): the odds-pages workflow may not have run", None
        raw = urllib.request.urlopen(MIRROR + f"{sport.lower()}/{mk}.html.gz", timeout=60).read()
        return gzip.decompress(raw).decode("utf-8", "replace"), "", age
    except Exception as e:
        return None, f"GitHub mirror failed: {type(e).__name__}: {e}", None

def cmd_fetchpages(a):
    """Download the props pages with the engine (no web-fetch tool). Writes <out>/<market>.html and <market>.txt
    (clean page text to transcribe into <market>.csv) and a status file that `close --fetch-status` turns into the
    'Close log' note."""
    os.makedirs(a.out, exist_ok=True)
    rows, ok = [], 0
    for mk, url in ODDS_PAGES[FETCH_SPORT].items():
        if a.markets and mk not in a.markets:
            continue
        html, err = _fetch_one(url)
        src = "direct"
        if html is None:  # v2.20: direct download blocked -> the GitHub mirror of the same page
            direct_err = err
            html, err, age = _mirror_one(FETCH_SPORT, mk)
            src = f"GitHub mirror, {age * 60:.0f} min old" if html is not None else src
            if html is None:
                err = f"{direct_err}; {err}"
        if html is not None:
            txt = _page_text(html)
            open(os.path.join(a.out, f"{mk}.html"), "w").write(html)
            open(os.path.join(a.out, f"{mk}.txt"), "w").write(txt)
            parsed = parse_props_html(html)
            cols = ["player", "matchup", "over_line", "over_odds", "over_book", "under_line", "under_odds", "under_book"]
            pd.DataFrame(parsed, columns=cols).to_csv(os.path.join(a.out, f"{mk}.csv"), index=False)
            ok += 1
            rows.append({"market": mk, "url": url, "status": "ok", "source": src, "bytes": len(html), "players": len(parsed)})
            print(f"  {mk:<10} ok  {len(parsed):>4} players ({src}) -> {a.out}/{mk}.csv")
            if not parsed:
                print(f"  {mk:<10} WARNING: page downloaded but 0 players parsed (layout changed?): read {a.out}/{mk}.txt and write the CSV by hand")
        else:
            rows.append({"market": mk, "url": url, "status": "failed", "error": err})
            print(f"  {mk:<10} FAILED  {url}  {err}")
    import json as _j
    _j.dump({"sport": FETCH_SPORT, "pages": rows}, open(a.status, "w"))
    print(f"FETCH {FETCH_SPORT}: {ok}/{len(rows)} pages downloaded" + ("" if ok == len(rows) else
          f" | {fetch_note(a.status)}"))

def fetch_note(path):
    """One-line summary of failed pages for the 'Close log' note."""
    import json as _j
    try:
        d = _j.load(open(path))
    except Exception:
        return ""
    bad = [p for p in d.get("pages", []) if p.get("status") != "ok"]
    if not bad:
        return ""
    errs = sorted({p.get("error", "") for p in bad})
    return "pages failed: " + ", ".join(p["market"] for p in bad) + " (" + "; ".join(errs) + ")"

def close_note(note, n_lines, a):
    """v2.19: name the exact pages and errors instead of 'blocked or stale?'."""
    extra = "; ".join(x for x in (fetch_note(a.fetch_status) if getattr(a, "fetch_status", None) else "",
                                  getattr(a, "fetch_note", None) or "") if x)
    if not n_lines:
        return "NO closing lines: " + (extra or "no props file passed (pages not fetched?)")
    return note + (f" | {extra}" if extra else "")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Verified Prop Engine - NBA (in-chat edition)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("board", "slate", "lines", "check", "log", "grade", "backtest", "parlay", "styles"):
        p = sub.add_parser(name)
        p.add_argument("--date"); p.add_argument("--bankroll", type=float); p.add_argument("--profile")
    for name in ("board", "slate", "check"):
        sub.choices[name].add_argument("--injuries", help='"Name:Out;Name:Questionable;Name:Out:PHI"')
    sub.choices["styles"].add_argument("--team"); sub.choices["styles"].add_argument("--stat")
    gm_ = sub.add_parser("games"); gm_.add_argument("--date"); gm_.add_argument("--games", required=True)
    gm_.add_argument("--injuries", default=""); gm_.add_argument("--out", default="game_bets.csv")
    gm_.add_argument("--log-out", default="games_log.json"); gm_.add_argument("--bankroll", type=float); gm_.add_argument("--profile")
    gm_.add_argument("--all", default="bets_all.csv"); gm_.add_argument("--bets", default="bets.csv"); gm_.add_argument("--hist"); gm_.add_argument("--book")
    gg_ = sub.add_parser("gamesgrade"); gg_.add_argument("--json", required=True); gg_.add_argument("--out", default="games_tab.json")
    sp_ = sub.add_parser("sgp"); sp_.add_argument("--date"); sp_.add_argument("--games", required=True)
    sp_.add_argument("--injuries", default=""); sp_.add_argument("--leg", action="append", required=True)
    sp_.add_argument("--price", required=True); sp_.add_argument("--min-ev", type=float, default=0.10)
    sp_.add_argument("--bankroll", type=float); sp_.add_argument("--profile")
    sp_.add_argument("--calib", default="calib_tab.json", help="Calib tab JSON: learned model/market mix (same as reprice)")
    sp_.add_argument("--ratings-out", default="sgp_ratings.json")
    sb_ = sub.add_parser("sgpbuild"); sb_.add_argument("--date"); sb_.add_argument("--games", required=True)
    sb_.add_argument("--all", default="bets_all.csv"); sb_.add_argument("--injuries", default="")
    sb_.add_argument("--calib", default="calib_tab.json"); sb_.add_argument("--exclude")
    sb_.add_argument("--boosts", help='e.g. "fd50:sgp:3leg:+100"'); sb_.add_argument("--boost-book", default="fanduel")
    sb_.add_argument("--boost", type=float); sb_.add_argument("--legs", type=int, default=3); sb_.add_argument("--max-legs", type=int)
    sb_.add_argument("--min-odds", type=float); sb_.add_argument("--per-game", type=int, default=7)
    sb_.add_argument("--top", type=int, default=5); sb_.add_argument("--all-games", action="store_true")
    sb_.add_argument("--min-ev", type=float, default=0.10, help="same TAKE bar as `sgp` (default +10%%)")
    sb_.add_argument("--bankroll", type=float); sb_.add_argument("--profile")
    vg_ = sub.add_parser("vig"); vg_.add_argument("--props", nargs="+", required=True); vg_.add_argument("--out", default="vig_obs.csv")
    fr = sub.add_parser("frames"); fr.add_argument("--video", required=True); fr.add_argument("--out", default="frames")
    fr.add_argument("--every", type=float, default=1.0); fr.add_argument("--diff", type=float, default=6.0)
    fr.add_argument("--max", type=int, default=60)
    tk = sub.add_parser("takeif"); tk.add_argument("--all", default="bets_all.csv"); tk.add_argument("--calib")
    tk.add_argument("--bet", action="append", required=True, help='"Player;market;Side;line;odds" (repeat)')
    for name in ("board", "check"):
        sub.choices[name].add_argument("--games", help="CSV away,home,home_spread,total")
    sub.choices["board"].add_argument("--team"); sub.choices["board"].add_argument("--player")
    sub.choices["board"].add_argument("--props")
    sub.choices["slate"].add_argument("--src", required=True); sub.choices["slate"].add_argument("--top", type=int, default=30)
    sub.choices["lines"].add_argument("--src", action="append", required=True); sub.choices["lines"].add_argument("--out", default="props.csv")
    sub.choices["check"].add_argument("--props", action="append", required=True)
    sub.choices["check"].add_argument("--out", default="bets.csv")
    sub.choices["check"].add_argument("--log", help="pick log (pulled from the sheet): units already staked tonight count toward the cap")
    sub.choices["check"].add_argument("--ignore-tip", action="store_true", help="evaluate games that already started")
    sub.choices["log"].add_argument("--placed", required=True); sub.choices["log"].add_argument("--log", default="nba_pick_log.csv")
    sub.choices["log"].add_argument("--all", default="bets_all.csv")
    sub.choices["log"].add_argument("--slip", default="bets.csv", help="tonight's slip from check (marks VPE picks)")
    sub.choices["log"].add_argument("--parlays", default="vpe_parlays.json", help="parlays VPE built (from parlay)")
    sub.choices["grade"].add_argument("--log", default="nba_pick_log.csv"); sub.choices["grade"].add_argument("--actuals")
    sub.choices["backtest"].add_argument("--start", required=True); sub.choices["backtest"].add_argument("--end", required=True)
    sub.choices["backtest"].add_argument("--step", type=int, default=1); sub.choices["backtest"].add_argument("--off", default="")
    sub.choices["backtest"].add_argument("--save"); sub.choices["backtest"].add_argument("--odds")
    sub.choices["parlay"].add_argument("--all", default="bets_all.csv")
    sub.choices["parlay"].add_argument("--bets", default="bets.csv")
    sub.choices["parlay"].add_argument("--stake", type=float, help="$ per parlay (default 5%% of bankroll)")
    sub.choices["parlay"].add_argument("--exclude", help='players to leave out "A;B" (e.g. DK price failed take-if)')
    sub.choices["parlay"].add_argument("--slip-out", default="slip.csv")
    sub.choices["parlay"].add_argument("--parlays-out", default="vpe_parlays.json")
    sub.choices["parlay"].add_argument("--boost", type=float, help="profit boost %% (gets its own BOOST parlay)")
    sub.choices["parlay"].add_argument("--boosts", help='several boosts in one run, e.g. "fd30:3leg:+100:win25,fd25,dk30"')
    sub.choices["parlay"].add_argument("--boost-book", default="draftkings", help="draftkings|fanduel (where the boost is)")
    sub.choices["log"].add_argument("--sheet-out", default="sheet_new.json", help="new rows for the Pick Log sheet")
    for name in ("check", "parlay", "log"):
        sub.choices[name].add_argument("--calib", help="Calib tab JSON (get_values): learned model weight")
    sub.choices["check"].add_argument("--shadow-out", default="shadow_new.json")
    sub.choices["check"].add_argument("--shadow-saved", help="get_values JSON of the Shadow tab's last rows (A:D): skip lines already saved (same-slate rerun)")
    sub.choices["log"].add_argument("--dry-run", action="store_true", help="rate the bets only, log nothing")
    sub.choices["parlay"].add_argument("--boost-min-leg-edge", type=float,
        help="v2.24: every boost leg needs at least this edge, e.g. 0.01 (default none: weakest-leg tilt on close calls)")
    for k, t in (("--boost-min-legs", int), ("--boost-max-legs", int), ("--boost-min-odds", float), ("--boost-max-win", float)):
        sub.choices["parlay"].add_argument(k, type=t)
    ss = sub.add_parser("status")
    ss.add_argument("--date"); ss.add_argument("--props", default="props.csv"); ss.add_argument("--games", default="games.csv")
    ss.add_argument("--injuries", help='inline list like check ("A:Out;B:Questionable") or a file; overrides --injuries-file')
    ss.add_argument("--injuries-file", default="inj.txt"); ss.add_argument("--calib", default="calib_tab.json")
    ss.add_argument("--shadow", default="shadow_new.json"); ss.add_argument("--ignore-tip", action="store_true")
    ss.add_argument("--v1-before", type=int); ss.add_argument("--v1-after", type=int)
    mn = sub.add_parser("menu"); mn.add_argument("--date"); mn.add_argument("--all", default="bets_all.csv")
    mn.add_argument("--bets", default="bets.csv"); mn.add_argument("--parlays", default="vpe_parlays.json")
    mn.add_argument("--games", default="games.csv"); mn.add_argument("--injuries", default="")
    mn.add_argument("--calib", default="calib_tab.json")
    nf_ = sub.add_parser("newsflash"); nf_.add_argument("--all", default="bets_all.csv"); nf_.add_argument("--prev"); nf_.add_argument("--runs", default="runs"); nf_.add_argument("--min-move", type=float, default=0.03); nf_.add_argument("--min-edge", type=float, default=0.0); nf_.add_argument("--top", type=int, default=12)
    sbp_ = sub.add_parser("scoreboard"); sbp_.add_argument("--log", default="nba_pick_log.csv"); sbp_.add_argument("--shadow", nargs="*"); sbp_.add_argument("--calib", default="calib_tab.json"); sbp_.add_argument("--since"); sbp_.add_argument("--since-week")
    bgp_ = sub.add_parser("bookgap"); bgp_.add_argument("--a", required=True); bgp_.add_argument("--b", required=True)
    bgp_.add_argument("--book-a", default="fanduel"); bgp_.add_argument("--book-b", default="draftkings")
    bgp_.add_argument("--games-a"); bgp_.add_argument("--games-b"); bgp_.add_argument("--fair"); bgp_.add_argument("--all", default="bets_all.csv"); bgp_.add_argument("--date")
    bgp_.add_argument("--out", default="bookgap.csv"); bgp_.add_argument("--sheet-out", default="bookgap_new.json"); bgp_.add_argument("--slip", default="slip.csv")
    sp2_ = sub.add_parser("sharepdf"); sp2_.add_argument("--all", default="bets_all.csv"); sp2_.add_argument("--bets", default="bets.csv"); sp2_.add_argument("--leans", default="leans.csv")
    sp2_.add_argument("--parlays", default="vpe_parlays.json"); sp2_.add_argument("--gaps", default="bookgap.csv"); sp2_.add_argument("--links", default="links.csv")
    sp2_.add_argument("--out", default="VPE_day_card.pdf"); sp2_.add_argument("--title"); sp2_.add_argument("--asof")
    sp2_.add_argument("--calib"); sp2_.add_argument("--source"); sp2_.add_argument("--status-file", default="status.txt"); sp2_.add_argument("--skip", default="")
    lk_ = sub.add_parser("links"); lk_.add_argument("--links", default="links.csv"); lk_.add_argument("--bets", default="bets.csv"); lk_.add_argument("--leans", default="leans.csv")
    lk_.add_argument("--parlays", default="vpe_parlays.json"); lk_.add_argument("--gaps", default="bookgap.csv")
    ld_ = sub.add_parser("linediff"); ld_.add_argument("--a", required=True, help="earlier props CSV"); ld_.add_argument("--b", required=True, help="later props CSV"); ld_.add_argument("--log"); ld_.add_argument("--date")
    ex2_ = sub.add_parser("exposure"); ex2_.add_argument("--log", default="nba_pick_log.csv"); ex2_.add_argument("--date"); ex2_.add_argument("--week")
    sw_ = sub.add_parser("sweep"); sw_.add_argument("--all", default="bets_all.csv"); sw_.add_argument("--bets", default="bets.csv")
    sw_.add_argument("--parlays", default="vpe_parlays.json"); sw_.add_argument("--top", type=int, default=10)
    sw_.add_argument("--heavy", action="store_true", help="final check: every player with a qualifying line")
    xc_ = sub.add_parser("crosscheck"); xc_.add_argument("--rec", required=True, help="props.csv from the recording")
    xc_.add_argument("--web", required=True, help="props_web.csv from `lines` (free pages)"); xc_.add_argument("--out", default="linecheck_new.json")
    xc_.add_argument("--slate"); xc_.add_argument("--week")
    xc_.add_argument("--line-gap", type=float, default=1.0); xc_.add_argument("--same-book-gap", type=float, default=0.04); xc_.add_argument("--better-gap", type=float, default=0.03)
    cl_ = sub.add_parser("close"); cl_.add_argument("--no-log", action="store_true"); cl_.add_argument("--log", default="nba_pick_log.csv"); cl_.add_argument("--props", nargs="+")
    cl_.add_argument("--date")
    cl_.add_argument("--snap", help="'Close NBA' tab JSON: closing lines saved on the night (for bets logged later)")
    cl_.add_argument("--snap-out", help="write tonight's closing lines as rows for the 'Close NBA' tab")
    cl_.add_argument("--status-out", help="write one status row for the 'Close log' tab")
    cl_.add_argument("--fetch-status", help="fetch_status.json from fetchpages: failed pages + errors go into the Close log note")
    cl_.add_argument("--fetch-note", help="free text for the Close log note (e.g. which web-fetch was refused and the error)")
    fp_ = sub.add_parser("fetchpages"); fp_.add_argument("--out", default="lines_close"); fp_.add_argument("--status", default="fetch_status.json")
    fp_.add_argument("--markets", nargs="+")
    ln = sub.add_parser("learn"); ln.add_argument("--log", help="pick log CSV: enables the 25-bet retune"); ln.add_argument("--tune-log", help="get_values JSON of the Tune log tab"); ln.add_argument("--tune-out", default="tune_new.json"); ln.add_argument("--json", nargs="+", required=True, help="Shadow window: one or more get_values JSON files, oldest first")
    ln.add_argument("--calib", default="calib_tab.json"); ln.add_argument("--learned", default="learned_tab.json")
    ln.add_argument("--out", default="learned_new.json")
    sw = sub.add_parser("shadow")
    sw.add_argument("--json", required=True, help="get_values JSON of the Shadow tab window")
    sw.add_argument("--first-row", type=int, required=True, help="sheet row of the first row in --json")
    sw.add_argument("--key", required=True, help="NBA: date YYYY-MM-DD | NFL: SEASON-Wweek e.g. 2026-W7")
    sw.add_argument("--calib", default="calib_tab.json"); sw.add_argument("--out", default="shadow_grade.json")
    sw.add_argument("--calib-out", default="calib_all.json")
    sh = sub.add_parser("sheet")
    sh.add_argument("--mode", choices=["pull", "push"], required=True)
    sh.add_argument("--json", default="sheet_tab.json"); sh.add_argument("--log", required=True)
    sh.add_argument("--out", default="sheet_all.json")
    rp_ = sub.add_parser("reprice")
    rp_.add_argument("--all", default="bets_all.csv"); rp_.add_argument("--calib")
    rp_.add_argument("--bet", action="append", required=True, help='"Player;market;Side;line;odds" (repeat)')
    rp_.add_argument("--parlay", action="store_true", help="also rate the parlay these legs make")
    rp_.add_argument("--parlay-odds", help="the app's total parlay price (default: legs multiplied)")
    rp_.add_argument("--boost", type=float); rp_.add_argument("--max-win", type=float); rp_.add_argument("--stake", type=float)
    ex_ = sub.add_parser("explain")
    for k, v in (("--all", "bets_all.csv"), ("--bets", "bets.csv"), ("--parlays", "vpe_parlays.json"), ("--leans", "leans.csv"),
                 ("--slip", "slip.csv"), ("--calib", None), ("--player", None), ("--market", None), ("--side", None),
                 ("--line", None)):
        ex_.add_argument(k, default=v)
    f = sub.add_parser("final")
    f.add_argument("--no-log", action="store_true", help="recheck only: do not write the pick log")
    f.add_argument("--boosts")
    for k, v in (("--slip", "slip.csv"), ("--all", "bets_all.csv"), ("--props", "props_close.csv"), ("--log", None),
                 ("--injuries", None), ("--exclude", None), ("--bets", "bets.csv"), ("--slip-out", "slip.csv"),
                 ("--close-out", "closing.csv"), ("--boost-book", "draftkings"), ("--calib", None)):
        f.add_argument(k, default=v)
    for k in ("--bankroll", "--stake", "--boost", "--boost-min-odds", "--boost-max-win"):
        f.add_argument(k, type=float)
    f.add_argument("--boost-min-legs", type=int); f.add_argument("--boost-max-legs", type=int)
    f.add_argument("--profile"); f.add_argument("--week", type=int); f.add_argument("--season", type=int); f.add_argument("--date")
    a = ap.parse_args()
    try:  # v2.19: parquet needs pyarrow; install it instead of stalling the run (11/14)
        import pyarrow  # noqa: F401
    except ImportError:
        import subprocess
        print("SETUP: pyarrow missing -> pip install pyarrow")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "pyarrow"], check=False)
    {"board": cmd_board, "slate": cmd_slate, "lines": cmd_lines, "check": cmd_check, "log": cmd_log,
     "grade": cmd_grade, "backtest": cmd_backtest, "parlay": cmd_parlay, "final": cmd_final, "sheet": cmd_sheet, "shadow": cmd_shadow, "learn": cmd_learn, "menu": cmd_menu, "sweep": cmd_sweep, "exposure": cmd_exposure, "linediff": cmd_linediff, "scoreboard": cmd_scoreboard, "newsflash": cmd_newsflash, "close": cmd_close, "status": cmd_status, "styles": cmd_styles,
     "takeif": cmd_takeif, "frames": cmd_frames, "games": cmd_games, "bookgap": cmd_bookgap, "links": cmd_links, "sharepdf": cmd_sharepdf, "gamesgrade": cmd_gamesgrade, "sgp": cmd_sgp, "sgpbuild": cmd_sgpbuild, "vig": cmd_vig, "explain": cmd_explain, "reprice": cmd_reprice, "crosscheck": cmd_crosscheck, "fetchpages": cmd_fetchpages}[a.cmd](a)
