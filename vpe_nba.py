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
VERSION = "NBA v2.9.0"

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
            note = (f"OUT: {', '.join(names)} | " if names else "") + \
                   f"{m2[pid] - P.at[pid, 'min']:+.1f} min, usage x{mult['fga']:.2f}, pts x{mult['pts']:.2f}"
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
    # recent role: last 3 games' minutes pull the projection 40% of the way (backtest 2025-26: minutes MAE 5.00 -> 4.69)
    if flags.get("role", True):
        ml3 = state.extra["min_l3"].reindex(sl.index) if "min_l3" in state.extra else pd.Series(np.nan, index=sl.index)
        mins = mins + m["role_weight"] * (ml3.fillna(sl["min"]) - sl["min"])
    # real games without tonight's biggest absence: blend his actual minutes in those games (weight n / (n + 32))
    if flags.get("wo", True) and "ws_n" in sl:
        n_ = sl["ws_n"].fillna(0).to_numpy()
        ok_ = n_ >= m["wo_min_games"]
        k_ = m["wo_min_k"]
        mins = pd.Series(np.where(ok_, (n_ * sl["ws_min"].fillna(0).to_numpy() + k_ * mins.to_numpy()) / (n_ + k_), mins.to_numpy()), index=sl.index)
    b2b = sl["b2b"].fillna(False).astype(bool)
    if flags.get("b2b", True):
        mins = mins * np.where(b2b, 1 + (sit["b2b_min"] - 1) * stren, 1.0)
    if flags.get("blowout", True):
        bf = [blowout_factor(sp, mn >= 26, cfg) for sp, mn in zip(sl["spread"], sl["min"])]
        mins = mins * (1 + (np.array(bf) - 1) * stren)
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
    sl["proj_min"] = np.minimum(mins * sl["nt_min"], m["max_minutes"])
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
        ef = np.power(env, tcfg["elasticity"].get(s, 0.0)) if flags.get("total", True) else 1.0
        sl["af_" + s] = af
        if flags.get("dvp", True) and len(state.dvp):
            idx = pd.MultiIndex.from_arrays([sl["opp"], sl["pos"]])
            dv = state.dvp[s].reindex(idx).fillna(1.0).to_numpy()
        else:
            dv = 1.0
        rate = P["r_" + s].reindex(sl.index).to_numpy() * sl["u_" + s].to_numpy()
        sl["mu_" + s] = sl["proj_min"].to_numpy() * rate * dv * pace_f * home_f * b2b_r * sl["nt_rate"].to_numpy() * af * ef
        sl["dvp_" + s] = dv
    sl["pace_f"] = pace_f
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
        po = (1 - prob if side == "Under" else prob) + PARLAY["model_weight"] * (po - po0)  # blended, moved with the line
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
        pm_over = cal_prob(prob_over(mu, var, mk, line))
        oo, uo = r["over_odds"], r["under_odds"]
        if pd.notna(oo) and pd.notna(uo):
            fair_over, _ = no_vig(oo, uo)
        else:
            # one-sided price (alt line / milestone like "6+ assists +250"): books charge far more than 7% there, so
            # read the market's chance off this player's main two-sided line (market-implied mean) instead
            a_, b_ = cfg["model"]["var"][mk]
            sdm = math.sqrt(max(var, 1e-9) / max(a_ * mu + b_ * mu * mu, 1e-9))
            fair_over = anchored_market_over(mk, line, mains.get((pid, mk)), cfg, sdm)
            if fair_over is None:
                one = oo if pd.notna(oo) else uo
                if -200 <= one <= 150:  # near-even single side (split lines): ordinary vig, old estimate is fine
                    fair = one_sided_fair(one)
                    fair_over = fair if pd.notna(oo) else 1 - fair
                else:
                    skipped.append((r["player"], mk, f"one-sided {int(one):+d} with no main line to price it against"))
                    continue
        pb_over = blend_leg(pm_over, fair_over, mk, "Over")  # win chance: learned model/market mix
        for side, odds, prob, pm in (("Over", oo, pb_over, pm_over), ("Under", uo, 1 - pb_over, 1 - pm_over)):
            if pd.isna(odds):
                continue
            edge = prob - implied_prob(odds)
            tier = _tier(edge, prob, cfg)
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
            mo, ll = _take_if(mu, var, mk, line, side, prob, shrink, cfg)
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
                         "returning": bool(p.get("returning", False))})
    ev = pd.DataFrame(rows)
    if len(ev):
        ev["market_gap"] = (ev["model_prob"] - ev["market_prob"]).abs().round(3)
    return ev, pd.DataFrame(skipped, columns=["player", "market", "reason"])

def held_mask(ev, cfg):
    m = cfg["model"]
    return ev["tier"].notna() & ((ev["market_gap"] > m["max_market_gap"]) | (ev["base_min"] < m["min_minutes_to_bet"]))

def used_units(log_path, mask_fn):
    """Units already staked (straights + parlays, not legs) in the pick log for this week/night, so a second run
    (e.g. Thursday game, then Sunday slate) shares the cap instead of resetting it."""
    if not log_path or not os.path.exists(log_path):
        return 0.0
    d = pd.read_csv(log_path, dtype=str)
    if d.empty:
        return 0.0
    m = mask_fn(d) & (d.get("bet_type", pd.Series("straight", index=d.index)).fillna("straight") != "leg")
    return round(float(pd.to_numeric(d.loc[m, "units"], errors="coerce").fillna(0).sum()), 2)

def apply_caps(ev, cfg):
    if ev.empty:
        return ev
    bets = ev[ev["tier"].notna() & ~held_mask(ev, cfg)].copy()
    bets["score"] = bets["ev_per_unit"] * bets["units"]
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
            "usage_beta": {"default": 0.5}, "wo_min_games": 3, "max_minutes": 40,
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
            "total": {"cap": 0.15, "elasticity": {"pts": 0.85, "reb": 1.25, "ast": 0.5, "fg3m": 1.25, "stl": 1.0, "blk": 1.5, "tov": 0.0}},
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
            "prob_calibration_shrink": 0.8, "max_market_gap": 0.15, "min_minutes_to_bet": 20,
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
  "one_sided_hold": {"short": 0.07, "mid": 0.07, "long": 0.15, "short_to": -200, "long_from": 150},
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
    cfg["used_units"] = used_units(a.log, lambda d: d.get("date", pd.Series("", index=d.index)).astype(str).str.lstrip("'") == str(date.date()))
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
                     "model_prob", "blend_prob", "edge", "units", "min_odds", "line_limit"]].to_string(index=False))
        print("\n" + picks[["player", "market", "dvp", "pace_f", "b2b", "spread", "new_team", "usage_note"]].to_string(index=False))
    else:
        print("none")
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
        ev.assign(date=str(date.date())).to_csv(a.out.replace(".csv", "_all.csv"), index=False)
        # run-stamped copy: a later rerun overwrites bets_all.csv, but `log` still finds bets from this run (10/26: 4 bets
        # from a replaced 6:00 slate were logged "unscored")
        os.makedirs("runs", exist_ok=True)
        ev.assign(date=str(date.date())).to_csv(f"runs/bets_all_{date.date()}_{pd.Timestamp.now(tz='America/New_York'):%H%M%S}.csv", index=False)
        print(f"\nsaved bets -> {a.out} (all lines -> {a.out.replace('.csv', '_all.csv')})")
    write_shadow(ev, str(date.date()), a.shadow_out)

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

def boosted_american(odds, boost_pct):
    """DraftKings profit boost: winnings x (1 + boost%). Returns the effective American odds."""
    if not boost_pct:
        return int(odds)
    return int(decimal_to_american(1 + (american_to_decimal(odds) - 1) * (1 + boost_pct / 100)))

def slip_keys(path):
    """(player, market, side) of every bet on the slip the last `check` produced; None if no slip was saved."""
    if not path or not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    kinds = s["kind"] if "kind" in s else ["pick"] * len(s)
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
        side, line, odds = str(b["side"]).title(), float(b["line"]), _odds(b["odds"])
        kind = None if slip is None else slip.get((norm_name(b["player"]), mk, side))
        sug = None if slip is None else kind is not None
        s, why = score(b["player"], mk, side, line, odds if odds is not None else -110) if mk else (None, "unknown market")
        note = "" if pd.isna(b["note"]) else str(b["note"])
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
            mo = s.get("min_odds")
            if mo is not None and pd.notna(mo) and american_to_decimal(odds) < american_to_decimal(mo) - 1e-9:
                v = f"SKIP rule broken: {int(odds):+d} worse than take-if {int(mo):+d}"
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
        # a placed parlay is a VPE pick only if it is EXACTLY a parlay VPE built tonight (same legs, nothing added/removed)
        lk = frozenset((norm_name(o[0]["player"]), o[0]["market"], o[0]["side"]) for o in out)
        vpe_par = next((v for v in VPE_PARLAYS if frozenset((norm_name(x[0]), x[1], x[2]) for x in v["legs"]) == lk), None)
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
                r.update({"tier": "sgp_correlated", "vpe_verdict": "same-game parlay: legs correlated, EV not rated "
                          "(use the sgp command before placing)"})
            elif any(x is None for x in probs):
                r.update({"tier": "unscored", "vpe_verdict": "unscored (a leg has no projection)"})
            else:
                prob = float(np.prod(probs))
                te, v, eb, ee = rate_pick(prob, podds, eff, cfg, r["suggested"] in ("yes", "lean") or (None if r["suggested"] == "unknown" else False))
                if r["suggested"] == "lean":
                    v = "VPE lean parlay (best available, no edge)" if ee <= 0 else "VPE lean parlay (+EV at the price taken)"
                if v == "small +EV, below tiers":
                    v = "+EV parlay" if eb > 0 else "+EV only with boost"
                r.update({"model_prob": round(prob, 3), "blend_prob": round(prob, 3), "tier": te or "off_model",
                          "source": "engine" if te else "off_model", "vpe_verdict": v, "edge_base": eb, "edge": ee,
                          "ev_per_unit": round(ev_per_unit(prob, eff), 3)})
        rows.append(r)
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
          "min_prob": 0.53, "max_gap": 0.15,          # model vs market gap: bigger = model probably missing news
          "min_odds": -200, "max_odds": 150, "min_minutes": 20, "min_minutes_ratio": 0.85,
          "books": ["draftkings", "fanduel"],
          "haircut": 0.05,           # price from another book -> assume ours pays 0.05 less in decimal odds (~6-8 cents)
          "junk_lines": {"rush_yds": 10.5, "rec_yds": 10.5, "receptions": 1.5, "rush_att": 3.5},  # line below -> skip
          "max_under_share": 1.0,    # v2.5.2: no Under cap (it guarded against the v2.1 home-team bug; the learned market gates now police each market x side)
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
LEAN = {"min_edge": -0.04,   # leans may be up to 4 pts below break-even (anything worse is listed as a pass)
        "min_prob": 0.50, "straights": 3, "units": 0.5, "parlay_target": 4.0, "parlay_max_legs": 3}

def lean_pool(allp, exclude=()):
    return parlay_pool(allp, exclude, min_edge=LEAN["min_edge"], min_prob=LEAN["min_prob"])

def lean_straights(allp, exclude, unit_usd):
    """Top straight bets by edge (one per game) when the staked slip is empty."""
    d = lean_pool(allp, exclude)
    if d.empty:
        return d
    d = d.sort_values(["leg_edge", "leg_prob"], ascending=False).drop_duplicates("game_id").head(LEAN["straights"]).copy()
    d["bet_odds"] = [decimal_to_american(x) for x in d["best_dec"]]
    d["units"] = LEAN["units"]
    d["stake_usd"] = round(LEAN["units"] * unit_usd, 2)
    d["cost_usd"] = (d["leg_edge"] * d["stake_usd"]).round(2)
    return d

def lean_parlay(allp, exclude):
    """Least-bad ~4x parlay (highest EV, one leg per game) when no +EV parlay exists."""
    pool = lean_pool(allp, exclude)
    if pool.empty:
        return None
    cand = pool.sort_values("leg_edge", ascending=False).head(PARLAY["pool"]).to_dict("records")
    best = None
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
                if best is None or ev > best[0] + 1e-9 or (abs(ev - best[0]) <= 1e-9 and p > best[1]):
                    best = (ev, p, dec, combo, bk)
    return best

def parlay_pool(allp, exclude=(), min_edge=None, min_prob=None):
    """Legs a parlay may use (one per player), ranked by leg prob. #1-3 Gold, #4-6 Silver."""
    d = allp.copy()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
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
    d["leg_edge"] = (d["leg_prob"] - 1 / d["best_dec"]).round(3)
    d = d[(d["leg_edge"] >= (PARLAY["min_edge"] if min_edge is None else min_edge))
          & (d["leg_prob"] >= (PARLAY["min_prob"] if min_prob is None else min_prob))]
    # take-if: worst odds (at either book) at which this leg still clears min_edge
    d["take_min"] = [int(decimal_to_american(1 / (p - PARLAY["min_edge"]))) for p in d["leg_prob"]]
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
            if under_share is not None and sum(c["side"] == "Under" for c in combo) > math.floor(under_share * n + 1e-9):
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
    set_calib(getattr(a, "calib", None), SPORT)
    allp = pd.read_csv(a.all)
    pool = parlay_pool(allp, [x for x in (a.exclude or "").split(";") if x.strip()])
    bank = a.bankroll or CONFIG["bankroll"]
    stake = a.stake if a.stake else max(1.0, round(bank * PARLAY["stake_pct"] * 2) / 2)
    boost, bbook = (a.boost or 0.0), (a.boost_book or "draftkings").lower().replace(" ", "")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    cols = [c for c in ("rank", "medal", "player", "team", "opp", "market", "side", "line", "odds", "src_book",
                        "price_dk", "price_fd", "bet_at", "leg_prob", "model_prob", "market_prob", "leg_edge",
                        "take_min", "line_limit") if c in pool]
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
        if heavy and sum(c["side"] == "Under" for c in heavy[3]) > math.floor(PARLAY["max_under_share"] * len(heavy[3]) + 1e-9):
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
        print(f"  -> build it at {BOOK_NAME[bk]}; if its parlay odds (before boost) are below {decimal_to_american(need):+d}"
              f" or a leg is worse than take_min, rerun with --exclude")
        keys += [(c["player"], c["market"], c["side"]) for c in combo]
        built.append({"name": name, "target": target, "book": bk, "odds": decimal_to_american(dec), "hit": round(p, 4),
                      "stake_usd": stake, "legs": [[c["player"], c["market"], c["side"], float(c["line"])] for c in combo]})
    bp = print_boost(a, allp, stake, cols, keys) if boost else None
    if bp:
        built.append(bp)
    excl = [x for x in (a.exclude or "").split(";") if x.strip()]
    lean_keys = [tuple(l[:3]) for l in bp["legs"]] if bp and bp.get("lean") else []
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
                      f"{r['units']:g}u (${r['stake_usd']:.2f}) | take only at {int(r['bet_odds']):+d} or better")
            lean_keys += list(zip(ls["player"], ls["market"], ls["side"]))
            lean_straight = set(zip(ls["player"], ls["market"], ls["side"]))
        else:
            print(f"  none: no line within {-LEAN['min_edge']:.0%} of break-even")
        ls.to_csv(getattr(a, "leans_out", None) or "leans.csv", index=False)
    elif os.path.exists(getattr(a, "leans_out", None) or "leans.csv"):
        os.remove(getattr(a, "leans_out", None) or "leans.csv")
    # ONE-PARLAY RULE (10/21-10/26: parlays were the whole loss, -$15.14 of -$15.89): recommend ONE parlay a night:
    # the BOOST parlay if there is one (+EV at market odds alone), else the biggest VPE could build (~10x), else ~4x.
    real = [b for b in built if not b.get("lean")]
    if built:
        # boost parlay (real, then lean) > biggest real parlay > lean parlay
        rec = (bp if bp and not bp.get("lean") else None) or (max(real, key=lambda b: b["target"]) if real else None) \
            or bp or built[-1]
        rec["recommended"] = True
        tag = (" - LEAN: no edge, the least-bad play with what we're given; logged as a VPE lean"
               if rec.get("lean") else "")
        print(f"\nONE-PARLAY RULE: place ONLY the {rec['name']} parlay at {BOOK_NAME[rec['book']]} "
              f"({rec['odds']:+d}, hit {rec['hit']:.1%}, ${rec['stake_usd']:.2f}){tag}. Any other parlay above = reference only.")
    else:
        print("\nONE-PARLAY RULE: nothing within reach tonight (no legs close to break-even) -> place none.")
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
    pd.concat([sk, lk], ignore_index=True).to_csv(a.slip_out, index=False)
    print(f"\nslip of every VPE leg/pick -> {a.slip_out} (use with log --slip and final --slip)")
    print("NOTE: legs treated as independent. Hit chances lean on the market on purpose; the model only tilts them.")


# ---------- learning loop: every scored line is graded the next day ("shadow"), the model weight is learned ----------
# One night of real lines (NBA 10/21/26, 619 line-sides) showed the raw model was overconfident: tiered picks said 63%
# and hit 48%; parlay-grade legs said 59% and hit 48%; the no-vig market alone scored best. So the model's weight is
# not fixed by hand any more: it starts at 40% and is re-fit from every graded line (shrunk toward 40% until
# ~1,500 lines are in), and each market's Over rate is corrected if it keeps landing off what VPE predicted.
CALIB_W = [round(i * 0.1, 1) for i in range(11)]
CALIB_COLS = ["sport", "market", "n", "sum_over", "sum_pm", "sum_pk"] + [f"ll_{int(round(x * 100)):03d}" for x in CALIB_W]
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
    for r in rows:
        n = _f(r, "n")
        pred = (w * _f(r, "sum_pm") + (1 - w) * _f(r, "sum_pk")) / n
        b = (_f(r, "sum_over") / n - pred) * n / (n + CALIB["n0_bias"])
        bias[str(r["market"])] = float(np.clip(b, -CALIB["bias_cap"], CALIB["bias_cap"]))
    return round(w, 3), bias, int(n_tot)


def set_calib(path, sport):
    w, bias, n = load_calib(path, sport)
    PARLAY["model_weight"], PARLAY["side_bias"] = w, bias
    b = ", ".join(f"{k} Over {v:+.1%}" for k, v in sorted(bias.items()) if abs(v) >= 0.005) or "none yet"
    src = f"{n} graded lines" if n else "no graded lines yet (prior)"
    print(f"CALIB: {src} -> win chance = {w:.0%} model + {1 - w:.0%} market | Over-rate corrections: {b}")
    set_learned(os.path.join(os.path.dirname(path) or ".", "learned_tab.json") if path else None, sport)
    return w, bias


def blend_leg(model_p, market_p, market=None, side=None):
    """VPE win chance = learned mix of model and no-vig market, + the market's learned Over-rate correction."""
    w = PARLAY["model_weight"]
    p = w * model_p + (1 - w) * market_p
    b = PARLAY.get("side_bias") or {}
    if market is not None and b:
        if isinstance(market, pd.Series):
            p = p + pd.Series(market).map(b).fillna(0.0).to_numpy() * np.where(np.asarray(side) == "Over", 1.0, -1.0)
        else:
            p = p + b.get(market, 0.0) * (1.0 if side == "Over" else -1.0)
    if LEARNED["tail"] or LEARNED["gate"]:
        p = _learn_adjust(p, market_p, market, side)
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
         "tail_bins": [0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0], "bet_edge": 0.025, "gate_n": 150}
LEARNED = {"tail": None, "gate": {}}
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
    LEARNED["tail"], LEARNED["gate"] = None, {}
    xs, ds, gates = [], [], {}
    for r in _learned_rows(path):
        if str(r.get("sport", "")).upper() != sport:
            continue
        if r["kind"] == "tail":
            xs.append(float(r["key"])); ds.append(float(r["value"]))
        elif r["kind"] == "gate" and str(r["value"]) in ("1", "2"):
            gates[str(r["key"])] = int(r["value"])
    if xs:
        o = np.argsort(xs)
        LEARNED["tail"] = ([0.5, LEARN["tail_bins"][0]] + [xs[i] for i in o], [0.0, 0.0] + [float(np.clip(ds[i], -LEARN["tail_cap"], LEARN["tail_cap"])) for i in o])
    LEARNED["gate"] = gates
    if path:
        t = "none" if not xs else ", ".join(f"{x:.0%}->{d:+.1%}" for x, d in zip(LEARNED["tail"][0][2:], LEARNED["tail"][1][2:]) if abs(d) >= 0.005) or "none needed"
        g = ", ".join(f"{k} {'HALF' if v == 1 else 'PAUSED'}" for k, v in sorted(gates.items())) or "none"
        print(f"LEARNED: tail fixes {t} | market gates {g}")


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


def cmd_learn(a):
    """Shadow window + Calib tab -> Learned tab (tail fixes + market gates). Prints what changed and why."""
    import json, datetime
    set_calib(a.calib, SPORT)
    LEARNED["tail"], LEARNED["gate"] = None, {}   # learn from the base blend, never from its own fixes
    vals = [r for j in a.json for r in _values(j)]   # several 2000-row reads, oldest first
    rows = [list(r) + [""] * (len(SHADOW_COLS) - len(r)) for r in vals if r and str(r[0]) != "key"]
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
                     "oo": str(r[4]).strip(), "uo": str(r[5]).strip()})
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
        if not ok.any():
            continue
        p_side = np.where(side == "Over", q, 1 - q)
        be = np.full(len(d), np.nan)
        be[ok.to_numpy()] = [implied_prob(float(o)) for o in odds[ok]]
        y = d["over"].to_numpy() if side == "Over" else 1 - d["over"].to_numpy()
        bet = ok.to_numpy() & (p_side - be >= LEARN["bet_edge"])
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
    json.dump(out, open(a.out, "w"))
    print(f"Learned table -> {a.out}: clear 'Learned {SPORT}'!A:G, then write it at 'Learned {SPORT}'!A1. Save it as learned_tab.json"
          f" next to calib_tab.json before check/parlay/final/takeif.")


# ---------- NEWS SWEEP (v2.6 / v15.9): who to check in the news before anything is placed ----------
# Every run: one news search per game + one per player VPE would put money on (slip, the parlay, top props, SGP legs).
# What the searches find goes into inj.txt (Out / Doubtful / Questionable / Limited) and `check` is rerun, so a minutes
# restriction mentioned in a presser but missing from the injury page still reaches the projection.
SWEEP = {"top": 10, "per_game_queries": 1}


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
    top = allp[(allp["edge"] >= 0.015) & allp["odds"].between(-200, 200) & (allp["market_gap"].fillna(0) <= 0.15)]
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
    print(f"NEWS SWEEP ({'HEAVY - final check' if a.heavy else 'every run'}): {len(why)} players in {len(games)} games")
    for g, ps in sorted(games.items()):
        print(f"\n  {g}")
        print(f'    search: "{g.replace(" vs ", " ")} injury news minutes restriction today"')
        n_q += 1
        for p in sorted(ps, key=rank):
            print(f'    - {p:<26} ({", ".join(why[p])})  search: "{p} status tonight minutes"')
            n_q += 1
    print(f"\n{n_q} searches. Any player reported out / doubtful / questionable / on a minutes limit / coming off the bench"
          " in a new role -> add to inj.txt (Out, Doubtful, Questionable, Limited) and rerun check + parlay + menu."
          " Quote one line in the header: NEWS: N searches, M changes (names) or NEWS: N searches, no changes.")


def shadow_rows(allp, key):
    """One row per line scored this run (Over view): what the model and the market said before the game."""
    out = {}
    for r in allp.to_dict("records"):
        if pd.isna(r.get("model_prob")) or pd.isna(r.get("market_prob")):
            continue
        k = (r["player"], r["market"], float(r["line"]))
        x = out.setdefault(k, {"key": key, "player": r["player"], "market": r["market"], "line": float(r["line"]),
                               "over_odds": "", "under_odds": "", "p_model": None, "p_market": None, "actual": "", "over": ""})
        o = r["side"] == "Over"
        x["over_odds" if o else "under_odds"] = int(r["odds"])
        if x["p_model"] is None or o:
            x["p_model"] = round(float(r["model_prob"] if o else 1 - r["model_prob"]), 4)
            x["p_market"] = round(float(r["market_prob"] if o else 1 - r["market_prob"]), 4)
    return pd.DataFrame(list(out.values()), columns=SHADOW_COLS)


def write_shadow(evald, key, path):
    import json
    if evald is None or evald.empty or not path:
        return
    s = shadow_rows(evald, key)
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
    rows = [list(r) + [""] * (len(SHADOW_COLS) - len(r)) for r in vals]
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
    ranked = pool.assign(_mv=pool["market_prob"] * pool[col]).sort_values(["_mv", "leg_prob"], ascending=False)
    ranked = ranked.drop_duplicates("player")
    # both sides in the candidate list (low-vig legs skew Under, and the Unders cap would block every combo)
    ranked = pd.concat([ranked[ranked["side"] == sd].head(PARLAY["pool"] // 2 + 1) for sd in ("Over", "Under")])
    cand, best = ranked.to_dict("records"), None
    for n in range(max(min_legs, 2), max_legs + 1):
        for combo in itertools.combinations(cand, n):
            if len({str(c["game_id"]) for c in combo}) < n:
                continue
            if sum(c["side"] == "Under" for c in combo) > math.floor(PARLAY["max_under_share"] * n + 1e-9):
                continue
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
            if best is None or g > best[6] + 1e-12:
                best = (p * dec_eff - 1, p, p_mkt, dec, dec_eff, combo, g)
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
            "odds": decimal_to_american(dec), "hit": round(p, 4), "lean": lean, "ev": round(ev, 3),
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
        if len(close):
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
    if a.log and os.path.exists(a.log) and a.props and os.path.exists(a.props):
        lg_ = read_log(a.log)
        cpr_ = _read_props([a.props])
        lg_, n_ = fill_closing(lg_, cpr_[cpr_["market"].notna()], str(today_et().date()), _cfg(a))
        lg_.to_csv(a.log, index=False)
        print(f"CLV: closing price written to {n_} placed bets/legs (every bet tonight, not only the slip)")
        clv_report(lg_, str(today_et().date()))
    if a.log and os.path.exists(a.log) and len(out):
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
    unit = cfg["bankroll"] * cfg["unit_pct_of_bankroll"]
    shrink = cfg["model"]["prob_calibration_shrink"]
    allp = _all_runs(a.all)

    def score(player, mk, side, line, odds):
        hit = allp[(allp["player"].map(norm_name) == norm_name(player)) & (allp["market"] == mk)] if len(allp) else allp
        if hit.empty:
            return None, "no projection saved for this player/market (run check first)"
        h = hit.iloc[0]
        po = cal_prob(prob_over(h["mean"], h["var"], mk, line))
        pr = po if side == "Over" else 1 - po
        mp = market_prob_for(allp, player, mk, side, line, odds)
        pb = blend_leg(pr, mp, mk, side)
        ex = hit[(hit["side"] == side) & (pd.to_numeric(hit["line"], errors="coerce") == float(line))]
        e0 = ex.iloc[0] if len(ex) else None
        return {"prob": pb, "leg_prob": pb, "team": h["team"], "opp": h["opp"],
                "rec_units": float(e0["units"]) if e0 is not None and pd.notna(e0.get("units")) else 0.0,
                "min_odds": e0.get("min_odds") if e0 is not None and pd.notna(e0.get("tier")) and str(e0.get("tier")) not in ("", "nan") else None}, ""

    slip = slip_keys(a.slip)
    import json as _j
    pp_ = getattr(a, "parlays", None) or "vpe_parlays.json"
    VPE_PARLAYS[:] = _j.load(open(pp_)) if os.path.exists(pp_) else []
    stamp = {"season": season_for(today_et()),
             "date": allp["date"].iloc[0] if len(allp) else str(today_et().date())}
    rows = build_log_rows(pd.read_csv(a.placed, dtype=str), score, cfg, unit, slip, stamp)
    if getattr(a, "dry_run", False):
        print_logged(rows)
        print("\nDRY RUN (rate only): nothing logged")
        return
    new = pd.DataFrame(rows).reindex(columns=LOG_COLS)
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
        "inj_beta": 0.42, "repl_gs_per_min": 0.25, "half_life": 60, "lam": 2.0, "prior_w": 6.0}
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

def cmd_games(a):
    """Rate spreads / totals / moneylines. RATE-ONLY: never part of a slip until promoted (see skill)."""
    cfg = _cfg(a)
    date, state, sched = _setup(a, cfg)
    model = team_ratings(state, date)
    g = _read_loose_csv(a.games)
    g["home"], g["away"] = g["home"].map(norm_team), g["away"].map(norm_team)
    for c in g.columns:
        if c not in ("home", "away"):
            g[c] = pd.to_numeric(g[c].astype(str).str.replace("+", "", regex=False).replace({"PK": "0", "pk": "0"}), errors="coerce")
    g = guard_games(g, sched, date)
    inj = parse_injuries(a.injuries)
    outs = build_outs(state, inj, {})
    lost = injury_shift(state, outs)
    rows, log = [], []
    for _, r in g.iterrows():
        home, away = r["home"], r["away"]
        cm, ct, m_mod, t_mod, m_mkt, t_mkt = game_center(model, home, away, r.get("home_spread"), r.get("total"), lost)
        inj_note = f"inj {GAME['inj_beta'] * (lost.get(away, 0) - lost.get(home, 0)):+.1f}" if (lost.get(home) or lost.get(away)) else ""
        def leg(name, p, odds, gap, held_at):
            odds = -110 if odds is None or pd.isna(odds) else int(odds)
            edge = p - implied_prob(odds)
            tier = _tier(edge, p, cfg)
            held = tier is not None and gap > held_at
            rows.append({"game": f"{away}@{home}", "bet": name, "odds": odds, "vpe_prob": round(p, 3), "edge": round(edge, 3),
                         "verdict": "HELD (model vs market gap)" if held else (tier or "pass"),
                         "model": f"margin {m_mod:+.1f} total {t_mod:.1f}", "market": f"margin {m_mkt:+.1f} total {t_mkt:.1f}",
                         "injuries": inj_note})
        hs = r.get("home_spread")
        if pd.notna(hs):
            ph = float(st.norm.sf(-hs, cm, GAME["sd_margin"]))
            leg(f"{home} {hs:+g}", ph, r.get("spread_home_odds"), abs(m_mod - m_mkt), GAME["held_margin"])
            leg(f"{away} {-hs:+g}", 1 - ph, r.get("spread_away_odds"), abs(m_mod - m_mkt), GAME["held_margin"])
        tot = r.get("total")
        if pd.notna(tot):
            po = float(st.norm.sf(tot, ct, GAME["sd_total"]))
            leg(f"Over {tot:g}", po, r.get("over_odds"), abs(t_mod - t_mkt), GAME["held_total"])
            leg(f"Under {tot:g}", 1 - po, r.get("under_odds"), abs(t_mod - t_mkt), GAME["held_total"])
        pw = float(st.norm.sf(0, cm, GAME["sd_margin"]))
        if pd.notna(r.get("home_ml")):
            leg(f"{home} ML", pw, r.get("home_ml"), abs(m_mod - m_mkt), GAME["held_margin"])
            leg(f"{away} ML", 1 - pw, r.get("away_ml"), abs(m_mod - m_mkt), GAME["held_margin"])
        log.append([str(date.date()), away, home, round(m_mod, 2), round(t_mod, 2),
                    "" if pd.isna(hs) else float(hs), "" if pd.isna(tot) else float(tot), inj_note, "", ""])
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"GAME MARKETS - RATE ONLY | prob = {1 - GAME['model_w']:.0%} market + {GAME['model_w']:.0%} injury-aware team ratings | "
          f"HELD = model vs market gap > {GAME['held_margin']:g} margin / {GAME['held_total']:g} total")
    print("  backtest: vs CLOSING lines the market alone is best; an edge can only exist before the line moves "
          "(the model's gap predicted the open->close move, corr 0.47). Track CLV before trusting any of this.")
    print(out.to_string(index=False) if len(out) else "no games")
    out.to_csv(a.out, index=False)
    import json
    json.dump(log, open(a.log_out, "w"))
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
        return {"kind": "total", "side": t[2].title(), "line": float(t[3]), "label": f"Game {t[2].title()} {t[3]}"}
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

def cmd_sgp(a):
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
    if len({frozenset((r["team"], r["opp"])) for _, r in proj.iterrows()}) > 1:
        sys.exit("legs are from different games - this checker is for same-game parlays")
    if len(proj):
        r0 = proj.iloc[0]; home, away = (r0["team"], r0["opp"]) if r0["home"] else (r0["opp"], r0["team"])
    else:
        tms = {l.get("team") for l in legs if l.get("team")}
        g0 = games[games["home"].isin(tms) | games["away"].isin(tms)].iloc[0]; home, away = g0["home"], g0["away"]
    gm = games[(games["home"] == home) & (games["away"] == away)] if games is not None else pd.DataFrame()
    hs = float(gm.iloc[0]["home_spread"]) if len(gm) and pd.notna(gm.iloc[0]["home_spread"]) else None
    tot = float(gm.iloc[0]["total"]) if len(gm) and pd.notna(gm.iloc[0]["total"]) else None
    cm, ct, *_ = game_center(team_ratings(state, date), home, away, hs, tot, injury_shift(state, outs))
    H = sgp_simulate(legs, proj, home, cm, ct)
    p_sim = H.mean(axis=1); lift = H.all(axis=0).mean() / max(np.prod(p_sim), 1e-9)
    probs = []
    for l, ps in zip(legs, p_sim):
        if l["kind"] == "player":
            r = proj.loc[l["pid"]]
            po = cal_prob(prob_over(float(r["mu_" + l["market"]]), float(r["var_" + l["market"]]), l["market"], l["line"]))
            p = po if l["side"] == "Over" else 1 - po
            l["model_p"] = p
            if l.get("odds"):
                o = l["odds"]
                mp = no_vig(*[float(x.replace("+", "")) for x in o.split("/")])[0] if "/" in o else one_sided_fair(o.replace("+", ""))
                p = 0.5 * p + 0.5 * mp
        else:
            p = float(ps)
        probs.append(p)
    joint = min(float(np.prod(probs)) * lift, min(probs))
    price = int(float(str(a.price).replace("+", ""))); dec = american_to_decimal(price)
    ev = joint * dec - 1
    print(f"\nSGP {away}@{home} - RATE ONLY | center: home margin {cm:+.1f}, total {ct:.1f} | {SGP_N:,} simulated games")
    for l, p in zip(legs, probs):
        extra = f" (model {l['model_p']:.1%}, 50/50 with book)" if l.get("odds") and l.get("model_p") is not None else ""
        print(f"  {l['label']:42s} VPE {p:5.1%}{extra}")
    print(f"  independent product {np.prod(probs):.2%} x correlation lift {lift:.2f} = joint {joint:.2%}")
    print(f"  book price {price:+d} (implies {1 / dec:.2%}) | fair price {decimal_to_american(1 / joint):+d} | EV {ev:+.0%}")
    print(f"  VERDICT: {'TAKE' if ev >= a.min_ev else 'SKIP'} (needs EV >= {a.min_ev:+.0%}; rate-only - correlations from last "
          "season, not yet proven against real SGP results)")
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
    pool = parlay_pool(allp)
    date, state, sched = _setup(a, cfg)
    print("\n" + "=" * 70 + "\nVPE MENU - best options tonight (ranked inside each section)\n" + "=" * 70)
    # 0. overall ranking: every scored prop on the slate with a real edge at its price (not held, not out), best first
    print("\n0) BEST PROPS OVERALL (every game, ranked by edge at the listed price; ★ = on the staked slip)")
    on_slip = set()
    if a.bets and os.path.exists(a.bets):
        bb = pd.read_csv(a.bets)
        on_slip = {(r["player"], r["market"], r["side"]) for _, r in bb.iterrows()}
    ov = allp.copy()
    for c in ("edge", "blend_prob", "odds", "market_gap"):
        ov[c] = pd.to_numeric(ov.get(c), errors="coerce")
    ok = (ov["edge"] >= MENU["min_edge"]) & ov["odds"].between(-200, 200) & (ov["market_gap"].fillna(0) <= 0.15)
    if "base_min" in ov:
        ok &= pd.to_numeric(ov["base_min"], errors="coerce").fillna(99) >= 20
    if "injury" in ov:
        ok &= ~ov["injury"].fillna("").astype(str).str.contains("Out|Doubtful", case=False)
    ov = ov[ok].sort_values("edge", ascending=False).head(MENU["top_overall"])
    if ov.empty:
        print("   none qualify")
    for i, (_, r) in enumerate(ov.iterrows(), 1):
        star = "★" if (r["player"], r["market"], r["side"]) in on_slip else " "
        tk = f" | take if odds >= {int(r['min_odds']):+d}" if pd.notna(r.get("min_odds")) else ""
        print(f"  {i:>2}. {star} {r['player']} ({r['team']} vs {r['opp']}) {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}, {r['book']})"
              f" | win {r['blend_prob']:.1%} | edge {r['edge']:+.1%}{tk}")
    # 1. straights
    print("\n1) BEST STRAIGHT BETS (the staked slip; max 2 per game)")
    if a.bets and os.path.exists(a.bets):
        b = pd.read_csv(a.bets)
        b = b[b.get("tier").notna()] if "tier" in b else b
        if len(b):
            for _, r in b.sort_values("edge", ascending=False).iterrows():
                print(f"   {r['player']} ({r['team']}) {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}) vs {r['opp']}"
                      f" | win {r['blend_prob']:.1%} | edge {r['edge']:+.1%} | {r['units']}u")
        else:
            print("   none beat the book tonight")
            lf = "leans.csv"
            if os.path.exists(lf) and len(pd.read_csv(lf)):
                print("   LEAN - best available (no edge; if you play, play these):")
                for _, r in pd.read_csv(lf).iterrows():
                    print(f"   ~ {r['player']} ({r['team']}) {r['side']} {r['line']} {r['market']} ({int(r['bet_odds']):+d} at {r['bet_at']})"
                          f" vs {r['opp']} | win {r['leg_prob']:.1%} | edge {r['leg_edge']:+.1%} | {r['units']:g}u")
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
        top = g.sort_values("leg_edge", ascending=False).head(MENU["props_per_game"])
        for _, r in top.iterrows():
            print(f"    prop: {r['player']} {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}, {r['bet_at']}) | win {r['leg_prob']:.1%}"
                  f" | edge {r['leg_edge']:+.1%} | take if odds >= {int(r['take_min']):+d}")
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
    m = (log["date"].astype(str) == str(date)) & log["closing_odds"].isna() & bt.isin(["straight", "leg"]) & log["market"].isin(list(MARKETS))
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
        d = d[d["date"].astype(str) == str(date)]
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
    log, n = fill_closing(log, close, date, cfg)
    log.to_csv(a.log, index=False)
    print(f"CLOSE {date}: closing price written to {n} bets/legs in {a.log}"
          + ("" if n else " (none matched: the closing file needs each bet player's MAIN O/U line with both prices)"))
    clv_report(log, date)


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
    if a.injuries_file and os.path.exists(a.injuries_file):
        txt = open(a.injuries_file).read().strip()
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
    gg_ = sub.add_parser("gamesgrade"); gg_.add_argument("--json", required=True); gg_.add_argument("--out", default="games_tab.json")
    sp_ = sub.add_parser("sgp"); sp_.add_argument("--date"); sp_.add_argument("--games", required=True)
    sp_.add_argument("--injuries", default=""); sp_.add_argument("--leg", action="append", required=True)
    sp_.add_argument("--price", required=True); sp_.add_argument("--min-ev", type=float, default=0.10)
    sp_.add_argument("--bankroll", type=float); sp_.add_argument("--profile")
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
    sub.choices["parlay"].add_argument("--stake", type=float, help="$ per parlay (default 5% of bankroll)")
    sub.choices["parlay"].add_argument("--exclude", help='players to leave out "A;B" (e.g. DK price failed take-if)')
    sub.choices["parlay"].add_argument("--slip-out", default="slip.csv")
    sub.choices["parlay"].add_argument("--parlays-out", default="vpe_parlays.json")
    sub.choices["parlay"].add_argument("--boost", type=float, help="profit boost %% (gets its own BOOST parlay)")
    sub.choices["parlay"].add_argument("--boost-book", default="draftkings", help="draftkings|fanduel (where the boost is)")
    sub.choices["log"].add_argument("--sheet-out", default="sheet_new.json", help="new rows for the Pick Log sheet")
    for name in ("check", "parlay", "log"):
        sub.choices[name].add_argument("--calib", help="Calib tab JSON (get_values): learned model weight")
    sub.choices["check"].add_argument("--shadow-out", default="shadow_new.json")
    sub.choices["log"].add_argument("--dry-run", action="store_true", help="rate the bets only, log nothing")
    for k, t in (("--boost-min-legs", int), ("--boost-max-legs", int), ("--boost-min-odds", float), ("--boost-max-win", float)):
        sub.choices["parlay"].add_argument(k, type=t)
    ss = sub.add_parser("status")
    ss.add_argument("--date"); ss.add_argument("--props", default="props.csv"); ss.add_argument("--games", default="games.csv")
    ss.add_argument("--injuries-file", default="inj.txt"); ss.add_argument("--calib", default="calib_tab.json")
    ss.add_argument("--shadow", default="shadow_new.json"); ss.add_argument("--ignore-tip", action="store_true")
    ss.add_argument("--v1-before", type=int); ss.add_argument("--v1-after", type=int)
    mn = sub.add_parser("menu"); mn.add_argument("--date"); mn.add_argument("--all", default="bets_all.csv")
    mn.add_argument("--bets", default="bets.csv"); mn.add_argument("--parlays", default="vpe_parlays.json")
    mn.add_argument("--games", default="games.csv"); mn.add_argument("--injuries", default="")
    mn.add_argument("--calib", default="calib_tab.json")
    sw_ = sub.add_parser("sweep"); sw_.add_argument("--all", default="bets_all.csv"); sw_.add_argument("--bets", default="bets.csv")
    sw_.add_argument("--parlays", default="vpe_parlays.json"); sw_.add_argument("--top", type=int, default=10)
    sw_.add_argument("--heavy", action="store_true", help="final check: every player with a qualifying line")
    cl_ = sub.add_parser("close"); cl_.add_argument("--log", default="nba_pick_log.csv"); cl_.add_argument("--props", nargs="+")
    cl_.add_argument("--date")
    ln = sub.add_parser("learn"); ln.add_argument("--json", nargs="+", required=True, help="Shadow window: one or more get_values JSON files, oldest first")
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
    f = sub.add_parser("final")
    for k, v in (("--slip", "slip.csv"), ("--all", "bets_all.csv"), ("--props", "props_close.csv"), ("--log", None),
                 ("--injuries", None), ("--exclude", None), ("--bets", "bets.csv"), ("--slip-out", "slip.csv"),
                 ("--close-out", "closing.csv"), ("--boost-book", "draftkings"), ("--calib", None)):
        f.add_argument(k, default=v)
    for k in ("--bankroll", "--stake", "--boost", "--boost-min-odds", "--boost-max-win"):
        f.add_argument(k, type=float)
    f.add_argument("--boost-min-legs", type=int); f.add_argument("--boost-max-legs", type=int)
    f.add_argument("--profile"); f.add_argument("--week", type=int); f.add_argument("--season", type=int); f.add_argument("--date")
    a = ap.parse_args()
    {"board": cmd_board, "slate": cmd_slate, "lines": cmd_lines, "check": cmd_check, "log": cmd_log,
     "grade": cmd_grade, "backtest": cmd_backtest, "parlay": cmd_parlay, "final": cmd_final, "sheet": cmd_sheet, "shadow": cmd_shadow, "learn": cmd_learn, "menu": cmd_menu, "sweep": cmd_sweep, "close": cmd_close, "status": cmd_status, "styles": cmd_styles,
     "takeif": cmd_takeif, "frames": cmd_frames, "games": cmd_games, "gamesgrade": cmd_gamesgrade, "sgp": cmd_sgp, "vig": cmd_vig}[a.cmd](a)
