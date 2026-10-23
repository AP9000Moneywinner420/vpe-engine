"""Verified Prop Engine - NBA, single-file in-chat edition (v1.1).
Data: free ESPN box scores + schedules that sportsdataverse publishes on GitHub (parquet files, no API key).

  python vpe_nba.py board  [--date 2026-10-20] [--team NY,PHI] [--player "A;B"] [--injuries "Name:Out[:TEAM]"] [--games games.csv]
  python vpe_nba.py slate  --src lines/points.csv [--date ...] [--top 30]     # who to pull DraftKings pages for
  python vpe_nba.py lines  --src pts=lines/points.csv --src fg3m=lines/threes.csv --out props.csv   # best-of-market rows
  python vpe_nba.py check  --props lines/dk.csv [--games games.csv] [--injuries "..."] [--bankroll 50] [--out bets.csv]
  python vpe_nba.py log    --placed placed.csv --log nba_pick_log.csv
  python vpe_nba.py grade  --log nba_pick_log.csv --date 2026-10-20 [--actuals actuals.csv]
  python vpe_nba.py backtest --start 2026-01-05 --end 2026-04-12 [--step 2]
"""
from scipy import stats as st
import io
import itertools
import math
import numpy as np
import os
import pandas as pd
import re
import time
import unicodedata
import urllib.error
import urllib.request

SPORT = "NBA"

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

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "oreb", "dreb", "fga", "fta", "fg3a"]
RAW = {"points": "pts", "rebounds": "reb", "assists": "ast", "three_point_field_goals_made": "fg3m", "steals": "stl",
       "blocks": "blk", "turnovers": "tov", "offensive_rebounds": "oreb", "defensive_rebounds": "dreb",
       "field_goals_attempted": "fga", "free_throws_attempted": "fta", "three_point_field_goals_attempted": "fg3a"}
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
    s = s.replace(".", "").replace("'", "").replace("’", "").replace("-", " ")
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
    return {"players": player_table(prev_box, m["prev_decay"], m["prev_decay"]), "tg": tg, "dvp": dvp,
            "pace": tg.groupby("team")["game_poss"].mean(), "lg_pace": tg["game_poss"].mean(),
            "pos_min": p.groupby("pos")["min"].sum(), "pos_sum": p.groupby("pos")[STATS].sum(),
            "team_tot": tg.groupby("team")[STATS].mean(), "lg_team_tot": tg[STATS].mean()}

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

def team_adjust(state, sl, outs, rosters, cfg, use_split=True):
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
        R["p"] = np.where(R.index.isin(sl.index), 1.0, av.fillna(m["unknown_play_rate"]).clip(0, 1))
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
            for s in USE_STATS:
                sl.at[pid, "u_" + s] = mult[s]
            note = (f"OUT: {', '.join(names)} | " if names else "") + \
                   f"{m2[pid] - P.at[pid, 'min']:+.1f} min, usage x{mult['fga']:.2f}, pts x{mult['pts']:.2f}"
            if use_split and big is not None:
                ws = without_split(state, pid, team, big.name)
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

def project(state, sl, outs, rosters, cfg, flags=None):
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
                     use_split=flags.get("split", True)) if flags.get("team", True) else \
        sl.assign(dmin=0.0, usage_note="", **{"u_" + s: 1.0 for s in USE_STATS})
    sit = m["situational"]
    stren = sit["strength"]
    mins = sl["min"] + sl["dmin"]
    b2b = sl["b2b"].fillna(False).astype(bool)
    if flags.get("b2b", True):
        mins = mins * np.where(b2b, 1 + (sit["b2b_min"] - 1) * stren, 1.0)
    if flags.get("blowout", True):
        bf = [blowout_factor(sp, mn >= 26, cfg) for sp, mn in zip(sl["spread"], sl["min"])]
        mins = mins * (1 + (np.array(bf) - 1) * stren)
    sl["proj_min"] = np.minimum(mins * sl["nt_min"], m["max_minutes"])
    pace = (state.pace.reindex(sl["opp"]).to_numpy() / state.lg_pace) if flags.get("pace", True) else 1.0
    pace_f = np.power(pace, m["pace_elasticity"])
    home_f = np.where(sl["home"].fillna(False).astype(bool), sit["home_rate"], sit["away_rate"]) if flags.get("home", True) else 1.0
    home_f = 1 + (np.asarray(home_f) - 1) * stren
    b2b_r = np.where(b2b, 1 + (sit["b2b_rate"] - 1) * stren, 1.0) if flags.get("b2b", True) else 1.0
    for s in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
        if flags.get("dvp", True) and len(state.dvp):
            idx = pd.MultiIndex.from_arrays([sl["opp"], sl["pos"]])
            dv = state.dvp[s].reindex(idx).fillna(1.0).to_numpy()
        else:
            dv = 1.0
        rate = P["r_" + s].reindex(sl.index).to_numpy() * sl["u_" + s].to_numpy()
        sl["mu_" + s] = sl["proj_min"].to_numpy() * rate * dv * pace_f * home_f * b2b_r * sl["nt_rate"].to_numpy()
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
    po0 = 0.5 + (prob_over(mu, var, mk, line) - 0.5) * shrink
    for _ in range(30):
        po = 0.5 + (prob_over(mu, var, mk, L) - 0.5) * shrink
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

def evaluate(props, proj, injuries, cfg):
    unit_pct, kf = cfg["unit_pct_of_bankroll"], cfg["staking"]["kelly_fraction"]
    profile = cfg["strategy_profiles"][cfg["strategy_profile"]]
    shrink = cfg["model"]["prob_calibration_shrink"]
    status = {norm_name(r["player"]): r["status"] for _, r in injuries.iterrows()} if injuries is not None else {}
    rows, skipped = [], []
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
        pm_over = 0.5 + (prob_over(mu, var, mk, line) - 0.5) * shrink
        oo, uo = r["over_odds"], r["under_odds"]
        if pd.notna(oo) and pd.notna(uo):
            fair_over, _ = no_vig(oo, uo)
        else:
            fair = implied_prob(oo if pd.notna(oo) else uo) / 1.07
            fair_over = fair if pd.notna(oo) else 1 - fair
        pb_over = blend_leg(pm_over, fair_over, mk, "Over")  # win chance: learned model/market mix
        for side, odds, prob, pm in (("Over", oo, pb_over, pm_over), ("Under", uo, 1 - pb_over, 1 - pm_over)):
            if pd.isna(odds):
                continue
            edge = prob - implied_prob(odds)
            tier = _tier(edge, prob, cfg)
            if tier == "tier_3" and not profile["allow_tier3"]:
                tier = None
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
                         "n_cur": int(p["n_cur"]), "min_odds": mo, "line_limit": ll})
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
    kept, pp, pg, total = [], {}, {}, 0.0
    for _, b in bets.iterrows():
        if pp.get(b["player"], 0) >= st_["max_props_per_player"]:
            continue
        if pg.get(b["game_id"], 0) + b["units"] > st_["max_units_per_game"] or total + b["units"] > cap_day:
            continue
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
            "prior_min_games": 3, "prior_rate_minutes": 400, "pos_shrink_minutes": 25,
            "dvp_shrink_games": 15, "dvp_prior_games": 10, "prior_trust": 0.5, "dvp_cap": [0.9, 1.1],
            "pace_prior_games": 10, "pace_elasticity": 1.0,
            "usage_min_out_minutes": 12, "usage_rep_factor": 0.85, "team_minutes": 240.7, "min_norm_alpha": 0.75,
            "bench_allowance_max": 35, "unknown_play_rate": 0.6, "team_total_prior_games": 8, "partial_beta_mult": 0.5, "partial_scale_cap": 1.0, "usage_mult_cap": 1.3,
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
            "new_team_sd_mult": 1.05, "early_sd_mult": 1.0,
            "prob_calibration_shrink": 0.8, "max_market_gap": 0.15, "min_minutes_to_bet": 20},
  "tiers": {"tier_1": {"min_edge": 0.04, "min_prob": 0.55, "max_units": 3.0},
            "tier_2": {"min_edge": 0.03, "min_prob": 0.52, "max_units": 2.0},
            "tier_3": {"min_edge": 0.025, "min_prob": 0.30, "max_units": 1.0}},
  "staking": {"kelly_fraction": 0.25, "questionable_stake_mult": 0.5, "max_props_per_player": 1, "max_units_per_game": 4},
  "parlays": {"enabled": False, "max_legs": 3, "units": 0.5},
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
    print(f"DATA: season {season - 1}-{str(season)[2:]}, slate date {date.date()}, this-season box scores: {n_games} games "
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
        mm = d["matchup"].map(parse_matchup)
        d["team"] = [x[0] for x in mm]
    else:
        d["team"] = None
    return d.reset_index(drop=True)

def _prepare(state, sched, date, props_players, injuries, games, cfg):
    """Map names -> ids, current teams, today's games, outs. Returns (proj, overrides, outs)."""
    overrides = {}
    pids = []
    for name, team in props_players:
        pid = state.find(name)
        if pid is not None:
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
    proj = project(state, sl, outs, rosters, cfg) if len(sl) else pd.DataFrame()
    return proj, overrides, outs

# ==================== commands ====================

def cmd_board(a):
    cfg = _cfg(a)
    date, state, sched = _setup(a, cfg)
    inj = parse_injuries(a.injuries)
    games = load_games_file(a.games)
    day = sched[sched["date"] == date]
    teams = set(day["home"]) | set(day["away"])
    if a.team:
        teams &= {norm_team(t) for t in a.team.split(",")}
    P = state.players
    props_players = []
    if a.props:
        pr = _read_props([a.props])
        props_players = list(zip(pr["player"], pr["team"]))
    pids = [(n, None) for n in P[(P["team"].isin(teams)) & (P["min"] >= 10)]["name"]]
    proj, _, outs = _prepare(state, sched, date, props_players + pids, inj, games, cfg)
    if proj.empty:
        print("No players with a game on this date.")
        return
    proj = proj[proj["team"].isin(teams) & ~proj.index.isin(set(outs["pid"]) if len(outs) else set())]
    if a.player:
        keys = {norm_name(x) for x in a.player.split(";")}
        proj = proj[proj["name"].map(norm_name).isin(keys)]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 80)
    cols = ["name", "team", "opp", "pos", "proj_min", "mu_pts", "mu_reb", "mu_ast", "mu_fg3m", "mu_pra", "dvp_pts",
            "pace_f", "new_team", "usage_note"]
    print(proj.sort_values(["team", "proj_min"], ascending=[True, False])[cols].round(2).to_string(index=False))

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
                      + (" -> page shows another slate, re-fetch it with ?date=" + str(date.date()) if k == 0 else ""))
        df = df[ok]
    df.to_csv(a.out, index=False)
    print(f"LINES: {len(df)} rows -> {a.out} | " + ", ".join(f"{k} {v}" for k, v in df["market"].value_counts().items()))

def cmd_check(a):
    cfg = _cfg(a)
    set_calib(getattr(a, "calib", None), SPORT)
    if a.ignore_tip:
        cfg["ignore_tip"] = True
    date, state, sched = _setup(a, cfg)
    props = _read_props(a.props)
    inj = parse_injuries(a.injuries)
    games = load_games_file(a.games)
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
    print("\n=== SKIPPED ===")
    print(skipped.drop_duplicates().to_string(index=False) if not skipped.empty else "none")
    if a.out:
        picks.assign(date=str(date.date())).to_csv(a.out, index=False)
        ev.assign(date=str(date.date())).to_csv(a.out.replace(".csv", "_all.csv"), index=False)
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
    return {(norm_name(p), canon_market(m) or m, str(sd).title()) for p, m, sd in zip(s.get("player", []),
            s.get("market", []), s.get("side", []))}

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
        sug = None if slip is None else (norm_name(b["player"]), mk, side) in slip
        s, why = score(b["player"], mk, side, line, odds if odds is not None else -110) if mk else (None, "unknown market")
        note = "" if pd.isna(b["note"]) else str(b["note"])
        r = {**stamp, "logged_at": now, "player": b["player"], "market": mk or b["market"], "side": side, "line": line,
             "odds": odds, "book": _book(b), "note": note,
             "suggested": "unknown" if sug is None else ("yes" if sug else "no")}
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
                  "boost_pct": boost or np.nan})
        if odds is None:
            r.update({"tier": "unscored", "source": "unscored: odds unreadable", "vpe_verdict": "unscored"})
            return r, None, sug, s
        eff = boosted_american(odds, boost)
        r["odds_eff"] = eff
        if not s:
            r.update({"tier": "unscored", "source": "unscored: " + why, "vpe_verdict": "unscored"})
        else:
            te, v, eb, ee = rate_pick(s["prob"], odds, eff, cfg, sug)
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
        r = {**stamp, "logged_at": now, "bet_type": "parlay", "parlay_id": f"{tag}-{p}", "market": "parlay",
             "player": " + ".join(f"{o[0]['player']} {o[0]['side']} {o[0]['line']:g} {o[0]['market']}" for o in out),
             "side": np.nan, "odds": podds, "book": _book(b0), "units": round(stake / unit, 2),
             "stake_usd": stake, "boost_pct": boost or np.nan,
             "suggested": "unknown" if any(o[2] is None for o in out) else ("yes" if all(o[2] for o in out) else "no"),
             "note": ("same-game legs: model treats them as independent, real odds of all hitting differ. " if same_game else "")
                     + ("" if pd.isna(b0["note"]) else str(b0["note"]))}
        if podds is None:
            r.update({"tier": "unscored", "vpe_verdict": "unscored (parlay odds unreadable)"})
        else:
            eff = boosted_american(podds, boost)
            r["odds_eff"] = eff
            if any(x is None for x in probs):
                r.update({"tier": "unscored", "vpe_verdict": "unscored (a leg has no projection)"})
            else:
                prob = float(np.prod(probs))
                te, v, eb, ee = rate_pick(prob, podds, eff, cfg, r["suggested"] == "yes" or (None if r["suggested"] == "unknown" else False))
                if v == "small +EV, below tiers":
                    v = "+EV parlay" if eb > 0 else "+EV only with boost"
                r.update({"model_prob": round(prob, 3), "blend_prob": round(prob, 3), "tier": te or "off_model",
                          "source": "engine" if te else "off_model", "vpe_verdict": v, "edge_base": eb, "edge": ee,
                          "ev_per_unit": round(ev_per_unit(prob, eff), 3)})
        rows.append(r)
    return rows

def _book(b):
    """Book from placed.csv (optional `book` column: DraftKings / FanDuel); default DraftKings."""
    v = str(b.get("book", "") if hasattr(b, "get") else "").strip()
    return {"fd": "FanDuel", "fanduel": "FanDuel", "dk": "DraftKings", "draftkings": "DraftKings"}.get(
        v.lower().replace(" ", ""), v if v and v.lower() != "nan" else "DraftKings")

def win_profit(r):
    """Units won on a WIN at the effective (boosted) odds."""
    o = r.get("odds_eff")
    o = r.get("odds") if o is None or pd.isna(o) else o
    if not r["units"] or o is None or pd.isna(o):
        return 0.0
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
          "max_under_share": 0.45,   # Unders <= 45% of a parlay's legs (2 legs: 0, 3-4: 1, 5-6: 2, 7: 3)
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
    keys = []
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
    if boost:
        print_boost(a, allp, stake, cols, keys)
    if len(pool):
        keys += list(zip(pool.head(6)["player"], pool.head(6)["market"], pool.head(6)["side"]))
    if a.bets and os.path.exists(a.bets):
        b = pd.read_csv(a.bets)
        keys += list(zip(b["player"], b["market"], b["side"]))
    pd.DataFrame(sorted(set(keys)), columns=["player", "market", "side"]).to_csv(a.slip_out, index=False)
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
    if isinstance(p, (pd.Series, np.ndarray)):
        return np.clip(p, 0.01, 0.99)
    return float(min(max(p, 0.01), 0.99))


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
    json.dump([[_cell(v) for v in r] for r in s.itertuples(index=False, name=None)], open(path, "w"))
    print(f"SHADOW: {len(s)} lines -> {path} (append to the Shadow tab; graded tomorrow to keep VPE calibrated)")


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
def boost_parlay(pool, book, boost, stake, min_legs=2, max_legs=4, min_dec=1.0, max_win=None):
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
            if p_mkt * dec_eff < 1:  # must be +EV even if the model adds nothing
                continue
            # least risk for the payout: maximize long-run bankroll growth at a 5% stake (penalizes longshots)
            g = p * math.log(1 + 0.05 * (dec_eff - 1)) + (1 - p) * math.log(0.95)
            if best is None or g > best[6] + 1e-12:
                best = (p * dec_eff - 1, p, p_mkt, dec, dec_eff, combo, g)
    return best[:6] if best and best[6] > 0 else None


def print_boost(a, allp, stake, cols, keys):
    boost, bbook = a.boost, (a.boost_book or "draftkings").lower().replace(" ", "")
    pool = parlay_pool(allp, [x for x in (a.exclude or "").split(";") if x.strip()], min_edge=-0.03, min_prob=0.5)
    min_dec = american_to_decimal(a.boost_min_odds) if a.boost_min_odds else 1.0
    r = boost_parlay(pool, bbook, boost, stake, a.boost_min_legs or 2, a.boost_max_legs or 4, min_dec, a.boost_max_win)
    print(f"\n=== BOOST PARLAY ({boost:g}% profit boost at {BOOK_NAME.get(bbook, bbook)}) ===")
    if not r:
        print("  no combination is +EV at market odds with this boost (too small, or min odds/legs too strict)")
        return
    ev, p, p_mkt, dec, dec_eff, combo = r
    print(f"  PLACE AT {BOOK_NAME.get(bbook, bbook).upper()} | {len(combo)} legs | est. odds {decimal_to_american(dec):+d} -> "
          f"{decimal_to_american(dec_eff):+d} boosted | ${stake:.2f} pays ~${stake * dec_eff:.2f} | hit chance {p:.1%} "
          f"(market {p_mkt:.1%}) | EV {ev:+.0%} (EV at market odds alone {p_mkt * dec_eff - 1:+.0%}) | "
          f"Unders {sum(c['side'] == 'Under' for c in combo)}")
    print(pd.DataFrame(combo)[cols].to_string(index=False))
    if a.boost_max_win:
        print(f"  (boost winnings capped at ${a.boost_max_win:g}: counted)")
    keys += [(c["player"], c["market"], c["side"]) for c in combo]



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
    if a.log and os.path.exists(a.log) and len(out):
        log = pd.read_csv(a.log, dtype={"result": "object"})
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
            "bet_type", "parlay_id", "suggested", "vpe_verdict", "boost_pct", "odds_eff", "edge_base", "note"]

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
        log = pd.read_csv(a.log, dtype={"result": "object"})
        out = sheet_rows(log, header=True)
        json.dump(out, open(a.out, "w"))
        print(f"{len(out) - 1} rows + header -> {a.out} (write it at A1 of the tab with update_values)")

def cmd_log(a):
    set_calib(getattr(a, "calib", None), SPORT)
    """Log bets actually placed (from DraftKings screenshots), including bets VPE did not suggest (profit boosts,
    own picks). Each is re-scored at the line/odds really taken, using the projections saved by the last `check`."""
    cfg = _cfg(a)
    unit = cfg["bankroll"] * cfg["unit_pct_of_bankroll"]
    shrink = cfg["model"]["prob_calibration_shrink"]
    allp = pd.read_csv(a.all) if os.path.exists(a.all) else pd.DataFrame()

    def score(player, mk, side, line, odds):
        hit = allp[(allp["player"].map(norm_name) == norm_name(player)) & (allp["market"] == mk)] if len(allp) else allp
        if hit.empty:
            return None, "no projection saved for this player/market (run check first)"
        h = hit.iloc[0]
        po = 0.5 + (prob_over(h["mean"], h["var"], mk, line) - 0.5) * shrink
        pr = po if side == "Over" else 1 - po
        mp = market_prob_for(allp, player, mk, side, line, odds)
        pb = blend_leg(pr, mp, mk, side)
        return {"prob": pb, "leg_prob": pb, "team": h["team"], "opp": h["opp"]}, ""

    slip = slip_keys(a.slip)
    stamp = {"season": season_for(today_et()),
             "date": allp["date"].iloc[0] if len(allp) else str(today_et().date())}
    rows = build_log_rows(pd.read_csv(a.placed, dtype=str), score, cfg, unit, slip, stamp)
    if getattr(a, "dry_run", False):
        print_logged(rows)
        print("\nDRY RUN (rate only): nothing logged")
        return
    new = pd.DataFrame(rows).reindex(columns=LOG_COLS)
    if os.path.exists(a.log):
        new = pd.concat([pd.read_csv(a.log, dtype={"result": "object"}), new], ignore_index=True).reindex(columns=LOG_COLS)
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
    log = pd.read_csv(a.log, dtype={"result": "object"})
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
    else:
        season = season_for(date or today_et())
        box = load_box(season, max_age_hours=1)
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
    recs = []
    for d in dates:
        state = State(cur, prev, d, cfg, pc, prev2_box=prev2)
        state.pv_box = prev
        today = cur[cur["date"] == d]
        played = today[today["played"]]
        sl = played.set_index("pid")[["team", "opp", "home"]].copy()
        sl = sl[sl.index.isin(state.players.index)]
        sl["b2b"] = [(d - pd.Timedelta(days=1)) in team_dates.get(t, set()) for t in sl["team"]]
        sl["spread"] = np.nan
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
                                      "out": has_out.to_numpy(), "new_team": pr["new_team"].to_numpy()}))
    df = pd.concat(recs, ignore_index=True)
    for c in ("mu", "var", "naive", "actual", "min", "base_min"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["naive"].notna() & df["mu"].notna() & (df["base_min"] >= 15)]
    shrink = cfg["model"]["prob_calibration_shrink"]
    df["line"] = np.floor(df["naive"]) + 0.5
    df["p_raw"] = [prob_over(m, v, mk, l) for m, v, mk, l in zip(df["mu"], df["var"], df["market"], df["line"])]
    df["p"] = 0.5 + (df["p_raw"] - 0.5) * shrink
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
    for name in ("board", "slate", "lines", "check", "log", "grade", "backtest", "parlay"):
        p = sub.add_parser(name)
        p.add_argument("--date"); p.add_argument("--bankroll", type=float); p.add_argument("--profile")
    for name in ("board", "slate", "check"):
        sub.choices[name].add_argument("--injuries", help='"Name:Out;Name:Questionable;Name:Out:PHI"')
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
    sub.choices["grade"].add_argument("--log", default="nba_pick_log.csv"); sub.choices["grade"].add_argument("--actuals")
    sub.choices["backtest"].add_argument("--start", required=True); sub.choices["backtest"].add_argument("--end", required=True)
    sub.choices["backtest"].add_argument("--step", type=int, default=1); sub.choices["backtest"].add_argument("--off", default="")
    sub.choices["backtest"].add_argument("--save")
    sub.choices["parlay"].add_argument("--all", default="bets_all.csv")
    sub.choices["parlay"].add_argument("--bets", default="bets.csv")
    sub.choices["parlay"].add_argument("--stake", type=float, help="$ per parlay (default 5% of bankroll)")
    sub.choices["parlay"].add_argument("--exclude", help='players to leave out "A;B" (e.g. DK price failed take-if)')
    sub.choices["parlay"].add_argument("--slip-out", default="slip.csv")
    sub.choices["parlay"].add_argument("--boost", type=float, help="profit boost %% (gets its own BOOST parlay)")
    sub.choices["parlay"].add_argument("--boost-book", default="draftkings", help="draftkings|fanduel (where the boost is)")
    sub.choices["log"].add_argument("--sheet-out", default="sheet_new.json", help="new rows for the Pick Log sheet")
    for name in ("check", "parlay", "log"):
        sub.choices[name].add_argument("--calib", help="Calib tab JSON (get_values): learned model weight")
    sub.choices["check"].add_argument("--shadow-out", default="shadow_new.json")
    sub.choices["log"].add_argument("--dry-run", action="store_true", help="rate the bets only, log nothing")
    for k, t in (("--boost-min-legs", int), ("--boost-max-legs", int), ("--boost-min-odds", float), ("--boost-max-win", float)):
        sub.choices["parlay"].add_argument(k, type=t)
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
     "grade": cmd_grade, "backtest": cmd_backtest, "parlay": cmd_parlay, "final": cmd_final, "sheet": cmd_sheet, "shadow": cmd_shadow}[a.cmd](a)
