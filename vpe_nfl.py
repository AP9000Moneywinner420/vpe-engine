"""Verified Prop Engine - NFL, single-file in-chat edition. Uses free nflverse CSVs (no API).

  python vpe_nfl.py board --team SF,ATL [--injuries "Name:Out"] [--usage-only]
  python vpe_nfl.py check --props props.csv [--injuries "Name:Out"] [--bankroll 50] [--out bets.csv]
  python vpe_nfl.py grade --log pick_log.csv --week 7
  python vpe_nfl.py lines --src rec_yds=sao_rec_yds.csv --src rush_yds=sao_rush_yds.csv --out props.csv
  python vpe_nfl.py log --placed placed.csv --log pick_log.csv --slip bets.csv
"""
from scipy import stats as st
import itertools
import math
import numpy as np
import os
import pandas as pd
import re
import time
import urllib.request

SPORT = "NFL"

# ==================== data ====================

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
URLS = {
    "stats": BASE + "/stats_player/stats_player_week_{season}.csv",
    "games": BASE + "/schedules/games.csv",
    "injuries": BASE + "/injuries/injuries_{season}.csv",
}

CACHE = os.path.join(os.path.expanduser("~"), ".vpe_cache")

def _download(url, path, max_age_hours):
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-prop-engine/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(path + ".tmp", path)
    except Exception as e:  # keep last good copy if download fails
        if os.path.exists(path):
            print(f"  ! download failed ({e}); using cached {os.path.basename(path)}")
        else:
            raise
    return path

def load_stats(season, max_age_hours=6):
    p = _download(URLS["stats"].format(season=season), os.path.join(CACHE, f"stats_{season}.csv"), max_age_hours)
    df = pd.read_csv(p, low_memory=False)
    return df[df["season_type"] == "REG"].copy()

def load_games(max_age_hours=6):
    p = _download(URLS["games"], os.path.join(CACHE, "games.csv"), max_age_hours)
    return pd.read_csv(p, low_memory=False)

def load_injuries(season, max_age_hours=6):
    try:
        p = _download(URLS["injuries"].format(season=season), os.path.join(CACHE, f"injuries_{season}.csv"), max_age_hours)
        return pd.read_csv(p, low_memory=False)
    except Exception as e:
        print(f"  ! injuries unavailable: {e}")
        return pd.DataFrame(columns=["week", "full_name", "team", "report_status"])

# ---------- name matching (sportsbook names vs nflverse names) ----------
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")

def norm_name(name):
    s = str(name).lower().replace(".", "").replace("'", "").replace("’", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

# ---------- play-level data for scheme matchups ----------
PBP_COLS = ["game_id", "play_id", "season", "season_type", "week", "posteam", "defteam", "pass_attempt", "rush_attempt",
            "sack", "qb_dropback", "two_point_attempt", "receiver_player_id", "passer_player_id", "rusher_player_id",
            "yards_gained", "complete_pass", "passing_yards"]
URLS.update({
    "pbp": BASE + "/pbp/play_by_play_{season}.csv",
    "ftn": BASE + "/ftn_charting/ftn_charting_{season}.csv",
    "part": BASE + "/pbp_participation/pbp_participation_{season}.csv",
})

def load_plays(season, max_age_hours=12):
    p = _download(URLS["pbp"].format(season=season), os.path.join(CACHE, f"pbp_{season}.csv"), max_age_hours)
    pbp = pd.read_csv(p, low_memory=False, usecols=lambda c: c in PBP_COLS)
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"] != 1)]
    try:
        f = _download(URLS["ftn"].format(season=season), os.path.join(CACHE, f"ftn_{season}.csv"), max_age_hours)
        ftn = pd.read_csv(f, low_memory=False, usecols=["nflverse_game_id", "nflverse_play_id", "n_blitzers",
                                                        "n_pass_rushers", "n_defense_box"])
        ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        pbp = pbp.merge(ftn, on=["game_id", "play_id"], how="left")
    except Exception as e:
        print(f"  ! FTN charting unavailable for {season}: {e}")
    try:
        q = _download(URLS["part"].format(season=season), os.path.join(CACHE, f"part_{season}.csv"), 24 * 30)
        part = pd.read_csv(q, low_memory=False, usecols=["nflverse_game_id", "play_id", "defense_coverage_type",
                                                         "defense_man_zone_type"])
        part = part.rename(columns={"nflverse_game_id": "game_id"})
        pbp = pbp.merge(part, on=["game_id", "play_id"], how="left")
    except Exception:
        pass  # coverage shells are only published after the season
    return pbp

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
    d = american_to_decimal(american)
    return prob * (d - 1) - (1 - prob)

def kelly_fraction(prob, american):
    b = american_to_decimal(american) - 1
    f = (b * prob - (1 - prob)) / b
    return max(f, 0.0)

# ==================== usage ====================

# which opportunity drives each market
OPP_FOR_MARKET = {
    "rec_yds": ["tgt"], "receptions": ["tgt"],
    "rush_yds": ["car"], "rush_att": ["car"],
    "rush_rec_yds": ["car", "tgt"], "anytime_td": ["car", "tgt"],
}
# markets where the backtest showed the usage shift helps (receiving). Others get a flag only.
TESTED_MARKETS = {"rec_yds", "receptions", "rush_rec_yds"}
# how strongly same-position teammates absorb vacated usage
POS_BOOST = {
    "tgt": {"same": 1.6, "other": 1.0},
    "car": {"RB": 1.0, "QB": 0.15, "WR": 0.1, "TE": 0.05},
}

class UsageModel:
    def __init__(self, s, week, outs, cfg):
        self.cfg = cfg
        self.week = week
        self.decay = cfg["recency_decay"]
        d = s[s["position"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
        team = d.groupby(["team", "week"]).agg(team_tgt=("targets", "sum"), team_car=("carries", "sum"),
                                               team_att=("attempts", "sum"))
        d = d.join(team, on=["team", "week"])
        d["tgt"] = (d["targets"].fillna(0) / d["team_tgt"].replace(0, np.nan)).fillna(0)
        d["car"] = (d["carries"].fillna(0) / d["team_car"].replace(0, np.nan)).fillna(0)
        d["att_share"] = (d["attempts"].fillna(0) / d["team_att"].replace(0, np.nan)).fillna(0)
        self.d = d
        self.team_weeks = d.groupby("team")["week"].apply(lambda x: sorted(set(x))).to_dict()
        self.outs = set(outs or [])
        latest = d.sort_values("week").groupby("player_id").tail(1).set_index("player_id")
        self.latest = latest
        self._cache = {}
        self._outs_cache = {}
        self._rows = {k: g for k, g in d.groupby(["player_id", "team"])}

    # recency-weighted mean of a share over the given weeks (0 for weeks the player missed is NOT counted:
    # shares are per game played)
    def _wshare(self, rows, col, weeks=None):
        r = rows if weeks is None else rows[rows["week"].isin(weeks)]
        if r.empty:
            return None
        ages = self.week - 1 - r["week"].to_numpy()
        w = self.decay ** (ages - ages.min())
        return float(np.sum(w * r[col].to_numpy()) / np.sum(w))

    def _prow(self, pid, team):
        return self._rows.get((pid, team), self.d.iloc[0:0])

    def team_outs(self, team):
        if team in self._outs_cache:
            return self._outs_cache[team]
        lt = self.latest
        cand = lt[(lt["team"] == team) & (lt["_key"].isin(self.outs))]
        res = []
        for pid, r in cand.iterrows():
            rows = self._prow(pid, team)
            tg, ca, at = (self._wshare(rows, c) or 0 for c in ("tgt", "car", "att_share"))
            if tg >= 0.08 or ca >= 0.15 or at >= 0.5:
                res.append({"pid": pid, "name": r["player_display_name"], "pos": r["position"],
                            "weeks": sorted(set(rows["week"])), "tgt": tg, "car": ca, "att": at})
        self._outs_cache[team] = res
        return res

    def _eff(self, rows, pos, num_col, den_col, k):
        num, den = rows[num_col].fillna(0).sum(), rows[den_col].fillna(0).sum()
        key = ("eff", pos, num_col, den_col)
        if key not in self._cache:
            p = self.d[self.d["position"] == pos]
            pd_ = p[den_col].fillna(0).sum()
            self._cache[key] = p[num_col].fillna(0).sum() / pd_ if pd_ else 0
        return (num + k * self._cache[key]) / (den + k)

    def _team_volume(self, team, col):
        t = self.d[self.d["team"] == team].groupby("week")[col].first()
        if t.empty:
            return 0
        ages = (self.week - 1 - t.index.to_numpy())
        w = self.decay ** (ages - ages.min())
        return float(np.sum(w * t.to_numpy()) / np.sum(w))

    def adjust(self, pid, market):
        key = (pid, market)
        if key in self._cache:
            return self._cache[key]
        out = (0.0, 1.0, "")
        if pid not in self.latest.index or not self.outs:
            self._cache[key] = out
            return out
        me = self.latest.loc[pid]
        team, pos = me["team"], me["position"]
        all_outs = [o for o in self.team_outs(team) if o["pid"] != pid]
        # QBs never hand their targets/carries to skill players (the backup QB takes the scrambles)
        outs = [o for o in all_outs if o["pos"] != "QB"]
        team_wks = self.team_weeks.get(team, [])
        # starting QB out, and the team has fewer than 2 games of data with the backup (else it's already in the stats)
        qb_out = [o for o in all_outs if o["pos"] == "QB" and o["att"] >= 0.5
                  and len([w for w in team_wks if w not in o["weeks"]]) < 2]
        if not outs and not qb_out:
            self._cache[key] = out
            return out

        my_rows = self._prow(pid, team)
        extra, notes = 0.0, []
        opps = OPP_FOR_MARKET.get(market) if (outs and market in TESTED_MARKETS) else None
        rb_out = [o for o in outs if o["car"] >= 0.15]
        if rb_out and market in ("rush_yds", "rush_att", "anytime_td") and pos == "RB":
            notes.append("backfield shift - carries not auto-adjusted (untested), use judgment")
        if opps and len(my_rows):
            with_wks = [w for w in team_wks if all(w in o["weeks"] for o in outs)]
            if len(with_wks) < 2:
                big = max(outs, key=lambda o: o["tgt"] + o["car"])
                with_wks = big["weeks"]
            wo_wks = [w for w in team_wks if all(w not in o["weeks"] for o in outs)]
            for o_col in opps:
                if pos == "QB":
                    continue
                all_share = self._wshare(my_rows, o_col) or 0
                with_share = self._wshare(my_rows, o_col, with_wks)
                if with_share is None:
                    with_share = all_share
                vacated = sum(self._wshare(self._prow(o["pid"], team), o_col, with_wks) or o[o_col]
                              for o in outs) * self.cfg.get("usage_retention_" + o_col, 0.7)
                if vacated < 0.02:
                    continue
                lt = self.latest
                mates = lt[(lt["team"] == team) & (~lt["_key"].isin(self.outs)) & (lt["week"] >= self.week - 3)
                           & (lt["position"] != "QB")]
                out_pos = {o["pos"] for o in outs if o[o_col] > 0.05}
                weights = {}
                for mid, mr in mates.iterrows():
                    rows = self._prow(mid, team)
                    sh = self._wshare(rows, o_col, with_wks)
                    if sh is None:
                        sh = self._wshare(rows, o_col) or 0
                    boost = (POS_BOOST["tgt"]["same"] if mr["position"] in out_pos else POS_BOOST["tgt"]["other"]) \
                        if o_col == "tgt" else POS_BOOST["car"].get(mr["position"], 0.05)
                    weights[mid] = max(sh, 0.0) * boost
                tw = sum(weights.values())
                prop = weights.get(pid, 0) / tw if tw > 0 else 0
                same = [mid for mid, mr in mates.iterrows()
                        if (mr["position"] in out_pos if o_col == "tgt" else mr["position"] == "RB")]
                eq = (1 / len(same)) if (pid in same and same) else 0
                a = self.cfg.get("usage_alloc_share_weight", 0.5)
                new_share = with_share + vacated * (a * prop + (1 - a) * eq)
                wo_rows = my_rows[my_rows["week"].isin(wo_wks)]
                if len(wo_rows):  # real games already played without them
                    k = self.cfg.get("usage_model_weight_games", 3)
                    new_share = (k * new_share + len(wo_rows) * self._wshare(wo_rows, o_col)) / (k + len(wo_rows))
                d_share = max(new_share - all_share, 0.0)  # teammates out never LOWER his opportunities (v11 fix)
                vol = self._team_volume(team, "team_tgt" if o_col == "tgt" else "team_car")
                d_opps = d_share * vol
                if o_col == "tgt":
                    eff = {"rec_yds": ("receiving_yards", 40), "receptions": ("receptions", 40),
                           "rush_rec_yds": ("receiving_yards", 40), "anytime_td": ("receiving_tds", 80)}[market]
                    per = self._eff(my_rows, pos, eff[0], "targets", eff[1])
                else:
                    eff = {"rush_yds": ("rushing_yards", 40), "rush_att": (None, 0),
                           "rush_rec_yds": ("rushing_yards", 40), "anytime_td": ("rushing_tds", 80)}[market]
                    per = 1.0 if eff[0] is None else self._eff(my_rows, pos, eff[0], "carries", eff[1])
                extra += d_opps * per
                label = "targets" if o_col == "tgt" else "carries"
                if abs(d_opps) >= 0.3:
                    notes.append(f"{d_opps:+.1f} {label}/g ({all_share:.0%}->{new_share:.0%} share)")

        eff_f = 1.0
        if qb_out and market in ("rec_yds", "receptions", "rush_rec_yds", "anytime_td") and pos != "QB":
            eff_f = self.cfg.get("backup_qb_efficiency", 0.9)
            notes.append(f"backup QB x{eff_f}")
        who = ", ".join(o["name"] for o in (outs + qb_out))
        res = (extra, eff_f, (f"OUT: {who} | " + "; ".join(notes)) if notes else "")
        self._cache[key] = res
        return res

# ==================== scheme ====================

COVER_GROUP = {"COVER_0": "man", "COVER_1": "man", "2_MAN": "man",
               "COVER_3": "single_high_zone",
               "COVER_2": "two_high_zone", "COVER_4": "two_high_zone", "COVER_6": "two_high_zone",
               "COVER_9": "two_high_zone"}

MARKET_DIMS = {  # which situations matter for which prop, and what "efficiency" means there
    "rec_yds":    ("tgt", "yards_gained", ["blitz", "coverage"]),
    "receptions": ("tgt", "complete_pass", ["blitz", "coverage"]),
    "pass_yds":   ("drop", "passing_yards", ["blitz", "coverage"]),
    "pass_comp":  ("drop", "complete_pass", ["blitz", "coverage"]),
    "rush_yds":   ("car", "yards_gained", ["box"]),
}

class SchemeModel:
    def __init__(self, plays_now, plays_prev, week, cfg):
        self.cfg = cfg.get("scheme", {})
        now = plays_now[plays_now["week"] < week].copy()
        now["_w"] = 1.0
        prev = plays_prev.copy() if plays_prev is not None else now.iloc[0:0]
        prev["_w"] = self.cfg.get("prev_season_weight", 0.5)
        p = pd.concat([prev, now], ignore_index=True)
        p["blitz"] = np.where(p["n_blitzers"].isna(), np.nan, (p["n_blitzers"] > 0).astype(float))
        p["box"] = np.where(p["n_defense_box"].isna(), np.nan, (p["n_defense_box"] >= 8).astype(float))
        if "defense_coverage_type" in p:
            p["coverage"] = p["defense_coverage_type"].map(COVER_GROUP)
        else:
            p["coverage"] = np.nan
        self.p = p
        self.now = p[p["_w"] == 1.0]
        self._cache = {}

    # plays of the right type for a player / a defense
    def _subset(self, kind, df):
        if kind == "tgt":
            return df[(df["pass_attempt"] == 1) & df["receiver_player_id"].notna()]
        if kind == "drop":
            return df[(df["qb_dropback"] == 1) & (df["sack"] != 1)]
        return df[df["rush_attempt"] == 1]

    def _groups(self, dim):
        return ["man", "single_high_zone", "two_high_zone"] if dim == "coverage" else [1.0, 0.0]

    def _league(self, kind, metric, dim):
        key = ("lg", kind, metric, dim)
        if key not in self._cache:
            d = self._subset(kind, self.p).dropna(subset=[dim])
            overall = np.average(d[metric].fillna(0), weights=d["_w"])
            ratio, mix = {}, {}
            for g in self._groups(dim):
                dg = d[d[dim] == g]
                ratio[g] = (np.average(dg[metric].fillna(0), weights=dg["_w"]) / overall) if len(dg) and overall else 1.0
                mix[g] = dg["_w"].sum() / d["_w"].sum()
            self._cache[key] = (ratio, mix)
        return self._cache[key]

    def _player_sens(self, pid, kind, metric, dim):
        col = {"tgt": "receiver_player_id", "drop": "passer_player_id", "car": "rusher_player_id"}[kind]
        d = self._subset(kind, self.p)
        d = d[(d[col] == pid)].dropna(subset=[dim])
        lg_ratio, _ = self._league(kind, metric, dim)
        k = self.cfg.get("player_shrink", {"tgt": 40, "drop": 80, "car": 50})[kind]
        if d.empty:
            return {g: 1.0 for g in self._groups(dim)}, 0
        overall = np.average(d[metric].fillna(0), weights=d["_w"])
        sens = {}
        for g in self._groups(dim):
            dg = d[d[dim] == g]
            n = dg["_w"].sum()
            if n == 0 or overall <= 0:
                sens[g] = 1.0
                continue
            raw = (np.average(dg[metric].fillna(0), weights=dg["_w"]) / overall) / max(lg_ratio[g], 1e-6)
            sens[g] = 1 + (raw - 1) * n / (n + k)
        return sens, d["_w"].sum()

    def _defense_mix(self, team, kind, dim):
        key = ("mix", team, kind, dim)
        if key in self._cache:
            return self._cache[key]
        _, lg_mix = self._league(kind, "yards_gained", dim)
        # current-season tendencies when charted (blitz, box); last season's for coverage shells
        src = self.now if self.now[dim].notna().any() else self.p
        d = self._subset(kind, src)
        d = d[(d["defteam"] == team)].dropna(subset=[dim])
        n = len(d)
        k = self.cfg.get("defense_shrink", {"blitz": 100, "box": 80, "coverage": 250})[dim]
        mix = {}
        for g in self._groups(dim):
            raw = (d[dim] == g).mean() if n else lg_mix[g]
            mix[g] = lg_mix[g] + (raw - lg_mix[g]) * n / (n + k)
        if dim == "coverage":  # last season's shells: only partly trusted (coordinators change)
            t = self.cfg.get("coverage_trust", 0.5)
            mix = {g: lg_mix[g] + (mix[g] - lg_mix[g]) * t for g in mix}
        self._cache[key] = (mix, n)
        return mix, n

    def factor(self, pid, opp, market):
        if market not in MARKET_DIMS or not self.cfg.get("enabled", True):
            return 1.0, ""
        key = ("f", pid, opp, market)
        if key in self._cache:
            return self._cache[key]
        kind, metric, dims = MARKET_DIMS[market]
        f, notes = 1.0, []
        for dim in dims:
            sens, n_p = self._player_sens(pid, kind, metric, dim)
            if n_p == 0:
                continue
            mix, _ = self._defense_mix(opp, kind, dim)
            _, lg_mix = self._league(kind, metric, dim)
            num = sum(mix[g] * sens[g] for g in sens)
            den = sum(lg_mix[g] * sens[g] for g in sens)
            fd = num / den if den else 1.0
            fd = 1 + (fd - 1) * self.cfg.get("strength", 1.0)
            f *= fd
            if abs(fd - 1) >= 0.02:
                label = {"blitz": "blitz", "coverage": "coverage shell", "box": "stacked box"}[dim]
                g_hi = max(mix, key=lambda g: mix[g] - lg_mix[g])
                gname = {1.0: "heavy" if dim != "box" else "8+ box", 0.0: "light", "man": "man",
                         "single_high_zone": "Cover 3", "two_high_zone": "2-high zone"}[g_hi]
                notes.append(f"{label}: D leans {gname} {mix[g_hi]:.0%} (lg {lg_mix[g_hi]:.0%}), "
                             f"player x{sens[g_hi]:.2f} there -> x{fd:.3f}")
        lo, hi = self.cfg.get("cap", [0.9, 1.1])
        f = float(min(max(f, lo), hi))
        note = "; ".join(notes)
        if market not in self.cfg.get("apply_markets", list(MARKET_DIMS)):
            # backtest: no gain for this market yet -> show the matchup, don't move the number
            note, f = (note + " (info only)") if note else "", 1.0
        res = (f, note)
        self._cache[key] = res
        return res

# ==================== projections ====================

# market -> definition
MARKETS = {
    "pass_yds":     {"cols": ["passing_yards"], "dist": "gamma", "env": "pass", "cv": 0.30, "pos": ["QB"]},
    "pass_tds":     {"cols": ["passing_tds"], "dist": "poisson", "env": "td", "pos": ["QB"]},
    "pass_att":     {"cols": ["attempts"], "dist": "normal", "env": "count", "cv": 0.17, "pos": ["QB"]},
    "pass_comp":    {"cols": ["completions"], "dist": "normal", "env": "count", "cv": 0.20, "pos": ["QB"]},
    "pass_int":     {"cols": ["passing_interceptions"], "dist": "poisson", "env": None, "pos": ["QB"]},
    "rush_yds":     {"cols": ["rushing_yards"], "dist": "gamma", "env": "rush", "cv": 0.55, "pos": ["QB", "RB", "WR"]},
    "rush_att":     {"cols": ["carries"], "dist": "normal", "env": "rush", "cv": 0.35, "pos": ["QB", "RB"]},
    "rec_yds":      {"cols": ["receiving_yards"], "dist": "gamma", "env": "rec", "cv": 0.65, "pos": ["RB", "WR", "TE"]},
    "receptions":   {"cols": ["receptions"], "dist": "poisson", "env": "count", "pos": ["RB", "WR", "TE"]},
    "rush_rec_yds": {"cols": ["rushing_yards", "receiving_yards"], "dist": "gamma", "env": "rush", "cv": 0.45, "pos": ["RB", "WR", "TE"]},
    "anytime_td":   {"cols": ["rushing_tds", "receiving_tds"], "dist": "poisson", "env": "td", "pos": ["QB", "RB", "WR", "TE"]},
}

MARKET_ALIASES = {
    "passing yards": "pass_yds", "pass yds": "pass_yds", "passing tds": "pass_tds", "pass tds": "pass_tds",
    "pass attempts": "pass_att", "passing attempts": "pass_att", "completions": "pass_comp",
    "pass completions": "pass_comp", "interceptions": "pass_int", "ints": "pass_int",
    "rushing yards": "rush_yds", "rush yds": "rush_yds", "rush attempts": "rush_att", "carries": "rush_att",
    "receiving yards": "rec_yds", "rec yds": "rec_yds", "receptions": "receptions", "rec": "receptions",
    "rush + rec yds": "rush_rec_yds", "rush+rec yds": "rush_rec_yds", "rushing + receiving yards": "rush_rec_yds",
    "anytime td": "anytime_td", "anytime touchdown": "anytime_td", "atd": "anytime_td",
    # DraftKings menu names
    "anytime td scorer": "anytime_td", "anytime touchdown scorer": "anytime_td", "passing touchdowns": "pass_tds",
    "interceptions thrown": "pass_int", "pass yards": "pass_yds", "rushing attempts": "rush_att",
    "rush + rec yards": "rush_rec_yds", "rushing + receiving yds": "rush_rec_yds", "receiving yds": "rec_yds",
    "rushing yds": "rush_yds", "passing yds": "pass_yds", "pass tds": "pass_tds",
}

def canon_market(m):
    m0 = str(m).strip().lower()
    return m0 if m0 in MARKETS else MARKET_ALIASES.get(m0)

def _val(df, cols):
    return df[cols].fillna(0).sum(axis=1)

class Projector:
    def __init__(self, stats_now, stats_prev, games, season, week, cfg, outs=None, weather=None, plays=None):
        self.cfg = cfg["model"]
        self.season, self.week = season, week
        self.s = stats_now[stats_now["week"] < week].copy()
        self.prev = stats_prev
        self.games = games[games["season"] == season].copy()
        self.s["_key"] = self.s["player_display_name"].map(norm_name)
        self._latest = self.s.sort_values("week").groupby("player_id").tail(1).set_index("player_id")
        self._build_env()
        self._opp_cache = {}
        self.usage = UsageModel(self.s, week, outs, self.cfg)
        self.weather = weather or {}  # {"BUF": (wind_mph, temp_f)} forecasts for upcoming outdoor games
        self.scheme = SchemeModel(plays[0], plays[1], week, self.cfg) if plays is not None else None

    # ---------- lookup ----------
    def find_player(self, name):
        key = norm_name(name)
        hit = self._latest[self._latest["_key"] == key]
        if hit.empty:  # try last-name + first initial fallback
            parts = key.split()
            if len(parts) >= 2:
                cand = self._latest[self._latest["_key"].str.endswith(" " + parts[-1])
                                    & self._latest["_key"].str.startswith(parts[0][0])]
                if len(cand) == 1:
                    hit = cand
        if hit.empty:
            return None
        if len(hit) > 1:  # same name: take the one with most recent game
            hit = hit.sort_values("week").tail(1)
        return hit.index[0]

    # ---------- game environment ----------
    def _build_env(self):
        g = self.games.dropna(subset=["spread_line", "total_line"])
        rows = []
        for _, r in g.iterrows():
            rows.append((r["week"], r["home_team"], r["away_team"], r["total_line"] / 2 + r["spread_line"] / 2, 1))
            rows.append((r["week"], r["away_team"], r["home_team"], r["total_line"] / 2 - r["spread_line"] / 2, 0))
        self.env = pd.DataFrame(rows, columns=["week", "team", "opp", "itt", "home"])
        past = self.env[self.env["week"] < self.week]
        self.team_avg_itt = past.groupby("team")["itt"].mean().to_dict()

    def matchup(self, team):
        e = self.env[(self.env["week"] == self.week) & (self.env["team"] == team)]
        if e.empty:
            g = self.games[(self.games["week"] == self.week)
                           & ((self.games["home_team"] == team) | (self.games["away_team"] == team))]
            if g.empty:
                return None
            r = g.iloc[0]
            opp = r["away_team"] if r["home_team"] == team else r["home_team"]
            return {"opp": opp, "itt": None, "home": int(r["home_team"] == team), "game_id": r["game_id"]}
        r = e.iloc[0]
        gid = self.games[(self.games["week"] == self.week)
                         & ((self.games["home_team"] == team) | (self.games["away_team"] == team))]["game_id"].iloc[0]
        return {"opp": r["opp"], "itt": r["itt"], "home": int(r["home"]), "game_id": gid}

    def _pos_baseline(self, pos, cols, own):
        if cols not in (["rushing_tds", "receiving_tds"], ["passing_tds"], ["passing_interceptions"]):
            return own
        ck = ("base", pos, tuple(cols))
        if ck in self._opp_cache:
            return self._opp_cache[ck]
        d = self.s[self.s["position"] == pos]
        per_player = d.groupby("player_id").apply(lambda x: _val(x, cols).mean(), include_groups=False)
        top = d.groupby("player_id")["fantasy_points_ppr"].mean().sort_values(ascending=False)
        n_keep = {"QB": 32, "RB": 48, "WR": 80, "TE": 40}.get(pos, 50)
        self._opp_cache[ck] = float(per_player.reindex(top.index[:n_keep]).mean())
        return self._opp_cache[ck]

    # ---------- opponent adjustment ----------
    def _opp_factor(self, opp, pos_group, cols):
        if self.cfg.get("opp_adjust_method", "raw") == "schedule":
            return self._opp_factor_sched(opp, pos_group, cols)
        key = (opp, pos_group, tuple(cols))
        if key in self._opp_cache:
            return self._opp_cache[key]
        d = self.s[self.s["position_group"] == pos_group].copy()
        d["_v"] = _val(d, cols)
        per_game = d.groupby(["opponent_team", "week"])["_v"].sum().reset_index()
        league = per_game["_v"].mean()
        mine = per_game[per_game["opponent_team"] == opp]
        n = len(mine)
        if n == 0 or league <= 0:
            f = 1.0
        else:
            ratio = mine["_v"].mean() / league
            k = self.cfg["opp_adjust_shrink_games"] * (2 if "tds" in cols[0] else 1)
            f = 1 + (ratio - 1) * n / (n + k)
            lo, hi = self.cfg["opp_adjust_cap"]
            f = float(min(max(f, lo), hi))
        self._opp_cache[key] = f
        return f

    def _opp_factor_sched(self, opp, pos_group, cols):
        key = ("sched", opp, pos_group, tuple(cols))
        if key in self._opp_cache:
            return self._opp_cache[key]
        if "_sched_tbl" not in self._opp_cache or self._opp_cache.get("_sched_cols") != (pos_group, tuple(cols)):
            pass
        tk = ("tbl", pos_group, tuple(cols))
        if tk not in self._opp_cache:
            d = self.s[self.s["position_group"] == pos_group][["player_id", "week", "opponent_team"] + cols].copy()
            d["_v"] = _val(d, cols)
            g = d.groupby("player_id")["_v"]
            d["_n"], d["_sum"] = g.transform("size"), g.transform("sum")
            d = d[d["_n"] >= 3]
            d["_exp"] = (d["_sum"] - d["_v"]) / (d["_n"] - 1)  # his average in his other games
            self._opp_cache[tk] = d[d["_exp"] > 0]
        d = self._opp_cache[tk]
        mine = d[d["opponent_team"] == opp]
        games = mine["week"].nunique()
        if games == 0 or mine["_exp"].sum() <= 0:
            f = 1.0
        else:
            ratio = mine["_v"].sum() / mine["_exp"].sum()
            k = self.cfg["opp_adjust_shrink_games"] * (2 if "tds" in cols[0] else 1)
            f = 1 + (ratio - 1) * games / (games + k)
            lo, hi = self.cfg["opp_adjust_cap"]
            f = float(min(max(f, lo), hi))
        self._opp_cache[key] = f
        return f

    # ---------- situational (measured on 2025-26 player-games, applied at ~60% strength) ----------
    def situational(self, team, market):
        sc = self.cfg.get("situational", {})
        g = self.games[(self.games["week"] == self.week)
                       & ((self.games["home_team"] == team) | (self.games["away_team"] == team))]
        if g.empty or not sc:
            return 1.0, ""
        r = g.iloc[0]
        f, notes = 1.0, []
        group = ("pass" if market in ("pass_yds", "pass_tds", "pass_comp") else
                 "rec" if market in ("rec_yds", "receptions") else
                 "rush" if market in ("rush_yds", "rush_att") else None)
        if group is None:
            return 1.0, ""
        outdoor = str(r.get("roof", "outdoors")) not in ("dome", "closed")
        wind, temp = r.get("wind"), r.get("temp")
        for t in (r["home_team"], r["away_team"]):
            if t in self.weather:
                wind, temp = self.weather[t]
        if outdoor and pd.notna(wind) and wind >= sc["wind_mph"]:
            f *= sc["wind"][group]; notes.append(f"wind {wind:.0f}mph x{sc['wind'][group]}")
        if outdoor and pd.notna(temp) and temp <= sc["cold_f"]:
            f *= sc["cold"][group]; notes.append(f"{temp:.0f}F x{sc['cold'][group]}")
        if r.get("div_game") == 1:
            f *= sc["div"][group]
            if sc["div"][group] != 1: notes.append(f"division game x{sc['div'][group]}")
        return f, "; ".join(notes)

    # ---------- core projection ----------
    def project(self, pid, market):
        m = MARKETS[market]
        hist = self.s[self.s["player_id"] == pid].sort_values("week")
        if len(hist) < self.cfg["min_games_this_season"]:
            return None
        info = self._latest.loc[pid]
        pos = info["position"]
        if pos not in m["pos"]:
            return None
        vals = _val(hist, m["cols"]).to_numpy(dtype=float)
        n = len(vals)
        w = self.cfg["recency_decay"] ** np.arange(n - 1, -1, -1)
        wmean = float(np.sum(w * vals) / np.sum(w))

        # prior: last season per-game average
        prior, prior_n = None, 0
        if self.prev is not None:
            ph = self.prev[self.prev["player_id"] == pid]
            if len(ph) >= 4:
                prior, prior_n = float(_val(ph, m["cols"]).mean()), len(ph)
        pw = self.cfg["prior_weight_games"]
        k = pw.get(market, pw["default"]) if isinstance(pw, dict) else pw
        if prior is None:  # no usable 2025 sample: lean on a position baseline instead
            prior, k = self._pos_baseline(pos, m["cols"], wmean), k / 2
        base = (n * wmean + k * prior) / (n + k)

        team = info["team"]
        mu = self.matchup(team)
        if mu is None:
            return None  # bye week / no game
        opp_f = self._opp_factor(mu["opp"], info["position_group"], m["cols"])

        env_f = 1.0
        if m["env"] and mu["itt"] is not None and team in self.team_avg_itt:
            el = self.cfg["game_env_elasticity"][m["env"]]
            env_f = (mu["itt"] / self.team_avg_itt[team]) ** el

        use_add, eff_f, use_note = self.usage.adjust(pid, market)
        new_base = max(base + use_add, 0.01)
        use_f = new_base / base if base > 0.01 else 1.0
        sit_f, sit_note = self.situational(team, market)
        sch_f, sch_note = self.scheme.factor(pid, mu["opp"], market) if self.scheme else (1.0, "")
        mc = self.cfg.get("mean_calib", {}).get(market, 1.0)  # measured bias fix (Weeks 3-6 backtest)
        mean = max(new_base * opp_f * env_f * eff_f * sit_f * sch_f * mc, 0.01)

        # spread estimate (player's own variance shrunk toward a market-typical CV)
        sd = None
        if m["dist"] in ("gamma", "normal"):
            s_player = (float(np.std(vals, ddof=1)) * use_f * eff_f * sit_f * sch_f * mc) if n > 1 else m["cv"] * mean
            k_sd = 4
            sd = math.sqrt((n * s_player ** 2 + k_sd * (m["cv"] * mean) ** 2) / (n + k_sd))
            sd = max(sd, 0.15 * mean, 1.0)

        return {
            "player_id": pid, "player": info["player_display_name"], "team": team, "pos": pos,
            "opp": mu["opp"], "home": mu["home"], "game_id": mu["game_id"], "market": market,
            "games": n, "season_avg": round(float(vals.mean()), 2), "last3_avg": round(float(vals[-3:].mean()), 2),
            "prior_avg": None if prior is None else round(prior, 2),
            "opp_factor": round(opp_f, 3), "env_factor": round(env_f, 3),
            "usage_factor": round(use_f * eff_f, 3), "usage_note": use_note,
            "situation_factor": round(sit_f, 3), "situation_note": sit_note,
            "scheme_factor": round(sch_f, 3), "scheme_note": sch_note,
            "mean": round(mean, 2), "sd": None if sd is None else round(sd, 2), "dist": m["dist"],
            "median": round(self.median(mean, sd, m["dist"]), 1),
        }

    # ---------- probabilities ----------
    @staticmethod
    def median(mean, sd, dist):
        if dist == "poisson":
            return float(st.poisson.median(mean))
        if dist == "gamma":
            k = (mean / sd) ** 2
            return float(st.gamma.median(k, scale=sd ** 2 / mean))
        return mean

    @staticmethod
    def prob_over(proj, line):
        mean, sd, dist = proj["mean"], proj["sd"], proj["dist"]
        if dist == "poisson":
            # P(X > line) for integer X; line like 0.5, 4.5
            return float(st.poisson.sf(math.floor(line), mean))
        if dist == "gamma":
            k = (mean / sd) ** 2
            return float(st.gamma.sf(line, k, scale=sd ** 2 / mean))
        # normal for volume stats; continuity correction for integer counts
        return float(st.norm.sf(math.floor(line) + 0.5, loc=mean, scale=sd))

    # ---------- full board ----------
    def board(self):
        recent = self._latest[(self._latest["week"] >= self.week - 2)
                              & (self._latest["position"].isin(["QB", "RB", "WR", "TE"]))]
        out = []
        for pid, r in recent.iterrows():
            for mk, m in MARKETS.items():
                if r["position"] not in m["pos"]:
                    continue
                p = self.project(pid, mk)
                if p is None:
                    continue
                # skip irrelevant (e.g. WR rushing yards with ~0 usage)
                if p["season_avg"] < {"rush_yds": 15, "rush_att": 4, "rec_yds": 15, "receptions": 1.5,
                                      "pass_yds": 120, "pass_att": 15, "pass_comp": 10,
                                      "rush_rec_yds": 25}.get(mk, 0):
                    continue
                if mk in ("pass_tds", "pass_int") and r["position"] == "QB":
                    att = self.s[self.s["player_id"] == pid]["attempts"].mean()
                    if att < 15:
                        continue
                out.append(p)
        return pd.DataFrame(out)

# ==================== selection ====================

ASSUMED_HOLD_ONE_SIDED = 0.07  # one-sided DK markets (anytime TD, milestones like 60+ yds): only for the info column

def injury_status(injuries, overrides, name, team, week):
    key = norm_name(name)
    if overrides is not None and not overrides.empty:
        o = overrides[overrides["_key"] == key]
        if not o.empty:
            return str(o.iloc[-1]["status"]).title()
    if injuries is None or injuries.empty:
        return None
    inj = injuries[(injuries["week"] == week) & (injuries["team"] == team)]
    hit = inj[inj["full_name"].map(norm_name) == key]
    if hit.empty:
        return None
    s = hit.iloc[-1]["report_status"]
    return None if pd.isna(s) else str(s)

def build_outs(injuries, overrides, week):
    outs = set()
    if injuries is not None and not injuries.empty:
        inj = injuries[(injuries["week"] == week) & injuries["report_status"].isin(["Out", "Doubtful"])]
        outs |= set(inj["full_name"].map(norm_name))
    if overrides is not None and not overrides.empty:
        for _, o in overrides.iterrows():
            st_ = str(o["status"]).title()
            if st_ in ("Out", "Doubtful", "Ir"):
                outs.add(o["_key"])
            else:
                outs.discard(o["_key"])
    return outs

def _take_if(p, line, side, prob, shrink, cfg):
    """Worst DraftKings price / line at which this side still clears Tier 3. Lines from free odds sites are
    best-of-market, so the user checks DK against these two numbers before betting."""
    c = cfg["tiers"]["tier_3"]
    be = prob - c["min_edge"]
    min_odds = int(decimal_to_american(1 / be)) if 0.02 < be < 0.98 else None
    if p["market"] == "anytime_td":
        return min_odds, None
    be110 = implied_prob(-110)
    limit, L = None, line
    po0 = 0.5 + (Projector.prob_over(p, line) - 0.5) * shrink
    for _ in range(25):
        po = 0.5 + (Projector.prob_over(p, L) - 0.5) * shrink
        po = (1 - prob if side == "Under" else prob) + PARLAY["model_weight"] * (po - po0)  # blended, moved with the line
        pr = po if side == "Over" else 1 - po
        if pr - be110 >= c["min_edge"] and pr >= c["min_prob"]:
            limit = L
            L = L + 1 if side == "Over" else L - 1
            if L < 0: break
        else:
            break
    return min_odds, limit

def _tier(edge, prob, cfg):
    for t in ("tier_1", "tier_2", "tier_3"):
        c = cfg["tiers"][t]
        if edge >= c["min_edge"] and prob >= c["min_prob"]:
            return t
    return None

def evaluate_lines(lines, proj: Projector, injuries, overrides, cfg):
    unit_pct = cfg["unit_pct_of_bankroll"]
    kf = cfg["staking"]["kelly_fraction"]
    profile = cfg["strategy_profiles"][cfg["strategy_profile"]]
    rows, skipped = [], []

    for _, r in lines.iterrows():
        if pd.isna(r.get("line")) or (pd.isna(r.get("over_odds")) and pd.isna(r.get("under_odds"))):
            continue
        mk = canon_market(r["market"])
        if mk is None:
            skipped.append((r["player"], r["market"], "unknown market"))
            continue
        pid = proj.find_player(r["player"])
        if pid is None:
            skipped.append((r["player"], mk, "player not found in 2026 stats"))
            continue
        p = proj.project(pid, mk)
        if p is None:
            skipped.append((r["player"], mk, "not enough games / no game this week / wrong position"))
            continue
        status = injury_status(injuries, overrides, p["player"], p["team"], proj.week)
        if status in ("Out", "Doubtful", "Ir"):
            skipped.append((p["player"], mk, f"injury: {status}"))
            continue

        line = float(r["line"])
        # calibration shrink from backtest.py (not for TD markets: Poisson is already calibrated there)
        shrink = 1.0 if mk == "anytime_td" else cfg["model"].get("prob_calibration_shrink", 1.0)
        p_model_over = 0.5 + (Projector.prob_over(p, line) - 0.5) * shrink
        oo, uo = r.get("over_odds"), r.get("under_odds")
        if pd.notna(oo) and pd.notna(uo):
            fair_over, _ = no_vig(oo, uo)
        else:
            side_odds = oo if pd.notna(oo) else uo
            fair = implied_prob(side_odds) / (1 + ASSUMED_HOLD_ONE_SIDED)
            fair_over = fair if pd.notna(oo) else 1 - fair
        p_over = blend_leg(p_model_over, fair_over, mk, "Over")

        for side, odds, prob, pm in (("Over", oo, p_over, p_model_over), ("Under", uo, 1 - p_over, 1 - p_model_over)):
            if pd.isna(odds):
                continue
            edge = prob - implied_prob(odds)
            ev = ev_per_unit(prob, odds)
            tier = _tier(edge, prob, cfg)
            if tier == "tier_3" and not profile["allow_tier3"]:
                tier = None
            units = 0.0
            if tier:
                units = min(kf * kelly_fraction(prob, odds) / unit_pct, cfg["tiers"][tier]["max_units"])
                if status == "Questionable":
                    units *= cfg["staking"]["questionable_stake_mult"]
                units = round(units * 4) / 4
                if units < 0.25:
                    tier, units = None, 0.0
            rows.append({
                **{k: p[k] for k in ("player", "team", "pos", "opp", "game_id", "market", "mean", "median",
                                     "season_avg", "last3_avg", "prior_avg", "opp_factor", "env_factor", "usage_factor",
                                     "usage_note", "situation_factor", "situation_note", "scheme_factor", "scheme_note", "games")},
                "line": line, "side": side, "odds": int(odds), "book": r.get("book", ""),
                "model_prob": round(pm, 3), "market_prob": round(fair_over if side == "Over" else 1 - fair_over, 3),
                "blend_prob": round(prob, 3), "breakeven": round(implied_prob(odds), 3),
                "edge": round(edge, 3), "ev_per_unit": round(ev, 3), "tier": tier, "units": units,
                "injury": status or "",
                **dict(zip(("min_odds", "line_limit"), _take_if(p, line, side, prob, shrink, cfg))),
            })
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["player", "market", "reason"])

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

def apply_caps(evald, cfg):
    if evald.empty:
        return evald
    bets = evald[evald["tier"].notna()].copy()
    if "market_gap" in bets:  # model vs market > 15 pts: usually news/role the stats haven't seen -> hold, don't stake
        bets = bets[bets["market_gap"] <= cfg["model"].get("max_market_gap", 0.15)]
    # carries aren't redistributed, so RB rush props on a team with an Out RB are judgment calls, not auto-stakes
    bets = bets[~bets["usage_note"].fillna("").str.contains("backfield shift")]
    bets["score"] = bets["ev_per_unit"] * bets["units"]
    bets = bets.sort_values("score", ascending=False)
    # one side per player-market
    bets = bets.drop_duplicates(["player", "market"])
    max_pp = cfg["staking"]["max_props_per_player"]
    max_pg = cfg["staking"]["max_units_per_game"]
    max_week = cfg["strategy_profiles"][cfg["strategy_profile"]]["max_units_per_week"] - cfg.get("used_units", 0.0)
    kept, per_player, per_game, total = [], {}, {}, 0.0
    for _, b in bets.iterrows():
        if per_player.get(b["player"], 0) >= max_pp:
            continue
        if per_game.get(b["game_id"], 0) + b["units"] > max_pg:
            continue
        if total + b["units"] > max_week:
            continue
        kept.append(b)
        per_player[b["player"]] = per_player.get(b["player"], 0) + 1
        per_game[b["game_id"]] = per_game.get(b["game_id"], 0) + b["units"]
        total += b["units"]
    return pd.DataFrame(kept).drop(columns="score") if kept else bets.iloc[0:0]

def build_parlays(picks, cfg, top_n=3):
    pc = cfg["parlays"]
    if not pc["enabled"] or picks.empty:
        return []
    order = ["tier_1", "tier_2", "tier_3"]
    pool = picks[picks["tier"].map(order.index) <= order.index(pc["min_leg_tier"])]
    pool = pool.sort_values("ev_per_unit", ascending=False).head(10)
    out = []
    for n in range(2, pc["max_legs"] + 1):
        for combo in itertools.combinations(pool.to_dict("records"), n):
            if not pc["allow_same_game"] and len({c["game_id"] for c in combo}) < n:
                continue
            prob, dec = 1.0, 1.0
            for c in combo:
                prob *= c["blend_prob"]
                dec *= american_to_decimal(c["odds"])
            ev = prob * (dec - 1) - (1 - prob)
            if ev > 0:
                out.append({"legs": [f'{c["player"]} {c["side"]} {c["line"]} {c["market"]} ({c["odds"]:+d})' for c in combo],
                            "n_legs": n, "prob": round(prob, 3), "odds": decimal_to_american(dec),
                            "ev_per_unit": round(ev, 3), "units": pc["units"]})
    out.sort(key=lambda x: x["ev_per_unit"], reverse=True)
    return out[:top_n]

# ==================== tracking ====================

def grade(log_path, stats, season, week):
    if not os.path.exists(log_path):
        print("No pick log yet.")
        return None
    log = pd.read_csv(log_path, dtype={"result": "object"})
    for c in ("actual", "profit_units", "closing_odds"):
        log[c] = pd.to_numeric(log[c], errors="coerce")
    wk = stats[stats["week"] == week].copy()
    if wk.empty:
        print(f"No box scores for week {week} yet (nflverse usually updates the morning after games).")
        return log
    wk["_key"] = wk["player_display_name"].map(norm_name)
    mask = (log["season"] == season) & (log["week"] == week) & log["result"].isna()
    for i in log[mask].index:
        r = log.loc[i]
        if r["market"] not in MARKETS:
            continue  # parlay rows are settled from their legs below
        row = wk[wk["_key"] == norm_name(r["player"])]
        if row.empty:
            log.loc[i, "result"] = "VOID"  # did not play -> books void/refund player props
            log.loc[i, "profit_units"] = 0.0
            continue
        actual = float(row[MARKETS[r["market"]]["cols"]].fillna(0).sum(axis=1).iloc[0])
        log.loc[i, "actual"] = actual
        if actual == r["line"]:
            res = "PUSH"
        elif (actual > r["line"]) == (r["side"] == "Over"):
            res = "WIN"
        else:
            res = "LOSS"
        log.loc[i, "result"] = res
        log.loc[i, "profit_units"] = (win_profit(r) if res == "WIN" else (-r["units"] if res == "LOSS" else 0.0))
    log = settle_parlays(log)
    log.to_csv(log_path, index=False)
    return log

def summarize(log):
    g = log[log["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    if "bet_type" in g:
        g = g[g["bet_type"].fillna("straight") != "leg"]  # legs are rated, not staked
    if g.empty:
        return None
    def agg(d):
        staked = d["units"].sum()
        wins = (d["result"] == "WIN").sum()
        losses = (d["result"] == "LOSS").sum()
        return pd.Series({
            "bets": len(d), "W": wins, "L": losses, "P": (d["result"] == "PUSH").sum(),
            "hit_rate": round(wins / max(wins + losses, 1), 3),
            "avg_pred_prob": round(d["blend_prob"].mean(), 3),
            "units_staked": round(staked, 2), "profit_units": round(d["profit_units"].sum(), 2),
            "roi": round(d["profit_units"].sum() / staked, 3) if staked else 0,
        })
    out = {"overall": agg(g).to_frame().T}
    for col in ("suggested", "vpe_verdict", "bet_type", "tier", "market", "side", "week", "source"):
        if col in g and g[col].notna().any():
            out[col] = g.groupby(col).apply(agg, include_groups=False)
    # closing line value: did we beat the closing price?
    c = g.dropna(subset=["closing_odds"])
    if not c.empty:
        c = c.assign(clv=c["closing_odds"].map(implied_prob) - c["odds"].map(implied_prob))
        out["clv"] = pd.DataFrame({"bets_with_close": [len(c)], "avg_clv_prob_pts": [round(c["clv"].mean(), 4)],
                                   "pct_beat_close": [round((c["clv"] > 0).mean(), 3)]})
    return out

# ==================== chat CLI ====================
CONFIG = {
  "season": 2026, "bankroll": 50.0, "unit_pct_of_bankroll": 0.02, "strategy_profile": "balanced",
  "strategy_profiles": {"conservative": {"max_units_per_week": 8, "allow_tier3": False},
                        "balanced": {"max_units_per_week": 14, "allow_tier3": True},
                        "aggressive": {"max_units_per_week": 20, "allow_tier3": True}},
  "model": {"recency_decay": 0.85, "prior_weight_games": {"default": 3, "pass_tds": 6, "anytime_td": 8, "pass_int": 8},
            "min_games_this_season": 2, "opp_adjust_shrink_games": 8, "opp_adjust_cap": [0.88, 1.12],
            "league_avg_team_total": 22.5,
            "game_env_elasticity": {"pass": 0.5, "rush": 0.4, "rec": 0.45, "td": 0.8, "count": 0.3},
            "model_weight_vs_market": 0.4,
            # Weeks 3-6 backtest, actual / projected: pass att x1.11, completions x1.07, pass yds x1.04 (QBs were
            # projected low -> fake Under edges on QB props). ~75% of each gap applied.
            "mean_calib": {"pass_att": 1.08, "pass_comp": 1.05, "pass_yds": 1.03, "receptions": 1.02,
                           "rush_att": 1.03, "anytime_td": 0.93}, "prob_calibration_shrink": 0.75, "max_market_gap": 0.15,
            "usage_alloc_share_weight": 0.3, "usage_retention_tgt": 0.7, "usage_retention_car": 0.0,
            "usage_model_weight_games": 3, "backup_qb_efficiency": 0.9,
            "situational": {"wind_mph": 15, "cold_f": 35, "wind": {"pass": 0.88, "rec": 0.88, "rush": 1.05},
                            "cold": {"pass": 0.97, "rec": 0.93, "rush": 1.05}, "div": {"pass": 0.98, "rec": 0.98, "rush": 1.0}},
            "scheme": {"enabled": True, "strength": 1.0, "prev_season_weight": 0.5, "coverage_trust": 0.5,
                       "player_shrink": {"tgt": 15, "drop": 30, "car": 20},
                       "defense_shrink": {"blitz": 100, "box": 80, "coverage": 250}, "cap": [0.8, 1.2],
                       "apply_markets": ["rec_yds", "receptions"]}},
  "tiers": {"tier_1": {"min_edge": 0.04, "min_prob": 0.55, "max_units": 3.0},
            "tier_2": {"min_edge": 0.03, "min_prob": 0.52, "max_units": 2.0},
            "tier_3": {"min_edge": 0.025, "min_prob": 0.30, "max_units": 1.0}},
  "staking": {"kelly_fraction": 0.25, "questionable_stake_mult": 0.5, "max_props_per_player": 2, "max_units_per_game": 5},
  "parlays": {"enabled": False, "max_legs": 3, "allow_same_game": False, "min_leg_tier": "tier_1", "units": 0.5},
}

def _cfg(a):
    import copy
    c = copy.deepcopy(CONFIG)
    if a.bankroll: c["bankroll"] = a.bankroll
    if a.profile: c["strategy_profile"] = a.profile
    if getattr(a, "season", None): c["season"] = a.season
    return c

def _setup(a, cfg):
    games = load_games()
    season = cfg["season"]
    g = games[(games["season"] == season) & (games["game_type"] == "REG")]
    week = a.week or int(g[g["home_score"].isna()]["week"].min())
    stats = load_stats(season)
    prev = load_stats(season - 1, max_age_hours=24 * 30)
    inj = load_injuries(season)
    print(f"DATA: season {season}, target week {week}, box scores through week {int(stats['week'].max())}, "
          f"official injury report for week {week}: {'YES' if week in set(inj.get('week', [])) else 'NOT YET'}")
    return games, stats, prev, inj, week

def _overrides(text):
    if not text:
        return None
    rows = [x.split(":") for x in text.split(";") if ":" in x]
    o = pd.DataFrame([(p.strip(), s.strip()) for p, s in rows], columns=["player", "status"])
    o["_key"] = o["player"].map(norm_name)
    return o

def _weather(text):
    w = {}
    for part in (text or "").split(";"):
        if part.count(":") == 2:
            t, wind, temp = part.split(":")
            w[t.strip().upper()] = (float(wind), float(temp))
    return w

def cmd_check(a):
    cfg = _cfg(a)
    set_calib(getattr(a, "calib", None), SPORT)
    games, stats, prev, inj, week = _setup(a, cfg)
    ov = _overrides(a.injuries)
    outs = build_outs(inj, ov, week)
    print(f"USAGE MODEL: {len(outs)} players Out/Doubtful -> targets redistributed to teammates")
    print("SCHEME: loading play-by-play + charting (blitz / coverage shells / box)...")
    plays = (load_plays(cfg["season"]), load_plays(cfg["season"] - 1))
    proj = Projector(stats, prev, games, cfg["season"], week, cfg, outs=outs, weather=_weather(a.weather), plays=plays)
    cfg["used_units"] = used_units(a.log, lambda d: (pd.to_numeric(d.get("season"), errors="coerce") == cfg["season"])
                                   & (pd.to_numeric(d.get("week"), errors="coerce") == week))
    cap = cfg["strategy_profiles"][cfg["strategy_profile"]]["max_units_per_week"]
    print(f"CAP: {cfg['used_units']:g}u already staked in week {week} (pick log) -> {max(cap - cfg['used_units'], 0):g}u of {cap}u left this week")
    lines = pd.read_csv(a.props)
    evald, skipped = evaluate_lines(lines, proj, inj, ov, cfg)
    if not evald.empty:
        evald["market_gap"] = (evald["model_prob"] - evald["market_prob"]).abs().round(3)
    picks = apply_caps(evald, cfg)
    parlays = build_parlays(picks, cfg)
    unit = cfg["bankroll"] * cfg["unit_pct_of_bankroll"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"\nBANKROLL ${cfg['bankroll']:.2f} | 1u = ${unit:.2f} | profile {cfg['strategy_profile']} | projections pure stats; win chance = learned model/market mix (CALIB)")
    print("\n=== ALL LINES (model view) ===")
    if not evald.empty:
        print(evald[["player", "team", "opp", "market", "side", "line", "odds", "mean", "median", "season_avg",
                     "last3_avg", "opp_factor", "env_factor", "usage_factor", "model_prob", "market_prob", "blend_prob", "edge",
                     "tier", "units", "injury", "usage_note", "situation_note", "scheme_note"]].to_string(index=False))
    print("\n=== BETS (after caps) ===")
    print(picks[["tier", "player", "team", "opp", "market", "side", "line", "odds", "book", "mean", "blend_prob", "edge",
                 "units", "min_odds", "line_limit"]].to_string(index=False) if not picks.empty else "none")
    print("  take-if: bet at DraftKings only if DK's line is no worse than line_limit (Over: <=, Under: >=) "
          "AND DK's odds are min_odds or better. Otherwise it's a pass.")
    if not evald.empty:
        held = evald[evald["tier"].notna() & ((evald["market_gap"] > cfg["model"]["max_market_gap"])
                                              | evald["usage_note"].fillna("").str.contains("backfield shift"))]
        held = held.sort_values("market_gap", ascending=False).drop_duplicates(["player", "market"])
        print(f"\n=== HELD ({len(held)}): model vs market > {cfg['model']['max_market_gap']:.0%} or backfield shift - judgment only, not staked ===")
        print(held[["player", "team", "market", "side", "line", "odds", "mean", "season_avg", "last3_avg", "prior_avg",
                    "model_prob", "market_prob", "usage_note"]].to_string(index=False) if not held.empty else "none")
    print("\n(parlays: built by the parlay command)")
    for p in parlays:
        print(f"  {p['odds']:+d} | prob {p['prob']:.1%} | EV/u {p['ev_per_unit']:+.3f} | {p['units']}u | " + " + ".join(p["legs"]))
    print("\n=== SKIPPED ===")
    print(skipped.to_string(index=False) if not skipped.empty else "none")
    if a.out:
        picks.assign(season=cfg["season"], week=week).to_csv(a.out, index=False)
        evald.to_csv(a.out.replace(".csv", "_all.csv"), index=False)
        print(f"\nsaved bets -> {a.out}")
    write_shadow(evald, f"{cfg['season']}-W{week}", a.shadow_out)

def _read_loose_csv(path):
    """Reads CSV text copied from a web page: drops ``` fences, MARKET= lines and blank lines."""
    import io
    keep = [l for l in open(path, encoding="utf-8").read().splitlines()
            if l.strip() and not l.strip().startswith("```") and not l.startswith("MARKET=")]
    return pd.read_csv(io.StringIO("\n".join(keep)), dtype=str).fillna("")

_NFL_ALIAS = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "GNB": "GB",
              "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB"}

def _nfl_team(x):
    s = str(x).strip().upper()
    return _NFL_ALIAS.get(s, s)

def slate_guard(df, a):
    """Odds pages can still show another week's games: keep only rows whose matchup (e.g. 'LA @ LV', 'DET vs GB')
    is a game in the target week; say per market how many were dropped."""
    cfg = _cfg(a)
    try:
        g = load_games()
    except Exception as e:
        print(f"  ! SLATE GUARD: schedule unavailable ({e}); rows kept unchecked")
        return df
    g = g[(g["season"] == cfg["season"]) & (g["game_type"] == "REG")]
    week = a.week or int(g[g["home_score"].isna()]["week"].min())
    games = {frozenset((h, w)) for h, w in zip(g[g["week"] == week]["home_team"], g[g["week"] == week]["away_team"])}
    def pair(m):
        parts = re.split(r"\s+(?:@|vs\.?|at)\s+", str(m).strip(), maxsplit=1, flags=re.I)
        return frozenset(_nfl_team(p) for p in parts) if len(parts) == 2 else None
    pairs = df["matchup"].map(pair)
    if pairs.isna().all():
        print("  ! SLATE GUARD: no readable matchups in the pages; rows kept unchecked")
        return df
    ok = pairs.map(lambda p: p in games if p is not None else False)
    for mk, grp in df.groupby("market"):
        k, n = int(ok[grp.index].sum()), len(grp)
        if k < n:
            print(f"  ! SLATE GUARD {mk}: dropped {n - k} of {n} rows not in week {week}"
                  + (" -> page shows another week: re-fetch it, or pass --week N if the schedule file lags" if k == 0 else ""))
    return df[ok]

def cmd_lines(a):
    """Turns odds-site rows (player,matchup,over_line,over_odds,over_book,under_line,under_odds,under_book)
    into props.csv. Same line both sides -> one row; different lines -> one row per side."""
    out = []
    for spec in a.src:
        mk, path = spec.split("=", 1)
        if canon_market(mk) is None:
            print(f"  ! unknown market {mk}"); continue
        d = _read_loose_csv(path).replace({"even": "+100", "EVEN": "+100", "Even": "+100", "EV": "+100"})
        for _, r in d.iterrows():
            ol, oo, ob = r.get("over_line", ""), r.get("over_odds", ""), r.get("over_book", "")
            ul, uo, ub = r.get("under_line", ""), r.get("under_odds", ""), r.get("under_book", "")
            mu = r.get("matchup", "")
            if ol and ul and oo and uo and float(ol) == float(ul):
                out.append((r["player"], mk, ol, oo, uo, f"{ob}/{ub}", mu))
            else:
                if ol and oo: out.append((r["player"], mk, ol, oo, "", ob, mu))
                if ul and uo: out.append((r["player"], mk, ul, "", uo, ub, mu))
    df = pd.DataFrame(out, columns=["player", "market", "line", "over_odds", "under_odds", "book", "matchup"])
    df = slate_guard(df, a)
    df.drop(columns="matchup").to_csv(a.out, index=False)
    dk = df["book"].str.contains("draftkings", case=False).sum()
    print(f"LINES: {len(df)} rows from {len(a.src)} market pages -> {a.out} ({dk} rows priced at DraftKings, "
          f"rest best-of-market) | by market: " + ", ".join(f"{k} {v}" for k, v in df["market"].value_counts().items()))

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
    print(f"SHADOW: {len(s)} lines -> {path}\n  >>> REQUIRED NEXT STEP: append_values {path} to 'Shadow NFL'!A1 and confirm Calib V2 grew by {len(s)}."
          " Skipping this breaks VPE's learning loop.")


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
    """NFL: actual stat for (player, market) in week `key` (e.g. 2026-W7); None if box scores aren't posted yet."""
    season, week = (int(x) for x in key.split("-W"))
    s = load_stats(season, max_age_hours=1)
    wk = s[s["week"] == week].copy()
    if wk.empty:
        return None
    wk["_key"] = wk["player_display_name"].map(norm_name)
    idx = {k: g.iloc[0] for k, g in wk.groupby("_key")}
    def f(player, market):
        r, mk = idx.get(norm_name(player)), canon_market(market)
        if r is None or mk not in MARKETS:
            return None
        return float(sum(0 if pd.isna(r[c]) else r[c] for c in MARKETS[mk]["cols"]))
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


LOG_COLS = ["season", "week", "logged_at", "player", "team", "opp", "market", "side", "line", "odds", "book", "tier",
            "units", "stake_usd", "source", "model_prob", "blend_prob", "edge", "ev_per_unit", "closing_odds",
            "actual", "result", "profit_units",
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
    """Log bets the user actually placed (from DK screenshots), including bets VPE did not suggest (profit boosts,
    own picks). Every bet is still run through the model at the price really taken."""
    cfg = _cfg(a)
    games, stats, prev, inj, week = _setup(a, cfg)
    ov = _overrides(a.injuries)
    outs = build_outs(inj, ov, week)
    plays = (load_plays(cfg["season"]), load_plays(cfg["season"] - 1))
    proj = Projector(stats, prev, games, cfg["season"], week, cfg, outs=outs, weather=_weather(a.weather), plays=plays)
    unit = cfg["bankroll"] * cfg["unit_pct_of_bankroll"]
    allp = pd.read_csv(a.all) if a.all and os.path.exists(a.all) else None

    def score(player, mk, side, line, odds):
        one = pd.DataFrame([{"player": player, "market": mk, "line": line,
                             "over_odds": odds if side == "Over" else np.nan,
                             "under_odds": odds if side == "Under" else np.nan, "book": "draftkings"}])
        ev, sk = evaluate_lines(one, proj, inj, ov, cfg)
        if ev.empty:
            return None, (sk["reason"].iloc[0] if not sk.empty else "?")
        e = ev.iloc[0]
        mp = market_prob_for(allp, player, mk, side, line, odds)
        pb = blend_leg(float(e["model_prob"]), mp, mk, side)
        return {"prob": pb, "leg_prob": pb,
                "team": e["team"], "opp": e["opp"]}, ""

    slip = slip_keys(a.slip)
    rows = build_log_rows(pd.read_csv(a.placed, dtype=str), score, cfg, unit, slip,
                          {"season": cfg["season"], "week": week})
    if getattr(a, "dry_run", False):
        print_logged(rows)
        print("\nDRY RUN (rate only): nothing logged")
        return
    new = pd.DataFrame(rows).reindex(columns=LOG_COLS)
    if a.log and os.path.exists(a.log):
        old = pd.read_csv(a.log, dtype={"result": "object"})
        new = pd.concat([old, new], ignore_index=True).reindex(columns=LOG_COLS)
    new.to_csv(a.log or "pick_log.csv", index=False)
    print_logged(rows)
    import json
    json.dump(sheet_rows(pd.DataFrame(rows)), open(a.sheet_out, "w"))
    print(f"sheet rows to append -> {a.sheet_out}")
    if slip is None:
        print(f"  ! no slip found at {a.slip}: run `check --out bets.csv` first so VPE picks can be told apart")
    print(f"\nlogged {len(rows)} rows -> {a.log or 'pick_log.csv'} (total rows {len(new)})")

def cmd_board(a):
    cfg = _cfg(a)
    games, stats, prev, inj, week = _setup(a, cfg)
    outs = build_outs(inj, _overrides(a.injuries), week)
    plays = (load_plays(cfg["season"]), load_plays(cfg["season"] - 1)) if a.scheme else None
    proj = Projector(stats, prev, games, cfg["season"], week, cfg, outs=outs, weather=_weather(a.weather), plays=plays)
    b = proj.board()
    if a.usage_only:
        b = b[b["usage_note"].fillna("") != ""]
    if a.team:
        b = b[b["team"].isin([t.strip().upper() for t in a.team.split(",")])]
    if a.player:
        keys = [norm_name(x) for x in a.player.split(";")]
        b = b[b["player"].map(norm_name).isin(keys)]
    if a.market:
        b = b[b["market"] == a.market]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(b.sort_values(["team", "player", "market"])[["player", "team", "pos", "opp", "market", "games", "season_avg",
          "last3_avg", "prior_avg", "opp_factor", "env_factor", "usage_factor", "situation_factor", "mean", "median", "usage_note", "situation_note"] + (["scheme_factor", "scheme_note"] if a.scheme else [])].to_string(index=False))

def cmd_grade(a):
    cfg = _cfg(a)
    stats = load_stats(cfg["season"], max_age_hours=1)
    log = pd.read_csv(a.log)
    for c in ("result", "actual", "profit_units", "closing_odds", "odds_eff", "bet_type", "parlay_id"):
        if c not in log.columns: log[c] = None
    log.to_csv(a.log, index=False)
    week = a.week or int(stats["week"].max())
    log = grade(a.log, stats, cfg["season"], week)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(log[["week", "player", "market", "side", "line", "odds", "odds_eff", "units", "suggested", "actual", "result",
               "profit_units"]].to_string(index=False))
    s = summarize(log)
    if s:
        for k, v in s.items():
            print(f"\n== {k} ==\n{v.to_string()}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Verified Prop Engine - NFL (in-chat edition)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "board", "grade", "lines", "log", "parlay"):
        p = sub.add_parser(name)
        p.add_argument("--week", type=int); p.add_argument("--bankroll", type=float)
        p.add_argument("--profile"); p.add_argument("--season", type=int)
    sub.choices["check"].add_argument("--props", required=True, help="CSV: player,market,line,over_odds,under_odds,book")
    sub.choices["check"].add_argument("--injuries", help='manual overrides "Name:Out;Name:Questionable"')
    sub.choices["check"].add_argument("--out")
    sub.choices["check"].add_argument("--log", help="pick log (pulled from the sheet): units already staked this week count toward the cap")
    sub.choices["board"].add_argument("--team"); sub.choices["board"].add_argument("--player")
    sub.choices["board"].add_argument("--market")
    sub.choices["board"].add_argument("--injuries", help='"Name:Out;Name:Questionable"')
    sub.choices["board"].add_argument("--usage-only", action="store_true", help="only players whose usage changed")
    for name in ("check", "board"):
        sub.choices[name].add_argument("--weather", help='outdoor forecasts "BUF:18:30;CHI:12:40" = team:wind_mph:temp_F')
    sub.choices["board"].add_argument("--scheme", action="store_true", help="add blitz/coverage/box matchup (slower)")
    sub.choices["grade"].add_argument("--log", required=True)
    sub.choices["lines"].add_argument("--src", action="append", required=True,
                                      help="market=file.csv (odds-site rows), repeat per market")
    sub.choices["lines"].add_argument("--out", default="props.csv")
    sub.choices["log"].add_argument("--placed", required=True, help="CSV: player,market,side,line,odds,stake")
    sub.choices["log"].add_argument("--log", default="pick_log.csv")
    sub.choices["log"].add_argument("--injuries"); sub.choices["log"].add_argument("--weather")
    sub.choices["log"].add_argument("--slip", default="bets.csv", help="the slip from this week's check (marks VPE picks)")
    sub.choices["log"].add_argument("--all", default="bets_all.csv", help="this week's bets_all.csv (market prices for parlay legs)")
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
    {"check": cmd_check, "board": cmd_board, "grade": cmd_grade, "lines": cmd_lines, "log": cmd_log,
     "parlay": cmd_parlay, "final": cmd_final, "sheet": cmd_sheet, "shadow": cmd_shadow}[a.cmd](a)

