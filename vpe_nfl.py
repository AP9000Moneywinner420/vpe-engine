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
import sys
import time
import urllib.request

SPORT = "NFL"
VERSION = "NFL v15.33"

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
    s = str(name).lower().replace(".", "").replace(",", " ").replace("'", "").replace("’", "").replace("-", " ")
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
        cand = lt[(lt["team"] == team) & is_out(lt["_key"], lt["team"], self.outs)]
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
                mates = lt[(lt["team"] == team) & ~is_out(lt["_key"], lt["team"], self.outs) & (lt["week"] >= self.week - 3)
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
            eff_f = self.cfg.get("backup_qb_efficiency", 1.0)
            if eff_f != 1.0:
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

    # ---------- v15.15 blind spots: things the book prices into the line that VPE's stats can't see ----------
    # 11/5 BAL vs JAX: 3rd-string QB Joe Fagnano (10/18, 60 yds) in a downpour; VPE took two Overs on BAL receivers.
    # Emergency QB = the team's usual starter is out and the next QB has < 200 pass attempts (2025 + 2026). Measured on
    # 2023-25 (167 team-games): team receiving yards x0.95, receptions x0.94, but swinging -16% to +4% by season. The
    # line already drops for it while VPE's history doesn't, so a model "edge" on an Over there is mostly fake.
    def qb_situation(self, team):
        if not hasattr(self, "_qb_cache"):
            self._qb_cache = {}
        if team in self._qb_cache:
            return self._qb_cache[team]
        res = None
        lt = self.usage.latest
        tq = lt[(lt["team"] == team) & (lt["position"] == "QB")]
        if len(tq):
            out_mask = is_out(tq["_key"], tq["team"], self.usage.outs)
            starters_out = [o for o in self.usage.team_outs(team) if o["pos"] == "QB" and o["att"] >= 0.5]
            if starters_out:
                att = lambda pid: float(self.s.loc[self.s["player_id"] == pid, "attempts"].fillna(0).sum()) + (
                    float(self.prev.loc[self.prev["player_id"] == pid, "attempts"].fillna(0).sum()) if self.prev is not None else 0.0)
                avail = [(att(pid), pid, r["player_display_name"]) for pid, r in tq[~out_mask].iterrows()]
                nxt = max(avail) if avail else (0.0, None, "unknown QB")
                if nxt[0] < self.cfg.get("emergency_qb_att", 200):
                    res = {"name": nxt[2], "att": int(nxt[0]), "out": ", ".join(o["name"] for o in starters_out)}
        self._qb_cache[team] = res
        return res

    def blind_spot(self, team, market):
        """Reason the Over side can't be trusted for this prop (heavy rain/snow, emergency QB), else ''."""
        why = []
        if market in ("pass_yds", "pass_tds", "pass_comp", "pass_att", "rec_yds", "receptions", "rush_rec_yds"):
            g = self.games[(self.games["week"] == self.week)
                           & ((self.games["home_team"] == team) | (self.games["away_team"] == team))]
            if len(g) and str(g.iloc[0].get("roof", "outdoors")) not in ("dome", "closed"):
                for t in (g.iloc[0]["home_team"], g.iloc[0]["away_team"]):
                    w = self.weather.get(t)
                    if w and len(w) > 2 and w[2]:
                        why.append(f"heavy {w[2]} forecast")
                        break
            q = self.qb_situation(team)
            if q:
                why.append(f"emergency QB {q['name']} ({q['att']} career att; {q['out']} out)")
        return "; ".join(why)

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
                wind, temp = self.weather[t][:2]
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
        q = self.qb_situation(team) if market in ("rec_yds", "receptions", "rush_rec_yds") and pos != "QB" else None
        if q:
            eq = self.cfg.get("emergency_qb_efficiency", 0.95)
            sit_f *= eq
            sit_note = "; ".join(x for x in (sit_note, f"emergency QB {q['name']} x{eq}") if x)
        blind = self.blind_spot(team, market)
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
            "situation_factor": round(sit_f, 3), "situation_note": sit_note, "blind_spot": blind,
            "scheme_factor": round(sch_f, 3), "scheme_note": sch_note,
            "mean": round(mean, 2), "sd": None if sd is None else round(sd, 2), "dist": m["dist"],
            "median": round(self.median(mean, sd, m["dist"]), 1),
            "_x": {"n": n, "wmean": _r(wmean, 2), "decay": self.cfg["recency_decay"], "prior": _r(prior, 2), "prior_n": prior_n,
                   "k": _r(k, 2), "base": _r(base, 2), "use_add": _r(use_add, 2), "eff": _r(eff_f, 3), "sit": _r(sit_f, 3),
                   "mc": _r(mc, 3), "itt": _r(mu["itt"], 1), "itt_avg": _r(self.team_avg_itt.get(team), 1),
                   "sd": None if sd is None else _r(sd, 2), "dist": m["dist"]},
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
                out.append({k_: v_ for k_, v_ in p.items() if k_ != "_x"})
        return pd.DataFrame(out)

# ==================== selection ====================

ASSUMED_HOLD_ONE_SIDED = 0.07  # one-sided DK markets (anytime TD, milestones like 60+ yds): only for the info column


def one_sided_hold(odds):
    """v15.26: MEASURED on 3,793 graded one-sided NFL lines (Shadow tab, weeks 9-10): real hold x0.04-0.07 up to +100,
    x0.10 at +100..+250, and x0.34 at +250 and longer (1,246 lines: the price implied 15.2%, 11.3% hit; VPE read 14.2%).
    The flat 7% made every longshot look ~3 pts better than it was (Goedert +250 and Lane +265 were Tier 3 picks)."""
    o = float(odds)
    return 0.30 if o >= 250 else (0.10 if o >= 100 else 0.06)

LONG_TERM = {"Ir", "Injured Reserve", "Out For Season", "Pup", "Nfi", "Suspended", "Reserve"}

def _override_for(overrides, key, team):
    if overrides is None or overrides.empty:
        return None
    o = overrides[(overrides["_key"] == key) & (overrides["team"].isna() | (overrides["team"] == team))]
    return None if o.empty else str(o.iloc[-1]["status"]).title()

def _official_for(injuries, key, team, week):
    if injuries is None or injuries.empty:
        return None
    inj = injuries[(injuries["week"] == week) & (injuries["team"] == team)]
    hit = inj[inj["full_name"].map(norm_name) == key]
    if hit.empty:
        return None
    s = hit.iloc[-1]["report_status"]
    return None if pd.isna(s) else str(s).title()

def injury_status(injuries, overrides, name, team, week):
    """Official report wins. The chat's override string only decides for long-term absences (IR, out for season,
    PUP, suspended) or when the official report doesn't list the player yet (reports post Wed-Fri). 10/25: a stale
    override 'Questionable' hid a real official 'Out'."""
    key = norm_name(name)
    ov = _override_for(overrides, key, team)
    if ov in LONG_TERM:
        return ov
    off = _official_for(injuries, key, team, week)
    return off if off is not None else ov

def build_outs(injuries, overrides, week):
    """Players the usage model treats as out, as 'name|TEAM' keys (or plain 'name' when an override has no team).
    Same precedence as injury_status: the official report wins unless the override is a long-term absence or the
    report doesn't list the player."""
    outs = set()
    listed = set()
    if injuries is not None and not injuries.empty:
        wk = injuries[injuries["week"] == week]
        listed = {(norm_name(n), t) for n, t in zip(wk["full_name"], wk["team"])}
        inj = wk[wk["report_status"].isin(["Out", "Doubtful"])]
        outs |= {f"{norm_name(n)}|{t}" for n, t in zip(inj["full_name"], inj["team"])}
    if overrides is not None and not overrides.empty:
        for _, o in overrides.iterrows():
            st_ = str(o["status"]).title()
            k, t = o["_key"], (None if pd.isna(o["team"]) else o["team"])
            on_report = any(n == k and (t is None or tt == t) for n, tt in listed)
            if st_ in LONG_TERM or (st_ in ("Out", "Doubtful") and not on_report):
                outs.add(f"{k}|{t}" if t else k)
    return outs

def is_out(keys, teams, outs):
    """Vectorized: name key (+ team) is in the outs set built by build_outs."""
    return pd.Series([f"{k}|{t}" in outs or k in outs for k, t in zip(keys, teams)], index=getattr(keys, "index", None))

def cal_prob(p, shrink):
    """Raw P(over) -> calibrated. With a prob_map in CONFIG (tails-only fix, like NBA v2.3) the map is used for every
    market except anytime TD (shrink 1.0 there: Poisson already calibrated); otherwise the flat shrink."""
    m = CONFIG["model"].get("prob_map")
    if m and shrink != 1.0:
        return float(np.interp(p, m["x"], m["y"]))
    return 0.5 + (p - 0.5) * shrink

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
    po0 = cal_prob(Projector.prob_over(p, line), shrink)
    for _ in range(25):
        po = cal_prob(Projector.prob_over(p, L), shrink)
        po = (1 - prob if side == "Under" else prob) + _mw(p["market"]) * (po - po0)  # blended, moved with the line
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
        if p["game_id"] in cfg.get("started", set()):
            skipped.append((p["player"], mk, "game already played / kicked off"))
            continue
        status = injury_status(injuries, overrides, p["player"], p["team"], proj.week)
        if status in ("Out", "Doubtful", "Ir"):
            skipped.append((p["player"], mk, f"injury: {status}"))
            continue

        line = float(r["line"])
        # calibration shrink from backtest.py (not for TD markets: Poisson is already calibrated there)
        shrink = 1.0 if mk == "anytime_td" else cfg["model"].get("prob_calibration_shrink", 1.0)
        raw_over = Projector.prob_over(p, line)
        p_model_over = cal_prob(raw_over, shrink)
        oo, uo = r.get("over_odds"), r.get("under_odds")
        fair_how = "2way"
        if pd.notna(oo) and pd.notna(uo):
            fair_over, _ = no_vig(oo, uo)
        else:
            fair_how = "1way"
            side_odds = oo if pd.notna(oo) else uo
            fair = implied_prob(side_odds) / (1 + one_sided_hold(side_odds))
            fair_over = fair if pd.notna(oo) else 1 - fair
        p_over = blend_leg(p_model_over, fair_over, mk, "Over")
        xj = _json.dumps({**p["_x"], "oo": None if pd.isna(oo) else int(oo), "uo": None if pd.isna(uo) else int(uo),
                          "p": why_prob(p_model_over, fair_over, mk, fair_how, raw_over)})  # v15.16: for `explain`

        for side, odds, prob, pm in (("Over", oo, p_over, p_model_over), ("Under", uo, 1 - p_over, 1 - p_model_over)):
            if pd.isna(odds):
                continue
            edge = prob - implied_prob(odds)
            ev = ev_per_unit(prob, odds)
            tier = None if _killed(mk, side, odds) else _tier(edge - (EST_MARGIN if fair_how == "1way" else 0.0), prob, cfg)  # v15.28 KILL
            if tier == "tier_3" and not profile["allow_tier3"]:
                tier = None
            if tier and side == "Over" and p.get("blind_spot"):
                tier = None  # v15.15: the line already reflects rain / an emergency QB; VPE's history doesn't
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
                                     "usage_note", "situation_factor", "situation_note", "blind_spot", "scheme_factor", "scheme_note", "games")},
                "line": line, "side": side, "odds": int(odds), "book": r.get("book", ""),
                "model_prob": round(pm, 3), "market_prob": round(fair_over if side == "Over" else 1 - fair_over, 3),
                "blend_prob": round(prob, 3), "breakeven": round(implied_prob(odds), 3),
                "edge": round(edge, 3), "ev_per_unit": round(ev, 3), "tier": tier, "units": units,
                "injury": status or "",
                **dict(zip(("min_odds", "line_limit"), _take_if(p, line, side, prob - (EST_MARGIN if fair_how == "1way" else 0.0), shrink, cfg))), "x": xj,
                "mkt_est": fair_how == "1way",
            })
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["player", "market", "reason"])

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
    on = len(b) > 0 and b.isin(["draftkings", "fanduel"]).all()
    LOCK["on"], LOCK["books"] = bool(on), " + ".join(sorted({BOOK_NAME_L.get(x, x) for x in b.unique()})) if on else ""
    return LOCK["on"]

def lock_from(allp):
    """Read the lock flag that `check` saved into bets_all.csv."""
    v = allp.get("locked") if allp is not None and len(allp) else None
    LOCK["on"] = bool(v is not None and v.astype(str).str.lower().isin(["true", "1"]).any())
    LOCK["books"] = str(allp["locked_books"].dropna().iloc[0]) if LOCK["on"] and "locked_books" in allp and allp["locked_books"].notna().any() else ""
    return LOCK["on"]

BOOK_NAME_L = {"draftkings": "DraftKings", "fanduel": "FanDuel"}

def apply_caps(evald, cfg):
    if evald.empty:
        return evald
    bets = evald[evald["tier"].notna()].copy()
    if "market_gap" in bets:  # model vs market > 15 pts: usually news/role the stats haven't seen -> hold, don't stake
        bets = bets[bets["market_gap"] <= cfg["model"].get("max_market_gap", 0.15)]
    # carries aren't redistributed, so RB rush props on a team with an Out RB are judgment calls, not auto-stakes
    bets = bets[~bets["usage_note"].fillna("").str.contains("backfield shift")]
    # v2.11: among bets that clear the edge bar at their price, the strongest prop (win chance) gets the cap room first
    bets["score"] = bets["blend_prob"] + 1e-3 * bets["ev_per_unit"] * bets["units"]
    bets = bets.sort_values("score", ascending=False)
    # one side per player-market
    bets = bets.drop_duplicates(["player", "market"])
    max_pp = cfg["staking"]["max_props_per_player"]
    max_pg = cfg["staking"]["max_units_per_game"]
    max_week = cfg["strategy_profiles"][cfg["strategy_profile"]]["max_units_per_week"] - cfg.get("used_units", 0.0)
    kept, per_player, per_game, total, ng = [], {}, {}, 0.0, {}
    for _, b in bets.iterrows():
        if per_player.get(b["player"], 0) >= max_pp:
            continue
        if per_game.get(b["game_id"], 0) + b["units"] > max_pg:
            continue
        if total + b["units"] > max_week:
            continue
        if ng.get(b["game_id"], 0) >= cfg["staking"].get("max_bets_per_game", 99):  # NBA 10/25: 69% of $ on one game
            continue
        ng[b["game_id"]] = ng.get(b["game_id"], 0) + 1
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
            # v15.6 two-season walk-forward check (2024 + 2025, ~20k player-weeks): pass_att x1.08 and rush_att x1.03
            # over-shot in BOTH seasons (Overs predicted 4-12 pts too often) -> 1.0; pass_comp mixed -> kept.
            # Tails-only probability map tested like NBA: made NFL worse both ways -> not used (flat 0.75 shrink stays).
            "mean_calib": {"pass_att": 1.0, "pass_comp": 1.05, "pass_yds": 1.03, "receptions": 1.02,
                           "rush_att": 1.0, "anytime_td": 0.93}, "prob_calibration_shrink": 0.75, "max_market_gap": 0.15,
            "usage_alloc_share_weight": 0.3, "usage_retention_tgt": 0.7, "usage_retention_car": 0.0,
            "usage_model_weight_games": 3, "backup_qb_efficiency": 1.0, "emergency_qb_att": 200,
            "emergency_qb_efficiency": 0.95,
            "situational": {"wind_mph": 15, "cold_f": 35, "wind": {"pass": 0.88, "rec": 0.88, "rush": 1.05},
                            "cold": {"pass": 0.97, "rec": 0.93, "rush": 1.05}, "div": {"pass": 0.98, "rec": 0.98, "rush": 1.0}},
            "scheme": {"enabled": True, "strength": 1.0, "prev_season_weight": 0.5, "coverage_trust": 0.5,
                       "player_shrink": {"tgt": 15, "drop": 30, "car": 20},
                       "defense_shrink": {"blitz": 100, "box": 80, "coverage": 250}, "cap": [0.8, 1.2],
                       "apply_markets": ["rec_yds", "receptions"]}},
  "tiers": {"tier_1": {"min_edge": 0.04, "min_prob": 0.55, "max_units": 3.0},
            "tier_2": {"min_edge": 0.03, "min_prob": 0.52, "max_units": 2.0},
            "tier_3": {"min_edge": 0.025, "min_prob": 0.30, "max_units": 1.0}},
  "staking": {"kelly_fraction": 0.25, "questionable_stake_mult": 0.5, "max_props_per_player": 2, "max_units_per_game": 5,
              "max_bets_per_game": 2},
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
    print(f"DATA [{VERSION}]: season {season}, target week {week}, box scores through week {int(stats['week'].max())}, "
          f"official injury report for week {week}: {'YES' if week in set(inj.get('week', [])) else 'NOT YET'}")
    return games, stats, prev, inj, week

def _overrides(text):
    """"Name:Status[:TEAM];..." -> rows. TEAM is optional; with it, a same-named player on another team isn't hit."""
    if not text:
        return None
    rows = [[y.strip() for y in x.split(":")] for x in text.split(";") if ":" in x]
    o = pd.DataFrame([(r[0], r[1], _nfl_team(r[2]) if len(r) > 2 and r[2] else None) for r in rows],
                     columns=["player", "status", "team"])
    o["_key"] = o["player"].map(norm_name)
    return o

def _weather(text):
    """"BUF:18:30;BAL:8:55:rain" = team:wind_mph:temp_F[:rain|snow] (4th field = heavy rain or snow forecast)."""
    w = {}
    for part in (text or "").split(";"):
        bits = [b.strip() for b in part.split(":")]
        if len(bits) in (3, 4) and bits[0]:
            prec = bits[3].lower() if len(bits) == 4 and bits[3].lower() in ("rain", "snow") else ""
            w[bits[0].upper()] = (float(bits[1]), float(bits[2]), prec)
    return w

def cmd_check(a):
    cfg = _cfg(a)
    set_calib(getattr(a, "calib", None), SPORT)
    games, stats, prev, inj, week = _setup(a, cfg)
    cfg["started"] = started_games(games[(games["season"] == cfg["season"]) & (games["week"] == week)]) if not getattr(a, "ignore_started", False) else set()
    ov = _overrides(a.injuries)
    outs = build_outs(inj, ov, week)
    print(f"USAGE MODEL: {len(outs)} players Out/Doubtful -> targets redistributed to teammates")
    print("SCHEME: loading play-by-play + charting (blitz / coverage shells / box)...")
    plays = (load_plays(cfg["season"]), load_plays(cfg["season"] - 1))
    proj = Projector(stats, prev, games, cfg["season"], week, cfg, outs=outs, weather=_weather(a.weather), plays=plays)
    cfg["used_units"] = used_units(a.log, lambda d: (pd.to_numeric(d.get("season"), errors="coerce") == cfg["season"])
                                   & (pd.to_numeric(d.get("week"), errors="coerce") == week),
                                   unit_usd=cfg["bankroll"] * cfg["unit_pct_of_bankroll"])
    cap = cfg["strategy_profiles"][cfg["strategy_profile"]]["max_units_per_week"]
    print(f"CAP: {cfg['used_units']:g}u already staked in week {week} (pick log) -> {max(cap - cfg['used_units'], 0):g}u of {cap}u left this week")
    lines = pd.read_csv(a.props)
    if detect_lock(lines["book"] if "book" in lines else []):
        print(f"PRICES LOCKED: every line is from your {LOCK['books']} recording/screenshots -> priced as recorded; every pick still shows its take-if (worst price still worth it) in case the price moves before you bet")
    evald, skipped = evaluate_lines(lines, proj, inj, ov, cfg)
    if not evald.empty:
        evald["market_gap"] = (evald["model_prob"] - evald["market_prob"]).abs().round(3)
    picks = apply_caps(evald, cfg)
    if PARLAY.get("boost_only") and not getattr(a, "all_picks", False) and not picks.empty:
        print(f"NFL MODE: boosts only -> {len(picks)} model straight pick(s) withheld (see CALIB). They stay in bets_all.csv; "
              "`parlay` builds the boost parlays and the BEST AVAILABLE table.")
        picks = picks.iloc[0:0]
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
    if True:  # v2.16: take-if on recordings too (prices move before bets go in)
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
        evald = evald.assign(locked=LOCK["on"], locked_books=LOCK["books"])
        evald.to_csv(a.out.replace(".csv", "_all.csv"), index=False)
        os.makedirs("runs", exist_ok=True)
        evald.to_csv(f"runs/bets_all_W{week}_{pd.Timestamp.now(tz='America/New_York'):%a%H%M%S}.csv", index=False)
        print(f"\nsaved bets -> {a.out}")
    write_shadow(evald, f"{cfg['season']}-W{week}", a.shadow_out, getattr(a, "shadow_saved", None))

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

def started_games(g):
    """Game ids already played or kicked off: a final score in the schedule OR kickoff time in the past (ET).
    10/23: NE@CHI was played Thursday but the schedule had no score yet, so it still looked upcoming."""
    kick = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].fillna("13:00").astype(str), errors="coerce")
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    return set(g.loc[g["result"].notna() | g["home_score"].notna() | (kick <= now), "game_id"])

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
    wk = g[g["week"] == week]
    done = started_games(wk)
    if done:
        print(f"  ! SLATE GUARD: {len(done)} week-{week} game(s) already played/kicked off -> their props dropped: "
              + ", ".join(f"{w} @ {h}" for h, w, gid in zip(wk["home_team"], wk["away_team"], wk["game_id"]) if gid in done))
    wk = wk[~wk["game_id"].isin(done)]
    games = {frozenset((h, w)) for h, w in zip(wk["home_team"], wk["away_team"])}
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
    df = merge_best(df)
    df.to_csv(a.out, index=False)  # matchup kept: the RUN STATUS gate reads it
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

SLIP_MEDAL = {}


def slip_keys(path):
    """(player, market, side) of every bet on the slip the last `check` produced; None if no slip was saved."""
    if not path or not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    kinds = s["kind"] if "kind" in s else ["pick"] * len(s)
    if "medal" in s:   # v15.26: Gold (#1-3) / Silver (#4-6) of that run's leg ranking -> stamped on the logged row
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
          "min_prob": 0.53, "max_gap": 0.15,          # model vs market gap: bigger = model probably missing news
          "min_odds": -200, "max_odds": 150, "min_minutes": 20, "min_minutes_ratio": 0.85,
          "books": ["draftkings", "fanduel"],
          "haircut": 0.05,           # price from another book -> assume ours pays 0.05 less in decimal odds (~6-8 cents)
          "junk_lines": {"rush_yds": 10.5, "rec_yds": 10.5, "receptions": 1.5, "rush_att": 3.5},  # line below -> skip
          "max_under_share": 0.45,   # v15.26 HARD RULE: Unders <= 45% of a parlay's legs, never under 1 (2-4 legs: 1, 5-6: 2) unless the math is
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
LEAN = {"min_dec": 1.0 + 100.0 / 129.0,  # v15.28 KILL: no lean at -130 or shorter (was -150)
        "_was_min_dec": 1.0 / 0.6,  # v2.20: no lean shorter than -150 (11/16: 5-9, -118.7u, mostly -180 to -200 favourites)
        "min_edge": -0.04,   # leans may be up to 4 pts below break-even (anything worse is listed as a pass)
        "min_prob": 0.50, "straights": 3, "units": 0.5, "parlay_target": 4.0, "parlay_max_legs": 3,
        # v15.31 (Adrian 11/23, with NBA v2.38): LEANS ARE OFF THE SLIP. Through 11/23 the leans went 7-12 across both
        # sports: a 37% hit rate at prices that mostly needed 60%+. Nothing is suggested, staked or logged as a "VPE lean"
        # any more. Hard rule 3 (never a bare "no bets") is kept by the BEST AVAILABLE table (ported from NBA v2.29, which
        # also ends the old "place none" line): rate-only, no stake, not a VPE pick. "off": False brings the LEAN plays back.
        "off": True, "record": "the old LEAN plays went 7-12 through 11/23"}

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
    """v15.31 (hard rule 3: never a bare "no bets"; from NBA v2.29): the closest lines to break-even, one per game,
    rate-only. KILL-list slices and prices at -130 or shorter are left out unless nothing else exists."""
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
        print("  BEST AVAILABLE: no scoreable line this week (every line is held, injured or outside the price range): say so, "
              "and offer a rerun on fresh lines")
        return []
    print(f"  BEST AVAILABLE (rate-only: NOT VPE picks, no stake suggested; if played they are logged as his own picks). "
          f"This week's closest lines to break-even, one per game. For context, {LEAN['record']}:")
    for _, r in ba.iterrows():
        print(f"    {r['player']} ({r['team']} vs {r['opp']}) {r['side']} {r['line']} {r['market']} "
              f"({int(r['bet_odds']):+d} at {r['bet_at']}) | win {r['leg_prob']:.1%} | edge {r['leg_edge']:+.1%}"
              + (" | on the KILL list (nothing else to show)" if r["on_kill"] else ""))
    return list(zip(ba["player"], ba["market"], ba["side"]))


def _under_max(n, share=None):
    """v15.27: most Unders allowed in an n-leg parlay: 45% of the legs, but never less than 1 (Adrian 11/16: 1 of 2 is fine).
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
    if "blind_spot" in d:  # v15.15: no Over legs on rain / emergency-QB props
        d = d[~((d["side"] == "Over") & d["blind_spot"].fillna("").astype(str).str.len().gt(0))]
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
    # v15.27: pick shrink (see PICK): a picked leg's chance moves a share k toward what the price itself implies
    d["two_sided"] = [k_ in _two for k_ in zip(d["player"], d["market"], pd.to_numeric(d["line"], errors="coerce"))]
    d["leg_prob_raw"] = d["leg_prob"]
    _fair = np.array([_price_fair(o, t, mp) for o, t, mp in zip(d["odds"], d["two_sided"], d["market_prob"])])
    d["leg_prob"] = np.round(d["leg_prob"] - _pick_k() * np.clip(d["leg_prob"] - _fair, 0, None), 3)
    d["leg_edge"] = (d["leg_prob"] - 1 / d["best_dec"]).round(3)
    d["_estm"] = (d["mkt_est"].astype(str).str.lower().isin(["true", "1"]) * EST_MARGIN) if "mkt_est" in d else 0.0
    d = d[(d["leg_edge"] - d["_estm"] >= (PARLAY["min_edge"] if min_edge is None else min_edge))
          & (d["leg_prob"] >= (PARLAY["min_prob"] if min_prob is None else min_prob))]
    if min_edge is None:   # v15.28 KILL: an unboosted parlay never uses a leg at -130 or shorter (boost builds may, down to -200)
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
    PROFIT["fee"] = float(getattr(a, "fee", 0) or 0) / 100
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
                        "take_min", "line_limit") if c in pool] + (["held_note"] if "held_note" in pool and pool["held_note"].astype(str).str.len().gt(0).any() else [])
    print(f"LEG POOL: {len(pool)} legs qualify | leg prob = {PARLAY['model_weight']:.0%} model + "
          f"{1 - PARLAY['model_weight']:.0%} market | edge >= {PARLAY['min_edge']:.1%} at DK/FD price | "
          f"gap <= {PARLAY['max_gap']:.0%} | odds {PARLAY['min_odds']} to +{PARLAY['max_odds']}")
    print(pool.head(6)[cols].to_string(index=False) if len(pool) else "none")
    print(f"STAKE: ${stake:.2f} per parlay ({PARLAY['stake_pct']:.0%} of ${bank:.0f} bankroll unless --stake given)"
          + (f" | {boost:g}% profit boost -> its own BOOST parlay at {BOOK_NAME.get(bbook, bbook)}" if boost else ""))
    keys, built = [], []
    if PARLAY.get("boost_only") and not getattr(a, "all_picks", False):
        print("NFL MODE: boosts only -> no model parlays (the ~4x / ~10x need a model edge); boost parlays + the BEST AVAILABLE table below")
    for name, target in ([] if PARLAY.get("boost_only") and not getattr(a, "all_picks", False) else PARLAY["targets"]):
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
        r_ = print_boost_spec(spec, allp, stake, cols, keys, used, a)
        if r_:
            bps.append(r_)
            built.append(r_)
            used += [l[0] for l in r_["legs"]]
    bp = next((b for b in bps if not b.get("lean")), bps[0] if bps else None)
    excl = [x for x in (a.exclude or "").split(";") if x.strip()]
    lean_keys = [tuple(l[:3]) for b in bps if b.get("lean") for l in b["legs"]]
    if LEAN.get("off"):
        lean_keys = []   # v15.31: a boost build that is not +EV is reference only, never logged as a VPE lean
    if not any(b["target"] for b in built) and not LEAN.get("off"):
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
    if not has_bets and LEAN.get("off"):
        # v15.31: no LEAN plays. The BEST AVAILABLE table keeps hard rule 3; its rows ride in leans.csv (0u) so the menu
        # and the links can still show them, labeled rate-only. They are NOT on the slip.
        print("\n=== BEST AVAILABLE STRAIGHTS (rate-only, not VPE picks) ===")
        _print_best_available(allp, excl)
        ls = best_available(allp, excl)
        if len(ls):
            ls = ls.assign(units=0.0, stake_usd=0.0, cost_usd=0.0)
        ls.to_csv(getattr(a, "leans_out", None) or "leans.csv", index=False)
    elif not has_bets:
        unit_usd = bank * CONFIG["unit_pct_of_bankroll"]
        ls = lean_straights(allp, excl, unit_usd)
        print(f"\n=== LEAN - BEST AVAILABLE STRAIGHTS: VPE says none of tonight's lines beat the book; "
              f"if you play, these are closest ({LEAN['units']:g}u each, one per game) ===")
        if len(ls):
            for _, r in ls.iterrows():
                print(f"  {r['player']} ({r['team']} vs {r['opp']}) {r['side']} {r['line']} {r['market']} "
                      f"({int(r['bet_odds']):+d} at {r['bet_at']}) | win {r['leg_prob']:.1%} | edge {r['leg_edge']:+.1%} | "
                      f"{r['units']:g}u (${r['stake_usd']:.2f})" + f" | take only at {int(r['bet_odds']):+d} or better")
            lean_keys += list(zip(ls["player"], ls["market"], ls["side"]))
            lean_straight = set(zip(ls["player"], ls["market"], ls["side"]))
        else:
            print(f"  none: no line within {-LEAN['min_edge']:.0%} of break-even")
        ls.to_csv(getattr(a, "leans_out", None) or "leans.csv", index=False)
    elif os.path.exists(getattr(a, "leans_out", None) or "leans.csv"):
        os.remove(getattr(a, "leans_out", None) or "leans.csv")
    # ONE-PARLAY RULE (10/21-10/26: parlays were the whole loss, -$15.14 of -$15.89): recommend ONE parlay a night:
    # the BOOST parlay if there is one (+EV at market odds alone), else the biggest VPE could build (~10x), else ~4x.
    if bps:   # v15.28 BOOST RANKER: every boost scored by true EV (after the pick shrink), best first
        print("\nBOOST RANK (" + ("value at FAIR prices after the books' cut and any fee, no model opinion" if PROFIT["on"] else "true EV after the pick shrink")
              + "; vig = what the legs' hold costs before the boost pays it back):")
        _evk = "ev_fair" if PROFIT["on"] else "ev"
        for i_, b in enumerate(sorted(bps, key=lambda x: -x.get(_evk, -9)), 1):
            ev_ = b.get(_evk, 0.0)
            if PROFIT["on"]:
                verdict = "SAVE IT (not +EV at fair prices: reference only)" if b.get("lean") or ev_ < 0 else ("USE IT" if ev_ >= PROFIT["use"] else "THIN: small plus, use only if the boost would expire unused")
            else:
                verdict = "USE IT" if ev_ >= 0.05 and not b.get("lean") else ("THIN: about break-even" if ev_ > -0.02 and not b.get("lean") else ("SAVE IT (not +EV: reference only)" if LEAN.get("off") else "SAVE IT / lean only"))
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
    elif built and ((real and not PROFIT["on"]) or (bp and not bp.get("lean")) or not LEAN.get("off")):
        # boost parlay (real, then lean) > biggest real parlay > lean parlay (v15.31: never a lean build)
        rec = (bp if bp and not bp.get("lean") else None) or (max(real, key=lambda b: b["target"]) if real else None) \
            or bp or built[-1]
        rec["recommended"] = True
        tag = (" - LEAN: no edge, the least-bad play with what we're given; logged as a VPE lean"
               if rec.get("lean") else "")
        print(f"\nONE-PARLAY RULE: place ONLY the {rec['name']} parlay at {BOOK_NAME[rec['book']]} "
              f"({rec['odds']:+d}, hit {rec['hit']:.1%}, ${rec['stake_usd']:.2f}{', take-if ' + format(rec['take_if'], '+d') if rec.get('take_if') is not None and not rec.get('lean') else ''}){tag}. Any other parlay above = reference only.")
    elif PROFIT["on"] and real:
        print("\nONE-PARLAY RULE (PROFIT MODE): no boost worth using, so VPE recommends NO parlay. The ~4x / ~10x builds above are "
              "REFERENCE ONLY (graded in the Shadow builds tab): without a boost every leg pays the books' cut and nothing pays it back. "
              "If he plays one anyway it is logged as his own pick. The BEST AVAILABLE table above is rate-only.")
    else:
        print("\nONE-PARLAY RULE: no VPE parlay this week (nothing is +EV). The plays are the BEST AVAILABLE lines "
              "printed above (rate-only); any boost build above is reference only (SAVE the boost).")
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
        print("NFL MODE: BOOSTS ONLY - on 3,000+ graded lines the most accurate mix is 0% model, so VPE's NFL"
              " numbers add nothing over the market. No straight picks / model parlays; boost parlays and the rate-only BEST AVAILABLE table only."
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
         "tail_bins": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0], "bet_edge": 0.025, "gate_n": 150, "est_hold": 1.045}
LEARNED = {"tail": None, "gate": {}, "pick": None, "kill": None}
# v15.27 PICK SHRINK (approved 11/16). The legs VPE picks for parlays and boosts are the lines where its chance sits furthest
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
    h = one_sided_hold(odds)
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
    # v15.27: pick shrink. Legs the pool would take (chance >= 50%, edge >= -3% at a real price): how much of VPE's lean
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
    # v15.29: retune every 25 graded bets (KILL list), logged to the 'Tune log' tab
    pk_ = {str(r["key"]) for (kd, _), r in prev.items() if kd == "kill" and str(r["value"]).strip() in ("1", "1.0")}
    if not any(kd == "kill" for (kd, _) in prev):
        pk_ = {"odds:short"}
    kill_, trows = (retune(a.log, getattr(a, "tune_log", None), pk_, today) if getattr(a, "log", None) and os.path.exists(a.log) else (pk_, []))
    for k_ in sorted(kill_ | pk_):
        out.append([SPORT, "kill", k_, 1 if k_ in kill_ else 0, "", "KILL list: no unboosted tier / lean / parlay leg here" if k_ in kill_ else "lifted", today])
    json.dump(trows, open(getattr(a, "tune_out", None) or "tune_new.json", "w"))
    if trows:
        print(f"  Tune log: {len(trows)} rows -> {getattr(a, 'tune_out', None) or 'tune_new.json'} (append to 'Tune log'!A1)")
    json.dump(out, open(a.out, "w"))
    print(f"Learned table -> {a.out}: clear 'Learned {SPORT}'!A:G, then write it at 'Learned {SPORT}'!A1. Save it as learned_tab.json"
          f" next to calib_tab.json before check/parlay/final/takeif.")


# ---------- MENU (v15.8): one view of the week's best options (no same-game sim in NFL yet) ----------
MENU = {"top_overall": 10, "min_edge": 0.015, "props_per_game": 3}


def cmd_menu(a):
    import json
    set_calib(getattr(a, "calib", None), SPORT)
    allp = pd.read_csv(a.all)
    lock_from(allp)
    pool = parlay_pool(allp)
    print("\n" + "=" * 70 + "\nVPE MENU - best options this week (ranked inside each section)\n" + "=" * 70)
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
    ov = ov[ok].sort_values(["blend_prob", "edge"], ascending=False).head(MENU["top_overall"])
    if ov.empty:
        print("   none qualify")
    for i, (_, r) in enumerate(ov.iterrows(), 1):
        star = "★" if (r["player"], r["market"], r["side"]) in on_slip else " "
        tk = f" | take if odds >= {int(r['min_odds']):+d}" if pd.notna(r.get("min_odds")) else ""
        print(f"  {i:>2}. {star} {r['player']} ({r['team']} vs {r['opp']}) {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}, {r['book']})"
              f" | win {r['blend_prob']:.1%} | edge {r['edge']:+.1%}{tk}")
    print("\n1) BEST STRAIGHT BETS (the staked slip; max 2 per game)")
    if a.bets and os.path.exists(a.bets):
        b = pd.read_csv(a.bets)
        b = b[b["tier"].notna()] if "tier" in b else b
        for _, r in b.sort_values(["blend_prob", "edge"], ascending=False).iterrows():
            print(f"   {r['player']} ({r['team']}) {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}) vs {r['opp']}"
                  f" | win {r['blend_prob']:.1%} | edge {r['edge']:+.1%} | {r['units']}u")
        if b.empty:
            print("   none beat the book this week")
            lf = "leans.csv"
            if os.path.exists(lf) and len(pd.read_csv(lf)):
                print("   BEST AVAILABLE (rate-only: not VPE picks, no stake suggested):" if LEAN.get("off") else "   LEAN - best available (no edge; if you play, play these):")
                for _, r in pd.read_csv(lf).iterrows():
                    print(f"   ~ {r['player']} ({r['team']}) {r['side']} {r['line']} {r['market']} ({int(r['bet_odds']):+d} at {r['bet_at']})"
                          f" vs {r['opp']} | win {r['leg_prob']:.1%} | edge {r['leg_edge']:+.1%} | {'rate-only' if LEAN.get('off') else format(r['units'], 'g') + 'u'}")
    else:
        print("   (run check first)")
    print("\n2) BEST CROSS-GAME PARLAY (ONE-PARLAY RULE)")
    pp = a.parlays if a.parlays and os.path.exists(a.parlays) else None
    rec = next((x for x in json.load(open(pp)) if x.get("recommended")), None) if pp else None
    if rec:
        print(f"   {rec['name']} at {BOOK_NAME[rec['book']]} {rec['odds']:+d} | hit {rec['hit']:.1%} | ${rec['stake_usd']:.2f}"
              + (" | LEAN: no edge, least-bad play" if rec.get("lean") else ""))
        for l in rec["legs"]:
            print(f"     - {l[0]} {l[2]} {l[3]} {l[1]}")
    else:
        print("   none this week (or run parlay first)")
    print("\n3) BY GAME: best props (same-game parlays: NBA only for now - NFL correlations not modeled yet)")
    if pool.empty:
        print("   no legs qualify")
        return
    for gid, g in pool.groupby("game_id", sort=False):
        r0 = g.iloc[0]
        print(f"\n  {r0['team']} vs {r0['opp']}")
        for _, r in g.sort_values(["leg_prob", "leg_edge"], ascending=False).head(MENU["props_per_game"]).iterrows():
            print(f"    prop: {r['player']} {r['side']} {r['line']} {r['market']} ({int(r['odds']):+d}, {r['bet_at']}) | win {r['leg_prob']:.1%}"
                  f" | edge {r['leg_edge']:+.1%}" + f" | take if odds >= {int(r['take_min']):+d}")


# ---------- NEWS SWEEP (v2.6 / v15.9): who to check in the news before anything is placed ----------
# Every run: one news search per game + one per player VPE would put money on (slip, the parlay, top props, SGP legs).
# What the searches find goes into inj.txt (Out / Doubtful / Questionable / Limited) and `check` is rerun, so a minutes
# restriction mentioned in a presser but missing from the injury page still reaches the projection.
SWEEP = {"top": 10, "per_game_queries": 1}


SB = {"min_bets": 30, "min_bucket": 400, "cal_max": 3.0, "clv_min": 0.50}  # sample-size guardrails for the scoreboard



KILL = {"short_odds": -130}   # v15.28 KILL list (approved 11/17): nothing priced -130 or shorter without a boost. 56 graded props there
# hit 53.6% vs 62.6% needed, 18 tickets returned -32%, and the Shadow tab lost ~17% on thousands of lines shorter than -200.


# v15.29 RETUNE LOOP (approved 11/17). Every 25 newly graded bets, `learn --log ... --tune-log ...` re-checks each slice of the
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
    return (market is not None and f"market:{market}" in ks) or (side is not None and f"side:{side}" in ks)


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
    for sd, x in p.groupby(p["side"].astype(str).str.title()):
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
    """v15.28 NEWS SPEED: after an injury / lineup change and a rerun of `check`, compare the new run with the run before it
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
        be_line(f"{sd}s (straights + legs)", props[props["side"].astype(str).str.title() == sd])
    for md in ("Gold", "Silver"):
        be_line(f"{md} legs", props[props["medal"] == md])
    for mk, x in props.groupby("market"):
        if len(x) >= 5:
            be_line(f"{mk}", x)
    npass = sum(v == "PASS" for _, v in lights)
    print("  SCOREBOARD: " + " | ".join(f"{k}: {v}" for k, v in lights) + f"  -> {npass} of 4 green")


GAME_MK = set()   # NFL game lines are not scored as props yet


def held_reason(allp, player, mk, side=None, line=None):
    """v15.30 (rule order step 1): why this player / market is HELD this week, or "" when it is not."""
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
    if len(d) and pd.notna(r.get("market_gap")) and float(r["market_gap"]) > m.get("max_market_gap", 0.15):
        return f"model vs market gap {float(r['market_gap']) * 100:.0f} pts (over {m.get('max_market_gap', 0.15) * 100:.0f})"
    return ""


# ---------- v15.30 (from NBA v2.31-2.35): cross-book gap finder ----------
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

def cmd_bookgap(a):
    """Props and game lines posted at BOTH of his books at the same line: where one book's price beats the other book's
    fair (no-vig) chance. TAKE = gap >= 2 pts and VPE's own win chance does not disagree. Rate + suggest; logged like picks."""
    import json
    na, nb = a.book_a.lower(), a.book_b.lower()
    A, B = _bg_props(a.a), _bg_props(a.b)
    for X in (A, B):
        X["lab_o"] = X["player"].astype(str) + " Over " + X["line"].map("{:g}".format) + " " + X["mk"]
        X["lab_u"] = X["player"].astype(str) + " Under " + X["line"].map("{:g}".format) + " " + X["mk"]
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
    date = a.date or str(pd.Timestamp.now(tz="America/New_York").date())
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
            want += [(r["player"], r["market"], r["side"], float(r["line"]), "pick" if p_ == a.bets else ("best available" if LEAN.get("off") else "lean")) for _, r in b.iterrows()]
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


# ---------- CLV capture (v15.10): closing price for every bet actually placed this week ----------
# NFL has no alt-line anchoring yet, so the closing no-vig chance is read at the SAME line (or one within 0.5).
def cmd_close(a):
    log = read_log(a.log)
    parts = [pd.read_csv(p, dtype=str).fillna("") for p in (a.props or [])]
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["player", "market", "line", "over_odds", "under_odds"])
    snaps = []
    if len(raw):
        tmp = raw.copy()
        tmp["market"] = tmp["market"].map(lambda m: canon_market(m) or m)
        snaps = snap_rows(tmp.assign(line=pd.to_numeric(tmp["line"], errors="coerce")).dropna(subset=["line"]), "", a.week)
    if getattr(a, "snap_out", None):
        import json as _j
        _j.dump(snaps, open(a.snap_out, "w"))
        print(f"SNAPSHOT: {len(snaps)} closing lines for week {a.week} -> {a.snap_out} (append to the 'Close NFL' tab)")
    if getattr(a, "snap", None):
        sp = snap_props(a.snap, week=a.week).astype(str)
        raw = pd.concat([raw, sp[["player", "market", "line", "over_odds", "under_odds"]]], ignore_index=True)
        print(f"SNAPSHOT: {len(sp)} saved closing lines for week {a.week} read from {a.snap}")
    raw["mk"] = raw["market"].map(lambda m: canon_market(m) or m)
    raw["_key"] = raw["player"].map(norm_name)
    for c in ("closing_odds", "line", "odds", "week"):
        log[c] = pd.to_numeric(log.get(c), errors="coerce")
    bt = log["bet_type"].fillna("straight").astype(str)
    m = (log["week"] == a.week) & log["closing_odds"].isna() & bt.isin(["straight", "leg"])
    n_open = int(m.sum())
    n = 0
    for i in log[m].index:
        r = log.loc[i]
        c = raw[(raw["_key"] == norm_name(r["player"])) & (raw["mk"] == r["market"]) & (raw["over_odds"] != "") & (raw["under_odds"] != "")]
        if c.empty or pd.isna(r["line"]):
            continue
        c = c.assign(d=(c["line"].astype(float) - float(r["line"])).abs()).sort_values("d")
        if c.iloc[0]["d"] > 0.5:
            continue
        po = no_vig(float(c.iloc[0]["over_odds"]), float(c.iloc[0]["under_odds"]))[0]
        p = min(max(po if str(r["side"]).title() == "Over" else 1 - po, 0.02), 0.98)
        log.loc[i, "closing_odds"] = decimal_to_american(1 / p)
        n += 1
    if getattr(a, "no_log", False):
        print("--no-log: pick log NOT written (recheck only)")
    else:
        log.to_csv(a.log, index=False)
    d = log[log["closing_odds"].notna() & log["odds"].notna() & (log["week"] == a.week)]
    print(f"CLOSE week {a.week}: closing price written to {n} bets/legs")
    if len(d):
        clv = d["closing_odds"].map(lambda x: 1 / american_to_decimal(x)) - d["odds"].map(implied_prob)
        print(f"CLV week {a.week}: {len(d)} bets/legs | beat the close {(clv > 0).mean():.0%} | avg {clv.mean():+.1%}")
    note = ("NO closing lines (odds pages blocked or stale?)" if not len(raw) else
            "no open bets this week" if not n_open else "ok" if n == n_open else f"{n_open - n} left blank (player/line not on the closing pages)")
    note = close_note(note, len(raw), a)
    write_close_status(getattr(a, "status_out", None), "NFL", f"week {a.week}", n_open, n, len(snaps), note)


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
            # v15.26: column K = the Over win chance VPE gave THAT night (learned mix + fixes), so the scoreboard's calibration
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
    stem = path[:-5] if path.endswith(".json") else path
    for i in range(0, len(rows), 100):
        json.dump(rows[i:i + 100], open(f"{stem}_{i // 100 + 1}.json", "w"))
    print(f"SHADOW chunks: {(len(rows) + 99) // 100} files of <= 100 rows ({stem}_1.json ...); append each in order")
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
BOOST_LEG = {"min_edge": None, "w": 5.0}  # v15.25: --boost-min-leg-edge sets a hard floor; w = close-call tilt
# v15.33 PROFIT MODE (Adrian 11/24: "it just has to make me money"). Through 11/23 his boosted tickets were +409u on 626u staked
# and everything without a boost was -317u on 789u. A boost is the one place the math is on his side before any model
# opinion, so the boost builder now works from FAIR prices first:
#   - a leg's fair chance = its no-vig market chance, minus prop_cut for player props (graded two-sided prop lines
#     returned -5.2% a side against -3.4% implied by the books' own prices; game lines carry about 2.3% a side);
#   - builds are scored on that fair chance with the model as a quarter-weight tilt (model_w), not the other way round;
#   - the verdict (USE IT / THIN / SAVE IT) comes from the fair-price value after any fee, never from the model alone;
#   - with no boost VPE recommends no parlay: the ~4x / ~10x builds are reference only.
# Everything switched off stays graded in the 'Shadow builds' tab, so this can be reversed on evidence. "on": False = old way.
PROFIT = {"on": True, "model_w": 0.25, "prop_cut": 0.015, "use": 0.05, "fee": 0.0}
SHADOW_ALT = []   # what the old model-first builder would have picked for each boost (graded in 'Shadow builds', never shown as a bet)


def _fair_leg(c):
    mp = float(c["market_prob"])
    return mp * (1 - PROFIT["prop_cut"]) if PROFIT["on"] and c["market"] not in GAME_MK else mp


def boost_stake_frac(p, dec_eff, cap=0.05):
    """Share of bankroll VPE would put on a boost build: the usual 5%, or half-Kelly when the build is long enough that
    half-Kelly is smaller. A build that is not +EV keeps 5% (its score stays negative, as before)."""
    ev_ = p * dec_eff - 1
    return min(cap, 0.5 * ev_ / (dec_eff - 1)) if ev_ > 0 and dec_eff > 1 else cap


def boost_parlay(*a, **k):
    """v15.32: the usual search first (score at a flat 5% stake, unchanged). Only when that finds nothing that is +EV does
    it look again with the score taken at the stake that fits a long build (half-Kelly). Before, a boost whose own terms
    force long odds ("4+ legs, +400 or longer") could never score above zero and was always told SAVE IT (11/22)."""
    r = _boost_parlay_core(*a, **k)
    if r is None and k.get("need_market_ev", True) and (len(a) < 9 or a[8]):
        r = _boost_parlay_core(*a, **dict(k, fitted=True))
    return r


def _boost_parlay_core(pool, book, boost, stake, min_legs=2, max_legs=4, min_dec=1.0, max_win=None, need_market_ev=True, fitted=False):
    col = "dec_" + book
    # low-vig legs first: value at market odds (no-vig chance x this book's price), not the shortest prices
    ranked = pool.assign(_mv=pool["market_prob"] * pool[col]).sort_values(["_mv", "leg_prob"], ascending=False)
    # v15.27: boosts are built from TWO-SIDED main lines (both prices recorded = real, lower hold: all two-sided sides in the
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
            p_mkt = float(np.prod([_fair_leg(c) for c in combo]))   # v15.33: fair price (props carry the measured extra cut)
            fee_ = 1 + (PROFIT["fee"] if PROFIT["on"] else 0.0)
            if need_market_ev and p_mkt * dec_eff < fee_:  # must be +EV at fair prices, after any fee, with no model opinion
                continue
            p_model = p
            if PROFIT["on"]:   # score on the fair chance; the model only tilts it
                p = p_mkt + PROFIT["model_w"] * (p_model - p_mkt)
            # least risk for the payout: maximize long-run bankroll growth at a 5% stake (penalizes longshots)
            f_ = boost_stake_frac(p, dec_eff) if fitted else 0.05   # v15.32: fitted = second look for long builds only
            g = p * math.log(1 + f_ * (dec_eff - 1)) + (1 - p) * math.log(1 - f_)
            ml = min(float(c.get("leg_edge", 0) or 0) for c in combo)
            if BOOST_LEG["min_edge"] is not None and ml < BOOST_LEG["min_edge"]:
                continue
            allc.append((p * dec_eff / fee_ - 1, p_model, p_mkt, dec, dec_eff, combo, g, ml, uh))
    if not allc:
        return None
    # v15.25: close calls go to the STRONGER legs: among combos within 10% of the best growth score, the score gets
    # +BOOST_LEG["w"] x the weakest leg's edge (w 5: a +1% weakest leg = +5%); outside close calls nothing changes.
    def _pick(xs):
        if not xs:
            return None
        gmax = max(x[6] for x in xs)
        close = [x for x in xs if x[6] >= gmax - 0.10 * abs(gmax)]
        return max(close, key=lambda x: x[6] * (1 + BOOST_LEG["w"] * x[7]) if x[6] > 0 else x[6] + BOOST_LEG["w"] * x[7] * 1e-4)
    # v15.26: Under cap (45% of legs) with the same "overwhelming math" override as the regular parlays
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
    # v15.29 NO-FILLER RULE: a leg stays only if the parlay is worth more with it. If dropping a leg leaves a legal build
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
        print("  REFERENCE ONLY (not +EV: SAVE this boost). If it is used anyway, this is the least-bad build; it is not a VPE pick" if LEAN.get("off")
              else "  LEAN - BEST AVAILABLE BOOST: no edge, but if you use this boost tonight, this is the least-bad way")
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
    return {"name": f"BOOST {boost:g}%" + ((" (reference)" if LEAN.get("off") else " (LEAN)") if lean else ""), "target": 0, "book": bbook,
            "boost_pct": boost, "odds": decimal_to_american(dec), "hit": round(p, 4), "lean": lean, "ev": round(ev, 3),
            "stake_usd": stake, "legs": [[c["player"], c["market"], c["side"], float(c["line"])] for c in combo]}



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


LOG_COLS = ["season", "week", "logged_at", "player", "team", "opp", "market", "side", "line", "odds", "book", "tier",
            "units", "stake_usd", "source", "model_prob", "blend_prob", "edge", "ev_per_unit", "closing_odds",
            "actual", "result", "profit_units",
            "bet_type", "parlay_id", "suggested", "vpe_verdict", "boost_pct", "odds_eff", "edge_base", "note",
            "face_usd", "bonus_bet", "rec_units", "rec_stake_usd", "held_result", "held_profit_units", "cashout_cost"]
LOG_NUM = ["season", "week", "line", "odds", "units", "stake_usd", "model_prob", "blend_prob", "edge", "ev_per_unit", "closing_odds", "actual",
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

def cmd_log(a):
    set_calib(getattr(a, "calib", None), SPORT)
    """Log bets the user actually placed (from DK screenshots), including bets VPE did not suggest (profit boosts,
    own picks). Every bet is still run through the model at the price really taken."""
    cfg = _cfg(a)
    games, stats, prev, inj, week = _setup(a, cfg)
    ov = _overrides(a.injuries)
    # v15.22: the play-by-play load + full projection rebuild was the slow part of logging (11/16). A bet whose exact
    # line is already in tonight's bets_all.csv reuses that projection (same model chance: the model never depends on
    # the price); the projector is only built, once, if a bet isn't in bets_all.csv.
    _pj = {}
    def get_proj():
        if "p" not in _pj:
            print("LOG: a bet isn't in bets_all.csv -> building projections (slower)")
            outs = build_outs(inj, ov, week)
            plays = (load_plays(cfg["season"]), load_plays(cfg["season"] - 1))
            _pj["p"] = Projector(stats, prev, games, cfg["season"], week, cfg, outs=outs, weather=_weather(a.weather), plays=plays)
        return _pj["p"]
    unit = LOG_UNIT_USD  # v2.16: the Pick Log unit is $1, so units == stake_usd (was 2% of --bankroll)
    allp = pd.read_csv(a.all) if a.all and os.path.exists(a.all) else None
    lock_from(allp)

    def score(player, mk, side, line, odds):
        e0 = None
        if allp is not None and len(allp):
            ex = allp[(allp["player"].map(norm_name) == norm_name(player)) & (allp["market"] == mk) & (allp["side"] == side)
                      & (pd.to_numeric(allp["line"], errors="coerce") == float(line))]
            e0 = ex.iloc[0] if len(ex) else None
        if e0 is not None and pd.notna(e0.get("model_prob")):
            e = e0
        else:
            one = pd.DataFrame([{"player": player, "market": mk, "line": line,
                                 "over_odds": odds if side == "Over" else np.nan,
                                 "under_odds": odds if side == "Under" else np.nan, "book": "draftkings"}])
            ev, sk = evaluate_lines(one, get_proj(), inj, ov, cfg)
            if ev.empty:
                return None, (sk["reason"].iloc[0] if not sk.empty else "?")
            e = ev.iloc[0]
        mp = market_prob_for(allp, player, mk, side, line, odds)
        pb = blend_leg(float(e["model_prob"]), mp, mk, side)
        return {"prob": pb, "leg_prob": pb, "team": e["team"], "opp": e["opp"],
                "rec_units": float(e0["units"]) if e0 is not None and pd.notna(e0.get("units")) else 0.0,
                "min_odds": e0.get("min_odds") if e0 is not None and pd.notna(e0.get("tier")) and str(e0.get("tier")) not in ("", "nan") else None}, ""

    slip = slip_keys(a.slip)
    import json as _j
    pp_ = getattr(a, "parlays", None) or "vpe_parlays.json"
    VPE_PARLAYS[:] = _j.load(open(pp_)) if os.path.exists(pp_) else []
    _sr = getattr(a, "sgp_ratings", None) or "sgp_ratings.json"
    SGP_RATINGS[:] = _j.load(open(_sr)) if os.path.exists(_sr) else []
    rows = build_log_rows(pd.read_csv(a.placed, dtype=str), score, cfg, unit, slip,
                          {"season": cfg["season"], "week": week})
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
    if a.log and os.path.exists(a.log):
        old = read_log(a.log)
        new = pd.concat([old, new], ignore_index=True).reindex(columns=LOG_COLS)
    new.to_csv(a.log or "pick_log.csv", index=False)
    print_logged(rows)
    import json
    json.dump(sheet_rows(pd.DataFrame(rows)), open(a.sheet_out, "w"))
    print(f"sheet rows to append -> {a.sheet_out}")
    if slip is None:
        print(f"  ! no slip found at {a.slip}: run `check --out bets.csv` first so VPE picks can be told apart")
    print(f"\nlogged {len(rows)} rows -> {a.log or 'pick_log.csv'} (total rows {len(new)})")


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
    """RUN STATUS gate (NFL): same idea as NBA. GO / PARTIAL (gaps named) / HOLD (no picks)."""
    import json
    cfg = _cfg(a)
    g = load_games()
    g = g[(g["season"] == cfg["season"]) & (g["game_type"] == "REG")]
    week = a.week or int(g[g["home_score"].isna()]["week"].min())
    wk = g[(g["week"] == week) & g["home_score"].isna()]
    games = {frozenset((h, w)) for h, w in zip(wk["home_team"], wk["away_team"])}
    rows, hold, partial = [], [], []
    def add(name, ok, detail, level="partial"):
        rows.append((name, ok, detail))
        if not ok:
            (hold if level == "hold" else partial).append(name)
    add("SLATE", len(games) > 0, f"week {week}: {len(games)} games not yet played", "hold")
    stats = load_stats(cfg["season"])
    last = int(stats["week"].max()) if len(stats) else 0
    add("STATS", last >= week - 1, f"stats through week {last}" + ("" if last >= week - 1 else f", week {week - 1} missing"))
    cov, n = {}, 0
    if a.props and os.path.exists(a.props):
        pr = pd.read_csv(a.props, dtype=str).fillna("")
        n = len(pr)
        if "matchup" in pr:
            pair = lambda m: (lambda ps: frozenset(_nfl_team(x) for x in ps) if len(ps) == 2 else None)(
                re.split(r"\s+(?:@|vs\.?|at)\s+", str(m).strip(), maxsplit=1, flags=re.I))
            for mk, gg in pr.groupby("market"):
                cov[mk] = len({pair(m) for m in gg["matchup"]} & games)
    best = max(cov.values()) if cov else 0
    add("LINES", best > 0 and best >= len(games), (" | ".join(f"{k} {v}/{len(games)}" for k, v in sorted(cov.items()))
        or "no props file / no matchup column") + f" ({n} rows)", "hold" if best == 0 else "partial")
    inj = load_injuries(cfg["season"])
    official = week in set(inj.get("week", []))
    _ij = getattr(a, "injuries", None)  # v15.24: same flag as check (a list or a file)
    news = bool(_ij and not os.path.exists(_ij) and _ij.strip()) or bool(
        (_ij or a.injuries_file) and os.path.exists(_ij or a.injuries_file) and open(_ij or a.injuries_file).read().strip() != "")
    add("INJURIES", bool(news), f"official report {'YES' if official else 'NOT YET'}; news list "
        + ("built" if news else "MISSING (usage model and Out filter blind)"), "hold")
    if a.calib and os.path.exists(a.calib):
        w, _, nn = load_calib(a.calib, SPORT)
        add("CALIB", True, f"{nn} graded lines, model weight {w:.0%}")
    else:
        add("CALIB", False, "Calib tab not loaded (prior 40% model weight used)")
    exp = len(json.load(open(a.shadow))) if a.shadow and os.path.exists(a.shadow) else None
    if exp is None:
        add("SHADOW", False, "no shadow_new.json (check not run?)")
    elif a.v2_before is None or a.v2_after is None:
        add("SHADOW", False, f"{exp} lines NOT confirmed saved: pass --v2-before/--v2-after after appending")
    else:
        got = a.v2_after - a.v2_before
        add("SHADOW", got == exp, f"{got}/{exp} lines saved" + ("" if got == exp else (" -> append the missing chunk(s), re-check"
            if got < exp else " -> too many rows: a chunk was appended twice; delete the duplicate rows")))
    verdict = "HOLD" if hold else ("PARTIAL" if partial else "GO")
    print(f"RUN STATUS [{VERSION}] week {week}: {verdict}")
    for name, ok, detail in rows:
        print(f"  {'OK ' if ok else 'XX '} {name:9s} {detail}")
    if hold:
        print(f"  -> HOLD: no picks until fixed: {', '.join(hold)}")
    elif partial:
        print(f"  -> PARTIAL: picks allowed; name these gaps on the slip: {', '.join(partial)}")

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
    log = read_log(a.log)
    for c in ("result", "actual", "profit_units", "closing_odds", "odds_eff", "bet_type", "parlay_id"):
        if c not in log.columns: log[c] = None
    log.to_csv(a.log, index=False)
    week = a.week or int(stats["week"].max())
    log = grade(a.log, stats, cfg["season"], week)
    if log is not None and len(log):
        st = stats.assign(_key=stats["player_display_name"].map(norm_name))

        def _actual(r):
            b = st[(st["_key"] == norm_name(r["player"])) & (st["week"] == pd.to_numeric(r.get("week"), errors="coerce"))]
            return None if b.empty else float(b[MARKETS[r["market"]]["cols"]].fillna(0).sum(axis=1).iloc[0])
        log = settle_cashouts(log, _actual)
        log.to_csv(a.log, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(log[["week", "player", "market", "side", "line", "odds", "odds_eff", "units", "suggested", "actual", "result",
               "profit_units"]].to_string(index=False))
    s = summarize(log)
    if s:
        for k, v in s.items():
            print(f"\n== {k} ==\n{v.to_string()}")


# ---------- v2.15 / v15.16: the full logic behind every prop VPE suggests (`explain`) ----------
# `check` stores each prop's whole chain in the "x" column of bets_all.csv: minutes (NBA) or base average (NFL), every
# factor, the distribution, the model's chance, the market's chance, the learned mix, the price, the tier and the stake.
# `explain` prints it as one card per suggested prop (straights, parlay legs, leans). Pure reporting: no number changes.
import json as _json


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
    estm = EST_MARGIN if (x.get("p") or {}).get("how") == "1way" else 0.0
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
            + (" (market estimated from one price: needs +1.5% extra)" if estm else "")), None


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
             f"spread {g.get('spread')} | total {g.get('total')} ({tot}) | style: {g.get('arch')} vs {g.get('opp_style')} defense")
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
                add(r["player"], r["market"], r["side"], r["line"], ("BEST AVAILABLE (rate-only, not a VPE pick)" if LEAN.get("off") else f"LEAN straight {float(r.get('units', 0.5)):g}u (best available, no edge)"))
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
        print(explain_card(r, x, labels, cfg) + "\n")



# ---------- v2.16 / v15.17: take-if on recordings, reprice, several boosts in one run, est. market margin ----------
# Lines from a recording are priced as recorded, but bets often go in hours later and the price moves (11/11: three
# boost legs moved 15-35 cents). So every pick, leg and parlay prints a take-if again, and `reprice` re-rates a few
# props (or the parlay they make) at the live price without a full rerun.
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
    """--boosts "fd30:3leg:+100:win25,fd25,dk30:4leg:+400": book + % and that boost's own terms, read off the boost.
    Terms (v15.32: forgiving spellings, more terms, and an unreadable term STOPS that boost instead of being ignored):
      4leg / 4+legs / min4leg = minimum legs · max6 = maximum legs · +400 / 400 / 400+ = minimum TOTAL odds before the boost
      leg-200 / each-200 = every leg at this price or longer · props = player props only · games = game lines only
      win25 = extra winnings capped at $25 · maxbet10 = most that can be staked · any = the boost has no terms."""
    out = []
    for tok in [t.strip() for t in (spec or "").split(",") if t.strip()]:
        parts = tok.lower().replace(" ", "").split(":")
        m = re.match(r"^(fd|fanduel|dk|draftkings)(\d+(?:\.\d+)?)%?$", parts[0])
        if not m:
            print(f"BOOSTS: can't read '{tok}' (format: fd30, dk25:4leg:+400:leg-200:win25)")
            continue
        b = {"book": "fanduel" if m.group(1).startswith("f") else "draftkings", "pct": float(m.group(2)),
             "label": tok, "given": len(parts) > 1}
        for q0 in parts[1:]:
            q = q0.replace("legs", "leg")
            if re.fullmatch(r"(?:min)?\d+\+?leg\+?", q):
                b["min_legs"] = int(re.search(r"\d+", q).group())
            elif re.fullmatch(r"max\d+(?:leg)?", q):
                b["max_legs"] = int(re.search(r"\d+", q).group())
            elif re.fullmatch(r"(?:legmin|leg|each)[+-]\d{3,}", q):
                b["leg_min"] = float(re.search(r"[+-]\d+", q).group())
            elif re.fullmatch(r"(?:min)?[+-]?\d{3,}\+?", q):
                b["min_odds"] = float(re.search(r"[+-]?\d+", q).group())
            elif re.fullmatch(r"win\d+(\.\d+)?", q):
                b["max_win"] = float(q[3:])
            elif re.fullmatch(r"(?:maxbet|bet|stake)\d+(\.\d+)?", q):
                b["max_bet"] = float(re.search(r"\d+(\.\d+)?", q).group())
            elif q in ("props", "propsonly", "playerprops"):
                b["only"] = "props"
            elif q in ("games", "gamelines", "gamesonly"):
                b["only"] = "games"
            elif q not in ("any", "norules"):
                b.setdefault("bad", []).append(q0)
                print(f"BOOSTS: can't read the term '{q0}' in {tok} -> that boost is NOT built until its terms are fixed")
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
    hi = max(hi, lo)   # v15.32: a "5+ legs" boost used to search 5..4 legs (nothing) and say SAVE IT
    if spec.get("bad"):   # v15.32 (11/22: a "4+ legs, +400" boost was built on a term the engine had silently ignored)
        print(f"\n=== BOOST {spec['label']}: NOT BUILT. Can't read the term(s) {', '.join(repr(x) for x in spec['bad'])}. Terms VPE reads: "
              "4leg (min legs), max6, +400 (min total odds), leg-200 (every leg this price or longer), props, games, win25, maxbet10, any. "
              "Fix the spec and rerun `parlay` with all boosts; never show a build for a boost whose terms were not all read. ===")
        return None
    if spec.get("max_bet"):
        stake = min(stake, float(spec["max_bet"]))
    if spec.get("only") and len(pool):
        gm_ = pool["market"].isin(GAME_MK)
        pool = pool[~gm_] if spec["only"] == "props" else pool[gm_]
    if spec.get("leg_min") is not None and len(pool):
        pool = pool[pool["dec_" + bbook] >= american_to_decimal(spec["leg_min"]) - 1e-9]
    print(f"\n=== BOOST PARLAY {spec['label']}: {boost:g}% profit boost at {BOOK_NAME.get(bbook, bbook)} ===")
    rules = ((f"exactly {lo} legs" if lo == hi else f"{lo}-{hi} legs") + (f", min total odds {int(spec['min_odds']):+d}" if spec.get("min_odds") else "")
             + (f", every leg {int(spec['leg_min']):+d} or longer" if spec.get("leg_min") is not None else "")
             + (f", {'player props' if spec['only'] == 'props' else 'game lines'} only" if spec.get("only") else "")
             + (f", stake capped at ${spec['max_bet']:g}" if spec.get("max_bet") else "")
             + (f", extra winnings capped at ${mw:g}" if mw else ""))
    ng_ = pool["game_id"].astype(str).nunique() if len(pool) else 0
    if ng_ < lo:
        print(f"  rules used: {rules}")
        print(f"  CAN'T QUALIFY TONIGHT: this boost needs {lo} legs and VPE builds one leg per game, but only {ng_} game(s) have a leg "
              "that passes these terms. Say exactly that; SAVE the boost (or ask whether it allows same-game legs).")
        return None
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
        print("  REFERENCE ONLY (not +EV: SAVE this boost). If it is used anyway, this is the least-bad build; it is not a VPE pick" if LEAN.get("off")
              else "  LEAN - BEST AVAILABLE BOOST: no edge; only if you use this boost tonight anyway, this is the least-bad way")
    ev, p, p_mkt, dec, dec_eff, combo = r
    print(f"  PLACE AT {BOOK_NAME.get(bbook, bbook).upper()} | {len(combo)} legs | est. odds {decimal_to_american(dec):+d} -> "
          f"{decimal_to_american(dec_eff):+d} boosted | ${stake:.2f} pays ~${stake * dec_eff:.2f} | hit chance {p:.1%} "
          f"(market {p_mkt:.1%}) | EV {ev:+.0%} (EV at market odds alone {p_mkt * dec_eff - 1:+.0%}) | "
          f"Unders {sum(c['side'] == 'Under' for c in combo)}")
    print(pd.DataFrame(combo)[[c for c in cols if c in combo[0]]].to_string(index=False))
    ev_fair = p_mkt * dec_eff / (1 + (PROFIT["fee"] if PROFIT["on"] else 0.0)) - 1
    if PROFIT["on"]:
        _n_game = sum(c["market"] in GAME_MK for c in combo)
        print(f"  FAIR-PRICE VALUE: {ev_fair:+.1%} of the stake (no model opinion: fair chance {p_mkt:.1%} x boosted payout"
              + (f", after the {PROFIT['fee']:.0%} fee" if PROFIT["fee"] else "") + f") = ${ev_fair * stake:+.2f} expected on ${stake:.2f} | "
              f"legs: {_n_game} game line(s), {len(combo) - _n_game} prop(s) | "
              + ("reference only" if lean else ("USE IT" if ev_fair >= PROFIT["use"] else "THIN: a small plus, worth using only if the boost would expire unused")))
        if not lean:   # what the old model-first builder would have picked: tracked, never shown as a bet
            import contextlib, io
            PROFIT["on"] = False
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    r2 = boost_parlay(pool, bbook, boost, stake, lo, hi, min_dec, mw)
            finally:
                PROFIT["on"] = True
            if r2 and {(c["player"], c["market"], c["side"]) for c in r2[5]} != {(c["player"], c["market"], c["side"]) for c in combo}:
                SHADOW_ALT.append({"name": f"BOOST {spec['label']} (model pick)", "book": bbook, "boost_pct": boost,
                                   "odds": decimal_to_american(r2[3]), "hit": round(r2[1], 4), "hit_fair": round(r2[2], 4),
                                   "ev_fair": round(r2[2] * r2[4] - 1, 3),
                                   "legs": [[c["player"], c["market"], c["side"], float(c["line"])] for c in r2[5]]})
    _f = boost_stake_frac(p, dec_eff)
    if not lean and p * math.log(1 + 0.05 * (dec_eff - 1)) + (1 - p) * math.log(0.95) <= 0:
        print(f"  LONG BUILD: +EV with the boost, but it hits {p:.0%} of the time, so VPE's own size for it is {_f:.1%} of bankroll "
              "(half-Kelly), not the usual 5%. VPE's suggestion only.")
    # v15.32 RULES CHECK: the build against every term of this boost, one line the run shows as a table
    _legs = [float(c["dec_" + bbook]) for c in combo]
    _ck = [(f"legs {len(combo)} (needs {lo}" + ("" if lo == hi else f" to {hi}") + ")", lo <= len(combo) <= hi)]
    if spec.get("min_odds"):
        _ck.append((f"total odds {decimal_to_american(dec):+d} before the boost (needs {int(spec['min_odds']):+d} or longer)", dec >= min_dec - 1e-9))
    if spec.get("leg_min") is not None:
        _ck.append((f"shortest leg {decimal_to_american(min(_legs)):+d} (needs {int(spec['leg_min']):+d} or longer)", min(_legs) >= american_to_decimal(spec["leg_min"]) - 1e-9))
    if spec.get("only"):
        _ok = all((c["market"] in GAME_MK) == (spec["only"] == "games") for c in combo)
        _ck.append((("player props only" if spec["only"] == "props" else "game lines only"), _ok))
    if spec.get("max_bet"):
        _ck.append((f"stake ${stake:.2f} (max ${spec['max_bet']:g})", stake <= spec["max_bet"] + 1e-9))
    _ck.append(("one leg per game", len({str(c["game_id"]) for c in combo}) == len(combo)))
    print("  RULES CHECK: " + " | ".join(f"{t_}: {'OK' if o_ else 'FAIL'}" for t_, o_ in _ck)
          + (" -> QUALIFIES for this boost" if all(o_ for _, o_ in _ck) else " -> DOES NOT QUALIFY: do not show this build")
          + ("" if spec.get("given") else " (boost terms were NOT given: defaults used)"))
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
    return {"name": f"BOOST {spec['label']}" + ((" (reference)" if LEAN.get("off") else " (LEAN)") if lean else ""), "target": 0, "book": bbook,
            "boost_pct": boost, "odds": decimal_to_american(dec), "take_if": decimal_to_american(tk), "hit": round(p, 4),
            "lean": lean, "ev": round(ev, 3), "ev_fair": round(ev_fair, 3), "hit_fair": round(p_mkt, 4), "stake_usd": stake,
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
        estm = EST_MARGIN if str(r.get("mkt_est")).lower() in ("true", "1") else 0.0
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
              f"{be:.1%} needed -> edge {edge:+.1%}{' (needs +1.5% more: market estimated)' if estm else ''} -> {v}")
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


def _mw(market=None):
    """Model weight for this market (v2.17: each market learns its own mix, shrunk toward the shared one)."""
    mw = PARLAY.get("mk_weight") or {}
    g = PARLAY["model_weight"]
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


FETCH_SPORT = "NFL"
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
        if "/odds-pages/" in MIRROR:   # v15.26: read by commit id, so a just-refreshed mirror is never served from a 5-minute cache
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
        sub.choices[name].add_argument("--weather", help='outdoor forecasts "BUF:18:30;BAL:8:55:rain" = team:wind_mph:temp_F[:rain|snow]')
    sub.choices["board"].add_argument("--scheme", action="store_true", help="add blitz/coverage/box matchup (slower)")
    sub.choices["grade"].add_argument("--log", required=True)
    sub.choices["lines"].add_argument("--src", action="append", required=True,
                                      help="market=file.csv (odds-site rows), repeat per market")
    sub.choices["lines"].add_argument("--out", default="props.csv")
    sub.choices["log"].add_argument("--placed", required=True, help="CSV: player,market,side,line,odds,stake")
    sub.choices["log"].add_argument("--log", default="pick_log.csv")
    sub.choices["log"].add_argument("--injuries"); sub.choices["log"].add_argument("--weather")
    sub.choices["check"].add_argument("--all-picks", action="store_true", help="NFL: ignore boosts+leans-only mode")
    sub.choices["log"].add_argument("--slip", default="bets.csv", help="the slip from this week's check (marks VPE picks)")
    sub.choices["log"].add_argument("--all", default="bets_all.csv", help="this week's bets_all.csv (market prices for parlay legs)")
    sub.choices["parlay"].add_argument("--all", default="bets_all.csv")
    sub.choices["parlay"].add_argument("--bets", default="bets.csv")
    sub.choices["parlay"].add_argument("--stake", type=float, help="$ per parlay (default 5%% of bankroll)")
    sub.choices["parlay"].add_argument("--exclude", help='players to leave out "A;B" (e.g. DK price failed take-if)')
    sub.choices["parlay"].add_argument("--slip-out", default="slip.csv")
    sub.choices["parlay"].add_argument("--all-picks", action="store_true", help="NFL: ignore boosts+leans-only mode")
    sub.choices["parlay"].add_argument("--boost", type=float, help="profit boost %% (gets its own BOOST parlay)")
    sub.choices["parlay"].add_argument("--boosts", help='several boosts in one run, e.g. "fd30:3leg:+100:win25,fd25,dk30"')
    sub.choices["parlay"].add_argument("--fee", type=float, default=0.0, help="v15.33: add-on fee as %% of the stake (Bet Protect+ / Early Exit+ = 3)")
    sub.choices["parlay"].add_argument("--boost-book", default="draftkings", help="draftkings|fanduel (where the boost is)")
    sub.choices["log"].add_argument("--sheet-out", default="sheet_new.json", help="new rows for the Pick Log sheet")
    for name in ("check", "parlay", "log"):
        sub.choices[name].add_argument("--calib", help="Calib tab JSON (get_values): learned model weight")
    sub.choices["check"].add_argument("--shadow-out", default="shadow_new.json")
    sub.choices["check"].add_argument("--shadow-saved", help="get_values JSON of the Shadow tab's last rows (A:D): skip lines already saved (same-slate rerun)")
    sub.choices["log"].add_argument("--dry-run", action="store_true", help="rate the bets only, log nothing")
    sub.choices["parlay"].add_argument("--boost-min-leg-edge", type=float,
        help="v15.25: every boost leg needs at least this edge, e.g. 0.01 (default none: weakest-leg tilt on close calls)")
    for k, t in (("--boost-min-legs", int), ("--boost-max-legs", int), ("--boost-min-odds", float), ("--boost-max-win", float)):
        sub.choices["parlay"].add_argument(k, type=t)
    sub.choices["log"].add_argument("--parlays", default="vpe_parlays.json", help="parlays VPE built (from parlay)")
    sub.choices["parlay"].add_argument("--parlays-out", default="vpe_parlays.json")
    fr = sub.add_parser("frames"); fr.add_argument("--video", required=True); fr.add_argument("--out", default="frames")
    fr.add_argument("--every", type=float, default=1.0); fr.add_argument("--diff", type=float, default=6.0)
    fr.add_argument("--max", type=int, default=60)
    tk = sub.add_parser("takeif"); tk.add_argument("--all", default="bets_all.csv"); tk.add_argument("--calib")
    tk.add_argument("--bet", action="append", required=True, help='"Player;market;Side;line;odds" (repeat)')
    ss = sub.add_parser("status")
    ss.add_argument("--week", type=int); ss.add_argument("--season", type=int); ss.add_argument("--bankroll", type=float)
    ss.add_argument("--profile"); ss.add_argument("--props", default="props.csv"); ss.add_argument("--injuries-file", default="inj.txt")
    ss.add_argument("--injuries", help='inline list like check ("A:Out;B:Questionable") or a file; overrides --injuries-file')
    ss.add_argument("--calib", default="calib_tab.json"); ss.add_argument("--shadow", default="shadow_new.json")
    ss.add_argument("--v2-before", type=int); ss.add_argument("--v2-after", type=int)
    mn = sub.add_parser("menu"); mn.add_argument("--all", default="bets_all.csv"); mn.add_argument("--bets", default="bets.csv")
    mn.add_argument("--parlays", default="vpe_parlays.json"); mn.add_argument("--calib", default="calib_tab.json")
    nf_ = sub.add_parser("newsflash"); nf_.add_argument("--all", default="bets_all.csv"); nf_.add_argument("--prev"); nf_.add_argument("--runs", default="runs"); nf_.add_argument("--min-move", type=float, default=0.03); nf_.add_argument("--min-edge", type=float, default=0.0); nf_.add_argument("--top", type=int, default=12)
    sbp_ = sub.add_parser("scoreboard"); sbp_.add_argument("--log", default="pick_log.csv"); sbp_.add_argument("--shadow", nargs="*"); sbp_.add_argument("--calib", default="calib_tab.json"); sbp_.add_argument("--since"); sbp_.add_argument("--since-week")
    bgp_ = sub.add_parser("bookgap"); bgp_.add_argument("--a", required=True); bgp_.add_argument("--b", required=True)
    bgp_.add_argument("--book-a", default="fanduel"); bgp_.add_argument("--book-b", default="draftkings")
    bgp_.add_argument("--games-a"); bgp_.add_argument("--games-b"); bgp_.add_argument("--fair"); bgp_.add_argument("--all", default="bets_all.csv"); bgp_.add_argument("--date")
    bgp_.add_argument("--out", default="bookgap.csv"); bgp_.add_argument("--sheet-out", default="bookgap_new.json"); bgp_.add_argument("--slip", default="slip.csv")
    lk_ = sub.add_parser("links"); lk_.add_argument("--links", default="links.csv"); lk_.add_argument("--bets", default="bets.csv"); lk_.add_argument("--leans", default="leans.csv")
    lk_.add_argument("--parlays", default="vpe_parlays.json"); lk_.add_argument("--gaps", default="bookgap.csv")
    ld_ = sub.add_parser("linediff"); ld_.add_argument("--a", required=True, help="earlier props CSV"); ld_.add_argument("--b", required=True, help="later props CSV"); ld_.add_argument("--log"); ld_.add_argument("--date")
    ex2_ = sub.add_parser("exposure"); ex2_.add_argument("--log", default="pick_log.csv"); ex2_.add_argument("--date"); ex2_.add_argument("--week")
    sw_ = sub.add_parser("sweep"); sw_.add_argument("--all", default="bets_all.csv"); sw_.add_argument("--bets", default="bets.csv")
    sw_.add_argument("--parlays", default="vpe_parlays.json"); sw_.add_argument("--top", type=int, default=10)
    sw_.add_argument("--heavy", action="store_true", help="final check: every player with a qualifying line")
    xc_ = sub.add_parser("crosscheck"); xc_.add_argument("--rec", required=True, help="props.csv from the recording")
    xc_.add_argument("--web", required=True, help="props_web.csv from `lines` (free pages)"); xc_.add_argument("--out", default="linecheck_new.json")
    xc_.add_argument("--slate"); xc_.add_argument("--week")
    xc_.add_argument("--line-gap", type=float, default=1.0); xc_.add_argument("--same-book-gap", type=float, default=0.04); xc_.add_argument("--better-gap", type=float, default=0.03)
    cl_ = sub.add_parser("close"); cl_.add_argument("--no-log", action="store_true"); cl_.add_argument("--log", default="pick_log.csv"); cl_.add_argument("--props", nargs="+")
    cl_.add_argument("--week", type=int, required=True)
    cl_.add_argument("--snap", help="'Close NFL' tab JSON: closing lines saved earlier (for bets logged later)")
    cl_.add_argument("--snap-out", help="write the closing lines as rows for the 'Close NFL' tab")
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
    {"check": cmd_check, "board": cmd_board, "grade": cmd_grade, "lines": cmd_lines, "log": cmd_log,
     "parlay": cmd_parlay, "final": cmd_final, "sheet": cmd_sheet, "shadow": cmd_shadow, "learn": cmd_learn, "menu": cmd_menu, "sweep": cmd_sweep, "exposure": cmd_exposure, "linediff": cmd_linediff, "bookgap": cmd_bookgap, "links": cmd_links, "scoreboard": cmd_scoreboard, "newsflash": cmd_newsflash, "close": cmd_close, "status": cmd_status,
     "takeif": cmd_takeif, "frames": cmd_frames, "explain": cmd_explain, "reprice": cmd_reprice, "crosscheck": cmd_crosscheck, "fetchpages": cmd_fetchpages}[a.cmd](a)

