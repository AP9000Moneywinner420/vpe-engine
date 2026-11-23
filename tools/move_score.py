"""VPE move score (11/23, report-only): after VPE scored a line, did the books' price move toward the side the model leaned?

A fast read on whether the model knows something: hundreds of lines a night instead of a handful of bets. It changes no
pick. Reads rows already saved in the Pick Log sheet; holds no odds data itself.

  python move_score.py --date 2026-11-22 --shadow lw_1.json lw_2.json --moves moves.json [--hist hist.json] --out row.json

--shadow  'Shadow NBA' rows, columns A:K (key, player, market, line, over_odds, under_odds, p_model, p_market, actual, over, p_vpe)
--moves   'Line moves' rows for that date, columns A:S
--hist    'Move score' tab rows (to print the running total); optional
Each file is a JSON list of rows or a {"values": [...]} object, with or without the header row.
"""
import argparse, json, math


def rows(path):
    j = json.load(open(path))
    if isinstance(j, dict):
        j = j.get("values") or j.get("content", {}).get("values") or []
    return [r for r in j if isinstance(r, list) and r]


def num(x):
    try:
        return float(str(x).replace("'", "").strip())
    except Exception:
        return None


def imp(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def score(date, shadow, moves):
    last = {}
    for r in moves:
        r = r + [""] * (19 - len(r))
        if str(r[0]).lstrip("'")[:10] != date or r[3] in ("ml", "spread", "total"):
            continue
        ln, lo, lu = num(r[14]), num(r[15]), num(r[16])
        if ln is None or not lo or not lu:
            continue
        last.setdefault((r[1], r[3], ln), []).append(imp(lo) / (imp(lo) + imp(lu)))
    out = {"n2": 0, "t2": 0, "n4": 0, "t4": 0, "nv": 0, "tv": 0, "matched": 0}
    seen = set()
    for r in shadow:
        r = r + [""] * (11 - len(r))
        if str(r[0]).lstrip("'")[:10] != date:
            continue
        k = (r[1], r[2], num(r[3]))
        pm, pk, pv = num(r[6]), num(r[7]), num(r[10])
        if k in seen or k not in last or pk is None or not num(r[4]) or not num(r[5]):
            continue
        seen.add(k)
        mv = sum(last[k]) / len(last[k]) - pk
        out["matched"] += 1
        if abs(mv) <= 0.005:
            continue
        for p, th, n_, t_ in ((pm, 0.02, "n2", "t2"), (pm, 0.04, "n4", "t4"), (pv, 0.02, "nv", "tv")):
            if p is not None and abs(p - pk) > th:
                out[n_] += 1
                out[t_] += int((p - pk > 0) == (mv > 0))
    return out


def pct(t, n):
    return f"{100 * t / n:.1f}% of {n} (+/-{100 * math.sqrt(0.25 / n):.1f})" if n else "no lines"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--shadow", nargs="+", required=True)
    ap.add_argument("--moves", required=True); ap.add_argument("--hist"); ap.add_argument("--out", default="move_score_row.json")
    a = ap.parse_args()
    sh = [r for f in a.shadow for r in rows(f)]
    s = score(a.date, sh, rows(a.moves))
    row = [a.date, s["matched"], s["n2"], s["t2"], s["n4"], s["t4"], s["nv"], s["tv"]]
    json.dump([row], open(a.out, "w"))
    if not s["matched"]:
        print(f"MOVE SCORE {a.date}: no lines matched (no line moves saved for that night, or no Shadow rows): nothing to append")
        return
    print(f"MOVE SCORE {a.date}: price moved toward the MODEL's side on {pct(s['t2'], s['n2'])} lines it leaned 2+ pts; "
          f"4+ pt leans {pct(s['t4'], s['n4'])}; VPE's final number {pct(s['tv'], s['nv'])}. 50% = no information.")
    if a.hist:
        tot = [0] * 6; nights = 0
        for r in rows(a.hist) + [row]:
            if str(r[0]).lstrip("'")[:4].isdigit() and (str(r[0]).lstrip("'")[:10] != a.date or r is row):
                v = [num(x) or 0 for x in (r + [0] * 8)[2:8]]
                tot = [x + y for x, y in zip(tot, v)]; nights += 1
        print(f"MOVE SCORE since 11/19 ({nights} nights): model 2+ pts {pct(int(tot[1]), int(tot[0]))}; 4+ pts {pct(int(tot[3]), int(tot[2]))}; "
              f"VPE final {pct(int(tot[5]), int(tot[4]))}. Report-only: it changes no pick.")


if __name__ == "__main__":
    main()
