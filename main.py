#!/usr/bin/env python3
"""Monte Carlo top-5 forecaster for make.fun `pumpfun-top5-forecast-01`.

Reads the house's 10-minute market-cap snapshots of every pump.fun coin
launched since the opening from $MAKEFUN_INPUT_DIR (default /input) and
writes one forecast to $MAKEFUN_OUTPUT_DIR (default /output), file name
from task.json ("forecast.json"):

  {"probabilities": [{"mint", "place", "p"}...],
   "top5": [mint x5 in order],
   "sizes": [{"place", "bin", "p"}...]}

Method
------
Per known coin (top ~80 by last-mark cap, plus recent top-5 holders) the
log10 market cap (in SOL) is simulated forward as a jump-diffusion with
a decaying drift (recent slope, shrunk toward the cross-sectional mean,
decaying to a negative fade within ~1h), per-coin volatility from realised
10-minute returns, a crash hazard (rug/dump), and a graduation jump for
bonding-curve coins close to completing (quote reserves near 85 SOL).
Momentum is short-lived: in the rehearsal the top 5 turns over almost
completely every 30 minutes, so the slope is capped at +/-0.25 log10/hour
and decays to a -0.12/hour fade with a 1-hour time constant.

Newcomers born after the history cut cannot be named, so they are
simulated as "phantoms": a Poisson flow of fresh coins whose peak cap is
resampled from the empirical peaks of the coins in the history that
reached the top 5 (so the peak scale follows the market itself), peaking
~20 minutes after birth and fading fast (a top-5 coin turns over almost
completely within 30 minutes). Their birth rate is the measured flow of
young coins (age under 2h) entering the top 5. Phantoms compete for
ranks in every simulated read, so the probability mass they take is
exactly the mass the forecast leaves unnamed at each place -- the
scorer shares unnamed mass equally over the answer chart's coins.

The one forecast is scored at every answer read, so the probabilities it
states are the Monte Carlo frequencies averaged over all simulated reads
(the mixture over reads is the log-score-optimal single forecast), and
the size distributions are the simulated per-place cap histograms,
smoothed so no bin is ever zero.

Deterministic: fixed seed, no wall clock, no network, stdlib only.
Two runs on the same history write byte-identical output.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
N_SIMS = 4000
MAX_COINS = 80
MIN_P_NAME = 0.006          # only name a coin at a place if its MC p beats this
SIZE_SMOOTH = 3.0           # Laplace smoothing counts per size bin
CHART_BOTTOM_LOG = math.log10(25.0)   # below this a coin is off the chart
DRIFT_FLOOR_PER_H = -0.12  # peaked coins fade ~ -0.12 log10 per hour
DRIFT_TAU_H = 1.0           # drift decays to the floor with this time constant
CRASH_PER_H_CURVE = 0.10    # per-hour crash hazard, ungraduated coins
CRASH_PER_H_POOL = 0.05     # per-hour crash hazard, graduated coins
CRASH_MEAN_LOG = 0.6        # mean log10 drop when a crash fires
GRAD_RESERVE_MIN = 40.0     # SOL in curve before completion is plausible
GRAD_RESERVE_FULL = 85.0    # SOL in curve at completion (pump.fun threshold)
GRAD_PER_H = 0.25           # per-hour completion prob at full reserves
GRAD_JUMP_LO = 0.10         # log10 jump on graduation
GRAD_JUMP_HI = 0.35
NEW_PEAK_AGE_MIN = 20.0     # minutes after birth when a newcomer peaks
NEW_FADE_PER_10MIN = 0.10   # log10 fade per 10 min after the peak (top-5
                            # coins turn over almost completely in 30 min)
NEW_RISE_PER_MIN = 0.01     # log10 still missing per minute before age 10
NEW_WINDOW_H = 2.0          # only newcomers born within this of a read matter


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log10_sol(lamports):
    v = int(lamports)
    if v <= 0:
        return None
    return math.log10(v) - 9.0


def erf(x):
    # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    s = 1 if x >= 0 else -1
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    t = 1.0 / (1.0 + p * abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return s * y


def norm_cdf(x, mu, sd):
    return 0.5 * (1.0 + erf((x - mu) / (sd * math.sqrt(2.0))))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1
    reads = [parse_time(r) for r in task["reads"]]
    out_name = task.get("output", "forecast.json")

    marks = []
    with open(os.path.join(in_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    marks.sort(key=lambda m: m["at"])
    cut = parse_time(marks[-1]["at"])

    # ---- build per-coin trajectories -------------------------------------
    # mint -> {obs: [(hours_since_cut, logcap, venue, reserves)], created, ...}
    coins = {}
    for m in marks:
        at = parse_time(m["at"])
        hours = (at - cut).total_seconds() / 3600.0  # <= 0
        for r in m["rows"]:
            mint = r["mint"]
            cap = log10_sol(r["market_cap_lamports"])
            if cap is None:
                continue
            c = coins.get(mint)
            if c is None:
                c = coins[mint] = {
                    "obs": [], "created": parse_time(r.get("created_at") or m["at"]),
                    "symbol": r.get("symbol", ""), "last_venue": r.get("venue", "curve"),
                    "last_reserves": float(r.get("quote_reserves_lamports") or 0) / 1e9,
                    "complete": bool(r.get("complete")),
                }
            c["obs"].append((hours, cap))
            c["last_venue"] = r.get("venue", "curve")
            c["last_reserves"] = float(r.get("quote_reserves_lamports") or 0) / 1e9
            c["complete"] = bool(r.get("complete"))
    last = marks[-1]["rows"]
    last_cap = {}
    for r in last:
        cap = log10_sol(r["market_cap_lamports"])
        if cap is not None:
            last_cap[r["mint"]] = cap
    last_rank = {r["mint"]: r["rank"] for r in last}

    # universe: top MAX_COINS by last cap, plus any coin top-5 in the last 12 marks
    cand = set(sorted(last_cap, key=lambda m: last_cap[m], reverse=True)[:MAX_COINS])
    for m in marks[-12:]:
        for r in m["rows"][:5]:
            cand.add(r["mint"])
    cand = [m for m in cand if m in coins]
    cand.sort(key=lambda m: last_cap.get(m, -99.0), reverse=True)
    cand = cand[:MAX_COINS]

    # ---- per-coin drift / vol --------------------------------------------
    # global 10-min return sd over all coins (for shrinkage)
    all_rets = []
    for c in coins.values():
        o = c["obs"]
        for (h1, x1), (h2, x2) in zip(o, o[1:]):
            if abs((h2 - h1) * 6.0 - 1.0) < 0.01:  # consecutive 10-min marks
                all_rets.append(x2 - x1)
    if all_rets:
        mu_r = sum(all_rets) / len(all_rets)
        g_sd = math.sqrt(sum((r - mu_r) ** 2 for r in all_rets) / len(all_rets))
    else:
        g_sd = 0.12
    g_sd = min(max(g_sd, 0.08), 0.30)

    # cross-sectional mean slope of the top 20 (per hour), for shrinkage
    def ols_slope(obs):
        n = len(obs)
        if n < 2:
            return None
        mh = sum(h for h, _ in obs) / n
        mx = sum(x for _, x in obs) / n
        den = sum((h - mh) ** 2 for h, _ in obs)
        if den <= 0:
            return None
        return sum((h - mh) * (x - mx) for h, x in obs) / den

    top20_slopes = []
    for m in sorted(last_cap, key=lambda k: last_cap[k], reverse=True)[:20]:
        s = ols_slope(coins[m]["obs"][-6:])
        if s is not None:
            top20_slopes.append(s)
    mean_slope = sum(top20_slopes) / len(top20_slopes) if top20_slopes else 0.0

    stats = []  # per coin: dict with x0, slope0, sd10, crash_p/h, grad info
    for m in cand:
        c = coins[m]
        obs = c["obs"][-6:]
        s = ols_slope(obs)
        n = len(obs)
        if s is None:
            slope0 = mean_slope
        else:
            slope0 = (n * s + 3.0 * mean_slope) / (n + 3.0)
        slope0 = max(min(slope0, 0.25), -0.25)
        # A coin sitting at its running peak and older than ~15 minutes has
        # almost certainly peaked: its rise slope is history, not its
        # future. Force the fade (the floor) instead of projecting the
        # rise forward for hours. Young peakers (<15 min) are still rising.
        peak_x = max(x for _, x in c["obs"])
        age_h = max(0.0, (cut - c["created"]).total_seconds() / 3600.0)
        if last_cap[m] >= peak_x - 0.03 and age_h > 0.25:
            slope0 = min(slope0, DRIFT_FLOOR_PER_H)
        rets = [x2 - x1 for (_, x1), (_, x2) in zip(obs, obs[1:])]
        if len(rets) >= 2:
            mr = sum(rets) / len(rets)
            sd = math.sqrt(sum((r - mr) ** 2 for r in rets) / len(rets))
            sd10 = (len(rets) * sd + 4.0 * g_sd) / (len(rets) + 4.0)
        else:
            sd10 = g_sd
        sd10 = min(max(sd10, 0.08), 0.30)
        venue = c["last_venue"]
        stats.append({
            "mint": m, "x0": last_cap[m], "slope0": slope0, "sd10": sd10,
            "crash_h": CRASH_PER_H_POOL if (venue == "pool" or c["complete"]) else CRASH_PER_H_CURVE,
            "venue": venue, "reserves": c["last_reserves"],
            "complete": c["complete"],
            "age_h": max(0.0, (cut - c["created"]).total_seconds() / 3600.0),
        })

    # ---- simulation grid: cut -> read1 -> read2 ... -----------------------
    bounds = [cut] + reads
    steps = []  # (dt_hours, elapsed_at_start_hours, read_time)
    for a, b in zip(bounds, bounds[1:]):
        dt = (b - a).total_seconds() / 3600.0
        el = (a - cut).total_seconds() / 3600.0
        if dt > 0:
            steps.append((dt, el, b))
    n_steps = len(steps)

    # precompute per-coin per-step drift mean and sd (deterministic)
    drift = []  # [coin][step]
    sig = []
    for st in stats:
        d_row, s_row = [], []
        for dt, el, _rt in steps:
            # drift(t) = floor + (slope0 - floor) * exp(-(el+t)/tau)
            integ = (DRIFT_FLOOR_PER_H * dt
                     + (st["slope0"] - DRIFT_FLOOR_PER_H) * DRIFT_TAU_H
                     * math.exp(-el / DRIFT_TAU_H) * (1.0 - math.exp(-dt / DRIFT_TAU_H)))
            d_row.append(integ)
            s_row.append(st["sd10"] * math.sqrt(max(dt, 0.0) * 6.0))
        drift.append(d_row)
        sig.append(s_row)

    # ---- newcomer flow: young coins entering the top 5 per hour --------
    # (measured over the history; drives the phantom birth rate) and the
    # empirical peaks of coins that reached the top 5 (phantom peak scale)
    entrant_rate = 0.0
    prev_top5 = None
    span_h = 0.0
    for m in marks:
        cur5 = m["rows"][:5]
        if prev_top5 is not None:
            for r in cur5:
                if r["mint"] not in prev_top5:
                    created = coins.get(r["mint"], {}).get("created")
                    if created is not None:
                        age_h = (parse_time(m["at"]) - created).total_seconds() / 3600.0
                        if age_h < 2.0:
                            entrant_rate += 1.0
            span_h += 1.0 / 6.0
        prev_top5 = {r["mint"] for r in cur5}
    if span_h > 0:
        entrant_rate /= span_h  # per hour
    lam_top5_h = min(max(entrant_rate, 0.5), 60.0)
    peak_pool = []
    for m, c in coins.items():
        xs = [x for _, x in c["obs"]]
        if xs and m in {r["mint"] for mm in marks for r in mm["rows"][:5]}:
            peak_pool.append(max(xs))
    if not peak_pool:
        peak_pool = [last_cap[m] for m in cand] or [2.0]
    peak_pool.sort()

    rng = random.Random(SEED)
    n_coins = len(stats)
    place_counts = [[0.0] * n_coins for _ in range(places)]
    place_phantom = [0.0] * places
    place_size_hist = [[0.0] * n_bins for _ in range(places)]
    rank_sum = [0.0] * n_coins          # for expected rank (top5)
    rank_n = [0] * n_coins
    total_reads = 0

    for _ in range(N_SIMS):
        xs = [st["x0"] for st in stats]
        alive = [True] * n_coins
        graduated = [st["complete"] or st["venue"] == "pool" for st in stats]
        for si, (dt, _el, read_t) in enumerate(steps):
            # advance known coins
            for i, st in enumerate(stats):
                if not alive[i]:
                    continue
                # crash?
                ch = st["crash_h"]
                p_crash = 1.0 - (1.0 - ch) ** dt
                x = xs[i] + drift[i][si] + sig[i][si] * rng.gauss(0.0, 1.0)
                if rng.random() < p_crash:
                    x -= rng.expovariate(1.0 / CRASH_MEAN_LOG)
                # graduation jump (once)
                if not graduated[i] and st["venue"] == "curve" and st["reserves"] >= GRAD_RESERVE_MIN:
                    p_g = 1.0 - (1.0 - GRAD_PER_H * min(st["reserves"] / GRAD_RESERVE_FULL, 1.0)) ** dt
                    if rng.random() < p_g:
                        x += rng.uniform(GRAD_JUMP_LO, GRAD_JUMP_HI)
                        graduated[i] = True
                xs[i] = x
                if x < CHART_BOTTOM_LOG:
                    alive[i] = False
            # phantoms for this read: newcomers born shortly before it,
            # peaks resampled from the empirical top-5 peak distribution
            window = min(NEW_WINDOW_H, (read_t - cut).total_seconds() / 3600.0)
            lam = lam_top5_h * window
            # Poisson draw (Knuth)
            l = math.exp(-lam)
            k, p = 0, 1.0
            while True:
                p *= rng.random()
                if p <= l:
                    break
                k += 1
            phantom_caps = []
            for _ in range(k):
                age_min = rng.uniform(0.0, window * 60.0)
                peak = peak_pool[rng.randrange(len(peak_pool))]
                peak += rng.gauss(0.0, 0.05)
                cap = peak - NEW_FADE_PER_10MIN * max(0.0, (age_min - NEW_PEAK_AGE_MIN) / 10.0)
                cap -= NEW_RISE_PER_MIN * max(0.0, 10.0 - age_min)
                if cap >= CHART_BOTTOM_LOG:
                    phantom_caps.append(cap)
            # rank: known alive coins + phantoms
            order = sorted((i for i in range(n_coins) if alive[i]),
                           key=lambda i: xs[i], reverse=True)
            # merge phantoms into ranking
            ranked_caps = [(xs[i], i) for i in order]
            for pc in phantom_caps:
                ranked_caps.append((pc, -1))
            ranked_caps.sort(key=lambda t: t[0], reverse=True)
            total_reads += 1
            for k in range(places):
                if k >= len(ranked_caps):
                    break
                cap, idx = ranked_caps[k]
                b = 0
                while b < len(edges) and edges[b] <= cap:
                    b += 1
                place_size_hist[k][b] += 1.0
                if idx >= 0:
                    place_counts[k][idx] += 1.0
                    rank_sum[idx] += (k + 1)
                    rank_n[idx] += 1
                else:
                    place_phantom[k] += 1.0
            # expected-rank penalty for coins not in top 5 this read
            top5_set = {idx for _, idx in ranked_caps[:places] if idx >= 0}
            for i in range(n_coins):
                if alive[i] and i not in top5_set:
                    rank_sum[i] += (places + 1)
                    rank_n[i] += 1

    # ---- build the forecast ----------------------------------------------
    denom = float(N_SIMS * n_steps) if n_steps else 1.0
    probabilities = []
    for k in range(places):
        rows = []
        for i, st in enumerate(stats):
            p = place_counts[k][i] / denom
            if p >= MIN_P_NAME:
                rows.append((p, st["mint"]))
        rows.sort(key=lambda t: (-t[0], t[1]))
        s = 0.0
        for p, mint in rows:
            q = math.floor(p * 1e6) / 1e6
            if q <= 0:
                continue
            if s + q > 1.0:
                q = math.floor((1.0 - s) * 1e6) / 1e6
                if q <= 0:
                    break
            probabilities.append({"mint": mint, "place": k + 1, "p": q})
            s += q
    # top5: lowest expected place first
    exp_rank = []
    for i, st in enumerate(stats):
        if rank_n[i]:
            exp_rank.append((rank_sum[i] / rank_n[i], st["mint"]))
        else:
            exp_rank.append((99.0, st["mint"]))
    exp_rank.sort(key=lambda t: (t[0], t[1]))
    top5 = [m for _, m in exp_rank[:places]]
    # sizes: smoothed histogram per place
    sizes = []
    for k in range(places):
        hist = place_size_hist[k]
        tot = sum(hist)
        if tot <= 0:
            continue
        s = 0.0
        for b in range(n_bins):
            p = (hist[b] + SIZE_SMOOTH) / (tot + SIZE_SMOOTH * n_bins)
            q = math.floor(p * 1e6) / 1e6
            if q <= 0:
                continue
            if s + q > 1.0:
                q = math.floor((1.0 - s) * 1e6) / 1e6
                if q <= 0:
                    break
            sizes.append({"place": k + 1, "bin": b, "p": q})
            s += q
    doc = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, separators=(",", ":")) + "\n")
    print("wrote %s: %d probability rows, top5 %s, %d size rows"
          % (os.path.join(out_dir, out_name), len(probabilities), top5, len(sizes)),
          file=sys.stderr)
    print("phantom (newcomer) share per place: %s; young top-5 entrant rate %.0f/h"
          % ([round(place_phantom[k] / denom, 3) for k in range(places)],
             entrant_rate), file=sys.stderr)


if __name__ == "__main__":
    main()
