#!/usr/bin/env python3
import argparse, itertools, os, subprocess, sys, csv, time, math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def parse_list(s, cast):
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [cast(x) for x in items]

def run_one(python_path, script_path, clusters, times, labels, classcsv,
            tmin, tmax, binsize, topn, L, hop, k, alphaH, seed, fr_min, fr_max, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cmd = [python_path, script_path, "--clusters", clusters, "--times", times, "--labels", labels, "--classcsv", classcsv,
           "--tmin", str(tmin), "--tmax", str(tmax), "--binsize", str(binsize), "--topn", str(topn),
           "--L", str(L), "--hop", str(hop), "--k", str(k), "--alphaH", str(alphaH),
           "--fr_min", str(fr_min), "--fr_max", str(fr_max), "--seed", str(seed), "--outdir", str(outdir)]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ok, msg = True, "ok"
    except subprocess.CalledProcessError as e:
        ok, msg = False, e.stdout if e.stdout else str(e)
    dur = time.time() - t0
    return ok, msg, dur

def load_WH(outdir):
    outdir = Path(outdir)
    W = np.load(outdir / "W_components.npy")
    H = np.load(outdir / "H_activations.npy")
    return W, H

def norm_templates(W):
    K, N, L = W.shape
    X = W.reshape(K, -1)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def match_and_stability(Wa, Wb):
    A = norm_templates(Wa); B = norm_templates(Wb)
    S = A @ B.T
    row_idx, col_idx = linear_sum_assignment(-S)
    matched = S[row_idx, col_idx]
    return float(np.mean(matched)), float(np.min(matched)), matched

def collect_basic_metrics(outdir):
    out = {"recon_err": math.nan, "H_nonzero_frac": math.nan}
    outdir = Path(outdir)
    try:
        with open(outdir / "reconstruction_err.txt") as f:
            out["recon_err"] = float(f.read().strip())
    except Exception:
        pass
    try:
        H = np.load(outdir / "H_activations.npy")
        eps = 1e-9
        out["H_nonzero_frac"] = float(np.mean(H > eps))
    except Exception:
        pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--script", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--classcsv", default=None)
    ap.add_argument("--fr_min", type=float, default=0.5)
    ap.add_argument("--fr_max", type=float, default=5.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=600.0)
    ap.add_argument("--binsize", required=True)
    ap.add_argument("--topn", required=True)
    ap.add_argument("--L", required=True)
    ap.add_argument("--hop", required=True)
    ap.add_argument("--k", required=True)
    ap.add_argument("--alphaH", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--outbase", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    def parse_list2(s, cast): return [cast(x) for x in s.split(",") if x.strip()]
    binsizes = parse_list2(args.binsize, float)
    topns    = parse_list2(args.topn,    int)
    Ls       = parse_list2(args.L,       int)
    hops     = parse_list2(args.hop,     int)
    ks       = parse_list2(args.k,       int)
    alphas   = parse_list2(args.alphaH,  float)
    seeds    = parse_list2(args.seeds,   int)
    if len(seeds) != 2: print("Provide exactly two seeds, e.g. --seeds 0,1"); sys.exit(1)

    outbase = Path(args.outbase); outbase.mkdir(parents=True, exist_ok=True)
    summary = []
    for (binsize, topn, L, hop, k, alphaH) in itertools.product(binsizes, topns, Ls, hops, ks, alphas):
        tagA = f"bs{binsize}_top{topn}_L{L}_hop{hop}_k{k}_a{alphaH}_seed{seeds[0]}"
        tagB = f"bs{binsize}_top{topn}_L{L}_hop{hop}_k{k}_a{alphaH}_seed{seeds[1]}"
        outA = outbase / tagA; outB = outbase / tagB
        for tag, outdir, seed in [(tagA, outA, seeds[0]), (tagB, outB, seeds[1])]:
            if args.resume and (outdir / "W_components.npy").exists():
                ok, msg, dur = True, "skipped", 0.0
            else:
                ok, msg, dur = run_one(args.python, args.script, args.clusters, args.times, args.labels, args.classcsv,
                       args.tmin, args.tmax, binsize, topn, L, hop, k, alphaH, seed,
                       args.fr_min, args.fr_max, outdir)
            print(f"[{'OK' if ok else 'ERR'}] {tag}  {dur:.1f}s")
        stab_mean = stab_min = math.nan
        try:
            Wa, _ = load_WH(outA); Wb, _ = load_WH(outB)
            stab_mean, stab_min, _ = match_and_stability(Wa, Wb)
        except Exception:
            pass
        metrics = collect_basic_metrics(outA)
        row = dict(outA=str(outA), outB=str(outB), binsize=binsize, topn=topn, L=L, hop=hop, k=k, alphaH=alphaH,
                   recon_err=metrics.get("recon_err", math.nan),
                   H_nonzero_frac=metrics.get("H_nonzero_frac", math.nan),
                   stability_mean=stab_mean, stability_min=stab_min)
        summary.append(row)

    csv_path = outbase / "grid_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"Wrote {csv_path}")
if __name__ == "__main__":
    main()