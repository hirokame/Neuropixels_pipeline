#!/usr/bin/env python3
"""
seqNMF-like analysis for Neuropixels spike trains.

Pipeline:
1) Load spike times & cluster IDs; keep only "good" units (unitType 1 or 2).
2) Bin spikes (neurons × time bins).
3) Build lagged patches (neurons×lags × sliding windows).
4) Run NMF on patches to extract sequence templates (W) and activations (H).
5) Save W (K×N×L), H (K×num_windows), unit lists, rough peak times, and quick plots.

Requires: numpy, pandas
Optional: scikit-learn (for fast NMF), matplotlib (for plots), scipy (for smoothing)
If scikit-learn is missing, falls back to a simple NumPy NMF (MU updates).
"""

import os
import argparse
import numpy as np
import pandas as pd

# Optional deps
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from scipy.ndimage import gaussian_filter1d as _gauss
    def _smooth(X, sigma=1.0):
        return _gauss(X, sigma=sigma, axis=1)
    HAS_SCIPY = True
except Exception:
    def _smooth(X, sigma=1.0):
        return X
    HAS_SCIPY = False

try:
    from sklearn.decomposition import NMF as SKNMF
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ------------------------
# I/O + preprocessing
# ------------------------

def load_data(spike_clusters_path, spike_times_path, unit_labels_tsv,
              good_types=(1, 2), class_csv=None, keep_class="MSN"):
    sc = np.load(spike_clusters_path)
    st = np.load(spike_times_path)

    labels = pd.read_csv(unit_labels_tsv, sep="\t")
    good_units = labels[labels["unitType"].isin(good_types)].index.values.astype(int)

    if class_csv is not None:
        cdf = pd.read_csv(class_csv)
        if "unit_id" not in cdf.columns or "cell_type" not in cdf.columns:
            raise ValueError("class_csv must contain columns: unit_id, cell_type")
        cdf["unit_id"] = cdf["unit_id"].astype(int)
        cdf["cell_type"] = cdf["cell_type"].astype(str).str.upper().str.strip()
        msn_units = cdf.loc[cdf["cell_type"] == str(keep_class).upper(), "unit_id"].to_numpy(dtype=int)
        good_units = np.intersect1d(good_units, msn_units)

    mask = np.isin(sc, good_units)
    return st[mask].astype(float), sc[mask].astype(int), good_units

def bin_spikes(times, units, bin_size=0.02, t_min=0.0, t_max=None,
               top_n=100, allowed_units=None, fr_min=None, fr_max=None):
    if t_max is None:
        t_max = float(times.max()) if times.size else t_min + 1.0
    sel_t = (times >= t_min) & (times <= t_max)
    t_win = times[sel_t]
    u_win = units[sel_t]

    # Restrict to allowed units (good ∩ MSN)
    if allowed_units is not None:
        keep_mask = np.isin(u_win, allowed_units)
        t_win = t_win[keep_mask]
        u_win = u_win[keep_mask]

    if t_win.size == 0:
        raise ValueError("No spikes in [tmin,tmax] after preliminary unit filtering.")

    # Compute rates in window and apply FR caps
    uniq, counts = np.unique(u_win, return_counts=True)
    duration = (t_max - t_min)
    rates_hz = counts / max(duration, 1e-9)
    keep = np.ones_like(uniq, dtype=bool)
    if fr_min is not None:
        keep &= rates_hz >= fr_min
    if fr_max is not None:
        keep &= rates_hz <= fr_max

    uniq = uniq[keep]
    if uniq.size == 0:
        raise ValueError("No units pass firing-rate filter; adjust --fr_min / --fr_max.")
    counts = counts[keep]

    # Choose top_n most active (by count) among the filtered units
    order = np.argsort(counts)[::-1]
    keep_units = uniq[order[:min(top_n, uniq.size)]]
    u2i = {u: i for i, u in enumerate(keep_units)}

    # Build binned matrix
    T = int(np.ceil(duration / bin_size))
    X = np.zeros((len(keep_units), T), dtype=np.float32)
    X = X / (X.mean(axis=1, keepdims=True) + 1e-9)  # normalize by mean rate
    t_idx = np.floor((t_win - t_min) / bin_size).astype(int)
    valid = (t_idx >= 0) & (t_idx < T)
    t_idx = t_idx[valid]
    u_idx = np.array([u2i.get(u, -1) for u in u_win[valid]])
    valid2 = u_idx >= 0
    np.add.at(X, (u_idx[valid2], t_idx[valid2]), 1.0)
    return X, keep_units


def build_lagged_patches(X, L=20, hop=5, smooth_sigma=1.0):
    """
    Build lagged patches: stack N×L slices into columns (N*L × num_windows).
    end indices = [L-1, L-1+hop, L-1+2*hop, ...]
    """
    Xs = _smooth(X, sigma=smooth_sigma) if smooth_sigma and smooth_sigma > 0 else X
    N, T = Xs.shape
    if T < L:
        raise ValueError("Time bins shorter than L.")
    ends = np.arange(L - 1, T, hop)
    num = len(ends)
    patches = np.empty((N * L, num), dtype=np.float32)
    # Fill columns
    for i, t in enumerate(ends):
        patches[:, i] = Xs[:, t - L + 1 : t + 1].reshape(-1)
    
    patches = patches / (patches.sum(axis=0, keepdims=True) + 1e-9)  # normalize columns
    return patches, ends


# ------------------------
# NMF backends
# ------------------------
def nmf_mu(V, rank, max_iter=300, tol=1e-4, alpha_H=0.0, random_state=0):
    """
    Simple Frobenius NMF with multiplicative updates:
      H <- H * (W^T V) / (W^T W H + alpha_H)
      W <- W * (V H^T) / (W H H^T)
    V: (m×n), rank: K
    Returns: W (m×K), H (K×n), err (Frobenius reconstruction)
    """
    rng = np.random.default_rng(random_state)
    m, n = V.shape
    V = np.maximum(V, 0)
    W = np.maximum(rng.random((m, rank)), 1e-6)
    H = np.maximum(rng.random((rank, n)), 1e-6)
    eps = 1e-9

    last_err = None
    for it in range(max_iter):
        # Update H
        WT = W.T
        numH = WT @ V
        denH = (WT @ W @ H) + (alpha_H if alpha_H > 0 else 0.0) + eps
        H *= numH / denH
        H = np.maximum(H, 0)

        # Update W
        HT = H.T
        numW = V @ HT
        denW = W @ (H @ HT) + eps
        W *= numW / denW
        W = np.maximum(W, 0)

        # Check error
        if (it + 1) % 20 == 0 or it == max_iter - 1:
            R = V - W @ H
            err = np.linalg.norm(R, "fro")
            if last_err is not None and abs(last_err - err) / (last_err + eps) < tol:
                break
            last_err = err

    R = V - W @ H
    err = np.linalg.norm(R, "fro")
    return W, H, err


def run_seqnmf_like(patches, n_components=6, max_iter=300, alpha_H=0.1, random_state=0):
    """Run NMF on patches matrix (N*L × num_windows). Prefer sklearn; fallback to MU."""
    V = np.maximum(patches, 0)
    if HAS_SKLEARN:
        model = SKNMF(
            n_components=n_components,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            l1_ratio=0.0,
            alpha_W=0.0,
            alpha_H=alpha_H,
            max_iter=max_iter,
            random_state=random_state,
        )
        W = model.fit_transform(V)
        H = model.components_
        err = model.reconstruction_err_
        return W, H, err
    else:
        W, H, err = nmf_mu(V, rank=n_components, max_iter=max_iter, alpha_H=alpha_H, random_state=random_state)
        return W, H, err


# ------------------------
# Post-processing & save
# ------------------------
def reshape_components(W, N, L):
    """W is (N*L × K) -> (K × N × L)."""
    K = W.shape[1]
    return W.reshape(N, L, K).transpose(2, 0, 1)


def save_top_units_per_component(W3, keep_units, out_csv, top_m=10):
    """For each component, rank units by max over lag."""
    K, N, L = W3.shape
    rows = []
    for k in range(K):
        weights = W3[k].max(axis=1)  # (N,)
        order = np.argsort(weights)[::-1][: min(top_m, N)]
        for rank, idx in enumerate(order, 1):
            rows.append(
                {"component": k, "rank": rank, "unit": int(keep_units[idx]), "weight": float(weights[idx])}
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def peak_times(H, hop, bin_size, t_min=0.0, topk=50, min_separation=0.15):
    """
    Rough peak-time picker on H (K × num_windows). Non-maximum suppression by time separation.
    Returns list of lists of peak times (seconds) per component.
    """
    K, Wn = H.shape
    peaks = []
    for k in range(K):
        h = H[k].copy()
        idxs = np.argsort(h)[::-1]
        events = []
        for i in idxs:
            t = t_min + (i * hop) * bin_size
            if all(abs(t - et) >= min_separation for et in events):
                events.append(t)
            if len(events) >= topk:
                break
        peaks.append(sorted(events))
    return peaks


def maybe_plot_H(H, outdir):
    if not HAS_MPL:
        return
    K = H.shape[0]
    fig, axes = plt.subplots(K, 1, figsize=(12, 1.5 * max(K, 1)), sharex=True)
    if K == 1:
        axes = [axes]
    for k in range(K):
        axes[k].plot(H[k])
        axes[k].set_ylabel(f"H{k}")
    axes[-1].set_xlabel("Window index")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "H_activations.png"), dpi=150)
    plt.close(fig)


def maybe_plot_W(W3, keep_units, outdir):
    if not HAS_MPL:
        return
    K, N, L = W3.shape
    for k in range(K):
        weights = W3[k].max(axis=1)
        order = np.argsort(weights)[::-1][: min(10, N)]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(W3[k][order, :], aspect="auto", origin="lower")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([str(int(keep_units[i])) for i in order])
        ax.set_xlabel("Lag (bins)")
        ax.set_ylabel("Unit")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"W_comp{k}.png"), dpi=150)
        plt.close(fig)


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="Path to spike_clusters.npy")
    ap.add_argument("--times",    required=True, help="Path to spike_seconds_adj.npy")
    ap.add_argument("--labels",   required=True, help="Path to templates._bc_unit_labels.tsv")
    ap.add_argument("--classcsv", default=None, help="Path to unit_classification_rulebased.csv; if given, keep only MSN units")
    ap.add_argument("--fr_min", type=float, default=0.5, help="Min firing rate (Hz) in [tmin,tmax]")
    ap.add_argument("--fr_max", type=float, default=5.0, help="Max firing rate (Hz) in [tmin,tmax]")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=600.0, help="Analyze up to this time (sec)")
    ap.add_argument("--binsize", type=float, default=0.02, help="Bin size (sec), e.g., 0.02=20ms")
    ap.add_argument("--topn", type=int, default=100, help="Most active units to include")
    ap.add_argument("--L", type=int, default=20, help="Sequence length in bins (L*binsize = duration)")
    ap.add_argument("--hop", type=int, default=5, help="Window hop (in bins)")
    ap.add_argument("--k", type=int, default=6, help="Number of sequence components")
    ap.add_argument("--alphaH", type=float, default=0.1, help="Sparsity on H (activations)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for NMF init")
    ap.add_argument("--outdir", default="./seqnmf_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load and build matrices
    times, units, allowed_units = load_data(
        args.clusters, args.times, args.labels,
        class_csv=args.classcsv, keep_class="MSN"
    )

    X, keep_units = bin_spikes(
        times, units,
        bin_size=args.binsize,
        t_min=args.tmin, t_max=args.tmax,
        top_n=args.topn,
        allowed_units=allowed_units,
        fr_min=args.fr_min, fr_max=args.fr_max
    )
    patches, ends = build_lagged_patches(X, L=args.L, hop=args.hop, smooth_sigma=1.0)
    N, T = X.shape

    # Fit NMF
    W, H, err = run_seqnmf_like(
        patches,
        n_components=args.k,
        alpha_H=args.alphaH,
        random_state=args.seed,
    )
    
    # Reshape W to (K×N×L)
    W3 = reshape_components(W, N, args.L)

    # Save outputs
    np.save(os.path.join(args.outdir, "X.npy"), X)
    np.save(os.path.join(args.outdir, "W_components.npy"), W3)
    np.save(os.path.join(args.outdir, "H_activations.npy"), H)
    with open(os.path.join(args.outdir, "reconstruction_err.txt"), "w") as f:
        f.write(str(err))

    # CSV: top units per component
    save_top_units_per_component(W3, keep_units, os.path.join(args.outdir, "top_units_per_component.csv"))

    # Peak time estimates (seconds)
    peaks = peak_times(H, hop=args.hop, bin_size=args.binsize, t_min=args.tmin, topk=100)
    with open(os.path.join(args.outdir, "component_peak_times_sec.txt"), "w") as f:
        for k, pk in enumerate(peaks):
            f.write(f"Component {k}: " + ", ".join(f"{p:.3f}" for p in pk) + "\n")

    # Save list of units kept
    pd.DataFrame({"unit": keep_units}).to_csv(os.path.join(args.outdir, "kept_units.csv"), index=False)

    # Plots (if matplotlib)
    maybe_plot_H(H, args.outdir)
    maybe_plot_W(W3, keep_units, args.outdir)

    # Notes for user
    note = []
    note.append(f"Saved outputs to: {os.path.abspath(args.outdir)}")
    note.append(f"Matrix X shape: {X.shape} (units × time bins)")
    note.append(f"Patches shape: {patches.shape} (units*L × windows), L={args.L}, hop={args.hop}, binsize={args.binsize}s")
    note.append(f"K={args.k}, alphaH={args.alphaH}, recon_err={err:.4f}")
    if not HAS_SKLEARN:
        note.append("scikit-learn not found; used NumPy NMF (MU updates). Install scikit-learn for faster/more stable results.")
    if not HAS_SCIPY:
        note.append("scipy not found; skipped smoothing. Install scipy for Gaussian smoothing of spike trains.")
    print("\n".join(note))


if __name__ == "__main__":
    main()