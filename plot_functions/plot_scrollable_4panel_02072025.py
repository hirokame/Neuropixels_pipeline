# -*- coding: utf-8 -*-
"""
Scrollable 4-panel figure (shared time axis):
(1) 13–30 Hz band-passed LFP (chan48, 500 Hz) + Hilbert envelope
(2) Delta, Low-beta, High-beta band power
(3) Center-of-mass (Snout/Tail average) speed from DLC
(4) Spike raster limited to last strobe time

Outputs: an interactive HTML (zoom/pan/scroll) saved next to this script.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, hilbert

# ----------------------------
# Paths (edit if your mount differs)
# ----------------------------
BASE = Path("/Volumes/Extreme SSD/Neuropixels/9153_02072025_tagging_g0/9153_02072025_tagging_g0_imec0/kilosort4")
LFP_PATH = BASE / "lfp_chan48_500Hz.npy"
BANDS_NPZ = BASE / "alignment_amp_tailAccel_02072025.npz"
SPIKE_SECONDS_PATH = BASE / "spike_seconds_adj.npy"      # REQUIRED: times (float64, 1-D)
SPIKE_UNIT_IDS_PATH = BASE / "spike_clusters.npy"        # OPTIONAL: per-spike unit ids (1-D int), same length as times
QMETRICS_TSV = (BASE.parent / "kilosort4qMetrics" / "templates._bc_unit_labels.tsv")
CLASS_CSV = BASE / "unit_classification_rulebased.csv"
STROBE_PATH = BASE / "strobe_seconds.npy"
EXPORT_STATIC = True
STATIC_PNG = BASE / "scrollable_4panel_02072025.png"
STATIC_PDF = BASE / "scrollable_4panel_02072025.pdf"
STATIC_WIDTH = 12000   # pixels (keep < 65000)
STATIC_HEIGHT = 1400   # pixels

DLC_H5 = Path("/Volumes/Extreme SSD/Neuropixels/DLC/9153/9153_Day13_4rewards2025-02-07T13_43_04DLC_HrnetW32_Neuropixel_9153Jun23shuffle1_detector_230_snapshot_210.h5")

# ----------------------------
# Config
# ----------------------------
FS_LFP = 500.0                          # Hz (for lfp_chan48_500Hz.npy)
BETA_BAND = (13.0, 30.0)                # band-pass for beta
SMOOTH_SPEED_SEC = 0.10                 # ~100 ms moving-average on speed (DLC)
STANDARDIZE_BANDS = True                # z-score delta/low-beta/high-beta for plotting
STANDARDIZE_SPEED = True                # z-score COM speed for plotting
HTML_OUT = BASE / "scrollable_4panel_02072025.html"

# ----------------------------
# Helpers
# ----------------------------
def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or not np.isfinite(s):
        s = 1.0
    return (x - m) / s

def bandpass_filt(x, fs, lo, hi, order=4):
    nyq = fs * 0.5
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, x)

def moving_average(x, win_samples):
    if win_samples <= 1:
        return x
    kernel = np.ones(win_samples, dtype=float) / win_samples
    return np.convolve(x, kernel, mode="same")

def build_dlc_timebase_from_strobe(strobe_seconds, n_rows):
    # Interpolate *evenly* from first → last strobe to DLC rows
    return np.linspace(strobe_seconds[0], strobe_seconds[-1], n_rows)

def extract_xy_from_dlc(df, bodypart, scorer=None):
    """
    DLC pandas HDF is usually MultiIndex (scorer, bodypart, coord).
    Returns x,y arrays (float). Raises if missing.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if scorer is None:
            scorer = sorted(set([c[0] for c in df.columns]))[0]
        cols = ( (scorer, bodypart, "x"), (scorer, bodypart, "y") )
        if all(c in df.columns for c in cols):
            x = df[cols[0]].astype(float).to_numpy()
            y = df[cols[1]].astype(float).to_numpy()
            return x, y
        else:
            raise KeyError(f"Bodypart {bodypart} with scorer {scorer} missing x/y columns.")
    else:
        # Single-level fallback (rare)
        xcol = f"{bodypart}_x"
        ycol = f"{bodypart}_y"
        x = df[xcol].astype(float).to_numpy()
        y = df[ycol].astype(float).to_numpy()
        return x, y

def robust_group_spikes(times_path, unit_ids_path=None, good_units=None, msn_units=None, tmax=None):
    """
    Group spikes into (unit_id, spike_times) pairs.
    - Accepts times-only (falls back to single-row raster).
    - If unit_ids_path is provided but lengths differ, truncates BOTH arrays to the common min length.
    - Clips to tmax (e.g., last strobe) BEFORE grouping.
    - Applies optional filters: good_units and msn_units.
    """
    import numpy as np
    from pathlib import Path
    from collections import defaultdict

    # Load spike times
    times = np.load(times_path, allow_pickle=True).astype(float).ravel()
    times = times[np.isfinite(times)]

    # If we don't have unit IDs, return a single-row raster (times only)
    if unit_ids_path is None or not Path(unit_ids_path).exists():
        if tmax is not None:
            times = times[times <= tmax]
        return [(-1, np.sort(times))]

    # Load per-spike unit IDs
    units = np.load(unit_ids_path, allow_pickle=True).astype(int).ravel()

    # --- HANDLE LENGTH MISMATCH BY TRUNCATING TO COMMON MIN LENGTH ---
    if len(units) != len(times):
        n = min(len(units), len(times))
        print(f"WARNING: spike_clusters length ({len(units)}) != spike_seconds length ({len(times)}). "
              f"Truncating to first {n} entries.")
        units = units[:n]
        times = times[:n]

    # Clip to tmax first (so raster is limited to last strobe)
    if tmax is not None:
        m = times <= tmax
        times = times[m]
        units = units[m]

    # Group by unit id
    buckets = defaultdict(list)
    for u, t in zip(units, times):
        if np.isfinite(t):
            buckets[int(u)].append(float(t))

    # Convert to arrays and apply filters (good + MSN)
    grouped = []
    for u in sorted(buckets.keys()):
        if (good_units is not None) and (u not in good_units):
            continue
        if (msn_units is not None) and (u not in msn_units):
            continue
        arr = np.sort(np.asarray(buckets[u], dtype=float))
        if arr.size:
            grouped.append((u, arr))

    # If filtering removed everything, fall back to unfiltered units (so you still see something)
    if not grouped:
        grouped = [(u, np.sort(np.asarray(buckets[u], dtype=float))) for u in sorted(buckets.keys()) if len(buckets[u])]

    return grouped

# ----------------------------
# 1) LFP 13–30 Hz with envelope
# ----------------------------
lfp = np.load(LFP_PATH).astype(float)  # 1D, 500 Hz
lfp_filt = bandpass_filt(lfp, FS_LFP, BETA_BAND[0], BETA_BAND[1])
lfp_env = np.abs(hilbert(lfp_filt))

# 2) Band power (delta/low-beta/high-beta)
bands = np.load(BANDS_NPZ)
time_lfp = bands["time_lfp"].astype(float)

def get_npz_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    raise KeyError(f"None of keys {candidates} found in NPZ. Available: {list(npz.files)}")

delta_key = get_npz_key(bands, ["delta_amp", "delta", "delta_power"])
blo_key   = get_npz_key(bands, ["beta_low_amp", "low_beta_amp", "beta_low"])
bhi_key   = get_npz_key(bands, ["beta_high_amp", "high_beta_amp", "beta_high"])

delta = bands[delta_key].astype(float)
beta_low = bands[blo_key].astype(float)
beta_high = bands[bhi_key].astype(float)

if len(time_lfp) == len(lfp):
    t_lfp = time_lfp
else:
    t_lfp = np.linspace(time_lfp[0], time_lfp[-1], len(lfp))

if STANDARDIZE_BANDS:
    delta_p = zscore(delta)
    blo_p = zscore(beta_low)
    bhi_p = zscore(beta_high)
else:
    delta_p, blo_p, bhi_p = delta, beta_low, beta_high

# 3) Center-of-mass speed from DLC (Snout/Tail average), DLC time from strobe
dlc_df = pd.read_hdf(DLC_H5)
n_rows = len(dlc_df)
strobe = np.load(STROBE_PATH).astype(float)
t_dlc = build_dlc_timebase_from_strobe(strobe, n_rows)
dt_dlc = float(np.median(np.diff(t_dlc)))
t_last_strobe = float(strobe[-1])

snout_x, snout_y = extract_xy_from_dlc(dlc_df, "Snout")
try:
    tail_x, tail_y = extract_xy_from_dlc(dlc_df, "Tail")
except Exception:
    tail_x = tail_y = None
    for alt in ["TailBase", "Tailbase", "TailTip", "Tail_tip"]:
        try:
            tail_x, tail_y = extract_xy_from_dlc(dlc_df, alt)
            break
        except Exception:
            pass
if tail_x is None:
    raise RuntimeError("No Tail-like bodypart found in DLC file.")

def clean_interp(a):
    a = np.asarray(a, dtype=float)
    a[a <= -0.5] = np.nan
    return pd.Series(a).interpolate(limit_direction="both").to_numpy()

snout_x, snout_y = clean_interp(snout_x), clean_interp(snout_y)
tail_x, tail_y   = clean_interp(tail_x),   clean_interp(tail_y)

com_x = 0.5 * (snout_x + tail_x)
com_y = 0.5 * (snout_y + tail_y)

vx = np.gradient(com_x, t_dlc)
vy = np.gradient(com_y, t_dlc)
speed = np.sqrt(vx**2 + vy**2)

# Smooth ~100 ms
win = max(3, int(round(SMOOTH_SPEED_SEC / max(dt_dlc, 1e-9))))
if win % 2 == 0:
    win += 1
kernel = np.ones(win)/win
speed_s = np.convolve(speed, kernel, mode="same")

speed_plot = zscore(speed_s) if STANDARDIZE_SPEED else speed_s

# 4) Spike raster limited to last strobe time; group by unit if possible
# "good" labels (1/2) and MSN only, if available
good_units = None
try:
    qm = pd.read_csv(QMETRICS_TSV, sep="\t")
    # Adjust column names if yours differ:
    good_units = set(qm["unitType"].isin([1, 2]).index.astype(int)).tolist()
except Exception:
    pass

msn_units = None
try:
    uclass = pd.read_csv(CLASS_CSV)
    msn_units = set(uclass.loc[uclass["cell_type"].str.upper()=="MSN", "unit_id"].astype(int).tolist())
except Exception:
    pass

spikes_kept = robust_group_spikes(SPIKE_SECONDS_PATH, SPIKE_UNIT_IDS_PATH if SPIKE_UNIT_IDS_PATH.exists() else None,
                                  good_units=good_units, msn_units=msn_units, tmax=t_last_strobe)

# ----------------------------
# Build interactive, shared-x figure (Plotly)
# ----------------------------
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.25, 0.25, 0.25, 0.25])

# (1) LFP filter and envelope
fig.add_trace(go.Scatter(x=t_lfp, y=lfp_filt, name="13–30 Hz LFP", line=dict(width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=t_lfp, y=lfp_env, name="Envelope", line=dict(width=1)),
              row=1, col=1)

# (2) Band powers
fig.add_trace(go.Scatter(x=time_lfp, y=delta_p, name="Delta"), row=2, col=1)
fig.add_trace(go.Scatter(x=time_lfp, y=blo_p,   name="Low beta"), row=2, col=1)
fig.add_trace(go.Scatter(x=time_lfp, y=bhi_p,   name="High beta"), row=2, col=1)

# (3) COM speed
fig.add_trace(go.Scatter(x=t_dlc, y=speed_plot, name="COM speed"), row=3, col=1)

# (4) Spike raster (units stacked or single row)
if len(spikes_kept) == 1 and spikes_kept[0][0] == -1:
    # single-row fallback
    times = spikes_kept[0][1]
    fig.add_trace(go.Scatter(x=times, y=np.ones_like(times), mode="markers",
                             marker=dict(size=2), name="All spikes ≤ last strobe"),
                  row=4, col=1)
    fig.update_yaxes(range=[0.5, 1.5], title_text="All spikes", row=4, col=1)
else:
    # multi-unit raster: offset each unit on the y-axis
    y_offset = 0
    for uid, times in spikes_kept:
        y = np.full_like(times, y_offset + 1.0, dtype=float)
        fig.add_trace(go.Scatter(x=times, y=y, mode="markers",
                         marker=dict(size=2, color="black"),
                         name=f"u{uid}", showlegend=False),
              row=4, col=1)
        y_offset += 1
    fig.update_yaxes(range=[0, y_offset + 1], title_text="Unit (good & MSN)", row=4, col=1)

# Shared x axis, wide layout (scroll/zoom via Plotly UI)
fig.update_layout(
    height=700,
    width=2400,   # big canvas; zoom/pan for more detail
    title_text="Scrollable 4-panel (shared x, spikes ≤ last strobe)",
    hovermode="x unified",
)

# Save static images (requires: pip install -U kaleido)
if EXPORT_STATIC:
    try:
        import plotly.io as pio
        # Share the same x-range up to last strobe for every subplot
        t0 = min(t_lfp[0], time_lfp[0], t_dlc[0])
        fig.update_xaxes(range=[t0, t_last_strobe])
        fig.write_image(STATIC_PNG, width=STATIC_WIDTH, height=STATIC_HEIGHT, scale=1)
        fig.write_image(STATIC_PDF, width=STATIC_WIDTH, height=STATIC_HEIGHT, scale=1)
        print(f"Saved {STATIC_PNG} and {STATIC_PDF}")
    except Exception as e:
        print("Static export failed. Install kaleido:  pip install -U kaleido")
        raise

# Axis labels
fig.update_yaxes(title_text="LFP (a.u.)", row=1, col=1)
fig.update_yaxes(title_text="Band power (z)" if STANDARDIZE_BANDS else "Band power", row=2, col=1)
fig.update_yaxes(title_text="COM speed (z)" if STANDARDIZE_SPEED else "COM speed", row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=4, col=1)

# Save to HTML
fig.write_html(HTML_OUT, include_plotlyjs="cdn")
print(f"Saved interactive HTML: {HTML_OUT}")
