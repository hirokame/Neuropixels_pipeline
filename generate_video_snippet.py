"""
Generate a 30-second composite video snippet combining:
  1. Actual recorded video (60 fps)
  2. Spike raster plot (rastermap-sorted, scrolling)
  3. LFP spectrogram (wavelet on CSD-preprocessed LFP, scrolling)
  4. Mouse speed from DLC (scrolling)
  5. Lick (black) & water delivery (red) vertical lines (scrolling)
  6. Firing rate per cell on 4-shank probe map
  7. Beta-band power heatmap on 4-shank probe map
  8. Gamma-band power heatmap on 4-shank probe map

Usage:
    python generate_video_snippet.py \
        --mouse 3546 --day 01072026 \
        --base-path "/Volumes/Extreme SSD/Neuropixels/Python/DemoData_3546" \
        --start-time 120.0 --duration 30.0 \
        --output composite_snippet.mp4
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import morlet2, cwt
from scipy.ndimage import gaussian_filter1d

import cv2
import imageio.v3 as iio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Add project root so we can import the data loaders
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "postanalysis"))

from postanalysis.data_loader import (
    load_session_data,
    SpikeDataLoader,
    DLCDataLoader,
    EventDataLoader,
    LFPDataLoader,
    StrobeDataLoader,
    print_data_summary,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: wavelet spectrogram
# ═══════════════════════════════════════════════════════════════════════════
def wavelet_spectrogram(signal, fs, freqs=None, n_cycles=6):
    """Compute a Morlet-wavelet spectrogram.

    Parameters
    ----------
    signal : 1-D array
    fs     : sampling rate (Hz)
    freqs  : array of frequencies to evaluate (default 1–150 Hz)
    n_cycles : number of wavelet cycles

    Returns
    -------
    power : (n_freqs, n_times) – squared magnitude
    freqs : frequency vector
    """
    if freqs is None:
        freqs = np.logspace(np.log10(1), np.log10(150), 80)

    widths = n_cycles * fs / (2 * np.pi * freqs)
    cwtm = cwt(signal, morlet2, widths, w=n_cycles)
    power = np.abs(cwtm) ** 2
    return power, freqs


# ═══════════════════════════════════════════════════════════════════════════
# Helper: band power per channel
# ═══════════════════════════════════════════════════════════════════════════
def band_power_per_channel(lfp_traces, fs, band, win_samples=None):
    """Compute average band power for each channel over a short window.

    Parameters
    ----------
    lfp_traces : (n_samples, n_channels)
    fs         : sampling rate
    band       : (lo, hi) in Hz
    win_samples: if given, use only the last `win_samples` of the trace

    Returns
    -------
    power : (n_channels,)
    """
    from scipy.signal import butter, sosfiltfilt

    if win_samples is not None:
        lfp_traces = lfp_traces[-win_samples:]

    lo, hi = band
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, lfp_traces, axis=0)
    power = np.mean(filtered ** 2, axis=0)
    return power


# ═══════════════════════════════════════════════════════════════════════════
# Helper: draw 4-shank probe map as smooth gradient heatmap
# ═══════════════════════════════════════════════════════════════════════════
def draw_probe_map(ax, channel_positions, values, title, cmap='hot',
                   vmin=None, vmax=None, n_interp=300, shank_width_um=60):
    """
    For each shank render a smooth vertical gradient strip (interpolated along
    depth), giving a clean heatmap appearance instead of scattered dots.
    Returns a ScalarMappable suitable for fig.colorbar().
    """
    from scipy.interpolate import interp1d

    ax.clear()
    vmin = vmin if vmin is not None else np.nanmin(values)
    vmax = vmax if vmax is not None else np.nanmax(values)
    norm  = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    x_all = channel_positions[:, 0]
    y_all = channel_positions[:, 1]

    # Identify shanks robustly: split unique x values at large gaps (inter-shank >> intra-shank)
    unique_x = np.unique(x_all)
    gaps = np.diff(unique_x)
    gap_threshold = np.max(gaps) * 0.5          # large gap = shank boundary
    boundaries = np.where(gaps > gap_threshold)[0]
    shank_x_groups = np.split(unique_x, boundaries + 1)   # list of x arrays per shank

    for x_group in shank_x_groups:
        mask = np.isin(x_all, x_group)
        if not np.any(mask):
            continue

        y_ch = y_all[mask]
        v_ch = values[mask]

        # Average across both columns at each depth level
        sort_idx = np.argsort(y_ch)
        y_s = y_ch[sort_idx]
        v_s = v_ch[sort_idx]

        # Collapse duplicate depths by averaging — avoids divide-by-zero in interp1d
        y_unique, inv = np.unique(y_s, return_inverse=True)
        v_unique = np.array([v_s[inv == i].mean() for i in range(len(y_unique))])
        y_s, v_s = y_unique, v_unique

        if len(y_s) < 2:
            continue

        # Interpolate values to a fine depth grid
        y_fine = np.linspace(y_s[0], y_s[-1], n_interp)
        f_interp = interp1d(y_s, v_s, kind='linear', bounds_error=False,
                            fill_value=(v_s[0], v_s[-1]))
        v_fine = f_interp(y_fine)

        # Draw each shank as a strip spanning its actual x extent
        x_lo = x_group.min() - shank_width_um * 0.3
        x_hi = x_group.max() + shank_width_um * 0.3
        strip = v_fine[:, np.newaxis]             # (n_interp, 1)
        ax.imshow(
            strip, aspect='auto', origin='upper',
            extent=[x_lo, x_hi, y_s[-1], y_s[0]],
            cmap=cmap_obj, norm=norm,
            interpolation='bilinear',
        )

    x_lo_all = x_all.min() - shank_width_um
    x_hi_all = x_all.max() + shank_width_um
    ax.set_xlim(x_lo_all, x_hi_all)
    y_min, y_max = y_all.min(), y_all.max()
    ax.set_ylim(y_max + 50, y_min - 50)          # depth increases downward
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_ylabel('Depth (µm)', fontsize=6)
    ax.set_xlabel('X (µm)', fontsize=5)
    ax.tick_params(labelsize=5)

    # Return ScalarMappable for colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    return sm


# ═══════════════════════════════════════════════════════════════════════════
# Main video generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_composite_video(
    mouse_id: str,
    day: str,
    base_path: str,
    start_time: float = 60.0,
    duration: float = 30.0,
    output_path: str = "composite_snippet.mp4",
    video_fps: int = 60,
    output_fps: int = 60,
    rastermap_subset: str = "msn_fsi",
    scrolling_window: float = 5.0,
):
    """Build the composite video frame-by-frame."""

    end_time = start_time + duration
    # Extend data window by one full scrolling_window before start so the
    # left side of every scrolling panel is always populated from frame 0.
    data_start = max(0.0, start_time - scrolling_window)

    # ------------------------------------------------------------------
    # 1. Resolve all paths
    # ------------------------------------------------------------------
    paths = load_session_data(mouse_id, day, base_path)
    print_data_summary(paths)

    # ------------------------------------------------------------------
    # Validate critical paths before proceeding
    # ------------------------------------------------------------------
    missing = []
    if not paths.neural_base or not paths.neural_base.exists():
        missing.append(f"Neural data (imec dir) — check mouse ID and date match a session folder in {base_path}")
    if not paths.kilosort_dir or not paths.kilosort_dir.exists():
        missing.append(f"Kilosort output dir: {paths.kilosort_dir}")
    if not paths.lfp_dir or not paths.lfp_dir.exists():
        missing.append(f"LFP dir: {paths.lfp_dir}")
    if not paths.strobe_seconds or not paths.strobe_seconds.exists():
        missing.append(f"strobe_seconds.npy in {paths.kilosort_dir}")
    if not paths.dlc_h5 or not paths.dlc_h5.exists():
        missing.append(f"DLC .h5 file in {base_path}/DLC/")
    if not paths.video_avi or not paths.video_avi.exists():
        missing.append(f"Video .avi file in {base_path}/Video/")
    if missing:
        print("\n❌ ERROR: Required files not found:")
        for m in missing:
            print(f"   • {m}")
        print(f"\nAvailable session folders in {base_path}:")
        for d in sorted(Path(base_path).glob("*_g0")):
            print(f"   {d.name}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load spikes
    # ------------------------------------------------------------------
    logger.info("Loading spike data ...")
    spike_loader = SpikeDataLoader(paths.neural_base)
    spike_data = spike_loader.load(paths.kilosort_dir)
    spike_times = spike_data['spike_times_sec']
    spike_clusters = spike_data['spike_clusters']
    unique_clusters = spike_data['unique_clusters']

    # ------------------------------------------------------------------
    # 3. Load rastermap sorting
    # ------------------------------------------------------------------
    logger.info("Loading rastermap sorting ...")
    rmap_dir = paths.rastermap_dir / rastermap_subset
    isort = np.load(rmap_dir / "isort.npy")
    rmap_cluster_ids = np.load(rmap_dir / "cluster_ids.npy")

    # Build cluster -> sorted-row mapping
    cluster_to_row = {int(cid): row for row, cid in enumerate(rmap_cluster_ids[isort])}
    n_sorted_units = len(rmap_cluster_ids)

    # ------------------------------------------------------------------
    # 4. Load channel positions (for probe maps)
    # ------------------------------------------------------------------
    channel_positions = np.load(paths.kilosort_dir / "channel_positions.npy")

    # Unit -> best channel position (from templates)
    templates = np.load(paths.kilosort_dir / "templates.npy")        # (n_units, n_time, n_ch)
    templates_ind = np.load(paths.kilosort_dir / "templates_ind.npy")  # (n_units, n_ch)
    unit_positions = {}
    for uid in unique_clusters:
        uid = int(uid)
        if uid < templates.shape[0]:
            best_ch_local = np.argmax(np.max(np.abs(templates[uid]), axis=0))
            if uid < templates_ind.shape[0] and best_ch_local < templates_ind.shape[1]:
                best_ch_global = templates_ind[uid, best_ch_local]
                if best_ch_global < len(channel_positions):
                    unit_positions[uid] = channel_positions[best_ch_global]

    # ------------------------------------------------------------------
    # 5. Load strobe -> frame <-> time mapping
    # ------------------------------------------------------------------
    logger.info("Loading strobe timestamps ...")
    strobe_loader = StrobeDataLoader(paths.neural_base)
    strobe_times = strobe_loader.load(paths.strobe_seconds)

    # ------------------------------------------------------------------
    # 6. Load DLC -> speed
    # ------------------------------------------------------------------
    logger.info("Loading DLC data & computing speed ...")
    dlc_loader = DLCDataLoader(paths.neural_base)
    df_dlc = dlc_loader.load(paths.dlc_h5)
    velocity, velocity_times = dlc_loader.calculate_velocity(
        df_dlc, video_fs=video_fps, strobe_path=paths.strobe_seconds
    )

    # ------------------------------------------------------------------
    # 7. Load events (lick & water)
    # ------------------------------------------------------------------
    logger.info("Loading lick & water events ...")
    # Lick & water times — load directly from pre-extracted npy files
    lick_npy   = paths.kilosort_dir / "licking_seconds.npy"
    reward_npy = paths.kilosort_dir / "reward_seconds.npy"

    if lick_npy.exists():
        lick_times = np.load(lick_npy).flatten()
        logger.info(f"Loaded {len(lick_times)} lick events from licking_seconds.npy")
    else:
        logger.warning("licking_seconds.npy not found — falling back to EventDataLoader")
        event_loader = EventDataLoader(paths.neural_base)
        lick_times = event_loader.get_event_times_by_type('licking', paths)

    if reward_npy.exists():
        water_times = np.load(reward_npy).flatten()
        logger.info(f"Loaded {len(water_times)} reward events from reward_seconds.npy")
    else:
        logger.warning("reward_seconds.npy not found — falling back to EventDataLoader")
        if 'event_loader' not in dir():
            event_loader = EventDataLoader(paths.neural_base)
        water_times = event_loader.get_event_times_by_type('reward', paths)

    # ------------------------------------------------------------------
    # 8. Load LFP (CSD reference)
    # ------------------------------------------------------------------
    logger.info("Loading LFP data (CSD reference) ...")
    lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
    lfp_fs = lfp_loader.fs

    all_ch_ids = list(lfp_loader.channel_ids)
    n_lfp_ch = len(all_ch_ids)  # 384

    logger.info(f"Loading LFP traces for spectrogram (t={data_start:.1f}–{end_time:.1f}) ...")
    lfp_full, lfp_full_t = lfp_loader.get_data(
        data_start, end_time, channels=None, reference='csd'
    )
    # Pick one channel (middle of shank 0) for the scrolling spectrogram
    lfp_spec_ch_idx = n_lfp_ch // 8
    lfp_spec_signal = lfp_full[:, lfp_spec_ch_idx]
    logger.info(f"Spectrogram channel: {lfp_spec_ch_idx} of {n_lfp_ch}")

    logger.info("Computing wavelet spectrogram ...")
    spec_power, spec_freqs = wavelet_spectrogram(lfp_spec_signal, lfp_fs)
    # Convert to dB
    spec_power_db = 10 * np.log10(spec_power + 1e-12)

    # ------------------------------------------------------------------
    # 9. Open behaviour video via imageio+ffmpeg (handles non-standard AVI)
    # ------------------------------------------------------------------
    logger.info(f"Opening video: {paths.video_avi}")
    video_path_str = str(paths.video_avi)

    # Verify we can read at least one frame
    try:
        _test = iio.imread(video_path_str, index=0, plugin='FFMPEG')
        logger.info(f"Video frame shape: {_test.shape}")
    except Exception as e:
        raise IOError(f"Cannot open video {paths.video_avi}: {e}")

    # Map start_time -> video frame via strobe
    start_frame_idx = int(np.searchsorted(strobe_times, start_time))
    end_frame_idx = int(np.searchsorted(strobe_times, end_time))
    n_snippet_frames = end_frame_idx - start_frame_idx

    # We'll render at output_fps (default 30), stepping through video frames
    step = max(1, video_fps // output_fps)  # e.g. 60/30 = skip every other frame
    frame_indices = np.arange(start_frame_idx, end_frame_idx, step)
    n_output_frames = len(frame_indices)

    logger.info(
        f"Video frames {start_frame_idx}–{end_frame_idx} "
        f"({n_snippet_frames} frames), outputting {n_output_frames} frames at {output_fps} fps"
    )

    # ------------------------------------------------------------------
    # 10. Pre-compute spike raster data in window
    # ------------------------------------------------------------------
    logger.info("Pre-computing spike raster data ...")
    mask = (spike_times >= data_start) & (spike_times < end_time)
    raster_times = spike_times[mask]
    raster_clusters = spike_clusters[mask]

    # Map to sorted row
    raster_rows = np.array([cluster_to_row.get(int(c), -1) for c in raster_clusters])
    valid = raster_rows >= 0
    raster_times = raster_times[valid]
    raster_rows = raster_rows[valid]

    # ------------------------------------------------------------------
    # 11. Pre-compute firing rates per unit (binned at video frame rate)
    # ------------------------------------------------------------------
    logger.info("Pre-computing per-unit firing rates ...")
    fr_bin_edges = strobe_times[start_frame_idx:end_frame_idx + 1]

    # Use the already-masked arrays from the raster step (spikes in snippet window)
    # raster_times / raster_clusters (before valid filter) are what we need here
    win_mask = (spike_times >= data_start) & (spike_times < end_time)
    win_times = spike_times[win_mask]
    win_clusters = spike_clusters[win_mask]

    unit_fr = {}
    n_bins = max(len(fr_bin_edges) - 1, 1)
    for uid in rmap_cluster_ids:
        uid = int(uid)
        uid_spikes = win_times[win_clusters == uid]
        if len(uid_spikes) > 0 and len(fr_bin_edges) > 1:
            counts, _ = np.histogram(uid_spikes, bins=fr_bin_edges)
            dt = np.diff(fr_bin_edges)
            dt[dt == 0] = 1.0 / video_fps
            unit_fr[uid] = counts / dt  # Hz
        else:
            unit_fr[uid] = np.zeros(n_bins)

    # ------------------------------------------------------------------
    # 12. Pre-compute band power snapshots per frame (sub-sampled)
    #     We compute every `power_step` frames and interpolate
    # ------------------------------------------------------------------
    power_step = step  # same as video step
    beta_band = (12, 30)
    gamma_band = (30, 80)

    logger.info("Pre-computing beta/gamma band power per channel ...")
    # We'll compute power in a sliding 500 ms window around each output frame
    power_win_sec = 0.5
    power_win_samples = int(power_win_sec * lfp_fs)

    # Store power snapshots: indexed by output frame index
    beta_power_frames = {}
    gamma_power_frames = {}

    for out_i, fi in enumerate(frame_indices):
        t_now = strobe_times[fi] if fi < len(strobe_times) else end_time
        t_win_start = t_now - power_win_sec / 2
        t_win_end = t_now + power_win_sec / 2

        # Map to LFP sample indices within pre-loaded lfp_full
        lfp_t0 = lfp_full_t[0]
        s0 = max(0, int((t_win_start - lfp_t0) * lfp_fs))
        s1 = min(len(lfp_full_t), int((t_win_end - lfp_t0) * lfp_fs))

        if s1 - s0 < 10:
            beta_power_frames[out_i] = np.zeros(n_lfp_ch)
            gamma_power_frames[out_i] = np.zeros(n_lfp_ch)
            continue

        chunk = lfp_full[s0:s1, :]  # all 384 channels

        beta_power_frames[out_i] = band_power_per_channel(chunk, lfp_fs, beta_band)
        gamma_power_frames[out_i] = band_power_per_channel(chunk, lfp_fs, gamma_band)

    # Global vmin/vmax for consistent colour scale
    all_beta = np.array(list(beta_power_frames.values()))
    all_gamma = np.array(list(gamma_power_frames.values()))
    beta_vmin, beta_vmax = np.percentile(all_beta[all_beta > 0], [5, 95]) if np.any(all_beta > 0) else (0, 1)
    gamma_vmin, gamma_vmax = np.percentile(all_gamma[all_gamma > 0], [5, 95]) if np.any(all_gamma > 0) else (0, 1)

    # FR global scale
    all_fr_vals = []
    for uid in unit_positions:
        if uid in unit_fr and len(unit_fr[uid]) > 0:
            all_fr_vals.append(np.percentile(unit_fr[uid], 95))
    fr_vmax = np.percentile(all_fr_vals, 95) if all_fr_vals else 10.0

    # ------------------------------------------------------------------
    # 13. Setup matplotlib figure layout
    # ------------------------------------------------------------------
    logger.info("Setting up figure layout ...")
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    # Total pixel size: 1920 x 1080

    # Original layout:
    #   Row 0: video (col 0) | FR probe (col 2) | Beta probe (col 3) | Gamma probe (col 4)
    #   Row 1: raster (all cols)
    #   Row 2: spectrogram (all cols)
    #   Row 3: speed + events (all cols)
    gs = gridspec.GridSpec(
        4, 4,
        height_ratios=[3, 2.5, 2, 1.2],
        width_ratios=[2.5, 1, 1, 1],
        hspace=0.38, wspace=0.32,
        left=0.05, right=0.97, top=0.96, bottom=0.04,
    )

    ax_video  = fig.add_subplot(gs[0, 0])
    ax_fr     = fig.add_subplot(gs[0, 1])
    ax_beta   = fig.add_subplot(gs[0, 2])
    ax_gamma  = fig.add_subplot(gs[0, 3])
    ax_raster = fig.add_subplot(gs[1, :])
    ax_spec   = fig.add_subplot(gs[2, :])
    ax_speed  = fig.add_subplot(gs[3, :])

    # Note: do NOT use sharex — clear() resets shared limits and causes blank panels.
    # Instead each panel sets its own xlim to (t_win_lo, t_win_hi) every frame.

    # ------------------------------------------------------------------
    # 13b. Create colorbars once (vmin/vmax are fixed for the whole video)
    # ------------------------------------------------------------------
    _sm_fr    = ScalarMappable(norm=Normalize(0, fr_vmax),               cmap=plt.get_cmap('viridis'))
    _sm_beta  = ScalarMappable(norm=Normalize(beta_vmin,  beta_vmax),    cmap=plt.get_cmap('hot'))
    _sm_gamma = ScalarMappable(norm=Normalize(gamma_vmin, gamma_vmax),   cmap=plt.get_cmap('inferno'))
    for sm in (_sm_fr, _sm_beta, _sm_gamma):
        sm.set_array([])

    cb_fr    = fig.colorbar(_sm_fr,    ax=ax_fr,    pad=0.02, fraction=0.08, label='FR (Hz)',     format='%.1f')
    cb_beta  = fig.colorbar(_sm_beta,  ax=ax_beta,  pad=0.02, fraction=0.08, label='Power',       format='%.2f')
    cb_gamma = fig.colorbar(_sm_gamma, ax=ax_gamma, pad=0.02, fraction=0.08, label='Power',       format='%.2f')
    _sm_spec = ScalarMappable(norm=Normalize(0, 1), cmap=plt.get_cmap('jet'))  # updated after spec computed
    _sm_spec.set_array([])

    # ------------------------------------------------------------------
    # 14. Set up video writer (imageio + ffmpeg — reliable on macOS)
    # ------------------------------------------------------------------
    frame_w, frame_h = 1920, 1080
    output_path = str(output_path)

    import imageio as iio2
    writer = iio2.get_writer(
        output_path,
        fps=output_fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
    )
    logger.info(f"Video writer opened (imageio/ffmpeg libx264) -> {output_path}")

    # ------------------------------------------------------------------
    # 15. Render frames
    # ------------------------------------------------------------------
    logger.info(f"Rendering {n_output_frames} frames ...")

    # Spectrogram extent (time axis maps to lfp_full_t)
    spec_extent = [lfp_full_t[0], lfp_full_t[-1], spec_freqs[0], spec_freqs[-1]]
    spec_vmin = np.percentile(spec_power_db, 5)
    spec_vmax = np.percentile(spec_power_db, 95)
    _sm_spec.norm = Normalize(spec_vmin, spec_vmax)
    cb_spec = fig.colorbar(_sm_spec, ax=ax_spec, pad=0.01, fraction=0.02,
                           label='Power (dB)', format='%.0f')

    for out_i, fi in tqdm(enumerate(frame_indices), total=n_output_frames,
                          desc="Rendering frames", unit="frame"):

        t_now = strobe_times[fi] if fi < len(strobe_times) else end_time
        t_win_lo = t_now - scrolling_window / 2
        t_win_hi = t_now + scrolling_window / 2

        # === (A) Video frame ===
        try:
            frame_rgb = iio.imread(video_path_str, index=int(fi), plugin='FFMPEG')
        except Exception:
            frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        ax_video.clear()
        ax_video.imshow(frame_rgb, aspect='equal')   # preserve video aspect ratio
        ax_video.set_title(f't = {t_now:.2f} s', fontsize=8)
        ax_video.axis('off')

        # Pre-compute lick/water events in this scrolling window (shared across panels)
        lick_in_win  = lick_times[(lick_times  >= t_win_lo) & (lick_times  < t_win_hi)]
        water_in_win = water_times[(water_times >= t_win_lo) & (water_times < t_win_hi)]

        def _draw_event_lines(ax, lick_color='k', water_color='red',
                              lick_lw=0.5, water_lw=1.2, alpha_lick=0.55, alpha_water=0.85):
            for lt in lick_in_win:
                ax.axvline(lt, color=lick_color, lw=lick_lw, alpha=alpha_lick)
            for wt in water_in_win:
                ax.axvline(wt, color=water_color, lw=water_lw, alpha=alpha_water)

        # === (B) Spike raster (scrolling window) ===
        ax_raster.clear()
        rmask = (raster_times >= t_win_lo) & (raster_times < t_win_hi)
        ax_raster.scatter(
            raster_times[rmask], raster_rows[rmask],
            s=2, c='k', marker='|', linewidths=0.5
        )
        _draw_event_lines(ax_raster, lick_lw=0.5, water_lw=1.2)
        ax_raster.axvline(t_now, color='cyan', lw=0.8, alpha=0.8)
        ax_raster.set_xlim(t_win_lo, t_win_hi)
        ax_raster.set_ylim(n_sorted_units, 0)
        ax_raster.set_ylabel('Unit (sorted)', fontsize=7)
        ax_raster.set_title('Spike Raster (rastermap) | Lick (black) | Water (red)', fontsize=8)
        ax_raster.tick_params(labelsize=6)
        ax_raster.set_xticklabels([])

        # === (C) LFP spectrogram (scrolling) ===
        ax_spec.clear()
        spec_t_mask = (lfp_full_t >= t_win_lo) & (lfp_full_t < t_win_hi)
        spec_t_idx = np.where(spec_t_mask)[0]
        if len(spec_t_idx) > 0:
            spec_chunk = spec_power_db[:, spec_t_idx[0]:spec_t_idx[-1]+1]
            im_spec = ax_spec.imshow(
                spec_chunk, aspect='auto', origin='lower',
                extent=[lfp_full_t[spec_t_idx[0]], lfp_full_t[spec_t_idx[-1]],
                        spec_freqs[0], spec_freqs[-1]],
                cmap='jet', vmin=spec_vmin, vmax=spec_vmax,
                interpolation='bilinear'
            )
        _draw_event_lines(ax_spec, lick_color='w', water_color='yellow',
                          lick_lw=0.5, water_lw=1.2)
        ax_spec.axvline(t_now, color='cyan', lw=0.8, alpha=0.8)
        ax_spec.set_xlim(t_win_lo, t_win_hi)
        ax_spec.set_ylim(spec_freqs[0], spec_freqs[-1])
        ax_spec.set_ylabel('Freq (Hz)', fontsize=7)
        ax_spec.set_title('LFP Spectrogram (CSD + wavelet) | Lick (white) | Water (yellow)', fontsize=8)
        ax_spec.tick_params(labelsize=6)
        ax_spec.set_xticklabels([])

        # === (D) Speed + lick/water (scrolling) ===
        ax_speed.clear()
        spd_mask = (velocity_times >= t_win_lo) & (velocity_times < t_win_hi)
        ax_speed.plot(velocity_times[spd_mask], velocity[spd_mask],
                      color='steelblue', lw=0.8)
        _draw_event_lines(ax_speed, lick_lw=0.5, water_lw=1.2)
        ax_speed.axvline(t_now, color='r', lw=0.8, alpha=0.7)
        ax_speed.set_xlim(t_win_lo, t_win_hi)
        ax_speed.set_ylabel('Speed\n(cm/s)', fontsize=7)
        ax_speed.set_xlabel('Time (s)', fontsize=7)
        ax_speed.set_title('Speed | Lick (black) | Water (red)', fontsize=8)
        ax_speed.tick_params(labelsize=6)

        # === (E) FR on probe map ===
        frame_rel = fi - start_frame_idx
        fr_uids = list(unit_positions.keys())
        fr_xy = np.array([unit_positions[u] for u in fr_uids])
        fr_vals = np.array([
            unit_fr[u][min(frame_rel, len(unit_fr[u])-1)] if u in unit_fr else 0.0
            for u in fr_uids
        ])
        draw_probe_map(ax_fr, fr_xy, fr_vals, 'FR (Hz)', cmap='viridis',
                       vmin=0, vmax=fr_vmax)

        # === (F) Beta power on probe map (all 384 channels) ===
        bp = beta_power_frames.get(out_i, np.zeros(n_lfp_ch))
        draw_probe_map(ax_beta, channel_positions, bp, 'Beta (12-30 Hz)',
                       cmap='hot', vmin=beta_vmin, vmax=beta_vmax)

        # === (G) Gamma power on probe map (all 384 channels) ===
        gp = gamma_power_frames.get(out_i, np.zeros(n_lfp_ch))
        draw_probe_map(ax_gamma, channel_positions, gp, 'Gamma (30-80 Hz)',
                       cmap='inferno', vmin=gamma_vmin, vmax=gamma_vmax)

        # === Render figure to image ===
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB

        # Resize to output resolution if needed (cv2 works on RGB too)
        if img.shape[1] != frame_w or img.shape[0] != frame_h:
            img = cv2.resize(img, (frame_w, frame_h))

        writer.append_data(img)

    # ------------------------------------------------------------------
    # 16. Cleanup
    # ------------------------------------------------------------------
    writer.close()
    plt.close(fig)
    logger.info(f"Done! Saved composite video to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate a 30-sec composite neural+behavioral video snippet."
    )
    parser.add_argument("--mouse", required=True, help="Mouse ID (e.g. 3546)")
    parser.add_argument("--day", required=True, help="Session date (MMDDYYYY, e.g. 01072026)")
    parser.add_argument("--base-path", required=True, help="Base data directory")
    parser.add_argument("--start-time", type=float, default=60.0,
                        help="Start time in seconds (default: 60)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration in seconds (default: 30)")
    parser.add_argument("--output", default="composite_snippet.mp4",
                        help="Output video path")
    parser.add_argument("--output-fps", type=int, default=60,
                        help="Output video FPS (default: 60, matches source)")
    parser.add_argument("--rastermap-subset", default="msn_fsi",
                        choices=["all_good_mua", "msn_fsi", "msn_only"],
                        help="Rastermap subset to use (default: msn_fsi)")
    parser.add_argument("--scrolling-window", type=float, default=5.0,
                        help="Width of scrolling window in seconds (default: 5)")

    args = parser.parse_args()

    generate_composite_video(
        mouse_id=args.mouse,
        day=args.day,
        base_path=args.base_path,
        start_time=args.start_time,
        duration=args.duration,
        output_path=args.output,
        output_fps=args.output_fps,
        rastermap_subset=args.rastermap_subset,
        scrolling_window=args.scrolling_window,
    )


if __name__ == "__main__":
    main()
