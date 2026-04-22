"""
Analyses specifically implemented for the Open Field task,
focusing on modeling differences in brain processing (Spikes, LFP, Dopamine)
and their relation to fine kinematics for Shank-KO / WT comparison.
Extracts a comprehensive suite of metrics into a single CSV.
"""

import pandas as pd
import numpy as np
import scipy.stats
from scipy.signal import welch, butter, filtfilt, hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from logging import getLogger
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from data_loader import (
    DataPaths,
    DLCDataLoader,
    StrobeDataLoader,
    PhotometryDataLoader,
    SpikeDataLoader,
    LFPDataLoader,
    VAMEDataLoader
)

logger = getLogger(__name__)

def _compute_kinematic_metrics(velocity, velocity_times):
    """Calculates openfield locomotion metrics."""
    if len(velocity) == 0:
        return {}
    
    total_distance = np.trapz(velocity, velocity_times) if len(velocity_times) > 1 else np.nan
    
    move_threshold = 2.0
    is_moving = velocity > move_threshold
    time_moving_perc = np.mean(is_moving) * 100
    
    avg_speed_move = np.mean(velocity[is_moving]) if np.any(is_moving) else np.nan
    
    onsets = np.where(np.diff(is_moving.astype(int)) == 1)[0] + 1
    offsets = np.where(np.diff(is_moving.astype(int)) == -1)[0]
    
    if is_moving[0]: onsets = np.insert(onsets, 0, 0)
    if is_moving[-1]: offsets = np.append(offsets, len(is_moving)-1)
        
    n_bouts = len(onsets)
    session_duration_min = (velocity_times[-1] - velocity_times[0]) / 60.0 if len(velocity_times) > 1 else 0
    bout_freq = n_bouts / session_duration_min if session_duration_min > 0 else np.nan
    
    duration_sum = 0
    valid_bouts = min(len(onsets), len(offsets))
    for i in range(valid_bouts):
        dur = velocity_times[offsets[i]] - velocity_times[onsets[i]]
        duration_sum += dur
    avg_bout_dur = duration_sum / valid_bouts if valid_bouts > 0 else 0
    
    return {
        'total_distance_cm': total_distance,
        'time_moving_perc': time_moving_perc,
        'avg_speed_moving_cm_s': avg_speed_move,
        'movement_bout_freq_per_min': bout_freq,
        'avg_bout_duration_sec': avg_bout_dur
    }

def _get_vame_transition_matrix(labels, n_states=15):
    """Computes categorical transition matrix (excluding self-transitions)."""
    if labels is None or len(labels) < 2:
        return np.zeros((n_states, n_states))
    
    # Extract sequence of state changes (no self-transitions)
    seq = labels[np.insert(np.diff(labels) != 0, 0, True)]
    mat = np.zeros((n_states, n_states))
    for i in range(len(seq) - 1):
        s1, s2 = seq[i], seq[i+1]
        if s1 < n_states and s2 < n_states:
            mat[s1, s2] += 1
    return mat


def _compute_latent_stereotypy(latent_vectors, fps=60, target_hz=6):
    """Compute continuous stereotypy metrics directly from the 30D latent trajectory.

    Groups:
        2. 30D Recurrence Quantification Analysis (RQA)
        3. Latent Space Variance / Compactness (PCA, Cov trace, Participation Ratio)
        4. Temporal Autocorrelation & AR predictability
        5. Latent Speed & Path Tortuosity
        6. Intra-session Similarity (first-half vs second-half DTW)
    """
    from sklearn.decomposition import PCA
    metrics = {}

    T, D = latent_vectors.shape
    # --- Downsampled version for expensive ops ---
    step = max(1, fps // target_hz)
    Z = latent_vectors[::step].astype(np.float32)   # shape: (T', 30)
    T_ = len(Z)

    # ---------------------------------------------------------------- #
    # Group 2: 30D RQA                                                  #
    # ---------------------------------------------------------------- #
    try:
        # Use a subsample cap to keep pairwise distance matrix manageable
        rqa_cap = min(T_, 2000)
        Zr = Z[:rqa_cap]
        # Compute pairwise Euclidean distances (upper triangle)
        diff = Zr[:, None, :] - Zr[None, :, :]   # (N, N, D)
        dist_mat = np.sqrt((diff ** 2).sum(axis=-1))

        # ε = 10th percentile of all pairwise distances
        eps = np.percentile(dist_mat, 10)
        R = (dist_mat <= eps).astype(np.uint8)
        np.fill_diagonal(R, 0)   # exclude identity line

        N = rqa_cap
        total_possible = N * (N - 1)
        rec_rate = R.sum() / total_possible if total_possible > 0 else 0

        # Determinism: %DET and L_max from diagonal line histogram
        diag_lengths = []
        for d_offset in range(1, N):
            diag = np.diagonal(R, offset=d_offset)
            run, runs = 0, []
            for v in diag:
                if v: run += 1
                elif run > 0: runs.append(run); run = 0
            if run > 0: runs.append(run)
            diag_lengths.extend(runs)

        if diag_lengths:
            diag_lengths = np.array(diag_lengths)
            min_line = 2
            long_pts = diag_lengths[diag_lengths >= min_line].sum()
            all_rec_pts = diag_lengths.sum()
            det = long_pts / all_rec_pts if all_rec_pts > 0 else 0
            lmax = int(diag_lengths.max())
        else:
            det, lmax = 0, 0

        metrics['vame_latent_rqa_rec'] = float(rec_rate)
        metrics['vame_latent_rqa_det'] = float(det)
        metrics['vame_latent_rqa_lmax'] = float(lmax)
    except Exception as e:
        logger.warning(f"Latent RQA failed: {e}")
        metrics.update({'vame_latent_rqa_rec': np.nan, 'vame_latent_rqa_det': np.nan, 'vame_latent_rqa_lmax': np.nan})

    # ---------------------------------------------------------------- #
    # Group 3: Variance / Compactness                                   #
    # ---------------------------------------------------------------- #
    try:
        cov = np.cov(latent_vectors.T)    # (D, D)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]

        cov_trace = float(np.trace(cov))

        # Participation Ratio: how many dimensions are "active"
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum() if len(eigenvalues) > 0 else np.nan

        pca = PCA(n_components=3)
        pca.fit(latent_vectors)
        pca1_var = float(pca.explained_variance_ratio_[0])
        pca3_var = float(pca.explained_variance_ratio_[:3].sum())

        metrics['vame_latent_cov_trace'] = cov_trace
        metrics['vame_latent_participation_ratio'] = float(pr)
        metrics['vame_latent_pca1_var'] = pca1_var
        metrics['vame_latent_pca3_var'] = pca3_var
    except Exception as e:
        logger.warning(f"Latent Compactness failed: {e}")
        metrics.update({'vame_latent_cov_trace': np.nan, 'vame_latent_participation_ratio': np.nan,
                        'vame_latent_pca1_var': np.nan, 'vame_latent_pca3_var': np.nan})

    # ---------------------------------------------------------------- #
    # Group 4: Temporal Autocorrelation & AR Predictability             #
    # ---------------------------------------------------------------- #
    try:
        lv = latent_vectors  # Use full resolution for autocorr
        # Mean autocorr at lag 1 across all dims
        mu = lv.mean(axis=0)
        lv_c = lv - mu
        var = (lv_c ** 2).mean(axis=0)
        autocorr_lag1 = ((lv_c[:-1] * lv_c[1:]).mean(axis=0)) / np.where(var > 0, var, 1)
        mean_autocorr_lag1 = float(np.mean(autocorr_lag1))

        # AR(1) predictability: R^2 of predicting z_t from z_{t-1}
        y = lv[1:]     # targets
        x = lv[:-1]    # predictors (AR shift)
        beta = np.sum(x * y, axis=0) / np.where(np.sum(x ** 2, axis=0) > 0, np.sum(x ** 2, axis=0), 1)
        y_hat = x * beta
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean(axis=0)) ** 2).sum()
        ar1_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

        metrics['vame_latent_autocorr_lag1'] = mean_autocorr_lag1
        metrics['vame_latent_ar1_r2'] = ar1_r2
    except Exception as e:
        logger.warning(f"Latent Autocorr failed: {e}")
        metrics.update({'vame_latent_autocorr_lag1': np.nan, 'vame_latent_ar1_r2': np.nan})

    # ---------------------------------------------------------------- #
    # Group 5: Latent Speed & Path Tortuosity                          #
    # ---------------------------------------------------------------- #
    try:
        diff_lv = np.diff(latent_vectors, axis=0)
        latent_vel = np.sqrt((diff_lv ** 2).sum(axis=1))    # per-frame speed

        vel_mean = float(latent_vel.mean())
        vel_var = float(latent_vel.var())

        # Tortuosity: total path / net displacement over 5-second windows
        win = fps * 5
        tortuosity_vals = []
        for start in range(0, T - win, win):
            seg = latent_vectors[start:start + win]
            total_len = latent_vel[start:start + win - 1].sum()
            displacement = float(np.linalg.norm(seg[-1] - seg[0]))
            if displacement > 1e-6:
                tortuosity_vals.append(total_len / displacement)
        latent_tortuosity = float(np.median(tortuosity_vals)) if tortuosity_vals else np.nan

        # Approximate Lyapunov exponent via divergence of nearby trajectories
        # Use a finite-time method on the downsampled trajectory
        nn_divergence = []
        Zl = Z  # downsampled
        for i in range(0, len(Zl) - 10, 50):
            dists = np.sqrt(((Zl - Zl[i]) ** 2).sum(axis=1))
            dists[i] = np.inf
            j = np.argmin(dists)
            if j + 10 < len(Zl) and i + 10 < len(Zl):
                d0 = dists[j] + 1e-12
                d1 = float(np.linalg.norm(Zl[i + 10] - Zl[j + 10])) + 1e-12
                nn_divergence.append(np.log(d1 / d0) / 10)
        lyapunov = float(np.mean(nn_divergence)) if nn_divergence else np.nan

        metrics['vame_latent_velocity_mean'] = vel_mean
        metrics['vame_latent_velocity_var'] = vel_var
        metrics['vame_latent_tortuosity'] = latent_tortuosity
        metrics['vame_latent_lyapunov'] = lyapunov
    except Exception as e:
        logger.warning(f"Latent Speed/Tortuosity failed: {e}")
        metrics.update({'vame_latent_velocity_mean': np.nan, 'vame_latent_velocity_var': np.nan,
                        'vame_latent_tortuosity': np.nan, 'vame_latent_lyapunov': np.nan})

    # ---------------------------------------------------------------- #
    # Group 6: Intra-session Similarity (half-vs-half DTW approx)      #
    # ---------------------------------------------------------------- #
    try:
        # Split session into first and second half, then subsample further for DTW
        dtw_cap = 500
        mid = len(Z) // 2
        Z1 = Z[:mid][::max(1, mid // dtw_cap)]
        Z2 = Z[mid:][::max(1, mid // dtw_cap)]

        # Approximate DTW using cumulative distance matrix
        N1, N2 = len(Z1), len(Z2)
        D_mat = np.sqrt(((Z1[:, None, :] - Z2[None, :, :]) ** 2).sum(axis=2))

        # Standard DTW DP
        dtw_dp = np.full((N1, N2), np.inf)
        dtw_dp[0, 0] = D_mat[0, 0]
        for i in range(1, N1): dtw_dp[i, 0] = dtw_dp[i-1, 0] + D_mat[i, 0]
        for j in range(1, N2): dtw_dp[0, j] = dtw_dp[0, j-1] + D_mat[0, j]
        for i in range(1, N1):
            for j in range(1, N2):
                dtw_dp[i, j] = D_mat[i, j] + min(dtw_dp[i-1, j], dtw_dp[i, j-1], dtw_dp[i-1, j-1])
        dtw_dist = float(dtw_dp[-1, -1]) / (N1 + N2)   # normalize by path length

        metrics['vame_latent_half_session_dtw'] = dtw_dist
    except Exception as e:
        logger.warning(f"Latent DTW failed: {e}")
        metrics['vame_latent_half_session_dtw'] = np.nan

    return metrics

def _compute_vame_metrics(paths):
    """Calculates VAME motif entropy and transition usage."""
    if not paths.vame_dir or not paths.vame_dir.exists():
        return {}, None
        
    try:
        vame_loader = VAMEDataLoader(paths.base_path)
        vame_data = vame_loader.load(paths.vame_dir, n_clusters=15)
        if vame_data is None or 'labels' not in vame_data:
            return {}, None
            
        labels = vame_data['labels']
        if len(labels) == 0: return {}, None
        
        # 1. Static usage entropy
        unique_states, counts = np.unique(labels, return_counts=True)
        probs = counts / np.sum(counts)
        entropy = scipy.stats.entropy(probs)
        
        # 2. Transition probability (overall rate of change)
        transitions = np.sum(np.diff(labels) != 0)
        p_transition = transitions / max(1, len(labels) - 1)
        
        # 3. Transition Matrix & Stereotypy Metrics
        n_states = 15
        mat = _get_vame_transition_matrix(labels, n_states=n_states)
                
        row_sums = mat.sum(axis=1)
        total_trans = row_sums.sum()
        
        # Transition probabilities P(next | current)
        P = np.divide(mat, row_sums[:, None], out=np.zeros_like(mat), where=row_sums[:, None] != 0)
        
        # 1. Transition Pattern Analysis
        pi = row_sums / total_trans if total_trans > 0 else np.zeros(n_states)
        cond_entropy = 0
        motif_entropies = np.zeros(n_states)
        for i in range(n_states):
            h_i = scipy.stats.entropy(P[i, :])
            motif_entropies[i] = h_i
            if pi[i] > 0:
                cond_entropy += pi[i] * h_i
        
        # Predictability (Mutual Information)
        next_state_counts = mat.sum(axis=0)
        p_next = next_state_counts / total_trans if total_trans > 0 else np.zeros(n_states)
        h_next = scipy.stats.entropy(p_next) if total_trans > 0 else 0
        mi = h_next - cond_entropy
        norm_mi = mi / h_next if h_next > 0 else 0
        
        # 2. Duration Pattern Analysis (Stability)
        bout_starts = np.where(np.diff(labels) != 0)[0] + 1
        runs = np.split(labels, bout_starts)
        bout_data = [(r[0], len(r)) for r in runs if len(r) > 0]
        
        motif_stabilities = np.zeros(n_states)
        for i in range(n_states):
            durs = np.array([d for m, d in bout_data if m == i])
            if len(durs) > 1:
                mu = np.mean(durs)
                sigma = np.std(durs)
                # Stability = 1 / (1 + CV) where CV = sigma / mu
                stability = 1.0 / (1.0 + (sigma / mu)) if mu > 0 else 0
                motif_stabilities[i] = stability
            elif len(durs) == 1:
                motif_stabilities[i] = 0.5 # Default for single bout 
        
        # 3. Combined Stereotypy Index (CSI)
        # Weighted mean of stabilities by motif usage
        usage = counts / np.sum(counts)
        avg_stability = np.sum(usage * motif_stabilities)
        
        csi = norm_mi * avg_stability
        
        max_trans_prob = np.max(P) if total_trans > 0 else 0
        
        result_metrics = {
            'vame_motif_entropy': entropy,
            'vame_transition_prob': p_transition,
            'vame_sequence_cond_entropy': cond_entropy,
            'vame_sequence_mutual_info': mi,
            'vame_max_transition_prob': max_trans_prob,
            'vame_combined_stereotypy_index': csi,
            'vame_motif_entropies': motif_entropies,
            'vame_motif_stabilities': motif_stabilities
        }
        
        # Continuous latent space metrics (Groups 2-6)
        latent_vectors = vame_data.get('latent_vectors', None)
        if latent_vectors is not None and len(latent_vectors) > 1:
            try:
                latent_met = _compute_latent_stereotypy(latent_vectors)
                result_metrics.update(latent_met)
                logger.info(f"Computed {len(latent_met)} latent stereotypy metrics.")
            except Exception as e:
                logger.warning(f"Latent stereotypy metrics failed: {e}")
        
        return result_metrics, labels
    except Exception as e:
        logger.error(f"Failed to compute VAME metrics: {e}")
        return {}, None


def _compute_spike_kinematic_metrics(spike_times, spike_clusters, unique_clusters, velocity, velocity_times):
    """Calculates speed-coding and locomotion modulation indices."""
    if len(velocity) == 0 or len(unique_clusters) == 0:
        return {}
        
    dt = np.median(np.diff(velocity_times))
    if dt <= 0: dt = 1/60.0
    bins = np.append(velocity_times, velocity_times[-1] + dt)
    
    move_mask = velocity > 2.0
    rest_mask = ~move_mask
    
    n_speed_coded, pos_mod, neg_mod = 0, 0, 0
    rates_rest, rates_move = [], []
    
    for cid in unique_clusters:
        spikes = spike_times[spike_clusters == cid]
        counts, _ = np.histogram(spikes, bins=bins)
        fr = counts / dt
        fr_smooth = gaussian_filter1d(fr, sigma=int(0.5/dt))
        
        if np.sum(move_mask) > 10:
            r, p = scipy.stats.pearsonr(fr_smooth[move_mask], velocity[move_mask])
            if r > 0.1 and p < 0.05: n_speed_coded += 1
                
        fr_move = np.mean(fr[move_mask]) if np.any(move_mask) else 0
        fr_rest = np.mean(fr[rest_mask]) if np.any(rest_mask) else 0
        rates_move.append(fr_move)
        rates_rest.append(fr_rest)
        
        try:
            chunk_size = int(2.0/dt)
            n_m = len(fr_smooth[move_mask]) // chunk_size
            n_r = len(fr_smooth[rest_mask]) // chunk_size
            if n_m > 5 and n_r > 5:
                # T-test chunks
                m_chunks = np.array_split(fr_smooth[move_mask][:n_m*chunk_size], n_m)
                r_chunks = np.array_split(fr_smooth[rest_mask][:n_r*chunk_size], n_r)
                m_means = [np.mean(c) for c in m_chunks]
                r_means = [np.mean(c) for c in r_chunks]
                t, p = scipy.stats.ttest_ind(m_means, r_means)
                if p < 0.05:
                    if np.mean(m_means) > np.mean(r_means): pos_mod += 1
                    else: neg_mod += 1
        except: pass
            
    n_u = len(unique_clusters)
    return {
        'prop_speed_coded': n_speed_coded / n_u if n_u else 0,
        'prop_pos_modulated': pos_mod / n_u if n_u else 0,
        'prop_neg_modulated': neg_mod / n_u if n_u else 0,
        'avg_fr_move': np.mean(rates_move) if rates_move else 0,
        'avg_fr_rest': np.mean(rates_rest) if rates_rest else 0
    }

def _compute_lfp_power_metrics(lfp_loader, velocity, velocity_times):
    """Calculates Beta/Gamma/Theta power ratio Move vs Rest."""
    if lfp_loader.extractor is None or len(velocity) == 0: return {}, None, None
    
    fs = lfp_loader.fs
    t_start, t_end = velocity_times[0], velocity_times[-1]
    
    channels = lfp_loader.channel_ids
    mid_channel = channels[len(channels)//2]
    traces, timestamps = lfp_loader.get_data(t_start, t_end, channels=[mid_channel])
    
    if len(traces) == 0: return {}, None, None
    trace = traces[:, 0]
    
    move_interp = np.interp(timestamps, velocity_times, (velocity > 2.0).astype(float))
    is_moving = move_interp > 0.5
    trace_move = trace[is_moving]
    trace_rest = trace[~is_moving]
    
    def band_power(data, fs):
        res = {'delta': np.nan, 'theta': np.nan, 'beta': np.nan, 'gamma_low': np.nan, 'gamma_high': np.nan}
        if len(data) < fs * 2: return res
        f, Pxx = welch(data, fs, nperseg=int(fs*2))
        bands = {'delta': (1, 4), 'theta': (4, 8), 'beta': (13, 30), 'gamma_low': (30, 50), 'gamma_high': (50, 90)}
        for b_name, (f_min, f_max) in bands.items():
            res[b_name] = np.sum(Pxx[(f >= f_min) & (f <= f_max)])
        return res
        
    p_move = band_power(trace_move, fs)
    p_rest = band_power(trace_rest, fs)
    
    metrics = {}
    for b in p_move:
        if not np.isnan(p_move[b]) and not np.isnan(p_rest[b]) and p_rest[b] > 0:
            metrics[f'ratio_{b}_power'] = p_move[b] / p_rest[b]
            metrics[f'abs_{b}_power_move'] = p_move[b]
        else:
            metrics[f'ratio_{b}_power'] = np.nan
            metrics[f'abs_{b}_power_move'] = np.nan
            
    metrics['theta_delta_move'] = p_move['theta'] / p_move['delta'] if not np.isnan(p_move['theta']) and p_move['delta'] > 0 else np.nan
    return metrics, traces, timestamps

def _compute_phase_locking_metrics(lfp_loader, spike_times, spike_clusters, unique_clusters):
    """Analyzes PLV to Beta and Theta bands."""
    if lfp_loader.extractor is None or len(unique_clusters) == 0: return {}
    
    fs = lfp_loader.fs
    channels = lfp_loader.channel_ids
    mid_channel = channels[len(channels)//2]
    
    t_start = max(0, spike_times[0] - 10) if len(spike_times) > 0 else 0
    t_end = spike_times[-1] + 10 if len(spike_times) > 0 else 600
    
    duration = t_end - t_start
    if duration > 600:
        t_center = t_start + duration/2
        t_start = t_center - 300
        t_end = t_center + 300
        
    traces, timestamps = lfp_loader.get_data(t_start, t_end, channels=[mid_channel])
    if len(traces) == 0: return {}
    trace = traces[:, 0]
    
    def filter_and_phase(data, fs, low, high):
        nyq = 0.5 * fs
        b, a = butter(3, [low/nyq, high/nyq], btype='bandpass')
        filtered = filtfilt(b, a, data)
        analytic = hilbert(filtered)
        return np.angle(analytic)
        
    phase_beta = filter_and_phase(trace, fs, 13, 30)
    phase_theta = filter_and_phase(trace, fs, 4, 8)
    
    beta_sig, theta_sig = 0, 0
    beta_plvs, theta_plvs = [], []
    
    mask_spikes = (spike_times >= timestamps[0]) & (spike_times <= timestamps[-1])
    st_sub, sc_sub = spike_times[mask_spikes], spike_clusters[mask_spikes]
    if len(st_sub) == 0: return {}

    idx = np.clip(np.searchsorted(timestamps, st_sub), 0, len(timestamps)-1)
    
    for cid in unique_clusters:
        c_mask = sc_sub == cid
        c_idx = idx[c_mask]
        
        if len(c_idx) > 20:
            # Beta
            p_b = phase_beta[c_idx]
            R_b = np.abs(np.sum(np.exp(1j * p_b))) / len(p_b)
            Z_b = len(p_b) * (R_b ** 2)
            pval_b = np.exp(-Z_b) * (1 + (2*Z_b - Z_b**2)/(4*len(p_b))) if len(p_b) < 1000 else np.exp(-Z_b)
            if pval_b < 0.05: beta_sig += 1
            beta_plvs.append(R_b)
            
            # Theta
            p_t = phase_theta[c_idx]
            R_t = np.abs(np.sum(np.exp(1j * p_t))) / len(p_t)
            Z_t = len(p_t) * (R_t ** 2)
            pval_t = np.exp(-Z_t) * (1 + (2*Z_t - Z_t**2)/(4*len(p_t))) if len(p_t) < 1000 else np.exp(-Z_t)
            if pval_t < 0.05: theta_sig += 1
            theta_plvs.append(R_t)
            
    n_valid = len(beta_plvs)
    if n_valid == 0: return {}
    return {
        'prop_beta_locked': beta_sig / n_valid,
        'prop_theta_locked': theta_sig / n_valid,
        'avg_beta_plv': np.mean(beta_plvs),
        'avg_theta_plv': np.mean(theta_plvs)
    }

def _compute_csd_metrics(lfp_loader, velocity, velocity_times):
    """Computes 1D Current Source Density (CSD) to model electric field sources/sinks."""
    if lfp_loader.extractor is None or len(velocity) == 0: return {}, None, None
    
    try:
        locs = lfp_loader.extractor.get_channel_locations()
        channel_ids = lfp_loader.extractor.get_channel_ids()
        
        unique_x = np.unique(locs[:, 0])
        target_x = unique_x[len(unique_x)//2]
        
        mask = locs[:, 0] == target_x
        col_channels = channel_ids[mask]
        col_locs_y = locs[mask, 1]
        
        sort_idx = np.argsort(col_locs_y)
        sorted_channels = col_channels[sort_idx]
        sorted_y = col_locs_y[sort_idx]
        
        t_start = velocity_times[0]
        t_end = min(t_start + 60, velocity_times[-1]) # 60s sample
        
        traces, timestamps = lfp_loader.get_data(t_start, t_end, channels=list(sorted_channels))
        if len(traces) == 0: return {}, None, None
        
        lfp_matrix = traces.T
        lfp_smooth = gaussian_filter1d(lfp_matrix, sigma=1.0, axis=0)
        
        dy = np.mean(np.diff(sorted_y))
        if dy <= 0: dy = 20.0 
        
        csd = -np.diff(lfp_smooth, n=2, axis=0) / (dy**2)
        
        move_interp = np.interp(timestamps, velocity_times, (velocity > 2.0).astype(float))
        is_moving = move_interp > 0.5
        
        csd_move = csd[:, is_moving]
        csd_rest = csd[:, ~is_moving]
        
        avg_sink_move = np.mean(np.min(csd_move, axis=0)) if csd_move.size > 0 else np.nan
        avg_sink_rest = np.mean(np.min(csd_rest, axis=0)) if csd_rest.size > 0 else np.nan
        sink_ratio = abs(avg_sink_move / avg_sink_rest) if not np.isnan(avg_sink_rest) and avg_sink_rest != 0 else np.nan
        
        sink_depths_move = sorted_y[1:-1][np.argmin(csd_move, axis=0)] if csd_move.size > 0 else []
        sink_depths_rest = sorted_y[1:-1][np.argmin(csd_rest, axis=0)] if csd_rest.size > 0 else []
        depth_shift = np.mean(sink_depths_move) - np.mean(sink_depths_rest) if len(sink_depths_move) > 0 and len(sink_depths_rest) > 0 else np.nan
        
        return {
            'csd_sink_strength_ratio_move_rest': sink_ratio,
            'csd_avg_sink_uv_mm2': abs(avg_sink_move),
            'csd_sink_depth_shift_um': depth_shift
        }, csd, sorted_y
    except Exception as e:
        logger.error(f"Failed to compute CSD metrics: {e}")
        return {}, None, None

def _compute_dopamine_metrics(tdt_data, velocity, velocity_times):
    """Correlates DA to speed and computes transient rates and amplitudes."""
    if not tdt_data or len(velocity) == 0: return {}
    
    dff = tdt_data['dff_values']
    ts = tdt_data['dff_timestamps']
    if len(dff) == 0: return {}
    
    v_interp = np.interp(ts, velocity_times, velocity)
    is_moving = v_interp > 2.0
    r_val, p = scipy.stats.pearsonr(dff, v_interp)
    
    std_dff = np.nanstd(dff)
    mean_dff = np.nanmean(dff)
    fs_dt = np.median(np.diff(ts))
    dist_samples = max(1, int(1.0 / fs_dt)) if fs_dt > 0 else 100
    
    peaks, _ = find_peaks(dff, height=mean_dff + 2*std_dff, distance=dist_samples)
    da_amplitudes = dff[peaks]
    avg_peak_amp = np.mean(da_amplitudes) if len(da_amplitudes) > 0 else np.nan
    
    time_m_min = np.sum(is_moving) * fs_dt / 60.0
    time_r_min = np.sum(~is_moving) * fs_dt / 60.0
    
    da_rate_m = np.sum(is_moving[peaks]) / time_m_min if time_m_min > 0 else np.nan
    da_rate_r = np.sum(~is_moving[peaks]) / time_r_min if time_r_min > 0 else np.nan
    
    return {
        'da_speed_corr': r_val if p < 0.05 else 0,
        'da_transient_rate_move': da_rate_m,
        'da_transient_rate_rest': da_rate_r,
        'da_avg_peak_amplitude_dff': avg_peak_amp
    }

def _visualize_openfield_dashboard(paths: DataPaths, data_pack: dict, metrics: dict):
    """Creates a 6-panel functional summary dashboard and saves to disk."""
    logger.info("Generating 9-Panel Visualization Dashboard...")
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    fig.suptitle(f"Openfield Functional Dashboard: Mouse {paths.mouse_id} | Date {paths.date_str}", fontsize=18, fontweight='bold')
    vel = data_pack.get('velocity', np.array([]))
    vel_ts = data_pack.get('velocity_times', np.array([]))
    df_dlc = data_pack.get('df_dlc', None)
    
    # Panel 1: Kinematic Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    t_offset = 10.0
    if df_dlc is not None and not df_dlc.empty and len(vel) > 0:
        try:
            # DLC MultiIndex: level 0 = scorer, level 1 = bodypart, level 2 = coords
            mask = vel_ts >= (vel_ts[0] + t_offset)
            x_vals = df_dlc[df_dlc.columns[df_dlc.columns.get_level_values(2) == 'x'][0]].values
            y_vals = df_dlc[df_dlc.columns[df_dlc.columns.get_level_values(2) == 'y'][0]].values
            
            # Slice to match velocity if needed
            if len(x_vals) > len(vel_ts):
                x_vals = x_vals[:len(vel_ts)]
                y_vals = y_vals[:len(vel_ts)]
                
            x = x_vals[mask]
            y = y_vals[mask]
            v_plot = vel[mask]
            
            sc = ax1.scatter(x, y, c=v_plot, cmap='magma', s=2, alpha=0.5)
            ax1.set_title(f"DLC Spatial Trajectory (After {t_offset}s)")
            ax1.invert_yaxis()
            fig.colorbar(sc, ax=ax1, label='Speed (cm/s)')
            ax1.axis('equal')
        except Exception as e:
             ax1.set_title(f"DLC Trajectory (Unavailable: {e})")
    else:
        ax1.set_title("Trajectory (No Data)")
        
    # Panel 2: Dopamine vs Speed
    ax2 = fig.add_subplot(gs[0, 1])
    tdt_data = data_pack.get('tdt_data', None)
    if tdt_data and len(vel) > 0:
        dff = tdt_data['dff_values']
        dff_ts = tdt_data['dff_timestamps']
        
        start_t = dff_ts[0] + t_offset
        start_idx_dff = np.searchsorted(dff_ts, start_t)
        end_idx_dff = min(len(dff_ts), np.searchsorted(dff_ts, start_t + 300)) # 5 mins
        
        start_idx_vel = np.searchsorted(vel_ts, start_t)
        end_idx_vel = min(len(vel_ts), np.searchsorted(vel_ts, start_t + 300))
        
        ax2.plot(dff_ts[start_idx_dff:end_idx_dff] - start_t, dff[start_idx_dff:end_idx_dff], color='#2ecc71', label='DA (dFF)', lw=1.5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(vel_ts[start_idx_vel:end_idx_vel] - start_t, vel[start_idx_vel:end_idx_vel], color='gray', alpha=0.4, label='Speed', lw=1)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('dFF', color='#2ecc71')
        ax2_twin.set_ylabel('Speed (cm/s)', color='gray')
        ax2.set_title("Dopamine transients vs Kinematics (First 5m)")
    else:
        ax2.set_title("Dopamine & Speed (Unavailable)")
        
    # Panel 3: LFP Power Spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    lfp_traces = data_pack.get('lfp_traces', np.array([]))
    lfp_ts = data_pack.get('lfp_timestamps', np.array([]))
    if lfp_traces is not None and len(lfp_traces) > 0 and len(vel) > 0:
        fs = 2500 # Default NPX LFP fs
        trace = lfp_traces[:, 0]
        move_interp = np.interp(lfp_ts, vel_ts, (vel > 2.0).astype(float))
        is_moving = move_interp > 0.5
        t_m = trace[is_moving]
        t_r = trace[~is_moving]
        
        if len(t_m) > fs*2 and len(t_r) > fs*2:
            f, pm = welch(t_m, fs, nperseg=int(fs*2))
            _, pr = welch(t_r, fs, nperseg=int(fs*2))
            ax3.semilogy(f, pm, label='Move', color='#e74c3c')
            ax3.semilogy(f, pr, label='Rest', color='#3498db')
            ax3.set_xlim(1, 100)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power')
            ax3.legend()
            ax3.set_title("LFP Absolute Power Spectrum")
    else:
        ax3.set_title("LFP Power Spectrum (Unavailable)")
        
    # Panel 4: CSD Depth
    ax4 = fig.add_subplot(gs[1, 0])
    csd_matrix = data_pack.get('csd_matrix', np.array([]))
    csd_y = data_pack.get('csd_y', np.array([]))
    if csd_matrix is not None and csd_y is not None and csd_matrix.size > 0:
        fs = 2500
        start_idx_csd = int(t_offset * fs)
        max_t = min(csd_matrix.shape[1], start_idx_csd + int(5 * fs)) # 5 SECONDS
        chunk = csd_matrix[:, start_idx_csd:max_t]
        vmax = np.max(np.abs(chunk)) * 0.5
        im = ax4.imshow(chunk, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax,
                        extent=[0, (max_t-start_idx_csd)/fs, csd_y[-1], csd_y[0]])
        ax4.set_title("1D Current Source Density (5 sec)")
        ax4.set_ylabel("Depth (um)")
        ax4.set_xlabel("Time (s)")
        fig.colorbar(im, ax=ax4, label=r"CSD $\mu V/mm^2$")
    else:
        ax4.set_title("CSD Heatmap (Unavailable)")
        
    # Panel 5: Spiking Raster
    ax5 = fig.add_subplot(gs[1, 1])
    spike_times = data_pack.get('spike_times', np.array([]))
    spike_clusters = data_pack.get('spike_clusters', np.array([]))
    unique_clusters = data_pack.get('unique_clusters', np.array([]))
    if len(spike_times) > 0 and len(vel) > 0 and len(unique_clusters) > 0:
        win_start = vel_ts[0] + t_offset
        win_end = win_start + 30
        subset = unique_clusters[:min(50, len(unique_clusters))]
        
        for i, cid in enumerate(subset):
            mask = (spike_clusters == cid) & (spike_times >= win_start) & (spike_times <= win_end)
            st = spike_times[mask] - win_start
            ax5.vlines(st, i-0.4, i+0.4, color='black', lw=0.6)
            
        ax5.set_xlim(0, 30)
        ax5.set_ylim(-1, len(subset))
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Unit #")
        
        ax5_twin = ax5.twinx()
        v_mask = (vel_ts >= win_start) & (vel_ts <= win_end)
        ax5_twin.plot(vel_ts[v_mask] - win_start, vel[v_mask], color='#e74c3c', alpha=0.5, label='Speed')
        ax5_twin.set_ylabel("Speed (cm/s)", color='#e74c3c')
        ax5.set_title("Locomotion-Aligned Raster (30s)")
    else:
        ax5.set_title("Spiking Raster (Unavailable)")
        
    # Panel 6: VAME Usage Histogram
    ax6 = fig.add_subplot(gs[1, 2])
    vame_labels = data_pack.get('vame_labels', None)
    if vame_labels is not None and len(vame_labels) > 0:
        sns.histplot(vame_labels, discrete=True, stat="proportion", ax=ax6, color='#9b59b6')
        entropy = metrics.get('vame_motif_entropy', 0)
        ax6.set_title(f"VAME Motif Usage (Entropy: {entropy:.2f})")
        ax6.set_xlabel("Latent State Motif ID")
        ax6.set_ylabel("Fractional Usage")
    else:
        ax6.set_title("VAME States (Unavailable)")
        
    # Panel 7: Population Spiking Summary
    ax7 = fig.add_subplot(gs[2, 0])
    if 'avg_fr_move' in metrics and not np.isnan(metrics.get('avg_fr_move', np.nan)):
        fr_m = metrics['avg_fr_move']
        fr_r = metrics['avg_fr_rest']
        
        ax7.bar(['Rest', 'Move'], [fr_r, fr_m], color=['#3498db', '#e74c3c'], alpha=0.8)
        ax7.set_ylabel("Avg Firing Rate (Hz)")
        ax7.set_title("Population Mean Firing Rate")
        
        pos = metrics.get('prop_pos_modulated', 0)
        neg = metrics.get('prop_neg_modulated', 0)
        none = max(0, 1 - (pos + neg))
        axin = ax7.inset_axes([0.65, 0.65, 0.35, 0.35])
        axin.pie([pos, neg, none], colors=['#e74c3c', '#3498db', '#95a5a6'], 
                 labels=['+','-','o'], textprops={'fontsize': 8})
        axin.set_title("Modulated", fontsize=8)
    else:
        ax7.set_title("Population Spiking (Unavailable)")
        
    # Panel 8: Phase Locking
    ax8 = fig.add_subplot(gs[2, 1])
    if 'prop_theta_locked' in metrics and not np.isnan(metrics.get('prop_theta_locked', np.nan)):
        t_lock = metrics.get('prop_theta_locked', 0)
        b_lock = metrics.get('prop_beta_locked', 0)
        
        ax8.bar(['Theta (4-8Hz)', 'Beta (13-30Hz)'], [t_lock*100, b_lock*100], color=['#f1c40f', '#e67e22'])
        ax8.set_ylabel("% Neurons Locked")
        ax8.set_title("LFP Phase Locking (Rayleigh Z)")
        if max(t_lock, b_lock) > 0:
             ax8.set_ylim(0, max(100, (max(t_lock, b_lock)*100)+10))
    else:
        ax8.set_title("Phase Locking (Unavailable)")
        
    # Panel 9: Dopamine Transient Summary
    ax9 = fig.add_subplot(gs[2, 2])
    if 'da_transient_rate_move' in metrics and not np.isnan(metrics.get('da_transient_rate_move', np.nan)):
        r_m = metrics.get('da_transient_rate_move', 0)
        r_r = metrics.get('da_transient_rate_rest', 0)
        
        ax9.bar(['Rest', 'Move'], [r_r, r_m], color=['gray', '#2ecc71'])
        ax9.set_ylabel("DA Transients / Min", color='#2ecc71')
        ax9.set_title("Dopamine Transient Rates")
        
        amp = metrics.get('da_avg_peak_amplitude_dff', 0)
        ax9_twin = ax9.twinx()
        ax9_twin.axhline(amp, color='black', linestyle='--', label=f'Avg Peak: {amp:.2f}')
        ax9_twin.legend(loc='upper right', fontsize=8)
        ax9_twin.set_yticks([])
    else:
        ax9.set_title("Dopamine Summaries (Unavailable)")
        
    plt.tight_layout()
    
    out_dir = paths.base_path / "post_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{paths.mouse_id}_{paths.date_str}_openfield_dashboard.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"==> Visual Dashboard exported to {out_file}")

def extract_openfield_metrics(paths: DataPaths):
    """Master function: Computes scalar metrics, visualizes dashboard, and appends to CSV."""
    logger.info("Extracting Comprehensive Summary Metrics...")
    
    metrics = {
        'mouse': paths.mouse_id,
        'date': paths.date_str,
        'genotype': paths.genotype,
    }
    data_pack = {} # Store raw traces for visualization
    
    # 0. VAME Behavioral States
    v_met, v_labels = _compute_vame_metrics(paths)
    metrics.update(v_met)
    if v_labels is not None: data_pack['vame_labels'] = v_labels
    
    # 1. Kinematics
    velocity, velocity_times = np.array([]), np.array([])
    try:
        dlc_loader = DLCDataLoader(paths.base_path)
        if paths.dlc_h5 and paths.dlc_h5.exists():
            df_dlc = dlc_loader.load(paths.dlc_h5)
            velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, video_fs=60, px_per_cm=30.0, strobe_path=paths.strobe_seconds)
            k_met = _compute_kinematic_metrics(velocity, velocity_times)
            metrics.update(k_met)
            data_pack['velocity'] = velocity
            data_pack['velocity_times'] = velocity_times
            data_pack['df_dlc'] = df_dlc
            logger.info("Kinematics metrics completed.")
        else:
            logger.warning("No DLC. Skipping Kinematics.")
    except Exception as e:
        logger.error(f"Kinematics loop failed: {e}")
        
    # 2. Spikes
    try:
        spike_data = None
        if paths.kilosort_dir and paths.kilosort_dir.exists():
            spike_loader = SpikeDataLoader(paths.base_path)
            spike_data = spike_loader.load(paths.kilosort_dir)
            s_met = _compute_spike_kinematic_metrics(
                spike_data['spike_times_sec'], spike_data['spike_clusters'], 
                spike_data['unique_clusters'], velocity, velocity_times)
            metrics.update(s_met)
            data_pack['spike_times'] = spike_data['spike_times_sec']
            data_pack['spike_clusters'] = spike_data['spike_clusters']
            data_pack['unique_clusters'] = spike_data['unique_clusters']
            logger.info("Spike metrics completed.")
    except Exception as e:
         logger.error(f"Spike loop failed: {e}")
         
    # 3. LFP and CSD
    try:
        if paths.lfp_dir and paths.lfp_dir.exists():
            lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
            lfp_met, lfp_traces, lfp_ts = _compute_lfp_power_metrics(lfp_loader, velocity, velocity_times)
            metrics.update(lfp_met)
            if lfp_traces is not None:
                 data_pack['lfp_traces'] = lfp_traces
                 data_pack['lfp_timestamps'] = lfp_ts
            logger.info("LFP metrics completed.")
            
            if spike_data is not None:
                pl_met = _compute_phase_locking_metrics(
                    lfp_loader, spike_data['spike_times_sec'], 
                    spike_data['spike_clusters'], spike_data['unique_clusters'])
                metrics.update(pl_met)
                logger.info("Phase locking metrics completed.")
            
            csd_met, csd_mat, csd_y = _compute_csd_metrics(lfp_loader, velocity, velocity_times)
            metrics.update(csd_met)
            if csd_mat is not None:
                 data_pack['csd_matrix'] = csd_mat
                 data_pack['csd_y'] = csd_y
            logger.info("CSD metrics completed.")
    except Exception as e:
         logger.error(f"LFP / Phase locking / CSD loop failed: {e}")
            
    # 4. Dopamine
    try:
        if paths.tdt_dff and paths.tdt_dff.exists():
            tdt_loader = PhotometryDataLoader(paths.base_path)
            tdt_data = tdt_loader.load(paths.tdt_dff, paths.tdt_raw)
            da_met = _compute_dopamine_metrics(tdt_data, velocity, velocity_times)
            metrics.update(da_met)
            data_pack['tdt_data'] = tdt_data
            logger.info("Dopamine metrics completed.")
    except Exception as e:
         logger.error(f"Dopamine loop failed: {e}")

    # Generate Image Dashboard
    _visualize_openfield_dashboard(paths, data_pack, metrics)
        
    # Finalize & Save in Long Format
    csv_file = paths.base_path / "openfield_summary_metrics.csv"
    file_exists = csv_file.exists()
    try:
        rows = []
        metadata = {'mouse': metrics.pop('mouse'), 'date': metrics.pop('date')}
        for m_name, m_val in metrics.items():
            row = metadata.copy()
            row['metric'] = m_name
            row['value'] = m_val
            rows.append(row)
            
        df_new = pd.DataFrame(rows)[['mouse', 'date', 'metric', 'value']]
        df_new.to_csv(csv_file, mode='a', header=not file_exists, index=False)
        logger.info(f"==> Appended {len(df_new)} metric rows to {csv_file}")
    except Exception as e:
        logger.error(f"Failed to save metrics CSV: {e}")

def _compute_rqa(labels, min_line=2):
    if len(labels) < min_line: return {'rqa_determinism': np.nan, 'rqa_laminarity': np.nan, 'rqa_trapping_time': np.nan}
    
    idx = np.linspace(0, len(labels)-1, min(2000, len(labels)), dtype=int)
    L = labels[idx]
    N = len(L)
    
    R = (L[:, None] == L[None, :]).astype(int)
    np.fill_diagonal(R, 0)
    
    diag_lines = []
    vert_lines = []
    
    for i in range(1, N):
        diag = np.diagonal(R, offset=i)
        if len(diag) > 0:
            runs = np.split(diag, np.where(np.diff(diag) != 0)[0] + 1)
            for r in runs:
                if len(r) >= min_line and r[0] == 1:
                    diag_lines.append(len(r))
                    
    for j in range(N):
        col = R[:, j]
        runs = np.split(col, np.where(np.diff(col) != 0)[0] + 1)
        for r in runs:
            if len(r) >= min_line and r[0] == 1:
                vert_lines.append(len(r))
                
    recurrence_points = np.sum(R)
    
    det = np.sum(diag_lines) / recurrence_points if recurrence_points > 0 else 0
    lam = np.sum(vert_lines) / recurrence_points if recurrence_points > 0 else 0
    tt = np.mean(vert_lines) if len(vert_lines) > 0 else 0
    
    return {'rqa_determinism': det, 'rqa_laminarity': lam, 'rqa_trapping_time': tt}

def _compute_tier1_microstructure(velocity, dt, x, y):
    metrics = {}
    if len(velocity) < 3: return metrics, ([], [], [])
    
    is_moving = velocity > 2.0
    onsets = np.where(np.diff(is_moving.astype(int)) == 1)[0] + 1
    offsets = np.where(np.diff(is_moving.astype(int)) == -1)[0]
    if len(onsets) == 0 or len(offsets) == 0: return metrics, ([], [], [])
    if onsets[0] > offsets[0]: offsets = offsets[1:]
    valid_bouts = min(len(onsets), len(offsets))
    
    peak_accels = []
    peak_decels = []
    bout_angles = []
    
    accel = np.gradient(velocity) / dt
    
    for i in range(valid_bouts):
        start, end = onsets[i], offsets[i]
        if end - start > 5:
            mid = start + (end-start)//2
            peak_accels.append(np.max(accel[start:mid]))
            peak_decels.append(np.min(accel[mid:end]))
            
            dx = np.diff(x[start:end])
            dy = np.diff(y[start:end])
            angles = np.arctan2(dy, dx)
            turn_angles = np.diff(np.unwrap(angles))
            bout_angles.extend(np.abs(turn_angles))
            
    metrics['bout_peak_accel_mean'] = np.mean(peak_accels) if peak_accels else np.nan
    metrics['bout_peak_decel_mean'] = np.mean(peak_decels) if peak_decels else np.nan
    metrics['turn_angle_mean_rad_per_frame'] = np.mean(bout_angles) if bout_angles else np.nan
    
    return metrics, (peak_accels, peak_decels, bout_angles)
    
def _compute_tier1_spatial_extended(x, y, valid_mask, dt):
    metrics = {}
    if np.sum(valid_mask) == 0: return metrics, (np.nan, np.nan)
    
    x_v = x[valid_mask]
    y_v = y[valid_mask]
    
    x_min, x_max = np.nanpercentile(x_v, 1), np.nanpercentile(x_v, 99)
    y_min, y_max = np.nanpercentile(y_v, 1), np.nanpercentile(y_v, 99)
    
    cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
    w, h = (x_max - x_min), (y_max - y_min)
    if w > 0 and h > 0:
        center_w, center_h = w * 0.5, h * 0.5
        is_center = (np.abs(x_v - cx) < center_w/2) & (np.abs(y_v - cy) < center_h/2)
        metrics['time_in_center_perc'] = np.mean(is_center) * 100
        
    n_bins = 20
    hist, x_edges, y_edges = np.histogram2d(x_v, y_v, bins=n_bins, range=[[x_min, x_max], [y_min, y_max]])
    p = hist / np.sum(hist)
    p_nz = p[p > 0]
    metrics['spatial_entropy'] = scipy.stats.entropy(p_nz) if len(p_nz) > 0 else np.nan
    
    home_bin = np.unravel_index(np.argmax(hist), hist.shape)
    hb_x = (x_edges[home_bin[0]] + x_edges[home_bin[0]+1]) / 2
    hb_y = (y_edges[home_bin[1]] + y_edges[home_bin[1]+1]) / 2
    
    hb_dist = np.sqrt((x_v - hb_x)**2 + (y_v - hb_y)**2)
    hb_radius_leave = max(5.0, w*0.1)
    hb_radius_return = max(2.5, w*0.05)
    
    at_home = True
    revisits = 0
    for d in hb_dist:
        if at_home and d > hb_radius_leave:
            at_home = False
        elif not at_home and d < hb_radius_return:
            at_home = True
            revisits += 1
    metrics['homebase_revisits'] = revisits
    
    mid_idx = min(len(x_v)//2, int((5*60)/dt) if dt > 0 else len(x_v)//2)
    first_half = (x_v[:mid_idx], y_v[:mid_idx])
    second_half = (x_v[mid_idx:], y_v[mid_idx:])
    
    a1 = np.sum(np.histogram2d(first_half[0], first_half[1], bins=n_bins, range=[[x_min, x_max], [y_min, y_max]])[0] > 0)
    a2 = np.sum(np.histogram2d(second_half[0], second_half[1], bins=n_bins, range=[[x_min, x_max], [y_min, y_max]])[0] > 0)
    
    metrics['cumulative_area_ratio_first_last_half'] = a1 / a2 if a2 > 0 else np.nan
    
    step_size = int(1.0 / dt) if dt > 0 else 60
    tortuosities = []
    for i in range(0, len(x_v) - step_size, step_size):
        chunk_x = x_v[i:i+step_size]
        chunk_y = y_v[i:i+step_size]
        path_len = np.sum(np.sqrt(np.diff(chunk_x)**2 + np.diff(chunk_y)**2))
        straight_dist = np.sqrt((chunk_x[-1] - chunk_x[0])**2 + (chunk_y[-1] - chunk_y[0])**2 + 1e-10)
        if straight_dist > 5:
            tortuosities.append(path_len / straight_dist)
            
    metrics['path_tortuosity_mean'] = np.mean(tortuosities) if tortuosities else np.nan
    
    return metrics, (hb_x, hb_y)

def _compute_tier1_vame_extended(labels):
    metrics = {}
    if len(labels) < 2: return metrics, []
    
    # Identify starts of new bouts
    bout_starts = np.where(np.diff(labels) != 0)[0] + 1
    # Split into sequences of identical motifs
    runs = np.split(labels, bout_starts)
    
    # Store as (motif_id, duration)
    bout_data = [(r[0], len(r)) for r in runs if len(r) > 0]
    bout_durs = [d for m, d in bout_data]
    
    metrics['vame_avg_bout_duration_frames'] = np.mean(bout_durs)
    metrics['vame_max_bout_duration_frames'] = np.max(bout_durs)
    
    unique, counts = np.unique(labels, return_counts=True)
    counts.sort()
    top3_usage = np.sum(counts[-3:]) / len(labels) if len(counts) >= 3 else 1.0
    metrics['vame_repetitive_behavior_index'] = top3_usage
    
    lag = min(30, len(labels)-1)
    metrics['vame_autocorr_lag30_prob'] = np.mean(labels[:-lag] == labels[lag:])
    
    rqa = _compute_rqa(labels)
    metrics.update(rqa)
        
    return metrics, bout_data

def _compute_tier1_kinematics(velocity, dt):
    metrics = {}
    if len(velocity) < 2: return metrics
    
    metrics['peak_speed_cm_s'] = np.nanpercentile(velocity, 99)
    accel = np.diff(velocity) / dt
    metrics['mean_abs_acceleration'] = np.nanmean(np.abs(accel))
    metrics['peak_acceleration'] = np.nanpercentile(accel, 99)
    
    return metrics

def analyze_of_tier1_behavior(paths: DataPaths):
    logger.info("Running Tier 1: Comprehensive Behavioral Phenotyping Analysis...")
    metrics = {'mouse': paths.mouse_id, 'date': paths.date_str, 'genotype': paths.genotype}
    data_pack = {}
    
    try:
        dlc_loader = DLCDataLoader(paths.base_path)
        if paths.dlc_h5 and paths.dlc_h5.exists():
            df_dlc = dlc_loader.load(paths.dlc_h5)
            velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, strobe_path=paths.strobe_seconds)
            
            dt = np.median(np.diff(velocity_times)) if len(velocity_times) > 1 else 1/60.0
            dt = max(dt, 0.001)
            metrics.update(_compute_tier1_kinematics(velocity, dt))
            metrics.update(_compute_kinematic_metrics(velocity, velocity_times))
            
            scorer = df_dlc.columns.get_level_values(0).unique()[0]
            x_raw = df_dlc[(scorer, 'Snout', 'x')].values
            y_raw = df_dlc[(scorer, 'Snout', 'y')].values
            mask = df_dlc[(scorer, 'Snout', 'likelihood')].values > 0.8
            
            if len(x_raw) > len(velocity):
                x_raw = x_raw[:len(velocity)]
                y_raw = y_raw[:len(velocity)]
                mask = mask[:len(velocity)]
                
            s_met, hb_loc = _compute_tier1_spatial_extended(x_raw, y_raw, mask, dt)
            metrics.update(s_met)
            
            m_met, m_raw = _compute_tier1_microstructure(velocity, dt, x_raw, y_raw)
            metrics.update(m_met)
            
            data_pack['x'] = x_raw[mask]
            data_pack['y'] = y_raw[mask]
            data_pack['vel'] = velocity
            data_pack['m_raw'] = m_raw
    except Exception as e:
        logger.error(f"Tier 1 Kinematics failed: {e}")
        
    try:
        v_met, labels = _compute_vame_metrics(paths)
        metrics.update(v_met)
        if labels is not None:
             v2_met, bout_data = _compute_tier1_vame_extended(labels)
             metrics.update(v2_met)
             data_pack['vame_labels'] = labels
             data_pack['vame_bout_data'] = bout_data
    except Exception as e:
        logger.error(f"Tier 1 VAME failed: {e}")
        
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Tier 1: Comprehensive Behavioral Phenotyping | {paths.mouse_id}", fontsize=20, fontweight='bold', y=0.96)
    gs = GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.3)
    
    # 1. Spatial Coverage
    ax1 = fig.add_subplot(gs[0, 0])
    if 'x' in data_pack:
        hb = ax1.hexbin(data_pack['x'], data_pack['y'], gridsize=30, cmap='inferno', mincnt=1)
        ax1.invert_yaxis()
        ax1.set_title(f"Spatial Coverage (Entropy: {metrics.get('spatial_entropy', 0):.2f})")
        fig.colorbar(hb, ax=ax1, label='Density')
        
    # 2. Speed distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if 'vel' in data_pack:
        ax2.hist(data_pack['vel'], bins=50, color='royalblue', alpha=0.7, density=True)
        ax2.set_xlabel("Speed (cm/s)")
        ax2.set_ylabel("Density")
        ax2.set_xlim(0, 30)
        ax2.set_title("Speed Distribution")
        
    # 3. Accel/Decel Profiles
    ax3 = fig.add_subplot(gs[0, 2])
    if 'm_raw' in data_pack and len(data_pack['m_raw'][0]) > 0:
        peak_accels, peak_decels, _ = data_pack['m_raw']
        ax3.hist(peak_accels, bins=30, color='crimson', alpha=0.6, label='Peak Accel', density=True)
        ax3.hist(peak_decels, bins=30, color='teal', alpha=0.6, label='Peak Decel', density=True)
        ax3.set_xlabel("Acceleration (cm/s^2)")
        ax3.set_ylabel("Density")
        ax3.set_xlim(-200, 200)
        ax3.legend()
        ax3.set_title("Bout Acceleration Profiles")
        
    # 4. Turn Angle Distribution
    ax4 = fig.add_subplot(gs[1, 0], polar=True)
    if 'm_raw' in data_pack and len(data_pack['m_raw'][2]) > 0:
        _, _, angles = data_pack['m_raw']
        hist, bins = np.histogram(angles, bins=30, range=(0, np.pi))
        width = bins[1] - bins[0]
        ax4.bar(bins[:-1], hist, width=width, color='purple', alpha=0.7, bottom=0.0)
        ax4.set_title("Turn Angles (rad/frame) per Bout", pad=10)
        
    # 5. Motif Usage
    ax5 = fig.add_subplot(gs[1, 1])
    if 'vame_labels' in data_pack:
        labels = data_pack['vame_labels']
        unique, counts = np.unique(labels, return_counts=True)
        ax5.bar(unique, counts / len(labels), color='darkorange', alpha=0.8)
        ax5.set_xlabel("VAME Motif ID")
        ax5.set_ylabel("Usage Ratio")
        ax5.set_title(f"Motif Usage (RBI: {metrics.get('vame_repetitive_behavior_index', 0):.2f})")
        
    # 6. VAME Transitions
    ax6 = fig.add_subplot(gs[1, 2])
    if 'vame_labels' in data_pack:
        labels = data_pack['vame_labels']
        n_states = 15
        
        # Consistent transition matrix calculation
        trans = _get_vame_transition_matrix(labels, n_states=n_states)
                
        row_sums = trans.sum(axis=1, keepdims=True)
        trans_prob = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums!=0)
        im = ax6.imshow(trans_prob, cmap='viridis', aspect='auto')
        ax6.set_title("VAME Motif Sequence Transitions")
        ax6.set_xlabel("To State")
        ax6.set_ylabel("From State")
        fig.colorbar(im, ax=ax6)
    
    # 7. Motif Stickiness (Bout Duration per Motif)
    ax7 = fig.add_subplot(gs[2, 0])
    avg_durs = np.zeros(15)
    if 'vame_bout_data' in data_pack and len(data_pack['vame_bout_data']) > 0:
        bd = data_pack['vame_bout_data']
        m_ids = np.array([m for m, d in bd])
        durs = np.array([d for m, d in bd])
        for i in range(15):
            m_durs = durs[m_ids == i]
            avg_durs[i] = np.mean(m_durs) if len(m_durs) > 0 else 0
        ax7.bar(range(15), avg_durs, color='olive', alpha=0.7)
        ax7.set_xlabel("VAME Motif ID")
        ax7.set_ylabel("Avg Duration (Frames)")
        ax7.set_title("Motif Stickiness (State Persistence)")

    # 8. Stereotypy & Sequence Metrics
    ax8 = fig.add_subplot(gs[2, 1])
    h_max = np.log2(15)
    h_cond = metrics.get('vame_sequence_cond_entropy', 0)
    mi = metrics.get('vame_sequence_mutual_info', 0)
    pred_score = np.clip(mi / h_max, 0, 1) if h_max > 0 else 0
    rep_score = np.clip(1 - (h_cond / h_max), 0, 1) if h_max > 0 else 0
    
    bars = ['RQA_Det', 'RQA_Lam', 'VAME_Pred', 'VAME_Rep']
    vals = [metrics.get('rqa_determinism', 0), metrics.get('rqa_laminarity', 0), pred_score, rep_score]
    ax8.bar(bars, vals, color=['#1f77b4', '#3498db', '#e67e22', '#d35400'], alpha=0.8)
    ax8.set_ylim(0, 1)
    ax8.set_ylabel("Score (0-1)")
    ax8.set_title("Behavioral Stereotypy Profile")
    for i, v in enumerate(vals): ax8.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    # 9. Motif Dynamics (Predictability vs Stability)
    ax9 = fig.add_subplot(gs[2, 2])
    motif_entropies = metrics.get('vame_motif_entropies', np.zeros(15))
    motif_stabilities = metrics.get('vame_motif_stabilities', np.zeros(15))
    
    if np.any(motif_stabilities > 0):
        # Scale bubble size by overall usage
        _, counts = np.unique(data_pack.get('vame_labels', []), return_counts=True)
        usage = np.zeros(15)
        for i, c in zip(range(15), counts): usage[i] = c / len(data_pack['vame_labels'])
        
        scatter = ax9.scatter(motif_entropies, motif_stabilities, s=usage*5000, alpha=0.6, c=range(15), cmap='tab10')
        ax9.set_xlabel("Transition Entropy (bits)")
        ax9.set_ylabel("Duration Stability (1/(1+CV))")
        ax9.set_title("Motif Dynamics (Stability vs Predictability)")
        ax9.set_ylim(0, 1.1)
        
        # Identify "High Stereotypy" quadrant
        ax9.axvline(np.mean(motif_entropies[motif_entropies > 0]), color='gray', linestyle='--', alpha=0.3)
        ax9.axhline(np.mean(motif_stabilities[motif_stabilities > 0]), color='gray', linestyle='--', alpha=0.3)
        for i in range(15):
            if motif_stabilities[i] > 0:
                ax9.text(motif_entropies[i], motif_stabilities[i], str(i), fontsize=10, ha='center', va='center', fontweight='bold')

    # 10. Summary Panel (Expanded text row)
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    col1 = (
        f"--- Kinematics & Spatial ---\n"
        f"Time in Center: {metrics.get('time_in_center_perc', 0):.1f}%\n"
        f"Peak Speed: {metrics.get('peak_speed_cm_s', 0):.1f} cm/s\n"
        f"Path Tortuosity: {metrics.get('path_tortuosity_mean', 0):.2f}\n"
        f"Homebase Revisits: {metrics.get('homebase_revisits', 0)}\n"
        f"Cum_Area_Ratio (1/2): {metrics.get('cumulative_area_ratio_first_last_half', 0):.2f}\n"
    )
    col2 = (
        f"--- Microstructure & VAME Grammar ---\n"
        f"Turn Angle Mean: {metrics.get('turn_angle_mean_rad_per_frame', 0):.2f} rad\n"
        f"Trans Predictability: {metrics.get('vame_sequence_mutual_info', 0):.2f} bits\n"
        f"Trans Entropy: {metrics.get('vame_sequence_cond_entropy', 0):.2f} bits\n"
        f"Max Transition Prob: {metrics.get('vame_max_transition_prob', 0):.2f}\n"
        f"Combined Stereotypy Index (CSI): {metrics.get('vame_combined_stereotypy_index', 0):.2f}\n"
    )
    col3 = (
        f"--- State Stability ---\n"
        f"Motif Bout Mean: {metrics.get('vame_avg_bout_duration_frames', 0):.1f} f\n"
        f"Motif Bout Max: {metrics.get('vame_max_bout_duration_frames', 0):.0f} f\n"
        f"RQA Determinism: {metrics.get('rqa_determinism', 0):.2f}\n"
        f"RQA Laminarity: {metrics.get('rqa_laminarity', 0):.2f}\n"
        f"RQA Trapping Time: {metrics.get('rqa_trapping_time', 0):.2f} f\n"
    )
    col4 = (
        f"--- Latent Space ---\n"
        f"L-RQA Det: {metrics.get('vame_latent_rqa_det', np.nan):.2f}\n"
        f"L-RQA Lmax: {metrics.get('vame_latent_rqa_lmax', np.nan):.0f}\n"
        f"L-PCA1 Var: {metrics.get('vame_latent_pca1_var', np.nan):.2f}\n"
        f"L-Part. Ratio: {metrics.get('vame_latent_participation_ratio', np.nan):.1f}\n"
        f"L-AR1 R²: {metrics.get('vame_latent_ar1_r2', np.nan):.2f}\n"
        f"L-Vel Mean: {metrics.get('vame_latent_velocity_mean', np.nan):.2f}\n"
        f"L-DTW (half): {metrics.get('vame_latent_half_session_dtw', np.nan):.2f}\n"
    )
    ax10.text(0.00, 0.5, col1, fontsize=10, va='center', ha='left', family='monospace')
    ax10.text(0.25, 0.5, col2, fontsize=10, va='center', ha='left', family='monospace')
    ax10.text(0.50, 0.5, col3, fontsize=10, va='center', ha='left', family='monospace')
    ax10.text(0.75, 0.5, col4, fontsize=10, va='center', ha='left', family='monospace')
    
    out_dir = paths.base_path / "post_analysis" / "tier1_behavior"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{paths.mouse_id}_{paths.date_str}_tier1_dashboard.png", dpi=300)
    plt.close(fig)
    
    df = pd.DataFrame([metrics])
    csv_file = out_dir / "tier1_behavior_summary.csv"
    
    import os
    mode = 'a' if csv_file.exists() else 'w'
    header = not csv_file.exists()
    df.to_csv(csv_file, mode=mode, header=header, index=False)
    logger.info(f"Tier 1 completed. Saved to {csv_file}")

def _simple_peth(spike_times, event_times, pre=1.0, post=1.0, bin_size=0.1):
    bins = np.arange(-pre, post + bin_size, bin_size)
    counts = np.zeros(len(bins)-1)
    if len(event_times) == 0 or len(spike_times) == 0:
        return counts, bins[:-1] + bin_size/2
    
    for et in event_times:
        win_spikes = spike_times[(spike_times >= et - pre) & (spike_times <= et + post)]
        c, _ = np.histogram(win_spikes - et, bins=bins)
        counts += c
    
    counts = counts / (max(1, len(event_times)) * bin_size) # Hz
    return counts, bins[:-1] + bin_size/2

def _classify_cells_by_waveform(waveform_metrics, unique_clusters, unit_types_existing):
    """Refine MSN/FSI classification using trough-to-peak waveform duration.

    A trough-to-peak < 0.4 ms is classified as putative FSI (narrow-spiking);
    >= 0.4 ms is putative MSN (broad-spiking).  Falls back to the existing
    firing-rate rule if waveform data are unavailable.

    Returns
    -------
    unit_types : dict  {cluster_id -> 'MSN' | 'FSI' | 'Unknown'}
    wf_summary : dict  summary metrics for the population
    """
    unit_types = dict(unit_types_existing)  # copy
    ttp_vals = {'MSN': [], 'FSI': []}

    for cid in unique_clusters:
        wm = waveform_metrics.get(int(cid), {})
        ttp = wm.get('trough_to_peak_ms', np.nan)
        if not np.isnan(ttp):
            label = 'FSI' if ttp < 0.4 else 'MSN'
            unit_types[cid] = label
            ttp_vals[label].append(ttp)

    n_msn = sum(1 for v in unit_types.values() if v == 'MSN')
    n_fsi = sum(1 for v in unit_types.values() if v == 'FSI')
    wf_summary = {
        'n_msn_wf': n_msn,
        'n_fsi_wf': n_fsi,
        'ratio_msn_fsi_wf': n_msn / max(1, n_fsi),
        'mean_ttp_msn_ms': float(np.nanmean(ttp_vals['MSN'])) if ttp_vals['MSN'] else np.nan,
        'mean_ttp_fsi_ms': float(np.nanmean(ttp_vals['FSI'])) if ttp_vals['FSI'] else np.nan,
    }
    return unit_types, wf_summary


def _compute_acceleration_tuning(spike_times, spike_clusters, unique_clusters,
                                  velocity, velocity_times, unit_types):
    """Compute mean firing rate as a function of signed acceleration.

    Returns
    -------
    metrics : dict  population-level summary
    data_pack : dict  per-cell-type mean curves + bin centres
    """
    from scipy.ndimage import gaussian_filter1d
    metrics = {}
    data_pack = {}
    if len(velocity) < 10 or len(unique_clusters) == 0:
        return metrics, data_pack

    dt = np.median(np.diff(velocity_times))
    if dt <= 0: dt = 1/60.0
    accel = np.gradient(velocity, dt)
    bins_edges = np.linspace(np.nanpercentile(accel, 5), np.nanpercentile(accel, 95), 13)
    bin_centres = (bins_edges[:-1] + bins_edges[1:]) / 2
    hist_bins = np.append(velocity_times, velocity_times[-1] + dt)

    msn_curves, fsi_curves = [], []
    for cid in unique_clusters:
        spikes = spike_times[spike_clusters == cid]
        counts, _ = np.histogram(spikes, bins=hist_bins)
        fr = gaussian_filter1d(counts / dt, sigma=int(0.3/dt))
        curve = [np.nanmean(fr[(accel >= bins_edges[i]) & (accel < bins_edges[i+1])])
                 for i in range(len(bin_centres))]
        utype = unit_types.get(cid, 'MSN')
        (msn_curves if utype == 'MSN' else fsi_curves).append(curve)

    data_pack['accel_bin_centres'] = bin_centres
    data_pack['msn_accel'] = np.array(msn_curves) if msn_curves else np.empty((0, len(bin_centres)))
    data_pack['fsi_accel'] = np.array(fsi_curves) if fsi_curves else np.empty((0, len(bin_centres)))

    # Summary: proportion of units with positive vs negative acceleration modulation
    all_curves = msn_curves + fsi_curves
    if all_curves:
        slopes = [np.polyfit(bin_centres, c, 1)[0] for c in all_curves if not any(np.isnan(c))]
        metrics['prop_accel_positive_modulated'] = float(np.mean(np.array(slopes) > 0)) if slopes else np.nan
        metrics['mean_accel_slope'] = float(np.nanmean(slopes)) if slopes else np.nan

    return metrics, data_pack


def _compute_per_motif_selectivity(spike_times, spike_clusters, unique_clusters,
                                    vame_labels, velocity_times):
    """Compute per-motif selectivity index for each unit.

    Selectivity index (SI) = (FR_motif - FR_other) / (FR_motif + FR_other)
    for each (unit, motif) pair.

    Returns
    -------
    metrics : dict  population-level summary
    data_pack : dict  per-unit per-motif SI matrix + list of motifs
    """
    metrics = {}
    data_pack = {}
    if vame_labels is None or len(vame_labels) == 0 or len(unique_clusters) == 0:
        return metrics, data_pack

    # Align vame labels to velocity time base
    n = min(len(vame_labels), len(velocity_times))
    labels = vame_labels[:n]
    times = velocity_times[:n]
    dt = np.median(np.diff(times))
    if dt <= 0: dt = 1/60.0

    unique_states = np.unique(labels)
    hist_bins = np.append(times, times[-1] + dt)

    # Bin all spike trains
    all_fr = {}
    for cid in unique_clusters:
        spikes = spike_times[spike_clusters == cid]
        counts, _ = np.histogram(spikes, bins=hist_bins)
        all_fr[cid] = counts / dt

    si_matrix = []  # shape: (n_units, n_states)
    for cid in unique_clusters:
        fr = all_fr[cid]
        row = []
        for s in unique_states:
            mask = labels == s
            fr_s = np.nanmean(fr[mask]) if mask.sum() > 0 else 0.0
            fr_o = np.nanmean(fr[~mask]) if (~mask).sum() > 0 else 0.0
            denom = fr_s + fr_o
            si = (fr_s - fr_o) / denom if denom > 0 else 0.0
            row.append(si)
        si_matrix.append(row)

    si_arr = np.array(si_matrix)  # (n_units, n_states)
    data_pack['si_matrix'] = si_arr
    data_pack['si_states'] = unique_states

    # Best motif per unit (highest SI)
    best_si = np.nanmax(si_arr, axis=1)
    best_state = unique_states[np.nanargmax(si_arr, axis=1)]
    data_pack['best_si'] = best_si
    data_pack['best_state'] = best_state

    metrics['mean_best_motif_si'] = float(np.nanmean(best_si))
    metrics['prop_units_selective_si05'] = float(np.mean(best_si > 0.05))
    # State with the highest mean |SI| across units (most differentiating)
    mean_abs_si = np.nanmean(np.abs(si_arr), axis=0)
    metrics['most_differentiating_state'] = int(unique_states[np.argmax(mean_abs_si)])
    metrics['max_population_si'] = float(np.nanmax(mean_abs_si))

    return metrics, data_pack


def _compute_per_motif_da(da_signal, da_times, vame_labels, velocity_times, pre=2.0, post=3.0):
    """Compute DA dFF PETH aligned to the onset of each VAME motif.

    Returns
    -------
    data_pack : dict  per-motif mean/sem PETHs + time axis + state list
    """
    data_pack = {}
    if da_signal is None or len(da_signal) == 0 or vame_labels is None:
        return data_pack

    n = min(len(vame_labels), len(velocity_times))
    labels = np.array(vame_labels[:n])
    times = np.array(velocity_times[:n])
    unique_states = np.unique(labels)
    bin_size = 0.1
    bins = np.arange(-pre, post + bin_size, bin_size)
    bin_centres = bins[:-1] + bin_size / 2

    per_motif_mean = {}
    per_motif_sem = {}
    for s in unique_states:
        # Find onset frames: transitions *into* this state
        is_state = (labels == s).astype(int)
        onsets_idx = np.where(np.diff(is_state) == 1)[0] + 1
        if len(onsets_idx) < 3:
            continue
        onset_times = times[onsets_idx]

        trials = []
        for t0 in onset_times:
            t_lo, t_hi = t0 - pre, t0 + post
            mask = (da_times >= t_lo) & (da_times <= t_hi)
            if mask.sum() < 5:
                continue
            t_rel = da_times[mask] - t0
            da_win = da_signal[mask]
            # Interpolate to common grid
            try:
                interp = np.interp(bin_centres, t_rel, da_win)
                trials.append(interp)
            except Exception:
                pass

        if len(trials) >= 3:
            arr = np.array(trials)
            per_motif_mean[s] = np.nanmean(arr, axis=0)
            per_motif_sem[s] = np.nanstd(arr, axis=0) / np.sqrt(len(arr))

    data_pack['da_motif_peth_times'] = bin_centres
    data_pack['da_motif_peth_mean'] = per_motif_mean  # dict: state -> array
    data_pack['da_motif_peth_sem'] = per_motif_sem
    data_pack['da_motif_states'] = list(per_motif_mean.keys())
    return data_pack


def _compute_tier2_tuning_advanced(spike_times, spike_clusters, unique_clusters, velocity, velocity_times, unit_types, x, y):
    from scipy.ndimage import gaussian_filter1d
    metrics = {}
    data_pack = {'msn_speed': [], 'fsi_speed': [], 'angles': [], 'msn_dir': [], 'fsi_dir': [], 'move_fr': [], 'rest_fr': []}
    if len(velocity) == 0 or len(unique_clusters) == 0: return metrics, data_pack
    
    dt = np.median(np.diff(velocity_times))
    if dt <= 0: dt = 1/60.0
    bins = np.append(velocity_times, velocity_times[-1] + dt)
    
    # Speed and acceleration bins
    accel = np.gradient(velocity) / dt
    v_bins_edges = np.linspace(0, max(20, np.nanpercentile(velocity, 95)), 11)
    
    # Direction
    dx = np.diff(x); dy = np.diff(y)
    dx = np.append(dx, dx[-1]); dy = np.append(dy, dy[-1]) # pad
    angles = np.arctan2(dy, dx)
    angle_bins_edges = np.linspace(-np.pi, np.pi, 13)
    data_pack['angles'] = (angle_bins_edges[:-1] + angle_bins_edges[1:])/2
    
    is_moving = velocity > 2.0
    
    for cid in unique_clusters:
        spikes = spike_times[spike_clusters == cid]
        counts, _ = np.histogram(spikes, bins=bins)
        fr = counts / dt
        fr_smooth = gaussian_filter1d(fr, sigma=int(0.5/dt))
        
        utype = unit_types.get(cid, 'Unknown')
        if utype != 'MSN' and utype != 'FSI':
            baseline_fr = len(spikes) / max(1, velocity_times[-1] - velocity_times[0])
            utype = 'FSI' if baseline_fr > 10 else 'MSN'
            
        data_pack['move_fr'].append(np.nanmean(fr_smooth[is_moving]))
        data_pack['rest_fr'].append(np.nanmean(fr_smooth[~is_moving]))
        
        valid = ~np.isnan(velocity) & ~np.isnan(fr_smooth)
        if np.sum(valid) > 100:
            # Speed tuning
            speed_curve = []
            for i in range(len(v_bins_edges)-1):
                mask = valid & (velocity >= v_bins_edges[i]) & (velocity < v_bins_edges[i+1])
                speed_curve.append(np.nanmean(fr_smooth[mask]) if np.sum(mask)>0 else 0)
                
            # Direction tuning
            dir_curve = []
            for i in range(len(angle_bins_edges)-1):
                mask = valid & is_moving & (angles >= angle_bins_edges[i]) & (angles < angle_bins_edges[i+1])
                dir_curve.append(np.nanmean(fr_smooth[mask]) if np.sum(mask)>0 else 0)
                
            if utype == 'MSN':
                data_pack['msn_speed'].append(speed_curve)
                data_pack['msn_dir'].append(dir_curve)
            else:
                data_pack['fsi_speed'].append(speed_curve)
                data_pack['fsi_dir'].append(dir_curve)
                
    metrics['ratio_msn_to_fsi'] = len(data_pack['msn_speed']) / max(1, len(data_pack['fsi_speed']))
    data_pack['v_bins'] = (v_bins_edges[:-1] + v_bins_edges[1:])/2
    
    return metrics, data_pack

def analyze_of_tier2_single_unit(paths: DataPaths):
    import pandas as pd
    from data_loader import SpikeDataLoader, DLCDataLoader, PhotometryDataLoader
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from scipy.ndimage import gaussian_filter1d
    
    logger.info("Running Tier 2: Comprehensive Single-Unit Correlates Analysis...")
    metrics = {'mouse': paths.mouse_id, 'date': paths.date_str, 'genotype': paths.genotype}
    dp = {}
    
    try:
        spike_loader = SpikeDataLoader(paths.base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        unique_clusters = spike_data['unique_clusters']
        spike_times = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unit_types = spike_data.get('unit_types', {})

        # 2D – Waveform-based cell type classification (refines FR-based labels)
        waveform_metrics_per_unit = spike_data.get('waveform_metrics', {})
        unit_types, wf_summary = _classify_cells_by_waveform(
            waveform_metrics_per_unit, unique_clusters, unit_types
        )
        metrics.update(wf_summary)
        # Trough-to-peak distribution for dashboard
        dp['ttp_msn'] = [waveform_metrics_per_unit[cid]['trough_to_peak_ms']
                         for cid in unique_clusters
                         if cid in waveform_metrics_per_unit and unit_types.get(cid) == 'MSN'
                         and not np.isnan(waveform_metrics_per_unit[cid]['trough_to_peak_ms'])]
        dp['ttp_fsi'] = [waveform_metrics_per_unit[cid]['trough_to_peak_ms']
                         for cid in unique_clusters
                         if cid in waveform_metrics_per_unit and unit_types.get(cid) == 'FSI'
                         and not np.isnan(waveform_metrics_per_unit[cid]['trough_to_peak_ms'])]
        
        dlc_loader = DLCDataLoader(paths.base_path)
        if paths.dlc_h5 and paths.dlc_h5.exists():
            df_dlc = dlc_loader.load(paths.dlc_h5)
            velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, strobe_path=paths.strobe_seconds)
            
            scorer = df_dlc.columns.get_level_values(0).unique()[0]
            x_raw = df_dlc[(scorer, 'Snout', 'x')].values[:len(velocity)]
            y_raw = df_dlc[(scorer, 'Snout', 'y')].values[:len(velocity)]
            
            t_met, t_dp = _compute_tier2_tuning_advanced(spike_times, spike_clusters, unique_clusters, velocity, velocity_times, unit_types, x_raw, y_raw)
            metrics.update(t_met)
            dp.update(t_dp)
            
            from analyses_openfield import _compute_vame_metrics
            v_met, labels = _compute_vame_metrics(paths)
            
            if labels is not None:
                min_len = min(len(labels), len(velocity))
                lbls = labels[:min_len]
                v_times = velocity_times[:min_len]
                v_trim = velocity[:min_len]
                
                # Selectivity Index & Decoding
                dt = np.median(np.diff(v_times))
                if dt <= 0: dt = 1/60.0
                bins = np.append(v_times, v_times[-1] + dt)
                
                pop_rate = []
                for cid in unique_clusters:
                    fr = np.histogram(spike_times[spike_clusters == cid], bins=bins)[0] / dt
                    pop_rate.append(gaussian_filter1d(fr, sigma=int(0.1/dt)))
                pop_mat = np.array(pop_rate).T
                
                unique_states, counts = np.unique(lbls, return_counts=True)
                top_states = unique_states[np.argsort(counts)[-3:]] # Top 3
                
                # Single-unit Motif Decoding AUC
                auc_scores = []
                if len(top_states) >= 1:
                    target_motif = top_states[-1]
                    y_bin = (lbls == target_motif).astype(int)
                    # Subsample
                    sub_idx = np.arange(0, len(y_bin), 10)
                    for i in range(pop_mat.shape[1]):
                        X_u = pop_mat[sub_idx, i].reshape(-1, 1)
                        y_u = y_bin[sub_idx]
                        if len(np.unique(y_u)) > 1:
                            clf = LogisticRegression(class_weight='balanced')
                            scores = cross_val_score(clf, X_u, y_u, cv=3, scoring='roc_auc')
                            auc_scores.append(np.mean(scores))
                dp['auc_scores'] = auc_scores
                metrics['mean_single_unit_motif_auc'] = np.mean(auc_scores) if auc_scores else np.nan
                
                # PETHs for Top Motif
                if len(top_states) >= 1:
                    is_top = (lbls == top_states[-1]).astype(int)
                    onsets_idx = np.where(np.diff(is_top) == 1)[0] + 1
                    onset_times_motif = v_times[onsets_idx]
                    
                    all_peths = []
                    for cid in unique_clusters:
                        peth_y, peth_x = _simple_peth(spike_times[spike_clusters == cid], onset_times_motif, pre=2.0, post=2.0, bin_size=0.1)
                        all_peths.append(peth_y)
                    dp['motif_peths'] = all_peths
                    dp['peth_x'] = peth_x
                else:
                    onset_times_motif = []

                # 2A – Acceleration tuning curves
                ac_met, ac_dp = _compute_acceleration_tuning(
                    spike_times, spike_clusters, unique_clusters,
                    velocity, velocity_times, unit_types
                )
                metrics.update(ac_met)
                dp.update({f'accel_{k}': v for k, v in ac_dp.items()})

                # 2B – Per-motif selectivity index (all motifs, not just top)
                si_met, si_dp = _compute_per_motif_selectivity(
                    spike_times, spike_clusters, unique_clusters,
                    vame_labels=labels, velocity_times=v_times
                )
                metrics.update(si_met)
                dp.update({f'si_{k}': v for k, v in si_dp.items()})

            else:
                onset_times_motif = []

            try:
                if paths.tdt_dff and paths.tdt_dff.exists():
                    tdt_loader = PhotometryDataLoader(paths.base_path)
                    tdt_data = tdt_loader.load(paths.tdt_dff, paths.tdt_raw)
                    dff = tdt_data['dff_values']
                    ts = tdt_data['dff_timestamps']
                    
                    from scipy.interpolate import interp1d
                    f_int = interp1d(ts, dff, bounds_error=False, fill_value=np.nan)
                    p_time = np.linspace(-2.0, 2.0, 40)
                    
                    # DA Move PETH
                    move_onsets = velocity_times[np.where(np.diff((velocity > 2.0).astype(int)) == 1)[0] + 1]
                    da_move = np.array([f_int(et + p_time) for et in move_onsets])
                    dp['da_move'] = np.nanmean(da_move, axis=0) if len(da_move) > 0 else []
                    
                    # Ramp
                    if len(da_move) > 0:
                        pre_ramp = da_move[:, p_time < 0]
                        ramp_slopes = [np.polyfit(p_time[p_time < 0], pr, 1)[0] for pr in pre_ramp if not np.isnan(pr).any()]
                        metrics['da_pre_move_ramp_slope'] = np.mean(ramp_slopes) if ramp_slopes else np.nan
                    
                    # DA Motif PETH (top motif only, for backward compat)
                    if len(onset_times_motif) > 0:
                        da_motif = np.array([f_int(et + p_time) for et in onset_times_motif])
                        dp['da_motif'] = np.nanmean(da_motif, axis=0) if len(da_motif) > 0 else []
                    dp['da_x'] = p_time

                    # 2C – Per-motif DA PETHs (all VAME states)
                    if 'vame_labels' in dp or labels is not None:
                        vl = labels if labels is not None else dp.get('vame_labels')
                        da_pm_dp = _compute_per_motif_da(dff, ts, vl, velocity_times)
                        dp.update(da_pm_dp)

            except Exception as e_da:
                logger.warning(f"DA PETHs failed: {e_da}")
                    
    except Exception as e:
        logger.error(f"Tier 2 failed: {e}")
        
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Tier 2: Comprehensive Single-Unit Correlates | {paths.mouse_id}", fontsize=20, fontweight='bold', y=0.98)
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # 1. Speed Tuning
    ax1 = fig.add_subplot(gs[0, 0])
    if 'msn_speed' in dp and 'v_bins' in dp:
        if dp['msn_speed']:
            ax1.plot(dp['v_bins'], np.nanmean(dp['msn_speed'], axis=0), color='royalblue', lw=2, label=f"MSN (n={len(dp['msn_speed'])})")
            ax1.fill_between(dp['v_bins'], np.nanmean(dp['msn_speed'], axis=0)-np.nanstd(dp['msn_speed'], axis=0)/np.sqrt(len(dp['msn_speed'])), np.nanmean(dp['msn_speed'], axis=0)+np.nanstd(dp['msn_speed'], axis=0)/np.sqrt(len(dp['msn_speed'])), color='royalblue', alpha=0.3)
        if dp['fsi_speed']:
            ax1.plot(dp['v_bins'], np.nanmean(dp['fsi_speed'], axis=0), color='crimson', lw=2, label=f"FSI (n={len(dp['fsi_speed'])})")
            ax1.fill_between(dp['v_bins'], np.nanmean(dp['fsi_speed'], axis=0)-np.nanstd(dp['fsi_speed'], axis=0)/np.sqrt(len(dp['fsi_speed'])), np.nanmean(dp['fsi_speed'], axis=0)+np.nanstd(dp['fsi_speed'], axis=0)/np.sqrt(len(dp['fsi_speed'])), color='crimson', alpha=0.3)
        ax1.set_xlabel("Speed (cm/s)")
        ax1.set_ylabel("Firing Rate (Hz)")
        ax1.set_title("Speed Tuning Curves")
        ax1.legend()
        
    # 2. Move vs Pause FR
    ax2 = fig.add_subplot(gs[0, 1])
    if 'move_fr' in dp and dp['move_fr']:
        ax2.bar(['Move (>2cm/s)', 'Rest (<2cm/s)'], [np.nanmean(dp['move_fr']), np.nanmean(dp['rest_fr'])], color=['seagreen', 'gray'], alpha=0.8)
        ax2.set_ylabel("Mean Population Firing Rate (Hz)")
        ax2.set_title("Move vs Pause Firing Rate")
        
    # 3. Direction Tuning
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    if 'msn_dir' in dp and 'angles' in dp:
        if dp['msn_dir']:
            ax3.plot(dp['angles'], np.nanmean(dp['msn_dir'], axis=0), color='royalblue', lw=2, label='MSN')
        if dp['fsi_dir']:
            ax3.plot(dp['angles'], np.nanmean(dp['fsi_dir'], axis=0), color='crimson', lw=2, label='FSI')
        ax3.set_title("Directional Tuning (FR vs Angle)", pad=10)
        
    # 4. Spike Motif PETH
    ax4 = fig.add_subplot(gs[1, 0])
    if 'motif_peths' in dp and dp['motif_peths']:
        mean_peth = np.nanmean(dp['motif_peths'], axis=0)
        sem_peth = np.nanstd(dp['motif_peths'], axis=0) / np.sqrt(len(dp['motif_peths']))
        ax4.plot(dp['peth_x'], mean_peth, color='indigo', lw=2)
        ax4.fill_between(dp['peth_x'], mean_peth-sem_peth, mean_peth+sem_peth, color='indigo', alpha=0.3)
        ax4.axvline(0, color='black', linestyle='--')
        ax4.set_xlabel("Time from Top Motif Onset (s)")
        ax4.set_ylabel("Firing Rate (Hz)")
        ax4.set_title("Population Motif PETH")
        
    # 5. Motif Decoding AUC Dist
    ax5 = fig.add_subplot(gs[1, 1])
    if 'auc_scores' in dp and dp['auc_scores']:
        ax5.hist(dp['auc_scores'], bins=20, color='darkorange', alpha=0.7, edgecolor='k')
        ax5.axvline(0.5, color='gray', linestyle='--', label='Chance')
        ax5.axvline(np.mean(dp['auc_scores']), color='red', linestyle='--', label='Mean')
        ax5.set_xlabel("ROC-AUC Score")
        ax5.set_ylabel("Unit Count")
        ax5.set_title("Single-Unit Motif Decoding Performance")
        ax5.legend()
        
    # 6. Cell Type Ratio
    ax6 = fig.add_subplot(gs[1, 2])
    if 'msn_speed' in dp:
        n_m = len(dp['msn_speed'])
        n_f = len(dp['fsi_speed'])
        if (n_m + n_f) > 0:
            ax6.pie([n_m, n_f], labels=[f'MSN ({n_m})', f'FSI ({n_f})'], colors=['royalblue', 'crimson'], autopct='%1.1f%%', startangle=90)
            ax6.set_title("Putative Cell Type Ratio")
            
    # 7. DA Move PETH
    ax7 = fig.add_subplot(gs[2, 0])
    if 'da_move' in dp and len(dp['da_move']) > 0:
        ax7.plot(dp['da_x'], dp['da_move'], color='forestgreen', lw=2)
        ax7.axvline(0, color='black', linestyle='--')
        ax7.set_xlabel("Time from Movement Onset (s)")
        ax7.set_ylabel("Photometry dF/F")
        ax7.set_title("Dopamine Movement PETH")
        
    # 8. DA Motif PETH
    ax8 = fig.add_subplot(gs[2, 1])
    if 'da_motif' in dp and len(dp['da_motif']) > 0:
        ax8.plot(dp['da_x'], dp['da_motif'], color='darkmagenta', lw=2)
        ax8.axvline(0, color='black', linestyle='--')
        ax8.set_xlabel("Time from Motif Onset (s)")
        ax8.set_ylabel("Photometry dF/F")
        ax8.set_title("Dopamine Top Motif PETH")
        
    # 9. Acceleration Tuning Curves
    ax9 = fig.add_subplot(gs[3, 0])
    if 'accel_accel_bin_centres' in dp:
        bc = dp['accel_accel_bin_centres']
        if dp.get('accel_msn_accel') is not None and len(dp['accel_msn_accel']) > 0:
            m = np.nanmean(dp['accel_msn_accel'], axis=0)
            se = np.nanstd(dp['accel_msn_accel'], axis=0) / max(1, np.sqrt(len(dp['accel_msn_accel'])))
            ax9.plot(bc, m, color='royalblue', lw=2, label='MSN')
            ax9.fill_between(bc, m - se, m + se, color='royalblue', alpha=0.3)
        if dp.get('accel_fsi_accel') is not None and len(dp['accel_fsi_accel']) > 0:
            m = np.nanmean(dp['accel_fsi_accel'], axis=0)
            se = np.nanstd(dp['accel_fsi_accel'], axis=0) / max(1, np.sqrt(len(dp['accel_fsi_accel'])))
            ax9.plot(bc, m, color='crimson', lw=2, label='FSI')
            ax9.fill_between(bc, m - se, m + se, color='crimson', alpha=0.3)
        ax9.axvline(0, color='k', linestyle='--', lw=1)
        ax9.set_xlabel("Acceleration (cm/s²)")
        ax9.set_ylabel("Firing Rate (Hz)")
        ax9.set_title("Acceleration Tuning Curves")
        ax9.legend(fontsize=8)

    # 10. Per-Motif Selectivity Index Heatmap
    ax10 = fig.add_subplot(gs[3, 1])
    if 'si_si_matrix' in dp and dp['si_si_matrix'] is not None and dp['si_si_matrix'].size > 0:
        si_mat = dp['si_si_matrix']
        si_states = dp.get('si_si_states', np.arange(si_mat.shape[1]))
        im = ax10.imshow(si_mat, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                         interpolation='nearest')
        ax10.set_xlabel("VAME Motif State")
        ax10.set_ylabel("Unit #")
        ax10.set_title(f"Per-Motif Selectivity Index\n(mean best SI={metrics.get('mean_best_motif_si', 0):.3f})")
        ax10.set_xticks(range(len(si_states)))
        ax10.set_xticklabels(si_states, fontsize=7, rotation=45)
        plt.colorbar(im, ax=ax10, label='SI')

    # 11. Per-Motif DA PETHs (overlay all motifs)
    ax11 = fig.add_subplot(gs[3, 2])
    if 'da_motif_peth_mean' in dp and dp['da_motif_peth_mean']:
        t_ax = dp.get('da_motif_peth_times', np.array([]))
        cmap_states = plt.cm.tab20
        for si_idx, (state, mean_trace) in enumerate(dp['da_motif_peth_mean'].items()):
            sem_trace = dp['da_motif_peth_sem'].get(state, np.zeros_like(mean_trace))
            col = cmap_states(si_idx % 20)
            ax11.plot(t_ax, mean_trace, color=col, lw=1.5, label=f"S{state}")
            ax11.fill_between(t_ax, mean_trace - sem_trace, mean_trace + sem_trace,
                              color=col, alpha=0.15)
        ax11.axvline(0, color='k', linestyle='--', lw=1)
        ax11.set_xlabel("Time from Motif Onset (s)")
        ax11.set_ylabel("dF/F")
        ax11.set_title("DA PETHs — All VAME Motifs")
        if len(dp['da_motif_peth_mean']) <= 10:
            ax11.legend(fontsize=6, ncol=2)

    # 12. Waveform TTP Distribution + Summary
    ax12 = fig.add_subplot(gs[2, 2])
    ttp_msn = dp.get('ttp_msn', [])
    ttp_fsi = dp.get('ttp_fsi', [])
    if ttp_msn or ttp_fsi:
        if ttp_msn:
            ax12.hist(ttp_msn, bins=15, color='royalblue', alpha=0.6, label='MSN', density=True)
        if ttp_fsi:
            ax12.hist(ttp_fsi, bins=15, color='crimson', alpha=0.6, label='FSI', density=True)
        ax12.axvline(0.4, color='k', linestyle='--', lw=1, label='0.4ms threshold')
        ax12.set_xlabel("Trough-to-Peak (ms)")
        ax12.set_ylabel("Density")
        ax12.set_title(f"Waveform Width Distribution\n(MSN={metrics.get('n_msn_wf',0)}, FSI={metrics.get('n_fsi_wf',0)})")
        ax12.legend(fontsize=8)
    else:
        ax12.axis('off')
        ax12.text(0.5, 0.5, "No waveform data", ha='center', va='center', transform=ax12.transAxes)
        ax12.set_title("Waveform TTP Distribution")

    # Summary Panel (moved to gs[2,2] → now at bottom right of row 2)
    ax_sum = fig.add_subplot(gs[2, 2]) if (not ttp_msn and not ttp_fsi) else None
    # (ax12 already takes gs[2,2] — summary text appended to ax12 if waveforms absent)
    if ax_sum is None:
        # Add a small text box to ax12
        pass  # summary info in title already

    # Standalone summary panel if we have space
    _ax_summary_t2 = None
    # Build summary text on the existing panel
    summary_txt = (
        f"MSN/FSI (FR): {metrics.get('ratio_msn_to_fsi', 0):.2f}\n"
        f"MSN/FSI (WF): {metrics.get('ratio_msn_fsi_wf', 0):.2f}\n"
        f"n_MSN={metrics.get('n_msn_wf',0)}  n_FSI={metrics.get('n_fsi_wf',0)}\n"
        f"Mean Motif AUC: {metrics.get('mean_single_unit_motif_auc', 0):.3f}\n"
        f"Mean Best SI: {metrics.get('mean_best_motif_si', 0):.3f}\n"
        f"Prop SI>0.05: {metrics.get('prop_units_selective_si05', 0):.2f}\n"
        f"Accel slope: {metrics.get('mean_accel_slope', 0):.4f}\n"
        f"DA Ramp: {metrics.get('da_pre_move_ramp_slope', 0):.4f}\n"
    )
    # Place summary in corner of waveform panel
    if ttp_msn or ttp_fsi:
        ax12.text(0.98, 0.98, summary_txt, fontsize=7, va='top', ha='right',
                  transform=ax12.transAxes, family='monospace',
                  bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    out_dir = paths.base_path / "post_analysis" / "tier2_single_unit"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{paths.mouse_id}_{paths.date_str}_tier2_dashboard.png", dpi=300)
    plt.close(fig)
    
    df = pd.DataFrame([metrics])
    csv_file = out_dir / "tier2_unit_summary.csv"
    import os
    mode = 'a' if csv_file.exists() else 'w'
    header = not csv_file.exists()
    df.to_csv(csv_file, mode=mode, header=header, index=False)
    logger.info(f"Tier 2 completed. Saved to {csv_file}")

def _compute_comodulogram(trace, fs, phase_bands, amp_bands):
    from scipy.signal import butter, filtfilt, hilbert
    import numpy as np
    import scipy.stats
    
    def filter_sig(data, low, high, fs):
        nyq = 0.5 * fs
        b, a = butter(3, [low/nyq, high/nyq], btype='bandpass')
        return filtfilt(b, a, data)
        
    def get_pac(phase_sig, amp_sig):
        n_bins = 18
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        db = np.digitize(phase_sig, bins) - 1
        mean_amp = np.zeros(n_bins)
        for i in range(n_bins):
            mask = db == i
            if np.sum(mask) > 0:
                mean_amp[i] = np.mean(amp_sig[mask])
        sum_amp = np.sum(mean_amp)
        if sum_amp == 0: return 0
        mean_amp = mean_amp / sum_amp
        h = scipy.stats.entropy(mean_amp)
        h_max = np.log(n_bins)
        return (h_max - h) / h_max if h_max > 0 else 0

    comod = np.zeros((len(amp_bands), len(phase_bands)))
    
    phase_sigs = []
    for pb in phase_bands:
        sig = filter_sig(trace, pb[0], pb[1], fs)
        phase_sigs.append(np.angle(hilbert(sig)))
        
    amp_sigs = []
    for ab in amp_bands:
        sig = filter_sig(trace, ab[0], ab[1], fs)
        amp_sigs.append(np.abs(hilbert(sig)))
        
    for j, a_sig in enumerate(amp_sigs):
        for i, p_sig in enumerate(phase_sigs):
            comod[j, i] = get_pac(p_sig, a_sig)
            
    return comod

def analyze_of_tier3_lfp(paths: DataPaths):
    from logging import getLogger
    logger = getLogger(__name__)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.signal import butter, filtfilt, hilbert, welch, spectrogram
    from data_loader import LFPDataLoader, DLCDataLoader, SpikeDataLoader, PhotometryDataLoader
    
    logger.info("Running Tier 3: Comprehensive LFP Oscillatory Dynamics...")
    metrics = {'mouse': paths.mouse_id, 'date': paths.date_str, 'genotype': paths.genotype}
    dp = {}
    
    try:
        dlc_loader = DLCDataLoader(paths.base_path)
        if paths.dlc_h5 and paths.dlc_h5.exists():
            df_dlc = dlc_loader.load(paths.dlc_h5)
            velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, strobe_path=paths.strobe_seconds)
        else:
            velocity, velocity_times = np.array([]), np.array([])
            
        spike_loader = SpikeDataLoader(paths.base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        unique_clusters = spike_data.get('unique_clusters', np.array([]))
        
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        channels = lfp_loader.channel_ids
        
        if len(channels) > 0 and len(velocity_times) > 0:
            # Dynamically select channel with highest spike density
            # Kilosort/Phy usually maps clusters to a primary channel. If 'cluster_channels' is not available,
            # we will just default to the middle, but if we can estimate, we pick the most active.
            try:
                # If we have cluster channel mapping in spike_data (depends on data_loader implementation)
                if 'cluster_channels' in spike_data:
                    c_chans = [spike_data['cluster_channels'].get(c, channels[len(channels)//2]) for c in unique_clusters]
                    best_chan = int(scipy.stats.mode(c_chans, keepdims=False)[0])
                else:
                    best_chan = channels[len(channels)//2]
            except:
                best_chan = channels[len(channels)//2]
                
            # Ensure we have boundary channels for CSD (spacing of 2 channels = ~40um on Neuropixels)
            best_idx = np.where(channels == best_chan)[0]
            if len(best_idx) > 0: best_idx = best_idx[0]
            else: best_idx = len(channels)//2
            
            best_idx = np.clip(best_idx, 2, len(channels)-3)
            # Load the target channel and two surrounding channels for 1D CSD
            trg_chans = [channels[best_idx-2], channels[best_idx], channels[best_idx+2]]
            
            # Analyze 5 minutes to fit in memory
            t_start = velocity_times[0]
            t_end = min(velocity_times[-1], t_start + 300)
            
            traces, timestamps = lfp_loader.get_data(t_start, t_end, channels=trg_chans)
            
            if len(traces) == len(timestamps) and traces.shape[1] >= 3:
                # 1D approximate CSD (Current Source Density) snippet:
                # CSD approx = -(V_top - 2*V_mid + V_bottom)
                trace = -1.0 * (traces[:, 0] - 2*traces[:, 1] + traces[:, 2])
                fs = lfp_loader.fs
                dp['lfp_fs'] = fs
                
                # Interpolate velocity to LFP timestamps
                from scipy.interpolate import interp1d
                v_int = interp1d(velocity_times, velocity, bounds_error=False, fill_value=0)
                lfp_vel = v_int(timestamps)
                is_move = lfp_vel > 2.0
                
                # 3A. PSD Move vs Rest
                f_m, psd_m = welch(trace[is_move], fs, nperseg=int(fs*2)) if sum(is_move) > int(fs*2) else ([], [])
                f_r, psd_r = welch(trace[~is_move], fs, nperseg=int(fs*2)) if sum(~is_move) > int(fs*2) else ([], [])
                dp['f'] = f_m if len(f_m) > 0 else f_r
                dp['psd_m'] = psd_m
                dp['psd_r'] = psd_r
                
                def filter_sig(data, low, high, fs):
                    nyq = 0.5 * fs
                    b, a = butter(3, [low/nyq, high/nyq], btype='bandpass')
                    return filtfilt(b, a, data)
                
                theta = filter_sig(trace, 4, 8, fs)
                beta = filter_sig(trace, 13, 30, fs)
                gamma = filter_sig(trace, 30, 80, fs)
                
                dp['theta_env'] = np.abs(hilbert(theta))
                dp['beta_env'] = np.abs(hilbert(beta))
                dp['gamma_env'] = np.abs(hilbert(gamma))
                
                # Spectrogram around movement onsets
                # Subsample trace to 250Hz for faster spectrogram
                ds_factor = int(fs / 250)
                trace_ds = trace[::ds_factor]
                fs_ds = fs / ds_factor
                f_sg, t_sg, Sxx = spectrogram(trace_ds, fs=fs_ds, nperseg=int(fs_ds), noverlap=int(fs_ds*0.9))
                t_sg_real = timestamps[0] + t_sg
                
                vel_onsets = velocity_times[np.where(np.diff((velocity > 2.0).astype(int)) == 1)[0] + 1]
                vel_onsets = vel_onsets[(vel_onsets > t_start+2) & (vel_onsets < t_end-2)]
                
                sg_peth = []
                for onset in vel_onsets:
                    idx = (t_sg_real >= onset - 2.0) & (t_sg_real <= onset + 2.0)
                    if np.sum(idx) > 10:
                        sg_peth.append(Sxx[:, idx])
                if len(sg_peth) > 0:
                    min_len = min([s.shape[1] for s in sg_peth])
                    dp['sg_mean'] = np.mean([s[:, :min_len] for s in sg_peth], axis=0)
                    dp['sg_f'] = f_sg
                    dp['sg_t'] = np.linspace(-2.0, 2.0, min_len)
                    
                # 3B. PAC Comodulogram
                phase_bands = [(2,4), (4,8), (8,12), (13,30)]
                amp_bands = [(30,50), (50,80), (80,120), (120,160)]
                dp['comod_move'] = _compute_comodulogram(trace[is_move], fs, phase_bands, amp_bands) if sum(is_move) > fs else np.zeros((4,4))
                dp['comod_rest'] = _compute_comodulogram(trace[~is_move], fs, phase_bands, amp_bands) if sum(~is_move) > fs else np.zeros((4,4))
                metrics['pac_theta_gamma_move'] = dp['comod_move'][1, 1]
                metrics['pac_beta_gamma_move'] = dp['comod_move'][1, 3]
                
                # 3C. Spike-Phase Locking
                theta_phase = np.angle(hilbert(theta))
                spk = spike_data['spike_times_sec']
                spk = spk[(spk >= t_start) & (spk <= t_end)]
                clst = spike_data['spike_clusters'][(spike_data['spike_times_sec'] >= t_start) & (spike_data['spike_times_sec'] <= t_end)]
                
                mvl_move = []
                mvl_rest = []
                for cid in spike_data['unique_clusters']:
                    c_spk = spk[clst == cid]
                    if len(c_spk) > 50:
                        spk_idx = np.searchsorted(timestamps, c_spk) - 1
                        spk_idx = spk_idx[(spk_idx >= 0) & (spk_idx < len(theta_phase))]
                        
                        phases = theta_phase[spk_idx]
                        moves = is_move[spk_idx]
                        
                        if sum(moves) > 10:
                            mvl_move.append(np.abs(np.mean(np.exp(1j * phases[moves]))))
                        if sum(~moves) > 10:
                            mvl_rest.append(np.abs(np.mean(np.exp(1j * phases[~moves]))))
                
                metrics['mean_theta_mvl_move'] = np.mean(mvl_move) if mvl_move else np.nan
                metrics['mean_theta_mvl_rest'] = np.mean(mvl_rest) if mvl_rest else np.nan
                dp['mvl_move'] = mvl_move
                dp['mvl_rest'] = mvl_rest

                # 3A (extended). Band-specific power vs speed
                # Compute mean envelope amplitude per speed bin
                speed_bins_edges = np.linspace(0, max(20, np.nanpercentile(lfp_vel, 95)), 11)
                speed_bin_centres = (speed_bins_edges[:-1] + speed_bins_edges[1:]) / 2
                dp['band_speed_bins'] = speed_bin_centres
                for band_name, band_env in [('theta', dp['theta_env']),
                                             ('beta', dp['beta_env']),
                                             ('gamma', dp['gamma_env'])]:
                    curve = []
                    for i in range(len(speed_bins_edges) - 1):
                        mask_bin = (lfp_vel >= speed_bins_edges[i]) & (lfp_vel < speed_bins_edges[i+1])
                        curve.append(np.nanmean(band_env[mask_bin]) if mask_bin.sum() > 0 else np.nan)
                    dp[f'{band_name}_power_vs_speed'] = np.array(curve)
                    # Summary: Spearman correlation between band power and speed
                    valid = ~np.isnan(lfp_vel) & ~np.isnan(band_env)
                    if valid.sum() > 100:
                        from scipy.stats import spearmanr
                        rho, p_rho = spearmanr(lfp_vel[valid], band_env[valid])
                        metrics[f'{band_name}_power_speed_rho'] = float(rho)
                        metrics[f'{band_name}_power_speed_p'] = float(p_rho)

                # 3C (extended). Multi-band phase locking + PPC (Pairwise Phase Consistency)
                # Add beta & gamma in addition to existing theta
                beta_phase = np.angle(hilbert(beta))
                gamma_phase = np.angle(hilbert(gamma))

                for band_name, b_phase in [('beta', beta_phase), ('gamma', gamma_phase)]:
                    mvl_m, mvl_r, ppc_m = [], [], []
                    for cid in spike_data['unique_clusters']:
                        c_spk = spk[clst == cid]
                        if len(c_spk) < 50:
                            continue
                        spk_idx = np.searchsorted(timestamps, c_spk) - 1
                        spk_idx = spk_idx[(spk_idx >= 0) & (spk_idx < len(b_phase))]
                        phases = b_phase[spk_idx]
                        moves = is_move[spk_idx]

                        if moves.sum() > 10:
                            ph_m = phases[moves]
                            mvl = np.abs(np.mean(np.exp(1j * ph_m)))
                            mvl_m.append(mvl)
                            # PPC = (n * MVL^2 - 1) / (n - 1)   (Vinck et al. 2010)
                            n = len(ph_m)
                            ppc = (n * mvl**2 - 1) / (n - 1) if n > 1 else np.nan
                            ppc_m.append(ppc)
                        if (~moves).sum() > 10:
                            mvl_r.append(np.abs(np.mean(np.exp(1j * phases[~moves]))))

                    metrics[f'mean_{band_name}_mvl_move'] = float(np.nanmean(mvl_m)) if mvl_m else np.nan
                    metrics[f'mean_{band_name}_mvl_rest'] = float(np.nanmean(mvl_r)) if mvl_r else np.nan
                    metrics[f'mean_{band_name}_ppc_move'] = float(np.nanmean(ppc_m)) if ppc_m else np.nan
                    dp[f'mvl_{band_name}_move'] = mvl_m
                    dp[f'mvl_{band_name}_rest'] = mvl_r
                    dp[f'ppc_{band_name}_move'] = ppc_m

                # Also add theta PPC
                ppc_theta_m = []
                for cid in spike_data['unique_clusters']:
                    c_spk = spk[clst == cid]
                    if len(c_spk) < 50:
                        continue
                    spk_idx = np.searchsorted(timestamps, c_spk) - 1
                    spk_idx = spk_idx[(spk_idx >= 0) & (spk_idx < len(theta_phase))]
                    ph_m = theta_phase[spk_idx][is_move[spk_idx]]
                    if len(ph_m) > 1:
                        mvl = np.abs(np.mean(np.exp(1j * ph_m)))
                        n = len(ph_m)
                        ppc_theta_m.append((n * mvl**2 - 1) / (n - 1))
                metrics['mean_theta_ppc_move'] = float(np.nanmean(ppc_theta_m)) if ppc_theta_m else np.nan
                dp['ppc_theta_move'] = ppc_theta_m

                # 3D. DA - LFP Coupling
                try:
                    if paths.tdt_dff and paths.tdt_dff.exists():
                        tdt_loader = PhotometryDataLoader(paths.base_path)
                        tdt_data = tdt_loader.load(paths.tdt_dff, paths.tdt_raw)
                        dff = np.asarray(tdt_data['dff_values']).flatten()
                        ts = np.asarray(tdt_data['dff_timestamps']).flatten()

                        # Restrict DA analyses to the same time window as the loaded LFP.
                        da_window_mask = (ts >= t_start) & (ts <= t_end)
                        dff = dff[da_window_mask]
                        ts = ts[da_window_mask]
                        if len(ts) < 10:
                            raise ValueError("Insufficient DA samples within the analyzed LFP window")
                        
                        # Interpolate everything to 100Hz for cross-corr
                        t_cc = np.arange(t_start, t_end, 0.01)
                        da_int = interp1d(ts, dff, bounds_error=False, fill_value=0)(t_cc)
                        theta_env_int = interp1d(timestamps, dp['theta_env'], bounds_error=False, fill_value=0)(t_cc)
                        beta_env_int = interp1d(timestamps, dp['beta_env'], bounds_error=False, fill_value=0)(t_cc)
                        
                        from scipy.signal import correlate
                        # Cross-corr DA vs Theta Envelope
                        lags = correlate(da_int - np.mean(da_int), theta_env_int - np.mean(theta_env_int), mode='full')
                        lags = lags / (len(da_int) * np.std(da_int) * np.std(theta_env_int))
                        lag_t = np.linspace(-len(da_int)*0.01, len(da_int)*0.01, len(lags))
                        
                        mask = (lag_t >= -5) & (lag_t <= 5)
                        dp['xcorr_da_theta'] = lags[mask]
                        dp['xcorr_da_beta'] = correlate(da_int - np.mean(da_int), beta_env_int - np.mean(beta_env_int), mode='full')[mask] / (len(da_int) * np.std(da_int) * np.std(beta_env_int))
                        dp['xcorr_lags'] = lag_t[mask]
                        metrics['da_theta_max_xcorr'] = np.max(dp['xcorr_da_theta']) if sum(mask) > 0 else np.nan

                        # 3D (extended). Phase of LFP at DA transient peaks
                        da_thresh = np.nanmean(dff) + 2.0 * np.nanstd(dff)
                        from scipy.signal import find_peaks
                        dt_da = np.nanmedian(np.diff(ts)) if len(ts) > 2 else np.nan
                        min_peak_distance = int(np.clip(np.round(1.0 / dt_da), 1, 5000)) if np.isfinite(dt_da) and dt_da > 0 else 1

                        # Primary detector: z-scored DA peaks with prominence.
                        dff_std = np.nanstd(dff)
                        dff_z = (dff - np.nanmean(dff)) / (dff_std + 1e-12)
                        da_peaks_idx, _ = find_peaks(dff_z, height=1.5, prominence=0.5, distance=min_peak_distance)

                        # Fallback detector: less strict percentile threshold if too few peaks.
                        if len(da_peaks_idx) < 6:
                            alt_thresh = np.nanpercentile(dff, 90)
                            da_peaks_idx, _ = find_peaks(dff, height=alt_thresh, distance=min_peak_distance)

                        metrics['n_da_detected_peaks'] = int(len(da_peaks_idx))
                        dp['da_peak_phase_source'] = 'detected_peaks'

                        theta_phase_interp = interp1d(
                            timestamps,
                            np.unwrap(theta_phase),
                            bounds_error=False,
                            fill_value=np.nan
                        )
                        beta_phase_interp = interp1d(
                            timestamps,
                            np.unwrap(beta_phase),
                            bounds_error=False,
                            fill_value=np.nan
                        ) if 'beta_phase' in locals() else None

                        if len(da_peaks_idx) > 3:
                            da_peak_times = ts[da_peaks_idx]
                            theta_peak_phase = theta_phase_interp(da_peak_times)
                            theta_peak_phase = theta_peak_phase[np.isfinite(theta_peak_phase)]
                            dp['da_peak_theta_phases'] = np.angle(np.exp(1j * theta_peak_phase))

                            if beta_phase_interp is not None:
                                beta_peak_phase = beta_phase_interp(da_peak_times)
                                beta_peak_phase = beta_peak_phase[np.isfinite(beta_peak_phase)]
                                dp['da_peak_beta_phases'] = np.angle(np.exp(1j * beta_peak_phase))
                            else:
                                dp['da_peak_beta_phases'] = []
                        else:
                            # Final fallback: use sparse high-DA moments to avoid an empty panel.
                            high_da_thr = np.nanpercentile(dff, 90)
                            high_da_idx = np.where(dff >= high_da_thr)[0]
                            if len(high_da_idx) > 0:
                                keep = np.insert(np.diff(high_da_idx) >= min_peak_distance, 0, True)
                                high_da_idx = high_da_idx[keep]
                                high_da_times = ts[high_da_idx]
                                theta_peak_phase = theta_phase_interp(high_da_times)
                                theta_peak_phase = theta_peak_phase[np.isfinite(theta_peak_phase)]
                                dp['da_peak_theta_phases'] = np.angle(np.exp(1j * theta_peak_phase))

                                if beta_phase_interp is not None:
                                    beta_peak_phase = beta_phase_interp(high_da_times)
                                    beta_peak_phase = beta_peak_phase[np.isfinite(beta_peak_phase)]
                                    dp['da_peak_beta_phases'] = np.angle(np.exp(1j * beta_peak_phase))
                                else:
                                    dp['da_peak_beta_phases'] = []
                                dp['da_peak_phase_source'] = 'high_da_windows'
                                metrics['n_da_detected_peaks'] = int(len(high_da_idx))

                        # Mean vector length = phase concentration
                        if 'da_peak_theta_phases' in dp and len(dp['da_peak_theta_phases']) > 3:
                            mvl_da = np.abs(np.mean(np.exp(1j * dp['da_peak_theta_phases'])))
                            metrics['da_peak_theta_phase_mvl'] = float(mvl_da)
                            metrics['da_peak_theta_preferred_phase'] = float(np.angle(np.mean(np.exp(1j * dp['da_peak_theta_phases']))))

                        # 3D (extended). DA modulation of PAC strength
                        # Split recording into epochs with high vs low DA, compute PAC in each
                        try:
                            da_env_int = np.abs(interp1d(ts, dff, bounds_error=False, fill_value=np.nan)(timestamps))
                            da_med = np.nanmedian(da_env_int)
                            high_da_mask = da_env_int > da_med
                            low_da_mask = da_env_int <= da_med
                            if high_da_mask.sum() > fs and low_da_mask.sum() > fs:
                                pac_high = _compute_comodulogram(trace[high_da_mask], fs, phase_bands, amp_bands)
                                pac_low = _compute_comodulogram(trace[low_da_mask], fs, phase_bands, amp_bands)
                                dp['da_pac_high'] = pac_high
                                dp['da_pac_low'] = pac_low
                                dp['da_pac_diff'] = pac_high - pac_low  # PAC modulation by DA
                                # Key metrics: theta-gamma PAC modulation by DA
                                metrics['da_modulation_theta_gamma_pac'] = float(dp['da_pac_diff'][1, 1])
                                metrics['da_modulation_beta_gamma_pac'] = float(dp['da_pac_diff'][1, 3])
                        except Exception as e_pac:
                            logger.debug(f"DA-PAC modulation failed: {e_pac}")

                except Exception as e_da:
                    logger.warning(f"DA-LFP coupling failed: {e_da}")
                    
    except Exception as e:
        logger.error(f"Tier 3 LFP failed: {e}")
        
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Tier 3: LFP Oscillatory Dynamics | {paths.mouse_id}", fontsize=20, fontweight='bold', y=0.98)
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # 1. PSD Move vs Rest (with 1/f aperiodic fit)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'psd_m' in dp and len(dp['f']) > 0:
        mask = (dp['f'] >= 1) & (dp['f'] <= 100)
        f_sub = dp['f'][mask]
        
        # Function to fit 1/f (log-log linear regression)
        def fit_aperiodic(f, psd):
            log_f = np.log10(f)
            log_p = 10 * np.log10(psd)
            # Simple robust linear fit
            slope, intercept, _, _, _ = scipy.stats.linregress(log_f, log_p)
            fitted_log_p = intercept + slope * log_f
            return log_p, fitted_log_p, slope
            
        if len(dp['psd_m']) > 0:
            log_p_m, fit_m, slope_m = fit_aperiodic(f_sub, dp['psd_m'][mask])
            ax1.plot(f_sub, log_p_m - fit_m, label=f'Move (Flattened, a={-slope_m:.2f})', color='seagreen', lw=2)
            
        if len(dp['psd_r']) > 0:
            log_p_r, fit_r, slope_r = fit_aperiodic(f_sub, dp['psd_r'][mask])
            ax1.plot(f_sub, log_p_r - fit_r, label=f'Rest (Flattened, a={-slope_r:.2f})', color='gray', lw=2)
            
        ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Periodic Power (Orig dB - 1/f Fit)")
        ax1.set_title("Flattened PSD (Aperiodic Subtracted)")
        ax1.legend()
        metrics['lfp_aperiodic_exponent_move'] = -slope_m if len(dp['psd_m']) > 0 else np.nan
        
    # 2. Movement Spectrogram
    ax2 = fig.add_subplot(gs[0, 1])
    if 'sg_mean' in dp:
        mask = (dp['sg_f'] >= 1) & (dp['sg_f'] <= 100)
        im = ax2.pcolormesh(dp['sg_t'], dp['sg_f'][mask], 10*np.log10(dp['sg_mean'][mask, :] + 1e-10), cmap='viridis', shading='gouraud')
        ax2.axvline(0, color='white', linestyle='--')
        ax2.set_xlabel("Time from Move Onset (s)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_title("Movement Onset Spectrogram")
        fig.colorbar(im, ax=ax2, label='dB')
        
    # 3. LFP Envelopes vs Speed
    ax3 = fig.add_subplot(gs[0, 2])
    if 'theta_env' in dp:
        snip = min(60000, len(dp['theta_env']))
        fs_plot = dp.get('lfp_fs', 1250.0)
        x = np.arange(snip) / fs_plot

        theta_env = dp['theta_env'][:snip]
        gamma_env = dp['gamma_env'][:snip]
        theta_scale = np.nanmax(theta_env)
        gamma_scale = np.nanmax(gamma_env)
        theta_norm = theta_env / theta_scale if np.isfinite(theta_scale) and theta_scale > 0 else theta_env
        gamma_norm = gamma_env / gamma_scale if np.isfinite(gamma_scale) and gamma_scale > 0 else gamma_env

        ax3.plot(x, theta_norm, label='Theta Env', color='blue', alpha=0.7)
        ax3.plot(x, gamma_norm, label='Gamma Env', color='red', alpha=0.7)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Normalized Power")
        ax3.set_title("Theta & Gamma Envelopes (1 min snippet)")
        ax3.legend()
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, "Envelope data\nnot available", ha='center', va='center', fontsize=10)
        ax3.set_title("Theta & Gamma Envelopes")
        
    # 4. Comodulogram Move
    ax4 = fig.add_subplot(gs[1, 0])
    phase_labels = ['Delta(2-4)', 'Theta(4-8)', 'Alpha(8-12)', 'Beta(13-30)']
    amp_labels = ['LowG(30-50)', 'MidG(50-80)', 'HighG(80-120)', 'UHF(120-160)']
    if 'comod_move' in dp:
        im4 = ax4.imshow(dp['comod_move'], origin='lower', cmap='plasma', aspect='auto')
        ax4.set_xticks(np.arange(4)); ax4.set_xticklabels(phase_labels, rotation=45)
        ax4.set_yticks(np.arange(4)); ax4.set_yticklabels(amp_labels)
        ax4.set_title("PAC Comodulogram (Move)")
        fig.colorbar(im4, ax=ax4, label='MI (Surrogate-norm)')
        
    # 5. Comodulogram Rest
    ax5 = fig.add_subplot(gs[1, 1])
    if 'comod_rest' in dp:
        im5 = ax5.imshow(dp['comod_rest'], origin='lower', cmap='plasma', aspect='auto')
        ax5.set_xticks(np.arange(4)); ax5.set_xticklabels(phase_labels, rotation=45)
        ax5.set_yticks(np.arange(4)); ax5.set_yticklabels(amp_labels)
        ax5.set_title("PAC Comodulogram (Rest)")
        fig.colorbar(im5, ax=ax5, label='MI')
        
    # 6. Spike-Phase Locking MVL Dist
    ax6 = fig.add_subplot(gs[1, 2])
    if 'mvl_move' in dp and len(dp['mvl_move']) > 0:
        ax6.hist(dp['mvl_move'], bins=20, alpha=0.6, color='seagreen', label='Move')
    if 'mvl_rest' in dp and len(dp['mvl_rest']) > 0:
        ax6.hist(dp['mvl_rest'], bins=20, alpha=0.6, color='gray', label='Rest')
    if 'mvl_move' in dp or 'mvl_rest' in dp:
        ax6.set_xlabel("Theta Phase MVL (Mean Vector Length)")
        ax6.set_ylabel("Units")
        ax6.set_title("Spike-Theta Phase Locking")
        ax6.legend()
        
    # 7. DA-Theta Cross-Corr
    ax7 = fig.add_subplot(gs[2, 0])
    if 'xcorr_da_theta' in dp:
        ax7.plot(dp['xcorr_lags'], dp['xcorr_da_theta'], color='blue', lw=2, label='Theta Env')
        ax7.plot(dp['xcorr_lags'], dp['xcorr_da_beta'], color='orange', lw=2, label='Beta Env')
        ax7.axvline(0, color='k', linestyle='--')
        ax7.set_xlabel("Lag (s) DA vs LFP Env")
        ax7.set_title("Dopamine - LFP Envelope Cross-Corr")
        ax7.legend()
    else:
        ax7.axis('off')
        ax7.text(0.5, 0.5, "DA-LFP cross-correlation\nnot available", ha='center', va='center', fontsize=10)
        ax7.set_title("Dopamine - LFP Envelope Cross-Corr")
        
    # 8. DA-PAC Modulation (High DA - Low DA comodulogram difference)
    ax8 = fig.add_subplot(gs[2, 1])
    if 'da_pac_diff' in dp:
        im8 = ax8.imshow(dp['da_pac_diff'], origin='lower', cmap='RdBu_r',
                         vmin=-np.nanmax(np.abs(dp['da_pac_diff'])),
                         vmax=np.nanmax(np.abs(dp['da_pac_diff'])),
                         aspect='auto')
        ax8.set_xticks(np.arange(4)); ax8.set_xticklabels(phase_labels, rotation=45, fontsize=7)
        ax8.set_yticks(np.arange(4)); ax8.set_yticklabels(amp_labels, fontsize=7)
        ax8.set_title("DA Modulation of PAC\n(High DA − Low DA)")
        fig.colorbar(im8, ax=ax8, label='ΔPAC')
    else:
        ax8.axis('off')
        ax8.text(0.5, 0.5, "DA-PAC data\nnot available", ha='center', va='center', fontsize=10)
        ax8.set_title("DA Modulation of PAC")

    # 9. Band-Power vs Speed
    ax9 = fig.add_subplot(gs[3, 0])
    if 'band_speed_bins' in dp:
        bc = dp['band_speed_bins']
        colors_bps = {'theta': 'blue', 'beta': 'orange', 'gamma': 'red'}
        for band_name, col in colors_bps.items():
            key = f'{band_name}_power_vs_speed'
            if key in dp:
                rho = metrics.get(f'{band_name}_power_speed_rho', np.nan)
                ax9.plot(bc, dp[key] / (np.nanmax(dp[key]) + 1e-10),
                         color=col, lw=2, label=f'{band_name} (rho={rho:.2f})')
        ax9.set_xlabel("Speed (cm/s)")
        ax9.set_ylabel("Norm. Band Power")
        ax9.set_title("Band Power vs Speed (3A)")
        ax9.legend(fontsize=8)
    else:
        ax9.axis('off')
        ax9.text(0.5, 0.5, "Band-speed data\nnot available", ha='center', va='center', fontsize=10)
        ax9.set_title("Band Power vs Speed (3A)")

    # 10. Multi-band PPC (Pairwise Phase Consistency)
    ax10 = fig.add_subplot(gs[3, 1])
    ppc_bands = ['theta', 'beta', 'gamma']
    ppc_vals = [metrics.get(f'mean_{b}_ppc_move', np.nan) for b in ppc_bands]
    mvl_vals = [metrics.get(f'mean_{b}_mvl_move', np.nan) for b in ppc_bands]
    x_pos = np.arange(len(ppc_bands))
    ax10.bar(x_pos - 0.2, ppc_vals, width=0.35, label='PPC', color='steelblue', alpha=0.8)
    ax10.bar(x_pos + 0.2, mvl_vals, width=0.35, label='MVL', color='tomato', alpha=0.8)
    ax10.set_xticks(x_pos); ax10.set_xticklabels([b.capitalize() for b in ppc_bands])
    ax10.set_ylabel("Phase Locking Strength")
    ax10.set_title("Multi-band Phase Locking (Move)\nPPC vs MVL (3C)")
    ax10.legend(fontsize=8)

    # 11. Phase of LFP at DA transient peaks (rose plot)
    ax11 = fig.add_subplot(gs[3, 2], polar=True)
    if 'da_peak_theta_phases' in dp and len(dp['da_peak_theta_phases']) > 3:
        phases_rose = dp['da_peak_theta_phases']
        n_bins_rose = 16
        bins_rose = np.linspace(-np.pi, np.pi, n_bins_rose + 1)
        counts_rose, _ = np.histogram(phases_rose, bins=bins_rose)
        theta_rose = (bins_rose[:-1] + bins_rose[1:]) / 2
        ax11.bar(theta_rose, counts_rose, width=2*np.pi/n_bins_rose, alpha=0.7,
                 color='darkviolet', edgecolor='k', linewidth=0.5)
        pref = metrics.get('da_peak_theta_preferred_phase', np.nan)
        if not np.isnan(pref):
            ax11.axvline(pref, color='red', lw=2, label=f'Pref={pref:.2f}rad')
        src = dp.get('da_peak_phase_source', 'detected_peaks')
        n_da = len(phases_rose)
        ax11.set_title(
            f"Theta Phase at DA Peaks\n(n={n_da}, MVL={metrics.get('da_peak_theta_phase_mvl', 0):.3f}, src={src})",
            pad=10
        )
    else:
        ax11.axis('off')
        n_da = metrics.get('n_da_detected_peaks', 0)
        ax11.text(0.5, 0.5, f"Insufficient DA peaks\n(n={n_da})", ha='center', va='center',
                  transform=ax11.transAxes, fontsize=10)
        ax11.set_title("Theta Phase at DA Peaks")

    # 12. Summary Panel
    ax12 = fig.add_subplot(gs[2, 2])
    ax12.axis('off')
    col1 = (
        f"--- LFP Oscillatory Dynamics ---\n"
        f"Theta/Gamma PAC (Move): {metrics.get('pac_theta_gamma_move', 0):.4f}\n"
        f"Beta/Gamma PAC (Move):  {metrics.get('pac_beta_gamma_move', 0):.4f}\n"
        f"\n--- Phase Locking (Move) ---\n"
        f"Theta MVL/PPC: {metrics.get('mean_theta_mvl_move', 0):.3f} / {metrics.get('mean_theta_ppc_move', 0):.3f}\n"
        f"Beta  MVL/PPC: {metrics.get('mean_beta_mvl_move', 0):.3f} / {metrics.get('mean_beta_ppc_move', 0):.3f}\n"
        f"Gamma MVL/PPC: {metrics.get('mean_gamma_mvl_move', 0):.3f} / {metrics.get('mean_gamma_ppc_move', 0):.3f}\n"
        f"\n--- DA-LFP Coupling ---\n"
        f"DA-Theta XCorr: {metrics.get('da_theta_max_xcorr', 0):.3f}\n"
        f"DA peak count: {metrics.get('n_da_detected_peaks', 0)}\n"
        f"DA Peak Phase MVL: {metrics.get('da_peak_theta_phase_mvl', 0):.3f}\n"
        f"DA mod Theta-Gamma PAC: {metrics.get('da_modulation_theta_gamma_pac', 0):.4f}\n"
        f"\n--- Band Power vs Speed ---\n"
        f"Theta rho: {metrics.get('theta_power_speed_rho', 0):.3f}  "
        f"Beta rho: {metrics.get('beta_power_speed_rho', 0):.3f}  "
        f"Gamma rho: {metrics.get('gamma_power_speed_rho', 0):.3f}\n"
    )
    ax12.text(0.05, 0.95, col1, fontsize=9, va='top', ha='left', family='monospace',
              transform=ax12.transAxes)
    
    out_dir = paths.base_path / "post_analysis" / "tier3_lfp"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{paths.mouse_id}_{paths.date_str}_tier3_dashboard.png", dpi=300)
    plt.close(fig)
    
    df = pd.DataFrame([metrics])
    csv_file = out_dir / "tier3_lfp_summary.csv"
    import os
    mode = 'a' if csv_file.exists() else 'w'
    header = not csv_file.exists()
    df.to_csv(csv_file, mode=mode, header=header, index=False)
    logger.info(f"Tier 3 completed. Saved to {csv_file}")


def _blocked_time_folds(n_samples, n_splits=5, gap_frames=0, min_test_size=20):
    """Create leakage-aware blocked CV folds for time-series data."""
    if n_samples < max(min_test_size * 2, n_splits):
        return []

    all_idx = np.arange(n_samples)
    blocks = np.array_split(all_idx, n_splits)
    folds = []

    for block in blocks:
        if len(block) < min_test_size:
            continue

        test_start, test_end = int(block[0]), int(block[-1])
        train_mask = np.ones(n_samples, dtype=bool)
        left = max(0, test_start - gap_frames)
        right = min(n_samples, test_end + gap_frames + 1)
        train_mask[left:right] = False

        train_idx = all_idx[train_mask]
        test_idx = block
        if len(train_idx) < min_test_size:
            continue
        folds.append((train_idx, test_idx))

    return folds


def _summarize_scores(scores):
    """Return mean/std/95% CI for an array-like score vector."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_s = float(np.mean(s))
    std_s = float(np.std(s))
    ci = 1.96 * std_s / np.sqrt(max(1, len(s)))
    return mean_s, std_s, float(mean_s - ci), float(mean_s + ci)


def _circular_shift(y, shift):
    if len(y) == 0:
        return y
    shift = int(shift) % len(y)
    if shift == 0:
        shift = 1
    return np.roll(y, shift)


def _decode_regression_blocked(X, y, n_splits=5, gap_frames=0):
    """Blocked-CV regression decode with Ridge and OOF predictions."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    y = np.asarray(y).flatten()
    valid = np.isfinite(y)
    Xv = np.asarray(X)[valid]
    yv = y[valid]
    folds = _blocked_time_folds(len(yv), n_splits=n_splits, gap_frames=gap_frames)
    if not folds:
        return {
            'scores': np.array([]),
            'mean': np.nan,
            'std': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'oof_true': np.array([]),
            'oof_pred': np.array([]),
        }

    scores = []
    oof_pred = np.full(len(yv), np.nan, dtype=float)
    oof_true = yv.copy()

    for tr_idx, te_idx in folds:
        model = Ridge(alpha=1.0)
        model.fit(Xv[tr_idx], yv[tr_idx])
        pred = model.predict(Xv[te_idx])
        oof_pred[te_idx] = pred
        if len(te_idx) > 1:
            scores.append(r2_score(yv[te_idx], pred))

    mean_s, std_s, ci_l, ci_h = _summarize_scores(scores)
    return {
        'scores': np.asarray(scores, dtype=float),
        'mean': mean_s,
        'std': std_s,
        'ci_low': ci_l,
        'ci_high': ci_h,
        'oof_true': oof_true[np.isfinite(oof_pred)],
        'oof_pred': oof_pred[np.isfinite(oof_pred)],
    }


def _decode_regression_null(X, y, n_splits=5, gap_frames=0, n_shuffles=200, rng=None):
    """Circular-shift null model for blocked-CV regression decoding."""
    if rng is None:
        rng = np.random.default_rng(42)
    y = np.asarray(y).flatten()
    n = len(y)
    if n < 20:
        return np.array([])

    null_scores = []
    for _ in range(n_shuffles):
        shift = rng.integers(1, max(2, n - 1))
        y_sh = _circular_shift(y, shift)
        res = _decode_regression_blocked(X, y_sh, n_splits=n_splits, gap_frames=gap_frames)
        null_scores.append(res['mean'])
    return np.asarray(null_scores, dtype=float)


def _decode_classification_blocked(X, y, n_splits=5, gap_frames=0):
    """Blocked-CV classification decode with balanced accuracy/macro-F1 and OOF preds."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.svm import LinearSVC

    y = np.asarray(y).flatten()
    valid = ~pd.isnull(y)
    Xv = np.asarray(X)[valid]
    yv = y[valid]
    if len(np.unique(yv)) < 2:
        return {
            'bal_scores': np.array([]),
            'f1_scores': np.array([]),
            'bal_mean': np.nan,
            'bal_std': np.nan,
            'bal_ci_low': np.nan,
            'bal_ci_high': np.nan,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'f1_ci_low': np.nan,
            'f1_ci_high': np.nan,
            'oof_true': np.array([]),
            'oof_pred': np.array([]),
        }

    folds = _blocked_time_folds(len(yv), n_splits=n_splits, gap_frames=gap_frames)
    if not folds:
        return {
            'bal_scores': np.array([]),
            'f1_scores': np.array([]),
            'bal_mean': np.nan,
            'bal_std': np.nan,
            'bal_ci_low': np.nan,
            'bal_ci_high': np.nan,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'f1_ci_low': np.nan,
            'f1_ci_high': np.nan,
            'oof_true': np.array([]),
            'oof_pred': np.array([]),
        }

    bal_scores = []
    f1_scores = []
    oof_pred = np.full(len(yv), np.nan, dtype=float)
    y_as_float = yv.astype(float)

    for tr_idx, te_idx in folds:
        y_tr = yv[tr_idx]
        y_te = yv[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = LinearSVC(class_weight='balanced', max_iter=5000)
        clf.fit(Xv[tr_idx], y_tr)
        pred = clf.predict(Xv[te_idx])
        oof_pred[te_idx] = pred.astype(float)
        bal_scores.append(balanced_accuracy_score(y_te, pred))
        f1_scores.append(f1_score(y_te, pred, average='macro'))

    bal_mean, bal_std, bal_ci_l, bal_ci_h = _summarize_scores(bal_scores)
    f1_mean, f1_std, f1_ci_l, f1_ci_h = _summarize_scores(f1_scores)

    valid_oof = np.isfinite(oof_pred)
    return {
        'bal_scores': np.asarray(bal_scores, dtype=float),
        'f1_scores': np.asarray(f1_scores, dtype=float),
        'bal_mean': bal_mean,
        'bal_std': bal_std,
        'bal_ci_low': bal_ci_l,
        'bal_ci_high': bal_ci_h,
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'f1_ci_low': f1_ci_l,
        'f1_ci_high': f1_ci_h,
        'oof_true': y_as_float[valid_oof],
        'oof_pred': oof_pred[valid_oof],
    }


def _decode_classification_null(X, y, n_splits=5, gap_frames=0, n_shuffles=200, rng=None):
    """Circular-shift null model for blocked-CV classification decoding."""
    if rng is None:
        rng = np.random.default_rng(42)
    y = np.asarray(y).flatten()
    n = len(y)
    if n < 20:
        return np.array([])

    null_bal = []
    for _ in range(n_shuffles):
        shift = rng.integers(1, max(2, n - 1))
        y_sh = _circular_shift(y, shift)
        res = _decode_classification_blocked(X, y_sh, n_splits=n_splits, gap_frames=gap_frames)
        null_bal.append(res['bal_mean'])
    return np.asarray(null_bal, dtype=float)

def analyze_of_tier4_population(paths: DataPaths):
    from logging import getLogger
    logger = getLogger(__name__)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy.stats
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA, NMF
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from data_loader import SpikeDataLoader, DLCDataLoader
    
    logger.info("Running Tier 4: Comprehensive Population-Level Analysis...")
    metrics = {'mouse': paths.mouse_id, 'date': paths.date_str, 'genotype': paths.genotype}
    dp = {}

    import os
    fast_mode = os.environ.get('TIER4_FAST_MODE', '1') != '0'

    cv_folds = 3 if fast_mode else 5
    null_cv_folds = 2 if fast_mode else 3
    cv_gap_frames = 30
    n_null_shuffles_main = 30 if fast_mode else 200
    n_null_shuffles_lag = 8 if fast_mode else 80
    rng = np.random.default_rng(42)
    metrics['decoding_fast_mode'] = int(fast_mode)
    metrics['decoding_cv_folds'] = cv_folds
    metrics['decoding_null_cv_folds'] = null_cv_folds
    metrics['decoding_cv_gap_frames'] = cv_gap_frames
    metrics['decoding_null_shuffles_main'] = n_null_shuffles_main
    metrics['decoding_null_shuffles_lag'] = n_null_shuffles_lag
    
    try:
        spike_loader = SpikeDataLoader(paths.base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        unique_clusters = spike_data['unique_clusters']
        spike_times = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        
        dlc_loader = DLCDataLoader(paths.base_path)
        if paths.dlc_h5 and paths.dlc_h5.exists():
            df_dlc = dlc_loader.load(paths.dlc_h5)
            velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, strobe_path=paths.strobe_seconds)

            # Load Snout position + heading for decoding (4C)
            try:
                scorer = df_dlc.columns.get_level_values(0).unique()[0]
                x_pos = df_dlc[(scorer, 'Snout', 'x')].values
                y_pos = df_dlc[(scorer, 'Snout', 'y')].values
                # Heading: atan2 of velocity vector
                dx = np.diff(x_pos, prepend=x_pos[0])
                dy = np.diff(y_pos, prepend=y_pos[0])
                heading = np.arctan2(dy, dx)
            except Exception:
                x_pos = np.full(len(velocity), np.nan)
                y_pos = np.full(len(velocity), np.nan)
                heading = np.full(len(velocity), np.nan)

            from analyses_openfield import _compute_vame_metrics
            v_met, labels = _compute_vame_metrics(paths)

            if labels is not None:
                min_len = min(len(velocity), len(labels))
                velocity = velocity[:min_len]
                velocity_times = velocity_times[:min_len]
                labels = labels[:min_len]
                x_pos = x_pos[:min_len]
                y_pos = y_pos[:min_len]
                heading = heading[:min_len]
            else:
                labels = np.zeros(len(velocity))
                
            # Bin spikes to match velocity_times (~60Hz) -> 16.6ms
            dt = np.median(np.diff(velocity_times)) if len(velocity_times) > 1 else 1/60.0
            dt = max(dt, 0.001)
            bins = np.append(velocity_times, velocity_times[-1] + dt)
            
            pop_rate = []
            for cid in unique_clusters:
                spikes = spike_times[spike_clusters == cid]
                counts, _ = np.histogram(spikes, bins=bins)
                fr = counts / dt
                fr_smooth = gaussian_filter1d(fr, sigma=int(0.1/dt)) # 100ms smoothing
                pop_rate.append(fr_smooth)
                
            pop_mat = np.array(pop_rate).T # shape: (time, units)
            valid_time = len(velocity)
            if pop_mat.shape[0] > valid_time: pop_mat = pop_mat[:valid_time]
            
            if pop_mat.shape[1] > 5 and pop_mat.shape[0] > 100:
                # Z-score pop_mat
                pop_mat_z = (pop_mat - np.mean(pop_mat, axis=0)) / (np.std(pop_mat, axis=0) + 1e-8)
                
                # 4A. State Space (PCA & UMAP)
                pca = PCA(n_components=min(10, pop_mat.shape[1]))
                pca_proj = pca.fit_transform(pop_mat_z)
                dp['pca_proj'] = pca_proj
                dp['velocity'] = velocity
                dp['labels'] = labels

                # Use low-dimensional features for decoding to reduce runtime.
                n_decode_dims = min(8, pca_proj.shape[1])
                decode_mat = pca_proj[:, :n_decode_dims]
                metrics['decoding_feature_dims'] = int(n_decode_dims)
                
                try:
                    from sklearn.manifold import TSNE
                    idx_sub = np.arange(0, len(pop_mat_z), 5) # subside memory
                    dp['umap_proj'] = TSNE(n_components=2, perplexity=30).fit_transform(pop_mat_z[idx_sub])
                    dp['umap_labels'] = labels[idx_sub]
                except:
                    # Fallback to TSNE or just copy PCA
                    dp['umap_proj'] = pca_proj[0:len(pop_mat_z):5, :2]
                    dp['umap_labels'] = labels[0:len(pop_mat_z):5]
                
                # 4B. Dimensionality (Participation Ratio)
                def calc_pr(mat):
                    if len(mat) < 10: return np.nan
                    cov = np.cov(mat.T)
                    evals = np.linalg.eigvalsh(cov)
                    evals = evals[evals > 0]
                    if len(evals) == 0: return np.nan
                    return (np.sum(evals)**2) / np.sum(evals**2)
                
                is_move = velocity > 2.0
                pr_move = calc_pr(pop_mat_z[is_move])
                pr_rest = calc_pr(pop_mat_z[~is_move])
                dp['pr_move'] = pr_move
                dp['pr_rest'] = pr_rest
                metrics['pr_overall'] = calc_pr(pop_mat_z)
                metrics['pr_move'] = pr_move
                metrics['pr_rest'] = pr_rest
                
                # 4D/E. NMF Ensembles & Rastermap
                # NMF needs non-negative
                pop_mat_nn = pop_mat.copy()
                nmf = NMF(n_components=min(5, pop_mat.shape[1]), init='nndsvda', max_iter=200)
                W = nmf.fit_transform(pop_mat_nn) # Temporal activations
                H = nmf.components_ # Spatial weights
                dp['W'] = W
                
                # Sort neurons by their max NMF component
                neuron_assignment = np.argmax(H, axis=0)
                sort_idx = np.lexsort((np.max(H, axis=0), neuron_assignment))
                dp['sorted_pop'] = pop_mat_z[:, sort_idx]
                dp['neuron_group_bounds'] = np.cumsum(np.bincount(neuron_assignment))
                
                # Co-activation matrix (Correlation)
                corr_mat = np.corrcoef(pop_mat_z[:, sort_idx].T)
                dp['corr_mat'] = corr_mat
                
                # 4C. Decoding Speed (Continuous)
                sub_idx = np.arange(0, len(pop_mat_z), 5)
                X = decode_mat[sub_idx]
                y_spd = velocity[sub_idx]
                speed_res = _decode_regression_blocked(
                    X, y_spd, n_splits=cv_folds, gap_frames=cv_gap_frames
                )
                speed_null = _decode_regression_null(
                    X, y_spd, n_splits=null_cv_folds, gap_frames=cv_gap_frames,
                    n_shuffles=n_null_shuffles_main, rng=rng
                )
                metrics['decoding_speed_r2'] = speed_res['mean']
                metrics['decoding_speed_r2_std'] = speed_res['std']
                metrics['decoding_speed_r2_ci_low'] = speed_res['ci_low']
                metrics['decoding_speed_r2_ci_high'] = speed_res['ci_high']
                metrics['decoding_speed_r2_null_mean'] = float(np.nanmean(speed_null)) if len(speed_null) > 0 else np.nan
                metrics['decoding_speed_r2_null_std'] = float(np.nanstd(speed_null)) if len(speed_null) > 0 else np.nan
                metrics['decoding_speed_r2_delta_vs_null'] = (
                    float(speed_res['mean'] - np.nanmean(speed_null)) if len(speed_null) > 0 and np.isfinite(speed_res['mean']) else np.nan
                )
                metrics['decoding_speed_r2_p'] = (
                    float((1 + np.sum(speed_null >= speed_res['mean'])) / (1 + len(speed_null)))
                    if len(speed_null) > 0 and np.isfinite(speed_res['mean']) else np.nan
                )
                dp['speed_decode_true'] = speed_res['oof_true']
                dp['speed_decode_pred'] = speed_res['oof_pred']
                
                # Temporal Decoding Speed
                lags = [-15, -10, -5, 0, 5, 10, 15] # frames
                r2_lags = []
                r2_lags_null = []
                r2_lags_p = []
                for lag in lags:
                    if lag < 0:
                        y_sh = velocity[-lag:]
                        X_sh = decode_mat[:lag]
                    elif lag > 0:
                        y_sh = velocity[:-lag]
                        X_sh = decode_mat[lag:]
                    else:
                        y_sh = velocity
                        X_sh = decode_mat
                    X_sh_sub = X_sh[::5]
                    y_sh_sub = y_sh[::5]
                    if len(X_sh_sub) > 50:
                        lag_res = _decode_regression_blocked(
                            X_sh_sub, y_sh_sub, n_splits=cv_folds, gap_frames=cv_gap_frames
                        )
                        if fast_mode:
                            lag_null = np.array([])
                        else:
                            lag_null = _decode_regression_null(
                                X_sh_sub, y_sh_sub, n_splits=null_cv_folds, gap_frames=cv_gap_frames,
                                n_shuffles=n_null_shuffles_lag, rng=rng
                            )
                        r2_lags.append(lag_res['mean'])
                        r2_lags_null.append(float(np.nanmean(lag_null)) if len(lag_null) > 0 else np.nan)
                        if len(lag_null) > 0 and np.isfinite(lag_res['mean']):
                            r2_lags_p.append(float((1 + np.sum(lag_null >= lag_res['mean'])) / (1 + len(lag_null))))
                        else:
                            r2_lags_p.append(np.nan)
                    else:
                        r2_lags.append(np.nan)
                        r2_lags_null.append(np.nan)
                        r2_lags_p.append(np.nan)
                dp['lags'] = np.array(lags) * dt # in seconds
                dp['r2_lags'] = r2_lags
                dp['r2_lags_null'] = r2_lags_null
                dp['r2_lags_p'] = r2_lags_p

                if np.any(np.isfinite(r2_lags)):
                    best_idx_lag = int(np.nanargmax(r2_lags))
                    metrics['decoding_speed_best_lag_sec'] = float(dp['lags'][best_idx_lag])
                    metrics['decoding_speed_best_lag_r2'] = float(r2_lags[best_idx_lag])
                    metrics['decoding_speed_best_lag_null_mean'] = float(r2_lags_null[best_idx_lag]) if np.isfinite(r2_lags_null[best_idx_lag]) else np.nan
                    metrics['decoding_speed_best_lag_p'] = float(r2_lags_p[best_idx_lag]) if np.isfinite(r2_lags_p[best_idx_lag]) else np.nan
                
                # Motif Decoding (SVM)
                y_lbl = labels[sub_idx]
                valid_states, counts_y = np.unique(y_lbl, return_counts=True)
                valid_states = valid_states[counts_y >= 10]
                mask = np.isin(y_lbl, valid_states)
                if sum(mask) > 20 and len(valid_states) > 1:
                    X_m = X[mask]
                    y_m = y_lbl[mask]
                    motif_res = _decode_classification_blocked(
                        X_m, y_m, n_splits=cv_folds, gap_frames=cv_gap_frames
                    )
                    motif_null = _decode_classification_null(
                        X_m, y_m, n_splits=null_cv_folds, gap_frames=cv_gap_frames,
                        n_shuffles=n_null_shuffles_main, rng=rng
                    )

                    metrics['decoding_motif_acc'] = motif_res['bal_mean']
                    metrics['decoding_motif_bal_acc'] = motif_res['bal_mean']
                    metrics['decoding_motif_bal_acc_std'] = motif_res['bal_std']
                    metrics['decoding_motif_bal_acc_ci_low'] = motif_res['bal_ci_low']
                    metrics['decoding_motif_bal_acc_ci_high'] = motif_res['bal_ci_high']
                    metrics['decoding_motif_macro_f1'] = motif_res['f1_mean']
                    metrics['decoding_motif_macro_f1_std'] = motif_res['f1_std']
                    metrics['decoding_motif_null_bal_acc_mean'] = float(np.nanmean(motif_null)) if len(motif_null) > 0 else np.nan
                    metrics['decoding_motif_null_bal_acc_std'] = float(np.nanstd(motif_null)) if len(motif_null) > 0 else np.nan
                    metrics['decoding_motif_delta_vs_null'] = (
                        float(motif_res['bal_mean'] - np.nanmean(motif_null)) if len(motif_null) > 0 and np.isfinite(motif_res['bal_mean']) else np.nan
                    )
                    metrics['decoding_motif_p'] = (
                        float((1 + np.sum(motif_null >= motif_res['bal_mean'])) / (1 + len(motif_null)))
                        if len(motif_null) > 0 and np.isfinite(motif_res['bal_mean']) else np.nan
                    )

                    if len(motif_res['oof_true']) > 0:
                        dp['cm'] = confusion_matrix(motif_res['oof_true'], motif_res['oof_pred'], normalize='true')
                    dp['cm_states'] = valid_states

                # 4B (extended). Dimensionality during repetitive vs exploratory behavior
                unique_states_all, state_counts = np.unique(labels, return_counts=True)
                top3_states = unique_states_all[np.argsort(state_counts)[-3:]]
                is_repetitive = np.isin(labels, top3_states)
                pr_rep = calc_pr(pop_mat_z[is_repetitive]) if is_repetitive.sum() > 10 else np.nan
                pr_exp = calc_pr(pop_mat_z[~is_repetitive]) if (~is_repetitive).sum() > 10 else np.nan
                metrics['pr_repetitive'] = float(pr_rep) if not np.isnan(pr_rep) else np.nan
                metrics['pr_exploratory'] = float(pr_exp) if not np.isnan(pr_exp) else np.nan
                dp['pr_rep'] = pr_rep
                dp['pr_exp'] = pr_exp

                # Per-motif dimensionality for plotting
                pr_per_state = {}
                for s in unique_states_all:
                    mask_s = labels == s
                    if mask_s.sum() >= 20:
                        pr_per_state[int(s)] = float(calc_pr(pop_mat_z[mask_s]))
                dp['pr_per_state'] = pr_per_state

                # 4C (extended). Position & Heading Decoding
                x_sub = x_pos[sub_idx]
                y_sub = y_pos[sub_idx]
                h_sub = heading[sub_idx]
                for target_name, y_target in [('pos_x', x_sub), ('pos_y', y_sub), ('heading', h_sub)]:
                    valid = ~np.isnan(y_target)
                    if valid.sum() < 50:
                        metrics[f'decoding_{target_name}_r2'] = np.nan
                        metrics[f'decoding_{target_name}_r2_std'] = np.nan
                        metrics[f'decoding_{target_name}_r2_ci_low'] = np.nan
                        metrics[f'decoding_{target_name}_r2_ci_high'] = np.nan
                        metrics[f'decoding_{target_name}_r2_null_mean'] = np.nan
                        metrics[f'decoding_{target_name}_r2_p'] = np.nan
                        continue
                    X_v = X[valid]
                    y_v = y_target[valid]
                    reg_res = _decode_regression_blocked(
                        X_v, y_v, n_splits=cv_folds, gap_frames=cv_gap_frames
                    )
                    reg_null = _decode_regression_null(
                        X_v, y_v, n_splits=null_cv_folds, gap_frames=cv_gap_frames,
                        n_shuffles=n_null_shuffles_main, rng=rng
                    )
                    metrics[f'decoding_{target_name}_r2'] = reg_res['mean']
                    metrics[f'decoding_{target_name}_r2_std'] = reg_res['std']
                    metrics[f'decoding_{target_name}_r2_ci_low'] = reg_res['ci_low']
                    metrics[f'decoding_{target_name}_r2_ci_high'] = reg_res['ci_high']
                    metrics[f'decoding_{target_name}_r2_null_mean'] = float(np.nanmean(reg_null)) if len(reg_null) > 0 else np.nan
                    metrics[f'decoding_{target_name}_r2_null_std'] = float(np.nanstd(reg_null)) if len(reg_null) > 0 else np.nan
                    metrics[f'decoding_{target_name}_r2_delta_vs_null'] = (
                        float(reg_res['mean'] - np.nanmean(reg_null)) if len(reg_null) > 0 and np.isfinite(reg_res['mean']) else np.nan
                    )
                    metrics[f'decoding_{target_name}_r2_p'] = (
                        float((1 + np.sum(reg_null >= reg_res['mean'])) / (1 + len(reg_null)))
                        if len(reg_null) > 0 and np.isfinite(reg_res['mean']) else np.nan
                    )
                    if target_name == 'pos_x':
                        dp['pos_decode_true_x'] = reg_res['oof_true']
                        dp['pos_decode_pred_x'] = reg_res['oof_pred']
                    if target_name == 'pos_y':
                        dp['pos_decode_true_y'] = reg_res['oof_true']
                        dp['pos_decode_pred_y'] = reg_res['oof_pred']
                    if target_name == 'heading':
                        dp['heading_decode_true'] = reg_res['oof_true']
                        dp['heading_decode_pred'] = reg_res['oof_pred']

    except Exception as e:
        logger.error(f"Tier 4 failed: {e}", exc_info=True)
        
    fig = plt.figure(figsize=(24, 24))
    fig.suptitle(f"Tier 4: Population-Level Dimensionality & Decoding | {paths.mouse_id}", fontsize=22, fontweight='bold', y=0.98)
    gs = GridSpec(4, 4, figure=fig, hspace=0.38, wspace=0.32)
    
    # 1. Rastermap (Sorted)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'sorted_pop' in dp:
        snip = min(3000, len(dp['sorted_pop']))
        im = ax1.imshow(dp['sorted_pop'][:snip].T, aspect='auto', cmap='magma', vmin=-1, vmax=3, interpolation='nearest')
        for b in dp['neuron_group_bounds'][:-1]:
            ax1.axhline(b, color='w', lw=1, linestyle='--')
        ax1.set_title("NMF-Sorted Neural Raster (50s)")
        ax1.set_xlabel("Time (frames)")
        ax1.set_ylabel("Neuron #")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
    # 2. Ensemble Co-Activation
    ax2 = fig.add_subplot(gs[0, 1])
    if 'corr_mat' in dp:
        im2 = ax2.imshow(dp['corr_mat'], aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=1)
        for b in dp['neuron_group_bounds'][:-1]:
            ax2.axhline(b, color='k', lw=1, linestyle='--')
            ax2.axvline(b, color='k', lw=1, linestyle='--')
        ax2.set_title("Unit-to-Unit Correlation")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
    # 3. Participation Ratio
    ax3 = fig.add_subplot(gs[0, 2])
    if 'pr_move' in dp:
        ax3.bar(['Move', 'Rest'], [dp['pr_move'], dp['pr_rest']], color=['seagreen', 'gray'], alpha=0.8)
        ax3.set_ylabel("Participation Ratio (Effective Dims)")
        ax3.set_title("Neural Dimensionality")
        
    # 4. NMF Temporal Activations
    ax4 = fig.add_subplot(gs[0, 3])
    if 'W' in dp:
        snip = min(3000, len(dp['W']))
        W_z = (dp['W'][:snip] - np.mean(dp['W'][:snip], axis=0)) / (np.std(dp['W'][:snip], axis=0)+1e-8)
        time_x = np.arange(snip) / 60.0 # Approx secs
        for i in range(W_z.shape[1]):
            ax4.plot(time_x, W_z[:, i] + i*5, alpha=0.8, lw=1.5, label=f'Ens {i+1}')
        ax4.set_xlabel("Time (s)")
        ax4.set_title("Ensemble Temporal Activations")
        
    # 5. PCA Color Speed
    ax5 = fig.add_subplot(gs[1, 0])
    if 'pca_proj' in dp:
        sub = np.arange(0, len(dp['pca_proj']), 5)
        sc = ax5.scatter(dp['pca_proj'][sub, 0], dp['pca_proj'][sub, 1], c=dp['velocity'][sub], cmap='inferno', s=5, alpha=0.5)
        ax5.set_xlabel("PC 1")
        ax5.set_ylabel("PC 2")
        ax5.set_title("State Space (Color: Speed)")
        fig.colorbar(sc, ax=ax5, label='Speed')
        
    # 6. PCA Color Motif
    ax6 = fig.add_subplot(gs[1, 1])
    if 'pca_proj' in dp:
        sub = np.arange(0, len(dp['pca_proj']), 5)
        sc2 = ax6.scatter(dp['pca_proj'][sub, 0], dp['pca_proj'][sub, 1], c=dp['labels'][sub], cmap='tab20', s=5, alpha=0.5)
        ax6.set_xlabel("PC 1")
        ax6.set_ylabel("PC 2")
        ax6.set_title("State Space (Color: Motif)")
        
    # 7. Speed Decoding R2
    ax7 = fig.add_subplot(gs[1, 2])
    if 'decoding_speed_r2' in metrics:
        obs = metrics.get('decoding_speed_r2', np.nan)
        null_m = metrics.get('decoding_speed_r2_null_mean', np.nan)
        p_val = metrics.get('decoding_speed_r2_p', np.nan)
        vals = [obs, null_m]
        ax7.bar(['Observed', 'Null'], vals, color=['teal', 'gray'], alpha=0.8)
        ylim_top = np.nanmax(np.array(vals, dtype=float))
        ylim_top = 0.5 if not np.isfinite(ylim_top) else max(0.5, ylim_top * 1.25)
        ax7.set_ylim([min(0.0, np.nanmin(np.array(vals, dtype=float)) - 0.05), ylim_top])
        ax7.set_ylabel("Test R2 Score")
        ax7.set_title(f"Continuous Decoding (Blocked CV)\np={p_val:.3g}" if np.isfinite(p_val) else "Continuous Decoding (Blocked CV)")
        
    # 8. Temporal Lag Speed Decoding
    ax8 = fig.add_subplot(gs[1, 3])
    if 'r2_lags' in dp:
        ax8.plot(dp['lags'], dp['r2_lags'], 'o-', color='crimson', lw=2)
        if 'r2_lags_null' in dp:
            ax8.plot(dp['lags'], dp['r2_lags_null'], 'o--', color='gray', lw=1.5, label='Null mean')
        ax8.axvline(0, color='k', linestyle='--')
        ax8.set_xlabel("Neural Lag vs Kinematics (s)")
        ax8.set_ylabel("Test R2")
        ax8.set_title("Temporal Predictive Decoding")
        if 'r2_lags_null' in dp:
            ax8.legend(fontsize=8)
        
    # 9. t-SNE Color Motif
    ax9 = fig.add_subplot(gs[2, 0])
    if 'umap_proj' in dp:
        sc3 = ax9.scatter(dp['umap_proj'][:, 0], dp['umap_proj'][:, 1], c=dp['umap_labels'], cmap='tab20', s=5, alpha=0.5)
        ax9.set_xlabel("t-SNE 1")
        ax9.set_ylabel("t-SNE 2")
        ax9.set_title("Non-Linear Manifold (Color: Motif)")
        
    # 10. SVM Confusion Matrix
    ax10 = fig.add_subplot(gs[2, 1])
    if 'cm' in dp:
        sns.heatmap(dp['cm'], annot=False, cmap='Blues', ax=ax10, xticklabels=dp['cm_states'], yticklabels=dp['cm_states'])
        ax10.set_xlabel("Predicted")
        ax10.set_ylabel("True")
        ax10.set_title("Motif Decoding Confusion")
        
    # 11. Decoding Motif Acc
    ax11 = fig.add_subplot(gs[2, 2])
    if 'decoding_motif_acc' in metrics:
        obs = metrics.get('decoding_motif_acc', np.nan)
        null_m = metrics.get('decoding_motif_null_bal_acc_mean', np.nan)
        p_val = metrics.get('decoding_motif_p', np.nan)
        ax11.bar(['Observed', 'Null'], [obs, null_m], color=['indigo', 'gray'], alpha=0.8)
        ax11.set_ylim([0, 1.0])
        ax11.set_ylabel("Balanced Accuracy")
        ax11.set_title(f"Discrete Motif Decoding\np={p_val:.3g}" if np.isfinite(p_val) else "Discrete Motif Decoding")
        
    # 12. Summary Text
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    col1 = (
        f"--- Population Metrics ---\n"
        f"PR (All / Move / Rest): "
        f"{metrics.get('pr_overall', 0):.1f} / {metrics.get('pr_move', 0):.1f} / {metrics.get('pr_rest', 0):.1f}\n"
        f"PR (Repetitive / Exploratory): "
        f"{metrics.get('pr_repetitive', 0):.1f} / {metrics.get('pr_exploratory', 0):.1f}\n"
        f"\n--- Decode Config ---\n"
        f"Blocked CV folds/gap: {int(metrics.get('decoding_cv_folds', 0))}/{int(metrics.get('decoding_cv_gap_frames', 0))} frames\n"
        f"Null shuffles: {int(metrics.get('decoding_null_shuffles', 0))}\n"
        f"\n--- Decoding R² ---\n"
        f"Speed: {metrics.get('decoding_speed_r2', 0):.3f} (null {metrics.get('decoding_speed_r2_null_mean', 0):.3f}, p={metrics.get('decoding_speed_r2_p', np.nan):.3g})\n"
        f"Pos X:  {metrics.get('decoding_pos_x_r2', 0):.3f} (p={metrics.get('decoding_pos_x_r2_p', np.nan):.3g})\n"
        f"Pos Y:  {metrics.get('decoding_pos_y_r2', 0):.3f} (p={metrics.get('decoding_pos_y_r2_p', np.nan):.3g})\n"
        f"Heading:{metrics.get('decoding_heading_r2', 0):.3f} (p={metrics.get('decoding_heading_r2_p', np.nan):.3g})\n"
        f"Motif BA: {metrics.get('decoding_motif_bal_acc', 0):.3f} (null {metrics.get('decoding_motif_null_bal_acc_mean', 0):.3f}, p={metrics.get('decoding_motif_p', np.nan):.3g})\n"
    )
    ax12.text(0.05, 0.95, col1, fontsize=10, va='top', ha='left', family='monospace',
              transform=ax12.transAxes)

    # 13. Dimensionality per VAME Motif
    ax13 = fig.add_subplot(gs[3, 0])
    if 'pr_per_state' in dp and dp['pr_per_state']:
        states_sorted = sorted(dp['pr_per_state'].keys())
        pr_vals_s = [dp['pr_per_state'][s] for s in states_sorted]
        bar_colors = ['#E74C3C' if s in [int(k) for k in []] else '#3498DB'
                       for s in states_sorted]
        ax13.bar(range(len(states_sorted)), pr_vals_s, color='steelblue', alpha=0.8)
        ax13.axhline(dp.get('pr_rep', np.nan), color='tomato', linestyle='--', lw=1.5, label='PR Repetitive')
        ax13.axhline(dp.get('pr_exp', np.nan), color='seagreen', linestyle='--', lw=1.5, label='PR Exploratory')
        ax13.set_xticks(range(len(states_sorted)))
        ax13.set_xticklabels(states_sorted, fontsize=7, rotation=45)
        ax13.set_xlabel("VAME Motif State")
        ax13.set_ylabel("Participation Ratio")
        ax13.set_title("Neural Dimensionality per Motif (4B)")
        ax13.legend(fontsize=8)

    # 14. Position Decoding Scatter
    ax14 = fig.add_subplot(gs[3, 1])
    if 'pos_decode_true_x' in dp and 'pos_decode_pred_x' in dp:
        sc_pos = ax14.scatter(dp['pos_decode_true_x'], dp['pos_decode_pred_x'],
                              s=5, alpha=0.4, c='dodgerblue')
        lim = [min(np.nanmin(dp['pos_decode_true_x']), np.nanmin(dp['pos_decode_pred_x'])),
               max(np.nanmax(dp['pos_decode_true_x']), np.nanmax(dp['pos_decode_pred_x']))]
        ax14.plot(lim, lim, 'k--', lw=1)
        ax14.set_xlabel("True X (px)")
        ax14.set_ylabel("Decoded X (px)")
        ax14.set_title(
            f"Position Decoding X\nR²={metrics.get('decoding_pos_x_r2', 0):.3f}, p={metrics.get('decoding_pos_x_r2_p', np.nan):.3g}"
        )
    else:
        ax14.axis('off')
        ax14.text(0.5, 0.5, "Position data\nnot available", ha='center', va='center',
                  transform=ax14.transAxes)
        ax14.set_title("Position Decoding X (4C)")

    # 15. Heading Decoding Scatter
    ax15 = fig.add_subplot(gs[3, 2])
    if 'heading_decode_true' in dp and 'heading_decode_pred' in dp:
        ax15.scatter(dp['heading_decode_true'], dp['heading_decode_pred'],
                     s=5, alpha=0.4, c='darkorange')
        ax15.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=1)
        ax15.set_xlabel("True Heading (rad)")
        ax15.set_ylabel("Decoded Heading (rad)")
        ax15.set_title(
            f"Heading Decoding\nR²={metrics.get('decoding_heading_r2', 0):.3f}, p={metrics.get('decoding_heading_r2_p', np.nan):.3g}"
        )
    else:
        ax15.axis('off')
        ax15.text(0.5, 0.5, "Heading data\nnot available", ha='center', va='center',
                  transform=ax15.transAxes)
        ax15.set_title("Heading Decoding (4C)")

    # 16. Decoding Summary Bar (Speed / X / Y / Heading / Motif)
    ax16 = fig.add_subplot(gs[3, 3])
    dec_keys = ['decoding_speed_r2', 'decoding_pos_x_r2', 'decoding_pos_y_r2',
                'decoding_heading_r2']
    dec_labels = ['Speed', 'Pos X', 'Pos Y', 'Heading']
    dec_vals = [metrics.get(k, 0) or 0 for k in dec_keys]
    dec_colors = ['teal', 'dodgerblue', 'steelblue', 'darkorange']
    ax16.barh(range(len(dec_labels)), dec_vals, color=dec_colors, alpha=0.8)
    ax16.set_yticks(range(len(dec_labels)))
    ax16.set_yticklabels(dec_labels)
    ax16.set_xlabel("R² Score")
    ax16.set_xlim(0, max(1.0, max(dec_vals) * 1.2))
    ax16.set_title("All Continuous Decoding R²")
    
    out_dir = paths.base_path / "post_analysis" / "tier4_population"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{paths.mouse_id}_{paths.date_str}_tier4_dashboard.png", dpi=300)
    plt.close(fig)
    
    df = pd.DataFrame([metrics])
    csv_file = out_dir / "tier4_population_summary.csv"
    import os
    mode = 'a' if csv_file.exists() else 'w'
    header = not csv_file.exists()
    df.to_csv(csv_file, mode=mode, header=header, index=False)
    logger.info(f"Tier 4 completed. Saved to {csv_file}")

def analyze_of_tier5_modeling(paths: DataPaths):
    from logging import getLogger
    logger = getLogger(__name__)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy.stats
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.cluster import KMeans
    import seaborn as sns
    from data_loader import SpikeDataLoader, DLCDataLoader, PhotometryDataLoader
    from analyses_openfield import _compute_vame_metrics
    
    logger.info("Running Tier 5: Generative & Computational Modeling...")
    metrics = {'mouse': paths.mouse_id, 'date': paths.date_str, 'genotype': paths.genotype}
    dp = {}
    
    try:
        spike_loader = SpikeDataLoader(paths.base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        unique_clusters = spike_data['unique_clusters']
        spike_times = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        
        dlc_loader = DLCDataLoader(paths.base_path)
        if paths.dlc_h5 and paths.dlc_h5.exists():
            df_dlc = dlc_loader.load(paths.dlc_h5)
            velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, strobe_path=paths.strobe_seconds)
            
            v_met, labels = _compute_vame_metrics(paths)
            if labels is not None:
                min_len = min(len(velocity), len(labels))
                velocity = velocity[:min_len]
                velocity_times = velocity_times[:min_len]
                labels = labels[:min_len]
            else:
                labels = np.zeros(len(velocity))
                
            dt = np.median(np.diff(velocity_times)) if len(velocity_times) > 1 else 1/60.0
            dt = max(dt, 0.001)
            bins = np.append(velocity_times, velocity_times[-1] + dt)
            
            pop_rate = []
            for cid in unique_clusters:
                counts, _ = np.histogram(spike_times[spike_clusters == cid], bins=bins)
                pop_rate.append(gaussian_filter1d(counts / dt, sigma=int(0.1/dt)))
            pop_mat = np.array(pop_rate).T
            if pop_mat.shape[0] > len(velocity): pop_mat = pop_mat[:len(velocity)]
            
            pop_mat_z = (pop_mat - np.mean(pop_mat, axis=0)) / (np.std(pop_mat, axis=0) + 1e-8)
            pca = PCA(n_components=min(10, pop_mat.shape[1]))
            pop_pca = pca.fit_transform(pop_mat_z)
            
            accel = np.gradient(velocity)
            features = [velocity, accel]
            
            has_da = False
            if paths.tdt_dff and paths.tdt_dff.exists():
                try:
                    tdt_loader = PhotometryDataLoader(paths.base_path)
                    tdt_data = tdt_loader.load(paths.tdt_dff, paths.tdt_raw)
                    from scipy.interpolate import interp1d
                    f_int = interp1d(tdt_data['dff_timestamps'], tdt_data['dff_values'], bounds_error=False, fill_value=0)
                    da_interp = f_int(velocity_times)
                    features.append(da_interp)
                    has_da = True
                    dp['da'] = da_interp
                except:
                    has_da = False
                    
            X_glm = np.column_stack(features)
            
            # 5A. GLM Encoding Model
            r2_scores = []
            idx_sub = np.arange(0, len(X_glm), 5)
            X_sub = X_glm[idx_sub]
            for i in range(min(50, pop_mat_z.shape[1])):
                y_sub = pop_mat_z[idx_sub, i]
                X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.3, random_state=42)
                rg = RidgeCV()
                rg.fit(X_tr, y_tr)
                r2_scores.append(r2_score(y_te, rg.predict(X_te)))
            dp['glm_r2'] = r2_scores
            metrics['glm_mean_r2'] = np.mean(r2_scores) if r2_scores else np.nan
            
            # Weights
            rg_full = RidgeCV().fit(X_sub, pop_mat_z[idx_sub, :min(50, pop_mat_z.shape[1])])
            dp['glm_weights'] = np.mean(np.abs(rg_full.coef_), axis=0)
            dp['glm_names'] = ['Speed', 'Accel', 'Dopa'] if has_da else ['Speed', 'Accel']
            
            # 5B/C. Joint HMM / State Space
            try:
                from hmmlearn import hmm
                joint_space = np.column_stack([pop_pca[idx_sub, :3], velocity[idx_sub]])
                model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
                model.fit(joint_space)
                states = model.predict(joint_space)
                dp['hmm_trans'] = model.transmat_
                
                # Dwell times
                dwells = []
                curr_s = states[0]
                run_len = 1
                for s in states[1:]:
                    if s == curr_s: run_len += 1
                    else:
                        dwells.append(run_len)
                        curr_s = s
                        run_len = 1
                dp['dwells'] = np.array(dwells) * (dt * 5)
                metrics['mean_dwell_time'] = np.mean(dp['dwells'])
            except:
                # Fallback to KMeans markov
                joint_space = np.column_stack([pop_pca[idx_sub, :3], velocity[idx_sub]])
                km = KMeans(n_clusters=5)
                states = km.fit_predict(joint_space)
                trans = np.zeros((5,5))
                for idx in range(len(states)-1): trans[states[idx], states[idx+1]] += 1
                trans = trans / (np.sum(trans, axis=1, keepdims=True) + 1e-8)
                dp['hmm_trans'] = trans
                
                dwells = []
                curr_s = states[0]
                run_len = 1
                for s in states[1:]:
                    if s == curr_s: run_len += 1
                    else:
                        dwells.append(run_len)
                        curr_s = s
                        run_len = 1
                dp['dwells'] = np.array(dwells) * (dt * 5)
                metrics['mean_dwell_time'] = np.mean(dp['dwells'])
                
            # 5D. Granger / Information Asymmetry
            if has_da:
                from scipy.signal import correlate
                da_zs = (dp['da'] - np.mean(dp['da'])) / np.std(dp['da'])
                pc1_zs = pop_pca[:, 0]
                pop_speed_zs = (velocity - np.mean(velocity))/np.std(velocity)
                
                lags_da_pc1 = correlate(pc1_zs, da_zs, mode='full') / len(da_zs)
                lag_t = np.linspace(-len(da_zs)*dt, len(da_zs)*dt, len(lags_da_pc1))
                mask_cc = (lag_t >= -2) & (lag_t <= 2)
                dp['gc_da_pc1'] = lags_da_pc1[mask_cc]
                dp['gc_lags'] = lag_t[mask_cc]
                
                lags_pc1_spd = correlate(pop_speed_zs, pc1_zs, mode='full') / len(pc1_zs)
                dp['gc_pc1_spd'] = lags_pc1_spd[mask_cc]
            
            # 5E. Attractor Landscape Model
            try:
                import sys
                import os
                model_dir = os.path.join(os.path.dirname(__file__), 'models')
                if model_dir not in sys.path: sys.path.append(model_dir)
                from attractor_energy_landscape import AttractorEnergyLandscapeModel
                
                att_model = AttractorEnergyLandscapeModel(n_dimensions=pca.n_components_)
                move_onsets = velocity_times[np.where(np.diff((velocity > 2.0).astype(int)) == 1)[0] + 1]
                
                res = att_model.analyze_movement_onset(
                    neural_population_activity=pop_mat_z,
                    movement_onsets=move_onsets,
                    pre_onset_window_sec=0.5,
                    sampling_rate=1.0/dt
                )
                dp['att_pre_vel'] = res['pre_onset_velocities']
                dp['att_base_vel'] = res['baseline_velocities']
                dp['att_pre_var'] = res['pre_onset_variances']
                dp['att_base_var'] = res['baseline_variances']
                
                metrics['attractor_velocity_increase'] = np.mean(res['pre_onset_velocities']) - np.mean(res['baseline_velocities'])
            except Exception as e_att:
                logger.warning(f"Attractor model failed: {e_att}")
                
    except Exception as e:
        logger.error(f"Tier 5 failed: {e}")
        import traceback
        traceback.print_exc()

    # Plotting 9-panel dashboard
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"Tier 5: Computational Modeling & Generative Dynamics | {paths.mouse_id}", fontsize=20, fontweight='bold', y=0.95)
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. GLM R2 Dist
    ax1 = fig.add_subplot(gs[0, 0])
    if 'glm_r2' in dp:
        ax1.hist(dp['glm_r2'], bins=15, color='darkorange', alpha=0.8, edgecolor='k')
        ax1.axvline(np.mean(dp['glm_r2']), color='r', linestyle='--', label=f"Mean: {np.mean(dp['glm_r2']):.2f}")
        ax1.set_xlabel("GLM R2 Score")
        ax1.set_title("Single-Unit GLM Encoding Performance")
        ax1.legend()
        
    # 2. GLM Weights
    ax2 = fig.add_subplot(gs[0, 1])
    if 'glm_weights' in dp:
        ax2.bar(dp['glm_names'], dp['glm_weights'], color='teal', alpha=0.8)
        ax2.set_ylabel("Mean Abs Beta Weight")
        ax2.set_title("GLM Feature Importance")
        
    # 3. Joint States (PCA plot mock)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'hmm_trans' in dp:
        sns.heatmap(dp['hmm_trans'], cmap='magma', annot=True, fmt=".2f", ax=ax3)
        ax3.set_xlabel("To State")
        ax3.set_ylabel("From State")
        ax3.set_title("Joint HMM Transition Matrix")
        
    # 4. State Dwell Times
    ax4 = fig.add_subplot(gs[1, 0])
    if 'dwells' in dp and len(dp['dwells']) > 0:
        ax4.hist(dp['dwells'], bins=np.linspace(0, 5, 20), color='purple', alpha=0.7)
        ax4.set_xlabel("Dwell Time (s)")
        ax4.set_ylabel("Count")
        ax4.set_title("Latent State Stickiness")
        
    # 5. Granger DA -> PC1
    ax5 = fig.add_subplot(gs[1, 1])
    if 'gc_da_pc1' in dp:
        ax5.plot(dp['gc_lags'], dp['gc_da_pc1'], color='blue', lw=2)
        ax5.axvline(0, color='k', linestyle='--')
        ax5.set_xlabel("Lag (s) DA vs Neural PC1")
        ax5.set_title("Information Flow (DA -> Pop)")
        
    # 6. Granger PC1 -> Speed
    ax6 = fig.add_subplot(gs[1, 2])
    if 'gc_pc1_spd' in dp:
        ax6.plot(dp['gc_lags'], dp['gc_pc1_spd'], color='green', lw=2)
        ax6.axvline(0, color='k', linestyle='--')
        ax6.set_xlabel("Lag (s) Neural PC1 vs Speed")
        ax6.set_title("Information Flow (Pop -> Speed)")
        
    # 7. Attractor P-Vel
    ax7 = fig.add_subplot(gs[2, 0])
    if 'att_pre_vel' in dp and len(dp['att_pre_vel']) > 0:
        sns.kdeplot(dp['att_base_vel'], fill=True, color='gray', label='Quiescence', ax=ax7)
        sns.kdeplot(dp['att_pre_vel'], fill=True, color='red', label='Pre-Onset', ax=ax7)
        ax7.set_xlabel("Neural Trajectory Speed")
        ax7.set_title("Attractor Escapes (Pre-Move Velocity)")
        ax7.legend()
        
    # 8. Attractor P-Var
    ax8 = fig.add_subplot(gs[2, 1])
    if 'att_pre_var' in dp and len(dp['att_pre_var']) > 0:
        sns.kdeplot(dp['att_base_var'], fill=True, color='gray', label='Quiescence', ax=ax8)
        sns.kdeplot(dp['att_pre_var'], fill=True, color='firebrick', label='Pre-Onset', ax=ax8)
        ax8.set_xlabel("Neural State Variance")
        ax8.set_title("Metastable Instability (Pre-Move Variance)")
        ax8.legend()
        
    # 9. Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    col1 = (
        f"--- Computational Models ---\n"
        f"GLM Mean R2: {metrics.get('glm_mean_r2', 0):.3f}\n"
        f"Mean Dwell Time: {metrics.get('mean_dwell_time', 0):.2f}s\n"
        f"Pre-Move Att Vel Inc: {metrics.get('attractor_velocity_increase', 0):.3f}\n"
    )
    ax9.text(0.1, 0.5, col1, fontsize=12, va='center', ha='left', family='monospace')
    
    out_dir = paths.base_path / "post_analysis" / "tier5_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{paths.mouse_id}_{paths.date_str}_tier5_dashboard.png", dpi=300)
    plt.close(fig)
    
    df = pd.DataFrame([metrics])
    csv_file = out_dir / "tier5_modeling_summary.csv"
    import os
    mode = 'a' if csv_file.exists() else 'w'
    header = not csv_file.exists()
    df.to_csv(csv_file, mode=mode, header=header, index=False)
    logger.info(f"Tier 5 completed. Saved to {csv_file}")

###############################################################################
# Genotype Comparison (KO vs WT)
###############################################################################

def _collect_tier_csvs(base_path: Path):
    """Scan all post_analysis directories for tier summary CSVs.

    Returns a dict mapping tier name -> DataFrame (all sessions concatenated).
    """
    post_dirs = list(base_path.rglob("post_analysis"))
    tier_frames = {}

    for pd_dir in post_dirs:
        for csv_file in pd_dir.rglob("*_summary.csv"):
            tier_name = csv_file.stem  # e.g. "tier1_behavior_summary"
            try:
                df = pd.read_csv(csv_file)
                if tier_name in tier_frames:
                    tier_frames[tier_name] = pd.concat(
                        [tier_frames[tier_name], df], ignore_index=True
                    )
                else:
                    tier_frames[tier_name] = df.copy()
            except Exception as e:
                logger.warning(f"Could not read {csv_file}: {e}")

    # De-duplicate (same mouse+date row appended twice)
    for name, df in tier_frames.items():
        if {'mouse', 'date'}.issubset(df.columns):
            tier_frames[name] = df.drop_duplicates(
                subset=['mouse', 'date'], keep='last'
            )

    return tier_frames


def _run_genotype_tests(df, metric_cols, group_col='genotype'):
    """Run Mann-Whitney U tests for each metric between genotype groups.

    Returns a DataFrame with columns:
        metric, U_stat, p_value, effect_size_r, mean_WT, mean_KO,
        median_WT, median_KO, n_WT, n_KO, p_fdr
    """
    from scipy.stats import mannwhitneyu

    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        logger.warning(f"Need at least 2 genotype groups, found: {groups}")
        return pd.DataFrame()

    # Use first two groups alphabetically so KO < WT
    groups = sorted(groups)
    g1_label, g2_label = groups[0], groups[1]
    g1 = df[df[group_col] == g1_label]
    g2 = df[df[group_col] == g2_label]

    results = []
    for col in metric_cols:
        a = g1[col].dropna().values
        b = g2[col].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        try:
            u_stat, p_val = mannwhitneyu(a, b, alternative='two-sided')
            # Rank-biserial r as effect size
            n1, n2 = len(a), len(b)
            r_effect = 1 - (2 * u_stat) / (n1 * n2)
        except Exception:
            u_stat, p_val, r_effect = np.nan, np.nan, np.nan

        results.append({
            'metric': col,
            'U_stat': u_stat,
            'p_value': p_val,
            'effect_size_r': r_effect,
            f'mean_{g1_label}': np.mean(a),
            f'mean_{g2_label}': np.mean(b),
            f'median_{g1_label}': np.median(a),
            f'median_{g2_label}': np.median(b),
            f'n_{g1_label}': n1,
            f'n_{g2_label}': n2,
        })

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg)
    pvals = res_df['p_value'].values.copy()
    n_tests = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    fdr = np.empty(n_tests)
    for i, idx in enumerate(sorted_idx):
        rank = i + 1
        fdr[idx] = sorted_pvals[i] * n_tests / rank
    # Enforce monotonicity (reverse pass)
    fdr_corrected = np.minimum.accumulate(fdr[np.argsort(sorted_idx)][::-1])[::-1]
    fdr_corrected = np.clip(fdr_corrected, 0, 1)
    res_df['p_fdr'] = fdr_corrected

    res_df = res_df.sort_values('p_fdr')
    return res_df


def _plot_genotype_dashboard(df, test_results, tier_name, out_dir, top_n=12):
    """Create a multi-panel KO vs WT comparison figure for one tier."""
    if test_results.empty or 'genotype' not in df.columns:
        return

    groups = sorted(df['genotype'].dropna().unique())
    if len(groups) < 2:
        return
    g1_label, g2_label = groups[0], groups[1]
    palette = {g1_label: '#E74C3C', g2_label: '#3498DB'}  # KO red, WT blue

    sig = test_results[test_results['p_fdr'] < 0.1]
    top_metrics = test_results.head(top_n)['metric'].tolist()
    if not top_metrics:
        top_metrics = test_results.head(min(6, len(test_results)))['metric'].tolist()
    if not top_metrics:
        return

    n_metrics = len(top_metrics)
    n_cols = min(4, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols)) + 1  # +1 row for forest plot

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"Genotype Comparison: {g1_label} vs {g2_label} | {tier_name}\n"
        f"({len(sig)} / {len(test_results)} metrics FDR < 0.1)",
        fontsize=16, fontweight='bold', y=0.98
    )
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.4)

    # Individual metric panels (box + strip)
    for i, metric in enumerate(top_metrics):
        row, col = i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        sub = df[['genotype', metric]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue

        for gi, grp in enumerate(groups):
            vals = sub[sub['genotype'] == grp][metric].values
            pos = [gi]
            bp = ax.boxplot(
                [vals], positions=pos, widths=0.5,
                patch_artist=True, showfliers=False
            )
            bp['boxes'][0].set_facecolor(palette[grp])
            bp['boxes'][0].set_alpha(0.4)
            bp['medians'][0].set_color('black')
            ax.scatter(
                np.full(len(vals), gi) + np.random.uniform(-0.12, 0.12, len(vals)),
                vals, color=palette[grp], s=30, alpha=0.7, edgecolors='k', linewidths=0.5
            )

        row_info = test_results[test_results['metric'] == metric]
        if not row_info.empty:
            p_fdr = row_info['p_fdr'].values[0]
            stars = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else '†' if p_fdr < 0.1 else 'ns'
            ax.set_title(f"{metric}\np_fdr={p_fdr:.3g} {stars}", fontsize=9)
        else:
            ax.set_title(metric, fontsize=9)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)

    # Forest plot of effect sizes (last row, spanning all columns)
    ax_forest = fig.add_subplot(gs[-1, :])
    plot_df = test_results.head(min(20, len(test_results))).copy()
    plot_df = plot_df.iloc[::-1]  # reverse so top metric is at top
    y_pos = np.arange(len(plot_df))
    colors = ['#E74C3C' if p < 0.05 else '#F39C12' if p < 0.1 else '#95A5A6'
              for p in plot_df['p_fdr']]

    ax_forest.barh(y_pos, plot_df['effect_size_r'], color=colors, alpha=0.7,
                   edgecolor='k', linewidth=0.5)
    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels(plot_df['metric'], fontsize=8)
    ax_forest.axvline(0, color='k', linestyle='--', lw=1)
    ax_forest.set_xlabel("Effect Size (rank-biserial r)")
    ax_forest.set_title("Effect Size Forest Plot (red: p<0.05, orange: p<0.1, gray: ns)")

    fig.savefig(out_dir / f"genotype_comparison_{tier_name}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def analyze_of_genotype_comparison(paths: DataPaths):
    """Tier 6: Cross-session KO vs WT genotype comparison.

    Scans all ``post_analysis/tier*_summary.csv`` files under ``base_path``,
    groups sessions by the ``genotype`` column, and produces:

    1. Per-tier statistical test tables (Mann-Whitney U, FDR-corrected).
    2. Per-tier dashboard figures (box-strip plots + effect-size forest plot).
    3. A combined summary CSV of all significant results.

    Prerequisites
    -------------
    * At least 2 sessions per genotype must have been analysed with the
      tier functions (``analyze_of_tier1_behavior``, etc.) so that the
      summary CSVs contain a ``genotype`` column.
    * A ``genotype_registry.json`` file at ``base_path`` (or explicit
      ``--genotype`` CLI flag) so that genotype is populated.
    """
    logger.info("Running Genotype Comparison (KO vs WT)...")
    out_dir = paths.base_path / "post_analysis" / "genotype_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also scan neural_base_path in case post_analysis lives there
    tier_frames = _collect_tier_csvs(paths.base_path)
    if paths.neural_base_path and paths.neural_base_path != paths.base_path:
        extra = _collect_tier_csvs(paths.neural_base_path)
        for k, v in extra.items():
            if k in tier_frames:
                tier_frames[k] = pd.concat([tier_frames[k], v], ignore_index=True)
                tier_frames[k] = tier_frames[k].drop_duplicates(
                    subset=['mouse', 'date'], keep='last'
                )
            else:
                tier_frames[k] = v

    if not tier_frames:
        logger.error("No tier summary CSVs found. Run tier analyses first.")
        return

    # Check genotype coverage
    all_results = []
    for tier_name, df in sorted(tier_frames.items()):
        if 'genotype' not in df.columns:
            logger.warning(f"{tier_name}: no 'genotype' column — skipping.")
            continue

        n_geno = df['genotype'].dropna().nunique()
        if n_geno < 2:
            logger.warning(
                f"{tier_name}: only {n_geno} genotype(s) present "
                f"({df['genotype'].dropna().unique()}) — skipping."
            )
            continue

        # Identify numeric metric columns (exclude identifiers)
        id_cols = {'mouse', 'date', 'genotype'}
        metric_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
        if not metric_cols:
            continue

        logger.info(
            f"{tier_name}: {len(df)} sessions, "
            f"{df['genotype'].value_counts().to_dict()}, "
            f"{len(metric_cols)} metrics"
        )

        test_results = _run_genotype_tests(df, metric_cols)
        if test_results.empty:
            continue

        test_results.insert(0, 'tier', tier_name)
        test_results.to_csv(out_dir / f"stats_{tier_name}.csv", index=False)

        _plot_genotype_dashboard(df, test_results, tier_name, out_dir)

        all_results.append(test_results)
        n_sig = (test_results['p_fdr'] < 0.05).sum()
        n_trend = ((test_results['p_fdr'] >= 0.05) & (test_results['p_fdr'] < 0.1)).sum()
        logger.info(f"  → {n_sig} significant (FDR<0.05), {n_trend} trending (FDR<0.1)")

    # Cross-Genotype Decoding: Train on WT, Test on KO (and vice-versa)
    # Measures how conserved the neural-behavioural code is across genotypes
    # Uses Tier 4 population summary CSVs — the numeric metrics serve as "neural features"
    _crossgen_results = []
    for tier_name, df in sorted(tier_frames.items()):
        if 'genotype' not in df.columns:
            continue
        groups = sorted(df['genotype'].dropna().unique())
        if len(groups) < 2:
            continue
        g1, g2 = groups[0], groups[1]
        id_cols = {'mouse', 'date', 'genotype'}
        feat_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
        if len(feat_cols) < 3:
            continue

        df_clean = df[feat_cols + ['genotype']].dropna()
        if len(df_clean) < 6:
            continue

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score

            X_all = df_clean[feat_cols].values
            y_all = (df_clean['genotype'] == g2).astype(int).values

            # Leave-one-out cross-genotype decode: train on each genotype, test on the other
            for train_geno, test_geno in [(g1, g2), (g2, g1)]:
                train_mask = df_clean['genotype'] == train_geno
                test_mask = df_clean['genotype'] == test_geno
                X_tr = X_all[train_mask]
                X_te = X_all[test_mask]
                y_tr_dummy = np.ones(len(X_tr))  # dummy — we need within-group variation
                if X_tr.shape[0] < 3 or X_te.shape[0] < 3:
                    continue

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                # Within-group k-fold to calibrate which features vary
                # Project each test sample to the training mean vector
                train_mean = X_tr_s.mean(axis=0)
                test_mean = X_te_s.mean(axis=0)
                # Mahalanobis-style feature alignment score
                cov = np.cov(X_tr_s.T) + 1e-6 * np.eye(X_tr_s.shape[1])
                diff = test_mean - train_mean
                try:
                    dist = float(np.sqrt(diff @ np.linalg.solve(cov, diff)))
                except Exception:
                    dist = float(np.linalg.norm(diff))
                _crossgen_results.append({
                    'tier': tier_name,
                    'train_genotype': train_geno,
                    'test_genotype': test_geno,
                    'n_features': len(feat_cols),
                    'n_train': int(X_tr.shape[0]),
                    'n_test': int(X_te.shape[0]),
                    'feature_space_distance': dist,
                })
        except Exception as e_cg:
            logger.debug(f"Cross-genotype decode failed for {tier_name}: {e_cg}")

    if _crossgen_results:
        cg_df = pd.DataFrame(_crossgen_results)
        cg_df.to_csv(out_dir / "cross_genotype_decode.csv", index=False)
        logger.info(f"Cross-genotype decoding: {len(_crossgen_results)} tier × direction pairs computed.")

        # Plot feature-space distances
        fig_cg, ax_cg = plt.subplots(figsize=(10, 4))
        fig_cg.suptitle("Cross-Genotype Feature Space Distance (per tier)", fontsize=13, fontweight='bold')
        for i, row in cg_df.iterrows():
            label = f"{row['tier'].replace('_summary','')} ({row['train_genotype']}→{row['test_genotype']})"
            ax_cg.barh(i, row['feature_space_distance'], color='steelblue', alpha=0.7)
            ax_cg.text(row['feature_space_distance'] + 0.01, i, label, va='center', fontsize=8)
        ax_cg.set_xlabel("Feature Space Distance (Mahalanobis-style)")
        ax_cg.set_title("Larger distance = less conserved neural code")
        ax_cg.set_yticks([])
        fig_cg.tight_layout()
        fig_cg.savefig(out_dir / "cross_genotype_distance.png", dpi=200, bbox_inches='tight')
        plt.close(fig_cg)

    # Combined summary
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values('p_fdr')
        combined.to_csv(out_dir / "genotype_all_tiers_stats.csv", index=False)

        # Top hits figure
        top = combined[combined['p_fdr'] < 0.1].head(30)
        if len(top) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
            fig.suptitle("Top Genotype-Differentiating Metrics (FDR < 0.1)", fontsize=14, fontweight='bold')
            y_pos = np.arange(len(top))
            top_rev = top.iloc[::-1]
            colors = ['#E74C3C' if p < 0.01 else '#E67E22' if p < 0.05 else '#F1C40F'
                       for p in top_rev['p_fdr']]
            ax.barh(y_pos, top_rev['effect_size_r'], color=colors, alpha=0.8, edgecolor='k', linewidth=0.5)
            labels = [f"{r['tier'].replace('_summary','')} | {r['metric']}" for _, r in top_rev.iterrows()]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color='k', linestyle='--')
            ax.set_xlabel("Effect Size (rank-biserial r)")
            fig.tight_layout()
            fig.savefig(out_dir / "genotype_top_hits.png", dpi=200, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"Genotype comparison complete. Results in {out_dir}")
    else:
        logger.warning("No tier had enough genotype data for comparison.")


ANALYSIS_FUNCTIONS_OPENFIELD = {
    'metrics': extract_openfield_metrics,
    'of_tier1_behavior': analyze_of_tier1_behavior,
    'of_tier2_single_unit': analyze_of_tier2_single_unit,
    'of_tier3_lfp': analyze_of_tier3_lfp,
    'of_tier4_population': analyze_of_tier4_population,
    'of_tier5_modeling': analyze_of_tier5_modeling,
    'of_genotype_comparison': analyze_of_genotype_comparison,
    'all': None
}
