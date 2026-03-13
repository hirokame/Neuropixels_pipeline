import pandas as pd
import numpy as np
from pathlib import Path
from postanalysis.data_loader import (
    DataPaths,
    convert_date_formats,
    SpikeDataLoader,
    DLCDataLoader,
    EventDataLoader,
    PhotometryDataLoader,
    StrobeDataLoader,
    LFPDataLoader,
    VAMEDataLoader
)
import json
from functools import lru_cache
from scipy.ndimage import gaussian_filter1d

from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import spikeinterface.core as si
from scipy.signal import find_peaks, butter, filtfilt, hilbert, welch
from scipy.stats import circmean, circstd, ttest_1samp, ttest_ind, sem
from scipy import stats as scipy_stats
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

# Module-level constants
DEFAULT_LFP_SAMPLING_RATE = 1000.0  # Hz
DEFAULT_DOPAMINE_SAMPLING_RATE = 100.0  # Hz
DEFAULT_PHASE_LOCKING_SIGNIFICANCE = 0.01  # p-value threshold
DEFAULT_MIN_SPIKES_FOR_PHASE = 10  # Minimum spikes for time-resolved analysis


def compute_statistics_for_tuning(data, method='ttest', pop_mean=0):
    """
    Computes summary statistics and significance tests for a distribution of tuning values.
    
    Args:
        data (np.ndarray): Array of tuning indices or firing rates.
        method (str): 'ttest' or 'wilcoxon'
        pop_mean (float): Population mean to test against (default 0).
        
    Returns:
        dict: Statistics including mean, sem, p-value, effect size.
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    if len(data) < 2:
        return {
            'mean': np.nan, 'sem': np.nan, 'std': np.nan,
            'p_value': np.nan, 'effect_size': np.nan, 'n': len(data),
            'ci_lower': np.nan, 'ci_upper': np.nan
        }
    
    mean_val = np.mean(data)
    sem_val = sem(data)
    std_val = np.std(data)
    
    if method == 'ttest':
        t_stat, p_val = ttest_1samp(data, pop_mean)
        # Cohen's d
        effect_size = (mean_val - pop_mean) / std_val if std_val > 0 else 0
    else:
        # Wilcoxon signed-rank test
        from scipy.stats import wilcoxon
        try:
            stat, p_val = wilcoxon(data - pop_mean)
        except:
            p_val = np.nan
        # Non-parametric effect size (r = Z / sqrt(N))
        effect_size = np.nan # Simplified
        
    # 95% Confidence Interval
    ci_lower, ci_upper = scipy_stats.t.interval(0.95, len(data)-1, loc=mean_val, scale=sem_val)
    
    return {
        'mean': mean_val,
        'sem': sem_val,
        'std': std_val,
        'p_value': p_val,
        'effect_size': effect_size,
        'n': len(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }




def _load_lfp_sampling_rate(lfp_dir: Path) -> float:
    """
    Load LFP sampling rate from file or return default.
    
    Args:
        lfp_dir: Path to LFP directory
        
    Returns:
        Sampling rate in Hz
    """
    fs_file = lfp_dir / 'sampling_rate.txt'
    if fs_file.exists():
        try:
            with open(fs_file, 'r') as f:
                return float(f.read().strip())
        except:
            pass
    return DEFAULT_LFP_SAMPLING_RATE

def _select_lfp_channel(lfp_data: np.ndarray, channel_index: int = None) -> np.ndarray:
    """
    Select LFP channel for analysis.
    
    Args:
        lfp_data: LFP data array (n_channels, n_samples)
        channel_index: Specific channel index, or None to use middle channel
        
    Returns:
        Selected LFP signal (n_samples,)
    """
    n_channels = lfp_data.shape[0]
    if channel_index is None:
        channel_index = n_channels // 2  # Use middle channel by default
    return lfp_data[channel_index, :]

def _get_unit_best_channels(paths: DataPaths, unique_clusters: np.ndarray) -> dict:
    """
    Helper to determine the best recording channel for each unit.
    
    Logic:
    1. Try simple heuristics/cluster_info if available (optional)
    2. Use templates.npy (Strategy 2) to find channel with max amplitude
    3. Use templates_ind.npy to map sparse indices to global channel indices
    
    Args:
        paths: DataPaths object
        unique_clusters: Array of cluster IDs
        
    Returns:
        dict: {cluster_id: best_channel_index}
    """
    unit_info = {}
    
    try:
        templates_path = paths.kilosort_dir / 'templates.npy' if paths.kilosort_dir else None
        if not templates_path or not templates_path.exists():
            templates_path = paths.neural_base / 'kilosort4' / 'sorter_output' / 'templates.npy'
        
        if templates_path and templates_path.exists():
            templates = np.load(templates_path) # (n_units, n_times, n_channels)
            
            # Check for sparse templates
            templates_ind_path = templates_path.parent / 'templates_ind.npy'
            templates_ind = None
            if templates_ind_path.exists():
                templates_ind = np.load(templates_ind_path) # (n_units, n_sparse_channels)
            
            print(f"  Loaded templates: {templates.shape}, Sparse inds: {templates_ind.shape if templates_ind is not None else 'None'}")
            
            mapped_count = 0
            for cid in unique_clusters:
                # Ensure cid is valid index for templates
                # Warning: In some versions, cluster_id might not match template index directly 
                # if curation/merging happened. For now assuming direct map or 0-indexed.
                if int(cid) < len(templates):
                    temp = templates[int(cid)]
                    
                    # Calculate Peak-to-Peak amplitude for each channel
                    ptps = temp.ptp(axis=0) # (n_channels,)
                    best_ch_local = np.argmax(ptps)
                    
                    if templates_ind is not None:
                        # Map local sparse index to global channel index
                        if int(cid) < len(templates_ind):
                            best_ch_global = templates_ind[int(cid), best_ch_local]
                        else:
                            best_ch_global = best_ch_local # Fallback
                    else:
                        best_ch_global = best_ch_local
                        
                    unit_info[cid] = int(best_ch_global)
                    mapped_count += 1
            
            print(f"  Derived channel mapping from templates for {mapped_count} units.")
        else:
            print("  Warning: templates.npy not found. Cannot derive channel mapping.")
            
    except Exception as e:
        print(f"  Warning deriving unit info: {e}")
        import traceback
        traceback.print_exc()
        
    return unit_info

def _get_kinematic_states(paths: DataPaths, speed_threshold: float = 5.0, px_per_cm: float = 30.0):
    """
    Segments behavior into 16 kinematic states based on Snout/Tail velocity:
    - 4 Port stays (Port_1, Port_2, Port_3, Port_4): Deceleration to Acceleration.
    - 12 Trajectories (X_to_Y): Acceleration to Deceleration.
    
    Returns a list of dicts: [{'start_time', 'end_time', 'label', 'duration'}]
    """
    # 1. Load Data
    try:
        base_path = paths.base_path
        
        # DLC for tracking and velocity
        dlc_loader = DLCDataLoader(base_path)
        # Check if dlc_h5 exists in paths
        if not paths.dlc_h5 or not paths.dlc_h5.exists():
            print("  Warning: DLC file not found.")
            return []
            
        df_dlc = dlc_loader.load(paths.dlc_h5)
        
        # Calculate average speed of Snout and Tail
        v_snout, _ = dlc_loader.calculate_velocity(df_dlc, video_fs=60, px_per_cm=px_per_cm, strobe_path=paths.strobe_seconds)
        v_tail, _ = dlc_loader.calculate_velocity(df_dlc, video_fs=60, px_per_cm=px_per_cm, strobe_path=paths.strobe_seconds)
        
        # Ensure length matches df_dlc
        if len(v_snout) < len(df_dlc):
            v_snout = np.pad(v_snout, (1, 0), mode='edge')
            v_tail = np.pad(v_tail, (1, 0), mode='edge')
        
        v_avg = (v_snout + v_tail) / 2.0
        
        # 2. Get Port IDs at every frame (ROI presence)
        event_loader = EventDataLoader(base_path)
        
        if not paths.event_corner or not paths.event_corner.exists():
            return []
        
        corner_df = event_loader.load(paths.event_corner, sync_to_dlc=True, dlc_data=df_dlc)
        port_ids = event_loader.infer_port_id(corner_df).values
        
        # 3. Timebase (Strobes or Fixed FS)
        try:
            strobe_loader = StrobeDataLoader(base_path)
            strobe_times = strobe_loader.load(paths.strobe_seconds)
        except Exception:
            strobe_times = np.arange(len(df_dlc)) / 60.0
            
        if len(strobe_times) < len(df_dlc):
            # Pad strobe times if slightly short
            dt = np.mean(np.diff(strobe_times)) if len(strobe_times) > 1 else 1/60.0
            extra = np.arange(1, len(df_dlc) - len(strobe_times) + 1) * dt
            strobe_times = np.concatenate([strobe_times, strobe_times[-1] + extra])
            
    except Exception as e:
        print(f"  Error loading data for kinematic states: {e}")
        return []

    # 4. State Segmentation
    is_moving = v_avg > speed_threshold
    n_frames = len(is_moving)
    segments = []
    
    curr = 0
    while curr < n_frames:
        start_frame = curr
        moving_val = is_moving[curr]
        while curr < n_frames and is_moving[curr] == moving_val:
            curr += 1
        end_frame = curr
        
        start_time = strobe_times[start_frame]
        end_time = strobe_times[min(end_frame, n_frames-1)]
        
        # Initial labeling
        if not moving_val: # Stationary
            block_ports = port_ids[start_frame:end_frame]
            active_ports = block_ports[block_ports > 0]
            if len(active_ports) > 0:
                # Most frequent port in this stationary block
                counts = np.bincount(active_ports)
                port_id = np.argmax(counts)
                label = f"Port_{port_id}"
            else:
                label = "Stationary_Other"
        else:
            label = "Movement"
            
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'label': label,
            'duration': end_time - start_time
        })

    # 5. Refine labels (Assign Trajectories)
    refined = []
    for i, seg in enumerate(segments):
        if seg['label'] == "Movement":
            # Finding previous and next port stays
            prev_port, next_port = 0, 0
            for j in range(i-1, -1, -1):
                if segments[j]['label'].startswith("Port_"):
                    prev_port = int(segments[j]['label'].split('_')[1])
                    break
            for j in range(i+1, len(segments)):
                if segments[j]['label'].startswith("Port_"):
                    next_port = int(segments[j]['label'].split('_')[1])
                    break
            
            if prev_port and next_port and prev_port != next_port:
                seg['label'] = f"{prev_port}_to_{next_port}"
            else:
                # Ignore random movements not between ports
                continue
        elif seg['label'] == "Stationary_Other":
            # Ignore non-port stationary periods
            continue
            
        # Merge if consecutive (shouldn't happen with current logic but for robustness)
        if refined and refined[-1]['label'] == seg['label']:
            refined[-1]['end_time'] = seg['end_time']
            refined[-1]['duration'] += seg['duration']
        else:
            refined.append(seg)
            
    return refined

def _plot_population_heatmap(df, output_path, title, xlabel, ylabel="Neuron ID", sort_col=None, cmap='viridis', z_score=True, unit_types=None):
    """
    Helper to plot a population heatmap from a DataFrame (Neurons x Features).
    Supports grouping by unit_types (MSN, FSI, Other) if provided.
    """
    try:
        if df.empty: return
        
        # Prepare data
        plot_data = df.copy()

        # Grouping and Sorting Logic
        transitions = []
        ylabel_text = ylabel

        if unit_types:
            # Map cluster IDs to types, default to 'Unknown'
            # Convert index to matching type in unit_types keys (int vs str)
            # Assuming unit_types keys are largely compatible with df index
            
            # Helper to safely get type
            def get_type(cid):
                val = unit_types.get(cid)
                if val is None:
                    try:
                        val = unit_types.get(int(cid))
                    except:
                        pass
                if val is None:
                    try:
                        val = unit_types.get(str(cid))
                    except:
                        pass
                return val if val else 'Unknown'

            plot_data['type'] = [get_type(cid) for cid in plot_data.index]
            
            # Find peak latency for sorting within groups
            peak_indices = np.argmax(plot_data.drop(columns=['type']).values, axis=1)
            plot_data['peak'] = peak_indices
            
            # Sort by Type (custom order) then Peak
            type_order = {'MSN': 1, 'FSI': 2, 'Other': 3}
            plot_data['type_rank'] = plot_data['type'].map(type_order).fillna(5)
            
            plot_data = plot_data.sort_values(by=['type_rank', 'peak'])
            
            # Find transition points for horizontal lines
            type_values = plot_data['type'].values
            transitions = np.where(type_values[:-1] != type_values[1:])[0] + 0.5
            
            # Create grouped Y-axis label
            counts = plot_data['type'].value_counts()
            labels = []
            for t in ['MSN', 'FSI', 'Other']:
                if t in counts:
                    labels.append(f"{t}")
            ylabel_text = "Neuron ID (Grouped: " + " / ".join(labels) + ")"
            
            # Clean up columns used for sorting
            final_plot_data = plot_data.drop(columns=['type', 'peak', 'type_rank'])
        else:
            # Default Sorting
            if sort_col:
                if sort_col == 'peak':
                    peak_indices = np.argmax(plot_data.values, axis=1)
                    sort_order = np.argsort(peak_indices)
                    final_plot_data = plot_data.iloc[sort_order]
                elif sort_col in plot_data.columns:
                    final_plot_data = plot_data.sort_values(sort_col, ascending=False)
                else:
                    final_plot_data = plot_data.sort_values(plot_data.columns[0], ascending=False)
            else:
                 final_plot_data = plot_data.sort_values(plot_data.columns[0], ascending=False)
                 
        # Normalize
        if z_score:
            means = final_plot_data.mean(axis=1)
            stds = final_plot_data.std(axis=1)
            stds[stds == 0] = 1.0
            final_plot_data = final_plot_data.sub(means, axis=0).div(stds, axis=0)
            vmin, vmax = -3, 3
            label = 'Z-scored Value'
        else:
            vmin, vmax = None, None
            label = 'Value'
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Calculate extent
        try:
             # Try to parse numeric range from columns (e.g. -1000ms ... 1000ms)
            if 'ms' in str(final_plot_data.columns[0]):
                x_start = float(final_plot_data.columns[0].replace('ms', ''))
                x_end = float(final_plot_data.columns[-1].replace('ms', ''))
            elif isinstance(final_plot_data.columns[0], (int, float)):
                 x_start = float(final_plot_data.columns[0])
                 x_end = float(final_plot_data.columns[-1])
            else:
                 raise ValueError("Non-numeric columns")
                 
            extent = [x_start, x_end, len(final_plot_data), 0]
            aspect = 'auto'
        except:
            # Fallback for categorical columns
            extent = [0, len(final_plot_data.columns), len(final_plot_data), 0]
            aspect = 'auto'

        im = ax.imshow(final_plot_data.values, aspect=aspect, cmap=cmap, interpolation='nearest', 
                       extent=extent, vmin=vmin, vmax=vmax)
        
        # Add separator lines
        for y in transitions:
            ax.axhline(y, color='white', linestyle='-', linewidth=1.5)

        plt.colorbar(im, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_text)
        ax.set_title(title)
        
        # Add vertical line at 0 if applicable
        if extent[0] < 0 and extent[1] > 0:
            ax.axvline(0, color='white', linestyle='--', alpha=0.5)

        # Draw group labels if grouped
        if unit_types and not final_plot_data.empty:
             unique_types_in_plot = []
             # Re-check types in sorted order
             sorted_types = [get_type(i) for i in final_plot_data.index]
             # Get unique preserving order
             seen = set()
             for t in sorted_types:
                 if t not in seen:
                     unique_types_in_plot.append(t)
                     seen.add(t)
             
             # Calculate centers
             boundaries = [0] + list(transitions) + [len(final_plot_data)]
             for i, t in enumerate(unique_types_in_plot):
                 if i < len(boundaries) - 1:
                     start = boundaries[i]
                     end = boundaries[i+1]
                     center = (start + end) / 2
                     
                     # Annotate on the right
                     # x position: slightly outside the plot range
                     x_range = extent[1] - extent[0]
                     x_pos = extent[1] + x_range * 0.02
                     
                     ax.text(x_pos, center, t, 
                             verticalalignment='center', fontweight='bold', rotation=270, fontsize=9)

        # Set x-ticks if categorical/fallback
        if extent == [0, len(final_plot_data.columns), len(final_plot_data), 0]:
             if len(final_plot_data.columns) <= 20:
                ax.set_xticks(np.arange(len(final_plot_data.columns)) + 0.5)
                ax.set_xticklabels(final_plot_data.columns, rotation=45, ha='right')
        
        plt.tight_layout() # This might clip manual text outside axes
        plt.savefig(output_path, bbox_inches='tight') # bbox_inches='tight' saves the outside text
        plt.close(fig)
    except Exception as e:
        print(f"  Could not generate heatmap for {title}: {e}")
        import traceback
        traceback.print_exc()

def _plot_metric_swarm(df, col_name, output_path, title, ylabel, p_val_col='p_value', outcome_col='significant', ax=None):
    """
    Plots a swarm plot for a specific metric, separated by neuron type.
    Points are colored by cell type and style/transparency indicates significance.
    """
    try:
        if df.empty or col_name not in df.columns:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Determine significance
        if outcome_col not in df.columns:
            if p_val_col in df.columns:
                df[outcome_col] = df[p_val_col] < 0.05
            else:
                # If no significance info, treat all as significant (opaque)
                df[outcome_col] = True
        
        save_figure = False
        if ax is None:
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            save_figure = True
        
        # Check if 'type' column exists - default to Other if not
        if 'type' not in df.columns and 'cell_type' in df.columns:
            df['type'] = df['cell_type']
        if 'type' not in df.columns:
            df['type'] = 'Other'
            
        # Define categories and colors
        categories = ['MSN', 'FSI', 'Other']
        colors = {'MSN': 'green', 'FSI': 'purple', 'Other': 'gray'}
        
        # Map types to x-positions
        available_types = [t for t in categories if t in df['type'].unique()]
        # Also include any types not in standard list
        for t in df['type'].unique():
            if t not in available_types:
                available_types.append(t)
                if t not in colors: colors[t] = 'gray'
        
        x_positions = {ctype: i for i, ctype in enumerate(available_types)}
        
        # Jitter parameters
        jitter_width = 0.2
        
        # Plot points
        for idx, row in df.iterrows():
            ctype = row['type']
            if ctype not in x_positions: continue
            
            x_base = x_positions[ctype]
            x_jitter = np.random.uniform(-jitter_width, jitter_width)
            x_pos = x_base + x_jitter
            
            y_pos = row[col_name]
            
            # Style based on significance
            if row[outcome_col]:
                alpha = 0.9
                edgecolor = 'black'
                zorder = 10
            else:
                alpha = 0.3
                edgecolor = 'none' 
                zorder = 5
                
            ax.scatter(x_pos, y_pos, 
                       c=colors.get(ctype, 'gray'), 
                       alpha=alpha, 
                       edgecolors=edgecolor, 
                       linewidth=0.5,
                       s=40,
                       zorder=zorder)
            
        # Formatting
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(list(x_positions.keys()))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add a horizontal line at 0
        if df[col_name].min() < 0 < df[col_name].max():
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        # Legend for Significance
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Significant (p<0.05)', markersize=8, alpha=0.9, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Not Significant', markersize=8, alpha=0.3)
        ]
        if save_figure:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Significant (p<0.05)', markersize=8, alpha=0.9, markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Not Significant', markersize=8, alpha=0.3)
            ]
            ax.legend(handles=legend_elements, loc='best')
        
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        
        if save_figure:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"  Swarm plot saved to {output_path}")

    except Exception as e:
        print(f"  Error generating swarm plot for {title}: {e}")
        import traceback
        traceback.print_exc()

def _calculate_block_tuning(trajectories, corner_order, start_time, end_time, spike_times, spike_clusters, unique_clusters):
    """
    Helper function to calculate strategy tuning index for a given block defined by [start_time, end_time].
    Uses pre-filtered kinematic trajectories.
    """
    if not trajectories: return {}
    
    cw_firing_rates = {cid: [] for cid in unique_clusters}
    ccw_firing_rates = {cid: [] for cid in unique_clusters}
    
    matching_trajs = 0
    for traj in trajectories:
        # Check if the trajectory is within our time window
        if traj['start_time'] >= start_time and traj['end_time'] <= end_time:
            matching_trajs += 1
            label = traj['label'] # e.g., "1_to_2"
            try:
                parts = label.split('_to_')
                if len(parts) != 2: continue
                
                start_port = int(parts[0])
                end_port = int(parts[1])
                
                is_cw = _is_move_correct(start_port, end_port, corner_order, True)
                is_ccw = _is_move_correct(start_port, end_port, corner_order, False)
                
                if not is_cw and not is_ccw:
                    continue
                
                # Get spikes for this trajectory
                duration = traj['end_time'] - traj['start_time']
                if duration <= 0: continue
                
                for cid in unique_clusters:
                    cluster_spikes = spike_times[spike_clusters == cid]
                    n_spikes = np.sum((cluster_spikes >= traj['start_time']) & (cluster_spikes < traj['end_time']))
                    rate = n_spikes / duration
                    
                    if is_cw:
                        cw_firing_rates[cid].append(rate)
                    if is_ccw:
                        ccw_firing_rates[cid].append(rate)
            except (ValueError, IndexError):
                continue
                
    if matching_trajs == 0:
        return {}
        
    # Calculate average rates and tuning index per cluster
    tuning_indices = {}
    for cid in unique_clusters:
        avg_cw = np.mean(cw_firing_rates[cid]) if cw_firing_rates[cid] else 0
        avg_ccw = np.mean(ccw_firing_rates[cid]) if ccw_firing_rates[cid] else 0
        
        if avg_cw + avg_ccw > 0:
            tuning_index = (avg_cw - avg_ccw) / (avg_cw + avg_ccw)
            tuning_indices[cid] = tuning_index
            
    return tuning_indices

def _load_channel_coords(paths):
    """Refined helper to load channel coordinates."""
    coords = None
    try:
        # Try kilsort output first
        f = paths.kilosort_dir / 'channel_positions.npy'
        if not f.exists():
            f = paths.neural_base / 'kilosort4' / 'sorter_output' / 'channel_positions.npy'
        if f.exists():
            coords = np.load(f)
    except:
        pass
    return coords

def _plot_shank_location(df, val_col, output_path, title, paths, p_val_col=None, significance_threshold=0.05, colormap_center=0):
    """
    Plots physical location of cells on the shank.
    """
    import matplotlib.pyplot as plt
    try:
        # 1. Load Channel Coordinates
        chan_pos = _load_channel_coords(paths)
        if chan_pos is None:
            print("  Error: channel_positions.npy not found. Cannot plot shank location.")
            return

        # 2. Get Unit Best Channels
        # We need unique clusters from df index
        unique_clusters = df.index.values
        unit_chans = _get_unit_best_channels(paths, unique_clusters)
        
        # 3. Map Units to Coordinates
        # Ensure df is unique-indexed to avoid row duplication during mapping
        df_unique = df.loc[~df.index.duplicated(keep='first')]
        
        valid_indices = []
        x_coords = []
        y_coords = []
        
        for cid in df_unique.index:
            if cid in unit_chans:
                ch_idx = unit_chans[cid]
                if ch_idx < len(chan_pos):
                    valid_indices.append(cid)
                    x_coords.append(chan_pos[ch_idx, 0])
                    y_coords.append(chan_pos[ch_idx, 1])
        
        if not valid_indices:
            print("  No units could be mapped to channels.")
            return
            
        plot_df = df_unique.loc[valid_indices].copy()
        plot_df['_x'] = x_coords
        plot_df['_y'] = y_coords
        figsize = (6, 12) 
        # --- Plot 1: Value Coded ---
        fig1, ax1 = plt.subplots(figsize=figsize)
        min_x, max_x = np.min(chan_pos[:, 0]), np.max(chan_pos[:, 0])
        min_y, max_y = np.min(chan_pos[:, 1]), np.max(chan_pos[:, 1])
        ax1.plot([min_x-20, max_x+20, max_x+20, min_x-20, min_x-20], 
                 [min_y-20, min_y-20, max_y+20, max_y+20, min_y-20], 
                 color='k', alpha=0.3, linewidth=1)
        
        # Also plot all channels as small grey dots
        ax1.scatter(chan_pos[:, 0], chan_pos[:, 1], s=5, color='lightgrey', alpha=0.5, label='Channels')
        
        values = plot_df[val_col]
        # Check if categorical or continuous
        if pd.api.types.is_numeric_dtype(values):
            if colormap_center is not None:
                # Center colormap at specified value
                delta = max(abs(values.min() - colormap_center), abs(values.max() - colormap_center))
                vmin, vmax = colormap_center - delta, colormap_center + delta
            else:
                vmin, vmax = None, None
            
            sc = ax1.scatter(plot_df['_x'], plot_df['_y'], c=values, cmap='bwr', s=60, edgecolors='k', linewidth=0.5, alpha=0.9, vmin=vmin, vmax=vmax)
            # Use fraction and pad to control colorbar size relative to axes
            plt.colorbar(sc, ax=ax1, label=val_col, fraction=0.046, pad=0.04)
        else:
            # Categorical
            cats = values.unique()
            for i, cat in enumerate(cats):
                mask = values == cat
                ax1.scatter(plot_df.loc[mask, '_x'], plot_df.loc[mask, '_y'], label=str(cat), s=60, edgecolors='k', linewidth=0.5, alpha=0.9)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        ax1.set_title(f"{title}\n({val_col})")
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (um)')
        ax1.set_ylabel('Y (um)')
        
        save_path1 = str(output_path).replace('.png', f'_{val_col}_shank.png')
        if not save_path1.endswith('.png'): save_path1 += '.png'
        
        plt.tight_layout()
        plt.savefig(save_path1, dpi=300)
        plt.close(fig1)
        print(f"  Saved shank plot 1: {save_path1}")
        
        # --- Plot 2: Significance Coded ---
        if p_val_col and p_val_col in plot_df.columns:
            fig2, ax2 = plt.subplots(figsize=figsize)
            
            # Outline & Channels
            ax2.plot([min_x-20, max_x+20, max_x+20, min_x-20, min_x-20], 
                     [min_y-20, min_y-20, max_y+20, max_y+20, min_y-20], 
                     color='k', alpha=0.3, linewidth=1)
            ax2.scatter(chan_pos[:, 0], chan_pos[:, 1], s=5, color='lightgrey', alpha=0.5)
            
            # Significant
            sig_mask = plot_df[p_val_col] < significance_threshold
            
            # Non-significant: Lighter/Translucent
            if (~sig_mask).any():
                ax2.scatter(plot_df.loc[~sig_mask, '_x'], plot_df.loc[~sig_mask, '_y'], 
                            c='gray', alpha=0.3, s=30, label='ns')
            
            # Significant: Brighter/Opaque
            if sig_mask.any():
                 if pd.api.types.is_numeric_dtype(values):
                     if colormap_center is not None:
                         # Center colormap at specified value
                         delta = max(abs(values.min() - colormap_center), abs(values.max() - colormap_center))
                         vmin, vmax = colormap_center - delta, colormap_center + delta
                     else:
                         vmin, vmax = None, None

                     # Use same colormap but full alpha
                     sc2 = ax2.scatter(plot_df.loc[sig_mask, '_x'], plot_df.loc[sig_mask, '_y'], 
                                       c=plot_df.loc[sig_mask, val_col], cmap='bwr', s=70, 
                                       edgecolors='k', linewidth=1.0, alpha=1.0, label=f'p<{significance_threshold}', vmin=vmin, vmax=vmax)
                     plt.colorbar(sc2, ax=ax2, label=val_col, fraction=0.046, pad=0.04)
                 else:
                     # Categorical
                     for i, cat in enumerate(cats):
                        mask = (values == cat) & sig_mask
                        if mask.any():
                            ax2.scatter(plot_df.loc[mask, '_x'], plot_df.loc[mask, '_y'], 
                                        s=70, edgecolors='k', linewidth=1.0, alpha=1.0, label=str(cat))
                     ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            ax2.set_title(f"{title}\n(Significance)")
            ax2.set_aspect('equal')
            ax2.set_xlabel('X (um)')
            ax2.set_ylabel('Y (um)')
            
            save_path2 = str(output_path).replace('.png', f'_significance_shank.png')
            if not save_path2.endswith('.png'): save_path2 += '.png'
            
            plt.tight_layout()
            plt.savefig(save_path2, dpi=300)
            plt.close(fig2)
            print(f"  Saved shank plot 2: {save_path2}")

    except Exception as e:
        print(f"  Error in _plot_shank_location: {e}")
        import traceback
        traceback.print_exc()
        
def _is_move_correct(start_port: int, end_port: int, cw_order: list, rule_is_cw: bool):
    """Helper to check if a move between ports follows the given rule."""
    try:
        # Find indices of ports in the clockwise order
        start_idx = cw_order.index(start_port)
        end_idx = cw_order.index(end_port)
    except (ValueError, TypeError):
        return False # Port not in the defined order
    
    if rule_is_cw: # Clockwise
        return (start_idx + 1) % len(cw_order) == end_idx
    else: # Counter-clockwise
        return (start_idx - 1 + len(cw_order)) % len(cw_order) == end_idx

def _load_switch_times(paths, event_loader, dlc_loader=None):
    """Loads switch times, correctly handling separate or embedded switch data."""
    if not paths.event_condition_switch:
        return np.array([])
        
    # Check if this maps to the corner file (embedded rule)
    if paths.event_condition_switch == paths.event_corner:
        # Rule switch is embedded in corner file (e.g. CW column)
        corner_df_full = event_loader.load(paths.event_corner, sync_to_dlc=False) # sync not needed just for times
        
        # Look for rule column
        rule_cols = ['CW', 'Condition', 'Rule', 'Protocol']
        rule_col = next((c for c in rule_cols if c in corner_df_full.columns), None)
        
        if rule_col:
            # Shift detects where values change
            df_rule = corner_df_full[rule_col].copy()
            
            # Convert to numeric to handle True/False
            if df_rule.dtype == bool or np.issubdtype(df_rule.dtype, np.bool_):
                df_rule = df_rule.map({True: 1, False: 0})
            
            # Forward/Backward fill NaNs
            df_rule = df_rule.ffill().bfill()
            
            rule_changes = df_rule.diff().fillna(0) != 0
            rule_changes.iloc[0] = False
            
            switch_df = corner_df_full[rule_changes]
            
            # Use strobe for absolute timing
            times = event_loader.get_event_times(switch_df, strobe_path=paths.strobe_seconds)
            return times

    if len(times) > 1:
        valid_times = [times[0]]
        for t in times[1:]:
            if t - valid_times[-1] > 2.0: # 2 second debounce
                valid_times.append(t)
        times = np.array(valid_times)
        
    print(f"  Detected {len(times)} event onsets as switches from {paths.event_condition_switch.name} (after debounce)")
    return times

def _get_behavioral_switch_points(switch_times, corner_times_onsets, corner_ids_onsets, 
                                  corner_df_full, corner_df_onsets, corner_order, 
                                  event_loader, strobe_path=None):
    """
    Identifies behavioral switch points (Decision and Success) for each rule switch.
    
    Returns a list of dicts, one per switch:
    [{'switch_time', 'decision_time', 'success_time', 'first_correct_trial_idx', 'rule_is_cw'}]
    """
    switch_points = []
    rule_col = 'CW'
    reward_col = 'Water'

    for t_idx, t_switch in enumerate(switch_times):
        post_switch_indices = np.where(corner_times_onsets >= t_switch)[0]
        
        if len(post_switch_indices) < 2:
            print(f"  Warning: Not enough corner events after switch {t_idx} at {t_switch:.1f}s.")
            continue
            
        first_post_idx = post_switch_indices[0]
        rule_is_cw = bool(corner_df_onsets.iloc[first_post_idx][rule_col])
        rule_str = "CW" if rule_is_cw else "CCW"
        found_correct = False
        
        # Filter indices to only those with valid ports (non-zero)
        valid_post_indices = [idx for idx in post_switch_indices if corner_ids_onsets[idx] != 0]
        
        for i in range(len(valid_post_indices) - 1):
            idx = valid_post_indices[i]
            next_idx = valid_post_indices[i+1]
            
            start_port = corner_ids_onsets[idx]
            end_port = corner_ids_onsets[next_idx]
            
            if _is_move_correct(start_port, end_port, corner_order, rule_is_cw):
                trial_start_frame = corner_df_onsets.index[idx]
                trial_end_frame = corner_df_onsets.index[next_idx]
                trial_segment = corner_df_full.loc[trial_end_frame : trial_end_frame + 120] 
                
                success_time = None
                if reward_col and reward_col in trial_segment.columns:
                    rewards = trial_segment[trial_segment[reward_col] == 1]
                    if not rewards.empty:
                        success_time = event_loader.get_event_times(rewards.iloc[[0]], strobe_path=paths.strobe_seconds)[0]
                
                if success_time is None:
                    success_time = corner_times_onsets[next_idx]
                
                decision_time = None
                start_port_col = f'Corner{start_port}'
                if start_port_col in corner_df_full.columns:
                    pre_trial_segment = corner_df_full.loc[trial_start_frame : trial_end_frame]
                    active_frames = pre_trial_segment[pre_trial_segment[start_port_col] == 1].index
                    if len(active_frames) > 0:
                        departure_frame = active_frames[-1]
                        decision_time = event_loader.get_event_times(corner_df_full.loc[[departure_frame]], strobe_path=paths.strobe_seconds)[0]
                
                if decision_time is None:
                    decision_time = corner_times_onsets[idx]
                
                switch_points.append({
                    'switch_time': t_switch,
                    'decision_time': decision_time,
                    'success_time': success_time,
                    'rule_is_cw': rule_is_cw,
                    'first_correct_trial_idx': idx
                })
                found_correct = True
                break
        
        if not found_correct:
            print(f"  Warning: No correct trial found after switch {t_idx} at {t_switch:.1f}s.")
                
    return switch_points


def analyze_lfp_movement_power(paths: DataPaths, time_window_ms: int = 1000, min_movement_duration_ms: int = 150):
    """
    Analyzes LFP power in beta and gamma bands around movement initiation.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_window_ms (int): Window in ms around movement onset to analyze.
        min_movement_duration_ms (int): Minimum duration for a bout of movement.
    """
    print("Analyzing LFP power around movement initiation...")

    # --- 1. Load LFP Data ---
    try:
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return

        lfp_fs = lfp_loader.fs
        print(f"  Initialized LFPDataLoader. FS={lfp_fs} Hz")
        
    except Exception as e:
        print(f"  Error loading LFP data: {e}")
        return

    # --- 2. Identify Movement Onsets using standardized kinematics ---
    kinematic_states = _get_kinematic_states(paths)
    
    if not kinematic_states:
        print("  No kinematic states found. Aborting.")
        return

    movement_start_times = []
    min_duration_sec = min_movement_duration_ms / 1000.0
    
    for state in kinematic_states:
        if (state['label'] == 'Movement' or '_to_' in state['label']):
            if state['duration'] >= min_duration_sec:
                movement_start_times.append(state['start_time'])
    
    print(f"  Found {len(movement_start_times)} movement initiation events.")

    if not movement_start_times:
        print("  No movement onsets detected. Aborting analysis.")
        return

    # --- 3. Analyze LFP Power Around Onsets ---
    results = []
    window_samples = int(time_window_ms / 1000 * lfp_fs)
    
    # --- Channel Selection Logic (Use LFP Loader Extractor) ---
    recording = lfp_loader.extractor
    locations = recording.get_channel_locations()
    channel_ids = recording.get_channel_ids()
    
    # Cluster X-coords to identify unique shanks
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    unique_x = np.unique(x_coords)
    
    shanks = [] 
    for x in unique_x:
        found = False
        for i, (sx, indices) in enumerate(shanks):
            if abs(x - sx) < 10.0:
                shanks[i][1].extend(np.where(x_coords == x)[0])
                found = True
                break
        if not found:
            shanks.append([x, list(np.where(x_coords == x)[0])])
    
    print(f"  Identified {len(shanks)} shanks.")
    
    selected_channels = []
    # Re-map shank indices to channel IDs properly
    for i, (sx, indices) in enumerate(sorted(shanks, key=lambda s: s[0])): 
        indices = np.array(indices)
        shank_y = y_coords[indices]
        shank_ch_ids = channel_ids[indices]
        
        top_local_idx = np.argmax(shank_y)
        bot_local_idx = np.argmin(shank_y)
        
        selected_channels.append({'id': shank_ch_ids[top_local_idx], 'shank': i+1, 'loc': 'top', 'depth': shank_y[top_local_idx]})
        selected_channels.append({'id': shank_ch_ids[bot_local_idx], 'shank': i+1, 'loc': 'bottom', 'depth': shank_y[bot_local_idx]})
        
    print(f"  Selected {len(selected_channels)} channels for analysis.")

    # Loop over channels
    for channel_info in selected_channels:
        chan_id = channel_info['id']
        
        t_max = max(movement_start_times) + (time_window_ms/1000.0) + 1.0
        try:
             # Load CSD for this channel from 0 to end of last event
             traces, timestamps = lfp_loader.get_data(0, t_max, channels=[chan_id], reference='csd')
             if len(traces) == 0: continue
             full_trace = traces[:, 0]
             trace_times = timestamps
        except Exception as e:
             print(f"    Error reading LFP Ch {chan_id}: {e}")
             continue

        for event_time in movement_start_times:
            # Find index in trace_times
            start_idx = np.searchsorted(trace_times, event_time)
            end_idx = start_idx + window_samples
            
            if start_idx < 0 or end_idx > len(full_trace):
                continue
                
            lfp_snippet = full_trace[start_idx:end_idx]
            if len(lfp_snippet) < window_samples: continue
            
            # Calculate power spectrum
            freqs, psd = welch(lfp_snippet, fs=lfp_fs, nperseg=min(len(lfp_snippet), 256))
            
            # Define frequency bands
            beta_band = (freqs >= 13) & (freqs <= 30)
            gamma_band = (freqs >= 30) & (freqs <= 80)
            theta_band = (freqs >= 4) & (freqs <= 8)
            
            # Calculate power
            beta_power = np.mean(psd[beta_band]) if np.sum(beta_band) > 0 else 0
            gamma_power = np.mean(psd[gamma_band]) if np.sum(gamma_band) > 0 else 0
            theta_power = np.mean(psd[theta_band]) if np.sum(theta_band) > 0 else 0
            
            results.append({
                'event_time': event_time,
                'channel_id': chan_id,
                'shank': channel_info['shank'],
                'location': channel_info['loc'],
                'depth': channel_info['depth'],
                'theta_power': theta_power,
                'beta_power': beta_power,
                'gamma_power': gamma_power
            })

    if not results:
        print("  Could not analyze any movement events. Aborting.")
        return

    # --- 4. Save and Display Results ---
    print("\n  LFP movement power analysis complete.")
    df_results = pd.DataFrame(results)

    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'lfp_movement_power_8ch.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

    # Summary Stats
    summary = df_results.groupby(['shank', 'location'])[['theta_power', 'beta_power', 'gamma_power']].mean()
    print("\n  Average power by channel location:")
    print(summary)

