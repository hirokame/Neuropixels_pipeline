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
    LFPDataLoader
)
import json
from functools import lru_cache
from scipy.ndimage import gaussian_filter1d

from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import spikeinterface.core as si
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from scipy.stats import circmean, circstd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# Module-level constants
DEFAULT_LFP_SAMPLING_RATE = 1000.0  # Hz
DEFAULT_DOPAMINE_SAMPLING_RATE = 100.0  # Hz
DEFAULT_PHASE_LOCKING_SIGNIFICANCE = 0.01  # p-value threshold
DEFAULT_MIN_SPIKES_FOR_PHASE = 10  # Minimum spikes for time-resolved analysis



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
        # filter df to those we have channel info for
        valid_indices = []
        x_coords = []
        y_coords = []
        
        for cid in df.index:
            if cid in unit_chans:
                ch_idx = unit_chans[cid]
                if ch_idx < len(chan_pos):
                    valid_indices.append(cid)
                    x_coords.append(chan_pos[ch_idx, 0])
                    y_coords.append(chan_pos[ch_idx, 1])
        
        if not valid_indices:
            print("  No units could be mapped to channels.")
            return
            
        plot_df = df.loc[valid_indices].copy()
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

def calculate_event_tuning(paths: DataPaths, event_file_type: str, time_window_ms: int = 2000, bin_size_ms: int = 10, bout_threshold_sec: float = 0.5):
    """
    Calculates the peri-event time histogram (PETH) for a given event type.
    
    Refactored to use modular data loading with proper schema validation and synchronization.
    """
    print(f"Running PETH for {event_file_type} events...")
    window_sec = time_window_ms / 1000.0
    bin_size_sec = bin_size_ms / 1000.0

    # --- 1. Load Event Data using modular loader ---
    try:
        
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path)
        dlc_loader = None
        if paths.dlc_h5 and paths.dlc_h5.exists():
            try:
                dlc_loader = DLCDataLoader(base_path)
            except Exception as e:
                print(f"  Warning: Could not create DLC loader for synchronization: {e}")
        
        event_times = event_loader.get_event_times_by_type(event_file_type, paths, dlc_loader=dlc_loader)
        
        print(f"  Loaded {len(event_times)} {event_file_type} events.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Load Spike Data using modular loader ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. Calculate PETH for each neuron ---
    n_bins = int(window_sec / bin_size_sec)
    peths = {cid: np.zeros(n_bins) for cid in unique_clusters}

    # Optimization: Pre-select and sort spikes for each cluster to avoid repeated boolean indexing
    spikes_by_cluster = {cid: np.sort(spike_times_sec[spike_clusters == cid]) for cid in unique_clusters}
    
    # DEBUG: Check time ranges
    if len(event_times) > 0 and len(spike_times_sec) > 0:
        if event_times.max() < spike_times_sec.min() or event_times.min() > spike_times_sec.max():
            print("  WARNING: Event times and Spike times do not overlap!")

    bin_edges = np.linspace(0, window_sec, n_bins + 1)
    
    for cid in tqdm(unique_clusters):
        cluster_spikes = spikes_by_cluster[cid]
        if len(cluster_spikes) == 0:
            continue
            
        # Vectorized search for all event windows for this cluster
        starts = event_times - (window_sec / 2)
        ends = starts + window_sec
        
        # Find indices of spikes falling into windows using binary search (O(log N))
        idx_starts = np.searchsorted(cluster_spikes, starts)
        idx_ends = np.searchsorted(cluster_spikes, ends)
        
        # Optimize: Collect all relative times first, then histogram once
        valid_mask = idx_ends > idx_starts
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            all_relative_times = [
                cluster_spikes[idx_starts[i]:idx_ends[i]] - starts[i] 
                for i in valid_indices
            ]
            if all_relative_times:
                all_relative_times_flat = np.concatenate(all_relative_times)
                hist, _ = np.histogram(all_relative_times_flat, bins=bin_edges)
                peths[cid] += hist

    # --- 4. Normalize by number of events and bin size to get firing rate (Hz) ---
    if len(event_times) > 0:
        for cid in unique_clusters:
            peths[cid] = peths[cid] / (len(event_times) * bin_size_sec)

    # --- 5. Save results ---
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    df_peth = pd.DataFrame.from_dict(peths, orient='index')
    time_bins = np.linspace(-time_window_ms / 2, time_window_ms / 2, n_bins)
    df_peth.columns = [f"{t:.0f}ms" for t in time_bins]
    df_peth.index.name = 'cluster_id'
    output_path = output_dir / f'PETH_{event_file_type}_data.csv'
    df_peth.to_csv(output_path)
    print(f"  PETH results for {event_file_type} saved to {output_path}")

    # --- 6. Generate Population Heatmap (All Neurons) ---
    try:
        if not df_peth.empty:
            heatmap_path = output_dir / f'PETH_{event_file_type}_heatmap.png'
            
            # Use the new signature with unit_types
            _plot_population_heatmap(df_peth, heatmap_path, 
                                    f"Population PETH Heatmap - {event_file_type}", 
                                    "Time from event (ms)", 
                                    sort_col='peak',
                                    unit_types=unit_types)
            print(f"  Population heatmap saved to {heatmap_path}")

    except Exception as e:
        print(f"  Could not generate PETH heatmap: {e}")

def calculate_movement_tuning(paths: DataPaths, video_fs: int = 60, px_per_cm: float = 30.0,
                                n_velocity_bins: int = 100):
    """
    Calculates the tuning of neural firing rates to the animal's movement velocity.
    
    Refactored to use modular data loading with proper schema validation.
    """
    print("Running movement tuning analysis...")

    # --- 1. Load DLC data and calculate kinematics ---
    try:
        if not paths.dlc_h5 or not paths.dlc_h5.exists():
            print(f"  Error: DLC file not found (path is {paths.dlc_h5}).")
            return
        
        base_path = paths.base_path
        dlc_loader = DLCDataLoader(base_path)
        
        df_dlc = dlc_loader.load(paths.dlc_h5)
        velocity, velocity_times = dlc_loader.calculate_velocity(
            df_dlc, video_fs=video_fs, px_per_cm=px_per_cm, strobe_path=paths.strobe_seconds
        )
        if velocity is None:
            return
    except Exception as e:
        print(f"  Error loading DLC data or calculating velocity: {e}")
        return

    # --- 2. Load Spike Data using modular loader ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return

    # --- 3. Bin Firing Rates and Kinematics ---
    session_duration = max(spike_times_sec.max(), velocity_times.max())
    bin_size_sec = 0.1
    n_time_bins = int(np.ceil(session_duration / bin_size_sec))
    time_bins = np.arange(0, n_time_bins * bin_size_sec, bin_size_sec)

    firing_rates_binned = {cid: np.zeros(n_time_bins) for cid in unique_clusters}
    for cid in unique_clusters:
        cts = spike_times_sec[spike_clusters == cid]
        hist, _ = np.histogram(cts, bins=n_time_bins, range=(0, n_time_bins * bin_size_sec))
        firing_rates_binned[cid] = hist / bin_size_sec

    # Helper to bin a kinematic variable
    def bin_kinematic(data, times):
        binned_data = np.zeros(n_time_bins)
        for i in range(n_time_bins):
            t_start, t_end = time_bins[i], time_bins[i] + bin_size_sec
            mask = (times >= t_start) & (times < t_end)
            if np.any(mask):
                binned_data[i] = np.mean(data[mask])
        return binned_data

    velocity_binned = bin_kinematic(velocity, velocity_times)

    # --- 4. ACCELERATION ANALYSIS ---
    print("  Calculating acceleration tuning...")
    dt = np.diff(velocity_times)
    dt[dt <= 0] = np.median(dt[dt > 0]) # Replace 0 dt with median
    acceleration = np.diff(velocity) / dt
    acceleration_times = velocity_times[1:]
    acceleration_binned = bin_kinematic(acceleration, acceleration_times)

    accel_bins = np.linspace(np.percentile(acceleration_binned, 1), np.percentile(acceleration_binned, 99), n_velocity_bins)
    digitized_accel = np.digitize(acceleration_binned, bins=accel_bins)
    
    accel_tuning_curves = {}
    for cid in tqdm(unique_clusters):
        curve = np.zeros(n_velocity_bins)
        for i in range(1, n_velocity_bins + 1):
            mask = digitized_accel == i
            if np.any(mask):
                curve[i-1] = np.mean(firing_rates_binned[cid][mask])
        accel_tuning_curves[cid] = curve

    # Save and plot acceleration results
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    df_accel_tuning = pd.DataFrame.from_dict(accel_tuning_curves, orient='index')
    df_accel_tuning.columns = [f"{a:.2f} cm/s^2" for a in accel_bins]
    df_accel_tuning.index.name = 'cluster_id'
    accel_csv_path = output_dir / 'FR_acceleration_data.csv'
    df_accel_tuning.to_csv(accel_csv_path)
    print(f"  Acceleration tuning curves saved to {accel_csv_path}")
    
    accel_heatmap_path = output_dir / 'FR_acceleration_heatmap.png'
    accel_heatmap_path = output_dir / 'FR_acceleration_heatmap.png'
    _plot_population_heatmap(df_accel_tuning, accel_heatmap_path, 
                             "Population Acceleration Tuning", "Acceleration (cm/s^2)", sort_col='peak',
                             unit_types=unit_types)

    # --- 5. TURNING ANALYSIS ---
    print("  Calculating turning vs. straight movement tuning...")
    try:
        scorer = df_dlc.columns.get_level_values(0)[0]
        bodyparts = df_dlc.columns.get_level_values(1).unique()
        
        # Try to find nose/head and tail/body parts
        head_part = next((p for p in ['Snout'] if p in bodyparts), None)
        tail_part = next((p for p in ['Tail'] if p in bodyparts), None)
        
        if head_part and tail_part:
            head_x = df_dlc[(scorer, head_part, 'x')].values
            head_y = df_dlc[(scorer, head_part, 'y')].values
            tail_x = df_dlc[(scorer, tail_part, 'x')].values
            tail_y = df_dlc[(scorer, tail_part, 'y')].values
            
            orientation = np.arctan2(head_y - tail_y, head_x - tail_x)
            orientation_diff = np.diff(np.unwrap(orientation))
            min_len = min(len(orientation_diff), len(dt))
            angular_velocity = orientation_diff[:min_len] / dt[:min_len]
            angular_velocity_times = velocity_times[1:1+min_len]
            
            # Bin angular velocity
            angular_velocity_binned = bin_kinematic(angular_velocity, angular_velocity_times)
            
            # Define turning and straight segments
            turn_threshold = np.percentile(np.abs(angular_velocity_binned), 75) # Top 25% is turning
            is_turning = np.abs(angular_velocity_binned) > turn_threshold
            is_straight = np.abs(angular_velocity_binned) < np.percentile(np.abs(angular_velocity_binned), 25)

            turn_analysis_results = {}
            for cid in tqdm(unique_clusters):
                rate_turning = np.mean(firing_rates_binned[cid][is_turning])
                rate_straight = np.mean(firing_rates_binned[cid][is_straight])
                turn_analysis_results[cid] = {
                    'rate_turning': rate_turning,
                    'rate_straight': rate_straight
                }
            
            df_turn = pd.DataFrame.from_dict(turn_analysis_results, orient='index')
            turn_csv_path = output_dir / 'FR_turn_data.csv'
            df_turn.to_csv(turn_csv_path)
            print(f"  Turning analysis saved to {turn_csv_path}")

        else:
            print("  Could not find required bodyparts for turning analysis (e.g., 'nose' and 'tail_base'). Skipping.")
    except Exception as e:
        print(f"  Failed to perform turning analysis: {e}")

    # --- 6. MOVEMENT ONSET ANALYSIS ---
    print("  Calculating movement onset PETH...")
    try:
        # Use new helper method
        movement_onset_times = dlc_loader.get_movement_onsets(
            df_dlc=df_dlc,
            video_fs=video_fs,
            px_per_cm=px_per_cm,
            smoothing_window_sec=0.1,
            threshold=2.0,
            strobe_path=paths.strobe_seconds
        )
        print(f"  Found {len(movement_onset_times)} movement onsets.")

        if len(movement_onset_times) > 10:
            window_ms = 2000
            bin_size_ms = 50
            window_sec = window_ms / 1000.0
            bin_size_sec = bin_size_ms / 1000.0
            n_bins = int(window_sec / bin_size_sec)
            
            onset_peths = {cid: np.zeros(n_bins) for cid in unique_clusters}
            bin_edges = np.linspace(0, window_sec, n_bins + 1)
            
            spikes_by_cluster = {cid: spike_times_sec[spike_clusters == cid] for cid in unique_clusters}

            for cid in tqdm(unique_clusters, desc="Movement Onset PETH"):
                cluster_spikes = spikes_by_cluster[cid]
                if len(cluster_spikes) == 0: continue
                
                starts = movement_onset_times - (window_sec / 2)
                ends = starts + window_sec
                
                idx_starts = np.searchsorted(cluster_spikes, starts)
                idx_ends = np.searchsorted(cluster_spikes, ends)
                
                for i in range(len(movement_onset_times)):
                    if idx_ends[i] > idx_starts[i]:
                        relative_times = cluster_spikes[idx_starts[i]:idx_ends[i]] - starts[i]
                        hist, _ = np.histogram(relative_times, bins=bin_edges)
                        onset_peths[cid] += hist
            
            # Normalize
            for cid in unique_clusters:
                onset_peths[cid] /= (len(movement_onset_times) * bin_size_sec)
            
            df_onset_peth = pd.DataFrame.from_dict(onset_peths, orient='index')
            time_labels = np.linspace(-window_ms / 2, window_ms / 2, n_bins)
            df_onset_peth.columns = [f"{t:.0f}ms" for t in time_labels]
            
            onset_peth_path = output_dir / 'PETH_movement_onset_data.csv'
            df_onset_peth.to_csv(onset_peth_path)
            print(f"  Movement onset PETH saved to {onset_peth_path}")
            
            onset_heatmap_path = output_dir / 'PETH_movement_onset_heatmap.png'
            _plot_population_heatmap(df_onset_peth, onset_heatmap_path, 
                                     "Population PETH at Movement Onset", "Time from Onset (ms)", sort_col='peak',
                                     unit_types=unit_types)
    except Exception as e:
        print(f"  Failed to perform movement onset analysis: {e}")

    # --- 7. VELOCITY TUNING (Original analysis) ---
    print("  Calculating velocity tuning...")
    velocity_bins = np.linspace(np.percentile(velocity_binned, 1), np.percentile(velocity_binned, 99), n_velocity_bins)
    digitized_velocity = np.digitize(velocity_binned, bins=velocity_bins)
    
    tuning_curves = {}
    for cid in tqdm(unique_clusters):
        tuning_curve = np.zeros(n_velocity_bins)
        for i in range(1, n_velocity_bins + 1):
            mask = digitized_velocity == i
            if np.any(mask):
                tuning_curve[i-1] = np.mean(firing_rates_binned[cid][mask])
        tuning_curves[cid] = tuning_curve
        
    # --- Save and Display Results ---
    df_tuning = pd.DataFrame.from_dict(tuning_curves, orient='index')
    df_tuning.columns = [f"{v:.2f} cm/s" for v in velocity_bins]
    df_tuning.index.name = 'cluster_id'
    output_path_csv = output_dir / 'FR_velocity_data.csv'
    df_tuning.to_csv(output_path_csv)
    print(f"  Movement tuning curves saved to {output_path_csv}")

    heatmap_path = output_dir / 'FR_velocity_heatmap.png'
    heatmap_path = output_dir / 'FR_velocity_heatmap.png'
    _plot_population_heatmap(df_tuning, heatmap_path, 
                             "Population Velocity Tuning", "Velocity (cm/s)", sort_col='peak',
                             unit_types=unit_types)

def calculate_lfp_peth(paths: DataPaths, event_file_type: str, 
                       frequency_bands: dict = None,
                       time_window_ms: int = 2000, bin_size_ms: int = 50,
                       video_fs: int = 60, px_per_cm: float = 30.0,
                       compute_spectrogram: bool = True):
    """
    Calculate peri-event time histogram (PETH) for LFP power in different frequency bands.
    
    This analysis extracts LFP data, filters it into frequency bands (beta, gamma, etc.),
    computes power envelope, and aligns it to behavioral events.
    
    Args:
        paths: DataPaths object
        event_file_type: Type of event to align to (e.g., 'reward', 'corner', 'licking')
        frequency_bands: Dictionary of frequency bands {'band_name': (low_freq, high_freq)}
                        Default: {'theta': (4, 8), 'beta': (13, 30), 'gamma': (30, 80)}
        time_window_ms: Total time window around event (ms)
        bin_size_ms: Bin size for temporal resolution (ms)
    
    Returns:
        DataFrame with LFP power PETH for each frequency band
    """
    print(f"Running LFP PETH for {event_file_type} events...")
    
    if frequency_bands is None:
        frequency_bands = {
            'theta': (4, 8),
            'beta': (13, 30),
            'low_gamma': (30, 60),
            'high_gamma': (60, 100)
        }
    
    window_sec = time_window_ms / 1000.0
    bin_size_sec = bin_size_ms / 1000.0
    
    # --- 1. Load Event Data ---
    try:
        base_path = paths.base_path
        if event_file_type == 'movement_onset':
            # Use DLC Loader for movement onsets
            print("  Detecting movement onsets from DLC...")
            dlc_loader = DLCDataLoader(base_path)
            df_dlc = dlc_loader.load(paths.dlc_h5)
            event_times = dlc_loader.get_movement_onsets(
                df_dlc=df_dlc,
                video_fs=video_fs,
                px_per_cm=px_per_cm,
                smoothing_window_sec=0.1,
                threshold=2.0,
                strobe_path=paths.strobe_seconds
        )
            print(f"  Loaded {len(event_times)} movement onset events.")
            
        else:
            event_loader = EventDataLoader(base_path)
            event_times = event_loader.get_event_times_by_type(event_file_type, paths)
            print(f"  Loaded {len(event_times)} {event_file_type} events.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- 2. Load LFP Data (Updated to use LFPDataLoader) ---
    try:
        
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return None
        
        recording = lfp_loader.extractor
        lfp_fs = lfp_loader.fs
        print(f"  Initialized LFPDataLoader. FS={lfp_fs} Hz. Sync params: {lfp_loader.sync_params}")

        # --- Channel Selection Logic (4 Shanks x Top/Bottom = 8 Channels) ---
        locations = recording.get_channel_locations()
        channel_ids = recording.get_channel_ids()
        
        # Cluster X-coords to identify unique shanks (allow small tolerance)
        x_coords = locations[:, 0]
        y_coords = locations[:, 1]
        unique_x = np.unique(x_coords)
        
        shanks = [] # List of (x_center, [channel_indices])
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
        
        selected_channels_info = [] 
        selected_channel_indices = []
        
        # Pick current 1, 3, 5, 7 (indices 0, 2, 4, 6) and rename to 1, 2, 3, 4.
        
        sorted_shanks = sorted(shanks, key=lambda s: s[0])
        target_indices = {0: 1, 2: 2, 4: 3, 6: 4}
        
        for i, (sx, indices) in enumerate(sorted_shanks): 
            if i not in target_indices:
                continue
                
            shank_id = target_indices[i] # Remapped ID 
            indices = np.array(indices)
            
            # Sort by Depth (Y)
            shank_y = y_coords[indices]
            
            # Sort indices by Y coordinate (local depth)
            sorted_local_order = np.argsort(shank_y)
            sorted_indices = indices[sorted_local_order]
            sorted_y = shank_y[sorted_local_order]
            
            # CSD requires i-1, i, i+1 valid.
            # LFPDataLoader sets edges (0 and -1) of a column to 0.
            # To be safe, we pick channels at least 2 steps from ends.
            
            n_ch = len(sorted_indices)
            if n_ch < 5:
                # Too few channels for robust CSD selection, fallback to edges but might be 0
                top_global_idx = sorted_indices[-1]
                bot_global_idx = sorted_indices[0]
            else:
                # Pick ~Top (index N-3) and ~Bottom (index 2)
                top_global_idx = sorted_indices[n_ch - 3]
                bot_global_idx = sorted_indices[2]

            top_id = channel_ids[top_global_idx]
            bot_id = channel_ids[bot_global_idx]
            
            selected_channel_indices.append(top_global_idx) # Just for our tracking
            selected_channel_indices.append(bot_global_idx)
            
            # Use actual depth of selected channel
            top_depth = y_coords[top_global_idx]
            bot_depth = y_coords[bot_global_idx]
            
            selected_channels_info.append({'id': top_id, 'idx': top_global_idx, 'shank': shank_id, 'loc': 'top', 'depth': top_depth})
            selected_channels_info.append({'id': bot_id, 'idx': bot_global_idx, 'shank': shank_id, 'loc': 'bottom', 'depth': bot_depth})
            
        print(f"  Selected {len(selected_channels_info)} representative channels: {[c['id'] for c in selected_channels_info]}")
        
        # Load CSD Trace for selected channels
        # Define the max time window needed.
        if len(event_times) == 0:
            print("  No valid events found. Skipping LFP extraction.")
            return None
            
        t_start = 0
        t_max = max(event_times) + (time_window_ms/1000.0)
        
        try:
             # Request CSD for our selected IDs
             req_ids = [c['id'] for c in selected_channels_info]
             traces_csd, timestamps_csd = lfp_loader.get_data(
                 start_time=t_start,
                 end_time=t_max,
                 channels=req_ids,
                 reference='csd'
             )
             
             if len(traces_csd) == 0:
                 print("  Error: No LFP data returned.")
                 return None
                 
        except Exception as e:
            print(f"  Error loading LFP CSD data: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Loop over channels (now columns in traces_csd)
        all_peth_results = []
        
        for ch_i, channel_info in enumerate(selected_channels_info):
            chan_id = channel_info['id']
            trace = traces_csd[:, ch_i]
            
            # Calculate power per band
            band_power = {}
            for band_name, (low_freq, high_freq) in frequency_bands.items():
                nyquist = lfp_fs / 2
                b, a = butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band')
                
                # Handle NaNs from CSD
                if np.isnan(trace).any():
                     trace_clean = np.nan_to_num(trace)
                else:
                     trace_clean = trace
                     
                filtered = filtfilt(b, a, trace_clean)
                envelope = gaussian_filter1d(np.abs(hilbert(filtered))**2, int(0.1 * lfp_fs))
                band_power[band_name] = envelope
            
            # Calculate PETH
            n_bins = int(window_sec / bin_size_sec)
            bin_edges = np.linspace(0, window_sec, n_bins + 1)
            
            # Vectorized PETH Calculation
            # 1. Filter events that fit within valid data range
            t_start = timestamps_csd[0] + (window_sec/2)
            t_end = timestamps_csd[-1] - (window_sec/2)
            valid_events = event_times[(event_times >= t_start) & (event_times <= t_end)]
            
            if len(valid_events) == 0:
                continue

            # 2. Find start indices for all events
            # timestamps_csd is monotonic
            idx_starts = np.searchsorted(timestamps_csd, valid_events - window_sec/2)
            
            # 3. Shape dimensions
            n_window_samples = int(window_sec * lfp_fs)
            n_bins = int(window_sec / bin_size_sec)
            samples_per_bin = n_window_samples // n_bins
            
            # Ensure divisibility for reshaping
            n_used_samples = n_bins * samples_per_bin
            
            # 4. Create indices matrix (N_events x N_samples) using broadcasting
            # shape: (N_events, n_used_samples)
            # Clip indices to be safe, though valid_events check should prevent this
            idx_matrix = idx_starts[:, None] + np.arange(n_used_samples)[None, :]
            # Clip using length of trace (all envelopes have same length)
            idx_matrix = np.clip(idx_matrix, 0, len(trace)-1)
            
            for band_name, power_envelope in band_power.items():
                # 5. Extract data (N_events x N_samples)
                stacked_power = power_envelope[idx_matrix]
                
                # 6. Binning: Reshape to (N_events, N_bins, Samples_per_bin)
                # Then mean over Samples_per_bin (axis 2)
                binned_power = stacked_power.reshape(len(valid_events), n_bins, samples_per_bin).mean(axis=2)
                
                # 7. Stats
                # 7. Stats
                avg_peth = np.mean(binned_power, axis=0) # (N_bins,)
                std_peth = np.std(binned_power, axis=0)  # (N_bins,)
                n_trials = len(valid_events)
                sem_peth = std_peth / np.sqrt(n_trials) if n_trials > 0 else std_peth
                
                for b_i, val in enumerate(avg_peth):
                    all_peth_results.append({
                        'channel_id': chan_id, 'shank': channel_info['shank'],
                        'location': channel_info['loc'], 'depth': channel_info['depth'],
                        'band': band_name, 
                        'time_bin': bin_edges[b_i] + (bin_size_sec/2) - (window_sec/2),
                        'power': val,
                        'power_sem': sem_peth[b_i]
                    })
                    

        # Save results
        if all_peth_results:
            df_peth = pd.DataFrame(all_peth_results)
            output_dir = paths.neural_base / 'post_analysis'
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f'LFP_PETH_{event_file_type}_8ch_data.csv'
            df_peth.to_csv(output_path, index=False)
            print(f"  LFP 8-channel PETH saved to {output_path}")

            # --- PLOTTING ---
            try:
                bands = df_peth['band'].unique()
                shanks = sorted(df_peth['shank'].unique())
                
                # Create grid: Rows = Bands, Cols = Shanks
                fig, axes = plt.subplots(len(bands), len(shanks), figsize=(4 * len(shanks), 3 * len(bands)), sharex=True)
                
                # Handle 1D axes cases
                if len(bands) == 1 and len(shanks) == 1:
                    axes = np.array([[axes]])
                elif len(bands) == 1:
                    axes = axes[np.newaxis, :]
                elif len(shanks) == 1:
                    axes = axes[:, np.newaxis]
                
                for r, band in enumerate(bands):
                    for c, shank in enumerate(shanks):
                        ax = axes[r, c]
                        
                        # Filter for this Band + Shank
                        subset = df_peth[(df_peth['band'] == band) & (df_peth['shank'] == shank)]
                        
                        if subset.empty:
                            continue
                            
                        # Plot Top (shallow) and Bottom (deep) channels
                        # 'location' column tells us 'top'/'bottom'
                        for loc, color in [('top', 'red'), ('bottom', 'blue')]:
                            trace_data = subset[subset['location'] == loc]
                            if not trace_data.empty:
                                # Sort by time
                                trace_data = trace_data.sort_values('time_bin')
                                x = trace_data['time_bin']
                                y = trace_data['power']
                                ax.plot(x, y, label=f'{loc}', color=color)
                                
                                # Plot SEM shading
                                if 'power_sem' in trace_data.columns:
                                    y_sem = trace_data['power_sem']
                                    ax.fill_between(x, y - y_sem, y + y_sem, color=color, alpha=0.2)
                                elif 'power_std' in trace_data.columns:
                                     # Fallback for compatibility
                                    y_std = trace_data['power_std']
                                    ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.2)
                        
                        if r == 0:
                            ax.set_title(f'Shank {shank}')
                        if c == 0:
                            ax.set_ylabel(f'{band}\nPower')
                        
                        if r == len(bands) - 1:
                            ax.set_xlabel('Time (s)')
                            
                        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
                        ax.legend(fontsize='x-small')
                        ax.grid(True, alpha=0.3)

                plt.suptitle(f'LFP Power PETH (Combined Shanks) - {event_file_type}')
                plt.tight_layout()
                
                plot_path = output_dir / f'LFP_PETH_{event_file_type}_summary.png'
                plt.savefig(plot_path)
                plt.close(fig)
                print(f"  LFP Summary Plot saved to {plot_path}")
            except Exception as e:
                print(f"  Could not generate LFP summary plot: {e}")

            return df_peth
        return None

    except Exception as e:
        print(f"  Error processing LFP PETH: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_dopamine_peth(paths: DataPaths, event_file_type: str,
                            time_window_ms: int = 2000, bin_size_ms: int = 50,
                            video_fs: int = 60, px_per_cm: float = 30.0):
    """
    Calculate peri-event time histogram (PETH) for dopamine signals.
    
    Aligns dopamine photometry signals to behavioral events to reveal
    event-related dopamine release dynamics.
    
    Args:
        paths: DataPaths object
        event_file_type: Type of event to align to (e.g., 'reward', 'corner', 'licking')
        time_window_ms: Total time window around event (ms)
        bin_size_ms: Bin size for temporal resolution (ms)
    
    Returns:
        DataFrame with dopamine PETH
    """
    print(f"Running Dopamine PETH for {event_file_type} events...")
    
    window_sec = time_window_ms / 1000.0
    bin_size_sec = bin_size_ms / 1000.0
    
    # --- 1. Load Event Data ---
    try:
        base_path = paths.base_path
        if event_file_type == 'movement_onset':
            # Use DLC Loader for movement onsets
            print("  Detecting movement onsets from DLC...")
            dlc_loader = DLCDataLoader(base_path)
            df_dlc = dlc_loader.load(paths.dlc_h5)
            event_times = dlc_loader.get_movement_onsets(
                df_dlc=df_dlc,
                video_fs=video_fs,
                px_per_cm=px_per_cm,
                smoothing_window_sec=0.1,
                threshold=2.0,
                strobe_path=paths.strobe_seconds
        )
            print(f"  Loaded {len(event_times)} movement onset events.")
            
        else:
            event_loader = EventDataLoader(base_path)
            event_times = event_loader.get_event_times_by_type(event_file_type, paths)
            print(f"  Loaded {len(event_times)} {event_file_type} events.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- 2. Load Dopamine Data ---
    try:  
        photometry_loader = PhotometryDataLoader(base_path)
        da_result = photometry_loader.load(paths.tdt_dff, paths.tdt_raw)
        
        da_signal = da_result['dff_values']
        da_times = da_result['dff_timestamps']
        
        # Validate time range
        if da_times.max() <= 1.0:
            print("  Warning: Dopamine timestamps appear normalized (max <= 1.0). Checking absolute time extraction.")

        print(f"  Dopamine signal shape: {da_signal.shape}, time range: {da_times.min():.2f}-{da_times.max():.2f}s")
        da_fs = 1.0 / np.median(np.diff(da_times))
        print(f"  Estimated sampling rate: {da_fs:.2f} Hz")
        
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- 3. Calculate Dopamine PETH ---
    print("  Calculating dopamine PETH...")
        
    n_bins = int(window_sec / bin_size_sec)
    bin_edges = np.linspace(0, window_sec, n_bins + 1)
    
    peth = np.zeros(n_bins)
    n_valid_events = 0
    trial_traces = []
    
    for event_time in event_times:
        start_time = event_time - (window_sec / 2)
        end_time = event_time + (window_sec / 2)
        
        # Find corresponding dopamine indices
        mask = (da_times >= start_time) & (da_times < end_time)
        
        if np.sum(mask) < 2:
            continue
        
        window_times_rel = da_times[mask] - start_time
        window_signal = da_signal[mask]
        
        # Bin the signal
        binned_signal, _ = np.histogram(window_times_rel, bins=bin_edges, weights=window_signal)
        bin_counts, _ = np.histogram(window_times_rel, bins=bin_edges)
        
        # Average signal per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            binned_signal = binned_signal / (bin_counts + 1e-10)
            binned_signal = np.nan_to_num(binned_signal, nan=0.0)
        
        peth += binned_signal
        trial_traces.append(binned_signal)
        n_valid_events += 1
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)

    if n_valid_events > 0:
        # Calculate Mean
        avg_peth = peth / n_valid_events
        
        # Create Output DataFrame for Average
        center_times = bin_edges[:-1] + (bin_size_sec / 2) - (window_sec / 2)
        df_peth = pd.DataFrame({'time': center_times, 'dFF': avg_peth})
        
        # Save Average Data
        output_path = output_dir / f'Dopamine_PETH_{event_file_type}_data.csv'
        df_peth.to_csv(output_path, index=False)
        print(f"  Dopamine PETH saved to {output_path}")
        
        # Save Individual Trials Data
        trials_arr = np.array(trial_traces)
        df_trials = pd.DataFrame(trials_arr, columns=[f"{t:.2f}s" for t in center_times])
        trials_path = output_dir / f'Dopamine_Trials_{event_file_type}_data.csv'
        df_trials.to_csv(trials_path, index_label='Trial')
        print(f"  Dopamine Trials data saved to {trials_path}")
        
        # --- PLOTTING ---
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
            
            # 1. Heatmap (Trials)
            extent = [center_times[0], center_times[-1], len(trials_arr), 0]
            # Use 'auto' aspect to fill plot irrespective of x/y ranges
            im = ax1.imshow(trials_arr, aspect='auto', cmap='viridis', extent=extent, interpolation='nearest')
            plt.colorbar(im, ax=ax1, label='dFF')
            ax1.set_ylabel('Trial')
            ax1.set_title(f'Dopamine Responses - {event_file_type}')
            ax1.axvline(0, color='white', linestyle='--', alpha=0.5)
            
            # 2. Average Trace
            sem = np.std(trials_arr, axis=0) / np.sqrt(len(trials_arr))
            ax2.plot(center_times, avg_peth, color='green', linewidth=2, label='Mean')
            ax2.fill_between(center_times, avg_peth - sem, avg_peth + sem, color='green', alpha=0.3, label='SEM')
            ax2.set_xlabel('Time from Event (s)')
            ax2.set_ylabel('dFF')
            ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plot_path = output_dir / f'Dopamine_PETH_{event_file_type}_plot.png'
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"  Dopamine Plot saved to {plot_path}")
        except Exception as e:
            print(f"  Could not generate dopamine plot: {e}")

        return df_peth
    else:
        print("  No valid events found for PETH.")
        return None

def analyze_behavioral_switch_response(paths: DataPaths, time_window_ms: int = 4000, corner_order: list = [1, 2, 4, 3]):
    """
    Analyzes neural activity aligned to the 'Behavioral Switch' - the first correct 
    trial performed after a rule switch.
    
    This aligns to:
    1. Success: The moment of first correct lick/reward at the target port.
    2. Decision: The moment the animal starts the trajectory towards the correct port 
       (departure from the previous port).
    """
    print("Analyzing behavioral switch response (Success vs. Decision)...")
    window_sec = time_window_ms / 1000.0

    # --- 1. Load Event Data with full temporal resolution ---
    try:
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path)
        
        # Load Corner Data
        corner_df_full = event_loader.load(event_path=paths.event_corner, sync_to_dlc=True)
        corner_df_onsets = event_loader.detect_onsets(corner_df_full)
        corner_times_onsets = event_loader.get_event_times(corner_df_onsets, strobe_path=paths.strobe_seconds)
        
        # Inferred valid corners logic remains the same
        ids = pd.Series(0, index=corner_df_onsets.index) # Default to 0
        for i in range(1, 4+1):
            col = f'Corner{i}'
            if col in corner_df_onsets.columns:
                mask = corner_df_onsets[col].fillna(0).astype(int) > 0
                ids[mask] = i
        corner_ids_onsets = ids.astype(int).values
            
        print(f"  Inferred corner IDs from boolean columns. Found {np.sum(corner_ids_onsets > 0)} valid visits.")
        
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids_onsets != 0
        corner_ids_onsets = corner_ids_onsets[valid_mask]
        corner_times_onsets = corner_times_onsets[valid_mask]
        
        print(f"  Filtering invalid (0) IDs: Retaining {len(corner_ids_onsets)} valid events.")

        # Load Switch Data
        if paths.event_condition_switch == paths.event_corner:
            # Rule is embedded in corner file, usually in a 'CW' or 'Condition' column
            rule_col = "CW"
            # Find where the rule changes (transitions between True/False or 1/0)
            rule_changes = corner_df_full[rule_col].diff().fillna(0) != 0
            # The first row of the file is also a "switch" to the initial rule
            rule_changes.iloc[0] = True 
            switch_df_raw = corner_df_full[rule_changes]
            switch_times = event_loader.get_event_times(switch_df_raw, strobe_path=paths.strobe_seconds)
        else:
            switch_df = event_loader.load(event_path=paths.event_condition_switch, sync_to_dlc=True)
            switch_df = event_loader.detect_onsets(switch_df)
            switch_times = event_loader.get_event_times(switch_df, strobe_path=paths.strobe_seconds)
        
        # Reward data is in the corner file under "Water" column
        lick_df_full = corner_df_full

    except Exception as e:
        print(f"  Error loading event data: {e}")
        return

    # --- 2. Identify Behavioral Switch Points ---
    switch_points = _get_behavioral_switch_points(
        switch_times, corner_times_onsets, corner_ids_onsets, 
        corner_df_full, corner_df_onsets, corner_order, 
        event_loader, strobe_path=paths.strobe_seconds
    )
    
    success_times = [pt['success_time'] for pt in switch_points]
    decision_times = [pt['decision_time'] for pt in switch_points]

    print(f"  Identified {len(success_times)} Success events and {len(decision_times)} Decision events.")

    # --- 3. Run PETH Analysis for both alignment types ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return

    session_duration = spike_times_sec[-1] if spike_times_sec.size > 0 else 1
    baseline_rates = {cid: len(spike_times_sec[spike_clusters == cid]) / session_duration for cid in unique_clusters}
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)

    def compute_and_save(event_times, suffix, title):
        results = {cid: [] for cid in unique_clusters}
        for event_time in event_times:
            start_time, end_time = event_time - (window_sec / 2), event_time + (window_sec / 2)
            for cid in unique_clusters:
                spikes = spike_times_sec[spike_clusters == cid]
                count = np.sum((spikes >= start_time) & (spikes < end_time))
                rate = count / window_sec
                brate = baseline_rates.get(cid, 0)
                results[cid].append(rate / brate if brate > 0.1 else rate)
        
        final = {cid: np.mean(rates) for cid, rates in results.items() if rates}
        df = pd.DataFrame.from_dict(final, orient='index', columns=['mean_normalized_rate'])
        df.index.name = 'cluster_id'
        
        csv_path = output_dir / f'behavioral_switch_{suffix}.csv'
        df.to_csv(csv_path)
        print(f"  Results saved for {suffix}")

    compute_and_save(success_times, "success", "Behavioral Switch - Success (Reward/Lick)")
    compute_and_save(decision_times, "decision", "Behavioral Switch - Decision (Trajectory Start)")

def analyze_port_to_port_trajectories(paths: DataPaths):
    """
    Analyzes neural activity during 16 behavioral states:
    - 4 Port stays (Port_1, Port_2, Port_3, Port_4): Deceleration to Acceleration.
    - 12 Trajectories (X_to_Y): Acceleration to Deceleration.
    
    This uses Snout and Tail velocity for precise segmentation.
    """
    print("Analyzing port-to-port trajectory activity (Kinematic 16-state)...")

    # --- 1. Load Kinematic States ---
    states = _get_kinematic_states(paths)
    if not states:
        print("  No valid kinematic states found. Aborting.")
        return
        
    print(f"  Identified {len(states)} kinematic behavioral segments.")
    traj_df = pd.DataFrame(states)
    traj_df.rename(columns={'label': 'trajectory_type'}, inplace=True)
    
    # --- 2. Load Spike Data ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return

    # --- 3. Calculate Firing Rates per State ---
    # We want to ensure all 16 possible states are represented if requested
    possible_ports = [1, 2, 4, 3]
    possible_trajs = [f"{p1}_to_{p2}" for p1 in possible_ports for p2 in possible_ports if p1 != p2]
    possible_stays = [f"Port_{p}" for p in possible_ports]
    all_possible_states = possible_stays + possible_trajs

    grouped_traj = traj_df.groupby('trajectory_type')
    results_by_traj = defaultdict(dict)
    
    for state_type in all_possible_states:
        if state_type not in grouped_traj.groups:
            # Initialize with NaNs if state never occurred
            for cid in unique_clusters:
                results_by_traj[state_type][cid] = np.nan
            continue
            
        group = grouped_traj.get_group(state_type)
        total_duration = group['duration'].sum()
        if total_duration == 0:
            for cid in unique_clusters:
                results_by_traj[state_type][cid] = np.nan
            continue
            
        for cid in unique_clusters:
            cluster_spike_times = spike_times_sec[spike_clusters == cid]
            
            total_spikes = 0
            for _, row in group.iterrows():
                spikes_in_segment = np.sum(
                    (cluster_spike_times >= row['start_time']) &
                    (cluster_spike_times < row['end_time'])
                )
                total_spikes += spikes_in_segment
            
            firing_rate = total_spikes / total_duration
            results_by_traj[state_type][cid] = firing_rate

    if not results_by_traj:
        print("  Could not calculate firing rates for any state. Aborting.")
        return
        
    # --- 4. Format and Save Results ---
    results_df = pd.DataFrame(results_by_traj).T
    results_df.index.name = 'behavioral_state'
    results_df.columns.name = 'cluster_id'

    print("\n  Port-to-port kinematic analysis complete.")
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'port_to_port_activity_data.csv'
    results_df.to_csv(output_path)
    print(f"  Results saved to {output_path}")

    # --- 5. Display Summary ---
    # Generate Heatmap (Neurons x States)
    # results_df is (State x Neuron), so we transpose it
    heatmap_path = output_dir / 'port_to_port_activity_heatmap.png'
    # Sort by 'peak' to organize neurons by their preferred behavior
    _plot_population_heatmap(results_df.T, heatmap_path, 
                             "Port-to-Port Kinematic Activity (16 States)", "Behavioral State", sort_col='peak')

def analyze_strategy_encoding(paths: DataPaths, corner_order: list = [1, 2, 4, 3], min_block_trials: int = 5):
    """
    Analyzes how neurons encode the current strategy (CW vs CCW).
    
    Calculates a strategy selectivity index for each neuron based on firing rates
    during CW vs CCW navigation blocks.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for CW navigation.
        min_block_trials (int): Minimum number of trials in a block to include it.
    """
    print("Analyzing strategy encoding...")
    
    # --- 1. Load Data ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing corner or condition switch event files.")
        return
    
    try:
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path)
        
        # Load corner events
        corner_df_full = event_loader.load(event_path=paths.event_corner, sync_to_dlc=True)
        corner_df_onsets = event_loader.detect_onsets(corner_df_full)
        corner_times = event_loader.get_event_times(corner_df_onsets, strobe_path=paths.strobe_seconds)
        
        # Get Corner IDs and Filter Invalid (0) Entries
        ids = pd.Series(0, index=corner_df_onsets.index)
        for i in range(1, 5):
            if f'Corner{i}' in corner_df_onsets.columns:
                ids[corner_df_onsets[f'Corner{i}'].fillna(False).astype(bool)] = i
        corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        assert len(corner_ids) == len(corner_times), f"Length mismatch: corner_ids={len(corner_ids)}, corner_times={len(corner_times)}"
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        if len(corner_ids) < 2:
            print("  Not enough valid corner events after filtering.")
            return
        
        # Load switch data
        if paths.event_condition_switch == paths.event_corner:
            rule_col = next((c for c in ['CW', 'Condition', 'Rule', 'Protocol'] if c in corner_df_full.columns), None)
            if rule_col:
                rule_changes = corner_df_full[rule_col].diff().fillna(0) != 0
                rule_changes.iloc[0] = True 
                switch_df_raw = corner_df_full[rule_changes]
                switch_times = event_loader.get_event_times(switch_df_raw, strobe_path=paths.strobe_seconds)
            else:
                print(f"  Warning: Could not find rule column in {paths.event_corner}. Using empty switch times.")
                switch_times = np.array([])
        else:
            switch_df = event_loader.load(event_path=paths.event_condition_switch, sync_to_dlc=True)
            switch_df = event_loader.detect_onsets(switch_df)
            switch_times = event_loader.get_event_times(switch_df, strobe_path=paths.strobe_seconds)
        
        # DEBUG OUTPUT
        print(f"  DEBUG: Loaded {len(corner_ids)} valid corner IDs, {len(switch_times)} switch times")
        print(f"  DEBUG: corner_order = {corner_order}")
        print(f"  DEBUG: Unique corner IDs in data: {np.unique(corner_ids)}")
        print(f"  DEBUG: First 10 corner IDs: {corner_ids[:min(10, len(corner_ids))]}")
        print(f"  DEBUG: switch_times = {switch_times[:min(5, len(switch_times))] if len(switch_times) > 0 else 'EMPTY'}")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- 2. Identify CW and CCW Blocks ---
    cw_segments = []
    ccw_segments = []
    
    block_boundaries = np.concatenate([[-np.inf], switch_times, [np.inf]])
    print(f"  DEBUG: Number of blocks to analyze: {len(block_boundaries) - 1}")
    
    for block_idx in range(len(block_boundaries) - 1):
        block_start = block_boundaries[block_idx]
        block_end = block_boundaries[block_idx + 1]
        
        # Find trials in this block
        block_mask = (corner_times > block_start) & (corner_times < block_end)
        block_indices = np.where(block_mask)[0]
        
        if len(block_indices) < min_block_trials:
            print(f"  DEBUG: Block {block_idx} skipped - only {len(block_indices)} trials (need {min_block_trials})")
            continue
        
        # Infer rule from majority of moves
        cw_moves = 0
        ccw_moves = 0
        for i in range(len(block_indices) - 1):
            idx = block_indices[i]
            if idx + 1 >= len(corner_ids):
                continue
            s, e = corner_ids[idx], corner_ids[idx + 1]
            if _is_move_correct(s, e, corner_order, True):
                cw_moves += 1
            if _is_move_correct(s, e, corner_order, False):
                ccw_moves += 1
        
        print(f"  DEBUG: Block {block_idx}: {len(block_indices)} trials, cw_moves={cw_moves}, ccw_moves={ccw_moves}")
        
        rule_is_cw = cw_moves > ccw_moves
        
        # Add block as a segment
        if rule_is_cw:
            cw_segments.append((corner_times[block_indices[0]], corner_times[block_indices[-1]]))
        else:
            ccw_segments.append((corner_times[block_indices[0]], corner_times[block_indices[-1]]))
    
    print(f"  Found {len(cw_segments)} CW blocks and {len(ccw_segments)} CCW blocks.")
    
    if not cw_segments or not ccw_segments:
        print("  Not enough data for both strategies. Aborting.")
        return
    
    # --- 3. Load Spike Data and Calculate Selectivity ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return
    
    results = {}
    total_cw_duration = sum(e - s for s, e in cw_segments)
    total_ccw_duration = sum(e - s for s, e in ccw_segments)
    
    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        # Calculate rate during CW strategy
        n_spikes_cw = sum(np.sum((cluster_spikes >= s) & (cluster_spikes < e)) for s, e in cw_segments)
        rate_cw = n_spikes_cw / total_cw_duration if total_cw_duration > 0 else 0
        
        # Calculate rate during CCW strategy
        n_spikes_ccw = sum(np.sum((cluster_spikes >= s) & (cluster_spikes < e)) for s, e in ccw_segments)
        rate_ccw = n_spikes_ccw / total_ccw_duration if total_ccw_duration > 0 else 0
        
        # Strategy selectivity index
        if rate_cw + rate_ccw > 0:
            selectivity_index = (rate_cw - rate_ccw) / (rate_cw + rate_ccw)
        else:
            selectivity_index = 0
        
        results[cid] = {
            'rate_cw': rate_cw,
            'rate_ccw': rate_ccw,
            'strategy_selectivity_index': selectivity_index
        }
    
    # --- 4. Save Results ---
    print("\n  Strategy encoding analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'strategy_encoding.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Generate Heatmap
    heatmap_path = output_dir / 'strategy_encoding_heatmap.png'
    cols_to_plot = ['rate_cw', 'rate_ccw']
    _plot_population_heatmap(df_results[cols_to_plot], heatmap_path, 
                             "Strategy Encoding (CW vs CCW)", "Strategy", sort_col='strategy_selectivity_index')

def analyze_directional_tuning(paths: DataPaths, corner_order: list = [1, 2, 4, 3], min_moves_per_direction: int = 5):
    """
    Analyzes preferred direction vectors for neurons (CW vs CCW tuning).
    
    Calculates directional preference for each neuron based on firing rates
    during clockwise vs counterclockwise movements.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for CW navigation.
        min_moves_per_direction (int): Minimum number of moves required per direction.
    """
    print("Analyzing directional tuning...")
    
    # --- 1. Load Data ---
    if not paths.event_corner or not paths.event_corner.exists():
        print(f"  Error: Corner event file not found.")
        return
    
    try:
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path)
        
        # Load corner events
        corner_df_full = event_loader.load(event_path=paths.event_corner, sync_to_dlc=True)
        corner_df_onsets = event_loader.detect_onsets(corner_df_full)
        corner_times = event_loader.get_event_times(corner_df_onsets, strobe_path=paths.strobe_seconds)
        
        # Get Corner IDs and Filter Invalid (0) Entries
        ids = pd.Series(0, index=corner_df_onsets.index)
        for i in range(1, 5):
            if f'Corner{i}' in corner_df_onsets.columns:
                ids[corner_df_onsets[f'Corner{i}'].fillna(False).astype(bool)] = i
        corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events (excluding non-corner onsets).")
        
    except Exception as e:
        print(f"  Error loading corner event data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- 2. Categorize Movements by Direction ---
    cw_segments = []
    ccw_segments = []
    
    for i in range(len(corner_times) - 1):
        if i + 1 >= len(corner_ids):
            continue
        
        start_port = corner_ids[i]
        end_port = corner_ids[i + 1]
        start_time = corner_times[i]
        end_time = corner_times[i + 1]
        
        if start_port == end_port or end_time <= start_time:
            continue
        
        # Check if movement is CW
        try:
            start_idx = corner_order.index(start_port)
            end_idx = corner_order.index(end_port)
            
            if (start_idx + 1) % len(corner_order) == end_idx:
                cw_segments.append((start_time, end_time))
            elif (start_idx - 1 + len(corner_order)) % len(corner_order) == end_idx:
                ccw_segments.append((start_time, end_time))
        except ValueError:
            continue
    
    print(f"  Found {len(cw_segments)} CW movements and {len(ccw_segments)} CCW movements.")
    
    if len(cw_segments) < min_moves_per_direction or len(ccw_segments) < min_moves_per_direction:
        print(f"  Not enough movements in both directions (min {min_moves_per_direction} required). Aborting.")
        return
    
    # --- 3. Load Spike Data ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return
    
    # --- 4. Calculate Directional Preference with Statistical Testing ---
    from scipy import stats
    
    results = {}
    total_cw_duration = sum(e - s for s, e in cw_segments)
    total_ccw_duration = sum(e - s for s, e in ccw_segments)
    
    # Helper for safe type retrieval
    def get_unit_type(cid):
        val = unit_types.get(cid)
        if val is None:
            try: val = unit_types.get(int(cid))
            except: pass
        if val is None:
            try: val = unit_types.get(str(cid))
            except: pass
        return val if val else 'Other'

    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        # Collect spike counts per segment for statistical testing
        cw_spike_counts = []
        ccw_spike_counts = []
        
        for s, e in cw_segments:
            duration = e - s
            n_spikes = np.sum((cluster_spikes >= s) & (cluster_spikes < e))
            # Normalize to rate (spikes/sec) per segment
            cw_spike_counts.append(n_spikes / duration if duration > 0 else 0)
        
        for s, e in ccw_segments:
            duration = e - s
            n_spikes = np.sum((cluster_spikes >= s) & (cluster_spikes < e))
            ccw_spike_counts.append(n_spikes / duration if duration > 0 else 0)
        
        cw_spike_counts = np.array(cw_spike_counts)
        ccw_spike_counts = np.array(ccw_spike_counts)
        
        # Calculate mean firing rates
        rate_cw = np.mean(cw_spike_counts) if len(cw_spike_counts) > 0 else 0
        rate_ccw = np.mean(ccw_spike_counts) if len(ccw_spike_counts) > 0 else 0
        
        # Directional preference index: -1 (CCW) to +1 (CW)
        if rate_cw + rate_ccw > 0:
            direction_index = (rate_cw - rate_ccw) / (rate_cw + rate_ccw)
        else:
            direction_index = 0
        
        # Mann-Whitney U test for statistical significance
        # Tests if CW and CCW firing rates come from different distributions
        if len(cw_spike_counts) >= 3 and len(ccw_spike_counts) >= 3:
            try:
                _, p_value = stats.mannwhitneyu(cw_spike_counts, ccw_spike_counts, alternative='two-sided')
            except ValueError:
                p_value = 1.0  # If all values are identical
        else:
            p_value = np.nan  # Not enough data for test
        
        # Determine significance (p < 0.05) and preferred direction
        is_significant = p_value < 0.05 if not np.isnan(p_value) else False
        
        # Preferred direction based on significance only
        if is_significant:
            if direction_index > 0:
                preferred = 'CW'
            else:
                preferred = 'CCW'
        else:
            preferred = 'None'
        
        results[cid] = {
            'rate_cw': rate_cw,
            'rate_ccw': rate_ccw,
            'direction_index': direction_index,
            'p_value': p_value,
            'significant': is_significant,
            'preferred_direction': preferred,
            'cell_type': get_unit_type(cid)
        }
    
    # --- 5. Save Results ---
    print("\n  Directional tuning analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'directional_tuning.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Summary statistics
    n_total = len(df_results)
    n_cw = np.sum(df_results['preferred_direction'] == 'CW')
    n_ccw = np.sum(df_results['preferred_direction'] == 'CCW')
    n_none = np.sum(df_results['preferred_direction'] == 'None')
    n_significant = n_cw + n_ccw  # Significantly tuned = CW + CCW
    
    print(f"\n  === DIRECTIONAL TUNING SUMMARY ===")
    print(f"  Total neurons analyzed: {n_total}")
    print(f"  Significantly tuned (p<0.05): {n_significant} ({100*n_significant/n_total:.1f}%)")
    print(f"    - CW-preferring: {n_cw} ({100*n_cw/n_total:.1f}%)")
    print(f"    - CCW-preferring: {n_ccw} ({100*n_ccw/n_total:.1f}%)")
    print(f"  Non-directional/NS: {n_none} ({100*n_none/n_total:.1f}%)")

    # --- SWARM PLOT (Z-scored Direction Index by Cell Type) ---
    try:
        print("  Generating directional tuning swarm plot...")
        
        # 1. Z-score the direction index across the entire population
        mean_idx = df_results['direction_index'].mean()
        std_idx = df_results['direction_index'].std()
        if std_idx == 0: std_idx = 1.0
        
        df_results['z_scored_index'] = (df_results['direction_index'] - mean_idx) / std_idx
        
        # 2. Use Helper
        swarm_path = output_dir / 'directional_tuning_swarm.png'
        _plot_metric_swarm(df_results, 'z_scored_index', swarm_path, 
                           'Directional Selectivity by Cell Type', 'Z-scored CW/CCW Index')
        
    except Exception as e:
        print(f"  Error generating swarm plot: {e}")
        import traceback
        traceback.print_exc()

def analyze_context_dependent_encoding(paths: DataPaths, corner_order: list = [1, 2, 4, 3]):
    """
    Analyzes context-dependent encoding: same stimulus encoded differently in CW vs CCW context.
    
    Examines whether neurons encode port visits differently depending on the current strategy.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners.
    """
    print("Analyzing context-dependent encoding...")
    
    # --- 1. Load Data ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing required event files.")
        return
    
    try:
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path)
        
        # Load corner events
        corner_df_full = event_loader.load(event_path=paths.event_corner, sync_to_dlc=True)
        corner_df_onsets = event_loader.detect_onsets(corner_df_full)
        corner_times = event_loader.get_event_times(corner_df_onsets, strobe_path=paths.strobe_seconds)
        
        # Get Corner IDs and Filter Invalid (0) Entries
        ids = pd.Series(0, index=corner_df_onsets.index)
        for i in range(1, 5):
            if f'Corner{i}' in corner_df_onsets.columns:
                ids[corner_df_onsets[f'Corner{i}'].fillna(False).astype(bool)] = i
        corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events.")
        
        # Load switch data
        if paths.event_condition_switch == paths.event_corner:
            rule_col = next((c for c in ['CW', 'Condition', 'Rule', 'Protocol'] if c in corner_df_full.columns), None)
            if rule_col:
                rule_changes = corner_df_full[rule_col].diff().fillna(0) != 0
                rule_changes.iloc[0] = True 
                switch_df_raw = corner_df_full[rule_changes]
                switch_times = event_loader.get_event_times(switch_df_raw, strobe_path=paths.strobe_seconds)
            else:
                print(f"  Warning: Could not find rule column in {paths.event_corner}. Using empty switch times.")
                switch_times = np.array([])
        else:
            switch_df = event_loader.load(event_path=paths.event_condition_switch, sync_to_dlc=True)
            switch_df = event_loader.detect_onsets(switch_df)
            switch_times = event_loader.get_event_times(switch_df, strobe_path=paths.strobe_seconds)
            
        print(f"  Loaded {len(switch_times)} rule switch events.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        return
    
    # --- 2. Categorize Port Visits by Context (Actual Movement Direction) ---
    port_visits_cw = {i: [] for i in range(1, 5)}
    port_visits_ccw = {i: [] for i in range(1, 5)}
    
    cw_count = 0
    ccw_count = 0
    
    # Iterate through corner events and classify based on the transition from the previous port
    for i in range(1, len(corner_ids)):
        prev_port = corner_ids[i-1]
        curr_port = corner_ids[i]
        curr_time = corner_times[i]
        
        if prev_port == curr_port:
            continue # Skip repeat visits to the same port
            
        try:
            # Check indices in clockwise order [1, 2, 4, 3]
            prev_idx = corner_order.index(prev_port)
            curr_idx = corner_order.index(curr_port)
            
            # Clockwise transition
            if (prev_idx + 1) % len(corner_order) == curr_idx:
                port_visits_cw[curr_port].append(curr_time)
                cw_count += 1
            # Counter-clockwise transition
            elif (prev_idx - 1 + len(corner_order)) % len(corner_order) == curr_idx:
                port_visits_ccw[curr_port].append(curr_time)
                ccw_count += 1
            # Other jumps are skipped to maintain pure direction context
        except (ValueError, TypeError):
            continue
            
    print(f"  Categorized {cw_count} CW visits and {ccw_count} CCW visits based on actual transitions.")
    
    # --- 3. Load Spike Data ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return
    
    # Helper for safe type retrieval
    def get_unit_type(cid):
        val = unit_types.get(cid)
        if val is None:
            try: val = unit_types.get(int(cid))
            except: pass
        if val is None:
            try: val = unit_types.get(str(cid))
            except: pass
        return val if val else 'Other'

    # DEBUG: Show port visit counts
    print(f"  DEBUG: Port visits per context:")
    for port in range(1, 5):
        print(f"    Port {port}: CW={len(port_visits_cw[port])}, CCW={len(port_visits_ccw[port])}")
    
    # --- 4. Calculate Firing Rates and Statistical Significance ---
    from scipy import stats
    window_sec = 1.0
    results = []
    
    for port in range(1, 5):
        if len(port_visits_cw[port]) < 3 or len(port_visits_ccw[port]) < 3:
            print(f"  DEBUG: Skipping port {port} - need >=3 visits in BOTH contexts")
            continue
        
        for cid in unique_clusters:
            cluster_spikes = spike_times_sec[spike_clusters == cid]
            
            # CW context: Collect spike counts per visit for statistical testing
            cw_rates = []
            for t in port_visits_cw[port]:
                # WINDOW: [t, t + window_sec] - captures activity AFTER arrival (post-deceleration)
                count = np.sum((cluster_spikes >= t) & (cluster_spikes < t + window_sec))
                cw_rates.append(count / window_sec)
            
            # CCW context
            ccw_rates = []
            for t in port_visits_ccw[port]:
                count = np.sum((cluster_spikes >= t) & (cluster_spikes < t + window_sec))
                ccw_rates.append(count / window_sec)
                
            rate_cw = np.mean(cw_rates)
            rate_ccw = np.mean(ccw_rates)
            
            # Mann-Whitney U test
            try:
                _, p_val = stats.mannwhitneyu(cw_rates, ccw_rates, alternative='two-sided')
            except ValueError:
                p_val = 1.0
            
            # Context modulation index
            if rate_cw + rate_ccw > 0:
                context_index = (rate_cw - rate_ccw) / (rate_cw + rate_ccw)
            else:
                context_index = 0
            
            is_significant = p_val < 0.05
            
            results.append({
                'cluster_id': cid,
                'port': port,
                'rate_cw_context': rate_cw,
                'rate_ccw_context': rate_ccw,
                'context_modulation_index': context_index,
                'p_value': p_val,
                'significant': is_significant,
                'preferred_context': 'CW' if (is_significant and context_index > 0) else ('CCW' if (is_significant and context_index < 0) else 'None'),
                'cell_type': get_unit_type(cid)
            })
    
    # --- 5. Save and Summarize Results ---
    print("\n  Context-dependent encoding analysis complete.")
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("  No significant results to verify.")
        return

    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'context_dependent_encoding.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    
    # Detailed Summary
    n_total_neurons = df_results['cluster_id'].nunique()
    sig_df = df_results[df_results['significant']]
    n_sig_neurons = sig_df['cluster_id'].nunique()
    n_sig_combos = len(sig_df)
    
    print(f"\n  === CONTEXT ENCODING SUMMARY ===")
    print(f"  Total unique neurons analyzed: {n_total_neurons}")
    print(f"  Neurons significant (p<0.05) at >=1 port: {n_sig_neurons} ({100*n_sig_neurons/n_total_neurons:.1f}%)")
    print(f"  Total significant port-neuron combinations: {n_sig_combos}")
    
    print("\n  Significant neurons by port:")
    for port in range(1, 5):
        n_port_sig = np.sum(df_results[df_results['port'] == port]['significant'])
        print(f"    Port {port}: {n_port_sig} neurons")
        
    # --- SWARM PLOT (Z-scored Context Index by Port) ---
    # --- SWARM PLOT (Z-scored Context Index by Port) ---
    try:
        print("  Generating context encoding swarm plots...")
        
        mean_idx = df_results['context_modulation_index'].mean()
        std_idx = df_results['context_modulation_index'].std()
        if std_idx == 0: std_idx = 1.0
        
        df_results['z_scored_index'] = (df_results['context_modulation_index'] - mean_idx) / std_idx
        
        # 2. Setup Plot (2x2 Grid for 4 Ports)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axes = axes.flatten()
        
        ports = sorted(df_results['port'].unique())
        
        for i, port in enumerate(ports):
            if i >= len(axes): break
            ax = axes[i]
            port_data = df_results[df_results['port'] == port]
            
            _plot_metric_swarm(port_data, 'z_scored_index', None, 
                              f'Port {port}', 'Z-scored CW/CCW Index', ax=ax)
            
            if i % 2 == 0:
                ax.set_ylabel('Z-scored CW/CCW Index')
            else:
                ax.set_ylabel('')
                
        # Clean up empty subplots
        for j in range(len(ports), len(axes)):
             fig.delaxes(axes[j])
             
        # Global Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Significant (p<0.05)', markersize=8, alpha=0.9, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Not Significant', markersize=8, alpha=0.3)
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))
        
        swarm_path = output_dir / 'context_dependent_encoding_swarm.png'
        plt.tight_layout()
        plt.savefig(swarm_path, dpi=300)
        plt.close(fig)
        print(f"  Swarm plot saved to {swarm_path}")
        
    except Exception as e:
        print(f"  Error generating swarm plot: {e}")
        import traceback
        traceback.print_exc()


def _interpolate_trajectory(x, y, n_points=100):
    """
    Interpolates a 2D trajectory to a fixed number of points.
    
    Args:
        x, y: 1D arrays of coordinates
        n_points: Number of points in output
        
    Returns:
        x_new, y_new: Interpolated coordinates
    """
    if len(x) < 2:
        return np.full(n_points, x[0]), np.full(n_points, y[0])
        
    # Calculate cumulative distance along path
    dist = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    dist = np.insert(dist, 0, 0)
    
    if dist[-1] == 0: # No movement
        return np.full(n_points, x[0]), np.full(n_points, y[0])
        
    # Normalized distance 0 to 1
    t = dist / dist[-1]
    
    # Interpolate
    t_new = np.linspace(0, 1, n_points)
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)
    
    return x_new, y_new

def analyze_trajectory_consistency(paths: DataPaths, output_dir: Path = None):
    """
    Analyzes and visualizes the spatial consistency of trajectories between ports.
    
    Logic:
    1. Extract kinematic states (Movement between ports).
    2. Group trajectories by start_port -> end_port.
    3. Normalize trajectories to static length (spatial interpolation).
    4. Calculate mean trajectory and variability.
    5. Compute consistency metric (e.g. mean distance from mean path).
    """
    if output_dir is None:
        output_dir = paths.neural_base / "post_analysis" / "trajectory_consistency"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing Trajectory Consistency for {paths.mouse_id}...")
    
    # 1. Load Data
    try:
        dlc_loader = DLCDataLoader(paths.base_path)
        df_dlc = dlc_loader.load(paths.dlc_h5)
        
        # Get pixels per cm from config if available, else default
        px_per_cm = 30.0 # Default
        
        # Extract X, Y (using 'Snout' generally)
        # Check if columns are MultiIndex
        if isinstance(df_dlc.columns, pd.MultiIndex):
            scorer = df_dlc.columns.get_level_values(0)[0]
            bodyparts = df_dlc.columns.get_level_values(1).unique()
            # Try to find a good bodypart
            bp = next((b for b in ['Snout', 'Head', 'Body', 'body', 'torso'] if b in bodyparts), bodyparts[0])
            
            x_raw = df_dlc[(scorer, bp, 'x')].values
            y_raw = df_dlc[(scorer, bp, 'y')].values
        else:
            # Flat columns fallback (unlikely given loader, but safe)
            x_raw = df_dlc.iloc[:, 0].values
            y_raw = df_dlc.iloc[:, 1].values
        
        # Strobe times for mapping states to frames
        try:
            strobe_loader = StrobeDataLoader(paths.base_path)
            strobe_times = strobe_loader.load(paths.strobe_seconds)
        except:
             # Fallback
            print("  Warning: generating linear timebase (60Hz)")
            strobe_times = np.arange(len(x_raw)) / 60.0

    except Exception as e:
        print(f"  Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Get Kinematic States
    states = _get_kinematic_states(paths)
    if not states:
        print("  No kinematic states found.")
        return
        
    # 3. Process Trajectories
    trajectories = defaultdict(list) # Key: "1_to_2", Value: list of (x, y) arrays
    trajectory_meta = [] # Store metadata for consistency over time analysis
    
    n_points = 100
    
    # Map time to frame index for fast lookup
    # Assuming monotonically increasing strobe_times
    
    for seg in tqdm(states, desc="Processing Trajectories"):
        if "_to_" in seg['label']:
            label = seg['label']
            start_t = seg['start_time']
            end_t = seg['end_time']
            
            # Find indices
            idx_start = np.searchsorted(strobe_times, start_t)
            idx_end = np.searchsorted(strobe_times, end_t)
            
            if idx_end - idx_start < 5: # Ignore very short segments (< ~80ms)
                continue
                
            xs = x_raw[idx_start:idx_end]
            ys = y_raw[idx_start:idx_end]
            
            # Interpolate spatially
            x_interp, y_interp = _interpolate_trajectory(xs, ys, n_points)
            
            trajectories[label].append(np.stack([x_interp, y_interp], axis=1)) # (100, 2)
            
            trajectory_meta.append({
                'label': label,
                'start_time': start_t,
                'data': np.stack([x_interp, y_interp], axis=1)
            })
            
    # 4. Analyze & Plot
    consistency_scores = []
    
    for label, trajs_list in trajectories.items():
        if len(trajs_list) < 5: continue
        
        # (N_trials, 100, 2)
        all_trajs = np.array(trajs_list) 
        
        # Mean Path
        mean_path = np.mean(all_trajs, axis=0) # (100, 2)
        
        # Variability (Std Dev of distance from mean at each point)
        # Dist = sqrt((x - mu_x)^2 + (y - mu_y)^2)
        dists = np.sqrt(np.sum((all_trajs - mean_path)**2, axis=2)) # (N_trials, 100)
        
        # Global Consistency Metric (Mean deviation across whole path)
        trial_deviations = np.mean(dists, axis=1) # (N_trials,) -> scalar per trial
        
        # Identify associated metadata
        these_meta = [m for m in trajectory_meta if m['label'] == label]
        
        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Overlay Plot
        ax = axes[0]
        # Plot individual lines (thin, alpha)
        for tr in all_trajs:
            ax.plot(tr[:, 0], tr[:, 1], 'k-', alpha=0.05, linewidth=1)
        
        # Plot mean path (red)
        ax.plot(mean_path[:, 0], mean_path[:, 1], 'r-', linewidth=2, label='Mean Path')
        
        # Plot start/end
        ax.plot(mean_path[0, 0], mean_path[0, 1], 'go', label='Start')
        ax.plot(mean_path[-1, 0], mean_path[-1, 1], 'bo', label='End')
        
        ax.set_title(f"Trajectories: {label} (n={len(all_trajs)})")
        ax.invert_yaxis() # Camera coords usually top-left origin
        ax.set_aspect('equal')
        ax.legend()
        
        # 2. Consistency Distribution / Time
        ax2 = axes[1]
        
        # Map trial_deviations back to time
        times = [m['start_time'] for m in these_meta]
        
        if len(times) == len(trial_deviations):
             ax2.scatter(times, trial_deviations, alpha=0.6, c='blue', s=10)
             
             # Trend line
             if len(times) > 10:
                 try:
                     z = np.polyfit(times, trial_deviations, 1)
                     p = np.poly1d(z)
                     ax2.plot(times, p(times), "r--", alpha=0.8, label=f"Trend")
                 except:
                     pass
        
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Spatial Deviation from Mean (px)")
        ax2.set_title(f"Trajectory Variability Over Time")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"trajectory_{label}.png")
        plt.close(fig)
        
        # Store summary stats
        avg_dev = np.mean(trial_deviations)
        std_dev = np.std(trial_deviations)
        consistency_scores.append({
            'Path': label,
            'N_trials': len(all_trajs),
            'Mean_Deviation_px': avg_dev,
            'Std_Deviation_px': std_dev
        })
        
    # Save Summary CSV
    if consistency_scores:
        df_scores = pd.DataFrame(consistency_scores)
        df_scores.to_csv(output_dir / "trajectory_consistency_summary.csv", index=False)
        print(f"  Saved consistency summary to {output_dir}")
    else:
        print("  No consistent trajectories found to analyze.")



    # 5. Predict Deviation from Neural Activity (Optional)
    # -------------------------------------------------------------------------
    try:
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.model_selection import cross_val_predict, KFold
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        
        print("  Predicting trajectory deviation from pre-movement neural activity...")
        
        # Load Spikes
        try:
            base_path = paths.neural_base.parent if paths.neural_base else Path('.')
            base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
            spike_loader = SpikeDataLoader(base_path)
            spike_data = spike_loader.load(paths.kilosort_dir)
            
            spike_times_sec = spike_data['spike_times_sec']
            spike_clusters = spike_data['spike_clusters']
            unique_clusters = spike_data['unique_clusters']
            unit_types = spike_data['unit_types']
            unit_labels = spike_data['unit_labels']
            
        except Exception as e:
            print(f"  Error loading spike data: {e}")
            return
            
        n_neurons = len(unique_clusters)
        


        # --- Feature Extraction Settings ---
        # 1. Time-Warped: 50 bins from Start to End
        n_bins_warped = 50
        
        # 2. Fixed Window: 100 bins of 5ms (500ms total) from Start
        n_bins_fixed = 100
        bin_size_fixed = 0.005
        window_fixed = n_bins_fixed * bin_size_fixed
        
        all_X_warped = []
        all_X_fixed = []
        all_y = []
        all_labels = []
        
        print(f"  Extracting features: Fixed (500ms, {n_bins_fixed} bins) vs Warped ({n_bins_warped} bins).")
        
        for label, trajs_list in trajectories.items():
            if len(trajs_list) < 5: continue
            
            # Identify trials for this path
            these_meta = [m for m in trajectory_meta if m['label'] == label]
            
            # --- y: Trajectory Deviation ---
            # Re-calculate deviation
            current_trajs = np.array([m['data'] for m in these_meta])
            current_mean = np.mean(current_trajs, axis=0) # (100, 2)
            current_dists = np.sqrt(np.sum((current_trajs - current_mean)**2, axis=2)) # (N, 100)
            y_subset = np.mean(current_dists, axis=1) # (N,) -> scalar per trial
            
            all_y.extend(y_subset)
            all_labels.extend([label] * len(y_subset))
            
            # --- X Extraction ---
            
            for i, meta in enumerate(these_meta):
                t_start = meta['start_time']
                
                # Find End Time for Warp
                t_end = next((s['end_time'] for s in states if s['start_time'] == t_start and s['label'] == label), None)
                if t_end is None: t_end = t_start + 0.5 # Fallback
                
                # --- A. Time-Warped Features ---
                warped_features = []
                bin_edges_warped = np.linspace(t_start, t_end, n_bins_warped + 1)
                duration = t_end - t_start
                
                # --- B. Fixed Window Features ---
                fixed_features = []
                t_end_fixed = t_start + window_fixed
                bin_edges_fixed = np.linspace(t_start, t_end_fixed, n_bins_fixed + 1)
                
                for cid in unique_clusters:
                    cluster_spikes = spike_times_sec[spike_clusters == cid]
                    
                    # 1. Warped
                    spikes_warped = cluster_spikes[(cluster_spikes >= t_start) & (cluster_spikes <= t_end)]
                    hist_warped, _ = np.histogram(spikes_warped, bins=bin_edges_warped)
                    if duration > 0:
                        hist_warped = hist_warped / (duration / n_bins_warped) # Hz
                    warped_features.append(hist_warped)
                    
                    # 2. Fixed
                    # Note: We capture spikes even if they are AFTER the current trial ends (if trial < 500ms)
                    # This is intentional for "fixed window from onset" analysis
                    spikes_fixed = cluster_spikes[(cluster_spikes >= t_start) & (cluster_spikes < t_end_fixed)]
                    hist_fixed, _ = np.histogram(spikes_fixed, bins=bin_edges_fixed)
                    hist_fixed = hist_fixed / bin_size_fixed # Hz
                    fixed_features.append(hist_fixed)
                
                all_X_warped.append(np.concatenate(warped_features))
                all_X_fixed.append(np.concatenate(fixed_features))

        if not all_y:
            print("  No valid data for prediction.")
            return
            
        X_warped = np.array(all_X_warped)
        X_fixed = np.array(all_X_fixed)
        y = np.array(all_y)
        
        print(f"  Data shapes: Warped={X_warped.shape}, Fixed={X_fixed.shape}, y={y.shape}")
        
        # --- Model Comparison ---
        # Models to test
        results = []
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        datasets = {
            'Time-Warped (50 bins)': X_warped,
            'Fixed 500ms (100 bins)': X_fixed
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        model = Ridge(alpha=100.0)
        
        for idx, (name, X_data) in enumerate(datasets.items()):
            print(f"  Training Ridge on {name}...")
            # Use pipeline to standardize features within CV
            pipeline = make_pipeline(StandardScaler(), model)
            
            try:
                y_pred = cross_val_predict(pipeline, X_data, y, cv=cv, n_jobs=-1)
                
                r2 = r2_score(y, y_pred)
                corr = np.corrcoef(y, y_pred)[0, 1] if np.std(y_pred) > 0 else 0
                
                # Plot
                ax = axes[idx]
                ax.scatter(y, y_pred, alpha=0.6, c='purple' if 'Fixed' in name else 'green', edgecolors='w')
                
                # Identity line
                min_val = min(y.min(), y_pred.min())
                max_val = max(y.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                ax.set_xlabel('Actual Deviation (px)')
                ax.set_ylabel('Predicted Deviation (px)')
                ax.set_title(f"{name}\nRidge: $R^2$={r2:.3f}, r={corr:.3f}")
                ax.grid(True, alpha=0.3)
                
                results.append({
                    'FeatureSet': name,
                    'Model': 'Ridge',
                    'R2_Score': r2,
                    'Correlation': corr
                })
                
            except Exception as e:
                print(f"    Failed to run {name}: {e}")
                
        plt.tight_layout()
        plt.savefig(output_dir / "deviation_prediction_comparison.png")
        plt.close(fig)
        
        # Save Prediction Summary
        if results:
            pd.DataFrame(results).to_csv(output_dir / "deviation_prediction_method_comparison.csv", index=False)
            print(f"  Saved prediction comparison to {output_dir}")

            
    except Exception as e:
        print(f"  Error in trajectory prediction: {e}")
        import traceback
        traceback.print_exc()


def analyze_spatial_rate_maps(paths: DataPaths, bin_size_cm: float = 2.0, sigma_cm: float = 2.0):
    """
    Computes and plots spatial rate maps (place fields) for all units.
    
    Args:
        paths: DataPaths object
        bin_size_cm: Size of spatial bins in cm
        sigma_cm: Standard deviation for Gaussian smoothing kernel in cm
    """
    print("Generating spatial rate maps...")
    

    # 1. Load Data
    base_path = paths.neural_base.parent if paths.neural_base else Path('.')
    base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
    spike_loader = SpikeDataLoader(base_path)
    spike_data = spike_loader.load(paths.kilosort_dir)
    
    spike_times_sec = spike_data['spike_times_sec']
    spike_clusters = spike_data['spike_clusters']
    unique_clusters = spike_data['unique_clusters']
    unit_types = spike_data['unit_types']
    unit_labels = spike_data['unit_labels']

    if not paths.dlc_h5 or not paths.dlc_h5.exists():
            print("  Error: No DLC file found for spatial mapping.")
            return
    
    dlc_loader = DLCDataLoader(paths.base_path)
    df_dlc = dlc_loader.load(paths.dlc_h5)
    bp = 'Snout' # Default assumption
    scorer = df_dlc.columns.levels[0][0]
    x = df_dlc[scorer][bp]['x'].values
    y = df_dlc[scorer][bp]['y'].values
    
    # Pixel to cm
    px_per_cm = 30.0 # Standard assumption
    x_cm = x / px_per_cm
    y_cm = y / px_per_cm
    
    strobe_loader = StrobeDataLoader(paths.base_path)
    t_pos = strobe_loader.load(paths.strobe_seconds)
        
    L = min(len(t_pos), len(x_cm))
    t_pos = t_pos[:L]
    x_cm = x_cm[:L]
    y_cm = y_cm[:L]
    
    # Remove NaNs
    valid_pos = ~np.isnan(x_cm) & ~np.isnan(y_cm)
    x_cm = x_cm[valid_pos]
    y_cm = y_cm[valid_pos]
    t_pos = t_pos[valid_pos]
    
    if len(x_cm) == 0:
            print("  No valid position data found.")
            return
            
    # 2. Define Grid
    x_min, x_max = np.nanmin(x_cm), np.nanmax(x_cm)
    y_min, y_max = np.nanmin(y_cm), np.nanmax(y_cm)
    
    x_edges = np.arange(x_min, x_max + bin_size_cm, bin_size_cm)
    y_edges = np.arange(y_min, y_max + bin_size_cm, bin_size_cm)
    
    # 3. Calculate Occupancy Map
    dt = np.mean(np.diff(t_pos)) if len(t_pos) > 1 else 1.0/60.0
    occupancy, _, _ = np.histogram2d(x_cm, y_cm, bins=[x_edges, y_edges])
    occupancy_seconds = occupancy * dt
    
    # 4. Calculate Rate Maps
    output_dir = paths.neural_base / 'post_analysis' / 'rate_maps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from scipy.interpolate import interp1d
    f_x = interp1d(t_pos, x_cm, bounds_error=False, fill_value=np.nan)
    f_y = interp1d(t_pos, y_cm, bounds_error=False, fill_value=np.nan)
    
    print(f"  Calculating maps for {len(unique_clusters)} clusters...")
    
    for cid in tqdm(unique_clusters, desc="Rate Maps"):
            spikes = spike_times_sec[spike_clusters == cid]
            
            # Get position at spike time
            spk_x = f_x(spikes)
            spk_y = f_y(spikes)
            
            valid_spk = ~np.isnan(spk_x)
            spk_x = spk_x[valid_spk]
            spk_y = spk_y[valid_spk]
            
            if len(spk_x) == 0: continue
            
            # Spike Histogram
            spike_hist, _, _ = np.histogram2d(spk_x, spk_y, bins=[x_edges, y_edges])
            
            # Smoothing
            kernel_sigma = sigma_cm / bin_size_cm
            
            smooth_spikes = gaussian_filter1d(gaussian_filter1d(spike_hist, kernel_sigma, axis=0), kernel_sigma, axis=1)
            smooth_occ = gaussian_filter1d(gaussian_filter1d(occupancy_seconds, kernel_sigma, axis=0), kernel_sigma, axis=1)
            
            rate_map = smooth_spikes / (smooth_occ + 1e-3)
            rate_map[smooth_occ < 0.1] = np.nan
            
            # Plot
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(rate_map.T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='jet', aspect='auto')
            plt.colorbar(im, label='Firing Rate (Hz)')
            ax.set_title(f'Cluster {cid} - {unit_types[cid]}')
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            
            plt.savefig(output_dir / f'rate_map_cluster_{cid}.png', dpi=100)
            plt.close(fig)
            
    print(f"  Rate maps saved to {output_dir}")

def analyze_pre_switch_activity(paths: DataPaths, pre_switch_window_sec: float = 10.0, min_trials_before_switch: int = 3, baseline_mode: str = 'stable_block'):
    """
    Analyzes neural changes before behavioral strategy switches.
    
    Examines whether neural activity changes in anticipation of a behavioral switch,
    comparing the period before the first correct choice after a rule change to a baseline.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        pre_switch_window_sec (float): Time window before switch to analyze.
        min_trials_before_switch (int): Minimum trials needed before switch.
        baseline_mode (str): Mode for baseline definition ('fixed_time' or 'stable_block').
                             'fixed_time': uses (pre_start - pre_switch_window_sec) as baseline.
                             'stable_block': uses middle 50% of the previous block (excluding transition periods).
    """
    print("Analyzing behavioral switch anticipation (pre-switch)...")
    
    # --- 1. Load Data ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing corner or condition switch event files.")
        return
    
    base_path = paths.base_path
    event_loader = EventDataLoader(base_path)
    
    # Load corner events
    corner_df_full = event_loader.load(event_path=paths.event_corner, sync_to_dlc=True)
    corner_df_onsets = event_loader.detect_onsets(corner_df_full)
    corner_times_onsets = event_loader.get_event_times(corner_df_onsets, strobe_path=paths.strobe_seconds)
    
    # Get Corner IDs and Filter Invalid (0) Entries
    ids = pd.Series(0, index=corner_df_onsets.index)
    for i in range(1, 4+1):
        col = f'Corner{i}'
        if col in corner_df_onsets.columns:
            mask = corner_df_onsets[col].fillna(0).astype(int) > 0
            ids[mask] = i
    corner_ids_onsets = ids.astype(int).values
        
    # FILTERING: Exclude 0s to preserve transition continuity
    valid_mask = corner_ids_onsets != 0
    corner_ids_onsets = corner_ids_onsets[valid_mask]
    corner_times_onsets = corner_times_onsets[valid_mask]
    
    print(f"  Filtering invalid (0) IDs: Retaining {len(corner_ids_onsets)} valid events.")   
    print(f"First 10 corner IDs: {corner_ids_onsets[:10]}")
        
    # Load switch data
    rule_changes = corner_df_full["CW"].diff().fillna(0) != 0
    rule_changes.iloc[0] = True 
    switch_df_raw = corner_df_full[rule_changes]
    switch_times = event_loader.get_event_times(switch_df_raw, strobe_path=paths.strobe_seconds)
    
    # Identify Behavioral Switch Point
    switch_points = _get_behavioral_switch_points(
        switch_times, corner_times_onsets, corner_ids_onsets, 
        corner_df_full, corner_df_onsets, [1, 2, 4, 3], # Default order
        event_loader, paths.strobe_seconds
    )
    
    print(f"  Identified {len(switch_points)} behavioral switch points.")
    
    # --- 2. Define Pre-Switch and Baseline Periods ---
    pre_switch_segments = []
    baseline_segments = []
    
    for i, pt in enumerate(switch_points):
        behavioral_switch_time = pt['decision_time']
        pre_start = behavioral_switch_time - pre_switch_window_sec
        pre_end = behavioral_switch_time
        n_trials_pre = np.sum((corner_times_onsets >= pre_start) & (corner_times_onsets < pre_end))
        if n_trials_pre >= min_trials_before_switch and pre_start > 0:
            baseline_start = None
            baseline_end = None
            if baseline_mode == 'stable_block':
                if i > 0:
                    prev_switch_time = switch_points[i-1]['decision_time']
                    block_start = prev_switch_time
                    block_end = behavioral_switch_time
                    block_trials_indices = np.where((corner_times_onsets >= block_start) & (corner_times_onsets < block_end))[0]
                    
                    if len(block_trials_indices) > 6: 
                        valid_indices = block_trials_indices[3:-3] 
                        if len(valid_indices) > 0:
                            t_start_valid = corner_times_onsets[valid_indices[0]]
                            t_end_valid = corner_times_onsets[valid_indices[-1]]
                            stable_duration = t_end_valid - t_start_valid
                            margin = stable_duration * 0.25
                            baseline_start = t_start_valid + margin
                            baseline_end = t_end_valid - margin
                            
                            if baseline_end > pre_start:
                                baseline_end = pre_start
                                if baseline_start >= baseline_end:
                                     baseline_start = None 
                
                if baseline_start is None:
                     print(f"  Warning: Could not determine stable baseline for switch {i}. Falling back to fixed time offset.")
            
            if baseline_start is None: 
                baseline_start = pre_start - pre_switch_window_sec
                baseline_end = pre_start
            
            if baseline_start > 0 and (baseline_end > baseline_start):
                 pre_switch_segments.append((pre_start, pre_end))
                 baseline_segments.append((baseline_start, baseline_end))
    
    print(f"  Found {len(pre_switch_segments)} valid pre-switch periods.")
    
    if len(pre_switch_segments) < 2:
        print("  Not enough pre-switch periods. Aborting.")
        return
    
    # --- 3. Load Spike Data ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return
    
    # --- 4. Calculate Firing Rates ---
    results = {}
    
    total_pre_duration = sum(e - s for s, e in pre_switch_segments)
    total_baseline_duration = sum(e - s for s, e in baseline_segments)
    
    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        # Pre-switch rate
        rates_pre = []
        for s, e in pre_switch_segments:
            duration = e - s
            if duration > 0:
                count = np.sum((cluster_spikes >= s) & (cluster_spikes < e))
                rates_pre.append(count / duration)
        
        rates_pre = np.array(rates_pre)
        rate_pre = np.mean(rates_pre) if len(rates_pre) > 0 else 0
        
        # Baseline rate
        rates_baseline = []
        for s, e in baseline_segments:
            duration = e - s
            if duration > 0:
                count = np.sum((cluster_spikes >= s) & (cluster_spikes < e))
                rates_baseline.append(count / duration)

        rates_baseline = np.array(rates_baseline)
        rate_baseline = np.mean(rates_baseline) if len(rates_baseline) > 0 else 0
        
        # Pre-switch modulation index
        if rate_pre + rate_baseline > 0:
            pre_switch_index = (rate_pre - rate_baseline) / (rate_pre + rate_baseline)
        else:
            pre_switch_index = 0
            
        # --- Statistical Test (Mann-Whitney U) ---
        p_val = np.nan
        stat = np.nan
        if len(rates_pre) > 0 and len(rates_baseline) > 0:
            try:
                from scipy import stats
                stat, p_val = stats.mannwhitneyu(rates_pre, rates_baseline, alternative='two-sided')
            except Exception:
                pass
        
        results[cid] = {
            'rate_pre_switch': rate_pre,
            'rate_baseline': rate_baseline,
            'pre_switch_modulation_index': pre_switch_index,
            'p_value': p_val,
            'statistic': stat,
            'type': unit_types.get(cid, 'Unknown')
        }
    
    # --- 5. Save Results ---
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'pre_switch_activity.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Generate Swarm Plot
    swarm_path = output_dir / 'pre_switch_activity_swarm.png'
    _plot_metric_swarm(df_results, 'pre_switch_modulation_index', swarm_path, 
                       "Pre-Switch Activity (Behavior Aligned)", "Modulation Index")

def analyze_outcome_encoding(paths: DataPaths, time_window_ms: int = 200):
    """
    Analyzes neural encoding of trial outcome (Reward vs Error/Omission).
    window for FR is from event time to 200ms after the event time.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_window_ms (int): The window in milliseconds around the outcome event.
    """
    print("Analyzing outcome encoding (Reward vs Error)...")
    window_sec = time_window_ms / 1000.0

    # --- 1. Load Outcome Data ---
    try:
        event_loader = EventDataLoader(paths.base_path)
        dlc_loader = DLCDataLoader(paths.base_path)
        
        reward_df_raw = event_loader.load(
            event_path=paths.event_reward,
            sync_to_dlc=True,
            dlc_data=dlc_loader.load(paths.dlc_h5)
        )
        
        # Explicitly filter for 'Water' column if it exists in a combined file
        target_column = 'Water' if 'Water' in reward_df_raw.columns else 'water'
        reward_df_onsets = event_loader.detect_onsets(reward_df_raw, target_column=target_column)
        reward_times = event_loader.get_event_times(reward_df_onsets, strobe_path=paths.strobe_seconds)
        reward_df = reward_df_onsets
        
        if len(reward_df) == 0:
            print(f"  Error: No outcome events found in {paths.event_reward}")
            return

        # --- Filter for First Lick Per Port Arrival using Corner Columns ---
        # Logic: 
        # 1. Determine "Current Port" from Corner columns in the RAW dataframe (frame-by-frame).
        # 2. Assign a unique "Visit ID" to each continuous port stay.
        # 3. Identify Lick Onsets.
        # 4. Keep only the first Lick Onset for each Visit ID.
        
        # We need to go back to reward_df_raw to resolve Visits *before* filtering to just sparse events
        # Check for Corner columns in reward_df_raw
        if any('Corner' in c for c in reward_df_raw.columns):
            print(f"  Using Corner columns to define Port Arrivals...")
            
            # 1. Map timestamps
            strobe_times = np.load(paths.strobe_seconds, mmap_mode='r').flatten()
            indices = reward_df_raw['Index'].values if 'Index' in reward_df_raw.columns else reward_df_raw.index.values
            valid_mask = (indices >= 0) & (indices < len(strobe_times))
            
            df_valid = reward_df_raw[valid_mask].copy()
            df_valid['timestamp'] = strobe_times[indices[valid_mask].astype(int)]
            
            # 2. Determine Port Visits
            df_valid['port'] = event_loader.infer_port_id(df_valid)
            visit_id = (df_valid['port'] != df_valid['port'].shift()).cumsum()
            df_valid['visit_id'] = visit_id
            
            in_port = df_valid[df_valid['port'] > 0]
            if in_port.empty:
                print("  No port visits found.")
                return
                
            df_visits = in_port.groupby('visit_id').agg(
                port=('port', 'first'),
                start_time=('timestamp', 'first'),
                end_time=('timestamp', 'last')
            ).reset_index()
            
            water_col = "Water"
            if water_col in df_valid.columns:
                water_agg = in_port.groupby('visit_id')[water_col].apply(lambda x: x.fillna(0).astype(bool).any()).reset_index()
                df_visits = df_visits.merge(water_agg, on='visit_id')
                df_visits.rename(columns={water_col: 'is_rewarded'}, inplace=True)
            else:
                df_visits['is_rewarded'] = False
                
            print(f"  Identified {len(df_visits)} Port Visits.")
            
            # 3. Load Lick Seconds
            lick_seconds_path = paths.kilosort_dir / 'licking_seconds.npy'
            if not lick_seconds_path.exists():
                print(f"  Error: {lick_seconds_path} not found. Cannot perform Lick analysis.")
                return
            
            print(f"  Loading Lick Timestamps from {lick_seconds_path}...")
            lick_seconds = np.load(lick_seconds_path)
            
            # 4. Find First Lick in each Visit
            start_idx = np.searchsorted(lick_seconds, df_visits['start_time'].values)
            valid_licks = start_idx < len(lick_seconds)
            
            df_visits['first_lick'] = np.nan
            if np.any(valid_licks):
                potential_licks = lick_seconds[start_idx[valid_licks]]
                valid_time = potential_licks <= df_visits['end_time'].values[valid_licks]
                
                # Assign only valid times
                extracted_licks = np.full(np.sum(valid_licks), np.nan)
                extracted_licks[valid_time] = potential_licks[valid_time]
                df_visits.loc[valid_licks, 'first_lick'] = extracted_licks
            
            # Drop visits without a lick
            df_visits = df_visits.dropna(subset=['first_lick'])
            print(f"  Found {len(df_visits)} First Licks matching valid Port Visits.")
            
            if df_visits.empty:
                print("  No licks found within port visits.")
                return
                
            # 5. Construct Final DataFrame
            reward_df = pd.DataFrame({
                'timestamp': df_visits['first_lick'],
                'is_rewarded': df_visits['is_rewarded'],
                'port_id': df_visits['port'],
                'Water': df_visits['is_rewarded']  # Match downstream
            })
            reward_times = reward_df['timestamp'].values
            
        else:
            print("  Warning: Corner columns not found. Cannot determine Port Visits.")
            return
            
        if len(reward_df) == 0:
            print("  No applicable outcome events found. Aborting.")
            return

        # Check for Water column
        water_col = "Water"
        
        if water_col and water_col in reward_df.columns:
            is_rewarded = reward_df[water_col].fillna(False).astype(bool)
            rewarded_times = reward_times[is_rewarded]
            unrewarded_times = reward_times[~is_rewarded]
            
            print(f"  Loaded {len(rewarded_times)} Rewarded trials and {len(unrewarded_times)} Unrewarded/Error trials.")
        else:
            print(f"  Warning: 'Water' column not found. Treating all events as Rewarded.")
            rewarded_times = reward_times
            unrewarded_times = np.array([])
            
    except Exception as e:
        print(f"  Error loading or processing outcome data: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(rewarded_times) == 0 and len(unrewarded_times) == 0:
        print("  No outcome events found. Aborting analysis.")
        return

    # --- 2. Load Spike Data ---
    base_path = paths.neural_base.parent if paths.neural_base else Path('.')
    base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
    spike_loader = SpikeDataLoader(base_path)
    spike_data = spike_loader.load(paths.kilosort_dir)
    
    spike_times_sec = spike_data['spike_times_sec']
    spike_clusters = spike_data['spike_clusters']
    unique_clusters = spike_data['unique_clusters']
    unit_types = spike_data['unit_types']
    unit_labels = spike_data['unit_labels']

    # --- 3. Calculate Firing Rates for Each Condition ---
    results = {}
    for cid in unique_clusters:
        cluster_spike_times = spike_times_sec[spike_clusters == cid]
        
        # --- Rewarded Trials ---
        rates_rewarded = []
        if len(rewarded_times) > 0:
            for event_time in rewarded_times:
                start_time = event_time
                end_time = event_time + window_sec
                spk_count = np.sum((cluster_spike_times >= start_time) & (cluster_spike_times < end_time))
                rates_rewarded.append(spk_count / window_sec)
            
            rate_rewarded = np.mean(rates_rewarded)
            rates_rewarded = np.array(rates_rewarded)
        else:
            rate_rewarded = 0
            rates_rewarded = np.array([])

        # --- Unrewarded Trials ---
        rates_unrewarded = []
        if len(unrewarded_times) > 0:
            for event_time in unrewarded_times:
                start_time = event_time
                end_time = event_time + window_sec
                spk_count = np.sum((cluster_spike_times >= start_time) & (cluster_spike_times < end_time))
                rates_unrewarded.append(spk_count / window_sec)

            rate_unrewarded = np.mean(rates_unrewarded)
            rates_unrewarded = np.array(rates_unrewarded)
        else:
            rate_unrewarded = np.nan
            rates_unrewarded = np.array([])

        # --- Outcome Modulation Index ---
        # (Reward - Error) / (Reward + Error)
        if not np.isnan(rate_unrewarded) and (rate_rewarded + rate_unrewarded > 0):
            mod_index = (rate_rewarded - rate_unrewarded) / (rate_rewarded + rate_unrewarded)
        else:
            mod_index = np.nan
            
        # --- Statistical Test (Mann-Whitney U) ---
        p_val = np.nan
        stat = np.nan
        if len(rates_rewarded) > 0 and len(rates_unrewarded) > 0:
            try:
                from scipy import stats
                stat, p_val = stats.mannwhitneyu(rates_rewarded, rates_unrewarded, alternative='two-sided')
            except Exception:
                pass

        results[cid] = {
            'firing_rate_reward': rate_rewarded,
            'firing_rate_error': rate_unrewarded,
            'outcome_modulation_index': mod_index,
            'p_value': p_val,
            'statistic': stat,
            'type': unit_types.get(cid, 'Unknown')
        }
        
    # --- 4. Save and Display Results ---
    print("\n  Outcome encoding analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'

    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'outcome_encoding.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")

    # Generate Heatmap
    # Generate Swarm Plot
    swarm_path = output_dir / 'outcome_encoding_swarm.png'
    _plot_metric_swarm(df_results, 'outcome_modulation_index', swarm_path, 
                       "Outcome Encoding (Reward vs Error)", "Modulation Index")

    # Shank Plot
    _plot_shank_location(
        df_results,
        'outcome_modulation_index',
        output_dir / 'outcome_shank_map.png',
        'Outcome Encoding',
        paths=paths,
        p_val_col='p_value',
        significance_threshold=0.05
    )

def analyze_reward_magnitude_encoding(paths: DataPaths, time_window_ms: int = 200):
    """
    Analyzes neural responses to the first vs. second reward at the same port.

    Compares firing rates in a window following reward delivery to see if neurons
    differentiate between the first and second rewards, which might have different values.
    window is event time to 200ms after the event time.
    calculate magnitude if the neuron fired both first and seconds reward window

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_window_ms (int): The window in milliseconds around the reward event.
    """
    print("Analyzing reward magnitude encoding (first vs. second reward)...")
    window_sec = time_window_ms / 1000.0

    # --- 1. Load Reward Data ---
    if not paths.event_reward or not paths.event_reward.exists():
        print(f"  Error: Reward event file not found at {paths.event_reward}.")
        return

    try:    
        event_loader = EventDataLoader(paths.base_path)
        dlc_loader = DLCDataLoader(paths.base_path)
        dlc_data = dlc_loader.load(paths.dlc_h5)
                
        reward_df_raw = event_loader.load(
            event_path=paths.event_reward,
            sync_to_dlc=(dlc_data is not None),
            dlc_data=dlc_data
        )

        target_column = 'Water'

        # 1. Map timestamps for all frames
        strobe_times = np.load(paths.strobe_seconds, mmap_mode='r').flatten()
        indices = reward_df_raw['Index'].values if 'Index' in reward_df_raw.columns else reward_df_raw.index.values
        valid_mask = (indices >= 0) & (indices < len(strobe_times))
        
        df_valid = reward_df_raw[valid_mask].copy()
        df_valid['timestamp'] = strobe_times[indices[valid_mask].astype(int)]
        
        # 2. Determine Port Visits
        df_valid['port'] = event_loader.infer_port_id(df_valid)
        visit_id = (df_valid['port'] != df_valid['port'].shift()).cumsum()
        df_valid['visit_id'] = visit_id
        
        # Filter only when Water is delivered
        reward_onsets = event_loader.detect_onsets(df_valid, target_column=target_column)
        
        if len(reward_onsets) == 0:
            print("  Error: No reward onsets found.")
            return
            
        # Group by visit_id to find the first and second reward in the SAME visit sequence!
        reward_onsets = reward_onsets[reward_onsets['port'] > 0] # Must be at a port
        reward_onsets['reward_order'] = reward_onsets.groupby('visit_id').cumcount()
        
        first_rewards = reward_onsets[reward_onsets['reward_order'] == 0]['timestamp'].values
        second_rewards = reward_onsets[reward_onsets['reward_order'] == 1]['timestamp'].values
        
        print(f"  Loaded {len(first_rewards)} first-reward events and {len(second_rewards)} second-reward events.")

    except Exception as e:
        print(f"  Error loading or processing reward data: {e}.")
        import traceback
        traceback.print_exc()
        return

    if len(first_rewards) == 0 or len(second_rewards) == 0:
        print("  Not enough data for both first and second rewards. Aborting analysis.")
        return

    # --- 2. Load Spike Data ---
    try:
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data['unit_labels']
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return

    # --- 3. Calculate Firing Rates for Each Condition ---
    post_results = {}
    pre_results = {}
    
    for cid in unique_clusters:
        cluster_spike_times = spike_times_sec[spike_clusters == cid]
        
        # --- First Rewards ---
        post_rates_first = []
        pre_rates_first = []
        for event_time in first_rewards:
            start_time = event_time
            end_time = event_time + window_sec
            post_count = np.sum((cluster_spike_times >= start_time) & (cluster_spike_times < end_time))
            post_rates_first.append(post_count / window_sec)

            start_time = event_time - window_sec
            end_time = event_time
            pre_count = np.sum((cluster_spike_times >= start_time) & (cluster_spike_times < end_time))
            pre_rates_first.append(pre_count / window_sec)

        post_rates_first = np.array(post_rates_first)
        pre_rates_first = np.array(pre_rates_first)
        post_rate_first_reward = np.mean(post_rates_first) if len(post_rates_first) > 0 else 0
        pre_rate_first_reward = np.mean(pre_rates_first) if len(pre_rates_first) > 0 else 0

        # --- Second Rewards ---
        post_rates_second = []
        pre_rates_second = []
        for event_time in second_rewards:
            start_time = event_time
            end_time = event_time + window_sec
            post_count = np.sum((cluster_spike_times >= start_time) & (cluster_spike_times < end_time))
            post_rates_second.append(post_count / window_sec)

            start_time = event_time - window_sec
            end_time = event_time
            pre_count = np.sum((cluster_spike_times >= start_time) & (cluster_spike_times < end_time))
            pre_rates_second.append(pre_count / window_sec)

        post_rates_second = np.array(post_rates_second)
        pre_rates_second = np.array(pre_rates_second)
        post_rate_second_reward = np.mean(post_rates_second) if len(post_rates_second) > 0 else 0
        pre_rate_second_reward = np.mean(pre_rates_second) if len(pre_rates_second) > 0 else 0

        # --- Magnitude Modulation Index ---
        if post_rate_second_reward > 0.05 and post_rate_first_reward > 0.05:
            post_magnitude_index = (post_rate_second_reward - post_rate_first_reward) / (post_rate_second_reward + post_rate_first_reward)
        else:
            post_magnitude_index = 0

        if pre_rate_second_reward > 0.05 and pre_rate_first_reward > 0.05:
            pre_magnitude_index = (pre_rate_second_reward - pre_rate_first_reward) / (pre_rate_second_reward + pre_rate_first_reward)
        else:
            pre_magnitude_index = 0
            
        # --- Statistical Test (Mann-Whitney U) ---
        post_p_val = np.nan
        post_stat = np.nan
        if len(post_rates_first) > 0 and len(post_rates_second) > 0:
            try:
                from scipy import stats
                post_stat, post_p_val = stats.mannwhitneyu(post_rates_first, post_rates_second, alternative='two-sided')
            except Exception:
                pass
        post_is_significant = post_p_val < 0.05 if not np.isnan(post_p_val) else False
        
        pre_p_val = np.nan
        pre_stat = np.nan
        if len(pre_rates_first) > 0 and len(pre_rates_second) > 0:
            try:
                from scipy import stats
                pre_stat, pre_p_val = stats.mannwhitneyu(pre_rates_first, pre_rates_second, alternative='two-sided')
            except Exception:
                pass
        pre_is_significant = pre_p_val < 0.05 if not np.isnan(pre_p_val) else False
            
        post_results[cid] = {
            'post_firing_rate_first_reward': post_rate_first_reward,
            'post_firing_rate_second_reward': post_rate_second_reward,
            'post_magnitude_modulation_index': post_magnitude_index,
            'post_p_value': post_p_val,
            'post_is_significant': post_is_significant,
            'post_statistic': post_stat,
            'type': unit_types.get(cid, 'Unknown')
        }

        pre_results[cid] = {
            'pre_firing_rate_first_reward': pre_rate_first_reward,
            'pre_firing_rate_second_reward': pre_rate_second_reward,
            'pre_magnitude_modulation_index': pre_magnitude_index,
            'pre_p_value': pre_p_val,
            'pre_is_significant': pre_is_significant,
            'pre_statistic': pre_stat,
            'type': unit_types.get(cid, 'Unknown')
        }
        
    # --- 4. Save and Display Results ---
    print("\n  Reward magnitude analysis complete.")
    post_df_results = pd.DataFrame.from_dict(post_results, orient='index')
    post_df_results.index.name = 'cluster_id'
    pre_df_results = pd.DataFrame.from_dict(pre_results, orient='index')
    pre_df_results.index.name = 'cluster_id'

    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    post_output_path = output_dir / 'reward_magnitude_encoding_postwindow.csv'
    pre_output_path = output_dir / 'reward_magnitude_encoding_prewindow.csv'
    post_df_results.to_csv(post_output_path)
    pre_df_results.to_csv(pre_output_path)
    print(f"  Results saved to {post_output_path} and {pre_output_path}")

    # Generate Swarm Plot
    _plot_metric_swarm(post_df_results, col_name='post_magnitude_modulation_index', output_path=output_dir/'reward_magnitude_swarm_postwindow.png', 
                       title=" FR diff First vs Second Reward (0~200ms)", ylabel="Modulation Index",
                       p_val_col='post_p_value', outcome_col='post_is_significant')
    _plot_metric_swarm(pre_df_results, col_name='pre_magnitude_modulation_index', output_path=output_dir/'reward_magnitude_swarm_prewindow.png', 
                       title="FR diff First vs Second Reward (-200ms~0ms)", ylabel="Modulation Index",
                       p_val_col='pre_p_value', outcome_col='pre_is_significant')

    # Generate Shank maps
    _plot_shank_location(post_df_results,'post_magnitude_modulation_index', output_dir/'reward_magnitude_shank_map_postwindow.png',
        title='FR diff First vs Second Reward (0~200ms)', paths=paths, p_val_col='post_p_value', significance_threshold=0.05)
    _plot_shank_location(pre_df_results,'pre_magnitude_modulation_index', output_dir/'reward_magnitude_shank_map_prewindow.png',
        title='FR diff First vs Second Reward (-200ms~0ms)', paths=paths, p_val_col='pre_p_value', significance_threshold=0.05)


def analyze_reward_history(paths: DataPaths, max_duration_sec: int = 30):
    """
    Analyzes how recent reward history affects firing rates during navigation.

    Compares firing rates on trials (port-to-port trajectories) that follow
    a rewarded trial vs. trials that follow an unrewarded trial.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        max_duration_sec (int): Max duration for a port-to-port trajectory to be
                                considered a valid trial.
    """
    import scipy.stats as stats
    print("Analyzing reward history effects...")
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)

    # --- 1. Load Corner and Reward Data ---    
    try:
        event_loader = EventDataLoader(paths.base_path)
        dlc_loader = DLCDataLoader(paths.base_path)
        dlc_data = dlc_loader.load(paths.dlc_h5)
        
        # Load Corner Events
        corner_df_raw = event_loader.load(
            paths.event_corner, 
            sync_to_dlc=(dlc_data is not None),
            dlc_data=dlc_data
        )
        
        
        # 1. Map timestamps for all frames
        corner_df_raw['timestamp'] = event_loader.get_event_times(corner_df_raw, strobe_path=paths.strobe_seconds)
        
        # 2. Determine Port Visits
        corner_df_raw['port'] = event_loader.infer_port_id(corner_df_raw).values
        visit_id = (corner_df_raw['port'] != corner_df_raw['port'].shift()).cumsum()
        corner_df_raw['visit_id'] = visit_id
        
        # 3. Filter valid visits and get onsets
        valid_visits = corner_df_raw[corner_df_raw['port'] > 0]
        visit_onsets = valid_visits.groupby('visit_id').first()
        
        corner_times = visit_onsets['timestamp'].values
        corner_ids = visit_onsets['port'].values
        
        print(f"  Filtering invalid (0) IDs: Retaining {len(corner_times)} valid corner visits (out of {len(corner_df_raw)} frames).")
        reward_df_raw = event_loader.load(
            event_path=paths.event_reward,
            sync_to_dlc=(dlc_data is not None),
            dlc_data=dlc_data
        )
        target_column = 'Water'
        reward_onsets = event_loader.detect_onsets(reward_df_raw, target_column=target_column)
        reward_times = event_loader.get_event_times(reward_onsets, strobe_path=paths.strobe_seconds)
        
        print(f"  Loaded {len(corner_times)} corner events and {len(reward_times)} reward events.")

    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Define Trials and Determine their Outcome ---
    if len(corner_times) == 0:
        print("  Error: No valid corner events found.")
        return
    corner_rewarded = np.zeros(len(corner_times), dtype=bool)
    
    for i in range(len(corner_times) - 1):
        t_start = corner_times[i]
        t_end = corner_times[i+1]
        if np.any((reward_times >= t_start) & (reward_times < t_end)):
            corner_rewarded[i] = True
            
    print(f"  Identified {np.sum(corner_rewarded)} rewarded corners out of {len(corner_times)} total.")

    bouts = _get_kinematic_states(paths)

    trials = []
    
    for i in range(len(corner_times) - 1):
        start_time, end_time = corner_times[i], corner_times[i+1]
        duration = end_time - start_time
        
        start_id = corner_ids[i]
        end_id = corner_ids[i+1]
        
        if 0 < duration < max_duration_sec:
            candidate_bouts = []
            for b in bouts:
                if b['start_time'] >= start_time and b['start_time'] <= end_time:
                    candidate_bouts.append(b)
            if not candidate_bouts:
                continue
            best_bout = max(candidate_bouts, key=lambda x: x['duration'])
            
            bout_start = best_bout['start_time']
            bout_end = best_bout['end_time']

            trials.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'corner_idx': i,
                'bout_start': bout_start,
                'bout_end': bout_end
            })
    
    if len(trials) < 2:
        print("  Fewer than 2 valid trials found. Aborting analysis.")
        return
    
    print(f"  Defined {len(trials)} trials based on trajectories.")

    # --- 3. Load Spike Data ---
    base_path = paths.neural_base.parent if paths.neural_base else Path('.')
    base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
    spike_loader = SpikeDataLoader(base_path)
    spike_data = spike_loader.load(paths.kilosort_dir)
    
    spike_times_sec = spike_data['spike_times_sec']
    spike_clusters = spike_data['spike_clusters']
    unique_clusters = spike_data['unique_clusters']
    unit_types = spike_data['unit_types']
    unit_labels = spike_data['unit_labels']

    # --- 4. Calculate Firing Rates Based on Previous Trial Outcome ---
    post_reward_segments = []
    post_no_reward_segments = []

    for i in range(len(trials)):
        current_trial = trials[i]
        is_post_reward = corner_rewarded[current_trial['corner_idx']]
        segment = (current_trial['bout_start'], current_trial['bout_end'])
        
        if is_post_reward:
            post_reward_segments.append(segment)
        else:
            post_no_reward_segments.append(segment)

    print(f"  Found {len(post_reward_segments)} trials following a reward.")
    print(f"  Found {len(post_no_reward_segments)} trials following no reward.")

    # --- 5. Behavioral Kinematics Analysis ---
    df_dlc = dlc_data if dlc_data is not None else None
    
    if df_dlc is not None:
            bodypart = 'Snout'
            if not any(bodypart in str(c) for c in df_dlc.columns):
                bp_candidates = set([col[0] if isinstance(col, tuple) else col.split('_')[0] for col in df_dlc.columns if 'x' in str(col) or 'y' in str(col)])
                if bp_candidates:
                    bodypart = list(bp_candidates)[0]
            
            x_col = next((c for c in df_dlc.columns if bodypart in str(c) and 'x' in str(c)), None)
            y_col = next((c for c in df_dlc.columns if bodypart in str(c) and 'y' in str(c)), None)
            
            velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, video_fs=60, px_per_cm=30.0, strobe_path=paths.strobe_seconds)
    else:
            velocity = np.array([])
            velocity_times = np.array([])
            x_col = None
            y_col = None
    
    # Load Licking Data
    lick_times = None
    lick_seconds_path = paths.kilosort_dir / 'licking_seconds.npy'
    if lick_seconds_path.exists():
        lick_times = np.load(lick_seconds_path)
    
    # Calculate Metrics for each trial
    traj_coords = defaultdict(list)
    temp_trials_data = []
    
    dlc_times = velocity_times

    for i in range(len(trials)):
        t = trials[i]
        prev_rewarded = corner_rewarded[t['corner_idx']]
        
        # 1. Speed
        mean_speed = np.nan
        resampled_xy = None
        traj_key = (corner_ids[i], corner_ids[i+1])
        if len(dlc_times) > 0:
            n_points = min(len(dlc_times), len(velocity))
            mask = (dlc_times[:n_points] >= t['start_time']) & (dlc_times[:n_points] <= t['end_time'])
            
            if np.any(mask):
                mean_speed = np.nanmean(velocity[:n_points][mask])
            
            # 2. Trajectory Coordinates
            if x_col and y_col:
                mask_full = (dlc_times >= t['start_time']) & (dlc_times <= t['end_time'])
                if len(dlc_times) == len(df_dlc):
                        df_safe = df_dlc.loc[mask_full]
                        xs = df_safe[x_col].values if x_col in df_safe else np.array([])
                        ys = df_safe[y_col].values if y_col in df_safe else np.array([])
                        
                        if len(xs) > 10:
                            dists = np.linspace(0, 1, len(xs))
                            target_dists = np.linspace(0, 1, 100)
                            f_x = interp1d(dists, xs, kind='linear')
                            f_y = interp1d(dists, ys, kind='linear')
                            resampled_xy = np.column_stack((f_x(target_dists), f_y(target_dists)))
                            traj_coords[traj_key].append(resampled_xy)
        
        # 3. Licking Count
        lick_freq = np.nan
        lick_len = np.nan
        
        if lick_times is not None:
            l_start = t['end_time']
            l_end = l_start + 3.0
            
            b_mask = (lick_times >= l_start) & (lick_times < l_end)
            bout_licks = lick_times[b_mask]
            
            if len(bout_licks) > 0:
                lick_freq = len(bout_licks) / 3.0
                lick_len = bout_licks[-1] - bout_licks[0]
            else:
                lick_freq = 0
                lick_len = 0
        
        temp_trials_data.append({
            'prev_rewarded': prev_rewarded,
            'mean_speed': mean_speed,
            'traj_key': traj_key,
            'xy_resampled': resampled_xy,
            'lick_freq': lick_freq,
            'lick_len': lick_len
        })

    mean_paths = {}
    for k, valid_paths in traj_coords.items():
        if len(valid_paths) > 0:
            mean_paths[k] = np.mean(np.array(valid_paths), axis=0)
            
    kinematic_results = []
    for d in temp_trials_data:
        mean_path = mean_paths.get(d['traj_key'])
        deviation = np.nan
        if mean_path is not None and d['xy_resampled'] is not None:
            diffs = np.linalg.norm(d['xy_resampled'] - mean_path, axis=1)
            deviation = np.mean(diffs)
        
        kinematic_results.append({
            'prev_rewarded': d['prev_rewarded'],
            'mean_speed': d['mean_speed'],
            'trajectory_deviation': deviation,
            'lick_frequency': d['lick_freq'],
            'lick_length': d['lick_len']
        })
        
    k_df = pd.DataFrame(kinematic_results)
    metrics = ['mean_speed', 'trajectory_deviation', 'lick_frequency', 'lick_length']
    
    print("\n  --- Kinematic Analysis Results (Post-Reward vs Post-No-Reward) ---")
    stats_results = {}
    
    for met in metrics:
        if met not in k_df.columns or k_df[met].isnull().all(): continue
        
        g1 = k_df[k_df['prev_rewarded'] == True][met].dropna()
        g2 = k_df[k_df['prev_rewarded'] == False][met].dropna()
        
        if len(g1) > 2 and len(g2) > 2:
            stat, p = stats.mannwhitneyu(g1, g2)
            mean1, mean2 = g1.mean(), g2.mean()
            print(f"    {met}: Reward={mean1:.2f}, NoReward={mean2:.2f} | p={p:.4f}")
            
            stats_results[met] = {'p': p, 'mean_reward': mean1, 'mean_noreward': mean2}
        else:
                print(f"    {met}: Not enough data.")
    
    k_output = output_dir / 'reward_history_kinematics.csv'
    k_df.to_csv(k_output)
    print(f"  Kinematics data saved to {k_output}")

    # --- 6. Neural Analysis (Existing) ---
    results = {}
    
    for cid in unique_clusters:
        cts = spike_times_sec[spike_clusters == cid]
        
        rates_post_reward = []
        for s, e in post_reward_segments:
            duration = e - s
            if duration > 0:
                count = np.sum((cts >= s) & (cts < e))
                rates_post_reward.append(count / duration)
                
        rates_post_reward = np.array(rates_post_reward)
        rate_post_reward = np.mean(rates_post_reward) if len(rates_post_reward) > 0 else 0
        
        rates_post_no_reward = []
        for s, e in post_no_reward_segments:
            duration = e - s
            if duration > 0:
                count = np.sum((cts >= s) & (cts < e))
                rates_post_no_reward.append(count / duration)
                
        rates_post_no_reward = np.array(rates_post_no_reward)
        rate_post_no_reward = np.mean(rates_post_no_reward) if len(rates_post_no_reward) > 0 else 0
        
        if rate_post_reward + rate_post_no_reward > 0:
            history_index = (rate_post_reward - rate_post_no_reward) / (rate_post_reward + rate_post_no_reward)
        else:
            history_index = 0
            
        p_val = np.nan
        stat = np.nan
        is_significant = False
        if len(rates_post_reward) > 0 and len(rates_post_no_reward) > 0:
            try:
                stat, p_val = stats.mannwhitneyu(rates_post_reward, rates_post_no_reward, alternative='two-sided')
                is_significant = p_val < 0.05 if not np.isnan(p_val) else False
            except Exception:
                pass
            
        results[cid] = {
            'firing_rate_after_reward': rate_post_reward,
            'firing_rate_after_no_reward': rate_post_no_reward,
            'reward_history_index': history_index,
            'p_value': p_val,
            'statistic': stat,
            'is_significant': is_significant,
            'type': unit_types.get(cid, 'Unknown')
        }

    # --- 7. Save and Display Neural Results ---
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    output_path = output_dir / 'reward_history_effects.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")

    # Generate Swarm Plot
    _plot_metric_swarm(df_results, col_name='reward_history_index', output_path=output_dir / 'reward_history_swarm.png', 
                       title="Reward History (After Reward vs No Reward)", ylabel="Modulation Index", p_val_col='p_value', outcome_col="is_significant")
    _plot_shank_location(df_results, val_col='reward_history_index', output_path=output_dir / 'reward_history_shank_location.png',
                         title="Reward History (After Reward vs No Reward)", paths=paths, p_val_col='p_value', significance_threshold=0.05)


def analyze_history_dependence_glm(paths: DataPaths, n_back: int = 5, corner_order_cw: list = [1, 2, 4, 3]):
    """
    Analyzes neural and kinematic dependence on trial history using a GLM.

    Constructs a design matrix with N-back features:
    - Reward Outcome (1 or 0) for t-1...t-N
    - Choice Accuracy/Direction (Correct=1/Error=0 or CW/CCW) for t-1...t-N
    - Lick Duration/Frequency (Continuous) for t-1...t-N

    Fits a Linear Regression model to predict:
    1. Current Trial Firing Rate (during navigation)
    2. Current Trial Mean Speed
    3. Current Trial Trajectory Deviation

    Args:
        paths (DataPaths): DataPaths object.
        n_back (int): Number of past trials to include.
        corner_order_cw (list): Order of corners for CW definition.
    """
    print(f"Analyzing history dependence (GLM) with {n_back}-back history...")

    # --- 1. Load Data & Build Trial Table ---
    try:
        from collections import defaultdict
        from scipy.interpolate import interp1d
        
        event_loader = EventDataLoader(paths.base_path)
        dlc_loader = DLCDataLoader(paths.base_path)
        df_dlc = dlc_loader.load(paths.dlc_h5)

        # Load Corner Events
        corner_df_raw = event_loader.load(paths.event_corner, sync_to_dlc=True, dlc_data=df_dlc)
        corner_onsets_df = event_loader.detect_onsets(corner_df_raw)
        corner_ids_full = event_loader.infer_port_id(corner_onsets_df).values
        corner_times_full = event_loader.get_event_times(corner_onsets_df, strobe_path=paths.strobe_seconds)
        
        valid_mask = corner_ids_full != 0
        if len(corner_times_full) == len(corner_ids_full):
            corner_times = corner_times_full[valid_mask]
            corner_ids = corner_ids_full[valid_mask]
        else:
            print("  Warning: corner times and IDs length mismatch.")
            return
            
        # Load Reward Events
        reward_df_raw = event_loader.load(paths.event_reward, sync_to_dlc=True, dlc_data=df_dlc)
        reward_col = next((c for c in ['Water', 'water', 'Reward'] if c in reward_df_raw.columns), None)
        if reward_col:
            reward_onsets = event_loader.detect_onsets(reward_df_raw, target_column=reward_col)
        else:
            print("  Warning: No Water column found. Using all reward events.")
            reward_onsets = event_loader.detect_onsets(reward_df_raw)
            
        reward_times = event_loader.get_event_times(reward_onsets, strobe_path=paths.strobe_seconds)

        # Load Licking Data
        lick_times = None
        lick_path = paths.base_path / "kilosort4/sorter_output" / 'licking_seconds.npy'
        if lick_path.exists():
            lick_times = np.load(lick_path, mmap_mode='r').flatten()
            
        # Load Kinematics (Speed)
        velocity, velocity_times = dlc_loader.calculate_velocity(df_dlc, video_fs=60, px_per_cm=30.0, strobe_path=paths.strobe_seconds)
        
        # Fix DLC times if mismatched
        if len(velocity_times) != len(df_dlc):
             velocity_times = np.arange(len(df_dlc)) / 60.0

        # Construct Trials
        trials = []
        
        # Pre-calculate mean paths for deviation
        traj_coords = defaultdict(list)
        
        bodypart = 'Snout'
        x_col = None
        y_col = None
        if isinstance(df_dlc.columns, pd.MultiIndex):
            bodyparts = list(df_dlc.columns.get_level_values(1).unique())
            if bodyparts and not any(bodypart in c for c in bodyparts):
                bodypart = bodyparts[0]
            scorer = df_dlc.columns.get_level_values(0)[0]
            if (scorer, bodypart, 'x') in df_dlc.columns:
                x_col = (scorer, bodypart, 'x')
                y_col = (scorer, bodypart, 'y')
        else:
            if not any(bodypart in c for c in df_dlc.columns):
                # Fallback to first bodypart
                cols = [c for c in df_dlc.columns if 'x' in c]
                if cols: bodypart = cols[0].replace('x', '')
            x_col = next((c for c in df_dlc.columns if bodypart in c and 'x' in c), None)
            y_col = next((c for c in df_dlc.columns if bodypart in c and 'y' in c), None)
        
        # First Pass: Collect Trajectories
        for i in range(len(corner_times)-1):
            start_t, end_t = corner_times[i], corner_times[i+1]
            if end_t - start_t > 30: continue # Skip long pauses
            
            mask = (velocity_times >= start_t) & (velocity_times <= end_t)
            if not np.any(mask): continue
            
            if x_col and y_col:
                xs = df_dlc.loc[mask, x_col].values
                ys = df_dlc.loc[mask, y_col].values
                if len(xs) > 10:
                    dists = np.linspace(0, 1, len(xs))
                    target = np.linspace(0, 1, 100)
                    fx = interp1d(dists, xs, kind='linear')
                    fy = interp1d(dists, ys, kind='linear')
                    path_xy = np.column_stack((fx(target), fy(target)))
                    
                    key = (corner_ids[i], corner_ids[i+1])
                    traj_coords[key].append(path_xy)
                    
        # Compute Means
        mean_paths = {k: np.mean(np.array(v), axis=0) for k, v in traj_coords.items() if len(v) > 0}
        
        # Second Pass: Build Features
        for i in range(len(corner_times) - 1):
            start_t, end_t = corner_times[i], corner_times[i+1]
            start_id, end_id = corner_ids[i], corner_ids[i+1]
            
            duration = end_t - start_t
            if duration > 30 or duration < 0.5: continue
            
            # Outcome (Reward)
            has_reward = np.any((reward_times >= end_t) & (reward_times <= end_t + 3.0))
            reward_val = 1 if has_reward else 0
            
            rewards_in_window = np.sum((reward_times >= end_t) & (reward_times <= end_t + 3.0))
            
            # Action (Correct/CW?)
            is_cw = _is_move_correct(start_id, end_id, corner_order_cw, True)
            is_ccw = _is_move_correct(start_id, end_id, corner_order_cw, False)
            
            # Encode Action: 1=CW, -1=CCW, 0=Other? 
            action_val = 0
            if is_cw: action_val = 1
            elif is_ccw: action_val = -1
            
            # Lick Duration/Count at Destination
            lick_dur = 0
            if lick_times is not None:
                licks = lick_times[(lick_times >= end_t) & (lick_times <= end_t + 3.0)]
                if len(licks) > 1:
                    lick_dur = licks[-1] - licks[0]
            
            # Kinematics during Run (the 'State' of this trial)
            n_points = min(len(velocity_times), len(velocity))
            safe_times = velocity_times[:n_points]
            safe_velocity = velocity[:n_points]
            
            mask = (safe_times >= start_t) & (safe_times < end_t)
            mean_speed = np.nanmean(safe_velocity[mask]) if np.any(mask) else np.nan
            
            deviation = np.nan
            if mean_paths and (start_id, end_id) in mean_paths:
                mean_p = mean_paths[(start_id, end_id)]
                if np.any(mask):
                     # Slice DF to match mask length (n_points)
                     df_safe = df_dlc.iloc[:n_points]
                     if len(df_safe.loc[mask]) > 10:
                        xs = df_safe.loc[mask, x_col].values
                        ys = df_safe.loc[mask, y_col].values
                        fx = interp1d(np.linspace(0,1,len(xs)), xs)
                        fy = interp1d(np.linspace(0,1,len(ys)), ys)
                        current_p = np.column_stack((fx(np.linspace(0,1,100)), fy(np.linspace(0,1,100))))
                        deviation = np.mean(np.linalg.norm(current_p - mean_p, axis=1))
            
            trials.append({
                'trial_idx': i,
                'start_time': start_t,
                'end_time': end_t,
                'action': action_val,
                'reward': reward_val, 
                'reward_mag': rewards_in_window,
                'lick_dur': lick_dur,
                'speed': mean_speed,
                'deviation': deviation
            })
            
        df_trials = pd.DataFrame(trials)
        if len(df_trials) < n_back + 10:
             print("  Not enough trials for history analysis.")
             return

    except Exception as e:
        print(f"  Error building trial table: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Build Design Matrix (X) and Targets (Y) ---
    valid_indices = range(n_back, len(df_trials))
    
    X = []
    
    # Kinematic Targets
    Y_speed = []
    Y_deviation = []
    
    # Trial indices for Neural Alignment
    target_trial_indices = [] # indices in df_trials
    
    for i in valid_indices:
        # Build history vector
        # Current Trial Controls (Lag 0) to separate movement effects
        current = df_trials.iloc[i]
        
        # Features: [Current_Action, Current_Speed, Current_Deviation]
        # Then History: [Rew_t-1, Act_t-1, Lick_t-1, ...]9*
        row = [current['action'], current['speed'], current['deviation']]
        
        for back in range(1, n_back + 1):
            past = df_trials.iloc[i - back]
            row.extend([past['reward_mag'], past['action'], past['lick_dur']])
            
        X.append(row)
        
        current = df_trials.iloc[i]
        Y_speed.append(current['speed'])
        Y_deviation.append(current['deviation'])
        
        target_trial_indices.append(i)
        
    X = np.array(X) 
    
    if np.isnan(X).any():
        print("  Warning: NaNs in design matrix. Filling with 0.")
        X = np.nan_to_num(X)
        
    # --- 3. Run Regression on Behavior (Kinematics) ---
    print("  Fitting GLM to Behavioral Kinematics...")
    from sklearn.linear_model import Ridge
    
    behavior_results = {}
    
    for name, Y_target in [('Speed', Y_speed), ('Deviation', Y_deviation)]:
        # Filter valid Y
        Y_arr = np.array(Y_target)
        valid_mask = ~np.isnan(Y_arr)
        
        # FIX: For Behavioral Model, Exclude 'Speed' (index 1) and 'Deviation' (index 2) from X
        # X columns: [Action, Speed, Deviation, Hist...]
        # We want: [Action, Hist...]
        # Action is col 0. Hist starts at col 3.
        kept_indices = [0] + list(range(3, X.shape[1]))
        X_beh = X[:, kept_indices]
        
        if np.sum(valid_mask) > 50:
            model = Ridge(alpha=1.0)
            model.fit(X_beh[valid_mask], Y_arr[valid_mask])
            behavior_results[name] = model.coef_
            print(f"    Modeled {name}. R2 = {model.score(X_beh[valid_mask], Y_arr[valid_mask]):.3f}")
        
    # --- 4. Run Regression on Neural Activity ---
    print("  Fitting GLM to Neural Activity...")
    
    try:
        base_path = getattr(paths, 'neural_base_path', getattr(paths, 'base_path', Path('.')))
        kilosort_dir = getattr(paths, 'kilosort_dir', base_path / 'kilosort4' / 'sorter_output')
        
        spike_loader = SpikeDataLoader(base_path)
        spike_data = spike_loader.load(kilosort_dir)
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data['unit_types']
        unit_labels = spike_data.get('unit_labels', {})
        
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        return
    
    neural_coefs = {} # cid -> coefs
    
    # Pre-calculate trial windows
    windows = []
    for idx in target_trial_indices:
        row = df_trials.iloc[idx]
        windows.append((row['start_time'], row['end_time']))
        
    windows = np.array(windows)
    durations = windows[:, 1] - windows[:, 0]
    
    # Check if tqdm is available contextually
    try:
        from tqdm import tqdm
        iterator = tqdm(unique_clusters, desc="Cells")
    except ImportError:
        iterator = unique_clusters
        
    for cid in iterator:
        st = spike_times_sec[spike_clusters == cid]
        
        rates = []
        for i, (start, end) in enumerate(windows):
            count = np.sum((st >= start) & (st < end))
            duration = max(durations[i], 0.001)
            rates.append(count / duration)
        
        rates = np.array(rates)
        
        # Fit Model
        model = Ridge(alpha=1.0)
        model.fit(X, rates)
        # Store Coefs
        neural_coefs[cid] = {
            'coefs': model.coef_,
            'intercept': model.intercept_,
            'score': model.score(X, rates),
            'type': unit_types.get(cid, 'Unknown')
        }
        
    # --- 5. Save and Visualize ---
    output_dir = getattr(paths, 'neural_base', getattr(paths, 'base_path', Path('.'))) / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Save Behavioral Kernels
    beh_data = []
    for target_name, coefs in behavior_results.items():
        # Handle Current Trial (X_beh Index 0 = Action)
        beh_data.append({'Target': target_name, 'Lag': 0, 'Feature': 'Action', 'Weight': coefs[0]})

        # Handle History (Indices 1+ in X_beh corresponding to Indices 3+ in X)
        history_coefs = coefs[1:] 
        for i, val in enumerate(history_coefs):
            lag = (i // 3) + 1
            feat_type = ['Reward', 'Action', 'Lick'][i % 3]
            beh_data.append({
                'Target': target_name,
                'Lag': lag,
                'Feature': feat_type,
                'Weight': val
            })

    pd.DataFrame(beh_data).to_csv(output_dir / 'history_glm_behavior_kernels.csv')
    
    # Save Neural Kernels
    neural_data = []
    for cid, res in neural_coefs.items():
        coefs = res['coefs']
        
        # Current Trial
        neural_data.append({'ClusterID': cid, 'Type': res['type'], 'Lag': 0, 'Feature': 'Action', 'Weight': coefs[0], 'R2': res['score']})
        neural_data.append({'ClusterID': cid, 'Type': res['type'], 'Lag': 0, 'Feature': 'Speed', 'Weight': coefs[1], 'R2': res['score']})
        neural_data.append({'ClusterID': cid, 'Type': res['type'], 'Lag': 0, 'Feature': 'Deviation', 'Weight': coefs[2], 'R2': res['score']}) # Changed from Lick to Deviation for current
        
        # History
        history_coefs = coefs[3:]
        for i, val in enumerate(history_coefs):
            lag = (i // 3) + 1
            feat_type = ['Reward', 'Action', 'Lick'][i % 3]
            neural_data.append({
                'ClusterID': cid,
                'Type': res['type'],
                'Lag': lag,
                'Feature': feat_type,
                'Weight': val,
                'R2': res['score']
            })
            
    df_neural = pd.DataFrame(neural_data)
    df_neural.to_csv(output_dir / 'history_glm_neural_kernels.csv')
    print(f"  Results saved to {output_dir}")
    
    # Plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if beh_data:
        try:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=pd.DataFrame(beh_data), x='Lag', y='Weight', hue='Feature', style='Target', markers=True)
            plt.title(f"Behavioral History Kernels (GLM)")
            plt.axhline(0, color='k', linestyle='--')
            plt.ylabel("Regression Weight")
            plt.tight_layout()
            plt.savefig(output_dir / 'history_glm_behavior.png')
            plt.close()
        except:
            pass
        
    if not df_neural.empty:
        try:
            plt.figure(figsize=(12, 8))
            # Filter for reasonable R2
            good_fits = df_neural[df_neural['R2'] > 0.01]
            if len(good_fits) > 0:
                good_fits = good_fits.copy()
                good_fits['AbsWeight'] = good_fits['Weight'].abs()
                
                g = sns.catplot(data=good_fits, x='Lag', y='AbsWeight', hue='Type', col='Feature', kind='point', errorbar='se')
                g.set_axis_labels("Trial Lag", "Mean |Regression Weight|")
                g.fig.suptitle("Neural History Encoding (Absolute Strength)", y=1.02)
                plt.savefig(output_dir / 'history_glm_neural_summary.png')
                plt.close()
        except:
            pass

def predict_reaction_time_multimodal(paths: DataPaths):
    """
    Predicts subsequent Reaction Time (RT) using Spikes, LFP, and Dopamine activity 
    measured in the critical 500ms immediately preceding the last lick before departure.
    
    Decodes which modality provides the most predictive generalized linear variance by tracking
    temporal sequences, oscillatory states, and neuromodulator ramping.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict
    from scipy.stats import pearsonr, linregress, ttest_ind, f_oneway
    from scipy.signal import welch, spectrogram
    from sklearn.metrics import r2_score
    import math
    import seaborn as sns

    print("Analyzing Advanced Multimodal RT Prediction (500ms Pre-Movement Window)...")
    
    # --- 1. Load Event Data ---
    if not hasattr(paths, 'event_corner') or not paths.event_corner or not paths.event_corner.exists():
        print("  Error: Corner event file not found.")
        return
        
    try:
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path)
        dlc_loader = DLCDataLoader(base_path)
        
        df_dlc = dlc_loader.load(paths.dlc_h5)
        corner_df_raw = event_loader.load(paths.event_corner, sync_to_dlc=True, dlc_data=df_dlc)
             
        corner_onsets_df = event_loader.detect_onsets(corner_df_raw)
        corner_ids_full = event_loader.infer_port_id(corner_onsets_df).values
        corner_times_full = event_loader.get_event_times(corner_onsets_df, paths.strobe_seconds)
        
        valid_mask = corner_ids_full != 0
        if len(corner_times_full) == len(corner_ids_full):
            corner_times = corner_times_full[valid_mask]
        else:
            print("  Warning: corner times and IDs length mismatch.")
            return

        lick_path = getattr(paths, 'kilosort_dir', paths.base_path / "kilosort4/sorter_output") / 'licking_seconds.npy'
        if lick_path.exists():
            lick_times = np.load(lick_path, mmap_mode='r').flatten()
        else:
            lick_times = event_loader.get_event_times_by_type('licking', paths)
        movement_onsets = dlc_loader.get_movement_onsets(df_dlc, strobe_path=paths.strobe_seconds)
        
        # Extract distinct Reward (Water) onsets
        if 'Water' in corner_df_raw.columns:
            reward_onsets = event_loader.detect_onsets(corner_df_raw, target_column="Water")
            reward_times = event_loader.get_event_times(reward_onsets, paths.strobe_seconds)
        else:
            reward_times = np.array([])
             
    except Exception as e:
        print(f"  Error loading event data: {e}")
        return

    # --- 2. Define Trials and Reaction Times ---
    trials = []
    
    PRE_MOVE_WINDOW = 0.5  # Focus on the 500ms right before departure
    
    if len(lick_times) > 0 and len(movement_onsets) > 0:
        for i in range(len(corner_times) - 1):
            current_port_time = corner_times[i]
            next_port_time = corner_times[i + 1]
            
            # Forward check for water pulse
            is_rewarded = np.any((reward_times >= current_port_time) & (reward_times <= current_port_time + 3.0)) if len(reward_times) > 0 else False
            
            licks_at_port = lick_times[(lick_times >= current_port_time) & (lick_times < next_port_time)]
            if len(licks_at_port) < 1: continue
            
            first_lick = licks_at_port[0]
            last_lick = licks_at_port[-1]
            
            lick_count = len(licks_at_port)
            lick_duration = last_lick - first_lick
            if lick_duration <= 0:
                lick_freq = np.nan
            else:
                lick_freq = lick_count / lick_duration
            
            # Require the window to be after arrival
            if last_lick - PRE_MOVE_WINDOW < current_port_time:
                continue
                
            # Filter outliers (require at least 500ms licking duration to align with our Pre-Movement window)
            if lick_duration < 0.5 or lick_duration > 10 or lick_freq > 20 or lick_count >= 100 or lick_count < 5 or np.isnan(lick_freq):
                continue
            
            next_onsets = movement_onsets[(movement_onsets > last_lick) & (movement_onsets < next_port_time)]
            if len(next_onsets) == 0: continue
            
            move_onset = next_onsets[0]
            rt = move_onset - last_lick
            
            if 0.1 < rt < 5.0:
                trials.append({
                    'window_start': last_lick - PRE_MOVE_WINDOW,
                    'window_end': last_lick,
                    'window_zero': last_lick,
                    'window_post': last_lick + 0.2, # Needed for Mechanisms plot
                    'rt': rt,
                    'is_rewarded': int(is_rewarded),
                    'lick_duration': lick_duration,
                    'lick_count': lick_count,
                    'lick_freq': lick_freq
                })
    
    if len(trials) < 10:
        print("  Not enough robust valid trials (Requires full 500ms pre-movement recorded RTs). Aborting.")
        return
        
    print(f"  Identified {len(trials)} trials for multimodal prediction.")
    
    # Store targets
    rts = np.array([t['rt'] for t in trials])
    y_target = np.log(rts) # Log transform due to classic long-tail RT distribution
    
    # --- 3. Load Multimodal Data and Extract Dynamic Features ---
    
    # A) Spikes: Track sequential rank order over 5 x 100ms bins
    spike_features = None
    try:
        spike_loader = SpikeDataLoader(getattr(paths, 'neural_base_path', base_path))
        spike_data = spike_loader.load(paths.kilosort_dir)
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        
        N_BINS = 10
        BIN_SIZE = PRE_MOVE_WINDOW / N_BINS
        
        # We will extract 3 manifold features per trial: Mean Rate, Trajectory Speed, State Shift Distance
        spike_features = np.zeros((len(trials), 3))
        print(f"  Extracting Spike Manifold Trajectory features ({len(unique_clusters)} clusters x {N_BINS} timebins)...")
        for i, trial in enumerate(trials):
            pop_vectors = np.zeros((N_BINS, len(unique_clusters)))
            for b in range(N_BINS):
                b_start = trial['window_start'] + b * BIN_SIZE
                b_end = b_start + BIN_SIZE
                mask = (spike_times_sec >= b_start) & (spike_times_sec < b_end)
                trial_clusters = spike_clusters[mask]
                counts = np.bincount(trial_clusters, minlength=np.max(unique_clusters)+1)
                pop_vectors[b, :] = counts[unique_clusters] / BIN_SIZE
                
            mean_rate = np.mean(pop_vectors)
            
            # Trajectory speed (mean step size)
            step_sizes = np.linalg.norm(np.diff(pop_vectors, axis=0), axis=1)
            traj_speed = np.mean(step_sizes)
            
            # State shift (distance from start to end of window)
            state_shift = np.linalg.norm(pop_vectors[-1, :] - pop_vectors[0, :])
            
            spike_features[i, 0] = mean_rate
            spike_features[i, 1] = traj_speed
            spike_features[i, 2] = state_shift
    except Exception as e:
        print(f"  Warning: Spikes not processed or missing. {e}")
        
    # B) LFP: Beta (15-30Hz) and Gamma (30-80Hz) phase-locked power
    lfp_features = None
    try:
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is not None:
             recording = lfp_loader.extractor
             lfp_fs = lfp_loader.fs
             
             # Subsample channels to speed up processing
             locations = recording.get_channel_locations()
             unique_x = np.unique(locations[:, 0])
             target_chans = []
             channel_ids = recording.get_channel_ids()
             for x in unique_x:
                 ch_idx = np.where(locations[:, 0] == x)[0]
                 sort_y = np.argsort(locations[ch_idx, 1])
                 if len(sort_y) > 0: target_chans.append(channel_ids[ch_idx[sort_y[0]]])
                 if len(sort_y) > 1: target_chans.append(channel_ids[ch_idx[sort_y[-1]]])
             
             target_chans = list(set(target_chans))
             print(f"  Extracting LFP Beta/Gamma oscillatory states from {len(target_chans)} channels...")
             
             lfp_features = np.zeros((len(trials), len(target_chans) * 2))
             for i, trial in enumerate(trials):
                 try:
                     # Calculate frames manually based on LFP sync params or just LFP fs
                     if hasattr(lfp_loader, 'sync_params') and lfp_loader.sync_params:
                         m = lfp_loader.sync_params['m']
                         c = lfp_loader.sync_params['c']
                         ratio = lfp_loader.sync_params.get('ratio', 30.0)
                         start_frame = int(((trial['window_start'] - c) / m) / ratio)
                         end_frame = int(((trial['window_end'] - c) / m) / ratio)
                     else:
                         start_frame = int(trial['window_start'] * lfp_fs)
                         end_frame = int(trial['window_end'] * lfp_fs)
                         
                     start_frame = max(0, start_frame)
                     end_frame = min(recording.get_num_samples(), end_frame)
                     if end_frame > start_frame:
                         traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=target_chans)
                         
                         for ch_i in range(len(target_chans)):
                             f, Pxx = welch(traces[:, ch_i], fs=lfp_fs, nperseg=int(lfp_fs * 0.25))
                             
                             beta_mask = (f >= 15) & (f <= 30)
                             gamma_mask = (f > 30) & (f <= 80)
                             
                             beta_power = np.mean(Pxx[beta_mask]) if np.any(beta_mask) else 0
                             gamma_power = np.mean(Pxx[gamma_mask]) if np.any(gamma_mask) else 0
                             
                             lfp_features[i, ch_i*2] = beta_power
                             lfp_features[i, ch_i*2 + 1] = gamma_power
                 except Exception as trial_e: 
                     pass # Silently drop the trial's LFP if bounds totally fail
    except Exception as e:
        print(f"  Warning: LFP not processed or missing. {e}")
        
    # C) Dopamine: Mean Level and Pre-movement Ramping (Slope)
    da_features = None
    try:
        photometry_loader = PhotometryDataLoader(base_path)
        da_result = photometry_loader.load(paths.tdt_dff, paths.tdt_raw)
        if da_result:
            da_signal = da_result['dff_values']
            da_times = da_result['dff_timestamps']
            print("  Extracting Dopamine ramping dynamics...")
            da_features = np.zeros((len(trials), 2)) # [Mean, Slope]
            for i, trial in enumerate(trials):
                mask = (da_times >= trial['window_start']) & (da_times < trial['window_end'])
                if np.sum(mask) > 5:  # Need points to fit slope
                    seg_times = da_times[mask]
                    seg_sig = da_signal[mask]
                    
                    da_features[i, 0] = np.mean(seg_sig)
                    slope, _, _, _, _ = linregress(seg_times, seg_sig)
                    da_features[i, 1] = slope
    except Exception as e:
        print(f"  Warning: Dopamine not processed or missing. {e}")


    # D) Licking Behavior
    lick_features = np.zeros((len(trials), 2))
    for i, trial in enumerate(trials):
        lick_features[i, 0] = trial['lick_duration']
        lick_features[i, 1] = trial['lick_freq']

    # --- 4. Predictive Modeling ---
    results = {}
    models = {
        'Spikes (Manifold)': spike_features,
        'LFP (Beta/Gamma)': lfp_features,
        'Dopamine (Ramp)': da_features,
        'Licking (Behavior)': lick_features
    }
    
    cv_folds = min(5, len(trials) // 3)
    best_pred = None
    best_r = -1
    best_modality = None
    
    for mod_name, X in models.items():
        if X is not None and np.any(np.std(X, axis=0) > 0):
            # Drop zero variance columns
            valid_cols = np.std(X, axis=0) > 0
            X_clean = X[:, valid_cols]
            
            pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20)))
            y_pred = cross_val_predict(pipeline, X_clean, y_target, cv=cv_folds)
            
            r2 = r2_score(y_target, y_pred)
            r, p = pearsonr(y_target, y_pred)
            
            results[mod_name] = {'R2': r2, 'Pearson_R': r, 'P_Value': p}
            
            if r > best_r:
                best_r = r
                best_pred = y_pred
                best_modality = mod_name
    
    # --- 5. Visualization & Export ---
    output_dir = getattr(paths, 'neural_base', getattr(paths, 'base_path', Path('.'))) / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    df_results = pd.DataFrame.from_dict(results, orient='index')
    if df_results.empty:
        print("  Error: Could not model any modality.")
        return
        
    df_results.to_csv(output_dir / 'predict_rt_results.csv')
    print(f"\n  Prediction Results:\n{df_results}")
    
    # Bar plot of correlations
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_results.index, df_results['Pearson_R'], color=['royalblue', 'indianred', 'mediumseagreen'][:len(df_results)])
    plt.axhline(0, color='black', linewidth=1)
    plt.ylabel('Out-of-Fold Pearson Correlation')
    plt.title('Predicting Subsequent RT from 500ms Pre-Movement Neural State')
    
    # Annotate p-values
    for bar, p in zip(bars, df_results['P_Value']):
        yval = bar.get_height()
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        offset = 0.02 if yval > 0 else -0.05
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, sig, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(output_dir / 'rt_prediction_comparison.png', dpi=300)
    plt.close()
    
    # Scatter plot of best modality
    if best_pred is not None:
        plt.figure(figsize=(6, 6))
        
        # Un-log to show absolute time relation visually (though modeled in log space)
        plt.scatter(np.exp(y_target), np.exp(best_pred), alpha=0.7, color='darkorange', edgecolor='k')
        
        # Identity line
        min_val = min(np.exp(y_target).min(), np.exp(best_pred).min())
        max_val = max(np.exp(y_target).max(), np.exp(best_pred).max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        # Line of best fit
        m, b = np.polyfit(np.exp(y_target), np.exp(best_pred), 1)
        plt.plot(np.exp(y_target), m*np.exp(y_target) + b, color='red', alpha=0.8, linewidth=2, label='Line of Best Fit')
        
        plt.xlabel('True Reaction Time (s)')
        plt.ylabel(f'Predicted Reaction Time [{best_modality}] (s)')
        plt.title(f'Best RT Predictor: {best_modality}\nPearson r = {best_r:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'rt_prediction_scatter.png', dpi=300)
        plt.close()
        
    print(f"  Saved comparison plots to {output_dir}")

    # -------------------------------------------------------------------------
    # --- 6. Plot 1: Behavioral Correlations ---
    # -------------------------------------------------------------------------
    print("  Generating Behavioral Correlations Plot...")
    df_trials = pd.DataFrame(trials)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot 1: Reward
    sns.boxplot(data=df_trials, x='is_rewarded', y='rt', ax=axes[0], palette='Set2')
    sns.stripplot(data=df_trials, x='is_rewarded', y='rt', ax=axes[0], color='black', alpha=0.3)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Unrewarded', 'Rewarded'])
    axes[0].set_xlabel("Port Status")
    axes[0].set_ylabel("Reaction Time (s)")
    if len(df_trials['is_rewarded'].unique()) > 1:
        rt_rew = df_trials[df_trials['is_rewarded'] == 1]['rt']
        rt_unrew = df_trials[df_trials['is_rewarded'] == 0]['rt']
        t_stat, p_val = ttest_ind(rt_rew, rt_unrew, equal_var=False)
        axes[0].set_title(f"RT vs Reward\np={p_val:.3f}")
    else:
        axes[0].set_title("RT vs Reward\n(Not enough variance)")
        
    # Plot 2: Lick Duration
    sns.regplot(data=df_trials, x='lick_duration', y='rt', ax=axes[1], scatter_kws={'alpha':0.5})
    r, p = pearsonr(df_trials['lick_duration'], df_trials['rt'])
    axes[1].set_xlabel("Licking Duration (s)")
    axes[1].set_ylabel("Reaction Time (s)")
    axes[1].set_title(f"RT vs Lick Duration\nr={r:.3f}, p={p:.3f}")
    
    # Plot 3: Lick Count
    sns.regplot(data=df_trials, x='lick_count', y='rt', ax=axes[2], scatter_kws={'alpha':0.5})
    r, p = pearsonr(df_trials['lick_count'], df_trials['rt'])
    axes[2].set_xlabel("Total Lick Count")
    axes[2].set_ylabel("Reaction Time (s)")
    axes[2].set_title(f"RT vs Lick Count\nr={r:.3f}, p={p:.3f}")
    
    # Plot 4: Lick Frequency
    sns.regplot(data=df_trials, x='lick_freq', y='rt', ax=axes[3], scatter_kws={'alpha':0.5})
    r, p = pearsonr(df_trials['lick_freq'], df_trials['rt'])
    axes[3].set_xlabel("Licking Frequency (Hz)")
    axes[3].set_ylabel("Reaction Time (s)")
    axes[3].set_title(f"RT vs Lick Frequency\nr={r:.3f}, p={p:.3f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'behavioral_rt_correlations.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # --- 7. Plot 2: Mechanisms Grouped by RT ---
    # -------------------------------------------------------------------------
    print("  Generating Mechanism Grouping Plot...")
    try:
        df_trials['RT_Group'] = pd.cut(df_trials['rt'], bins=[0, 0.4, 0.9, 6.0], labels=['Short RT', 'Middle RT', 'Long RT'])
        # Drop any that might somehow be outside (none should be due to 0.1-5.0 filter)
        df_trials = df_trials.dropna(subset=['RT_Group'])
        fig, axes = plt.subplots(3, 3, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        colors = {'Short RT': 'red', 'Middle RT': 'green', 'Long RT': 'blue'}
        
        POST_MOVE_WINDOW = 0.2
        lfp_chan = target_chans[int(len(target_chans)/2)] if 'target_chans' in locals() and len(target_chans) > 0 else None
        
        for col_idx, condition in enumerate(['Short RT', 'Middle RT', 'Long RT']):
            group_trials = df_trials[df_trials['RT_Group'] == condition].to_dict('records')
            N = len(group_trials)
            
            # --- Spikes Raster ---
            ax_spike = axes[0, col_idx]
            if 'spike_times_sec' in locals() and 'spike_clusters' in locals():
                all_spikes_x, all_spikes_y = [], []
                y_offset = 0
                for t in group_trials:
                    mask = (spike_times_sec >= t['window_start']) & (spike_times_sec <= t['window_post'])
                    t_spikes = spike_times_sec[mask] - t['window_zero']
                    t_clusters = spike_clusters[mask]
                    valid_u = np.isin(t_clusters, unique_clusters[:15])
                    all_spikes_x.extend(t_spikes[valid_u])
                    all_spikes_y.extend(t_clusters[valid_u] + y_offset)
                    y_offset += 20
                ax_spike.scatter(all_spikes_x, all_spikes_y, s=1, color='black', alpha=0.5)
            
            ax_spike.axvline(0, color='red', linestyle='--', linewidth=2, label='Last Lick')
            mean_rt = np.mean([t['rt'] for t in group_trials])
            ax_spike.axvline(mean_rt, color='blue', linestyle='-.', linewidth=2, label='Mean Move Onset')
            ax_spike.set_title(f"{condition} (n={N})\nMean RT = {mean_rt:.2f}s", fontsize=14, fontweight='bold', color=colors[condition])
            ax_spike.set_xlim([-PRE_MOVE_WINDOW, POST_MOVE_WINDOW])
            ax_spike.set_yticks([])
            if col_idx == 0: ax_spike.set_ylabel("Trials (Neurons 1-15)", fontsize=12)
            if col_idx == 2: ax_spike.legend(loc='upper right')
            
            # --- LFP Spectrogram ---
            ax_lfp = axes[1, col_idx]
            if 'recording' in locals() and lfp_chan is not None:
                all_traces = []
                m = lfp_loader.sync_params.get('m', 1/lfp_fs) if hasattr(lfp_loader, 'sync_params') and getattr(lfp_loader, 'sync_params', None) else 1/lfp_fs
                c = lfp_loader.sync_params.get('c', 0) if hasattr(lfp_loader, 'sync_params') and getattr(lfp_loader, 'sync_params', None) else 0
                ratio = lfp_loader.sync_params.get('ratio', 30.0) if hasattr(lfp_loader, 'sync_params') and getattr(lfp_loader, 'sync_params', None) else 1.0
                for t in group_trials:
                    s_frame = max(0, int(((t['window_start'] - c) / m) / ratio))
                    e_frame = min(recording.get_num_samples(), int(((t['window_post'] - c) / m) / ratio))
                    if e_frame > s_frame:
                        trace = recording.get_traces(start_frame=s_frame, end_frame=e_frame, channel_ids=[lfp_chan])[:, 0]
                        target_len = int((PRE_MOVE_WINDOW + POST_MOVE_WINDOW) * lfp_fs)
                        if len(trace) > target_len: trace = trace[:target_len]
                        elif len(trace) < target_len: trace = np.pad(trace, (0, target_len - len(trace)))
                        all_traces.append(trace)
                if all_traces:
                    mean_trace = np.mean(all_traces, axis=0)
                    f, t_spec, Sxx = spectrogram(mean_trace, fs=lfp_fs, nperseg=int(lfp_fs*0.1), noverlap=int(lfp_fs*0.09))
                    t_spec_adj = t_spec - PRE_MOVE_WINDOW
                    f_mask = f <= 100
                    Sxx_db = 10 * np.log10(Sxx[f_mask, :] + 1e-10)
                    ax_lfp.pcolormesh(t_spec_adj, f[f_mask], Sxx_db, shading='gouraud', cmap='viridis')
            ax_lfp.axvline(0, color='red', linestyle='--', linewidth=2)
            ax_lfp.axhspan(15, 30, color='white', alpha=0.2, linestyle=':')
            ax_lfp.set_ylim([0, 100])
            ax_lfp.set_xlim([-PRE_MOVE_WINDOW, POST_MOVE_WINDOW])
            if col_idx == 0: ax_lfp.set_ylabel("Frequency (Hz)", fontsize=12)
            ax_lfp.set_title("Average LFP Spectrogram", fontsize=12)
            
            # --- DA Trace ---
            ax_da = axes[2, col_idx]
            if 'da_signal' in locals() and 'da_times' in locals():
                fs_da = 1 / np.median(np.diff(da_times)) if len(da_times) > 1 else 300.0
                all_da_traces = []
                for t in group_trials:
                    mask = (da_times >= t['window_start']) & (da_times <= t['window_post'])
                    seg = da_signal[mask]
                    target_len = int((PRE_MOVE_WINDOW + POST_MOVE_WINDOW) * fs_da)
                    if len(seg) > target_len: seg = seg[:target_len]
                    elif len(seg) < target_len: seg = np.pad(seg, (0, target_len - len(seg)), constant_values=np.nan)
                    all_da_traces.append(seg)
                if all_da_traces:
                    da_mat = np.array(all_da_traces)
                    da_mean = np.nanmean(da_mat, axis=0)
                    da_err = np.nanstd(da_mat, axis=0) / np.sqrt(N)
                    t_axis = np.linspace(-PRE_MOVE_WINDOW, POST_MOVE_WINDOW, target_len)
                    ax_da.plot(t_axis, da_mean, color=colors[condition], linewidth=2)
                    ax_da.fill_between(t_axis, da_mean - da_err, da_mean + da_err, color=colors[condition], alpha=0.2)
            ax_da.axvline(0, color='red', linestyle='--', linewidth=2)
            ax_da.set_xlim([-PRE_MOVE_WINDOW, POST_MOVE_WINDOW])
            ax_da.set_xlabel("Time from Last Lick (s)", fontsize=12)
            if col_idx == 0: ax_da.set_ylabel("Dopamine dF/F", fontsize=12)
            ax_da.set_title("Average DA Trace", fontsize=12)
            
        fig.suptitle("Multimodal Mechanisms of Reaction Time (T=0 Last Lick)", fontsize=22, fontweight='bold', y=0.98)
        plt.savefig(output_dir / 'mechanisms_rt_groups.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved detailed visual plots to {output_dir}")
    except Exception as e:
        print(f"  Error generating mechanisms plot: {e}")

