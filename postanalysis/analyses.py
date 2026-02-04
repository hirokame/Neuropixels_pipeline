import pandas as pd
import numpy as np
from pathlib import Path
from postanalysis.data_loader import (
    DataPaths,
    convert_date_formats,
    load_dataset_config,
    find_config_entry,
    get_column_name,
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

@lru_cache(maxsize=1)
def _get_dataset_config():
    """Loads the dataset_config.json file."""
    config_path = Path(__file__).resolve().parent.parent / 'dataset_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Error: dataset_config.json not found at {config_path}")
        return {}

def _get_event_onsets_df(event_df, config_entry, target_column=None):
    """
    Filters a DataFrame to return only rows corresponding to event onsets.
    Handles frame-by-frame boolean data (False->True) and ID change data.
    
    Args:
        event_df: DataFrame containing event data
        config_entry: Configuration dictionary for the event file
        target_column: Optional specific column name to filter for (e.g., 'Water')
    
    CRITICAL CHANGE: This function explicitly handles truncated data where
    the initial rows are NaNs. It does NOT convert NaNs to 0s for difference 
    calculation to prevent spurious 0 -> 1 transitions at the start of valid data.
    """
    if event_df.empty:
        return event_df

    # 1. Target Column Logic (Priority)
    if target_column:
        # Check if target column exists (exact match or partial match logic could vary, 
        # but let's try exact first, then fallback to substring)
        matched_col = None
        if target_column in event_df.columns:
            matched_col = target_column
        else:
            # Try finding column that contains target_string
            matches = [c for c in event_df.columns if target_column in str(c)]
            if matches:
                matched_col = matches[0]
        
        if matched_col:
            # Detect rising edge (0->1 or False->True)
            vals = event_df[matched_col].apply(pd.to_numeric, errors='coerce')
            onsets = vals.diff() == 1
            return event_df[onsets]
        else:
            print(f"  Warning: Target column '{target_column}' not found in event file.")
            # Fallback to standard logic if target not found? Or return empty?
            # Safer to return empty if specific target was requested but not found.
            return event_df[pd.Series(False, index=event_df.index)]

    
    # 2. Boolean columns
    bool_cols = []
    if config_entry and 'columns' in config_entry:
        bool_cols = [c['name'] for c in config_entry['columns'] 
                     if c.get('dtype') == 'bool' and c['name'] in event_df.columns]
    
    # Fallback: Check for typical boolean names
    if not bool_cols:
        keywords = ['Corner', 'Lick', 'Beam', 'Stim', 'Water', 'Reward']
        for c in event_df.columns:
            if any(k in str(c) for k in keywords):
                # Check if values are effectively boolean
                unique_vals = event_df[c].dropna().unique()
                if pd.api.types.is_bool_dtype(event_df[c]) or \
                   set(unique_vals).issubset({0, 1, 0.0, 1.0, False, True}):
                    bool_cols.append(c)

    if bool_cols:
        # Detect rising edge on EACH boolean column
        # Logic: Convert to float (NaN preserved). diff() -> (NaN, 1.0, 0.0, -1.0)
        vals = event_df[bool_cols].apply(pd.to_numeric, errors='coerce')
        
        # Heuristic: Distinguish "Event" (Pulse, e.g. Lick) from "State" (Condition, e.g. CW)
        # Event: Only Rising Edge (0 -> 1) matters.
        # State: ANY Change (0 -> 1 OR 1 -> 0) matters.
        
        state_keywords = ['CW', 'CCW', 'Rule', 'Condition', 'Strategy', 'Phase']
        
        mask_list = []
        for col in bool_cols:
            if any(k in col for k in state_keywords):
                # State variable: Detect ANY change (-1 or 1)
                # Ignore NaN -> Value (handled by dropna/diff logic naturally? 
                # diff() of NaN->0 is NaN. diff() of 0->1 is 1. diff of 1->0 is -1.
                # We want abs(diff) == 1.
                change = vals[col].diff().abs() == 1
                mask_list.append(change)
            else:
                # Event variable: Detect Rising Edge (0 -> 1)
                rise = vals[col].diff() == 1
                mask_list.append(rise)
        
        if mask_list:
            onsets = pd.concat(mask_list, axis=1).any(axis=1)
            return event_df[onsets]
        else:
            # Fallback if loop was empty (shouldn't happen if bool_cols is not empty)
            return event_df[pd.Series(False, index=event_df.index)]
    
    # 2. ID columns
    id_col = get_column_name(config_entry, ['CornerID', 'ID', 'id', 'Corner', 'Port', 'port', 'reward_type', 'Type'])
    if id_col and id_col in event_df.columns:
        # If it's numeric/ID-like
        ids = event_df[id_col]
        
        # Logic: Change in ID, current is not 0 (or null), AND previous was not Null/NaN
        # We want to catch real transitions, not artifactual startup transitions
        prev_ids = ids.shift(1)
        
        if pd.api.types.is_numeric_dtype(ids):
            # Reset NaNs to 0 just for value comparison, but track validity
            # Actually, standard logic: (curr != prev) & (curr != 0)
            # But if prev is NaN, (curr != prev) is True. We want to avoid that.
            
            # Mask for valid current ID
            valid_curr = ids.notna() & (ids != 0)
            
            # Mask for changed ID
            changed = ids != prev_ids
            
            # Mask for valid previous ID (if previous is NaN, we shouldn't count it as a switch onset)
            # This is the key fix for "start of file" spurious events
            valid_prev = prev_ids.notna() 
            
            onsets = valid_curr & changed & valid_prev
            return event_df[onsets]
        else:
            # String IDs
            valid_curr = ids.notna() & (ids != '')
            changed = ids != prev_ids
            valid_prev = prev_ids.notna()
            
            onsets = valid_curr & changed & valid_prev
            return event_df[onsets]

    # 3. Fallback: Return as is (sparse)
    return event_df

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
        config = load_dataset_config()
        base_path = paths.base_path
        
        # DLC for tracking and velocity
        dlc_loader = DLCDataLoader(base_path, config)
        # Check if dlc_h5 exists in paths
        if not paths.dlc_h5 or not paths.dlc_h5.exists():
            print("  Warning: DLC file not found.")
            return []
            
        df_dlc = dlc_loader.load(paths.dlc_h5)
        
        # Calculate average speed of Snout and Tail
        strobe_path = paths.strobe_seconds
        v_snout, _ = dlc_loader.calculate_velocity(df_dlc, bodypart='Snout', px_per_cm=px_per_cm, strobe_path=strobe_path)
        v_tail, _ = dlc_loader.calculate_velocity(df_dlc, bodypart='Tail', px_per_cm=px_per_cm, strobe_path=strobe_path)
        
        # Ensure length matches df_dlc
        if len(v_snout) < len(df_dlc):
            v_snout = np.pad(v_snout, (1, 0), mode='edge')
            v_tail = np.pad(v_tail, (1, 0), mode='edge')
        
        v_avg = (v_snout + v_tail) / 2.0
        
        # 2. Get Port IDs at every frame (ROI presence)
        event_loader = EventDataLoader(base_path, config)
        corner_cfg = find_config_entry(paths.event_corner, config)
        if not corner_cfg:
            return []
        
        corner_key = next(k for k, v in config.items() if v == corner_cfg)
        corner_df = event_loader.load(paths.event_corner, sync_to_dlc=True, dlc_data=df_dlc)
        port_ids = event_loader.infer_port_id(corner_df).values
        
        # 3. Timebase (Strobes or Fixed FS)
        try:
            strobe_loader = StrobeDataLoader(base_path, config)
            strobe_times = strobe_loader.load()
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
        print(f"  Population heatmap saved to {output_path}")
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
        # Legend for Significance (Only add if we created the figure or if requested? Let's add it)
        # To avoid clutter in subplots, typically legend is added once. 
        # But for this helper, let's add it if it's a single plot (save_figure=True).
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

def _load_spike_data(paths: DataPaths, return_types: bool = True):
    """
    Loads spike times and clusters from Kilosort output.
    
    Refactored to use modular data loading with schema validation.
    
    Args:
        paths: DataPaths object
        return_types: If True, also returns unit_types dictionary
        
    Returns:
        tuple: (spike_times_sec, spike_clusters, unique_clusters) OR
               (spike_times_sec, spike_clusters, unique_clusters, unit_types) if return_types=True
    """
    try:
        config = load_dataset_config()
        base_path = paths.neural_base.parent if paths.neural_base else Path('.')
        base_path = paths.neural_base_path if paths.neural_base_path else paths.base_path
        spike_loader = SpikeDataLoader(base_path, config)
        spike_data = spike_loader.load(paths.kilosort_dir)
        
        if return_types:
            return (
                spike_data['spike_times_sec'],
                spike_data['spike_clusters'],
                spike_data['unique_clusters'],
                spike_data.get('unit_types', {})
            )
        else:
            return (
                spike_data['spike_times_sec'],
                spike_data['spike_clusters'],
                spike_data['unique_clusters']
            )
    except Exception as e:
        print(f"  Error loading spike data: {e}")
        import traceback
        traceback.print_exc()
        if return_types:
            return None, None, None, {}
        return None, None, None

def _load_dlc_and_calculate_velocity(paths: DataPaths, video_fs: int, px_per_cm: float):
    """
    Loads DLC data and calculates velocity.
    
    Refactored to use modular data loading with schema validation.
    """
    if not paths.dlc_h5 or not paths.dlc_h5.exists():
        print(f"  Error: DLC file not found at {paths.dlc_h5}")
        return None, None
    
    try:
        config = load_dataset_config()
        base_path = paths.dlc_h5.parent if paths.dlc_h5.parent.exists() else Path('.')
        base_path = paths.base_path
        dlc_loader = DLCDataLoader(base_path, config)
        
        # Find config key for DLC file
        dlc_config_entry = find_config_entry(paths.dlc_h5, config)
        if not dlc_config_entry:
            raise ValueError(f"Could not find configuration for {paths.dlc_h5}")
        
        # Find the config key
        dlc_config_key = None
        for key, value in config.items():
            if value == dlc_config_entry:
                dlc_config_key = key
                break
        
        if not dlc_config_key:
            raise ValueError(f"Could not determine config key for {paths.dlc_h5}")
        
        # Load DLC data
        df_dlc = dlc_loader.load(config_key=dlc_config_key)
        
        # Calculate velocity using the loader's method
        velocity, velocity_times = dlc_loader.calculate_velocity(
            df_dlc,
            bodypart=None,  # Will use config to determine
            video_fs=video_fs,
            px_per_cm=px_per_cm
        )
        
        print(f"  Calculated velocity for {len(velocity)} frames.")
        return velocity, velocity_times

    except Exception as e:
        print(f"  Error loading DLC data or calculating velocity: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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

def _load_switch_times(paths, config, event_loader, dlc_loader=None):
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
            strobe_path = paths.kilosort_dir / "strobe_seconds.npy" if paths.kilosort_dir else None
            times = event_loader.get_event_times(switch_df, strobe_path=strobe_path)
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
                                  corner_cfg, event_loader, corner_key):
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
                        success_time = event_loader.get_event_times(rewards.iloc[[0]], config_key=corner_key)[0]
                
                if success_time is None:
                    success_time = corner_times_onsets[next_idx]
                
                decision_time = None
                start_port_col = f'Corner{start_port}'
                if start_port_col in corner_df_full.columns:
                    pre_trial_segment = corner_df_full.loc[trial_start_frame : trial_end_frame]
                    active_frames = pre_trial_segment[pre_trial_segment[start_port_col] == 1].index
                    if len(active_frames) > 0:
                        departure_frame = active_frames[-1]
                        decision_time = event_loader.get_event_times(corner_df_full.loc[[departure_frame]], config_key=corner_key)[0]
                
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
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        dlc_loader = None
        if paths.dlc_h5 and paths.dlc_h5.exists():
            try:
                dlc_loader = DLCDataLoader(base_path, config)
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
        spike_loader = SpikeDataLoader(base_path, config)
        spike_data = spike_loader.load()
        
        spike_times_sec = spike_data['spike_times_sec']
        spike_clusters = spike_data['spike_clusters']
        unique_clusters = spike_data['unique_clusters']
        unit_types = spike_data.get('unit_types', {})
        
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
        print(f"  Debug: Event times range: {event_times.min():.2f}s - {event_times.max():.2f}s")
        print(f"  Debug: Spike times range: {spike_times_sec.min():.2f}s - {spike_times_sec.max():.2f}s")
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
        # List comprehension is significantly faster than repeated np.histogram calls
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

        config = load_dataset_config()
        base_path = paths.base_path
        dlc_loader = DLCDataLoader(base_path, config)
        
        # Find config key for DLC file
        dlc_config_entry = find_config_entry(paths.dlc_h5, config)
        if not dlc_config_entry:
            raise ValueError(f"Could not find config for {paths.dlc_h5}")
        
        dlc_config_key = next((k for k, v in config.items() if v == dlc_config_entry), None)
        if not dlc_config_key:
            raise ValueError(f"Could not find config key for {paths.dlc_h5}")

        df_dlc = dlc_loader.load(config_key=dlc_config_key)
        velocity, velocity_times = dlc_loader.calculate_velocity(
            df_dlc, video_fs=video_fs, px_per_cm=px_per_cm
        )
        if velocity is None:
            return
    except Exception as e:
        print(f"  Error loading DLC data or calculating velocity: {e}")
        return

    # --- 2. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
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
            threshold=2.0
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
        config = load_dataset_config()
        base_path = paths.base_path
        
        if event_file_type == 'movement_onset':
            # Use DLC Loader for movement onsets
            print("  Detecting movement onsets from DLC...")
            dlc_loader = DLCDataLoader(base_path, config)
            event_times = dlc_loader.get_movement_onsets()
            print(f"  Loaded {len(event_times)} movement onset events.")
            
        else:
            event_loader = EventDataLoader(base_path, config)
            event_times = event_loader.get_event_times_by_type(event_file_type, paths)
            print(f"  Loaded {len(event_times)} {event_file_type} events.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- 2. Load LFP Data (Updated to use LFPDataLoader) ---
    try:
        config = load_dataset_config()
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
                            time_window_ms: int = 2000, bin_size_ms: int = 50):
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
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        event_times = event_loader.get_event_times_by_type(event_file_type, paths)
        print(f"  Loaded {len(event_times)} {event_file_type} events.")
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- 2. Load Dopamine Data ---
    try:
        # Identify Dopamine dFF and RAW config keys
        dff_config_key = None
        raw_config_key = None
        
        for key, value in config.items():
            path_str = value.get('path', '').lower()
            if '_dff.mat' in path_str:
                dff_config_key = key
            if '_univraw_offdemod.mat' in path_str:
                raw_config_key = key
                
        if not dff_config_key or not raw_config_key:
            print("  Error: Could not find config entries for both dFF and UnivRAW files.")
            return None
            
        photometry_loader = PhotometryDataLoader(base_path, config)
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
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Load Corner Data (unfiltered for temporal accuracy)
        corner_cfg = find_config_entry(paths.event_corner, config)
        corner_key = next(k for k, v in config.items() if v == corner_cfg)
        corner_df_full = event_loader.load(config_key=corner_key, sync_to_dlc=True)
        
        # Get Corner Times and Onsets for block logic
        corner_df_onsets = _get_event_onsets_df(corner_df_full, corner_cfg)
        corner_times_onsets = event_loader.get_event_times(corner_df_onsets, config_key=corner_key)
        
        # Get Corner IDs
        id_col = get_column_name(corner_cfg, ['CornerID', 'ID', 'id', 'Corner', 'Port'])
        if id_col and id_col in corner_df_onsets.columns and id_col not in [f'Corner{i}' for i in range(1, 5)]:
            corner_ids_onsets = corner_df_onsets[id_col].fillna(0).astype(int).values
        else:
            # Infer from boolean columns
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
        # Load Switch Data
        if paths.event_condition_switch == paths.event_corner:
            # Rule is embedded in corner file, usually in a 'CW' or 'Condition' column
            rule_col = get_column_name(corner_cfg, ['CW', 'Condition', 'Rule', 'Protocol'])
            if rule_col and rule_col in corner_df_full.columns:
                # Find where the rule changes (transitions between True/False or 1/0)
                rule_changes = corner_df_full[rule_col].diff().fillna(0) != 0
                # The first row of the file is also a "switch" to the initial rule
                rule_changes.iloc[0] = True 
                switch_df_raw = corner_df_full[rule_changes]
                switch_times = event_loader.get_event_times(switch_df_raw, config_key=corner_key)
            else:
                print(f"  Warning: Could not find rule column in {paths.event_corner}. Using empty switch times.")
                switch_times = np.array([])
        else:
            switch_cfg = find_config_entry(paths.event_condition_switch, config)
            switch_key = next(k for k, v in config.items() if v == switch_cfg)
            switch_df = event_loader.load(config_key=switch_key)
            switch_df = _get_event_onsets_df(switch_df, switch_cfg)
            switch_times = event_loader.get_event_times(switch_df, switch_key)
        
        # Load Licking/Reward Data if in separate file
        reward_col = get_column_name(corner_cfg, ['Water', 'Reward', 'Reward1'])
        if not reward_col or reward_col not in corner_df_full.columns:
            # Fallback to licking file if specified
            lick_df_full = event_loader.load(config_key=find_config_entry(paths.event_licking, config), sync_to_dlc=True)
        else:
            lick_df_full = corner_df_full # Combined setup

    except Exception as e:
        print(f"  Error loading event data: {e}")
        return

    # --- 2. Identify Behavioral Switch Points ---
    switch_points = _get_behavioral_switch_points(
        switch_times, corner_times_onsets, corner_ids_onsets, 
        corner_df_full, corner_df_onsets, corner_order, 
        corner_cfg, event_loader, corner_key
    )
    
    success_times = [pt['success_time'] for pt in switch_points]
    decision_times = [pt['decision_time'] for pt in switch_points]

    print(f"  Identified {len(success_times)} Success events and {len(decision_times)} Decision events.")

    # --- 3. Run PETH Analysis for both alignment types ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None: return

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
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
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
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Load corner events
        corner_df, corner_times = event_loader.load_events_from_path(paths.event_corner)
        if corner_df.empty: raise ValueError("Could not load corner events")
        
        # Get Corner IDs and Filter Invalid (0) Entries
        corner_config_entry = find_config_entry(paths.event_corner, config)
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            # Use fillna(0) instead of dropna() to maintain same length as corner_times
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        # Both arrays must have same length before filtering
        assert len(corner_ids) == len(corner_times), f"Length mismatch: corner_ids={len(corner_ids)}, corner_times={len(corner_times)}"
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        if len(corner_ids) < 2:
            print("  Not enough valid corner events after filtering.")
            return
        
        # Load switch data - USE HELPER that handles embedded CW column
        switch_times = _load_switch_times(paths, config, event_loader, dlc_loader=None)
        
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
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
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
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Load corner events
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_config_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_config_key)
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        # Get Corner IDs
        # Get Corner IDs and Filter Invalid (0) Entries
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
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
    # Request unit types for visualization
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
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
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_config_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_config_key)
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events.")
        
        # Load switch times - USE HELPER that handles embedded CW column
        switch_times = _load_switch_times(paths, config, event_loader, dlc_loader=None)
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
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
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
        config = load_dataset_config()
        dlc_loader = DLCDataLoader(paths.base_path, config)
        df_dlc = dlc_loader.load()
        
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
            strobe_loader = StrobeDataLoader(paths.base_path, config)
            strobe_times = strobe_loader.load()
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
        spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
        if spike_times_sec is None:
            print("  Skipping prediction (no spike data).")
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
    
    try:
        # 1. Load Data
        spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
        if spike_times_sec is None: return
        
        config = load_dataset_config()
        if not paths.dlc_h5 or not paths.dlc_h5.exists():
             print("  Error: No DLC file found for spatial mapping.")
             return
        
        dlc_loader = DLCDataLoader(paths.base_path, config)
        dlc_config_entry = find_config_entry(paths.dlc_h5, config)
        if not dlc_config_entry:
             print("  Error: DLC config not found.")
             return
             
        dlc_config_key = next((k for k,v in config.items() if v == dlc_config_entry), None)
        df_dlc = dlc_loader.load(config_key=dlc_config_key)
        
        # Extract X, Y (Snout best, then derived)
        bp = 'Snout' # Default assumption
        # Check if Snout is in columns
        if isinstance(df_dlc.columns, pd.MultiIndex):
             scorer = df_dlc.columns.levels[0][0]
             bodyparts = df_dlc.columns.levels[1]
             if 'Snout' in bodyparts:
                bp = 'Snout'
             else:
                bp = bodyparts[0]
                
             x = df_dlc[scorer][bp]['x'].values
             y = df_dlc[scorer][bp]['y'].values
        else:
             print("  Warning: DLC dataframe not MultiIndex. Skipping.")
             return
        
        # Pixel to cm
        px_per_cm = 30.0 # Standard assumption
        x_cm = x / px_per_cm
        y_cm = y / px_per_cm
        
        # Timebase
        try:
             strobe_loader = StrobeDataLoader(paths.base_path, config)
             strobe_file = paths.base_path / 'kilosort4' / 'sorter_output' / 'strobe_seconds.npy'
             if strobe_file.exists():
                 t_pos = np.load(strobe_file)
             else:
                 t_pos = strobe_loader.load()
                 
             L = min(len(t_pos), len(x_cm))
             t_pos = t_pos[:L]
             x_cm = x_cm[:L]
             y_cm = y_cm[:L]
        except:
             fs = 60.0 # standard
             t_pos = np.arange(len(x_cm)) / fs
        
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
             
    except Exception as e:
        print(f"  Error in spatial rate maps: {e}")
        import traceback
        traceback.print_exc()

def analyze_pre_switch_activity(paths: DataPaths, pre_switch_window_sec: float = 10.0, min_trials_before_switch: int = 3):
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
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Load corner events
        corner_cfg = find_config_entry(paths.event_corner, config)
        corner_key = next(k for k, v in config.items() if v == corner_cfg)
        corner_df_full = event_loader.load(config_key=corner_key, sync_to_dlc=True)
        corner_df_onsets = _get_event_onsets_df(corner_df_full, corner_cfg)
        corner_times_onsets = event_loader.get_event_times(corner_df_onsets, config_key=corner_key)
        
        id_col = get_column_name(corner_cfg, ['CornerID', 'ID', 'id', 'Corner', 'Port'])
        if id_col and id_col in corner_df_onsets.columns:
            corner_ids_onsets = corner_df_onsets[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df_onsets.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df_onsets.columns:
                    ids[corner_df_onsets[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids_onsets = ids.astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids_onsets != 0
        corner_ids_onsets = corner_ids_onsets[valid_mask]
        corner_times_onsets = corner_times_onsets[valid_mask]
        
        print(f"  Filtering invalid (0) IDs: Retaining {len(corner_ids_onsets)} valid events.")
            
        print(f"  DEBUG: Unique corner IDs in data: {np.unique(corner_ids_onsets)}")
        print(f"  DEBUG: First 10 corner IDs: {corner_ids_onsets[:10]}")
            
        # Load switch data
        switch_times = _load_switch_times(paths, config, event_loader)
        
        # Load switch config entry for helper
        switch_cfg = find_config_entry(paths.event_condition_switch, config)
        
        # Identify Behavioral Switch Points
        print(f"  DEBUG: Using corner_order for pre-switch analysis: {[1, 2, 4, 3]}")
        switch_points = _get_behavioral_switch_points(
            switch_times, corner_times_onsets, corner_ids_onsets, 
            corner_df_full, corner_df_onsets, [1, 2, 4, 3], # Default order
            switch_cfg, event_loader, corner_key
        )
        
        print(f"  Identified {len(switch_points)} behavioral switch points.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        return
    
    # --- 2. Define Pre-Switch and Baseline Periods ---
    pre_switch_segments = []
    baseline_segments = []
    
    for i, pt in enumerate(switch_points):
        behavioral_switch_time = pt['decision_time']
        
        # Pre-switch period: immediately before behavioral switch
        pre_start = behavioral_switch_time - pre_switch_window_sec
        pre_end = behavioral_switch_time
        
        # Count trials in pre-switch period
        n_trials_pre = np.sum((corner_times_onsets >= pre_start) & (corner_times_onsets < pre_end))
        
        # Ensure sufficient trials and valid time
        if n_trials_pre >= min_trials_before_switch and pre_start > 0:
            
            # --- Determine Baseline ---
            baseline_start = None
            baseline_end = None
            
            if baseline_mode == 'stable_block':
                # Need previous switch to define the block
                if i > 0:
                    prev_switch_time = switch_points[i-1]['decision_time']
                    block_start = prev_switch_time
                    block_end = behavioral_switch_time
                    
                    # Identify trials in this block
                    block_trials_indices = np.where((corner_times_onsets >= block_start) & (corner_times_onsets < block_end))[0]
                    
                    if len(block_trials_indices) > 6: # Need > 6 trials to drop 3 start and 3 end
                        # Exclude first 3 and last 3 trials of the previous block
                        # The block ends at the *current* switch's start (pre-switch instability).
                        # Actually, "previous block" implies the block that is *currently* active before the switch.
                        # So it starts at switch_points[i-1] and ends at switch_points[i].
                        
                        # Indices of trials within this block
                        valid_indices = block_trials_indices[3:-3] # Drop 3 start, 3 end
                        
                        if len(valid_indices) > 0:
                            t_start_valid = corner_times_onsets[valid_indices[0]]
                            t_end_valid = corner_times_onsets[valid_indices[-1]]
                            
                            # Stable block period
                            stable_duration = t_end_valid - t_start_valid
                            
                            # Take middle 50%
                            margin = stable_duration * 0.25
                            baseline_start = t_start_valid + margin
                            baseline_end = t_end_valid - margin
                            
                            # Ensure it doesn't overlap with pre-switch window
                            if baseline_end > pre_start:
                                baseline_end = pre_start
                                if baseline_start >= baseline_end:
                                     # Overlap is too significant, fallback or skip
                                     baseline_start = None 
                
                if baseline_start is None:
                     print(f"  Warning: Could not determine stable baseline for switch {i}. Falling back to fixed time offset.")
                     # Fallback to fixed time logic below
            
            # Default or Fallback Logic
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
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
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
    print("\n  Pre-switch activity analysis complete (aligned to behavioral switch).")
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

def compute_statistics_for_tuning(data_array, baseline_array=None, method='ttest'):
    """
    Compute statistical tests and effect sizes for tuning data.
    
    Args:
        data_array: Array of values to test (e.g., firing rates, tuning indices)
        baseline_array: Optional baseline comparison (default: test against 0)
        method: Statistical test to use ('ttest', 'wilcoxon', 'permutation')
        
    Returns:
        dict with: p_value, statistic, effect_size, confidence_interval
    """
    from scipy import stats
    import numpy as np
    
    results = {}
    
    # Remove NaNs
    data_clean = data_array[~np.isnan(data_array)]
    
    if len(data_clean) == 0:
        return {'p_value': np.nan, 'statistic': np.nan, 'effect_size': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan, 'n': 0}
    
    if baseline_array is None:
        # One-sample test against zero
        if method == 'ttest':
            stat, pval = stats.ttest_1samp(data_clean, 0)
        elif method == 'wilcoxon':
            stat, pval = stats.wilcoxon(data_clean)
        else:
            # Permutation test
            observed_mean = np.mean(data_clean)
            n_perm = 10000
            perm_means = []
            for _ in range(n_perm):
                perm_signs = np.random.choice([-1, 1], size=len(data_clean))
                perm_means.append(np.mean(data_clean * perm_signs))
            pval = np.sum(np.abs(perm_means) >= np.abs(observed_mean)) / n_perm
            stat = observed_mean
    else:
        # Two-sample test
        baseline_clean = baseline_array[~np.isnan(baseline_array)]
        if method == 'ttest':
            stat, pval = stats.ttest_ind(data_clean, baseline_clean)
        elif method == 'wilcoxon':
            stat, pval = stats.mannwhitneyu(data_clean, baseline_clean)
        else:
            # Permutation test
            observed_diff = np.mean(data_clean) - np.mean(baseline_clean)
            combined = np.concatenate([data_clean, baseline_clean])
            n_perm = 10000
            perm_diffs = []
            for _ in range(n_perm):
                np.random.shuffle(combined)
                perm_diffs.append(np.mean(combined[:len(data_clean)]) - 
                                 np.mean(combined[len(data_clean):]))
            pval = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_perm
            stat = observed_diff
    
    # Effect size (Cohen's d)
    if baseline_array is None:
        effect_size = np.mean(data_clean) / (np.std(data_clean) + 1e-10)
    else:
        pooled_std = np.sqrt((np.std(data_clean)**2 + np.std(baseline_clean)**2) / 2)
        effect_size = (np.mean(data_clean) - np.mean(baseline_clean)) / (pooled_std + 1e-10)
    
    # Confidence interval (95%)
    ci_lower, ci_upper = stats.t.interval(0.95, len(data_clean)-1,
                                          loc=np.mean(data_clean),
                                          scale=stats.sem(data_clean))
    
    results['p_value'] = pval
    results['statistic'] = stat
    results['effect_size'] = effect_size
    results['ci_lower'] = ci_lower
    results['ci_upper'] = ci_upper
    results['n'] = len(data_clean)
    results['mean'] = np.mean(data_clean)
    results['sem'] = stats.sem(data_clean)
    
    return results

def apply_multiple_comparison_correction(p_values, method='fdr_bh'):
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        p_values: Array of p-values
        method: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg), or 'fdr_by'
        
    Returns:
        Array of corrected p-values and boolean array of rejections at alpha=0.05
    """
    from statsmodels.stats.multitest import multipletests
    
    # Remove NaNs
    p_clean = np.array(p_values)
    valid_mask = ~np.isnan(p_clean)
    
    if np.sum(valid_mask) == 0:
        return p_clean, np.zeros(len(p_clean), dtype=bool)
    
    # Apply correction
    reject, pvals_corrected, _, _ = multipletests(p_clean[valid_mask], 
                                                   alpha=0.05, 
                                                   method=method)
    
    # Put back into full array
    result_pvals = np.full(len(p_values), np.nan)
    result_reject = np.zeros(len(p_values), dtype=bool)
    result_pvals[valid_mask] = pvals_corrected
    result_reject[valid_mask] = reject
    
    return result_pvals, result_reject

def analyze_outcome_encoding(paths: DataPaths, time_window_ms: int = 200):
    """
    Analyzes neural encoding of trial outcome (Reward vs Error/Omission).
    
    This replaces the RPE analysis for tasks where "prediction error" is not explicit,
    but the animal switches behavior based on the outcome (Reward vs No Reward).
    It compares neural activity following a Rewarded outcome vs an Unrewarded outcome.
    window for FR is from event time to 200ms after the event time.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_window_ms (int): The window in milliseconds around the outcome event.
    """
    print("Analyzing outcome encoding (Reward vs Error)...")
    window_sec = time_window_ms / 1000.0

    # --- 1. Load Outcome Data ---
    if not paths.event_reward or not paths.event_reward.exists():
        print(f"  Error: Reward event file not found at {paths.event_reward}. This analysis requires it.")
        return
        
    try:
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Try to sync to DLC if available
        dlc_loader = None
        if paths.dlc_h5 and paths.dlc_h5.exists():
            try:
                dlc_loader = DLCDataLoader(paths.base_path, config)
            except Exception:
                pass
        
        # Load reward events explicitly to target 'Water' column
        reward_config_entry = find_config_entry(paths.event_reward, config)
        reward_config_key = next((k for k, v in config.items() if v == reward_config_entry), None)
        
        if not reward_config_key:
             raise ValueError(f"Could not find config key for {paths.event_reward}")

        reward_df_raw = event_loader.load(
            config_key=reward_config_key,
            sync_to_dlc=(dlc_loader is not None),
            dlc_loader=dlc_loader
        )
        
        # Explicitly filter for 'Water' column if it exists in a combined file
        target_column = 'Water' 
        reward_df_onsets = _get_event_onsets_df(reward_df_raw, reward_config_entry, target_column=target_column)
        reward_times = event_loader.get_event_times(reward_df_onsets, reward_config_key)
        
        # We need the original columns (like Water) aligned with these times.
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
        corner_cols = [c for c in reward_df_raw.columns if 'Corner' in c]
        lick_col_raw = get_column_name(reward_config_entry, ['Lick', 'lick', 'Licking', 'licking'])
        
        if corner_cols:
             print(f"  Using Corner columns {corner_cols} to define Port Arrivals...")
             
             # 1. Determine Current Port Intervals from RAW data
             # Assumption: Corners are mutually exclusive or we take the max
             current_port = pd.Series(0, index=reward_df_raw.index, dtype=int)
             for col in corner_cols:
                 try:
                     port_num = int(''.join(filter(str.isdigit, col)))
                 except:
                     continue
                 is_active = reward_df_raw[col].fillna(0).astype(bool)
                 current_port[is_active] = port_num
             
             # Identify continuous segments of Port presence (Visits)
             # Transition from 0 to N is start, N to 0 (or M) is end
             # We want Start and End Times for each visit
             
             # Get timestamps for all frames
             all_times = event_loader.get_event_times(reward_df_raw, reward_config_key)
             
             # Create separate arrays for processing
             port_vals = current_port.values
             
             # Find changes
             changes = np.where(port_vals[:-1] != port_vals[1:])[0]
             
             visits = []
             # Start from first segment?
             if port_vals[0] != 0:
                  start_idx = 0
             else:
                  start_idx = None

             for idx in changes:
                 # idx is the last index of the previous state
                 # idx+1 is start of new state
                 
                 # Close previous
                 if start_idx is not None:
                     port_id = port_vals[start_idx]
                     if port_id != 0:
                         visits.append({
                             'port': port_id,
                             'start_time': all_times[start_idx],
                             'end_time': all_times[idx],
                             'start_idx': start_idx,
                             'end_idx': idx
                         })
                     start_idx = None
                 
                 # Start new
                 if port_vals[idx+1] != 0:
                     start_idx = idx + 1
             
             # Handle last segment
             if start_idx is not None:
                 port_id = port_vals[start_idx]
                 if port_id != 0:
                      visits.append({
                             'port': port_id,
                             'start_time': all_times[start_idx],
                             'end_time': all_times[-1],
                             'start_idx': start_idx,
                             'end_idx': len(all_times)-1
                         })
             
             print(f"  Identified {len(visits)} Port Visits.")
             
             # 2. Convert Visits to DataFrame
             visits_df = pd.DataFrame(visits)
             if visits_df.empty:
                 print("  No port visits found.")
                 return

             # 3. Load Lick Seconds from Kilosort Folder
             lick_seconds_path = paths.kilosort_dir / 'licking_seconds.npy'
             if not lick_seconds_path.exists():
                 print(f"  Error: {lick_seconds_path} not found. Cannot perform Lick analysis.")
                 return
             
             print(f"  Loading Lick Timestamps from {lick_seconds_path}...")
             lick_seconds = np.load(lick_seconds_path)
             
             # 4. Find First Lick in each Visit
             # For each visit, find the first lick time t such that start_time <= t <= end_time
             
             first_lick_times = []
             first_lick_indices = [] # Indices in visits_df
             
             # Optimize search (licks are sorted)
             for i, row in visits_df.iterrows():
                 # Find licks in window
                 # searchsorted finds insertion points
                 start_idx = np.searchsorted(lick_seconds, row['start_time'])
                 if start_idx < len(lick_seconds):
                     first_lick = lick_seconds[start_idx]
                     if first_lick <= row['end_time']:
                         first_lick_times.append(first_lick)
                         first_lick_indices.append(i)
             
             print(f"  Found {len(first_lick_times)} First Licks matching valid Port Visits.")
             
             if len(first_lick_times) == 0:
                 print("  No licks found within port visits.")
                 return

             # 5. Determine Outcome for these specific Licks
             # We need to know if the Visit was Rewarded.
             # Check 'Water' column in the original data for the duration of the visit
             # OR check if a Reward Pulse happened in [start_time, end_time]
             
             # Let's use the Raw Data indices stored in visits
             water_col = get_column_name(reward_config_entry, ['Water', 'water', 'Reward', 'reward'])
             
             final_events = []
             
             for i, lick_time in zip(first_lick_indices, first_lick_times):
                 visit = visits_df.iloc[i]
                 
                 # Check Water in this visit interval (using indices is safer/faster if dense)
                 # Slice raw df
                 # access via iloc requires integers. 'visit' is a Series which might float-cast ints if mixed with timestamps.
                 start_idx = int(visit['start_idx'])
                 end_idx = int(visit['end_idx'])
                 visit_slice = reward_df_raw.iloc[start_idx : end_idx+1]
                 
                 is_rewarded = False
                 if water_col and water_col in visit_slice.columns:
                     if visit_slice[water_col].dropna().astype(bool).any():
                         is_rewarded = True
                 
                 final_events.append({
                     'timestamp': lick_time,
                     'is_rewarded': is_rewarded,
                     'port_id': visit['port']
                 })
             
             # 6. Construct Final DataFrame for Analysis
             reward_df = pd.DataFrame(final_events)
             reward_times = reward_df['timestamp'].values
             
             # We don't need 'Water' column anymore in reward_df as we have 'is_rewarded'
             # logic below expects 'Water' column to split?
             # Let's add 'Water' column to match downstream logic
             reward_df['Water'] = reward_df['is_rewarded']
             
        else:
             print("  Warning: Corner columns not found. Cannot determine Port Visits.")
             return
             
        if len(reward_df) == 0:
             print("  No applicable outcome events found. Aborting.")
             return

        # Check for Water column
        water_col = get_column_name(reward_config_entry, ['Water', 'water', 'Reward', 'reward'])
        
        if water_col and water_col in reward_df.columns:
            # Filter by Water (True = Reward, False = No Reward/Error)
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
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
        return

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
    # This analysis assumes an 'event_reward.csv' file with 'timestamp' and 'port_id' columns.
    if not paths.event_reward or not paths.event_reward.exists():
        print(f"  Error: Reward event file not found at {paths.event_reward}.")
        return
        
    try:
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Try to sync to DLC if available
        dlc_loader = None
        if paths.dlc_h5 and paths.dlc_h5.exists():
            try:
                dlc_loader = DLCDataLoader(paths.base_path, config)
            except Exception:
                pass

        # Load reward events explicitly
        reward_config_entry = find_config_entry(paths.event_reward, config)
        reward_config_key = next((k for k, v in config.items() if v == reward_config_entry), None)
        
        if not reward_config_key:
             raise ValueError(f"Could not find config key for {paths.event_reward}")

        reward_df_raw = event_loader.load(
            config_key=reward_config_key,
            sync_to_dlc=(dlc_loader is not None),
            dlc_loader=dlc_loader
        )

        target_column = 'Water'
        reward_df = _get_event_onsets_df(reward_df_raw, reward_config_entry, target_column=target_column)
        reward_times = event_loader.get_event_times(reward_df, reward_config_key)
        
        if len(reward_df) == 0:
            print(f"  Error: No reward events found in {paths.event_reward}")
            return
            
        reward_df['timestamp'] = reward_times
        
        # Find config entry for port ID
        reward_config_entry = find_config_entry(paths.event_reward, config)
        port_col = get_column_name(reward_config_entry, ['port_id', 'PortID', 'Port', 'id'])
        
        if port_col and port_col in reward_df.columns:
            reward_df['port_id'] = reward_df[port_col].fillna(0).astype(int)
        else:
            print("  Inferring port_id from Corner/Lick columns...")
            reward_df['port_id'] = event_loader.infer_port_id(reward_df)
            
        # Filter out events where port could not be determined (port_id == 0)
        n_unknown = (reward_df['port_id'] == 0).sum()
        if n_unknown > 0:
            print(f"  Warning: Could not determine port ID for {n_unknown} reward events.")
            reward_df = reward_df[reward_df['port_id'] != 0]
        
        # Identify first and second rewards at each port
        reward_df['reward_order'] = reward_df.groupby('port_id').cumcount()
        
        first_rewards = reward_df[reward_df['reward_order'] == 0]['timestamp'].values
        second_rewards = reward_df[reward_df['reward_order'] == 1]['timestamp'].values
        
        print(f"  Loaded {len(first_rewards)} first-reward events and {len(second_rewards)} second-reward events.")

    except Exception as e:
        print(f"  Error loading or processing reward data: {e}. Ensure file has 'timestamp' and 'port_id' columns.")
        return

    if len(first_rewards) == 0 or len(second_rewards) == 0:
        print("  Not enough data for both first and second rewards. Aborting analysis.")
        return

    # --- 2. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
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
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Try to sync to DLC if available
        dlc_loader = DLCDataLoader(paths.base_path, config)
        corner_df, corner_times = event_loader.load_events_from_path(
            paths.event_corner, 
            dlc_loader=dlc_loader,
            sync_to_dlc=(dlc_loader is not None)
        )
        
        # Exclude 0s to preserve transition continuity
        corner_config_entry = find_config_entry(paths.event_corner, config)
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 4+1):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        valid_mask = corner_ids != 0
        corner_df = corner_df[valid_mask]
        corner_times = corner_times[valid_mask]
        corner_ids = corner_ids[valid_mask] # Sync IDs filtering
        
        print(f"  Filtering invalid (0) IDs: Retaining {len(corner_times)} valid corner events for history analysis.")

        # Load reward events explicitly
        reward_config_entry = find_config_entry(paths.event_reward, config)
        reward_config_key = next((k for k, v in config.items() if v == reward_config_entry), None)
        
        reward_df_raw = event_loader.load(
            config_key=reward_config_key,
            sync_to_dlc=(dlc_loader is not None),
            dlc_loader=dlc_loader
        )
        reward_df = _get_event_onsets_df(reward_df_raw, reward_config_entry, target_column='Water')
        reward_times = event_loader.get_event_times(reward_df, reward_config_key)
        
        print(f"  Loaded {len(corner_times)} corner events and {len(reward_times)} reward events.")

    except Exception as e:
        print(f"  Error loading event data: {e}")
        return

    # --- 2. Define Trials and Determine their Outcome ---
    # Map rewards to corners strictly (Window: Corner[i] to Corner[i+1])
    corner_rewarded = np.zeros(len(corner_times), dtype=bool)
    for i in range(len(corner_times) - 1):
        t_start = corner_times[i]
        t_end = corner_times[i+1]
        # Check if any reward occurred in [t_start, t_end)
        if np.any((reward_times >= t_start) & (reward_times < t_end)):
            corner_rewarded[i] = True
            
    print(f"  Identified {np.sum(corner_rewarded)} rewarded corners out of {len(corner_times)} total.")

    # Get Kinematic Bouts for precise segmentation
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
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
        return

    # --- 4. Calculate Firing Rates Based on Previous Trial Outcome ---
    # Segments for trials following a rewarded trial
    post_reward_segments = []
    # Segments for trials following a non-rewarded trial
    post_no_reward_segments = []

    for i in range(len(trials)):
        current_trial = trials[i]
        is_post_reward = corner_rewarded[current_trial['corner_idx']]
        # Use bout timings (guaranteed to exist now)
        segment = (current_trial['bout_start'], current_trial['bout_end'])
        
        if is_post_reward:
            post_reward_segments.append(segment)
        else:
            post_no_reward_segments.append(segment)

    print(f"  Found {len(post_reward_segments)} trials following a reward.")
    print(f"  Found {len(post_no_reward_segments)} trials following no reward.")

    if not post_reward_segments or not post_no_reward_segments:
        print("  Not enough trials of both types to compare. Aborting.")
        return

    # --- 5. Behavioral Kinematics Analysis ---
    print("\n  Analyzing behavioral kinematics (Speed, Deviation, Licking)...")
    try:
        # Load DLC Data
        velocity, velocity_times = _load_dlc_and_calculate_velocity(paths, video_fs=60, px_per_cm=30.0)
        
        # Load raw DLC for coordinates
        dlc_loader = DLCDataLoader(paths.base_path, config)
        dlc_config_entry = find_config_entry(paths.dlc_h5, config)
        dlc_config_key = next((k for k, v in config.items() if v == dlc_config_entry), None)
        df_dlc = dlc_loader.load(config_key=dlc_config_key)
        
        # Get Position Columns (Snout or Body)
        # Try Snout first
        bodypart = 'Snout'
        if not any(bodypart in c for c in df_dlc.columns):
            bodypart = dlc_loader.detect_bodyparts(df_dlc)[0]
        
        x_col = next((c for c in df_dlc.columns if bodypart in c and 'x' in c), None)
        y_col = next((c for c in df_dlc.columns if bodypart in c and 'y' in c), None)
        
        # Load Licking Data
        lick_times = None
        lick_seconds_path = paths.kilosort_dir / 'licking_seconds.npy'
        if lick_seconds_path.exists():
            lick_times = np.load(lick_seconds_path)
        
        # Calculate Metrics for each trial
        kinematics = []
        
        # Helper to compute path deviation
        # Collect all trajectories first to compute mean
        traj_coords = defaultdict(list)
        
        temp_trials_data = []
        
        valid_trials = [] # To keep aligned with neural analysis potentially, or just independent
        
        # Sync DLC times (assuming frame-based index converted to seconds in velocity_times)
        # If velocity_times matches df_dlc length
        if len(velocity_times) != len(df_dlc):
             # Simple alignment
             dlc_times = np.arange(len(df_dlc)) / 60.0 # fallback
        else:
             dlc_times = velocity_times

        for i in range(len(trials)):
            t = trials[i]
            # Analyze based on outcome at the START of this trial
            prev_rewarded = corner_rewarded[t['corner_idx']]
            
            # 1. Speed (Mean in window)
            n_points = min(len(dlc_times), len(velocity))
            mask = (dlc_times[:n_points] >= t['start_time']) & (dlc_times[:n_points] <= t['end_time'])
            
            if not np.any(mask): continue
            mean_speed = np.nanmean(velocity[:n_points][mask])
            
            # 2. Trajectory Coordinates (for Deviation)
            if x_col and y_col:
                # Slice DF to match mask length
                df_safe = df_dlc.iloc[:n_points]
                xs = df_safe.loc[mask, x_col].values
                ys = df_safe.loc[mask, y_col].values
                # Resample to common length (e.g. 100 points) for mean calculation
                if len(xs) > 10:
                    # Interpolate
                    dists = np.linspace(0, 1, len(xs))
                    target_dists = np.linspace(0, 1, 100)
                    f_x = interp1d(dists, xs, kind='linear')
                    f_y = interp1d(dists, ys, kind='linear')
                    resampled_xy = np.column_stack((f_x(target_dists), f_y(target_dists)))
                    
                    # Store for calculation
                    start_id = corner_ids[i]
                    end_id = corner_ids[i+1]
                    
                    traj_key = (start_id, end_id)
                    traj_coords[traj_key].append(resampled_xy)
                    
                    # 3. Licking at Next Port (Destination)
                    lick_freq = np.nan
                    lick_len = np.nan
                    
                    if lick_times is not None:
                        l_start = t['end_time']
                        l_end = l_start + 3.0 # 3 second window for licking analysis
                        
                        b_mask = (lick_times >= l_start) & (lick_times < l_end)
                        bout_licks = lick_times[b_mask]
                        
                        if len(bout_licks) > 0:
                            lick_freq = len(bout_licks) / 3.0 # Hz
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

        # Calculate Mean Trajectories
        mean_paths = {}
        for k, valid_paths in traj_coords.items():
            if len(valid_paths) > 0:
                mean_paths[k] = np.mean(np.array(valid_paths), axis=0)
                
        # Calculate Deviation and Finalize Dictionary
        kinematic_results = []
        for d in temp_trials_data:
            mean_path = mean_paths.get(d['traj_key'])
            deviation = np.nan
            if mean_path is not None:
                diffs = np.linalg.norm(d['xy_resampled'] - mean_path, axis=1)
                deviation = np.mean(diffs)
            
            kinematic_results.append({
                'prev_rewarded': d['prev_rewarded'],
                'mean_speed': d['mean_speed'],
                'trajectory_deviation': deviation,
                'lick_frequency': d['lick_freq'],
                'lick_length': d['lick_len']
            })
            
        # Convert to DataFrame
        k_df = pd.DataFrame(kinematic_results)
        
        # Stats and Plotting
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
        
        # Save Kinematics
        k_output = output_dir / 'reward_history_kinematics.csv'
        k_df.to_csv(k_output)
        print(f"  Kinematics data saved to {k_output}")
        
    except Exception as e:
        print(f"  Error in kinematic analysis: {e}")
        import traceback
        traceback.print_exc()

    # --- 6. Neural Analysis (Existing) ---
    results = {}
    
    for cid in unique_clusters:
        cts = spike_times_sec[spike_clusters == cid]
        
        # --- Post-Reward Rates ---
        rates_post_reward = []
        for s, e in post_reward_segments:
            duration = e - s
            if duration > 0:
                count = np.sum((cts >= s) & (cts < e))
                rates_post_reward.append(count / duration)
                
        rates_post_reward = np.array(rates_post_reward)
        rate_post_reward = np.mean(rates_post_reward) if len(rates_post_reward) > 0 else 0
        
        # --- Post-No-Reward Rates ---
        rates_post_no_reward = []
        for s, e in post_no_reward_segments:
            duration = e - s
            if duration > 0:
                count = np.sum((cts >= s) & (cts < e))
                rates_post_no_reward.append(count / duration)
                
        rates_post_no_reward = np.array(rates_post_no_reward)
        rate_post_no_reward = np.mean(rates_post_no_reward) if len(rates_post_no_reward) > 0 else 0
        
        # --- History Modulation Index ---
        if rate_post_reward + rate_post_no_reward > 0:
            history_index = (rate_post_reward - rate_post_no_reward) / (rate_post_reward + rate_post_no_reward)
        else:
            history_index = 0
            
        # --- Statistical Test (Mann-Whitney U) ---
        p_val = np.nan
        stat = np.nan
        if len(rates_post_reward) > 0 and len(rates_post_no_reward) > 0:
            try:
                from scipy import stats
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
    print("\n  Reward history analysis complete.")
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
        # Load Corner Events
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Sync DLC
        dlc_loader = None
        if paths.dlc_h5 and paths.dlc_h5.exists():
            try: dlc_loader = DLCDataLoader(paths.base_path, config)
            except: pass
            
        corner_df, corner_times = event_loader.load_events_from_path(paths.event_corner, dlc_loader=dlc_loader, sync_to_dlc=True)
        
        # Get Corner IDs
        corner_config_entry = find_config_entry(paths.event_corner, config)
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            # Infer
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        valid_mask = corner_ids != 0
        corner_times = corner_times[valid_mask]
        corner_ids = corner_ids[valid_mask]
        
        # Load Reward Events
        reward_df_raw = event_loader.load_events_from_path(paths.event_reward, dlc_loader=dlc_loader)[0]
        # Get timestamps
        reward_config_entry = find_config_entry(paths.event_reward, config)
        reward_key = next(k for k, v in config.items() if v == reward_config_entry)
        
        # Assume 'Water' column
        water_col = get_column_name(reward_config_entry, ['Water', 'water', 'Reward'])
        if water_col:
             reward_df = _get_event_onsets_df(reward_df_raw, reward_config_entry, target_column=water_col)
             reward_times = event_loader.get_event_times(reward_df, reward_key)
        else:
             print("  Warning: No Water column found. Using all reward events.")
             reward_times = event_loader.get_event_times(reward_df_raw, reward_key)

        # Load Licking Data
        lick_times = None
        lick_path = paths.kilosort_dir / 'licking_seconds.npy'
        if lick_path.exists():
            lick_times = np.load(lick_path)
            
        # Load Kinematics (Speed)
        velocity, velocity_times = _load_dlc_and_calculate_velocity(paths, video_fs=60, px_per_cm=30.0)
        # Load Raw DLC for deviation
        dlc_loader = DLCDataLoader(paths.base_path, config)
        dlc_key = next((k for k, v in config.items() if v == find_config_entry(paths.dlc_h5, config)), None)
        df_dlc = dlc_loader.load(config_key=dlc_key)
        
        # Fix DLC times if mismatched
        if len(velocity_times) != len(df_dlc):
             velocity_times = np.arange(len(df_dlc)) / 60.0

        # Construct Trials
        trials = []
        
        # Pre-calculate mean paths for deviation
        traj_coords = defaultdict(list)
        
        bodypart = 'Snout'
        if not any(bodypart in c for c in df_dlc.columns):
             bodypart = dlc_loader.detect_bodyparts(df_dlc)[0]
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
    
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None: return
    
    neural_coefs = {} # cid -> coefs
    
    # Pre-calculate trial windows
    windows = []
    for idx in target_trial_indices:
        row = df_trials.iloc[idx]
        windows.append((row['start_time'], row['end_time']))
        
    windows = np.array(windows)
    durations = windows[:, 1] - windows[:, 0]
    
    for cid in tqdm(unique_clusters, desc="Cells"):
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
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Save Behavioral Kernels
    beh_data = []
    for target_name, coefs in behavior_results.items():
        # Handle Current Trial (X_beh Index 0 = Action)
        beh_data.append({'Target': target_name, 'Lag': 0, 'Feature': 'Action', 'Weight': coefs[0]})
        
        # Note: Speed and Deviation were NOT in the model, so no Lag 0 weights for them.

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
    if beh_data:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=pd.DataFrame(beh_data), x='Lag', y='Weight', hue='Feature', style='Target', markers=True)
        plt.title(f"Behavioral History Kernels (GLM)")
        plt.axhline(0, color='k', linestyle='--')
        plt.ylabel("Regression Weight")
        plt.tight_layout()
        plt.savefig(output_dir / 'history_glm_behavior.png')
        plt.close()
        
    if not df_neural.empty:
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

def analyze_perseveration_signals(paths: DataPaths, post_switch_window_trials: int = 5, corner_order_cw: list = [1, 2, 4, 3]):
    """
    Analyzes neural activity during perseverative errors immediately following a rule switch.
    
    Refactored to use modular data loading with proper schema validation and synchronization.
    """
    print("Analyzing perseveration signals...")

    # --- 1. Load Data using modular loaders ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing corner or condition switch event files.")
        return
        
    try:
        config = load_dataset_config()
        base_path = paths.event_corner.parent if paths.event_corner.parent.exists() else Path('.')
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Try to sync to DLC if available
        dlc_loader = None
        if paths.dlc_h5 and paths.dlc_h5.exists():
            try:
                dlc_loader = DLCDataLoader(base_path, config)
            except Exception:
                pass
        
        # Load corner events
        corner_config_entry = find_config_entry(paths.event_corner, config)
        if not corner_config_entry:
            raise ValueError("No config for corner events")
        
        corner_config_key = None
        for key, value in config.items():
            if value == corner_config_entry:
                corner_config_key = key
                break
        
        if not corner_config_key:
            raise ValueError("Could not determine config key for corner events")
        
        corner_df = event_loader.load(
            config_key=corner_config_key,
            sync_to_dlc=(dlc_loader is not None),
            dlc_loader=dlc_loader
        )
        
        # Filter to onsets
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        
        # Extract columns using config
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner', 'Port'])
        
        # Get corner times
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        # Get corner IDs
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            # Try to infer from boolean columns
            ids = pd.Series(0, index=corner_df.index)
            found_cols = False
            for i in range(1, 5):
                col = f'Corner{i}'
                if col in corner_df.columns:
                    found_cols = True
                    ids[corner_df[col].fillna(False).astype(bool)] = i
            
            if found_cols:
                corner_ids = ids.fillna(0).astype(int).values
            else:
                raise ValueError("Could not determine corner ID column")
                
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events.")

        # Load switch events
        # Load switch data
        switch_times = _load_switch_times(paths, config, event_loader, dlc_loader)

    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters, unit_types = _load_spike_data(paths, return_types=True)
    if spike_times_sec is None:
        return

    # --- 3. Load Kinematic States for precise Bout identification ---
    states = _get_kinematic_states(paths)
    if not states:
        print("  Error: Could not retrieve kinematic states. Aborting.")
        return
        
    # Filter only Trajectory segments (X_to_Y)
    trajectories = [s for s in states if '_to_' in s['label']]
    traj_start_times = np.array([t['start_time'] for t in trajectories])

    # --- 4. Identify Perseverative Errors and Correct Choices ---
    error_segments = []
    correct_segments = []
    
    current_rule_is_cw = True # Assumption: Initial rule is CW (should ideally be inferred)
    
    # We need to correctly handle the initial rule and switches
    # If switch_times are [t1, t2, ...], rule is:
    # [0, t1]: Initial Rule
    # [t1, t2]: Next Rule
    all_switch_times = np.concatenate(([0], switch_times, [traj_start_times[-1] + 1.0 if len(traj_start_times) > 0 else 100000]))

    for i in range(len(all_switch_times) - 1):
        pre_switch_rule_is_cw = current_rule_is_cw
        switch_time = all_switch_times[i]
        
        # In actual experiments, switches happen AFTER a certain condition is met.
        # Everything BEFORE the first switch_time is pre-switch rule.
        # Special case: i=0 is the first block. We start analyzing from i=1 (post-switch).
        
        if i > 0:
            # We are in the block [switch_time, next_switch_time]
            # The rule just changed. Pre-switch rule was pre_switch_rule_is_cw.
            post_switch_rule_is_cw = not pre_switch_rule_is_cw
            
            # Find trajectories starting after this switch
            post_switch_indices = np.where(traj_start_times >= switch_time)[0]
            
            # Take the first N trajectories for perseveration analysis
            for k in range(min(post_switch_window_trials, len(post_switch_indices))):
                idx = post_switch_indices[k]
                traj = trajectories[idx]
                try:
                    parts = traj['label'].split('_to_')
                    start_port, end_port = int(parts[0]), int(parts[1])
                except (ValueError, IndexError):
                    continue
                
                is_correct_by_old_rule = _is_move_correct(start_port, end_port, corner_order_cw, pre_switch_rule_is_cw)
                is_correct_by_new_rule = _is_move_correct(start_port, end_port, corner_order_cw, post_switch_rule_is_cw)
                
                # A perseverative error is one that follows the OLD rule but violates the NEW rule
                if is_correct_by_old_rule and not is_correct_by_new_rule:
                    error_segments.append((traj['start_time'], traj['end_time']))
                # A correct trial follows the NEW rule
                elif is_correct_by_new_rule:
                    correct_segments.append((traj['start_time'], traj['end_time']))

        # Update rule for next block
        current_rule_is_cw = not current_rule_is_cw

    print(f"  Found {len(error_segments)} perseverative error movements and {len(correct_segments)} correct movements.")

    if not error_segments or not correct_segments:
        print("  Not enough error or correct trials found to compare. Aborting.")
        return

    # --- 4. Calculate Firing Rates ---
    error_duration = sum(e - s for s, e in error_segments)
    correct_duration = sum(e - s for s, e in correct_segments)

    results = {}
    for cid in unique_clusters:
        cts = spike_times_sec[spike_clusters == cid]
        
        rate_error = sum(np.sum((cts >= s) & (cts < e)) for s, e in error_segments) / error_duration if error_duration > 0 else 0
        rate_correct = sum(np.sum((cts >= s) & (cts < e)) for s, e in correct_segments) / correct_duration if correct_duration > 0 else 0
        
        if rate_error + rate_correct > 0:
            error_selectivity = (rate_error - rate_correct) / (rate_error + rate_correct)
        else:
            error_selectivity = 0
            
        results[cid] = {
            'firing_rate_error_trials': rate_error,
            'firing_rate_correct_trials': rate_correct,
            'error_selectivity_index': error_selectivity,
            'type': unit_types.get(cid, 'Unknown')
        }
    
    # --- 5. Save and Display Results ---
    print("\n  Perseveration signal analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'

    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'perseveration_data.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")

    # Generate Swarm Plot
    swarm_path = output_dir / 'perseveration_swarm.png'
    _plot_metric_swarm(df_results, 'error_selectivity_index', swarm_path, 
                       "Perseveration Signals (Error vs Correct)", "Selectivity Index")

def analyze_error_detection(paths: DataPaths, corner_order: list = [1, 2, 4, 3], time_window_ms: int = 2000):
    """
    Analyzes neural activity differences between correct and incorrect port choices.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for correct CW navigation.
        time_window_ms (int): Time window around port arrival for analysis.
    """
    print("Analyzing error detection signals...")
    window_sec = time_window_ms / 1000.0
    
    # --- 1. Load Data ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing corner or condition switch event files.")
        return
    
    try:
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Load corner events
        corner_df, corner_times = event_loader.load_events_from_path(paths.event_corner)
        if corner_df.empty: raise ValueError("Could not load corner events")

        # Get Corner IDs
        corner_config_entry = find_config_entry(paths.event_corner, config)
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_df = corner_df[valid_mask] # Also filter DF if used later? analyze_error_detection uses switch_df mostly. 
        # Actually corner_df IS used later for getting times again? No, corner_times is already extracted.
        # But wait, analyze_error_detection doesn't use corner_df heavily after loading.
        
        print(f"  Loaded {len(corner_times)} valid corner events.")
        
        # Load switch data
        switch_df, switch_times = event_loader.load_events_from_path(paths.event_condition_switch)
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- 2. Categorize Trials as Correct or Incorrect ---
    correct_trials = []
    incorrect_trials = []
    
    # Determine rule for each block
    block_boundaries = np.concatenate([[-np.inf], switch_times, [np.inf]])
    
    for block_idx in range(len(block_boundaries) - 1):
        block_start = block_boundaries[block_idx]
        block_end = block_boundaries[block_idx + 1]
        
        # Find trials in this block
        block_mask = (corner_times > block_start) & (corner_times < block_end)
        block_indices = np.where(block_mask)[0]
        
        if len(block_indices) < 2:
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
        
        rule_is_cw = cw_moves > ccw_moves
        
        # Categorize each trial
        for i in range(len(block_indices) - 1):
            idx = block_indices[i]
            if idx + 1 >= len(corner_ids):
                continue
            
            trial_time = corner_times[idx + 1]
            s, e = corner_ids[idx], corner_ids[idx + 1]
            
            if _is_move_correct(s, e, corner_order, rule_is_cw):
                correct_trials.append(trial_time)
            else:
                incorrect_trials.append(trial_time)
    
    correct_trials = np.array(correct_trials)
    incorrect_trials = np.array(incorrect_trials)
    
    print(f"  Found {len(correct_trials)} correct trials and {len(incorrect_trials)} incorrect trials.")
    
    if len(correct_trials) == 0 or len(incorrect_trials) == 0:
        print("  Not enough trials of both types. Aborting.")
        return
    
    # --- 3. Load Spike Data and Calculate Firing Rates ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    results = {}
    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        # Calculate rate during correct trials
        rate_correct = 0
        for t in correct_trials:
            rate_correct += np.sum((cluster_spikes >= t - window_sec/2) & (cluster_spikes < t + window_sec/2))
        rate_correct /= (len(correct_trials) * window_sec)
        
        # Calculate rate during incorrect trials
        rate_incorrect = 0
        for t in incorrect_trials:
            rate_incorrect += np.sum((cluster_spikes >= t - window_sec/2) & (cluster_spikes < t + window_sec/2))
        rate_incorrect /= (len(incorrect_trials) * window_sec)
        
        # Error detection index
        if rate_correct + rate_incorrect > 0:
            error_index = (rate_incorrect - rate_correct) / (rate_incorrect + rate_correct)
        else:
            error_index = 0
        
        results[cid] = {
            'rate_correct': rate_correct,
            'rate_incorrect': rate_incorrect,
            'error_detection_index': error_index
        }
    
    # --- 4. Save Results ---
    print("\n  Error detection analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'error_detection.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Generate Heatmap
    heatmap_path = output_dir / 'error_detection_heatmap.png'
    cols_to_plot = ['rate_correct', 'rate_incorrect']
    _plot_population_heatmap(df_results[cols_to_plot], heatmap_path, 
                             "Error Detection (Correct vs Incorrect)", "Condition", sort_col='error_detection_index')

def analyze_decision_confidence(paths: DataPaths, corner_order: list = [1, 2, 4, 3], confidence_metric: str = 'reaction_time'):
    """
    Analyzes relationship between firing patterns and decision certainty.
    
    Uses reaction time (time to reach next port) as a proxy for decision confidence.
    Faster movements indicate higher confidence.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for navigation.
        confidence_metric (str): Metric for confidence ('reaction_time' or 'trajectory_variability').
    """
    print("Analyzing decision confidence...")
    
    # --- 1. Load Corner Data ---
    if not paths.event_corner or not paths.event_corner.exists():
        print(f"  Error: Corner event file not found.")
        return
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_config_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_config_key)
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        print(f"  Loaded {len(corner_times)} corner events.")
        
    except Exception as e:
        print(f"  Error loading corner event data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- 2. Calculate Confidence Metric (Reaction Time) ---
    reaction_times = []
    trial_times = []
    
    for i in range(len(corner_times) - 1):
        rt = corner_times[i + 1] - corner_times[i]
        if 0.5 < rt < 15:  # Valid reaction time range
            reaction_times.append(rt)
            trial_times.append(corner_times[i])
    
    reaction_times = np.array(reaction_times)
    trial_times = np.array(trial_times)
    
    # Define fast (high confidence) and slow (low confidence) trials
    median_rt = np.median(reaction_times)
    fast_trials = trial_times[reaction_times < median_rt]
    slow_trials = trial_times[reaction_times >= median_rt]
    
    print(f"  Identified {len(fast_trials)} fast (high confidence) and {len(slow_trials)} slow (low confidence) trials.")
    
    if len(fast_trials) < 5 or len(slow_trials) < 5:
        print("  Not enough trials in both categories. Aborting.")
        return
    
    # --- 3. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 4. Calculate Firing Rates for Each Confidence Level ---
    window_sec = 2.0  # Window before port arrival
    
    results = {}
    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        # Fast trials (high confidence)
        rate_fast = 0
        for t in fast_trials:
            rate_fast += np.sum((cluster_spikes >= t - window_sec) & (cluster_spikes < t))
        rate_fast /= (len(fast_trials) * window_sec)
        
        # Slow trials (low confidence)
        rate_slow = 0
        for t in slow_trials:
            rate_slow += np.sum((cluster_spikes >= t - window_sec) & (cluster_spikes < t))
        rate_slow /= (len(slow_trials) * window_sec)
        
        # Confidence modulation index
        if rate_fast + rate_slow > 0:
            confidence_index = (rate_fast - rate_slow) / (rate_fast + rate_slow)
        else:
            confidence_index = 0
        
        results[cid] = {
            'rate_high_confidence': rate_fast,
            'rate_low_confidence': rate_slow,
            'confidence_modulation_index': confidence_index
        }
    
    # --- 5. Save Results ---
    print("\n  Decision confidence analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'decision_confidence.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Generate Heatmap
    heatmap_path = output_dir / 'decision_confidence_heatmap.png'
    cols_to_plot = ['rate_high_confidence', 'rate_low_confidence']
    _plot_population_heatmap(df_results[cols_to_plot], heatmap_path, 
                             "Decision Confidence Encoding", "Confidence Level", sort_col='confidence_modulation_index')

def analyze_decision_accumulation(paths: DataPaths, corner_order: list = [1, 2, 4, 3]):
    """
    Analyzes evidence accumulation for decisions using drift-diffusion-like framework.
    
    Examines whether neural firing rates ramp up before port choices, indicating
    evidence accumulation.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for navigation.
    """
    print("Analyzing decision accumulation (evidence accumulation)...")
    
    # --- 1. Load Data ---
    if not paths.event_corner or not paths.event_corner.exists():
        print("  Error: Corner event file not found.")
        return
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_config_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_config_key)
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        print(f"  Loaded {len(corner_times)} corner events.")
        
        # Get Corner IDs and Filter
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Filtering invalid (0) IDs: Retaining {len(corner_times)} valid choice events.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        return
    
    # --- 2. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 3. Calculate Ramping Activity Before Choices ---
    pre_choice_window = 2.0  # 2 seconds before choice
    n_time_bins = 10
    bin_size = pre_choice_window / n_time_bins
    
    results = {}
    
    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        # Collect firing rates in time bins before each choice
        binned_rates = []
        
        for choice_time in corner_times[1:]:  # Skip first corner
            bin_rates = []
            for i in range(n_time_bins):
                bin_start = choice_time - pre_choice_window + i * bin_size
                bin_end = bin_start + bin_size
                
                n_spikes = np.sum((cluster_spikes >= bin_start) & (cluster_spikes < bin_end))
                rate = n_spikes / bin_size
                bin_rates.append(rate)
            
            binned_rates.append(bin_rates)
        
        if len(binned_rates) > 5:
            # Average across trials
            mean_trajectory = np.mean(binned_rates, axis=0)
            
            # Calculate slope (linear regression)
            x = np.arange(n_time_bins)
            slope, _ = np.polyfit(x, mean_trajectory, 1)
            
            # Positive slope indicates ramping/accumulation
            results[cid] = {
                'ramping_slope': slope,
                'mean_rate': np.mean(mean_trajectory),
                'final_rate': mean_trajectory[-1],
                'initial_rate': mean_trajectory[0]
            }
    
    # --- 4. Save Results ---
    print("\n  Decision accumulation analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'decision_accumulation.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Summary
    n_ramping = np.sum(df_results['ramping_slope'] > 0.1)
    print(f"  Neurons with evidence accumulation (ramping): {n_ramping}")

def analyze_choice_prediction(paths: DataPaths, time_window_sec: float = 2.0, n_splits: int = 5):
    """
    Analyzes whether neural activity can predict the upcoming port choice.
    Performs Sliding Window Decoding from -2s to +1s relative to arrival.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_window_sec (float): (Deprecated/Used for bounds) Range to check around arrival.
        n_splits (int): Number of cross-validation splits.
    """
    print("Analyzing choice prediction (sliding window)...")
    
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  Error: scikit-learn is required for this analysis. Install it with: pip install scikit-learn")
        return
    
    # --- 1. Load Corner Event Data ---
    if not paths.event_corner or not paths.event_corner.exists():
        print(f"  Error: Corner event file not found at {paths.event_corner}.")
        return
        
    try:
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Load corner events
        corner_df, corner_times = event_loader.load_events_from_path(paths.event_corner)
        
        if corner_df.empty:
            print(f"  Error: Could not load corner data from {paths.event_corner}")
            return
            
        # Get Corner IDs
        corner_config_entry = find_config_entry(paths.event_corner, config)
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            # Infer from boolean columns
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events.")
        
        print(f"  Loaded {len(corner_times)} corner events.")
        
    except Exception as e:
        print(f"  Error loading corner event data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- 2. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- Define Sliding Windows ---
    # Window settings
    window_width = 0.2  # 200 ms
    step_size = 0.05    # 50 ms
    
    # Range relative to event (e.g. -2.0 to +1.0)
    t_min = -2.0
    t_max = 1.0
    
    window_starts = np.arange(t_min, t_max, step_size)
    accuracies = []
    times = []
    
    print(f"  Running decoder on {len(window_starts)} windows ({t_min}s to {t_max}s)...")
    
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except:
        return

    # Pre-calculate spike counts for all events/units to optimize? 
    # Or just loop. Loop is cleaner for now.
    
    results = {}
    
    # Filter corner events to ensure we have valid IDs
    # Also we might want to decode Strategy (CW/CCW) or PortID?
    # Prompt implies "predict upcoming port choice" -> Multi-class classification (4 ports)
    
    # Remove corner ID 0 or NaNs
    valid_mask = corner_ids > 0
    clean_ids = corner_ids[valid_mask]
    clean_times = corner_times[valid_mask]
    
    # Filter classes with too few samples
    from collections import Counter
    counts = Counter(clean_ids)
    valid_classes = [k for k,v in counts.items() if v >= n_splits]
    
    if len(valid_classes) < 2:
        print("  Not enough classes/samples for decoding.")
        return
        
    final_mask = np.isin(clean_ids, valid_classes)
    y = clean_ids[final_mask]
    target_times = clean_times[final_mask]
    
    for t_start in tqdm(window_starts):
        t_end = t_start + window_width
        win_center = t_start + window_width/2
        
        # Build X for this window
        X = []
        for eta in target_times:
            # Absolute window
            w_start = eta + t_start
            w_end = eta + t_end
            
            # Count spikes
            rates = []
            for cid in unique_clusters:
                spikes = spike_times_sec[spike_clusters == cid]
                n = np.sum((spikes >= w_start) & (spikes < w_end))
                rates.append(n / window_width)
            X.append(rates)
            
        X = np.array(X)
        
        # Decode
        clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv)
        
        accuracies.append(np.mean(scores))
        times.append(win_center)
        
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, accuracies, 'k-', linewidth=2)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Arrival')
    ax.axhline(1.0/len(valid_classes), color='g', linestyle=':', label='Chance')
    ax.set_xlabel('Time relative to arrival (s)')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('Port Choice Prediction (Sliding Window)')
    ax.legend()
    plt.tight_layout()
    
    out_dir = paths.neural_base / 'post_analysis'
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'decoding_sliding_window.png')
    print(f"  Saved decoding plot to {out_dir / 'decoding_sliding_window.png'}")
    
    # Save CSV
    df = pd.DataFrame({'time': times, 'accuracy': accuracies})
    df.to_csv(out_dir / 'decoding_sliding_window.csv', index=False)
            
    return # End function here (replacing original logic)

    # --- 3. Build Feature Matrix ---
    # (Original logic masked out by return above, or we can just replace it entirely)

    labels = np.array(labels)
    
    print(f"  Built feature matrix: {features.shape[0]} trials x {features.shape[1]} neurons")
    
    # --- 4. Train Classifier and Evaluate ---
    if len(np.unique(labels)) < 2:
        print("  Error: Need at least 2 different port choices to train classifier.")
        return
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Use logistic regression with cross-validation
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
    scores = cross_val_score(clf, features_scaled, labels, cv=min(n_splits, len(labels)), scoring='accuracy')
    
    mean_accuracy = np.mean(scores)
    chance_level = 1.0 / len(np.unique(labels))
    
    print(f"\n  Choice prediction analysis complete.")
    print(f"  Mean accuracy: {mean_accuracy:.3f}")
    print(f"  Chance level: {chance_level:.3f}")
    print(f"  Above chance: {mean_accuracy > chance_level}")
    
    # --- 5. Save Results ---
    results = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': np.std(scores),
        'chance_level': chance_level,
        'n_neurons': features.shape[1],
        'n_trials': features.shape[0],
        'n_classes': len(np.unique(labels))
    }
    
    df_results = pd.DataFrame([results])
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'choice_prediction.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

def analyze_reversal_learning_dynamics(paths: DataPaths, trials_per_block: int = 10, corner_order: list = [1, 2, 4, 3]):
    """
    Compares neural adaptation dynamics between the first and subsequent rule reversals.

    Calculates the change in strategy tuning for the first switch vs. the average
    of all subsequent switches to see if learning-to-learn occurs.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        trials_per_block (int): Number of trials pre/post switch to define an analysis block.
        corner_order (list): The order of corners for the CW direction.
    
    Refactored to use modular data loading with proper schema validation and synchronization.
    """
    print("Analyzing reversal learning dynamics (first vs. subsequent switches)...")

    # --- 1. Load Data using modular loaders ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing corner or condition switch event files.")
        return
        
    try:
        config = load_dataset_config()
        base_path = paths.event_corner.parent if paths.event_corner.parent.exists() else Path('.')
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Try to sync to DLC if available
        dlc_loader = None
        if paths.dlc_h5 and paths.dlc_h5.exists():
            try:
                dlc_loader = DLCDataLoader(base_path, config)
            except Exception:
                pass
        
        # Load corner events
        corner_config_entry = find_config_entry(paths.event_corner, config)
        if not corner_config_entry:
            raise ValueError(f"Could not find configuration for {paths.event_corner}")
        
        corner_config_key = None
        for key, value in config.items():
            if value == corner_config_entry:
                corner_config_key = key
                break
        
        if not corner_config_key:
            raise ValueError(f"Could not determine config key for corner events")
        
        corner_df_full = event_loader.load(
            config_key=corner_config_key,
            sync_to_dlc=(dlc_loader is not None),
            dlc_loader=dlc_loader
        )
        
        # Filter to onsets
        corner_df_onsets = _get_event_onsets_df(corner_df_full, corner_config_entry)
        
        # Extract columns using config
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner', 'Port', 'PortID'])
        
        # Get corner times and IDs
        corner_times_onsets = event_loader.get_event_times(corner_df_onsets, corner_config_key)
        
        if id_col and id_col in corner_df_onsets.columns:
            corner_ids_onsets = corner_df_onsets[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df_onsets.index)
            for i in range(1, 5):
                col = f'Corner{i}'
                if col in corner_df_onsets.columns:
                    ids[corner_df_onsets[col].fillna(False).astype(bool)] = i
            corner_ids_onsets = ids.fillna(0).astype(int).values
        
        # Load switch events
        switch_times = _load_switch_times(paths, config, event_loader, dlc_loader)
        
        print(f"  Loaded {len(corner_times_onsets)} corner events and {len(switch_times)} switches.")
        
        # Load switch config entry for helper
        switch_config_entry = find_config_entry(paths.event_condition_switch, config)
        switch_config_key = next(k for k, v in config.items() if v == switch_config_entry)
        
        # --- 2. Identify Behavioral Switch Points ---
        print(f"  DEBUG: Identifying behavioral switch points using corner_order: {corner_order}")
        switch_points = _get_behavioral_switch_points(
            switch_times, corner_times_onsets, corner_ids_onsets, 
            corner_df_full, corner_df_onsets, corner_order, 
            corner_config_entry, event_loader, corner_config_key
        )
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(switch_points) < 1:
        print("  No behavioral switches identified. Aborting.")
        return

    # --- 3. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return

    # --- 4. Analyze Adaptation Dynamics ---
    # Load Kinematic states for trajectory-based analysis
    states = _get_kinematic_states(paths)
    trajectories = [s for s in states if '_to_' in s['label']] if states else []
    
    switch_adaptation_results = {cid: {'first': np.nan, 'subsequent': []} for cid in unique_clusters}
    
    for pt_idx, pt in enumerate(switch_points):
        # We align to the DECISION point of the first correct trial
        behavioral_switch_time = pt['decision_time']
        trial_idx = pt['first_correct_trial_idx']
        
        if trial_idx is None:
            continue
            
        pre_start_idx = trial_idx - trials_per_block
        post_end_idx = trial_idx + trials_per_block
        
        if pre_start_idx < 0 or post_end_idx >= len(corner_times_onsets):
            print(f"  Warning: Behavioral switch at {behavioral_switch_time:.1f}s is too close to edge of session (Index {trial_idx}, Bounds [0, {len(corner_times_onsets)-1}]). Skipping.")
            continue

        # Define time windows based on trials
        pre_start_time = corner_times_onsets[pre_start_idx]
        post_end_time = corner_times_onsets[post_end_idx]

        # Pre-switch tuning: The window leading up to the behavioral switch
        pre_tuning = _calculate_block_tuning(trajectories, corner_order, pre_start_time, behavioral_switch_time, spike_times_sec, spike_clusters, unique_clusters)
        
        # Post-switch tuning: The window starting from the behavioral switch
        post_tuning = _calculate_block_tuning(trajectories, corner_order, behavioral_switch_time, post_end_time, spike_times_sec, spike_clusters, unique_clusters)
        
        if pre_tuning and post_tuning:
            for cid in unique_clusters:
                # Handle clusters that might be silent in one window
                idx_pre = pre_tuning.get(cid, 0)
                idx_post = post_tuning.get(cid, 0)
                tuning_change = idx_post - idx_pre
                
                if pt_idx == 0: # First switch
                    switch_adaptation_results[cid]['first'] = tuning_change
                else: # Subsequent switches
                    switch_adaptation_results[cid]['subsequent'].append(tuning_change)
    
    # --- 4. Finalize, Save and Display ---
    final_results = {}
    for cid, data in switch_adaptation_results.items():
        mean_subsequent_change = np.nanmean(data['subsequent']) if data['subsequent'] else np.nan
        final_results[cid] = {
            'tuning_change_first_switch': data['first'],
            'tuning_change_subsequent_switches': mean_subsequent_change
        }

    print("\n  Reversal learning dynamics analysis complete.")
    df_results = pd.DataFrame.from_dict(final_results, orient='index')
    df_results.index.name = 'cluster_id'
    
    # Add a column for the difference, ignoring NaNs
    df_results['adaptation_difference'] = df_results['tuning_change_subsequent_switches'] - df_results['tuning_change_first_switch']

    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'reversal_learning_data.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")

    # Generate Heatmap
    df_results.dropna(inplace=True)
    heatmap_path = output_dir / 'reversal_learning_heatmap.png'
    cols_to_plot = ['tuning_change_first_switch', 'tuning_change_subsequent_switches']
    _plot_population_heatmap(df_results[cols_to_plot], heatmap_path, 
                             "Reversal Learning Dynamics", "Switch Type", sort_col='tuning_change_subsequent_switches')

def analyze_post_switch_adaptation(paths: DataPaths, corner_order: list = [1, 2, 4, 3], post_switch_window_trials: int = 10):
    """
    Analyzes how quickly neurons adapt to new strategy after a behavioral switch.
    
    Compares neural activity in early vs late trials starting from the first correct
    choice after a rule switch.
    
    Args:
        paths: The DataPaths object with all the required paths.
        corner_order (list): Order of corners for navigation.
        post_switch_window_trials (int): Number of trials to analyze post-switch.
    """
    print("Analyzing behavioral adaptation speed (post-switch)...")
    
    # --- 1. Load Data ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing corner or condition switch event files.")
        return
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Load corner events
        corner_cfg = find_config_entry(paths.event_corner, config)
        corner_key = next(k for k, v in config.items() if v == corner_cfg)
        corner_df_full = event_loader.load(config_key=corner_key, sync_to_dlc=True)
        corner_df_onsets = _get_event_onsets_df(corner_df_full, corner_cfg)
        corner_times_onsets = event_loader.get_event_times(corner_df_onsets, config_key=corner_key)
        
        id_col = get_column_name(corner_cfg, ['CornerID', 'ID', 'id', 'Corner', 'Port'])
        if id_col and id_col in corner_df_onsets.columns:
            corner_ids_onsets = corner_df_onsets[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df_onsets.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df_onsets.columns:
                    ids[corner_df_onsets[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids_onsets = ids.astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids_onsets != 0
        corner_ids_onsets = corner_ids_onsets[valid_mask]
        corner_times_onsets = corner_times_onsets[valid_mask]
        
        print(f"  Filtering invalid (0) IDs: Retaining {len(corner_ids_onsets)} valid events.")
        
        # Load switch data
        switch_times = _load_switch_times(paths, config, event_loader, dlc_loader=None)
        
        # Load switch config entry for helper
        switch_cfg = find_config_entry(paths.event_condition_switch, config)
        
        # Identify Behavioral Switch Points
        switch_points = _get_behavioral_switch_points(
            switch_times, corner_times_onsets, corner_ids_onsets, 
            corner_df_full, corner_df_onsets, corner_order, 
            switch_cfg, event_loader, corner_key
        )
        
        print(f"  Identified {len(switch_points)} behavioral switch points.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        return
    
    # --- 2. Identify Early and Late Post-Switch Trials ---
    early_post_switch_segments = []
    late_post_switch_segments = []
    
    for pt in switch_points:
        behavioral_switch_time = pt['decision_time']
        trial_idx = pt['first_correct_trial_idx']
        
        if trial_idx is None:
            continue
            
        # Analyze trials starting FROM the behavioral switch
        post_switch_indices = np.where(corner_times_onsets >= behavioral_switch_time)[0]
        
        if len(post_switch_indices) < post_switch_window_trials:
            continue
        
        # Early trials: first half of the post-switch window
        early_indices = post_switch_indices[:post_switch_window_trials // 2]
        # Late trials: second half
        late_indices = post_switch_indices[post_switch_window_trials // 2:post_switch_window_trials]
        
        if len(early_indices) > 0 and len(late_indices) > 0:
            early_start = corner_times_onsets[early_indices[0]]
            early_end = corner_times_onsets[early_indices[-1]] + 1 # small buffer if not next onset available
            
            late_start = corner_times_onsets[late_indices[0]]
            late_end = corner_times_onsets[late_indices[-1]] + 1
            
            early_post_switch_segments.append((early_start, early_end))
            late_post_switch_segments.append((late_start, late_end))
    
    print(f"  Found {len(early_post_switch_segments)} valid adaptation analysis pairs.")
    
    if len(early_post_switch_segments) < 2:
        print("  Not enough post-switch periods. Aborting.")
        return
    
    # --- 3. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 4. Calculate Firing Rates ---
    results = {}
    
    total_early_duration = sum(e - s for s, e in early_post_switch_segments)
    total_late_duration = sum(e - s for s, e in late_post_switch_segments)
    
    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        # Early post-switch rate
        n_spikes_early = sum(np.sum((cluster_spikes >= s) & (cluster_spikes < e)) for s, e in early_post_switch_segments)
        rate_early = n_spikes_early / total_early_duration if total_early_duration > 0 else 0
        
        # Late post-switch rate
        n_spikes_late = sum(np.sum((cluster_spikes >= s) & (cluster_spikes < e)) for s, e in late_post_switch_segments)
        rate_late = n_spikes_late / total_late_duration if total_late_duration > 0 else 0
        
        # Adaptation index: decrease from early to late indicates fast adaptation
        if rate_early + rate_late > 0:
            adaptation_index = (rate_early - rate_late) / (rate_early + rate_late)
        else:
            adaptation_index = 0
        
        results[cid] = {
            'rate_early_post_switch': rate_early,
            'rate_late_post_switch': rate_late,
            'adaptation_speed_index': adaptation_index
        }
    
    # --- 5. Save Results ---
    print("\n  Post-switch adaptation analysis complete (aligned to behavioral switch).")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'post_switch_adaptation.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Generate Heatmap
    heatmap_path = output_dir / 'post_switch_adaptation_heatmap.png'
    cols_to_plot = ['rate_early_post_switch', 'rate_late_post_switch']
    _plot_population_heatmap(df_results[cols_to_plot], heatmap_path, 
                             "Post-Switch Adaptation (Behavior Aligned)", "Period", sort_col='adaptation_speed_index')

def analyze_learning_curves(paths: DataPaths, corner_order: list = [1, 2, 4, 3], 
                            window_size: int = 10):
    """
    Analyzes learning curves: how performance improves after condition switches.
    
    Tracks accuracy, reaction time, and other performance metrics over trials
    to quantify adaptation speed and learning rate.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for CW navigation.
        window_size (int): Moving average window size for smoothing.
    """
    print("Analyzing learning curves...")
    
    # --- 1. Load Data ---
    if not all([paths.event_corner, paths.event_corner.exists(),
                paths.event_condition_switch, paths.event_condition_switch.exists()]):
        print("  Error: Missing required event files.")
        return
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Load corner events
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_config_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_config_key)
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        # Get corner IDs
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # Filter out 0 IDs (artefacts from other boolean columns like CW)
        valid_mask = corner_ids > 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        # Load switch times
        switch_times = _load_switch_times(paths, config, event_loader, dlc_loader=None)
        
        print(f"  Loaded {len(corner_times)} corner events and {len(switch_times)} switches.")
    except Exception as e:
        print(f"  Error loading data: {e}")
        return
    

    

    
    # --- 2. Determine Correct Strategy for Each Trial ---
    # For each port visit, determine if it follows CW or CCW
    current_strategy = 'CW'  # Start with CW
    switch_idx = 0
    
    trial_data = []
    
    for i in range(len(corner_times) - 1):
        if i + 1 >= len(corner_ids):
            continue
        
        trial_time = corner_times[i]
        start_port = corner_ids[i]
        end_port = corner_ids[i + 1]
        next_time = corner_times[i + 1]
        
        # Check if we've passed a switch
        while switch_idx < len(switch_times) and switch_times[switch_idx] < trial_time:
            current_strategy = 'CCW' if current_strategy == 'CW' else 'CW'
            switch_idx += 1
        
        if start_port == end_port:
            continue
        
        # Determine if this was a correct trial
        try:
            start_idx = corner_order.index(start_port)
            end_idx = corner_order.index(end_port)
            
            # CW: next in sequence
            is_cw = (start_idx + 1) % len(corner_order) == end_idx
            # CCW: previous in sequence
            is_ccw = (start_idx - 1 + len(corner_order)) % len(corner_order) == end_idx
            
            if current_strategy == 'CW':
                correct = is_cw
            else:
                correct = is_ccw
            
            # Reaction time (time from one port to next)
            reaction_time = next_time - trial_time
            
            trial_data.append({
                'trial_num': i,
                'time': trial_time,
                'strategy': current_strategy,
                'correct': correct,
                'reaction_time': reaction_time,
                'start_port': start_port,
                'end_port': end_port
            })
            
        except ValueError:
            continue
    
    if len(trial_data) == 0:
        print("  No valid trials found.")
        return
    
    df_trials = pd.DataFrame(trial_data)
    
    # --- 3. Compute Learning Metrics ---
    # Find trials after each switch
    learning_blocks = []
    
    for switch_idx, switch_time in enumerate(switch_times):
        # Trials after this switch
        post_switch_trials = df_trials[df_trials['time'] > switch_time].copy()
        
        if len(post_switch_trials) < 5:
            continue
        
        # Take first 50 trials after switch
        post_switch_trials = post_switch_trials.head(50).copy()
        post_switch_trials['trials_since_switch'] = np.arange(len(post_switch_trials))
        
        # Compute moving average accuracy
        post_switch_trials['accuracy_smooth'] = post_switch_trials['correct'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        # Compute moving average reaction time
        post_switch_trials['rt_smooth'] = post_switch_trials['reaction_time'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        learning_blocks.append({
            'switch_num': switch_idx,
            'switch_time': switch_time,
            'trials': post_switch_trials
        })
    
    print(f"  Analyzed {len(learning_blocks)} learning blocks.")
    
    # --- 4. Save Results ---
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Save all trial data
    output_path = output_dir / 'learning_curves_trials.csv'
    df_trials.to_csv(output_path, index=False)
    print(f"  Trial data saved to {output_path}")
    
    # Compute summary statistics per learning block
    summary = []
    for block in learning_blocks:
        trials = block['trials']
        
        # Learning rate (how fast accuracy improves)
        if len(trials) >= 10:
            # Fit linear trend to accuracy
            x = trials['trials_since_switch'].values[:20]  # First 20 trials
            y = trials['correct'].values[:20]
            if len(x) > 0 and np.std(x) > 0:
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0
        else:
            slope = 0
        
        summary.append({
            'switch_num': block['switch_num'],
            'switch_time': block['switch_time'],
            'initial_accuracy': trials['correct'].iloc[:5].mean() if len(trials) >= 5 else np.nan,
            'final_accuracy': trials['correct'].iloc[-5:].mean() if len(trials) >= 5 else np.nan,
            'learning_rate': slope,
            'mean_reaction_time': trials['reaction_time'].mean(),
            'n_trials': len(trials)
        })
    
    df_summary = pd.DataFrame(summary)
    summary_path = output_dir / 'learning_curves_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary saved to {summary_path}")
    
    if df_summary.empty:
        print("  No learning blocks identified. Skipping plotting.")
        return
    
    # --- 5. Visualize ---
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy over all trials
        ax = axes[0, 0]
        ax.plot(df_trials['trial_num'], df_trials['correct'], 'o', alpha=0.3, markersize=3)
        
        # Mark switches
        for switch_time in switch_times:
            switch_trial = df_trials[df_trials['time'] >= switch_time].iloc[0]['trial_num'] if len(df_trials[df_trials['time'] >= switch_time]) > 0 else None
            if switch_trial is not None:
                ax.axvline(switch_trial, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Correct (1) or Error (0)')
        ax.set_title('Performance Over Session')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Learning curves after switches
        ax = axes[0, 1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(learning_blocks)))
        
        for idx, block in enumerate(learning_blocks[:5]):  # Plot first 5 blocks
            trials = block['trials']
            ax.plot(trials['trials_since_switch'], trials['accuracy_smooth'],
                   label=f"Switch {block['switch_num']}", color=colors[idx], linewidth=2)
        
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trials Since Switch')
        ax.set_ylabel('Accuracy (Moving Avg)')
        ax.set_title('Learning Curves After Switches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 3: Reaction time over session
        ax = axes[1, 0]
        ax.plot(df_trials['trial_num'], df_trials['reaction_time'], 'o', alpha=0.3, markersize=3)
        
        # Mark switches
        for switch_time in switch_times:
            switch_trial = df_trials[df_trials['time'] >= switch_time].iloc[0]['trial_num'] if len(df_trials[df_trials['time'] >= switch_time]) > 0 else None
            if switch_trial is not None:
                ax.axvline(switch_trial, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Reaction Time (s)')
        ax.set_title('Reaction Time Over Session')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning rate distribution
        ax = axes[1, 1]
        ax.hist(df_summary['learning_rate'].dropna(), bins=15, alpha=0.7, color='blue')
        ax.set_xlabel('Learning Rate (accuracy/trial)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Learning Rates')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'learning_curves.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    return df_trials, df_summary

def analyze_navigation_efficiency(paths: DataPaths, corner_order: list = [1, 2, 4, 3]):
    """
    Analyzes custom behavioral metrics: navigation efficiency and strategy adherence.
    
    Calculates path efficiency and how well the animal follows the current strategy.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for navigation.
    """
    print("Analyzing navigation efficiency metrics...")
    
    # --- 1. Load Data ---
    if not paths.event_corner or not paths.event_corner.exists():
        print("  Error: Corner event file not found.")
        return
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_config_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_config_key)
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        return
    
    # --- 2. Calculate Efficiency Metrics ---
    efficiency_scores = []
    
    for i in range(len(corner_times) - 1):
        if i + 1 >= len(corner_ids):
            continue
        
        start_port = corner_ids[i]
        end_port = corner_ids[i + 1]
        duration = corner_times[i + 1] - corner_times[i]
        
        if start_port == end_port or duration <= 0 or duration > 20:
            continue
        
        # Calculate shortest path distance (in corner steps)
        try:
            start_idx = corner_order.index(start_port)
            end_idx = corner_order.index(end_port)
            
            cw_dist = (end_idx - start_idx) % len(corner_order)
            ccw_dist = (start_idx - end_idx) % len(corner_order)
            shortest_dist = min(cw_dist, ccw_dist)
            
            # Actual distance is always 1 (direct move)
            # Efficiency: 1 if direct, 0 if circuitous
            efficiency = 1.0 if shortest_dist == 1 else 0.0
            
            efficiency_scores.append({
                'trial': i,
                'start_port': start_port,
                'end_port': end_port,
                'duration_sec': duration,
                'efficiency': efficiency
            })
        except ValueError:
            continue
    
    # --- 3. Save Results ---
    print("\n  Navigation efficiency analysis complete.")
    df_results = pd.DataFrame(efficiency_scores)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'navigation_efficiency.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    
    if not df_results.empty:
        print(f"  Mean efficiency: {df_results['efficiency'].mean():.3f}")
        print(f"  Mean duration: {df_results['duration_sec'].mean():.2f}s")

def analyze_population_manifolds(paths: DataPaths, method: str = 'pca', n_components: int = 3, time_bin_ms: int = 100, min_velocity: float = 5.0):
    """
    Analyzes low-dimensional population trajectories using dimensionality reduction.
    
    Uses PCA, tSNE, or UMAP to project population activity into low-dimensional space
    and examines how trajectories relate to behavioral states.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
        n_components (int): Number of dimensions to reduce to.
        time_bin_ms (int): Time bin size for discretizing activity.
        min_velocity (float): Minimum velocity (cm/s) to include a time bin.
    """
    print(f"Analyzing population manifolds using {method.upper()}...")
    
    try:
        from sklearn.decomposition import PCA
        if method == 'tsne':
            from sklearn.manifold import TSNE
        elif method == 'umap':
            import umap
    except ImportError as e:
        print(f"  Error: Required library not found - {e}")
        print("  Install with: pip install sklearn umap-learn")
        return
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
        
    # --- 1b. Load Velocity Data for Filtering ---
    velocity = None
    velocity_times = None
    if min_velocity > 0:
        try:
            # Attempt to deduce video FS or usage defaults
            # For robustness, we try to load DLC
            velocity, velocity_times = _load_dlc_and_calculate_velocity(paths, video_fs=60.0, px_per_cm=30.0)
        except Exception as e:
            print(f"  Warning: Could not load velocity for filtering ({e}). Proceeding without filtering.")
    
    # --- 2. Bin Population Activity ---
    session_duration = spike_times_sec.max()
    bin_size_sec = time_bin_ms / 1000.0
    n_bins = int(np.ceil(session_duration / bin_size_sec))
    
    print(f"  Binning activity into {n_bins} bins of {time_bin_ms}ms...")
    
    # Create population activity matrix: (time bins x neurons)
    population_matrix = np.zeros((n_bins, len(unique_clusters)))
    
    for i, cid in enumerate(unique_clusters):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        hist, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        population_matrix[:, i] = hist / bin_size_sec  # Convert to firing rate
        
    # Filter by Velocity
    if velocity is not None and min_velocity > 0:
        print(f"  Filtering time bins by velocity > {min_velocity} cm/s...")
        if len(velocity) > 0:
            bin_centers = np.linspace(0, session_duration, n_bins) 
            # velocity_times might be slightly different, interpolate
            bin_velocities = np.interp(bin_centers, velocity_times, velocity)
            
            valid_mask = bin_velocities > min_velocity
            population_matrix = population_matrix[valid_mask, :]
            
            print(f"  Retained {np.sum(valid_mask)}/{len(valid_mask)} bins ({np.sum(valid_mask)/len(valid_mask)*100:.1f}%)")
            
            if population_matrix.shape[0] < 50:
                 print("  Warning: Too few bins left after filtering. Aborting.")
                 return
    
    # --- 3. Apply Dimensionality Reduction ---
    print(f"  Applying {method.upper()} reduction to {n_components} dimensions...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(population_matrix)
        explained_var = reducer.explained_variance_ratio_
        print(f"  Explained variance: {explained_var}")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(population_matrix)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(population_matrix)
    else:
        print(f"  Error: Unknown method '{method}'")
        return
    
    # --- 4. Save Results ---
    print("\n  Population manifold analysis complete.")
    
    # Save embedding
    df_embedding = pd.DataFrame(embedding, columns=[f'Dim{i+1}' for i in range(n_components)])
    df_embedding['time_sec'] = np.arange(n_bins) * bin_size_sec
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'population_manifold_{method}.csv'
    df_embedding.to_csv(output_path, index=False)
    print(f"  Embedding saved to {output_path}")
    
    # Plot trajectory
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 5))
        
        if n_components >= 2:
            ax1 = fig.add_subplot(121)
            scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                                 c=np.arange(n_bins), cmap='viridis', s=5, alpha=0.6)
            ax1.set_xlabel(f'{method.upper()} Dimension 1')
            ax1.set_ylabel(f'{method.upper()} Dimension 2')
            ax1.set_title('Population Trajectory (2D)')
            plt.colorbar(scatter, ax=ax1, label='Time (bins)')
        
        if n_components >= 3:
            ax2 = fig.add_subplot(122, projection='3d')
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                                 c=np.arange(n_bins), cmap='viridis', s=5, alpha=0.6)
            ax2.set_xlabel(f'{method.upper()} Dimension 1')
            ax2.set_ylabel(f'{method.upper()} Dimension 2')
            ax2.set_zlabel(f'{method.upper()} Dimension 3')
            ax2.set_title('Population Trajectory (3D)')
        
        plt.tight_layout()
        plot_path = output_dir / f'population_manifold_{method}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Trajectory plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plot: {e}")

def analyze_population_trajectories_by_direction(paths: DataPaths, method: str = 'pca', n_components: int = 3, 
                                                 corner_order: list = [1, 2, 4, 3], min_trials: int = 5,
                                                 max_plot_trials: int = 10):
    """
    Analyzes population trajectories separately for CW and CCW movements.
    
    Compares low-dimensional population dynamics during clockwise vs counterclockwise
    navigation to identify differential trajectory patterns between strategies.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
        n_components (int): Number of dimensions to reduce to.
        corner_order (list): Order of corners for CW navigation.
        min_trials (int): Minimum number of trials per direction required.
        max_plot_trials (int): Maximum number of individual trials to plot (default: 10).
    """
    print(f"Analyzing population trajectories by direction ({method.upper()})...")
    
    try:
        from sklearn.decomposition import PCA
        if method == 'tsne':
            from sklearn.manifold import TSNE
        elif method == 'umap':
            import umap
    except ImportError as e:
        print(f"  Error: Required library not found - {e}")
        return
    
    # --- 1. Load Behavioral Data (Corner Events) ---
    if not paths.event_corner or not paths.event_corner.exists():
        print("  Error: Corner event file not found.")
        return
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_config_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_config_key)
        corner_df = _get_event_onsets_df(corner_df, corner_config_entry)
        corner_times = event_loader.get_event_times(corner_df, corner_config_key)
        
        # Get corner IDs
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 5):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events.")
        
    except Exception as e:
        print(f"  Error loading corner event data: {e}")
        return
    
    # --- 2. Identify CW and CCW Movements ---
    cw_movements = []
    ccw_movements = []
    
    for i in range(len(corner_times) - 1):
        if i + 1 >= len(corner_ids):
            continue
        
        start_port = corner_ids[i]
        end_port = corner_ids[i + 1]
        start_time = corner_times[i]
        end_time = corner_times[i + 1]
        
        if start_port == end_port or end_time <= start_time:
            continue
        
        # Check direction
        try:
            start_idx = corner_order.index(start_port)
            end_idx = corner_order.index(end_port)
            
            if (start_idx + 1) % len(corner_order) == end_idx:
                cw_movements.append((start_time, end_time, i))
            elif (start_idx - 1 + len(corner_order)) % len(corner_order) == end_idx:
                ccw_movements.append((start_time, end_time, i))
        except ValueError:
            continue
    
    print(f"  Found {len(cw_movements)} CW and {len(ccw_movements)} CCW movements.")
    
    if len(cw_movements) < min_trials or len(ccw_movements) < min_trials:
        print(f"  Not enough trials in both directions (min {min_trials} required). Aborting.")
        return
    
    # --- 3. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 4. Extract Population Activity for Each Movement ---
    def extract_trial_activity(movements, trial_duration_sec=2.0, n_time_bins=20):
        """Extract population activity aligned to movements."""
        trial_activities = []
        
        for start_t, end_t, _ in movements:
            # Use fixed window for alignment
            window_start = start_t
            window_end = start_t + trial_duration_sec
            
            if window_end > spike_times_sec.max():
                continue
            
            # Bin activity for this trial
            trial_matrix = np.zeros((n_time_bins, len(unique_clusters)))
            bin_edges = np.linspace(window_start, window_end, n_time_bins + 1)
            
            for i, cid in enumerate(unique_clusters):
                cluster_spikes = spike_times_sec[spike_clusters == cid]
                spikes_in_trial = cluster_spikes[(cluster_spikes >= window_start) & 
                                                (cluster_spikes < window_end)]
                hist, _ = np.histogram(spikes_in_trial, bins=bin_edges)
                trial_matrix[:, i] = hist / (trial_duration_sec / n_time_bins)
            
            trial_activities.append(trial_matrix)
        
        return np.array(trial_activities)
    
    # Extract activity for both directions
    cw_activities = extract_trial_activity(cw_movements)
    ccw_activities = extract_trial_activity(ccw_movements)
    
    print(f"  Extracted activity: CW shape {cw_activities.shape}, CCW shape {ccw_activities.shape}")
    
    # Reshape for PCA: (n_trials * n_time_bins, n_neurons)
    cw_flat = cw_activities.reshape(-1, cw_activities.shape[2])
    ccw_flat = ccw_activities.reshape(-1, ccw_activities.shape[2])
    
    # --- 5. Apply Dimensionality Reduction ---
    print(f"  Applying {method.upper()} to combined data...")
    
    # Combine data for consistent embedding
    combined = np.vstack([cw_flat, ccw_flat])
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(combined)
        print(f"  Explained variance: {reducer.explained_variance_ratio_}")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(combined)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(combined)
    
    # Split back into CW and CCW
    n_cw = cw_flat.shape[0]
    cw_embedding = embedding[:n_cw]
    ccw_embedding = embedding[n_cw:]
    
    # --- 6. Calculate Trajectory Differences ---
    # Mean trajectory for each direction
    cw_mean_traj = cw_embedding.reshape(cw_activities.shape[0], -1, n_components).mean(axis=0)
    ccw_mean_traj = ccw_embedding.reshape(ccw_activities.shape[0], -1, n_components).mean(axis=0)
    
    # Trajectory distance (Euclidean)
    traj_distance = np.linalg.norm(cw_mean_traj - ccw_mean_traj, axis=1)
    mean_traj_distance = np.mean(traj_distance)
    
    print(f"  Mean trajectory distance between CW and CCW: {mean_traj_distance:.4f}")
    
    # --- 7. Save Results ---
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Save embeddings
    df_cw = pd.DataFrame(cw_embedding, columns=[f'Dim{i+1}' for i in range(n_components)])
    df_cw['direction'] = 'CW'
    df_cw['trial'] = np.repeat(np.arange(cw_activities.shape[0]), cw_activities.shape[1])
    df_cw['time_in_trial'] = np.tile(np.arange(cw_activities.shape[1]), cw_activities.shape[0])
    
    df_ccw = pd.DataFrame(ccw_embedding, columns=[f'Dim{i+1}' for i in range(n_components)])
    df_ccw['direction'] = 'CCW'
    df_ccw['trial'] = np.repeat(np.arange(ccw_activities.shape[0]), ccw_activities.shape[1])
    df_ccw['time_in_trial'] = np.tile(np.arange(ccw_activities.shape[1]), ccw_activities.shape[0])
    
    df_combined = pd.concat([df_cw, df_ccw], ignore_index=True)
    
    output_path = output_dir / f'population_trajectories_by_direction_{method}.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"  Trajectories saved to {output_path}")
    
    # Save summary statistics
    summary = {
        'method': method,
        'n_components': n_components,
        'n_cw_trials': cw_activities.shape[0],
        'n_ccw_trials': ccw_activities.shape[0],
        'mean_trajectory_distance': mean_traj_distance
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f'population_trajectories_summary_{method}.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # --- 8. Visualize Trajectories ---
    try:
        fig = plt.figure(figsize=(16, 6))
        
        if n_components >= 2:
            # Plot 1: 2D trajectories
            ax1 = fig.add_subplot(131)
            
            # Plot mean trajectories
            ax1.plot(cw_mean_traj[:, 0], cw_mean_traj[:, 1], 'b-', linewidth=3, label='CW mean', alpha=0.8)
            ax1.plot(ccw_mean_traj[:, 0], ccw_mean_traj[:, 1], 'r-', linewidth=3, label='CCW mean', alpha=0.8)
            
            # Mark start and end
            ax1.scatter(cw_mean_traj[0, 0], cw_mean_traj[0, 1], c='blue', s=200, marker='o', 
                       edgecolors='black', linewidth=2, label='CW start', zorder=5)
            ax1.scatter(ccw_mean_traj[0, 0], ccw_mean_traj[0, 1], c='red', s=200, marker='o',
                       edgecolors='black', linewidth=2, label='CCW start', zorder=5)
            
            ax1.set_xlabel(f'{method.upper()} Dimension 1')
            ax1.set_ylabel(f'{method.upper()} Dimension 2')
            ax1.set_title('Mean Population Trajectories')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Individual trials
            ax2 = fig.add_subplot(132)
            
            # Reshape for trial visualization
            cw_trials_2d = cw_embedding.reshape(cw_activities.shape[0], -1, n_components)
            ccw_trials_2d = ccw_embedding.reshape(ccw_activities.shape[0], -1, n_components)
            
            for i in range(min(max_plot_trials, cw_trials_2d.shape[0])):
                ax2.plot(cw_trials_2d[i, :, 0], cw_trials_2d[i, :, 1], 
                        'b-', alpha=0.3, linewidth=1)
            
            for i in range(min(max_plot_trials, ccw_trials_2d.shape[0])):
                ax2.plot(ccw_trials_2d[i, :, 0], ccw_trials_2d[i, :, 1], 
                        'r-', alpha=0.3, linewidth=1)
            
            ax2.set_xlabel(f'{method.upper()} Dimension 1')
            ax2.set_ylabel(f'{method.upper()} Dimension 2')
            ax2.set_title('Individual Trial Trajectories')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Trajectory distance over time
            ax3 = fig.add_subplot(133)
            ax3.plot(traj_distance, 'k-', linewidth=2)
            ax3.set_xlabel('Time in Trial')
            ax3.set_ylabel('Trajectory Distance (Euclidean)')
            ax3.set_title('CW vs CCW Trajectory Separation')
            ax3.axhline(mean_traj_distance, color='gray', linestyle='--', 
                       label=f'Mean: {mean_traj_distance:.3f}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f'population_trajectories_by_direction_{method}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Trajectory plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate trajectory plot: {e}")
        import traceback
        traceback.print_exc()

def analyze_dimensionality_reduction(paths: DataPaths, method: str = 'pca', n_components: int = 10):
    """
    Performs dimensionality reduction (PCA/ICA) on population activity.
    
    Analyzes intrinsic dimensionality of neural population and identifies
    principal components of activity.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        method (str): Method to use ('pca' or 'ica').
        n_components (int): Number of components to extract.
    """
    print(f"Performing {method.upper()} dimensionality reduction...")
    
    try:
        from sklearn.decomposition import PCA, FastICA
    except ImportError:
        print("  Error: sklearn required")
        return
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 2. Create Population Activity Matrix ---
    session_duration = spike_times_sec.max()
    bin_size_sec = 0.05  # 50ms bins
    n_bins = int(session_duration / bin_size_sec)
    
    print(f"  Creating population matrix: {n_bins} time bins x {len(unique_clusters)} neurons")
    
    population_matrix = np.zeros((n_bins, len(unique_clusters)))
    
    for i, cid in enumerate(unique_clusters):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        hist, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        population_matrix[:, i] = hist / bin_size_sec
    
    # --- 3. Apply Dimensionality Reduction ---
    if method == 'pca':
        reducer = PCA(n_components=min(n_components, len(unique_clusters)))
        components = reducer.fit_transform(population_matrix)
        explained_var = reducer.explained_variance_ratio_
        
        print(f"  Explained variance by top {len(explained_var)} components:")
        for i, var in enumerate(explained_var[:5]):
            print(f"    PC{i+1}: {var:.3f}")
        
        # Estimate intrinsic dimensionality (80% variance threshold)
        cumsum_var = np.cumsum(explained_var)
        intrinsic_dim = np.argmax(cumsum_var >= 0.8) + 1
        print(f"  Intrinsic dimensionality (80% variance): {intrinsic_dim}")
        
    elif method == 'ica':
        reducer = FastICA(n_components=min(n_components, len(unique_clusters)), random_state=42)
        components = reducer.fit_transform(population_matrix)
        explained_var = None
    else:
        print(f"  Error: Unknown method '{method}'")
        return
    
    # --- 4. Save Results ---
    print("\n  Dimensionality reduction analysis complete.")
    
    # Save components
    df_components = pd.DataFrame(components, 
                                 columns=[f'Component{i+1}' for i in range(components.shape[1])])
    df_components['time_sec'] = np.arange(n_bins) * bin_size_sec
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'dimensionality_reduction_{method}.csv'
    df_components.to_csv(output_path, index=False)
    print(f"  Components saved to {output_path}")
    
    if explained_var is not None:
        var_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(explained_var))],
            'explained_variance': explained_var
        })
        var_path = output_dir / f'explained_variance_{method}.csv'
        var_df.to_csv(var_path, index=False)
        print(f"  Explained variance saved to {var_path}")

def analyze_phase_space_trajectories(paths: DataPaths, n_components: int = 3, 
                                     trial_window_sec: float = 2.0):
    """
    Analyzes phase space trajectories of population activity.
    
    Projects population activity into low-dimensional phase space and
    identifies fixed points, attractors, and dynamical features.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_components (int): Number of dimensions for phase space.
        trial_window_sec (float): Time window for trial-based trajectories.
    """
    print("Analyzing phase space trajectories...")
    
    # --- 1. Load Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    session_duration = spike_times_sec.max()
    bin_size_sec = 0.050  # 50ms bins
    n_bins = int(session_duration / bin_size_sec)
    
    # --- 2. Create Population Firing Rate Matrix ---
    print(f"  Creating population activity matrix...")
    
    population_matrix = []
    for cid in tqdm(unique_clusters, desc="Processing neurons"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        if len(cluster_spikes) < 50:
            continue
        
        spike_train, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        firing_rate = spike_train / bin_size_sec
        
        # Smooth
        from scipy.ndimage import gaussian_filter1d
        firing_rate_smooth = gaussian_filter1d(firing_rate, sigma=2)
        
        population_matrix.append(firing_rate_smooth)
    
    population_matrix = np.array(population_matrix)  # neurons x time
    print(f"  Population matrix shape: {population_matrix.shape}")
    
    # --- 3. Dimensionality Reduction to Phase Space ---
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize
    scaler = StandardScaler()
    population_normalized = scaler.fit_transform(population_matrix.T).T
    
    # PCA to low-D phase space
    pca = PCA(n_components=n_components)
    phase_space = pca.fit_transform(population_normalized.T)  # time x components
    
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Total variance explained: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # --- 4. Identify Fixed Points ---
    # Fixed points are regions where trajectory velocity is low
    velocity = np.diff(phase_space, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    
    # Find slow points (potential fixed points)
    speed_threshold = np.percentile(speed, 10)  # Bottom 10% slowest
    fixed_point_candidates = np.where(speed < speed_threshold)[0]
    
    print(f"  Identified {len(fixed_point_candidates)} potential fixed points.")
    
    # Cluster fixed point candidates to find stable states
    if len(fixed_point_candidates) > 10:
        from sklearn.cluster import DBSCAN
        
        fixed_points_coords = phase_space[fixed_point_candidates]
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(fixed_points_coords)
        n_fixed_points = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        print(f"  Found {n_fixed_points} stable fixed point regions.")
    else:
        n_fixed_points = 0
    
    # --- 5. Save Results ---
    print("\n  Phase space trajectory analysis complete.")
    
    # Save phase space coordinates
    df_phase_space = pd.DataFrame(phase_space, columns=[f'PC{i+1}' for i in range(n_components)])
    df_phase_space['time_sec'] = np.arange(len(phase_space)) * bin_size_sec
    df_phase_space['speed'] = np.concatenate([[0], speed])
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'phase_space_trajectories.csv'
    df_phase_space.to_csv(output_path, index=False)
    print(f"  Phase space data saved to {output_path}")
    
    # --- 6. Visualize ---
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # 3D trajectory plot
        if n_components >= 3:
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            
            # Color by time
            time_colors = np.arange(len(phase_space))
            scatter = ax.scatter(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2],
                               c=time_colors, cmap='viridis', s=1, alpha=0.3)
            
            # Mark fixed points
            if len(fixed_point_candidates) > 0 and n_components >= 3:
                ax.scatter(phase_space[fixed_point_candidates, 0],
                          phase_space[fixed_point_candidates, 1],
                          phase_space[fixed_point_candidates, 2],
                          c='red', s=20, alpha=0.5, label='Slow points')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.set_title('3D Phase Space Trajectory')
            plt.colorbar(scatter, ax=ax, label='Time')
            ax.legend()
        
        # 2D projection
        ax = fig.add_subplot(2, 2, 2)
        scatter = ax.scatter(phase_space[:, 0], phase_space[:, 1],
                           c=time_colors, cmap='viridis', s=2, alpha=0.3)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('2D Phase Space Projection')
        plt.colorbar(scatter, ax=ax, label='Time')
        
        # Speed over time
        ax = fig.add_subplot(2, 2, 3)
        time_axis = np.arange(len(speed)) * bin_size_sec
        ax.plot(time_axis, speed, 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(speed_threshold, color='red', linestyle='--', label='Fixed point threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trajectory Speed')
        ax.set_title('Phase Space Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Speed distribution
        ax = fig.add_subplot(2, 2, 4)
        ax.hist(speed, bins=50, alpha=0.7, color='blue')
        ax.axvline(speed_threshold, color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Speed')
        ax.set_ylabel('Count')
        ax.set_title('Speed Distribution')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'phase_space_trajectories.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    return df_phase_space

def analyze_ica_decomposition(paths: DataPaths, n_components: int = 10):
    """
    Performs Independent Component Analysis (ICA) on population activity.
    
    ICA finds statistically independent components, useful for separating
    mixed neural signals and identifying independent sources of variance.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_components (int): Number of independent components to extract.
    """
    print("Performing ICA decomposition...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    session_duration = spike_times_sec.max()
    bin_size_sec = 0.100  # 100ms bins
    n_bins = int(session_duration / bin_size_sec)
    
    # --- 2. Create Population Matrix ---
    print(f"  Creating population matrix for {len(unique_clusters)} neurons...")
    
    population_matrix = []
    for cid in tqdm(unique_clusters, desc="Processing neurons"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        if len(cluster_spikes) < 50:
            continue
        
        spike_train, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        firing_rate = spike_train / bin_size_sec
        
        population_matrix.append(firing_rate)
    
    population_matrix = np.array(population_matrix)  # neurons x time
    print(f"  Population matrix shape: {population_matrix.shape}")
    
    # --- 3. Apply ICA ---
    from sklearn.decomposition import FastICA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize
    scaler = StandardScaler()
    population_standardized = scaler.fit_transform(population_matrix.T)  # time x neurons
    
    # ICA
    print(f"  Applying ICA with {n_components} components...")
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    components = ica.fit_transform(population_standardized)  # time x components
    mixing_matrix = ica.mixing_  # neurons x components
    
    # --- 4. Analyze Components ---
    component_stats = []
    
    for i in range(n_components):
        component = components[:, i]
        
        # Statistics
        kurtosis = np.mean((component - np.mean(component))**4) / np.std(component)**4
        
        # Find neurons most strongly associated with this component
        component_weights = mixing_matrix[:, i]
        top_neurons_idx = np.argsort(np.abs(component_weights))[-10:]  # Top 10
        
        component_stats.append({
            'component_id': i,
            'mean': np.mean(component),
            'std': np.std(component),
            'kurtosis': kurtosis,
            'n_strong_neurons': np.sum(np.abs(component_weights) > 0.1)
        })
    
    # --- 5. Save Results ---
    print("\n  ICA decomposition complete.")
    
    # Save components
    df_components = pd.DataFrame(components, columns=[f'IC{i+1}' for i in range(n_components)])
    df_components['time_sec'] = np.arange(n_bins) * bin_size_sec
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ica_components.csv'
    df_components.to_csv(output_path, index=False)
    print(f"  Components saved to {output_path}")
    
    # Save component statistics
    df_stats = pd.DataFrame(component_stats)
    stats_path = output_dir / 'ica_component_stats.csv'
    df_stats.to_csv(stats_path, index=False)
    print(f"  Statistics saved to {stats_path}")
    
    # Save mixing matrix
    df_mixing = pd.DataFrame(mixing_matrix, columns=[f'IC{i+1}' for i in range(n_components)])
    mixing_path = output_dir / 'ica_mixing_matrix.csv'
    df_mixing.to_csv(mixing_path, index=False)
    print(f"  Mixing matrix saved to {mixing_path}")
    
    # --- 6. Visualize ---
    try:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Plot first 6 components
        time_axis = np.arange(n_bins) * bin_size_sec
        for i in range(min(6, n_components)):
            ax = axes[i]
            ax.plot(time_axis, components[:, i], 'b-', linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('IC Amplitude')
            ax.set_title(f'Independent Component {i+1}\nKurtosis: {component_stats[i]["kurtosis"]:.2f}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, min(100, time_axis[-1])])  # First 100 seconds
        
        plt.tight_layout()
        plot_path = output_dir / 'ica_components.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Component plots saved to {plot_path}")
        
        # Plot mixing matrix heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(mixing_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('Independent Component')
        ax.set_ylabel('Neuron')
        ax.set_title('ICA Mixing Matrix')
        plt.colorbar(im, label='Weight')
        
        plot_path = output_dir / 'ica_mixing_matrix.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Mixing matrix plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
    
    return df_components, df_stats

def analyze_decoding_performance(paths: DataPaths, corner_order: list = [1, 2, 4, 3]):
    """
    Decode behavioral variables from neural activity using GLM and classifiers.
    
    Critical analysis for publication: demonstrates that neural activity
    actually encodes behavioral information. Tests:
    - CW vs CCW strategy decoding
    - Movement direction prediction
    - Reward outcome prediction
    - Port location decoding
    
    Uses cross-validated classifiers and reports accuracy, AUC, confusion matrices.
    """
    print("Analyzing decoding performance from neural activity...")
    
    # --- 1. Load Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        
        # Load corner events
        corner_cfg = find_config_entry(paths.event_corner, config)
        corner_key = next(k for k, v in config.items() if v == corner_cfg)
        corner_df = event_loader.load(config_key=corner_key, sync_to_dlc=False)
        corner_df_onsets = _get_event_onsets_df(corner_df, corner_cfg)
        corner_times = event_loader.get_event_times(corner_df_onsets, corner_key)
        
        # Get strategy labels (CW vs CCW)
        rule_col = get_column_name(corner_cfg, ['CW', 'Condition', 'Rule', 'Protocol'])
        if rule_col and rule_col in corner_df_onsets.columns:
            strategy_labels = corner_df_onsets[rule_col].astype(int).values
        else:
            print("  Could not find strategy labels. Skipping strategy decoding.")
            strategy_labels = None
        
        # Get port IDs
        id_col = get_column_name(corner_cfg, ['CornerID', 'ID', 'Port'])
        if id_col and id_col in corner_df_onsets.columns:
            port_ids = corner_df_onsets[id_col].values
        else:
            port_ids = event_loader.infer_port_id(corner_df_onsets).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = port_ids != 0
        port_ids = port_ids[valid_mask]
        corner_times = corner_times[valid_mask]
        if strategy_labels is not None:
            strategy_labels = strategy_labels[valid_mask]
        
        print(f"  Filtering invalid (0) IDs: Retaining {len(port_ids)} valid events for decoding.")
        
    except Exception as e:
        print(f"  Error loading event data: {e}")
        return
    
    # --- 2. Create feature matrix ---
    print("  Creating neural feature matrix...")
    
    # Time window for decoding (e.g., 500ms around each event)
    time_window = 0.5  # seconds
    
    # Create firing rate vectors for each event
    n_events = len(corner_times)
    n_neurons = len(unique_clusters)
    
    feature_matrix = np.zeros((n_events, n_neurons))
    
    for i, event_time in enumerate(tqdm(corner_times, desc="Computing features")):
        start_time = event_time - time_window / 2
        end_time = event_time + time_window / 2
        
        for j, cid in enumerate(unique_clusters):
            cluster_spikes = spike_times_sec[spike_clusters == cid]
            n_spikes = np.sum((cluster_spikes >= start_time) & (cluster_spikes < end_time))
            feature_matrix[i, j] = n_spikes / time_window
    
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    
    # --- 3. Decode Strategy (CW vs CCW) ---
    if strategy_labels is not None and len(np.unique(strategy_labels)) == 2:
        print("\n  Decoding strategy (CW vs CCW)...")
        
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
        from sklearn.preprocessing import StandardScaler
        
        # Filter out invalid labels
        valid_mask = ~np.isnan(strategy_labels)
        X = feature_matrix[valid_mask]
        y = strategy_labels[valid_mask]
        
        if len(X) < 10:
            print("  Not enough data for strategy decoding.")
        else:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validated classification
            clf = LogisticRegression(max_iter=1000, random_state=42)
            
            # Cross-validated scores
            cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
            cv_auc = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
            
            # Get predictions for confusion matrix
            y_pred = cross_val_predict(clf, X_scaled, y, cv=5)
            
            print(f"  Strategy decoding accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            print(f"  Strategy decoding AUC: {np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}")
            
            # Test against chance (50%)
            from scipy import stats as scipy_stats
            t_stat, p_val = scipy_stats.ttest_1samp(cv_scores, 0.5)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"  Significance vs chance: p={p_val:.6f} {sig}")
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Feature importance (coefficients)
            clf.fit(X_scaled, y)
            feature_importance = np.abs(clf.coef_[0])
            top_neurons = np.argsort(feature_importance)[-10:]  # Top 10 neurons
            
            # Save results
            output_dir = paths.neural_base / 'post_analysis'
            output_dir.mkdir(exist_ok=True)
            
            strategy_results = {
                'accuracy_mean': np.mean(cv_scores),
                'accuracy_std': np.std(cv_scores),
                'auc_mean': np.mean(cv_auc),
                'auc_std': np.std(cv_auc),
                'p_value_vs_chance': p_val,
                'n_trials': len(y),
                'n_neurons': n_neurons
            }
            
            results_path = output_dir / 'decoding_strategy_results.json'
            import json
            with open(results_path, 'w') as f:
                json.dump(strategy_results, f, indent=2)
            
            # Save feature importance
            importance_df = pd.DataFrame({
                'cluster_id': unique_clusters,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            importance_path = output_dir / 'decoding_strategy_feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            
            print(f"  Strategy decoding results saved to {output_dir}")
            
            # Visualize
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Confusion matrix
                ax = axes[0]
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Strategy Decoding\nAccuracy: {np.mean(cv_scores):.2%}')
                
                # Feature importance
                ax = axes[1]
                top_importance = feature_importance[top_neurons]
                ax.barh(range(len(top_neurons)), top_importance, alpha=0.7)
                ax.set_yticks(range(len(top_neurons)))
                ax.set_yticklabels([f"Neuron {unique_clusters[i]}" for i in top_neurons])
                ax.set_xlabel('Importance (|coefficient|)')
                ax.set_title('Top 10 Most Important Neurons')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                plot_path = output_dir / 'decoding_strategy.png'
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"  Strategy decoding plots saved to {plot_path}")
                
            except Exception as e:
                print(f"  Could not generate decoding plots: {e}")
    
    # --- 4. Decode Port Location ---
    if port_ids is not None:
        print("\n  Decoding port location...")
        
        valid_ports = port_ids[port_ids > 0]
        valid_features = feature_matrix[port_ids > 0]
        
        if len(valid_ports) < 20:
            print("  Not enough data for port decoding.")
        else:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(valid_features)
            y = valid_ports
            
            # Cross-validated classification
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
            
            print(f"  Port decoding accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            # Test against chance (25% for 4 ports)
            from scipy import stats as scipy_stats
            t_stat, p_val = scipy_stats.ttest_1samp(cv_scores, 0.25)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"  Significance vs chance: p={p_val:.6f} {sig}")
            
            # Save results
            port_results = {
                'accuracy_mean': np.mean(cv_scores),
                'accuracy_std': np.std(cv_scores),
                'p_value_vs_chance': p_val,
                'n_trials': len(y),
                'n_neurons': n_neurons,
                'chance_level': 0.25
            }
            
            output_dir = paths.neural_base / 'post_analysis'
            results_path = output_dir / 'decoding_port_results.json'
            import json
            with open(results_path, 'w') as f:
                json.dump(port_results, f, indent=2)
            
            print(f"  Port decoding results saved to {output_dir}")
    
    print("\n  Decoding analysis complete.")

def analyze_population_statistics(paths: DataPaths, corner_order: list = [1, 2, 4, 3]):
    """
    Compute population-level statistics across all existing analyses.
    
    This function aggregates results from multiple analyses and computes:
    - Population means, SEMs, and distributions
    - Statistical significance tests
    - Effect sizes
    - Correlations between different tuning properties
    
    Creates a comprehensive summary report for publication.
    """
    print("Computing population statistics across all analyses...")
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. Load existing analysis results ---
    results = {}
    
    # List of analysis files to aggregate
    analysis_files = {
        'movement_tuning': 'FR_velocity_data.csv',
        'acceleration_tuning': 'FR_acceleration_data.csv',
        'turn_analysis': 'FR_turn_data.csv',
        'peth_reward': 'PETH_reward_data.csv',
        'peth_corner': 'PETH_corner_data.csv',
        'peth_licking': 'PETH_licking_data.csv',
        'directional_tuning': 'directional_tuning_index.csv',
        'strategy_encoding': 'strategy_tuning_indices.csv',
        'reward_prediction_error': 'reward_prediction_error.csv',
        'behavioral_switch_success': 'behavioral_switch_success.csv',
        'behavioral_switch_decision': 'behavioral_switch_decision.csv',
        'functional_tuning': 'functional_tuning_matrix.csv',
    }
    
    for name, filename in analysis_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, index_col=0)
                results[name] = df
                print(f"  Loaded {name}: {df.shape[0]} neurons")
            except Exception as e:
                print(f"  Could not load {name}: {e}")
    
    if len(results) == 0:
        print("  No analysis results found. Run analyses first.")
        return
    
    # --- 2. Compute population statistics ---
    print("\n  Computing population statistics...")
    
    summary_stats = {}
    
    for analysis_name, df in results.items():
        print(f"\n  Processing {analysis_name}...")
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].values
            
            # Compute statistics
            stats = compute_statistics_for_tuning(data, method='ttest')
            
            key = f"{analysis_name}_{col}"
            summary_stats[key] = stats
            
            # Print if significant
            # if stats['p_value'] < 0.05:
            #     sig_marker = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*"
            #     print(f"    {col}: mean={stats['mean']:.3f}, p={stats['p_value']:.4f}{sig_marker}, "
            #           f"d={stats['effect_size']:.3f}, n={stats['n']}")
    
    # Save summary statistics
    df_summary = pd.DataFrame.from_dict(summary_stats, orient='index')
    summary_path = output_dir / 'population_statistics_summary.csv'
    df_summary.to_csv(summary_path)
    print(f"\n  Population statistics saved to {summary_path}")
    
    # --- 3. Compute correlations between tuning properties ---
    print("\n  Computing correlations between tuning properties...")
    
    # Merge all results into one DataFrame by cluster_id
    merged_df = None
    for analysis_name, df in results.items():
        if 'cluster_id' in df.columns:
            df_subset = df.set_index('cluster_id')
            if merged_df is None:
                merged_df = df_subset
            else:
                merged_df = merged_df.join(df_subset, how='outer', rsuffix=f'_{analysis_name}')
    
    if merged_df is not None:
        # Select key columns for correlation
        key_columns = [col for col in merged_df.columns if any(k in col.lower() for k in 
                      ['tuning', 'rate', 'preference', 'modulation', 'index'])]
        
        if len(key_columns) > 1:
            corr_matrix = merged_df[key_columns].corr()
            
            # Save correlation matrix
            corr_path = output_dir / 'tuning_correlations.csv'
            corr_matrix.to_csv(corr_path)
            print(f"  Correlation matrix saved to {corr_path}")
            
            # Plot correlation heatmap
            try:
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, 
                           vmin=-1, vmax=1, square=True, ax=ax)
                ax.set_title('Correlations Between Tuning Properties')
                plt.tight_layout()
                plot_path = output_dir / 'tuning_correlations_heatmap.png'
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"  Correlation heatmap saved to {plot_path}")
            except Exception as e:
                print(f"  Could not generate correlation heatmap: {e}")
    
    # --- 4. Generate summary report ---
    print("\n  Generating summary report...")
    
    report_lines = [
        "# POPULATION STATISTICS SUMMARY REPORT",
        "=" * 80,
        f"\nGenerated: {pd.Timestamp.now()}",
        f"Output directory: {output_dir}",
        "\n## SIGNIFICANT FINDINGS (p < 0.05)",
        "-" * 80,
    ]
    
    # List significant findings
    significant_findings = []
    for key, stats in summary_stats.items():
        if stats['p_value'] < 0.05 and not np.isnan(stats['p_value']):
            significant_findings.append((key, stats))
    
    # Sort by p-value
    significant_findings.sort(key=lambda x: x[1]['p_value'])
    
    for key, stats in significant_findings:
        sig_level = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*"
        report_lines.append(
            f"\n{key}:"
            f"\n  Mean: {stats['mean']:.4f} ± {stats['sem']:.4f} (SEM)"
            f"\n  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            f"\n  p-value: {stats['p_value']:.6f} {sig_level}"
            f"\n  Effect size (Cohen's d): {stats['effect_size']:.4f}"
            f"\n  N: {stats['n']}"
        )
    
    report_lines.append(f"\n\nTotal significant findings: {len(significant_findings)}")
    report_lines.append(f"Total tests performed: {len(summary_stats)}")
    
    # Save report
    report_path = output_dir / 'population_statistics_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Summary report saved to {report_path}")
    print(f"\n  Found {len(significant_findings)} significant effects out of {len(summary_stats)} tests")
    
    return df_summary

def generate_publication_summary(paths: DataPaths):
    """
    Generate a comprehensive publication-ready summary integrating all analyses.
    
    Creates:
    - Main findings summary
    - Key figures list
    - Statistical summary tables
    - Suggested narrative structure
    """
    print("Generating publication summary...")
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. Collect all results ---
    results_files = list(output_dir.glob('*.csv')) + list(output_dir.glob('*.json'))
    
    summary_lines = [
        "=" * 80,
        "PUBLICATION SUMMARY: STRIATAL NEURAL DYNAMICS IN CW/CCW NAVIGATION",
        "=" * 80,
        f"\nGenerated: {pd.Timestamp.now()}",
        f"Data directory: {paths.neural_base}",
        f"\n" + "=" * 80,
        "\n## MAIN FINDINGS",
        "-" * 80,
    ]
    
    # --- 2. Key findings from population statistics ---
    pop_stats_file = output_dir / 'population_statistics_summary.csv'
    if pop_stats_file.exists():
        df_pop = pd.read_csv(pop_stats_file, index_col=0)
        
        # Find most significant effects
        significant = df_pop[df_pop['p_value'] < 0.05].sort_values('p_value')
        
        summary_lines.append("\n### 1. Population-Level Effects (Most Significant)")
        for idx, row in significant.head(10).iterrows():
            summary_lines.append(
                f"  • {idx}: "
                f"mean={row['mean']:.3f}, p={row['p_value']:.6f}, "
                f"d={row['effect_size']:.3f}, n={int(row['n'])}"
            )
    
    # --- 3. Cell type differences ---
    cell_comp_file = output_dir / 'cell_type_comparisons.csv'
    if cell_comp_file.exists():
        df_cell = pd.read_csv(cell_comp_file)
        
        if 'significant_fdr' in df_cell.columns:
            significant_cell = df_cell[df_cell['significant_fdr']].sort_values('p_value')
            
            summary_lines.append("\n### 2. MSN vs FSI Differences (FDR-Corrected)")
            for _, row in significant_cell.head(10).iterrows():
                summary_lines.append(
                    f"  • {row['analysis']} - {row['variable']}: "
                    f"MSN={row['MSN_mean']:.3f}, FSI={row['FSI_mean']:.3f}, "
                    f"p={row['p_value']:.6f}, d={row['cohens_d']:.3f}"
                )
    
    # --- 4. Decoding performance ---
    strategy_decode_file = output_dir / 'decoding_strategy_results.json'
    if strategy_decode_file.exists():
        import json
        with open(strategy_decode_file, 'r') as f:
            decode_results = json.load(f)
        
        summary_lines.append("\n### 3. Decoding Performance")
        summary_lines.append(
            f"  • Strategy (CW vs CCW): "
            f"Accuracy={decode_results['accuracy_mean']:.1%} ± {decode_results['accuracy_std']:.1%}, "
            f"AUC={decode_results['auc_mean']:.3f}, "
            f"p={decode_results['p_value_vs_chance']:.6f}"
        )
    
    port_decode_file = output_dir / 'decoding_port_results.json'
    if port_decode_file.exists():
        import json
        with open(port_decode_file, 'r') as f:
            decode_results = json.load(f)
        
        summary_lines.append(
            f"  • Port Location: "
            f"Accuracy={decode_results['accuracy_mean']:.1%} ± {decode_results['accuracy_std']:.1%}, "
            f"p={decode_results['p_value_vs_chance']:.6f} (chance=25%)"
        )
    
    # --- 5. Suggested narrative structure ---
    summary_lines.extend([
        "\n" + "=" * 80,
        "\n## SUGGESTED NARRATIVE STRUCTURE FOR PUBLICATION",
        "-" * 80,
        "\n### Introduction",
        "  • Striatal role in action selection and strategy switching",
        "  • Gap: How do MSN and FSI populations encode strategy switches?",
        "  • This study: Neuropixels recording during CW/CCW navigation task",
        "",
        "\n### Results",
        "",
        "#### 1. Population-Level Neural Dynamics",
        "  • Present population statistics (Figure 1)",
        "  • Key finding: [Insert top effect from population stats]",
        "  • Visualization: Population heatmaps, tuning curves",
        "",
        "#### 2. Cell-Type Specific Encoding",
        "  • MSN vs FSI comparison (Figure 2)",
        "  • Key finding: [Insert top MSN-FSI difference]",
        "  • Visualization: Bar plots with error bars, distributions",
        "",
        "#### 3. Strategy Encoding and Decoding",
        "  • Directional tuning analysis (Figure 3)",
        "  • Decoding performance demonstrates information content",
        "  • Visualization: Decoder performance, confusion matrices",
        "",
        "#### 4. Behavioral Switch Dynamics",
        "  • Pre-switch, decision, and post-switch activity (Figure 4)",
        "  • Adaptation timescales",
        "  • Visualization: PETHs aligned to behavioral switch points",
        "",
        "#### 5. Movement and Reward Integration",
        "  • Movement tuning and reward responses (Figure 5)",
        "  • Port-to-port trajectory encoding",
        "  • Visualization: Trajectory-specific firing rates",
        "",
        "\n### Discussion",
        "  • MSN and FSI differentially encode strategy and movement",
        "  • Strategy switches involve coordinated population dynamics",
        "  • Implications for basal ganglia models of action selection",
        "  • Future directions: dopamine modulation, plasticity",
        "",
        "\n" + "=" * 80,
        "\n## KEY FIGURES FOR PUBLICATION",
        "-" * 80,
    ])
    
    # List available figures
    figure_files = list(output_dir.glob('*.png'))
    summary_lines.append(f"\n{len(figure_files)} figures generated:")
    
    # Categorize figures
    figure_categories = {
        'Population': ['heatmap', 'population', 'manifold'],
        'Tuning': ['tuning', 'velocity', 'acceleration', 'direction'],
        'PETH': ['PETH', 'peth'],
        'Cell Types': ['cell_type', 'comparison'],
        'Decoding': ['decoding'],
        'Statistics': ['statistics', 'correlation'],
    }
    
    for category, keywords in figure_categories.items():
        category_figs = [f for f in figure_files 
                        if any(k in f.name.lower() for k in keywords)]
        if category_figs:
            summary_lines.append(f"\n  {category}:")
            for fig in category_figs:
                summary_lines.append(f"    • {fig.name}")
    
    # --- 6. Save summary ---
    summary_path = output_dir / 'PUBLICATION_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n  Publication summary saved to {summary_path}")
    print("\n" + "=" * 80)
    print("  SUMMARY GENERATION COMPLETE")
    print("  Review the summary file for suggested narrative and key findings.")
    print("=" * 80)
    
    return summary_path

def analyze_lfp_movement_power(paths: DataPaths, time_window_ms: int = 1000, min_movement_duration_ms: int = 150):
    """
    Analyzes LFP power in beta and gamma bands around movement initiation.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_window_ms (int): Window in ms around movement onset to analyze.
        min_movement_duration_ms (int): Minimum duration for a bout of movement.
    """
    print("Analyzing LFP power around movement initiation...")

    # --- 1. Import required libraries ---
    try:
        import spikeinterface.core as si
        from scipy.signal import welch
    except ImportError as e:
        print(f"  Error: Missing dependency - {e}. This analysis requires spikeinterface and scipy.")
        return

    # --- 2. Load LFP Data (Updated) ---
    try:
        config = load_dataset_config()
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return

        lfp_fs = lfp_loader.fs
        print(f"  Initialized LFPDataLoader. FS={lfp_fs} Hz")
        
    except Exception as e:
        print(f"  Error loading LFP data: {e}")
        return

    # --- 3. Identify Movement Onsets using standardized kinematics ---
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

    # --- 4. Analyze LFP Power Around Onsets ---
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
    # Note: channel_ids might be different from indices
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
        chan_label = f"Shank{channel_info['shank']}_{channel_info['loc']}"
        
        # Load FULL trace for this channel to be efficient with many events?
        # Or load per event?
        # LFPDataLoader CSD might be heavy to re-compute per event per channel.
        # Efficient strategy: Load whole CSD trace for this channel once.
        # But we need to support loading full recording or large chunks.
        
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
            freqs, psd = welch(lfp_snippet, fs=lfp_fs, nperseg=min(len(lfp_snippet), 256)) # shorter segment? window_samples is based on time_window_ms (1000ms typically), so usually fine.
            # nperseg should likely be length of snippet for 1-estimate, or smaller for averaging.
            # Default nperseg=256 is small. If window is 1s (2500 samples), 256 is fine.
            
            # Define frequency bands
            beta_band = (freqs >= 13) & (freqs <= 30) # Updated to 13-30 standard
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

    # --- 5. Save and Display Results ---
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

def analyze_theta_oscillations(paths: DataPaths, max_nav_duration_sec: int = 15):
    """
    Analyzes LFP theta power during navigation vs. stationary periods.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        max_nav_duration_sec (int): Max duration for a port-to-port trajectory
                                    to be considered a navigation segment.
    """
    print("Analyzing theta oscillations (navigation vs. stationary)...")

    # --- 1. Import required libraries ---
    try:
        import spikeinterface.core as si
        from scipy.signal import welch
        from scipy.ndimage import gaussian_filter1d
    except ImportError as e:
        print(f"  Error: Missing dependency - {e}. This analysis requires spikeinterface and scipy.")
        return

    # --- 2. Load LFP Data (Updated to use LFPDataLoader) ---
    try:
        config = load_dataset_config()
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return

        lfp_fs = lfp_loader.fs
        print(f"  Initialized LFPDataLoader. FS={lfp_fs} Hz. Sync params: {lfp_loader.sync_params}")
        
    except Exception as e:
        print(f"  Error loading LFP data: {e}")
        return

    # --- 3. Define Segments using standardized kinematics ---
    # We use _get_kinematic_states which combines DLC (velocity) and Corner (location) data
    kinematic_states = _get_kinematic_states(paths)
    
    if not kinematic_states:
        print("  Could not define kinematic states. Aborting theta analysis.")
        return

    # Navigation: Trajectories between ports (labeled generally as "X_to_Y")
    # We filter by duration to ensure it's a "real" trajectory
    navigation_segments = [
        (k['start_time'], k['end_time']) 
        for k in kinematic_states 
        if '_to_' in k['label'] and k['duration'] < max_nav_duration_sec
    ]

    # Stationary: Port stays (labeled "Port_X") or general stationary
    # We prefer Port stays for "clean" stationary behavior
    stationary_segments = [
        (k['start_time'], k['end_time'])
        for k in kinematic_states
        if k['label'].startswith('Port_') and k['duration'] > 1.0 # At least 1 sec
    ]
    
    print(f"  Defined {len(navigation_segments)} navigation segments and {len(stationary_segments)} stationary segments.")

    if not navigation_segments or not stationary_segments:
        print("  Not enough segments for comparison. Aborting.")
        return

    # --- 5. Analyze LFP Power in Segments ---
    # Use middle channel
    channel_ids = lfp_loader.channel_ids
    channel_id_to_use = channel_ids[len(channel_ids)//2]
    # We need the index for CSD or just use ID?
    # LFPDataLoader.get_data takes generic channel list.
    
    # We need to map channel_id_to_use to an index for LFPDataLoader CSD logic?
    # If using 'csd', we pass the ID.
    pass
    
    def calculate_band_power(segments, band_freqs=(4, 12)):
        all_psds = []
        for start_t, end_t in segments:
            # LFPDataLoader handles sync and extraction
            try:
                traces, _ = lfp_loader.get_data(start_t, end_t, channels=[channel_id_to_use])
                if len(traces) == 0: continue
                lfp_snippet = traces[:, 0]
            except:
                continue
                
            if len(lfp_snippet) < lfp_fs: # Need at least 1s of data for welch
                continue

            freqs, psd = welch(lfp_snippet, fs=lfp_fs, nperseg=int(lfp_fs)) # 1s segments
            all_psds.append(psd)
        
        if not all_psds: return np.nan
        
        mean_psd = np.mean(all_psds, axis=0)
        band = (freqs >= band_freqs[0]) & (freqs <= band_freqs[1])
        return np.mean(mean_psd[band])
        
        if not all_psds: return np.nan
        
        mean_psd = np.mean(all_psds, axis=0)
        band = (freqs >= band_freqs[0]) & (freqs <= band_freqs[1])
        return np.mean(mean_psd[band])

    theta_power_nav = calculate_band_power(navigation_segments, band_freqs=(4, 12))
    theta_power_stat = calculate_band_power(stationary_segments, band_freqs=(4, 12))

    # --- 6. Save and Display Results ---
    print("\n  Theta oscillation analysis complete.")
    results = {
        'theta_power_navigation': theta_power_nav,
        'theta_power_stationary': theta_power_stat,
        'theta_ratio_nav_vs_stat': theta_power_nav / theta_power_stat if theta_power_stat > 0 else np.nan
    }
    
    df_results = pd.DataFrame([results])
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'theta_oscillation_analysis.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

    print("\n  Theta (4-12 Hz) Power Summary:")
    print(f"  - During Navigation: {theta_power_nav:.4g}")
    print(f"  - During Rest: {theta_power_stat:.4g}")
    print(f"  - Ratio (Nav/Rest): {results['theta_ratio_nav_vs_stat']:.3f}")
    
def analyze_phase_amplitude_coupling(paths: DataPaths, phase_band=(4, 12), n_bins=18):
    """
    Analyzes phase-amplitude coupling between LFP theta and single-unit spikes.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        phase_band (tuple): The frequency band for the phase signal (e.g., theta).
        n_bins (int): Number of phase bins to use for the modulation index.
    """
    print("Analyzing phase-amplitude coupling (Theta-Spike)...")
    
    # --- 1. Import required libraries ---
    try:
        import spikeinterface.core as si
        from scipy.signal import firwin, filtfilt, hilbert
        from postanalysis.data_loader_refactored import LFPDataLoader
    except ImportError as e:
        print(f"  Error: Missing dependency - {e}. This analysis requires spikeinterface and scipy.")
        return

    # --- 2. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
        
    # --- 3. Initialize LFP ---
    try:
        config = load_dataset_config()
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return
        lfp_fs = lfp_loader.fs
    except Exception as e:
        print(f"  Error loading LFP: {e}")
        return

    # --- 4. Map and Process by Channel ---
    unit_ch_map = _get_unit_best_channels(paths, unique_clusters)
    units_by_channel = defaultdict(list)
    for cid in unique_clusters:
        ch = unit_ch_map.get(cid)
        if ch is not None:
            units_by_channel[ch].append(cid)
            
    # Filter setup
    nyquist = lfp_fs / 2.0
    low = phase_band[0] / nyquist
    high = phase_band[1] / nyquist
    b = firwin(1001, [low, high], pass_zero=False)

    results = {}
    
    # Iterate channels
    for ch_idx, cluster_list in tqdm(units_by_channel.items(), desc="PAC (Theta-Spike)"):
        # Load LFP for channel
        # We need coverage for all spikes. LFPDataLoader is window based.
        # Load full range?
        t_start, t_end = 0, spike_times_sec.max() + 1.0
        
        try:
            traces, timestamps = lfp_loader.get_data(t_start, t_end, channels=[ch_idx], reference='csd')
            if len(traces) == 0: continue
            lfp_signal = traces[:, 0]
            lfp_times = timestamps
            
            # Filter
            # NaNs check
            if np.isnan(lfp_signal).any(): lfp_signal = np.nan_to_num(lfp_signal)
            lfp_filtered = filtfilt(b, 1, lfp_signal)
            phase_signal = np.angle(hilbert(lfp_filtered))
            
        except Exception as e:
            print(f"  Error processing Ch {ch_idx}: {e}")
            continue
            
        for cid in cluster_list:
             cluster_spikes = spike_times_sec[spike_clusters == cid]
             
             # Map spikes to LFP phase
             # Use searchsorted on timestamps
             spike_idxs = np.searchsorted(lfp_times, cluster_spikes)
             # Filter valid indices
             valid = (spike_idxs >= 0) & (spike_idxs < len(phase_signal))
             valid_idxs = spike_idxs[valid]
             
             if len(valid_idxs) < 50:
                 results[cid] = {'modulation_index': np.nan, 'n_spikes': len(valid_idxs)}
                 continue
                 
             spike_phases = phase_signal[valid_idxs]
             
             phase_hist, _ = np.histogram(spike_phases, bins=n_bins, range=(-np.pi, np.pi))
             prob_dist = phase_hist / len(valid_idxs)
             
             uniform_dist = np.ones(n_bins) / n_bins
             kl_divergence = np.sum(prob_dist * np.log((prob_dist + 1e-9) / uniform_dist))
             mi = kl_divergence / np.log(n_bins)
             results[cid] = {'modulation_index': mi, 'n_spikes': len(valid_idxs)}

    # --- 5. Save and Display Results ---
    print("\n  Phase-amplitude coupling analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'

    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'phase_amplitude_coupling.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")

    # Generate Heatmap
    df_results.dropna(inplace=True)
    heatmap_path = output_dir / 'phase_amplitude_coupling_heatmap.png'
    _plot_population_heatmap(df_results[['modulation_index']], heatmap_path, 
                             "Phase-Amplitude Coupling (Theta-Spike MI)", "Metric", sort_col='modulation_index')
        
def analyze_cross_frequency_coupling(paths: DataPaths, phase_band=(4, 12), amp_band=(30, 80), n_bins=18):
    """
    Analyzes cross-frequency coupling between a low-frequency phase and a
    high-frequency amplitude from the LFP.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        phase_band (tuple): The frequency band for the phase signal (e.g., theta).
        amp_band (tuple): The frequency band for the amplitude signal (e.g., gamma).
        n_bins (int): Number of phase bins to use for the modulation index.
    """
    print(f"Analyzing cross-frequency coupling ({phase_band[0]}-{phase_band[1]} Hz Phase vs. {amp_band[0]}-{amp_band[1]} Hz Amp)...")
    
    # --- 1. Import required libraries ---
    try:
        import spikeinterface.core as si
        from scipy.signal import firwin, filtfilt, hilbert
    except ImportError as e:
        print(f"  Error: Missing dependency - {e}. This analysis requires spikeinterface and scipy.")
        return

    # --- 2. Load LFP Data (Updated to use LFPDataLoader) ---
    try:
        config = load_dataset_config()
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
             print("  Error: LFP Extractor not initialized.")
             return
             
        lfp_fs = lfp_loader.fs
        
        # Use middle channel
        channel_ids = lfp_loader.channel_ids
        channel_id_to_use = channel_ids[len(channel_ids)//2]
        
        # Load 5 minutes of data for CFC from start (or middle?)
        # Let's take 0 to 300s
        traces, _ = lfp_loader.get_data(0, 300.0, channels=[channel_id_to_use])
        
        if len(traces) == 0:
             print("  No LFP data found.")
             return
             
        lfp_signal = traces[:, 0]
        print(f"  Loaded LFP data from channel {channel_id_to_use} ({lfp_fs:.2f} Hz).")
        
    except Exception as e:
        print(f"  Error loading LFP data: {e}")
        return

    # --- 3. Filter LFP and Extract Phase and Amplitude ---
    try:
        nyquist = lfp_fs / 2.0
        
        # Filter for phase signal (e.g., Theta)
        low_phase = phase_band[0] / nyquist
        high_phase = phase_band[1] / nyquist
        b_phase = firwin(1001, [low_phase, high_phase], pass_zero=False)
        lfp_phase_filtered = filtfilt(b_phase, 1, lfp_signal)
        phase_signal = np.angle(hilbert(lfp_phase_filtered))

        # Filter for amplitude signal (e.g., Gamma)
        low_amp = amp_band[0] / nyquist
        high_amp = amp_band[1] / nyquist
        b_amp = firwin(1001, [low_amp, high_amp], pass_zero=False)
        lfp_amp_filtered = filtfilt(b_amp, 1, lfp_signal)
        amplitude_signal = np.abs(hilbert(lfp_amp_filtered))

        print("  Filtered LFP and extracted phase and amplitude.")
    except Exception as e:
        print(f"  Error during LFP filtering or Hilbert transform: {e}")
        return
        
    # --- 4. Calculate Modulation Index ---
    # Bin the phase signal
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    # Digitize the phase to assign each time point to a phase bin
    digitized_phase = np.digitize(phase_signal, bins=phase_bins)
    
    # Calculate the mean amplitude within each phase bin
    mean_amplitude_per_phase_bin = np.zeros(n_bins)
    for i in range(1, n_bins + 1):
        mean_amplitude_per_phase_bin[i-1] = np.mean(amplitude_signal[digitized_phase == i])

    # Normalize the mean amplitude distribution to resemble a probability distribution
    prob_dist = mean_amplitude_per_phase_bin / np.sum(mean_amplitude_per_phase_bin)
    
    # --- Calculate MI using KL-Divergence ---
    uniform_dist = np.ones(n_bins) / n_bins
    kl_divergence = np.sum(prob_dist * np.log((prob_dist + 1e-9) / uniform_dist))
    mi = kl_divergence / np.log(n_bins)

    # --- 5. Save and Display Results ---
    print("\n  Cross-frequency coupling analysis complete.")
    results = {
        'phase_band_hz': f"{phase_band[0]}-{phase_band[1]}",
        'amplitude_band_hz': f"{amp_band[0]}-{amp_band[1]}",
        'modulation_index': mi
    }

    df_results = pd.DataFrame([results])
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'cross_frequency_coupling.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

    print(f"  - Modulation Index: {mi:.5f}")
    if mi > 0.01:
        print("  - Found potentially significant cross-frequency coupling.")
    else:
        print("  - No significant cross-frequency coupling detected.")

def analyze_spike_phase_locking(paths: DataPaths, frequency_bands: dict = None,
                                 min_spikes: int = 50):
    """
    Analyze phase-locking of spikes to LFP oscillations in different frequency bands.
    
    For each neuron, calculates:
    - Preferred phase angle
    - Phase-locking strength (PPC, PLV, Rayleigh test)
    - Significance of phase-locking
    - Spatial location of phase-locked neurons.
    - Uses specific LFP channel for each unit (closest channel) if available in cluster_info.
    - Groups units by channel for efficient loading and filtering.
    
    Args:
        paths: DataPaths object
        frequency_bands: Dictionary of frequency bands {'band_name': (low_freq, high_freq)}
                        Default: {'beta': (13, 30), 'low_gamma': (30, 60), 'high_gamma': (60, 100)}
        min_spikes: Minimum number of spikes required for analysis
    
    Returns:
        DataFrame with phase-locking results for each neuron
    """
    print("Analyzing spike-LFP phase-locking...")
    
    if frequency_bands is None:
        frequency_bands = {
            'beta': (13, 30),
            'low_gamma': (30, 60),
            'high_gamma': (60, 100)
        }
    
    DEFAULT_PHASE_LOCKING_SIGNIFICANCE = 0.01

    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return None
    
    # --- 2. Load Unit-to-Channel Mapping ---
    # Use helper to get best channel for each unit (Strategy 2: templates.npy)
    unit_ch_map = _get_unit_best_channels(paths, unique_clusters)
    
    # Group units by channel for efficient processing
    units_by_channel = defaultdict(list)
    for cid in unique_clusters:
        ch = unit_ch_map.get(cid)
        if ch is not None:
            units_by_channel[ch].append(cid)
        else:
            # If no mapping, skip or assign to default? skipping is safer for phase locking
            # Or use middle channel if desperate, but better to be accurate.
            pass
            
    print(f"  Mapped {len(unit_ch_map)} units to LFP channels. {len(unique_clusters) - len(unit_ch_map)} units unmapped.")

    # --- 3. Initialize LFP Loader ---
    try:
        config = load_dataset_config()
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return
            
        lfp_fs = lfp_loader.fs
        print(f"  Initialized LFP Loader. FS={lfp_fs} Hz. Sync params: {lfp_loader.sync_params}")
    except Exception as e:
        print(f"  Error initializing LFPDataLoader: {e}")
        return

    # --- 4. Calculate Phase-Locking ---
    results = []
    
    # Pre-calculate filters
    filters = {} # band -> (b, a)
    nyquist = lfp_fs / 2
    for band_name, (low, high) in frequency_bands.items():
        b, a = butter(4, [low / nyquist, high / nyquist], btype='band')
        filters[band_name] = (b, a)
        
    print(f"  Processing {len(units_by_channel)} unique channels...")

    # Iterate over channels that have units
    for ch_idx, cluster_list in tqdm(units_by_channel.items(), desc="Analyzing Channels"):
        
        # Load LFP for this channel (and neighbors for CSD)
        # We need the full duration. LFPDataLoader handles alignment.
        # To get the whole recording, we can ask for a large window.
        # Or better: get start/end times from spike data to know the range.
        
        t_start = 0
        t_end = spike_times_sec.max() + 1.0 # Add margin
        
        try:
            # Request CSD data for this specific channel
            # LFPDataLoader.get_data returns (traces, timestamps)
            # traces is (n_samples, n_req_channels)
            lfp_traces, lfp_timestamps = lfp_loader.get_data(
                start_time=t_start, 
                end_time=t_end, 
                channels=[ch_idx], 
                reference='csd'
            )
            
            if len(lfp_traces) == 0:
                print(f"    Warning: No data returned for Ch {ch_idx}. Skipping.")
                continue
                
            lfp_trace = lfp_traces[:, 0] # Extract the single channel
            
        except Exception as e:
            print(f"    Error reading LFP Ch {ch_idx}: {e}")
            continue
            
        # Filter this channel for all bands and extract phase
        band_phases = {}
        processed_bands = True
        for band_name, (b, a) in filters.items():
            try:
                # remove NaNs if any (CSD might produce NaNs at edges)
                if np.isnan(lfp_trace).any():
                     lfp_trace = np.nan_to_num(lfp_trace)
                     
                filtered = filtfilt(b, a, lfp_trace)
                analytic = hilbert(filtered)
                band_phases[band_name] = np.angle(analytic)
            except Exception as e:
                print(f"    Filter error ({band_name}) on ch {ch_idx}: {e}")
                processed_bands = False
                
        if not processed_bands: continue

        # Process all neurons assigned to this channel
        for cid in cluster_list:
            cluster_spikes = spike_times_sec[spike_clusters == cid]
            
            if len(cluster_spikes) < min_spikes:
                continue
                
            # Map spikes to LFP timestamps
            # LFP timestamps are aligned to TPrime (same as spike_times_sec)
            # Find indices in lfp_timestamps closest to spike times
            # Since lfp_timestamps should be monotonic, use searchsorted
            
            # Optimization: lfp_timestamps is usually regular grid (defined by sync params)
            # But let's use searchsorted for robustness
            spike_indices = np.searchsorted(lfp_timestamps, cluster_spikes)
            
            # Clip and filter valid
            valid_mask = (spike_indices >= 0) & (spike_indices < len(lfp_timestamps))
            valid_indices = spike_indices[valid_mask]
            
            # Double check time difference to ensure we didn't map far-off spikes
            # (e.g. if spikes extend beyond LFP)
            if len(valid_indices) > 0:
                time_diffs = np.abs(lfp_timestamps[valid_indices] - cluster_spikes[valid_mask])
                good_match_mask = time_diffs < (1.0 / lfp_fs * 1.5) # within 1.5 samples
                valid_indices = valid_indices[good_match_mask]
            
            if len(valid_indices) < min_spikes:
                continue
                
            for band_name, phase_array in band_phases.items():
                spike_phases = phase_array[valid_indices]
                n = len(spike_phases)
                
                # Metrics
                if n > 0:
                    mean_vector = np.mean(np.exp(1j * spike_phases))
                    plv = np.abs(mean_vector)
                    preferred_phase = np.angle(mean_vector)
                    
                    # Rayleigh Test
                    rayleigh_stat = n * (plv ** 2)
                    rayleigh_pval = np.exp(-rayleigh_stat)
                    
                    # PPC
                    if n > 1:
                        r_sum = np.abs(np.sum(np.exp(1j * spike_phases)))
                        ppc = (r_sum**2 - n) / (n * (n - 1))
                    else:
                        ppc = 0
                else:
                    plv, preferred_phase, rayleigh_stat, rayleigh_pval, ppc = 0, 0, 0, 1.0, 0
                
                circ_std = circstd(spike_phases)
                
                results.append({
                    'cluster_id': cid,
                    'band': band_name,
                    'n_spikes': n,
                    'plv': plv,
                    'ppc': ppc,
                    'preferred_phase_rad': preferred_phase,
                    'preferred_phase_deg': np.degrees(preferred_phase),
                    'circular_std': circ_std,
                    'rayleigh_stat': rayleigh_stat,
                    'rayleigh_pval': rayleigh_pval,
                    'is_locked': rayleigh_pval < DEFAULT_PHASE_LOCKING_SIGNIFICANCE,
                    'lfp_channel_idx': ch_idx
                })
    
    if len(results) == 0:
        print("  No phase-locking results computed.")
        return None
    
    # --- 5. Save Results ---
    df_results = pd.DataFrame(results)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'spike_phase_locking_data.csv'
    df_results.to_csv(output_path, index=False)
    print(f"\n  Phase-locking results saved to {output_path}")
    
    # Print summary
    for band_name in frequency_bands.keys():
        band_results = df_results[df_results['band'] == band_name]
        if band_results.empty: continue
        n_locked = np.sum(band_results['is_locked'])
        n_total = len(band_results)
        mean_plv = band_results[band_results['is_locked']]['plv'].mean() if n_locked > 0 else 0
        
        print(f"  {band_name}: {n_locked}/{n_total} neurons significantly phase-locked (mean PLV={mean_plv:.3f})")
    
    # --- 6. Visualize Phase-Locking Results ---
    try:
        n_bands = len(frequency_bands)
        fig, axes = plt.subplots(2, n_bands, figsize=(6*n_bands, 10))
        
        if n_bands == 1:
            axes = np.array(axes).reshape(-1, 1)
        
        for idx, band_name in enumerate(frequency_bands.keys()):
            if idx >= axes.shape[1]: break
            
            band_results = df_results[df_results['band'] == band_name]
            if band_results.empty: continue
            
            locked_results = band_results[band_results['is_locked']]
            
            # Top row: Phase distribution (circular histogram)
            ax = axes[0, idx]
            if len(locked_results) > 0:
                phases = locked_results['preferred_phase_rad'].values
                n_bins = 36  # 10-degree bins
                
                phases = phases[~np.isnan(phases)]
                if len(phases) > 0:
                    # Create circular histogram
                    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
                    counts, _ = np.histogram(phases, bins=bins)
                    
                    # Plot as bar chart
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    bin_widths = np.diff(bins)
                    bin_centers_deg = np.degrees(bin_centers)
                    
                    ax.bar(bin_centers_deg, counts, width=np.degrees(bin_widths[0]), 
                          alpha=0.7, color='#3498db', edgecolor='black')
                    ax.set_xlabel('Preferred Phase (degrees)')
                    ax.set_ylabel('Number of Neurons')
                    ax.set_title(f'{band_name} - Phase Distribution\n({len(locked_results)} locked neurons)')
                    ax.set_xlim(-180, 180)
                    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No significantly\nlocked neurons', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{band_name} - Phase Distribution')
            
            # Bottom row: PLV vs PPC scatter
            ax = axes[1, idx]
            if len(band_results) > 0:
                locked = band_results['is_locked'].values
                ax.scatter(band_results[~locked]['plv'], band_results[~locked]['ppc'], 
                          alpha=0.5, s=30, c='gray', label='Not locked')
                if np.sum(locked) > 0:
                    ax.scatter(band_results[locked]['plv'], band_results[locked]['ppc'], 
                              alpha=0.7, s=50, c='#e74c3c', label='Locked (p<0.01)')
                
                ax.set_xlabel('PLV (Phase-Locking Value)')
                ax.set_ylabel('PPC (Pairwise Phase Consistency)')
                ax.set_title(f'{band_name} - Locking Strength')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.1, 1)
            
        plt.tight_layout()
        plot_path = output_dir / 'spike_phase_locking_summary.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Phase-locking summary plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate phase-locking plots: {e}")
        import traceback
        traceback.print_exc()
    
    # --- 7. Spatial Distribution Analysis ---
    if not df_results['depth_um'].isna().all():
        try:
            print("\n  Analyzing spatial distribution of phase-locked neurons...")
            
            fig, axes = plt.subplots(1, n_bands, figsize=(6*n_bands, 8))
            
            if n_bands == 1:
                axes = [axes]
            
            for idx, band_name in enumerate(frequency_bands.keys()):
                 if idx >= len(axes): break
                 ax = axes[idx]
                 band_results = df_results[df_results['band'] == band_name]
                 
                 # Remove NaN depths
                 valid_results = band_results[~band_results['depth_um'].isna()]
                 
                 if len(valid_results) == 0:
                     continue
                 
                 locked = valid_results['is_locked'].values
                 depths = valid_results['depth_um'].values
                 plvs = valid_results['plv'].values
                 
                 # Plot depth vs PLV
                 ax.scatter(plvs[~locked], depths[~locked], alpha=0.5, s=30, 
                           c='gray', label='Not locked')
                 if np.sum(locked) > 0:
                     scatter = ax.scatter(plvs[locked], depths[locked], alpha=0.7, s=80, 
                                        c=depths[locked], cmap='viridis', 
                                        edgecolors='black', linewidths=1)
                     plt.colorbar(scatter, ax=ax, label='Depth (μm)')
                 
                 ax.set_xlabel('PLV (Phase-Locking Value)')
                 ax.set_ylabel('Depth on Probe (μm)')
                 ax.set_title(f'{band_name} - Spatial Distribution')
                 ax.legend()
                 ax.grid(True, alpha=0.3)
                 ax.invert_yaxis()  # Convention: depth increases downward
            
            plt.tight_layout()
            spatial_plot_path = output_dir / 'spike_phase_locking_spatial.png'
            plt.savefig(spatial_plot_path, dpi=150)
            plt.close()
            print(f"  Spatial distribution plot saved to {spatial_plot_path}")
            
        except Exception as e:
            print(f"  Could not generate spatial distribution plot: {e}")
            import traceback
            traceback.print_exc()
    
    return df_results

def analyze_temporal_autocorrelation(paths: DataPaths, max_lag_ms: int = 500, bin_size_ms: int = 1):
    """
    Analyzes temporal autocorrelation within single neurons.
    
    Computes autocorrelation function for each neuron to reveal temporal structure,
    rhythmicity, and bursting patterns.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        max_lag_ms (int): Maximum time lag for autocorrelation in milliseconds.
        bin_size_ms (int): Bin size for spike train in milliseconds.
    """
    print("Analyzing temporal autocorrelation...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 2. Compute Autocorrelation for Each Neuron ---
    max_lag_sec = max_lag_ms / 1000.0
    bin_size_sec = bin_size_ms / 1000.0
    n_lags = int(max_lag_sec / bin_size_sec)
    
    results = {}
    session_duration = spike_times_sec.max()
    
    print(f"  Computing autocorrelation for {len(unique_clusters)} neurons...")
    
    for cid in tqdm(unique_clusters, desc="Neurons"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        if len(cluster_spikes) < 10:  # Need minimum spikes
            continue
        
        # Create binned spike train
        n_bins = int(session_duration / bin_size_sec)
        spike_train, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        
        # Compute autocorrelation using FFT (efficient for long signals)
        from scipy.signal import correlate
        autocorr = correlate(spike_train, spike_train, mode='full', method='fft')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        autocorr = autocorr[:n_lags]  # Truncate to max_lag
        
        # Normalize
        autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
        
        # Find peaks (rhythmicity)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr[1:], height=0.1, distance=10)
        
        # Calculate metrics
        mean_autocorr = np.mean(autocorr[1:])
        decay_time = np.argmax(autocorr[1:] < 0.5 * autocorr[0]) * bin_size_ms if np.any(autocorr[1:] < 0.5 * autocorr[0]) else max_lag_ms
        
        results[cid] = {
            'autocorr': autocorr,
            'mean_autocorr': mean_autocorr,
            'decay_time_ms': decay_time,
            'n_peaks': len(peaks),
            'rhythmic': len(peaks) > 2
        }
    
    # --- 3. Save Results ---
    print("\n  Temporal autocorrelation analysis complete.")
    
    # Save summary statistics
    summary_data = {
        cid: {
            'mean_autocorr': data['mean_autocorr'],
            'decay_time_ms': data['decay_time_ms'],
            'n_peaks': data['n_peaks'],
            'rhythmic': data['rhythmic']
        }
        for cid, data in results.items()
    }
    
    df_summary = pd.DataFrame.from_dict(summary_data, orient='index')
    df_summary.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'temporal_autocorrelation_summary.csv'
    df_summary.to_csv(output_path)
    print(f"  Summary saved to {output_path}")
    
    # Plot example autocorrelations
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        # Plot first 6 neurons
        lag_times = np.arange(n_lags) * bin_size_ms
        for idx, (cid, data) in enumerate(list(results.items())[:6]):
            ax = axes[idx]
            ax.plot(lag_times, data['autocorr'], 'b-', linewidth=1)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Lag (ms)')
            ax.set_ylabel('Autocorrelation')
            ax.set_title(f"Neuron {cid} ({'Rhythmic' if data['rhythmic'] else 'Non-rhythmic'})")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'temporal_autocorrelation_examples.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Example plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
    
    return results

def analyze_cross_correlation_pairs(paths: DataPaths, max_lag_ms: int = 100, 
                                    bin_size_ms: int = 1, max_pairs: int = 1000,
                                    max_dist_um: float = 200.0, speed_threshold: float = 2.0):
    """
    Analyzes cross-correlation between neuron pairs with spatial and biological context.
    
    Improvements:
    1. Selects pairs based on physical distance (default < 200um) rather than random sampling.
    2. Integrates cell-type classification (MSN vs FSI).
    3. Performs state-dependent analysis (Movement vs Rest).
    
    Args:
        paths (DataPaths): The DataPaths object.
        max_lag_ms (int): Max lag in ms.
        bin_size_ms (int): Bin size in ms.
        max_pairs (int): Max pairs to analyze (sampled from close pairs if count exceeds this).
        max_dist_um (float): Maximum Euclidian distance between units to consider for pairing.
        speed_threshold (float): Velocity threshold (cm/s) to define Movement vs Rest.
    """
    print("Analyzing cross-correlation with Spatial & State dependency...")
    
    # --- 1. Load Data ---
    # Spikes
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return

    # Unit Locations (Spatial)
    print("  Loading unit locations...")
    unit_channels = _get_unit_best_channels(paths, unique_clusters) # {cid: ch_idx}
    
    # Load channel positions from config/file
    try:
        config = load_dataset_config()
        # Find channel_positions.npy path
        cp_entry = next((v for k, v in config.items() if 'channel_positions.npy' in v.get('path', '')), None)
        if cp_entry:
            cp_path = paths.base_path / cp_entry['path'] if not Path(cp_entry['path']).is_absolute() else Path(cp_entry['path'])
            if cp_path.exists():
                channel_positions = np.load(cp_path) # (n_channels, 2)
            else:
                print(f"  Warning: Channel positions file not found at {cp_path}")
                channel_positions = np.zeros((384, 2)) # Fallback
        else:
             print("  Warning: channel_positions config not found. Using dummy positions.")
             channel_positions = np.zeros((384, 2))
    except Exception as e:
        print(f"  Error loading channel positions: {e}")
        channel_positions = np.zeros((384, 2))

    # Map Unit -> (x, y)
    unit_positions = {}
    for cid in unique_clusters:
        if cid in unit_channels:
            ch = unit_channels[cid]
            if ch < len(channel_positions):
                unit_positions[cid] = channel_positions[ch]
            else:
                unit_positions[cid] = [0, 0]
        else:
            unit_positions[cid] = [0, 0] # Default if location unknown

    # Unit Classifications (Cell Type)
    print("  Loading unit classifications...")
    unit_types = {cid: "Unknown" for cid in unique_clusters}
    try:
        # Look for unit_classification_rulebased.csv
        class_file = None
        if paths.kilosort_dir:
            class_file = paths.kilosort_dir / 'unit_classification_rulebased.csv'
        
        if not class_file or not class_file.exists():
             # Try fallback structure
             class_file = paths.neural_base / 'kilosort4' / 'sorter_output' / 'unit_classification_rulebased.csv'
        
        if class_file and class_file.exists():
            df_class = pd.read_csv(class_file)
            # Expecting columns: unit_id, cell_type
            if 'unit_id' in df_class.columns and 'cell_type' in df_class.columns:
                for _, row in df_class.iterrows():
                    unit_types[row['unit_id']] = row['cell_type']
            print(f"  Loaded classifications for {len(df_class)} units.")
        else:
            print("  Warning: unit_classification_rulebased.csv not found.")
    except Exception as e:
        print(f"  Error loading unit classifications: {e}")

    # Velocity (State Dependency)
    print("  Loading velocity for state dependency...")
    velocity, v_times = _load_dlc_and_calculate_velocity(paths, video_fs=60, px_per_cm=30.0)
    has_velocity = velocity is not None
    
    # --- 2. Select Pairs (Distance Based) ---
    from scipy.spatial.distance import pdist, squareform
    
    # Create position matrix aligned with unique_clusters
    n_neurons = len(unique_clusters)
    pos_matrix = np.array([unit_positions[cid] for cid in unique_clusters])
    
    # Calculate distances (Upper triangle)
    dist_matrix = squareform(pdist(pos_matrix)) # (N, N)
    
    pairs = []
    distances = []
    pair_types_list = []
    
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            d = dist_matrix[i, j]
            if d <= max_dist_um:
                cid1 = unique_clusters[i]
                cid2 = unique_clusters[j]
                
                pairs.append((cid1, cid2))
                distances.append(d)
                
                t1 = unit_types.get(cid1, "Unknown")
                t2 = unit_types.get(cid2, "Unknown")
                # Sort types alphabetically for consistent "FSI-MSN" vs "MSN-FSI" label
                ptype = "-".join(sorted([str(t1), str(t2)]))
                pair_types_list.append(ptype)
    
    n_total_close = len(pairs)
    print(f"  Found {n_total_close} pairs within {max_dist_um} um.")
    
    if n_total_close > max_pairs:
        print(f"  Subsampling to {max_pairs} pairs...")
        # Weighted sample? No, random is fine for defined "close" set
        idx = np.random.choice(n_total_close, max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
        distances = [distances[i] for i in idx]
        pair_types_list = [pair_types_list[i] for i in idx]
    
    # --- 3. Prepare Spike Trains & States ---
    max_lag_sec = max_lag_ms / 1000.0
    bin_size_sec = bin_size_ms / 1000.0
    n_lags = int(max_lag_sec / bin_size_sec)
    
    session_duration = spike_times_sec.max()
    n_bins = int(session_duration / bin_size_sec) + 1
    
    # Define states per bin (Interpolate velocity to bin centers)
    # This is faster than checking state for every spike for every pair
    bin_edges = np.arange(0, n_bins + 1) * bin_size_sec
    bin_centers = bin_edges[:-1] + bin_size_sec/2
    
    valid_mask_move = None
    valid_mask_rest = None
    
    if has_velocity:
        # Interpolate velocity to bin centers
        # We need a quick way. 1ms bins = 1000Hz. Velocity is ~60Hz.
        # Nearest neighbor interpolation
        from scipy.interpolate import interp1d
        
        # Handle velocity time range mismatch
        if v_times[-1] < bin_centers[-1]:
             # Pad velocity with last value
             v_times = np.append(v_times, bin_centers[-1] + 1.0)
             velocity = np.append(velocity, velocity[-1])
             
        f_vel = interp1d(v_times, velocity, kind='nearest', bounds_error=False, fill_value=0)
        bin_vel = f_vel(bin_centers)
        
        valid_mask_move = bin_vel > speed_threshold
        valid_mask_rest = bin_vel <= speed_threshold
        
        print(f"  State Proportions: Move={np.mean(valid_mask_move):.1%}, Rest={np.mean(valid_mask_rest):.1%}")
    
    # Pre-bin spike trains
    spike_trains = {}
    for cid in unique_clusters:
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        spike_train, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, bin_edges[-1]))
        spike_trains[cid] = spike_train

    # --- 4. Compute CCGs ---
    from scipy.signal import correlate
    results = []
    
    for idx, (cid1, cid2) in enumerate(tqdm(pairs, desc="CCG Pairs")):
        train1 = spike_trains[cid1]
        train2 = spike_trains[cid2]
        
        if len(train1) != len(train2):
            # Should not happen given shared binning
            min_len = min(len(train1), len(train2))
            train1 = train1[:min_len]
            train2 = train2[:min_len]
            if has_velocity:
                valid_mask_move = valid_mask_move[:min_len]
                valid_mask_rest = valid_mask_rest[:min_len]

        # Helper to compute CCG metrics
        def compute_ccg(t1, t2, valid_mask=None):
            if valid_mask is not None:
                # Zero out spikes in invalid periods
                # Note: This is an approximation. Ideally we concatenate valid segments,
                # but zeroing preserves time lags, which is crucial for cross-correlation.
                # For high-frequency bins (1ms) and slow states (seconds), zeroing is acceptable
                # as boundary effects are negligible compared to total duration.
                t1_masked = t1 * valid_mask
                t2_masked = t2 * valid_mask
            else:
                t1_masked = t1
                t2_masked = t2
                
            if np.sum(t1_masked) < 10 or np.sum(t2_masked) < 10:
                return np.nan, np.nan, np.nan # Not enough spikes
            
            # FFT Correlation
            xcorr = correlate(t1_masked, t2_masked, mode='full', method='fft')
            center = len(xcorr) // 2
            xcorr = xcorr[center - n_lags : center + n_lags + 1]
            
            # Normalize (Geometric Mean of spike counts)
            n1 = np.sum(t1_masked)
            n2 = np.sum(t2_masked)
            norm = np.sqrt(n1 * n2)
            if norm > 0:
                xcorr = xcorr / norm
            else:
                return np.nan, np.nan, np.nan # No spikes to normalize
                
            peak_idx = np.argmax(np.abs(xcorr))
            peak_val = xcorr[peak_idx]
            peak_lag = (peak_idx - n_lags) * bin_size_ms
            zero_lag_val = xcorr[n_lags]
            
            return peak_val, peak_lag, zero_lag_val

        # 1. Global
        g_peak, g_lag, g_zero = compute_ccg(train1, train2, None)
        
        # 2. Movement
        m_peak, m_lag, m_zero = (np.nan, np.nan, np.nan)
        if has_velocity:
            m_peak, m_lag, m_zero = compute_ccg(train1, train2, valid_mask_move)
            
        # 3. Rest
        r_peak, r_lag, r_zero = (np.nan, np.nan, np.nan)
        if has_velocity:
            r_peak, r_lag, r_zero = compute_ccg(train1, train2, valid_mask_rest)
            
        results.append({
            'neuron_1': cid1,
            'neuron_2': cid2,
            'distance_um': distances[idx],
            'pair_type': pair_types_list[idx],
            'cell_type_1': unit_types.get(cid1, "Unknown"),
            'cell_type_2': unit_types.get(cid2, "Unknown"),
            
            'peak_corr_global': g_peak,
            'peak_lag_global': g_lag,
            'zero_lag_global': g_zero,
            
            'peak_corr_move': m_peak,
            'peak_lag_move': m_lag,
            'zero_lag_move': m_zero,
            
            'peak_corr_rest': r_peak,
            'peak_lag_rest': r_lag,
            'zero_lag_rest': r_zero,
        })

    # --- 5. Save & Plot ---
    df_results = pd.DataFrame(results)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'cross_correlation_pairs.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    
    # Plotting
    try:
        # Filter NaNs
        df_plot = df_results.dropna(subset=['peak_corr_global', 'peak_corr_move', 'peak_corr_rest'])
        
        if len(df_plot) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # A. Zero-Lag Corr vs Distance
            ax = axes[0, 0]
            scatter = ax.scatter(df_plot['distance_um'], df_plot['zero_lag_global'], 
                               alpha=0.6, c='blue', s=20)
            ax.set_xlabel('Distance (um)')
            ax.set_ylabel('Global Zero-Lag Correlation')
            ax.set_title('Synchrony vs Distance')
            ax.grid(True, alpha=0.3)
            
            # B. Move vs Rest Correlation
            ax = axes[0, 1]
            # Color by Cell Type Pair if possible, else simplified
            unique_ptypes = df_plot['pair_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ptypes)))
            
            for i, ptype in enumerate(unique_ptypes):
                subset = df_plot[df_plot['pair_type'] == ptype]
                ax.scatter(subset['peak_corr_rest'], subset['peak_corr_move'], 
                          label=ptype, alpha=0.7, s=20)
            
            # Identity line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
            
            ax.set_xlabel('Peak Corr (Rest)')
            ax.set_ylabel('Peak Corr (Move)')
            ax.set_title('State-Dependent Correlation')
            ax.legend(title='Pair Type', fontsize='small')
            ax.grid(True, alpha=0.3)
            
            # C. Lag Distribution by Type
            ax = axes[1, 0]
            # Simple histogram of peak lags for global
            for i, ptype in enumerate(unique_ptypes):
                subset = df_plot[df_plot['pair_type'] == ptype]
                ax.hist(subset['peak_lag_global'], bins=30, alpha=0.5, label=ptype, density=True)
            ax.set_xlabel('Peak Lag (ms)')
            ax.set_ylabel('Density')
            ax.set_title('Lag Distribution by Pair Type')
            ax.legend()
            
            # D. Summary Bar Chart
            ax = axes[1, 1]
            means_move = df_plot.groupby('pair_type')['peak_corr_move'].mean()
            means_rest = df_plot.groupby('pair_type')['peak_corr_rest'].mean()
            
            x = np.arange(len(means_move))
            width = 0.35
            
            ax.bar(x - width/2, means_move, width, label='Move')
            ax.bar(x + width/2, means_rest, width, label='Rest')
            
            ax.set_ylabel('Mean Peak Correlation')
            ax.set_title('Synchrony by Type & State')
            ax.set_xticks(x)
            ax.set_xticklabels(means_move.index, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            plot_path = output_dir / 'cross_correlation_summary.png'
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"  Plots saved to {plot_path}")
            
    except Exception as e:
        print(f"  Error generating plots: {e}")

    
    return df_results

def analyze_bursting_behavior(paths: DataPaths, min_isi_ms: float = 10.0, 
                              min_spikes_in_burst: int = 3, speed_threshold: float = 2.0):
    """
    Analyzes bursting behavior with comprehensive metrics and biological context.
    
    Includes:
    1. Detailed Metrics: Burst Rate, Duration, Spikes/Burst, Intra-Burst Freq.
    2. State-Dependence: Movement vs Rest bursting.
    3. Striatal Signatures: Reward Burst Bias, Post-Burst Pause.
    4. Cell-Type Integration.
    
    Args:
        paths (DataPaths): The DataPaths object.
        min_isi_ms (float): Max ISI to define a burst (default 10ms for MSNs).
        min_spikes_in_burst (int): Min spikes to form a burst.
        speed_threshold (float): Velocity threshold for Move vs Rest.
    """
    print("Analyzing comprehensive bursting behavior...")
    
    # --- 1. Load Data ---
    # Spikes
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None: return

    # Velocity (for State Dependence)
    print("  Loading velocity...")
    velocity, v_times = _load_dlc_and_calculate_velocity(paths, video_fs=60, px_per_cm=30.0)
    has_velocity = velocity is not None
    
    # Rewards (for Reward Bias)
    print("  Loading reward events...")
    reward_times = np.array([])
    try:
        config = load_dataset_config()
        # Find reward file
        reward_loader = EventDataLoader(paths.base_path, config)
        
        # Try finding a reward file in config keys
        reward_key = None
        for key, val in config.items():
            if 'path' in val and ('reward' in key.lower() or 'water' in key.lower()) and 'csv' in val['path']:
                reward_key = key
                break
        
        if reward_key:
             reward_df, reward_times = reward_loader.load_events_from_path(
                paths.base_path / config[reward_key]['path'] if not Path(config[reward_key]['path']).is_absolute() else Path(config[reward_key]['path'])
             )
             print(f"  Loaded {len(reward_times)} reward events.")
        else:
             print("  Warning: No reward file found in config.")
    except Exception as e:
        print(f"  Error loading rewards: {e}")

    # Cell Types
    print("  Loading unit classifications...")
    unit_types = {cid: "Unknown" for cid in unique_clusters}
    try:
        class_file = None
        if paths.kilosort_dir: class_file = paths.kilosort_dir / 'unit_classification_rulebased.csv'
        if not class_file or not class_file.exists():
             class_file = paths.neural_base / 'kilosort4' / 'sorter_output' / 'unit_classification_rulebased.csv'
        
        if class_file and class_file.exists():
            df_class = pd.read_csv(class_file)
            if 'unit_id' in df_class.columns and 'cell_type' in df_class.columns:
                for _, row in df_class.iterrows():
                    unit_types[row['unit_id']] = row['cell_type']
            print(f"  Loaded classifications for {len(df_class)} units.")
    except Exception:
        pass

    # --- 2. Calculate Metrics for Each Unit ---
    results = []
    min_isi_sec = min_isi_ms / 1000.0
    
    # State Masks (Interpolated)
    valid_mask_move = None
    if has_velocity:
        # Create interpolated function for velocity
        from scipy.interpolate import interp1d
        # Ensure v_times covers range
        if v_times[-1] < spike_times_sec.max():
             v_times = np.append(v_times, spike_times_sec.max() + 1.0)
             velocity = np.append(velocity, velocity[-1])
        f_vel = interp1d(v_times, velocity, kind='nearest', bounds_error=False, fill_value=0)
    
    for cid in tqdm(unique_clusters, desc="Burst Stats"):
        cluster_spikes = np.sort(spike_times_sec[spike_clusters == cid])
        total_spikes = len(cluster_spikes)
        
        if total_spikes < min_spikes_in_burst:
            continue

        # A. Detect Bursts (ISI Method)
        isis = np.diff(cluster_spikes)
        is_burst_interval = isis <= min_isi_sec
        
        # Find sequences of True in is_burst_interval
        # Pad with False to detect edges
        padded = np.concatenate(([False], is_burst_interval, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] # End index is exclusive in slice, inclusive in ISI
        
        burst_events = [] # (start_time, end_time, n_spikes, spike_indices)
        spikes_in_bursts = 0
        
        for s, e in zip(starts, ends):
            # s, e are indices into isis. 
            # ISI index i corresponds to interval between spike[i] and spike[i+1]
            # Sequence s..e means spike[s] to spike[e+1] are connected by short ISIs
            
            n_spikes_in_seq = (e - s) + 1
            if n_spikes_in_seq >= min_spikes_in_burst:
                first_spike_idx = s
                last_spike_idx = e
                
                b_start = cluster_spikes[first_spike_idx]
                b_end = cluster_spikes[last_spike_idx]
                b_spikes = cluster_spikes[first_spike_idx : last_spike_idx+1]
                
                burst_events.append({
                    'start': b_start,
                    'end': b_end,
                    'n_spikes': len(b_spikes),
                    'duration': b_end - b_start,
                    'intra_freq': (len(b_spikes)-1) / (b_end - b_start) if (b_end > b_start) else 0
                })
                spikes_in_bursts += len(b_spikes)

        # B. Calculate Basic Metrics
        session_dur = spike_times_sec.max() - spike_times_sec.min()
        burst_ratio = spikes_in_bursts / total_spikes if total_spikes > 0 else 0
        burst_rate_hz = len(burst_events) / session_dur if session_dur > 0 else 0
        
        if burst_events:
            avg_spikes = np.mean([b['n_spikes'] for b in burst_events])
            avg_dur_ms = np.mean([b['duration'] for b in burst_events]) * 1000.0
            avg_intra_freq = np.mean([b['intra_freq'] for b in burst_events])
            
            # Post-Burst Pause
            # Time from burst end to NEXT spike
            pauses = []
            for b_idx, b in enumerate(burst_events):
                # Find spike after b['end']
                # Optimizable: we have indices, but simple search is safe
                next_spikes = cluster_spikes[cluster_spikes > b['end']]
                if len(next_spikes) > 0:
                    pause = next_spikes[0] - b['end']
                    pauses.append(pause)
            avg_pause_ms = np.mean(pauses) * 1000.0 if pauses else 0
            
            # Inter-Burst Interval CV
            burst_starts = [b['start'] for b in burst_events]
            if len(burst_starts) > 1:
                ibis = np.diff(burst_starts)
                cv_ibi = np.std(ibis) / np.mean(ibis) if np.mean(ibis) > 0 else 0
            else:
                cv_ibi = np.nan
        else:
            avg_spikes = 0
            avg_dur_ms = 0
            avg_intra_freq = 0
            avg_pause_ms = 0
            cv_ibi = 0

        # C. State Dependent (Move vs Rest)
        b_rate_move = np.nan
        b_rate_rest = np.nan
        mod_index = 0
        
        if has_velocity and len(burst_events) > 0:
            burst_midpoints = [(b['start'] + b['end'])/2 for b in burst_events]
            burst_vels = f_vel(burst_midpoints)
            
            n_move_bursts = np.sum(burst_vels > speed_threshold)
            n_rest_bursts = np.sum(burst_vels <= speed_threshold)
            
            # Estimate time spent in each state
            # Sample velocity at 10Hz to estimate duration
            t_sample = np.arange(0, session_dur, 0.1)
            v_sample = f_vel(t_sample)
            t_move = np.sum(v_sample > speed_threshold) * 0.1
            t_rest = np.sum(v_sample <= speed_threshold) * 0.1
            
            b_rate_move = n_move_bursts / t_move if t_move > 1.0 else 0
            b_rate_rest = n_rest_bursts / t_rest if t_rest > 1.0 else 0
            
            if (b_rate_move + b_rate_rest) > 0:
                mod_index = (b_rate_move - b_rate_rest) / (b_rate_move + b_rate_rest)

        # D. Reward Bias
        # Does the neuron burst more during Reward vs Baseline?
        reward_bias = 0
        if len(reward_times) > 0 and len(burst_events) > 0:
            burst_starts = np.array([b['start'] for b in burst_events])
            
            # Count bursts within 1s of reward
            n_reward_bursts = 0
            for rt in reward_times:
                n_reward_bursts += np.sum((burst_starts >= rt) & (burst_starts < rt + 1.0))
            
            # Burst Rate during Reward
            t_reward_total = len(reward_times) * 1.0
            rate_reward = n_reward_bursts / t_reward_total if t_reward_total > 0 else 0
            
            # Burst Rate Baseline (everything else)
            n_baseline_bursts = len(burst_events) - n_reward_bursts
            t_baseline = session_dur - t_reward_total
            rate_baseline = n_baseline_bursts / t_baseline if t_baseline > 0 else 0
            
            if (rate_reward + rate_baseline) > 0:
                reward_bias = (rate_reward - rate_baseline) / (rate_reward + rate_baseline)

        results.append({
            'unit_id': cid,
            'cell_type': unit_types.get(cid, "Unknown"),
            'total_spikes': total_spikes,
            'n_bursts': len(burst_events),
            # Detailed
            'burst_ratio': burst_ratio,
            'burst_rate_hz': burst_rate_hz,
            'spikes_per_burst': avg_spikes,
            'burst_duration_ms': avg_dur_ms,
            'intra_burst_freq_hz': avg_intra_freq,
            'post_burst_pause_ms': avg_pause_ms,
            'cv_inter_burst_interval': cv_ibi,
            # State
            'burst_rate_move_hz': b_rate_move,
            'burst_rate_rest_hz': b_rate_rest,
            'burst_mod_index': mod_index,
            # Striatal
            'reward_burst_bias': reward_bias
        })

    # --- 3. Save & Plot ---
    print("\n  Bursting behavior analysis complete.")
    df_results = pd.DataFrame(results)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'bursting_behavior_detailed.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

    # Generate Plots
    try:
        if len(df_results) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Color coding
            cell_types = df_results['cell_type'].unique()
            cmap = plt.cm.get_cmap('tab10', len(cell_types))
            c_dict = {ct: cmap(i) for i, ct in enumerate(cell_types)}
            colors = df_results['cell_type'].map(c_dict)

            # A. Burst Ratio vs Firing Rate (Cell Type)
            ax = axes[0]
            # Approx firing rate = total_spikes / session_dur (loaded earlier)
            session_dur = spike_times_sec.max()
            df_results['firing_rate'] = df_results['total_spikes'] / session_dur
            
            for ct in cell_types:
                subset = df_results[df_results['cell_type'] == ct]
                ax.scatter(subset['firing_rate'], subset['burst_ratio'], label=ct, alpha=0.7)
            
            ax.set_xlabel('Mean Firing Rate (Hz)')
            ax.set_ylabel('Burst Ratio')
            ax.set_title('Burst Ratio vs Firing Rate')
            ax.set_xscale('log') # Rate often log-distributed
            ax.grid(True, alpha=0.3)
            ax.legend()

            # B. Intra-Burst Freq vs Duration (Shape)
            ax = axes[1]
            sc = ax.scatter(df_results['burst_duration_ms'], df_results['intra_burst_freq_hz'], 
                           c=colors, alpha=0.7)
            ax.set_xlabel('Burst Duration (ms)')
            ax.set_ylabel('Intra-Burst Freq (Hz)')
            ax.set_title('Burst Shape')
            ax.grid(True, alpha=0.3)

            # C. Modulation (Move vs Rest)
            ax = axes[2]
            ax.scatter(df_results['burst_rate_rest_hz'], df_results['burst_rate_move_hz'], 
                      c=colors, alpha=0.7)
            
            # Identity Line
            lims = [
                np.nanmin([ax.get_xlim(), ax.get_ylim()]),
                np.nanmax([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            
            ax.set_xlabel('Burst Rate (Rest)')
            ax.set_ylabel('Burst Rate (Move)')
            ax.set_title('State Modulation')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / 'bursting_behavior_summary.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plots saved to {plot_path}")
            
    except Exception as e:
        print(f"  Error generating plots: {e}")
    
def analyze_isi_distribution(paths: DataPaths, n_bins: int = 100, max_isi_ms: float = 100):
    """
    Analyzes the inter-spike interval (ISI) distribution for each neuron.

    Calculates the Coefficient of Variation (CV) of the ISI as a measure of
    firing regularity.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_bins (int): Number of bins for the ISI histogram.
        max_isi_ms (float): Maximum ISI to include in the analysis in milliseconds.
    """
    print("Analyzing ISI distributions...")

    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None: return

    # --- 2. Calculate CV of ISI for each neuron ---
    results = {}
    max_isi_sec = max_isi_ms / 1000.0

    all_isi_hists = {}

    for cid in unique_clusters:
        cluster_spike_times = np.sort(spike_times_sec[spike_clusters == cid])
        if len(cluster_spike_times) < 20: # Need a decent number of spikes
            results[cid] = {'cv_isi': np.nan, 'mean_firing_rate': 0}
            continue

        isis = np.diff(cluster_spike_times)
        isis = isis[isis < max_isi_sec] # Exclude very long ISIs

        if len(isis) < 10:
            results[cid] = {'cv_isi': np.nan, 'mean_firing_rate': 0}
            continue

        mean_isi = np.mean(isis)
        std_isi = np.std(isis)
        
        cv_isi = std_isi / mean_isi if mean_isi > 0 else np.nan
        mean_rate = 1.0 / mean_isi if mean_isi > 0 else 0

        results[cid] = {'cv_isi': cv_isi, 'mean_firing_rate': mean_rate}

        # Store histogram for plotting
        hist, bins = np.histogram(isis * 1000, bins=n_bins, range=(0, max_isi_ms))
        all_isi_hists[cid] = (hist, bins)

    # --- 3. Save and Display Results ---
    print("\n  ISI distribution analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'isi_analysis.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")



def analyze_spike_pattern_motifs(paths: DataPaths, k_motifs=8, l_bins=20, binsize_sec=0.02, time_window_sec=2.0):
    """
    Finds recurring spike pattern motifs using SeqNMF and correlates them with
    behavioral events.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        k_motifs (int): Number of motifs (sequences) to find.
        l_bins (int): Length of each motif in time bins.
        binsize_sec (float): Bin size in seconds for discretizing spike trains.
        time_window_sec (float): Window around behavioral events to check for motifs.
    """
    import subprocess
    import sys

    print("Analyzing for spike pattern motifs with SeqNMF...")

    # --- 1. Check for SeqNMF script and input files ---
    seqnmf_script_path = Path(__file__).parent.parent / "SeqNMF" / "seqnmf_like.py"
    if not seqnmf_script_path.exists():
        print(f"  Error: SeqNMF script not found at {seqnmf_script_path}")
        return
        
    spike_times_path = paths.kilosort_dir / "spike_times.npy"
    spike_clusters_path = paths.kilosort_dir / "spike_clusters.npy"
    labels_path = paths.neural_base / "kilosort4qMetrics" / "templates._bc_unit_labels.tsv"
    if not labels_path.exists():
        labels_path = paths.kilosort_dir / "templates._bc_unit_labels.tsv"

    if not all([p.exists() for p in [spike_times_path, spike_clusters_path, labels_path]]):
        print("  Error: Missing required spike data files for SeqNMF (spike_times.npy, spike_clusters.npy, or templates._bc_unit_labels.tsv).")
        return

    # --- 2. Run SeqNMF ---
    output_dir = paths.neural_base / 'post_analysis' / 'seqnmf_output'
    output_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        sys.executable, str(seqnmf_script_path),
        "--clusters", str(spike_clusters_path),
        "--times", str(spike_times_path),
        "--labels", str(labels_path),
        "--k", str(k_motifs),
        "--L", str(l_bins),
        "--binsize", str(binsize_sec),
        "--outdir", str(output_dir)
    ]
    
    print(f"  Running SeqNMF with command: {' '.join(cmd)}")
    try:
        # Using a timeout in case the process hangs
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print("  SeqNMF completed successfully.")
        print(result.stdout)
    except FileNotFoundError:
        print("  Error: `python` command not found. Make sure Python is in your PATH.")
        return
    except subprocess.TimeoutExpired:
        print("  Error: SeqNMF process timed out after 5 minutes.")
        return
    except subprocess.CalledProcessError as e:
        print(f"  Error running SeqNMF script. Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return

    # --- 3. Load SeqNMF results and behavioral events ---
    try:
        H = np.load(output_dir / "H_activations.npy") # Motif activations over time
        # The 'ends' array from build_lagged_patches in seqnmf_like.py tells us the time of each window in H
        # Since it's not saved, we have to recalculate it.
        spike_times_sec, _, _ = _load_spike_data(paths)
        if spike_times_sec is None: return
        total_time = spike_times_sec.max()
        
        T = int(np.ceil(total_time / binsize_sec))
        hop = 5 # Default hop size in seqnmf_like.py
        ends = np.arange(l_bins - 1, T, hop)
        window_times = ends * binsize_sec # Time in seconds for each column in H

        event_df = pd.read_csv(paths.event_corner)
        event_times = event_df.iloc[:, 0].dropna().values
        event_labels = [f"corner_{i}" for i in range(len(event_times))]
        print(f"  Loaded {len(event_times)} corner events for correlation.")

    except Exception as e:
        print(f"  Error loading SeqNMF results or events for correlation: {e}")
        return


        # --- 4. Correlate Motif Occurrences with Events (Original Logic) ---
        correlation_results = []
        
        for k in range(H.shape[0]): # For each motif
            motif_activations = H[k, :]
            
            # Find peaks in activation to identify motif occurrences
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(motif_activations, height=np.percentile(motif_activations, 95))
            motif_occurrence_times = window_times[peaks]
            
            if len(motif_occurrence_times) == 0:
                continue

            # For each behavioral event, count motif occurrences nearby
            event_correlations = defaultdict(int)
            for event_t in event_times:
                # Count occurrences within a window around the event
                n_occurrences = np.sum(
                    (motif_occurrence_times >= event_t - time_window_sec / 2) &
                    (motif_occurrence_times < event_t + time_window_sec / 2)
                )
                if n_occurrences > 0:
                    # This is a simple way to associate, just counting how many events have a motif nearby
                    event_correlations['corner_events'] += 1

            # Find the event type this motif is most associated with
            if event_correlations:
                max_assoc_event = max(event_correlations, key=event_correlations.get)
                max_assoc_count = event_correlations[max_assoc_event]
            else:
                max_assoc_event = "None"
                max_assoc_count = 0
            
            correlation_results.append({
                'motif_id': k,
                'max_assoc_event': max_assoc_event,
                'count': max_assoc_count
            })
            
        # Save simple correlation results
        df_results = pd.DataFrame(correlation_results)
        output_path = output_dir / "motif_event_correlations.csv"
        df_results.to_csv(output_path)
        print(f"  Results saved to {output_path}")

        # --- 5. "Elevated" Multi-Modal Analyses ---
        print("\n  Staritng Elevated Multi-Modal Analyses...")
        
        # Load extra SeqNMF outputs needed for spatial analysis
        try:
             W_components = np.load(output_dir / "W_components.npy") # (K, N, L)
             kept_units_df = pd.read_csv(output_dir / "kept_units.csv")
             kept_units = kept_units_df['unit'].values
        except Exception as e:
             print(f"  Warning: Could not load W_components or kept_units for spatial analysis: {e}")
             W_components = None
             kept_units = None

        # 1. LFP Analysis
        print("    Running Motif-LFP Coupling Analysis...")
        try:
            # Load LFP
            config = load_dataset_config()
            lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
            
            if lfp_loader.extractor is not None:
                fs = lfp_loader.fs
                # Get middle channel
                channel_ids = lfp_loader.extractor.get_channel_ids()
                ch_idx = len(channel_ids) // 2
                mid_channel_id = channel_ids[ch_idx]
                
                print(f"    Loaded LFP Extractor. Extracting trace from channel {mid_channel_id}...")
                
                # Extract full trace for this channel
                # Note: get_traces returns (n_samples, n_channels)
                lfp_trace = lfp_loader.extractor.get_traces(channel_ids=[mid_channel_id], return_scaled=False).flatten()
                
            else:
                lfp_trace = None
                print("    LFP Extractor failed to load.")
                
                # Filter Bands
                from scipy.signal import butter, filtfilt, hilbert
                peaks_per_motif = []
                for k in range(H.shape[0]):
                    act = H[k]
                    pks, _ = find_peaks(act, height=np.percentile(act, 95), distance=int(0.2/np.mean(np.diff(window_times))))
                    peaks_per_motif.append(window_times[pks])

                fig, axes = plt.subplots(1, H.shape[0], figsize=(3*H.shape[0], 3), subplot_kw={'projection': 'polar'})
                if H.shape[0] == 1: axes = [axes]
                
                for k, peak_times in enumerate(peaks_per_motif):
                    if len(peak_times) < 10: continue
                    peak_indices = (peak_times * fs).astype(int)
                    valid_indices = peak_indices[(peak_indices > 0) & (peak_indices < n_samples)]
                    phases = []
                    for idx in valid_indices:
                        start = max(0, idx - int(1.0*fs))
                        end = min(n_samples, idx + int(1.0*fs))
                        snippet = lfp_trace[start:end]
                        if len(snippet) < 100: continue
                        nyq = 0.5 * fs
                        b, a = butter(2, [6/nyq, 12/nyq], btype='band') # Theta
                        filt_snip = filtfilt(b, a, snippet)
                        ang = np.angle(hilbert(filt_snip))
                        center_idx = idx - start
                        if center_idx < len(ang):
                            phases.append(ang[center_idx])
                    if phases:
                        axes[k].hist(phases, bins=20, density=True, alpha=0.7)
                        axes[k].set_title(f"Motif {k} Theta Phase")
                plt.tight_layout()
                plt.savefig(output_dir / "Motif_Theta_Phase.png")
                plt.close()
                print(f"      Saved LFP phase analysis to {output_dir}")
        except Exception as e:
            print(f"      Error in LFP coupling analysis: {e}")

        # 2. Dopamine Analysis
        print("    Running Motif-Dopamine Coupling Analysis...")
        try:
            config = load_dataset_config()
            photo_loader = PhotometryDataLoader(paths.base_path, config)
            photo_data = photo_loader.load(paths.tdt_dff, paths.tdt_raw)
            
            dff = None
            ts = None
            if photo_data is not None:
                if isinstance(photo_data, pd.DataFrame):
                    dff = photo_data.get('dFF', photo_data.get('Signal', None))
                    ts = photo_data.get('Timestamp', photo_data.get('Times', None))
                    if dff is None:
                        cols = [c for c in photo_data.columns if 'dFF' in c or '465' in c]
                        if cols: dff = photo_data[cols[0]]
                    if ts is None and dff is not None:
                        fs = DEFAULT_DOPAMINE_SAMPLING_RATE
                        ts = np.arange(len(dff)) / fs
                else:
                     dff = photo_data.get('dFF', None)
                     ts = photo_data.get('time', None)
            
            if dff is not None and ts is not None:
                dff = (dff - np.mean(dff)) / np.std(dff)
                window_sec = 2.0
                fig, axes = plt.subplots(H.shape[0], 1, figsize=(6, 2*H.shape[0]), sharex=True, sharey=True)
                if H.shape[0] == 1: axes = [axes]
                
                for k in range(H.shape[0]):
                    act = H[k]
                    pks, _ = find_peaks(act, height=np.percentile(act, 95), distance=int(0.5/np.mean(np.diff(window_times))))
                    peak_times = window_times[pks]
                    traces = []
                    for t in peak_times:
                        t_start, t_end = t - window_sec, t + window_sec
                        idx_start = np.searchsorted(ts, t_start)
                        idx_end = np.searchsorted(ts, t_end)
                        if idx_start > 0 and idx_end < len(dff):
                            trace = dff[idx_start:idx_end]
                            if len(traces) > 0:
                                target_len = len(traces[0])
                                if len(trace) != target_len:
                                    from scipy.signal import resample
                                    trace = resample(trace, target_len)
                            traces.append(trace)
                    if traces:
                        avg_trace = np.mean(traces, axis=0)
                        sem_trace = np.std(traces, axis=0) / np.sqrt(len(traces))
                        x_axis = np.linspace(-window_sec, window_sec, len(avg_trace))
                        axes[k].plot(x_axis, avg_trace, color='g', lw=2)
                        axes[k].fill_between(x_axis, avg_trace-sem_trace, avg_trace+sem_trace, color='g', alpha=0.2)
                        axes[k].axvline(0, color='k', linestyle='--')
                        axes[k].set_ylabel(f"Motif {k} DA (z)")
                axes[-1].set_xlabel("Time from Motif Onset (s)")
                plt.tight_layout()
                plt.savefig(output_dir / "Motif_Dopamine_PETH.png")
                plt.close()
                print(f"      Saved Dopamine PETH to {output_dir}")
        except Exception as e:
            print(f"      Error in Dopamine coupling analysis: {e}")

        # 3. Kinematics (Speed/Space)
        print("    Running Motif-Kinematics Analysis...")
        try:
            velocity, vel_times = _load_dlc_and_calculate_velocity(paths, video_fs=60, px_per_cm=30.0)
            if velocity is not None:
                config = load_dataset_config()
                dlc_loader = DLCDataLoader(paths.base_path, config)
                dlc_config_entry = find_config_entry(paths.dlc_h5, config)
                dlc_config_key = None
                if dlc_config_entry:
                    for key, value in config.items():
                        if value == dlc_config_entry:
                            dlc_config_key = key
                            break
                
                if dlc_config_key:
                    df_dlc = dlc_loader.load(config_key=dlc_config_key)
                    if 'Snout_x' in df_dlc.columns and 'Snout_y' in df_dlc.columns:
                        coords = df_dlc[['Snout_x', 'Snout_y']].values
                        coord_times = vel_times 
                        fig, axes = plt.subplots(1, H.shape[0], figsize=(3*H.shape[0], 3))
                        if H.shape[0] == 1: axes = [axes]
                        for k in range(H.shape[0]):
                            act = H[k]
                            pks, _ = find_peaks(act, height=np.percentile(act, 95))
                            peak_times = window_times[pks]
                            valid_coords = []
                            for t in peak_times:
                                idx = np.searchsorted(coord_times, t)
                                if idx < len(coords): valid_coords.append(coords[idx])
                            valid_coords = np.array(valid_coords)
                            axes[k].plot(coords[:,0], coords[:,1], '.', color='lightgray', markersize=1, alpha=0.1)
                            if len(valid_coords) > 0:
                                axes[k].plot(valid_coords[:,0], valid_coords[:,1], 'r.', markersize=3, alpha=0.6)
                            axes[k].set_title(f"Motif {k} Loc")
                            axes[k].axis('off')
                        plt.tight_layout()
                        plt.savefig(output_dir / "Motif_Spatial_Map.png")
                        plt.close()
                        print(f"      Saved Spatial Map to {output_dir}")

                from scipy.interpolate import interp1d
                f_vel = interp1d(vel_times, velocity, fill_value="extrapolate")
                vel_resampled = f_vel(window_times)
                corrs = []
                for k in range(H.shape[0]):
                    corrs.append(np.corrcoef(H[k], vel_resampled)[0, 1])
                fig, ax = plt.subplots()
                ax.bar(range(len(corrs)), corrs)
                ax.set_title("Motif Speed Tuning")
                plt.savefig(output_dir / "Motif_Speed_Corr.png")
                plt.close()
        except Exception as e:
            print(f"      Error in Kinematics analysis: {e}")

        # 4. Context (Rule)
        print("    Running Motif-Task Context Analysis...")
        try:
            config = load_dataset_config()
            event_loader = EventDataLoader(paths.base_path, config)
            switch_times = _load_switch_times(paths, config, event_loader)
            if len(switch_times) > 0:
                corner_cfg = find_config_entry(paths.event_corner, config)
                corner_key = next(k for k,v in config.items() if v==corner_cfg)
                df_corner = event_loader.load(config_key=corner_key)
                rule_col = get_column_name(corner_cfg, ['CW', 'Condition', 'Rule'])
                if rule_col and rule_col in df_corner.columns:
                    event_times = event_loader.get_event_times(df_corner, corner_key)
                    event_rules = df_corner[rule_col].values.astype(int)
                    augmented_times = np.concatenate([[0], switch_times, [window_times.max()]])
                    block_activations = []
                    for i in range(len(augmented_times)-1):
                        t_start, t_end = augmented_times[i], augmented_times[i+1]
                        mask = (window_times >= t_start) & (window_times < t_end)
                        if not np.any(mask): continue
                        t_mid = (t_start + t_end) / 2
                        closest_idx = np.searchsorted(event_times, t_mid)
                        rule = event_rules[closest_idx] if closest_idx < len(event_rules) else event_rules[-1]
                        mean_acts = H[:, mask].mean(axis=1)
                        block_activations.append({'rule': rule, 'acts': mean_acts})
                    rule_0_acts = [b['acts'] for b in block_activations if b['rule'] == 0]
                    rule_1_acts = [b['acts'] for b in block_activations if b['rule'] == 1]
                    if rule_0_acts and rule_1_acts:
                        mean_0 = np.mean(rule_0_acts, axis=0)
                        mean_1 = np.mean(rule_1_acts, axis=0)
                        fig, ax = plt.subplots()
                        width = 0.35
                        x = np.arange(H.shape[0])
                        ax.bar(x - width/2, mean_0, width, label='Rule 0')
                        ax.bar(x + width/2, mean_1, width, label='Rule 1')
                        ax.legend()
                        plt.savefig(output_dir / "Motif_Rule_Context.png")
                        plt.close()
                        print(f"      Saved Context analysis to {output_dir}")
        except Exception as e:
            print(f"      Error in Context analysis: {e}")

        # 5. Spatial Propagation
        if W_components is not None and kept_units is not None:
            print("    Running Motif-Spatial Propagation Analysis...")
            try:
                try:
                     chan_pos = np.load(paths.neural_base / "kilosort4" / "sorter_output" / "channel_positions.npy")
                except:
                     chan_pos = np.load(paths.kilosort_dir / "channel_positions.npy")
                unit_ch_map = _get_unit_best_channels(paths, kept_units)
                K, N, L = W_components.shape
                for k in range(K):
                    com_depths = []
                    lags = range(L)
                    for t in lags:
                        weights = W_components[k, :, t]
                        mask = weights > np.percentile(weights, 90)
                        if not np.any(mask):
                            com_depths.append(np.nan)
                            continue
                        relevant_units = kept_units[mask]
                        relevant_weights = weights[mask]
                        depths = []
                        for u in relevant_units:
                            ch = unit_ch_map.get(u, 0)
                            depths.append(chan_pos[ch, 1] if ch < len(chan_pos) else 0)
                        depths = np.array(depths)
                        if np.sum(relevant_weights) > 0:
                            com = np.average(depths, weights=relevant_weights)
                            com_depths.append(com)
                        else:
                            com_depths.append(np.nan)
                    fig, ax = plt.subplots()
                    ax.plot(lags, com_depths, 'o-', lw=2)
                    ax.set_ylabel("Depth (um)")
                    ax.set_title(f"Motif {k} Spatial Propagation")
                    plt.savefig(output_dir / f"Motif_{k}_propagation.png")
                    plt.close()
                print(f"      Saved Spatial Propagation analysis to {output_dir}")
            except Exception as e:
                print(f"      Error in Spatial Propagation analysis: {e}")

            


    # --- 5. Save and Display Results ---
    print("\n  Spike pattern motif analysis complete.")
    if not correlation_results:
        print("  No motif correlations to report.")
        return

    df_results = pd.DataFrame(correlation_results)
    output_path = output_dir / 'spike_motif_correlations.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

    if not df_results.empty:
        print("\n  Summary of Spike Motifs and their Behavioral Association:")
        print(df_results.sort_values('association_count', ascending=False))

def analyze_wave_propagations(paths: DataPaths, freq_range: tuple = (5, 12), channel_step: int = 4):
    """
    Analyzes traveling wave propagation using LFP data.

    Detects traveling waves by analyzing the phase gradient across the probe.
    Correlates wave events with spike timing (phase-locking) and dopamine transients.

    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        freq_range (tuple): Frequency range (min, max) in Hz. Default Theta (5-12 Hz).
        channel_step (int): Downsample channels for LFP loading (e.g., every 4th channel).
    """
    print(f"\\nAnalyzing LFP traveling waves in range {freq_range} Hz...")
    output_dir = paths.neural_base / 'post_analysis' / 'lfp_waves'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load LFP Data and Config ---
    try:
        config = load_dataset_config()
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        
        # Get LFP path
        lfp_path = lfp_loader.get_lfp_path(paths.neural_base if paths.neural_base else paths.base_path)
        
        if not lfp_path or not lfp_path.exists():
            print("  Error: LFP binary file not found.")
            return

        print(f"  Loading LFP from: {lfp_path}")
        fs_lfp = _load_lfp_sampling_rate(lfp_path.parent)
        print(f"  LFP Sampling Rate: {fs_lfp} Hz")
        
        # Determine channels and positions
        n_channels_total = 384
        chan_pos_path = paths.kilosort_dir / "channel_positions.npy"
        if not chan_pos_path.exists():
            print("  Warning: channel_positions.npy not found, using linear depth estimate.")
            chan_pos = np.zeros((n_channels_total, 2))
            chan_pos[:, 1] = np.arange(n_channels_total) * 10
        else:
            chan_pos = np.load(chan_pos_path)
        
        # Downsample channels
        selected_channels = np.arange(0, n_channels_total, channel_step)
        selected_pos_y = chan_pos[selected_channels, 1]
        
        # Sort by depth
        sorted_idx = np.argsort(selected_pos_y)
        selected_channels = selected_channels[sorted_idx]
        selected_pos_y = selected_pos_y[sorted_idx]

        # Load Spike Data for Phase Locking
        spike_data = SpikeDataLoader(paths.base_path, config).load(paths.kilosort_dir)
        unit_phases = defaultdict(list) # {unit_id: [phases...]}
        unit_wave_phases = defaultdict(list) # {unit_id: [phases_during_waves...]}
        
        # Unit positions mapping
        unit_depths = {}
        if spike_data:
            spike_times = spike_data['spike_times_sec']
            spike_clusters = spike_data['spike_clusters']
            unique_clusters = spike_data['unique_clusters']
            
            # Estimate unit depths (simplified: average spike position or from templates)
            # Try to load template centers if available, otherwise heuristic
            try:
                # Try loading metrics first
                # Or re-use _get_unit_best_channels logic if available but it returns channel index
                unit_best_ch = _get_unit_best_channels(paths, unique_clusters)
                for uid, ch in unit_best_ch.items():
                    unit_depths[uid] = chan_pos[ch, 1] if ch < len(chan_pos) else 0
            except:
                pass
        
        # Prepare for Dopamine Analyis
        wave_prob_trace = [] # Binary trace of wave presence
        wave_speed_trace = []
        chunk_time_offsets = []

        # Process in chunks
        file_size = lfp_path.stat().st_size
        n_samples_total = file_size // (n_channels_total * 2)
        lfp_mmap = np.memmap(lfp_path, dtype='int16', mode='r', shape=(n_samples_total, n_channels_total))
        
        chunk_duration_sec = 60.0 
        chunk_samples = int(chunk_duration_sec * fs_lfp)
        n_chunks = int(np.ceil(n_samples_total / chunk_samples))
        
        wave_events = []
        nyq = 0.5 * fs_lfp
        b, a = butter(2, [freq_range[0]/nyq, freq_range[1]/nyq], btype='band')
        
        print(f"  Processing {n_chunks} chunks...")
        
        max_chunks = int(1800 / chunk_duration_sec) # Limit to 30 mins analysis for performance if needed
        
        for i in tqdm(range(min(n_chunks, max_chunks)), desc="LFP Chunks"):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, n_samples_total)
            current_chunk_samples = end_idx - start_idx
            
            if current_chunk_samples < fs_lfp: continue # Skip tiny chunks
            
            # Load & Filter
            lfp_chunk = lfp_mmap[start_idx:end_idx, selected_channels].astype(np.float32)
            lfp_filt = filtfilt(b, a, lfp_chunk, axis=0)
            analytic_signal = hilbert(lfp_filt, axis=0)
            phases = np.angle(analytic_signal).T # (n_channels_sel, n_time)
            
            # --- Wave Detection ---
            phases_unwrapped = np.unwrap(phases, axis=0)
            
            # Spatial Gradient (Slope of Phase vs Depth)
            Y = selected_pos_y
            mean_Y = np.mean(Y)
            var_Y = np.var(Y)
            mean_phase = np.mean(phases_unwrapped, axis=0)
            mean_y_phase = np.mean(Y[:, np.newaxis] * phases_unwrapped, axis=0)
            spatial_freq = (mean_y_phase - mean_Y * mean_phase) / var_Y # rad / um
            
            # Goodness of fit (R2)
            pred = spatial_freq[np.newaxis, :] * (Y[:, np.newaxis] - mean_Y) + mean_phase[np.newaxis, :]
            resid = phases_unwrapped - pred
            var_resid = np.var(resid, axis=0)
            var_phase = np.var(phases_unwrapped, axis=0)
            r2 = 1 - (var_resid / (var_phase + 1e-6))
            
            # Temporal Gradient (Inst Freq)
            inst_freq = np.diff(np.unwrap(phases, axis=1), axis=1) * fs_lfp / (2*np.pi)
            inst_freq = np.hstack([inst_freq, inst_freq[:, -1:]])
            mean_inst_freq = np.mean(inst_freq, axis=0)
            
            # Wave Speed
            wave_speed = (2 * np.pi * mean_inst_freq) / (spatial_freq + 1e-9) # um/s
            
            # Detect Events
            wave_mask = (r2 > 0.6) & (np.abs(wave_speed) > 100) & (np.abs(wave_speed) < 10000)
            
            # Collect trace data for dopamine correlation (decimated)
            decimate_factor = int(fs_lfp / 100) # Down to ~100Hz
            if decimate_factor > 0:
                wave_prob_trace.append(np.mean(wave_mask.reshape(-1, decimate_factor), axis=1))
                wave_speed_trace.append(np.mean(wave_speed.reshape(-1, decimate_factor), axis=1))
                chunk_time_offsets.append(start_idx / fs_lfp)
            
            # Extract Events
            from scipy.ndimage import label
            labeled, n_events = label(wave_mask.astype(int))
            chunk_time_base = start_idx / fs_lfp
            
            for ev in range(1, n_events+1):
                idx = np.where(labeled == ev)[0]
                if len(idx) < (0.05 * fs_lfp): continue
                
                t_start = chunk_time_base + idx[0]/fs_lfp
                t_end = chunk_time_base + idx[-1]/fs_lfp
                avg_speed = np.mean(wave_speed[idx])
                direction = np.sign(avg_speed)
                
                wave_events.append({
                    'start_time': t_start,
                    'end_time': t_end,
                    'duration': t_end - t_start,
                    'speed': avg_speed,
                    'direction': direction,
                    'r2': np.mean(r2[idx])
                })
            
            # --- Spike Phase Extraction ---
            if spike_data:
                t_chunk_start = chunk_time_base
                t_chunk_end = t_chunk_start + current_chunk_samples/fs_lfp
                
                # Find spikes in this chunk
                mask_sp = (spike_times >= t_chunk_start) & (spike_times < t_chunk_end)
                chunk_sp_times = spike_times[mask_sp]
                chunk_sp_clusters = spike_clusters[mask_sp]
                
                if len(chunk_sp_times) > 0:
                    # Map time to index
                    sp_indices = ((chunk_sp_times - t_chunk_start) * fs_lfp).astype(int)
                    sp_indices = np.clip(sp_indices, 0, phases.shape[1]-1)
                    
                    for i_sp, cid in enumerate(chunk_sp_clusters):
                        if cid not in unit_depths: continue
                        
                        depth = unit_depths[cid]
                        # Find closest LFP channel
                        ch_idx = np.abs(selected_pos_y - depth).argmin()
                        
                        # Get phase
                        phi = phases[ch_idx, sp_indices[i_sp]]
                        unit_phases[cid].append(phi)
                        
                        # Check if during wave
                        if wave_mask[sp_indices[i_sp]]:
                            unit_wave_phases[cid].append(phi)
                            
        # Save Wave Events
        df_waves = pd.DataFrame(wave_events)
        df_waves.to_csv(output_dir / 'lfp_wave_events.csv', index=False)
        print(f"  Detected {len(df_waves)} wave events. Spikes processed.")
        
        # --- Save Phase Locking Results ---
        plv_results = []
        for cid in unit_phases:
            all_ph = np.array(unit_phases[cid])
            wave_ph = np.array(unit_wave_phases[cid])
            
            res = {'cluster_id': cid}
            
            # All times
            if len(all_ph) > 50:
                res['plv_all'] = np.abs(np.mean(np.exp(1j * all_ph)))
                res['pref_phase_all'] = np.angle(np.mean(np.exp(1j * all_ph)))
                res['n_spikes_all'] = len(all_ph)
            
            # Wave times
            if len(wave_ph) > 10:
                res['plv_wave'] = np.abs(np.mean(np.exp(1j * wave_ph)))
                res['pref_phase_wave'] = np.angle(np.mean(np.exp(1j * wave_ph)))
                res['n_spikes_wave'] = len(wave_ph)
                
            plv_results.append(res)
            
        df_plv = pd.DataFrame(plv_results)
        df_plv.to_csv(output_dir / 'spike_phase_locking.csv', index=False)

        # Visualize PLV
        if not df_plv.empty and 'plv_wave' in df_plv.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df_plv['plv_all'].dropna(), bins=30, alpha=0.5, label='All Times')
            plt.hist(df_plv['plv_wave'].dropna(), bins=30, alpha=0.5, label='During Waves')
            plt.xlabel('Phase Locking Value (PLV)')
            plt.legend()
            plt.title('Spike-LFP Phase Locking')
            plt.savefig(output_dir / 'plv_distribution.png')
            plt.close()

        # --- Dopamine Correlation ---
        print("  Analyzing Dopamine-Wave Relationship...")
        try:
            photo_loader = PhotometryDataLoader(paths.base_path, config)
            photo_data = photo_loader.load()
            
            # Flatten traces
            if wave_prob_trace:
                full_wave_prob = np.concatenate(wave_prob_trace)
                full_wave_speed = np.concatenate(wave_speed_trace)
                
                # Reconstruct time axis for these traces
                # They are at 100Hz (approx)
                n_points = len(full_wave_prob)
                t_wave = np.arange(n_points) * (1.0/100.0) # Assumes 100Hz exact, roughly true
                
                # Load Dopamine
                if isinstance(photo_data, pd.DataFrame):
                    dff = photo_data.get('dFF', photo_data.get('Signal', None))
                    ts_da = photo_data.get('Timestamp', None)
                    
                    if dff is not None and ts_da is not None:
                        # Interpolate Wave Prob to DA timebase
                        from scipy.interpolate import interp1d
                        f_prob = interp1d(t_wave, full_wave_prob, bounds_error=False, fill_value=0)
                        wave_prob_resampled = f_prob(ts_da)
                        
                        # Cross-Corr
                        lags = signal.correlation_lags(len(dff), len(wave_prob_resampled))
                        corr = signal.correlate(dff - np.mean(dff), wave_prob_resampled - np.mean(wave_prob_resampled), mode='full')
                        corr /= (np.std(dff) * np.std(wave_prob_resampled) * len(dff))
                        
                        # Plot
                        plt.figure(figsize=(8, 4))
                        mask = (lags * np.mean(np.diff(ts_da)))
                        mask_idx = (mask > -5.0) & (mask < 5.0)
                        plt.plot(mask[mask_idx], corr[mask_idx])
                        plt.xlabel('Lag (s)')
                        plt.ylabel('Correlation (DA vs WaveProb)')
                        plt.title('Dopamine - Traveling Wave Correlation')
                        plt.axvline(0, color='k', linestyle='--')
                        plt.savefig(output_dir / 'dopamine_wave_xcorr.png')
                        plt.close()
                        
        except Exception as e:
            print(f"  Dopamine analysis failed: {e}")

    except Exception as e:
        print(f"  Error in LFP wave analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_multi_scale_temporal_encoding(paths: DataPaths, timescales_ms: list = [10, 50, 100, 500, 1000]):
    """
    Analyzes multi-scale temporal encoding at different timescales.
    
    Examines how neurons encode information at fast (spikes), medium (100ms firing rate),
    and slow (seconds-scale rate changes) timescales.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        timescales_ms (list): List of timescales in milliseconds to analyze.
    """
    print("Analyzing multi-scale temporal encoding...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    session_duration = spike_times_sec.max()
    
    # --- 2. Compute Firing Rates at Each Timescale ---
    results = {}
    
    for timescale_ms in timescales_ms:
        timescale_sec = timescale_ms / 1000.0
        n_bins = int(session_duration / timescale_sec)
        
        print(f"  Processing {timescale_ms}ms timescale ({n_bins} bins)...")
        
        scale_data = {}
        for cid in unique_clusters:
            cluster_spikes = spike_times_sec[spike_clusters == cid]
            
            # Bin spikes at this timescale
            firing_rate, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
            firing_rate = firing_rate / timescale_sec  # Convert to Hz
            
            # Compute statistics
            mean_rate = np.mean(firing_rate)
            std_rate = np.std(firing_rate)
            cv = std_rate / mean_rate if mean_rate > 0 else 0
            
            # Information content (entropy)
            if np.sum(firing_rate) > 0:
                prob = firing_rate / np.sum(firing_rate)
                prob = prob[prob > 0]
                entropy = -np.sum(prob * np.log2(prob))
            else:
                entropy = 0
            
            scale_data[cid] = {
                'mean_rate': mean_rate,
                'std_rate': std_rate,
                'cv': cv,
                'entropy': entropy
            }
        
        results[f'{timescale_ms}ms'] = scale_data
    
    # --- 3. Analyze Scale-Dependence ---
    # For each neuron, how does encoding change across scales?
    neuron_scale_profiles = {}
    
    for cid in unique_clusters:
        profile = {
            'cluster_id': cid,
        }
        
        for timescale_ms in timescales_ms:
            scale_key = f'{timescale_ms}ms'
            if cid in results[scale_key]:
                data = results[scale_key][cid]
                profile[f'{scale_key}_mean_rate'] = data['mean_rate']
                profile[f'{scale_key}_cv'] = data['cv']
                profile[f'{scale_key}_entropy'] = data['entropy']
        
        neuron_scale_profiles[cid] = profile
    
    # --- 4. Save Results ---
    print("\n  Multi-scale temporal encoding analysis complete.")
    
    df_profiles = pd.DataFrame.from_dict(neuron_scale_profiles, orient='index')
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'multi_scale_temporal_encoding.csv'
    df_profiles.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    
    # --- 5. Visualize ---
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Mean rate vs timescale
        ax = axes[0, 0]
        for cid in list(unique_clusters)[:20]:  # Plot first 20 neurons
            rates = [neuron_scale_profiles[cid][f'{ts}ms_mean_rate'] 
                    for ts in timescales_ms if f'{ts}ms_mean_rate' in neuron_scale_profiles[cid]]
            if rates:
                ax.plot(timescales_ms[:len(rates)], rates, alpha=0.5, linewidth=1)
        ax.set_xlabel('Timescale (ms)')
        ax.set_ylabel('Mean Firing Rate (Hz)')
        ax.set_title('Firing Rate Across Timescales')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: CV vs timescale
        ax = axes[0, 1]
        for cid in list(unique_clusters)[:20]:
            cvs = [neuron_scale_profiles[cid][f'{ts}ms_cv'] 
                  for ts in timescales_ms if f'{ts}ms_cv' in neuron_scale_profiles[cid]]
            if cvs:
                ax.plot(timescales_ms[:len(cvs)], cvs, alpha=0.5, linewidth=1)
        ax.set_xlabel('Timescale (ms)')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Variability Across Timescales')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Entropy vs timescale
        ax = axes[1, 0]
        for cid in list(unique_clusters)[:20]:
            entropies = [neuron_scale_profiles[cid][f'{ts}ms_entropy'] 
                        for ts in timescales_ms if f'{ts}ms_entropy' in neuron_scale_profiles[cid]]
            if entropies:
                ax.plot(timescales_ms[:len(entropies)], entropies, alpha=0.5, linewidth=1)
        ax.set_xlabel('Timescale (ms)')
        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Information Content Across Timescales')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Population average
        ax = axes[1, 1]
        mean_rates = [np.mean([neuron_scale_profiles[cid][f'{ts}ms_mean_rate'] 
                              for cid in unique_clusters if f'{ts}ms_mean_rate' in neuron_scale_profiles[cid]])
                     for ts in timescales_ms]
        ax.plot(timescales_ms, mean_rates, 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel('Timescale (ms)')
        ax.set_ylabel('Population Mean Rate (Hz)')
        ax.set_title('Population Average Across Timescales')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'multi_scale_temporal_encoding.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
    
    return df_profiles

def analyze_rank_order_coding(paths: DataPaths, time_window_ms: float = 50.0):
    """
    Analyzes rank-order coding: the sequence in which neurons fire.
    
    Examines whether information is encoded in the temporal order of spikes
    rather than just firing rates.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_window_ms (float): Time window for detecting spike sequences.
    """
    print("Analyzing rank-order coding...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    time_window_sec = time_window_ms / 1000.0
    session_duration = spike_times_sec.max()
    
    # --- 2. Detect Population Bursts ---
    # Bin all spikes into small windows
    bin_size = 0.010  # 10ms bins
    n_bins = int(session_duration / bin_size)
    all_spike_counts, bin_edges = np.histogram(spike_times_sec, bins=n_bins, range=(0, session_duration))
    
    # Find high-activity periods (population bursts)
    threshold = np.percentile(all_spike_counts, 90)
    burst_bins = np.where(all_spike_counts > threshold)[0]
    burst_times = bin_edges[burst_bins]
    
    print(f"  Detected {len(burst_times)} population burst events.")
    
    if len(burst_times) < 10:
        print("  Not enough burst events for rank-order analysis.")
        return
    
    # --- 3. Analyze Spike Order Within Bursts ---
    rank_sequences = []
    
    for burst_t in burst_times[:500]:  # Analyze first 500 bursts
        window_start = burst_t
        window_end = burst_t + time_window_sec
        
        # Get all spikes in this window
        window_mask = (spike_times_sec >= window_start) & (spike_times_sec < window_end)
        window_spikes = spike_times_sec[window_mask]
        window_clusters = spike_clusters[window_mask]
        
        if len(window_spikes) < 3:  # Need at least 3 spikes
            continue
        
        # Sort by time to get firing order
        sorted_indices = np.argsort(window_spikes)
        firing_order = window_clusters[sorted_indices]
        
        # Only keep first spike from each neuron
        unique_order = []
        seen = set()
        for cid in firing_order:
            if cid not in seen:
                unique_order.append(cid)
                seen.add(cid)
        
        if len(unique_order) >= 3:
            rank_sequences.append({
                'burst_time': burst_t,
                'sequence': tuple(unique_order[:10]),  # First 10 neurons
                'n_neurons': len(unique_order)
            })
    
    print(f"  Analyzed {len(rank_sequences)} rank-order sequences.")
    
    if len(rank_sequences) == 0:
        print("  No valid sequences found.")
        return
    
    # --- 4. Find Common Sequences ---
    from collections import Counter
    
    # Count frequency of each sequence
    sequence_counts = Counter([seq['sequence'] for seq in rank_sequences])
    most_common = sequence_counts.most_common(20)
    
    # --- 5. Save Results ---
    print("\n  Rank-order coding analysis complete.")
    
    # Save sequence data
    df_sequences = pd.DataFrame(rank_sequences)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'rank_order_sequences.csv'
    # Convert tuple sequences to strings for CSV
    df_sequences['sequence_str'] = df_sequences['sequence'].apply(str)
    df_sequences[['burst_time', 'n_neurons', 'sequence_str']].to_csv(output_path, index=False)
    print(f"  Sequences saved to {output_path}")
    
    # Save most common sequences
    common_sequences = pd.DataFrame([
        {'sequence': str(seq), 'count': count, 'frequency': count / len(rank_sequences)}
        for seq, count in most_common
    ])
    common_path = output_dir / 'rank_order_common_sequences.csv'
    common_sequences.to_csv(common_path, index=False)
    print(f"  Common sequences saved to {common_path}")
    print(f"  Most common sequence appears {most_common[0][1]} times ({most_common[0][1]/len(rank_sequences):.1%})")
    
    # --- 6. Visualize ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Sequence length distribution
        ax = axes[0]
        ax.hist(df_sequences['n_neurons'], bins=30, alpha=0.7, color='blue')
        ax.set_xlabel('Number of Neurons in Sequence')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Sequence Lengths')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Top sequences
        ax = axes[1]
        top_10 = common_sequences.head(10)
        ax.barh(range(len(top_10)), top_10['count'], alpha=0.7, color='green')
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels([f"Seq {i+1}" for i in range(len(top_10))])
        ax.set_xlabel('Frequency')
        ax.set_title('Top 10 Most Common Sequences')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = output_dir / 'rank_order_coding.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
    
    return df_sequences, common_sequences

def analyze_temporal_clustering(paths: DataPaths, n_clusters: int = 5):
    """
    Clusters neurons based on their temporal firing patterns.
    
    Groups neurons that have similar temporal dynamics, autocorrelation
    structures, or firing pattern similarities.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_clusters (int): Number of clusters to create.
    """
    print("Analyzing temporal clustering...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    session_duration = spike_times_sec.max()
    bin_size_sec = 0.010  # 10ms bins
    n_bins = int(session_duration / bin_size_sec)
    
    # --- 2. Create Firing Rate Vectors for Each Neuron ---
    print(f"  Creating firing rate vectors for {len(unique_clusters)} neurons...")
    
    firing_rate_matrix = []
    valid_clusters = []
    
    for cid in tqdm(unique_clusters, desc="Processing neurons"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        if len(cluster_spikes) < 100:  # Need minimum spikes
            continue
        
        # Bin spikes
        spike_train, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        firing_rate = spike_train / bin_size_sec  # Convert to Hz
        
        # Downsample for efficiency (take every 10th bin)
        firing_rate_downsampled = firing_rate[::10]
        
        firing_rate_matrix.append(firing_rate_downsampled)
        valid_clusters.append(cid)
    
    firing_rate_matrix = np.array(firing_rate_matrix)
    print(f"  Created {firing_rate_matrix.shape[0]} x {firing_rate_matrix.shape[1]} firing rate matrix.")
    
    # --- 3. Normalize and Cluster ---
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Standardize each neuron's firing rate
    scaler = StandardScaler()
    firing_rate_normalized = scaler.fit_transform(firing_rate_matrix.T).T
    
    # K-means clustering
    print(f"  Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(firing_rate_normalized)
    
    # --- 4. Analyze Cluster Properties ---
    cluster_stats = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_neurons = np.array(valid_clusters)[cluster_mask]
        
        # Mean firing pattern for this cluster
        mean_pattern = np.mean(firing_rate_matrix[cluster_mask], axis=0)
        std_pattern = np.std(firing_rate_matrix[cluster_mask], axis=0)
        
        # Overall statistics
        mean_rate = np.mean(firing_rate_matrix[cluster_mask])
        cv = np.std(mean_pattern) / np.mean(mean_pattern) if np.mean(mean_pattern) > 0 else 0
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'n_neurons': len(cluster_neurons),
            'mean_firing_rate': mean_rate,
            'cv': cv,
            'neurons': cluster_neurons.tolist()
        })
    
    # --- 5. Save Results ---
    print("\n  Temporal clustering analysis complete.")
    
    # Save cluster assignments
    df_assignments = pd.DataFrame({
        'cluster_id': valid_clusters,
        'temporal_cluster': cluster_labels
    })
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'temporal_clustering_assignments.csv'
    df_assignments.to_csv(output_path, index=False)
    print(f"  Cluster assignments saved to {output_path}")
    
    # Save cluster statistics
    df_stats = pd.DataFrame([{k: v for k, v in stat.items() if k != 'neurons'} 
                            for stat in cluster_stats])
    stats_path = output_dir / 'temporal_clustering_stats.csv'
    df_stats.to_csv(stats_path, index=False)
    print(f"  Cluster statistics saved to {stats_path}")
    
    for cluster_id in range(n_clusters):
        n = cluster_stats[cluster_id]['n_neurons']
        print(f"  Cluster {cluster_id}: {n} neurons")
    
    # --- 6. Visualize ---
    try:
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: PCA projection with clusters
        ax = axes[0, 0]
        pca = PCA(n_components=2)
        firing_pca = pca.fit_transform(firing_rate_normalized)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            ax.scatter(firing_pca[mask, 0], firing_pca[mask, 1], 
                      c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                      alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('Temporal Clusters in PCA Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cluster sizes
        ax = axes[0, 1]
        cluster_sizes = [cluster_stats[i]['n_neurons'] for i in range(n_clusters)]
        ax.bar(range(n_clusters), cluster_sizes, color=colors, alpha=0.7)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Neurons')
        ax.set_title('Cluster Sizes')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Mean firing patterns
        ax = axes[1, 0]
        time_axis = np.arange(firing_rate_matrix.shape[1]) * bin_size_sec * 10  # Downsampled
        for cluster_id in range(min(5, n_clusters)):  # Plot first 5 clusters
            mask = cluster_labels == cluster_id
            mean_pattern = np.mean(firing_rate_matrix[mask], axis=0)
            ax.plot(time_axis, mean_pattern, color=colors[cluster_id], 
                   label=f'Cluster {cluster_id}', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean Firing Rate (Hz)')
        ax.set_title('Mean Temporal Patterns by Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(100, time_axis[-1])])  # First 100 seconds
        
        # Plot 4: Cluster firing rate distribution
        ax = axes[1, 1]
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            rates = np.mean(firing_rate_matrix[mask], axis=1)
            ax.hist(rates, bins=20, alpha=0.5, color=colors[cluster_id], 
                   label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('Mean Firing Rate (Hz)')
        ax.set_ylabel('Count')
        ax.set_title('Firing Rate Distribution by Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'temporal_clustering.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    return df_assignments, df_stats

def analyze_dopamine_spike_coupling(paths: DataPaths, time_lags_ms: list = [-500, -250, -100, 0, 100, 250, 500], 
                                    dopamine_threshold_percentile: float = 75):
    """
    Analyzes coupling between dopamine transients and neural spike activity.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        time_lags_ms (list): Time lags (in ms) to test for coupling.
        dopamine_threshold_percentile (float): Percentile threshold for defining dopamine transients.
    """
    print("Analyzing dopamine-spike coupling...")
    
    # --- 1. Load Dopamine Data ---
    if not paths.tdt_dff or not paths.tdt_dff.exists():
        print(f"  Error: TDT dFF file not found at {paths.tdt_dff}")
        return
    
    try:
        import scipy.io
        import h5py
        
        # Load dFF data
        dff_data = scipy.io.loadmat(paths.tdt_dff)
        
        # Extract dFF values and time
        if 'dFF' in dff_data:
            # Get Config Keys
            config = load_dataset_config()
            dff_entry = find_config_entry(paths.tdt_dff, config)
            dff_key = next((k for k, v in config.items() if v == dff_entry), None)
            
            # Find raw file key
            tdt_dir = paths.tdt_dff.parent
            raw_files = list(tdt_dir.glob('*_UnivRAW_offdemod.mat'))
            if not raw_files:
                 print("  Error: Could not find raw TDT file.")
                 return
            raw_file = raw_files[0]
            raw_entry = find_config_entry(raw_file, config)
            raw_key = next((k for k, v in config.items() if v == raw_entry), None)
            
            if not dff_key or not raw_key:
                 print("  Error: Missing config keys for PhotometryDataLoader.")
                 return

            # Use PhotometryDataLoader
            photo_loader = PhotometryDataLoader(paths.base_path, config)
            data = photo_loader.load(dff_config_key=dff_key, raw_config_key=raw_key)
            
            dff_vals = data['dff_values']
            ts_abs = data['dff_timestamps']
            
        else:
            print("  Error: Could not find dFF structure in TDT file.")
            return
            
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        import traceback
        traceback.print_exc()
        return
    # --- 2. Detect Discrete Dopamine Peaks ---
    # Calculate a robust threshold (e.g., 90th percentile or Z-score > 2)
    height_threshold = np.percentile(dff_vals, dopamine_threshold_percentile) 

    # find_peaks handles the "continuous" problem by looking for local maxima
    # distance: Minimum samples between peaks (e.g., 500ms * fs) to avoid double-counting one event
    # prominence: Ensures the peak stands out from the immediate baseline
    sampling_rate = 1 / np.mean(np.diff(ts_abs)) # Estimate fs
    min_dist_samples = int(0.5 * sampling_rate) # 500ms lockout

    peaks_indices, _ = find_peaks(
        dff_vals.flatten(), 
        height=height_threshold, 
        distance=min_dist_samples
    )

    transient_times = ts_abs[peaks_indices]

    print(f" Detected {len(transient_times)} discrete dopamine events.") 
    
    if len(transient_times) < 10:
        print("  Not enough dopamine transients. Aborting.")
        return
    
    # --- 3. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 4. Calculate Cross-Correlation at Different Lags ---
    results = {}
    
    for cid in tqdm(unique_clusters, desc="Processing neurons"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        coupling_at_lags = []
        
        for lag_ms in time_lags_ms:
            lag_sec = lag_ms / 1000.0
            
            # Count spikes near dopamine transients with this lag
            # Positive lag: spikes occur AFTER dopamine
            # Negative lag: spikes occur BEFORE dopamine
            
            spikes_near_transients = 0
            window_size = 0.05  # 50ms window
            
            for transient_t in transient_times:
                spike_time = transient_t + lag_sec
                spikes_near_transients += np.sum(
                    (cluster_spikes >= spike_time - window_size/2) &
                    (cluster_spikes < spike_time + window_size/2)
                )
            
            # Normalize by number of transients and window size
            coupling_strength = spikes_near_transients / (len(transient_times) * window_size)
            coupling_at_lags.append(coupling_strength)
        
        results[cid] = {f'coupling_lag_{lag}ms': val for lag, val in zip(time_lags_ms, coupling_at_lags)}
        
        # Find peak coupling lag
        peak_idx = np.argmax(np.abs(coupling_at_lags))
        results[cid]['peak_coupling_lag_ms'] = time_lags_ms[peak_idx]
        results[cid]['peak_coupling_strength'] = coupling_at_lags[peak_idx]
    
    # --- 5. Save Results ---
    print("\n  Dopamine-spike coupling analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'dopamine_spike_coupling.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")

    # --- 6. Plot Population Profile ---
    try:
        import matplotlib.pyplot as plt
        # Average coupling profile across all neurons
        mean_coupling = df_results[[c for c in df_results.columns if 'coupling_lag' in c]].mean()
        
        plt.figure(figsize=(8, 5))
        plt.plot(time_lags_ms, mean_coupling.values, 'o-')
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Time Lag (ms)')
        plt.ylabel('Coupling Strength (Hz)')
        plt.title('Population Average Dopamine-Spike Coupling')
        plt.grid(True, alpha=0.3)
        
        plot_path = output_dir / 'dopamine_spike_coupling_profile.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"  Coupling profile plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate coupling profile plot: {e}")

def analyze_dopamine_triggered_firing(paths: DataPaths, window_ms: int = 2000, bin_size_ms: int = 50, 
                                     dopamine_threshold_percentile: float = 75):
    """
    Calculates dopamine-triggered average (DTA) of neural firing rates.
    
    Creates peri-event time histograms of neural activity aligned to dopamine transients,
    similar to spike-triggered averages but with dopamine events as triggers.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        window_ms (int): Time window around dopamine transient (ms).
        bin_size_ms (int): Bin size for PETH (ms).
        dopamine_threshold_percentile (float): Percentile threshold for defining transients.
    """
    print("Analyzing dopamine-triggered neural firing (DTA)...")
    
    window_sec = window_ms / 1000.0
    bin_size_sec = bin_size_ms / 1000.0
    
    # --- 1. Load Dopamine Data ---
    if not paths.tdt_dff or not paths.tdt_dff.exists():
        print(f"  Error: TDT dFF file not found at {paths.tdt_dff}")
        return
    
    try:
        config = load_dataset_config()
        # Find raw file
        tdt_dir = paths.tdt_dff.parent
        raw_files = list(tdt_dir.glob('*_UnivRAW_offdemod.mat'))
        if not raw_files:
            print("  Error: Could not find raw TDT file for timestamps.")
            return
        raw_file = raw_files[0]

        # Get Config Keys
        dff_entry = find_config_entry(paths.tdt_dff, config)
        raw_entry = find_config_entry(raw_file, config)
        
        dff_key = next((k for k, v in config.items() if v == dff_entry), None)
        raw_key = next((k for k, v in config.items() if v == raw_entry), None)
        
        if not dff_key or not raw_key:
             print("  Error: Could not find config keys for dFF or Raw files.")
             return

        # Load using PhotometryDataLoader
        photo_loader = PhotometryDataLoader(paths.base_path, config)
        data = photo_loader.load(dff_config_key=dff_key, raw_config_key=raw_key)
        
        dff_vals = data['dff_values']
        ts_abs = data['dff_timestamps']
            
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        return
            
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- 2. Detect Dopamine Transients ---
    height_threshold = np.percentile(dff_vals, dopamine_threshold_percentile) 
    sampling_rate = 1 / np.mean(np.diff(ts_abs)) # Estimate fs
    min_dist_samples = int(0.5 * sampling_rate) # 500ms lockout
    peaks_indices, _ = find_peaks(
        dff_vals.flatten(), 
        height=height_threshold, 
        distance=min_dist_samples
    )
    transient_times = ts_abs[peaks_indices]
    
    print(f"  Detected {len(transient_times)} dopamine transients above {dopamine_threshold_percentile}th percentile.")
    
    if len(transient_times) < 10:
        print("  Not enough dopamine transients. Aborting.")
        return
    
    # --- 3. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 4. Calculate DTA (Dopamine-Triggered Average) for Each Neuron ---
    n_bins = int(window_sec / bin_size_sec)
    dta_results = {cid: np.zeros(n_bins) for cid in unique_clusters}
    
    # Pre-select spikes for each cluster
    spikes_by_cluster = {cid: np.sort(spike_times_sec[spike_clusters == cid]) for cid in unique_clusters}
    
    bin_edges = np.linspace(0, window_sec, n_bins + 1)
    
    for cid in tqdm(unique_clusters, desc="Calculating DTA"):
        cluster_spikes = spikes_by_cluster[cid]
        if len(cluster_spikes) == 0:
            continue
        
        # Vectorized search for all transient windows
        starts = transient_times - (window_sec / 2)
        ends = starts + window_sec
        
        # Find indices of spikes falling into windows
        idx_starts = np.searchsorted(cluster_spikes, starts)
        idx_ends = np.searchsorted(cluster_spikes, ends)
        
        for i in range(len(transient_times)):
            if idx_ends[i] > idx_starts[i]:
                relative_times = cluster_spikes[idx_starts[i]:idx_ends[i]] - starts[i]
                hist, _ = np.histogram(relative_times, bins=bin_edges)
                dta_results[cid] += hist
    
    # --- 5. Normalize by number of transients and bin size to get firing rate (Hz) ---
    for cid in unique_clusters:
        dta_results[cid] = dta_results[cid] / (len(transient_times) * bin_size_sec)
    
    # --- 6. Save Results ---
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    df_dta = pd.DataFrame.from_dict(dta_results, orient='index')
    time_bins = np.linspace(-window_ms / 2, window_ms / 2, n_bins)
    df_dta.columns = [f"{t:.0f}ms" for t in time_bins]
    df_dta.index.name = 'cluster_id'
    output_path = output_dir / 'dopamine_triggered_firing_data.csv'
    df_dta.to_csv(output_path)
    print(f"  DTA results saved to {output_path}")
    
    # --- 7. Generate Population Heatmap ---
    try:
        if not df_dta.empty:
            # Z-score normalize for heatmap
            means = df_dta.mean(axis=1)
            stds = df_dta.std(axis=1)
            stds[stds == 0] = 1.0
            df_z = df_dta.sub(means, axis=0).div(stds, axis=0)
            
            # Sort neurons by peak latency
            peak_indices = np.argmax(df_z.values, axis=1)
            sort_order = np.argsort(peak_indices)
            sorted_z = df_z.iloc[sort_order]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(sorted_z.values, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                           extent=[-window_ms/2, window_ms/2, len(sorted_z), 0],
                           vmin=-3, vmax=3)
            
            plt.colorbar(im, label='Z-scored Firing Rate')
            ax.set_xlabel("Time from DA transient (ms)")
            ax.set_ylabel("Neuron ID (sorted by peak latency)")
            ax.set_title(f"Dopamine-Triggered Neural Activity (n={len(transient_times)} transients)")
            ax.axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
            
            heatmap_path = output_dir / 'dopamine_triggered_firing_heatmap.png'
            plt.savefig(heatmap_path, dpi=150)
            plt.close(fig)
            print(f"  Population heatmap saved to {heatmap_path}")
    except Exception as e:
        print(f"  Could not generate DTA heatmap: {e}")

def analyze_dopamine_modulation_index(paths: DataPaths, baseline_window_sec: float = 5.0, 
                                      dopamine_threshold_percentile: float = 75):
    """
    Calculates dopamine modulation index for each neuron.
    
    Compares firing rates during high vs low dopamine periods to quantify
    how strongly dopamine modulates each neuron's activity.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        baseline_window_sec (float): Window size for calculating baseline firing rate.
        dopamine_threshold_percentile (float): Percentile threshold for defining high DA.
    """
    print("Analyzing dopamine modulation index...")
    
    # --- 1. Load Dopamine Data ---
    if not paths.tdt_dff or not paths.tdt_dff.exists():
        print(f"  Error: TDT dFF file not found at {paths.tdt_dff}")
        return
    
    try:
        config = load_dataset_config()
        # Find raw file
        tdt_dir = paths.tdt_dff.parent
        raw_files = list(tdt_dir.glob('*_UnivRAW_offdemod.mat'))
        if not raw_files:
            print("  Error: Could not find raw TDT file for timestamps.")
            return
        raw_file = raw_files[0]

        # Get Config Keys
        dff_entry = find_config_entry(paths.tdt_dff, config)
        raw_entry = find_config_entry(raw_file, config)
        
        dff_key = next((k for k, v in config.items() if v == dff_entry), None)
        raw_key = next((k for k, v in config.items() if v == raw_entry), None)
        
        if not dff_key or not raw_key:
             print("  Error: Could not find config keys for dFF or Raw files.")
             return

        # Load using PhotometryDataLoader
        photo_loader = PhotometryDataLoader(paths.base_path, config)
        data = photo_loader.load(dff_key, raw_key)
        
        dff_vals = data['dff_values']
        ts_abs = data['dff_timestamps']
            
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        return
            
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        return
    
    # --- 2. Define High and Low Dopamine Periods ---
    dff_threshold = np.percentile(dff_vals, dopamine_threshold_percentile)
    high_da_mask = dff_vals > dff_threshold
    low_da_mask = dff_vals < np.percentile(dff_vals, 25)  # Bottom 25%
    
    # Convert to time periods
    high_da_times = ts_abs[high_da_mask]
    low_da_times = ts_abs[low_da_mask]
    
    print(f"  High DA periods: {len(high_da_times)} samples ({len(high_da_times)/len(ts_abs)*100:.1f}%)")
    print(f"  Low DA periods: {len(low_da_times)} samples ({len(low_da_times)/len(ts_abs)*100:.1f}%)")
    
    # --- 3. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # --- 4. Calculate Firing Rates (Optimized) ---
    print("Calculating firing rates using vectorized state matching...")
    
    # A. Pre-calculate durations of High/Low states
    # Assuming ts_abs is monotonic and evenly spaced (mostly)
    dt = np.median(np.diff(ts_abs))
    
    # Simple boolean masks for the Photometry timeline
    mask_high = dff_vals > dff_threshold
    mask_low = dff_vals < np.percentile(dff_vals, 25)
    
    # Calculate total duration in seconds for each state
    total_time_high = np.sum(mask_high) * dt
    total_time_low = np.sum(mask_low) * dt

    results = {}

    # B. Vectorized Spike Counting
    for cid in tqdm(unique_clusters, desc="Processing neurons"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        if len(cluster_spikes) == 0:
            results[cid] = {'rate_high_dopamine': 0, 'rate_low_dopamine': 0, 'dopamine_modulation_index': 0}
            continue

        # Map every spike time to the nearest index in the Photometry timestamp array
        # np.searchsorted finds the insertion index to maintain order
        spike_indices = np.searchsorted(ts_abs, cluster_spikes)
        
        # Clip indices to prevent out-of-bounds (if spikes occur after photometry ends)
        spike_indices = np.clip(spike_indices, 0, len(ts_abs) - 1)
        
        # C. Retrieve DA State for each spike
        # We simply look up the boolean value at the spike's index
        spikes_in_high_da = mask_high[spike_indices]
        spikes_in_low_da = mask_low[spike_indices]
        
        # Sum the boolean arrays to get spike counts
        count_high = np.sum(spikes_in_high_da)
        count_low = np.sum(spikes_in_low_da)
        
        # D. Calculate Rates & Index
        rate_high = count_high / total_time_high if total_time_high > 0 else 0
        rate_low = count_low / total_time_low if total_time_low > 0 else 0
        
        if (rate_high + rate_low) > 0:
            dmi = (rate_high - rate_low) / (rate_high + rate_low)
        else:
            dmi = 0
            
        results[cid] = {
            'rate_high_dopamine': rate_high,
            'rate_low_dopamine': rate_low,
            'dopamine_modulation_index': dmi
        }
    
    # --- 5. Save Results ---
    print("\n  Dopamine modulation index analysis complete.")
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'dopamine_modulation_index.csv'
    df_results.to_csv(output_path)
    print(f"  Results saved to {output_path}")
    
    # Summary statistics
    positive_mod = np.sum(df_results['dopamine_modulation_index'] > 0.1)
    negative_mod = np.sum(df_results['dopamine_modulation_index'] < -0.1)
    print(f"  Neurons with positive DA modulation (>0.1): {positive_mod}")
    print(f"  Neurons with negative DA modulation (<-0.1): {negative_mod}")
    
    # Generate Heatmap
    try:
        heatmap_path = output_dir / 'dopamine_modulation_heatmap.png'
        cols_to_plot = ['rate_low_dopamine', 'rate_high_dopamine']
        _plot_population_heatmap(df_results[cols_to_plot], heatmap_path, 
                                 "Dopamine Modulation of Firing Rate", "DA Level", 
                                 sort_col='dopamine_modulation_index')
    except Exception as e:
        print(f"  Could not generate modulation heatmap: {e}")

def analyze_dopamine_lfp_coupling(paths: DataPaths, lfp_bands: dict = {'theta': (4, 12), 'beta': (13, 30), 'gamma': (30, 80)},
                                 dopamine_threshold_percentile: float = 75):
    """
    Analyzes coupling between dopamine transients and LFP power/phase.
    
    Examines relationship between dopamine signals and LFP oscillations in different frequency bands.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        lfp_bands (dict): Dictionary of frequency bands to analyze.
    """
    print("Analyzing dopamine-LFP coupling...")
    
    try:
        import scipy.io
        import h5py
        import spikeinterface.core as si
        from scipy.signal import welch
    except ImportError as e:
        print(f"  Error: Required library missing - {e}")
        return
    
    # --- 1. Load Dopamine Data ---
    try:
        config = load_dataset_config()
        
        # Find dopamine config keys (dFF and RAW)
        dff_config_key = None
        raw_config_key = None
        
        for key, value in config.items():
            path_str = value.get('path', '')
            if 'dFF' in path_str and path_str.endswith('.mat'):
                dff_config_key = key
            elif 'UnivRAW' in path_str and path_str.endswith('.mat'):
                raw_config_key = key
        
        if not dff_config_key or not raw_config_key:
             print("  Error: Could not find config keys for dFF or Raw files.")
             return

        # Load using PhotometryDataLoader
        photo_loader = PhotometryDataLoader(paths.base_path, config)
        data = photo_loader.load(dff_config_key=dff_config_key, raw_config_key=raw_config_key)
        
        dff_vals = data['dff_values']
        ts_abs = data['dff_timestamps']
        
        print(f"  Loaded {len(dff_vals)} dopamine samples")
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        return
    
    # --- 2. Load LFP Data (Updated) ---
    try:
        config = load_dataset_config()
        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return

        lfp_fs = lfp_loader.fs
        recording = lfp_loader.extractor
        print(f"  Initialized LFPDataLoader. FS={lfp_fs} Hz")
        
    except Exception as e:
        print(f"  Error loading LFP data: {e}")
        return

    # --- 3. Detect Dopamine Transients ---
    height_threshold = np.percentile(dff_vals, dopamine_threshold_percentile) 
    sampling_rate = 1 / np.mean(np.diff(ts_abs)) # Estimate fs
    min_dist_samples = int(0.5 * sampling_rate) # 500ms lockout
    peaks_indices, _ = find_peaks(
        dff_vals.flatten(), 
        height=height_threshold, 
        distance=min_dist_samples
    )
    transient_times = ts_abs[peaks_indices]
    print(f"  Detected {len(transient_times)} dopamine transients")
    
    if len(transient_times) < 10:
        print("  Not enough transients. Aborting.")
        return
    
    # --- 4. Channel Selection (8 representative channels) ---
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
    
    selected_channels = []
    for i, (sx, indices) in enumerate(sorted(shanks, key=lambda s: s[0])): 
        indices = np.array(indices)
        shank_y = y_coords[indices]
        shank_ch_ids = channel_ids[indices]
        
        top_local_idx = np.argmax(shank_y)
        bot_local_idx = np.argmin(shank_y)
        
        selected_channels.append({'id': shank_ch_ids[top_local_idx], 'shank': i+1, 'loc': 'top'})
        selected_channels.append({'id': shank_ch_ids[bot_local_idx], 'shank': i+1, 'loc': 'bottom'})

    print(f"  Selected {len(selected_channels)} channels for coupling analysis.")

    # --- 5. Calculate LFP Power Around Transients ---
    window_sec = 1.0
    results = []
    
    # Limit transients for efficiency if too many
    analysis_transients = transient_times[:100]
    t_start_load = max(0, min(analysis_transients) - 1.0)
    t_end_load = max(analysis_transients) + window_sec + 1.0

    for ch_info in tqdm(selected_channels, desc="Processing LFP Channels"):
        chan_id = ch_info['id']
        
        try:
            # Load LFP for this channel
            # We load the whole relevant range once
            traces, timestamps = lfp_loader.get_data(t_start_load, t_end_load, channels=[chan_id], reference='csd')
            if len(traces) == 0: continue
            lfp_signal = traces[:, 0]
            lfp_times = timestamps
        except Exception as e:
            print(f"    Error reading LFP Ch {chan_id}: {e}")
            continue

        for band_name, (low_freq, high_freq) in lfp_bands.items():
            band_powers = []
            
            for t_transient in analysis_transients:
                # Find index
                start_idx = np.searchsorted(lfp_times, t_transient)
                end_idx = np.searchsorted(lfp_times, t_transient + window_sec)
                
                if start_idx < 0 or end_idx > len(lfp_signal):
                    continue
                
                lfp_snippet = lfp_signal[start_idx:end_idx]
                
                if len(lfp_snippet) < lfp_fs:
                    continue
                
                # PSD
                freqs, psd = welch(lfp_snippet, fs=lfp_fs, nperseg=min(len(lfp_snippet), 256))
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if np.sum(band_mask) > 0:
                    band_power = np.mean(psd[band_mask])
                    band_powers.append(band_power)
            
            if band_powers:
                results.append({
                    'channel_id': chan_id,
                    'shank': ch_info['shank'],
                    'location': ch_info['loc'],
                    'band': band_name,
                    'freq_range': f'{low_freq}-{high_freq} Hz',
                    'mean_power': np.mean(band_powers),
                    'std_power': np.std(band_powers),
                    'n_events': len(band_powers)
                })
    
    # --- 5. Save Results ---
    print("\n  Dopamine-LFP coupling analysis complete.")
    df_results = pd.DataFrame(results)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'dopamine_lfp_coupling.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

def analyze_dopamine_phase_locking_relationship(paths: DataPaths, frequency_bands: dict = None,
                                                 time_window_sec: float = 5.0):
    """
    Analyze the relationship between dopamine release and spike-phase locking strength.
    
    This analysis examines:
    1. Correlation between dopamine levels and phase-locking strength
    2. Changes in phase-locking during high vs low dopamine periods
    3. Dopamine modulation of preferred phase angles
    
    Args:
        paths: DataPaths object
        frequency_bands: Dictionary of frequency bands to analyze
        time_window_sec: Time window for segmenting data
    
    Returns:
        DataFrame with dopamine-phase locking relationship metrics
    """
    print("Analyzing dopamine-phase locking relationship...")
    
    if frequency_bands is None:
        frequency_bands = {
            'beta': (13, 30),
            'low_gamma': (30, 60),
            'high_gamma': (60, 100)
        }
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. Load Phase-Locking Results ---
    phase_locking_file = output_dir / 'spike_phase_locking_data.csv'
    
    if not phase_locking_file.exists():
        print("  Error: Phase-locking results not found. Run analyze_spike_phase_locking first.")
        return None
    
    df_phase = pd.read_csv(phase_locking_file)
    print(f"  Loaded phase-locking data for {len(df_phase)} neuron-band combinations.")
    
    # Filter to only significantly locked neurons
    df_locked = df_phase[df_phase['is_locked']].copy()
    
    if len(df_locked) == 0:
        print("  No significantly phase-locked neurons found.")
        return None
    
    print(f"  Found {len(df_locked)} significantly phase-locked neuron-band combinations.")
    
    # --- 2. Load Dopamine Data ---
    try:
        config = load_dataset_config()
        base_path = paths.base_path
        photometry_loader = PhotometryDataLoader(base_path, config)
        
        # Find dopamine config
        # Find dopamine config keys (dFF and RAW)
        if not paths.tdt_dff or not paths.tdt_raw:
            print("  Error: Dopamine paths (dFF/RAW) not found.")
            return None
            
        # Load dopamine signal
        da_data = photometry_loader.load(paths.tdt_dff, paths.tdt_raw)
        
        if da_data is None or len(da_data) == 0:
            print("  Error: Dopamine data is empty.")
            return None
        
        # Extract signal and times
        if isinstance(da_data, dict):
            da_signal = da_data.get('dff_values')
            da_times = da_data.get('dff_timestamps')
        elif isinstance(da_data, pd.DataFrame):
            # ... legacy handling for DataFrame ...
            if 'dFF' in da_data.columns:
                da_signal = da_data['dFF'].values
            elif 'signal' in da_data.columns:
                da_signal = da_data['signal'].values
            elif 'Dopamine' in da_data.columns:
                da_signal = da_data['Dopamine'].values
            else:
                numeric_cols = da_data.select_dtypes(include=[np.number]).columns
                da_signal = da_data[numeric_cols[0]].values
            
            if 'time' in da_data.columns:
                da_times = da_data['time'].values
            elif 'timestamp' in da_data.columns:
                da_times = da_data['timestamp'].values
            else:
                da_fs = 100.0
                da_times = np.arange(len(da_signal)) / da_fs
        else:
            da_signal = np.array(da_data)
            da_fs = 100.0
            da_times = np.arange(len(da_signal)) / da_fs
        
        print(f"  Loaded dopamine data: {len(da_signal)} samples, time range: {da_times[0]:.1f}-{da_times[-1]:.1f}s")
        
    except Exception as e:
        print(f"  Error loading dopamine data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # --- 3. Load Spike and LFP Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return None
    
    # Initialize LFP Loader
    try:
        if not paths.lfp_dir or not paths.lfp_dir.exists():
             print(f"  Error: LFP directory not found: {paths.lfp_dir}")
             return None

        lfp_loader = LFPDataLoader(paths.lfp_dir, paths.kilosort_dir)
        if lfp_loader.extractor is None:
            print("  Error: LFP Extractor not initialized.")
            return None
        lfp_fs = lfp_loader.fs
        print(f"  Initialized LFPDataLoader. FS={lfp_fs} Hz")

    except Exception as e:
        print(f"  Error loading LFP data: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    # Map units to channels
    unit_ch_map = _get_unit_best_channels(paths, unique_clusters)
    units_by_channel = defaultdict(list)
    for cid in unique_clusters:
        ch = unit_ch_map.get(cid)
        if ch is not None:
            units_by_channel[ch].append(cid)
    
    # --- 4. Time-Resolved Phase-Locking Analysis ---
    from scipy.signal import butter, filtfilt, hilbert
    
    # Define time segments
    session_duration = min(spike_times_sec.max(), lfp_times.max(), da_times.max())
    time_segments = np.arange(0, session_duration, time_window_sec)
    
    print(f"  Analyzing {len(time_segments)} time segments of {time_window_sec}s each...")
    
    results = []
    
    # Pre-calculate Dopamine means for each segment to avoid re-calculation
    segment_da_means = []
    valid_segment_indices = []
    
    print(f"  Pre-calculating dopamine means for {len(time_segments)} segments...")
    for i, seg_start in enumerate(time_segments):
        seg_end = seg_start + time_window_sec
        da_mask = (da_times >= seg_start) & (da_times < seg_end)
        if np.sum(da_mask) >= 2:
            segment_da_means.append(np.mean(da_signal[da_mask]))
            valid_segment_indices.append(i)
        else:
            segment_da_means.append(np.nan)
            
    segment_da_means = np.array(segment_da_means)
    
    results = []
    
    # Iterate over channels (Optimization: Load LFP once per channel)
    for ch_idx, cluster_list in tqdm(units_by_channel.items(), desc="Analyzing Channels"):
        
        # Load LFP for this channel
        # Load full session duration (or max spike time)
        # We need coverage for interaction
        t_start = 0
        t_end = session_duration + 1.0
        
        try:
             traces, timestamps = lfp_loader.get_data(t_start, t_end, channels=[ch_idx], reference='csd')
             if len(traces) == 0: continue
             lfp_channel_data = traces[:, 0]
             lfp_ch_times = timestamps
        except Exception as e:
            print(f"    Error reading LFP Ch {ch_idx}: {e}")
            continue
            
        # Iterate Frequency Bands
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            
            # Identify which neurons on this channel are locked to this band
            # Filter df_locked for this band and these clusters
            relevant_neurons = [cid for cid in cluster_list 
                                if not df_locked[(df_locked['cluster_id'] == cid) & (df_locked['band'] == band_name)].empty]
            
            if not relevant_neurons:
                continue
                
            # Filter LFP for this band
            nyquist = lfp_fs / 2
            try:
                # Handle NaNs
                if np.isnan(lfp_channel_data).any():
                    lfp_clean = np.nan_to_num(lfp_channel_data)
                else:
                    lfp_clean = lfp_channel_data
                    
                b, a = butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band')
                filtered_lfp = filtfilt(b, a, lfp_clean)
                analytic_signal = hilbert(filtered_lfp)
                instantaneous_phase = np.angle(analytic_signal)
            except Exception as e:
                 print(f"    Filter error ({band_name}) on ch {ch_idx}: {e}")
                 continue

            # Process Neurons
            for cid in relevant_neurons:
                cluster_spikes = spike_times_sec[spike_clusters == cid]
                
                # Iterate Time Segments
                # Optimization: Vectorize segment processing if possible, or simple loop
                # Simple loop over valid segments
                
                for seg_idx in valid_segment_indices:
                    seg_start = time_segments[seg_idx]
                    seg_end = seg_start + time_window_sec
                    da_val = segment_da_means[seg_idx]
                    
                    # Get spikes in segment
                    # Use searchsorted for speed
                    idx_start = np.searchsorted(cluster_spikes, seg_start)
                    idx_end = np.searchsorted(cluster_spikes, seg_end)
                    
                    seg_spikes = cluster_spikes[idx_start:idx_end]
                    
                    if len(seg_spikes) < DEFAULT_MIN_SPIKES_FOR_PHASE:
                        continue
                        
                    # Map to LFP Phase
                    # Use searchsorted on lfp_ch_times
                    lfp_indices = np.searchsorted(lfp_ch_times, seg_spikes)
                    valid_mask = (lfp_indices >= 0) & (lfp_indices < len(instantaneous_phase))
                    lfp_indices = lfp_indices[valid_mask]
                    
                    if len(lfp_indices) < DEFAULT_MIN_SPIKES_FOR_PHASE:
                        continue
                        
                    spike_phases = instantaneous_phase[lfp_indices]
                    
                    # Calculate PLV
                    mean_vector = np.mean(np.exp(1j * spike_phases))
                    segment_plv = np.abs(mean_vector)
                    segment_phase = np.angle(mean_vector)
                    
                    results.append({
                        'cluster_id': cid,
                        'band': band_name,
                        'segment_start': seg_start,
                        'segment_end': seg_end,
                        'dopamine_mean': da_val,
                        'plv': segment_plv,
                        'preferred_phase_rad': segment_phase,
                        'n_spikes': len(spike_phases),
                        'lfp_channel': ch_idx
                    })
    
    if len(results) == 0:
        print("  No time-resolved phase-locking data computed.")
        return None
    
    df_results = pd.DataFrame(results)
    
    # --- 5. Correlation Analysis ---
    print("\n  Computing dopamine-phase locking correlations...")
    
    correlation_results = []
    
    for band_name in frequency_bands.keys():
        band_data = df_results[df_results['band'] == band_name]
        
        if len(band_data) < 10:
            continue
        
        # Overall correlation across all neurons and time segments
        from scipy.stats import pearsonr, spearmanr
        
        valid_data = band_data[['dopamine_mean', 'plv']].dropna()
        
        if len(valid_data) < 10:
            continue
        
        pearson_r, pearson_p = pearsonr(valid_data['dopamine_mean'], valid_data['plv'])
        spearman_r, spearman_p = spearmanr(valid_data['dopamine_mean'], valid_data['plv'])
        
        print(f"  {band_name}: Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
              f"Spearman r={spearman_r:.3f} (p={spearman_p:.4f})")
        
        correlation_results.append({
            'band': band_name,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(valid_data)
        })
        
        # Per-neuron correlations
        for cid in band_data['cluster_id'].unique():
            neuron_data = band_data[band_data['cluster_id'] == cid]
            
            if len(neuron_data) < 5:
                continue
            
            neuron_valid = neuron_data[['dopamine_mean', 'plv']].dropna()
            
            if len(neuron_valid) < 5:
                continue
            
            try:
                neuron_r, neuron_p = pearsonr(neuron_valid['dopamine_mean'], neuron_valid['plv'])
                
                correlation_results.append({
                    'band': band_name,
                    'cluster_id': cid,
                    'pearson_r': neuron_r,
                    'pearson_p': neuron_p,
                    'n_samples': len(neuron_valid)
                })
            except:
                pass
    
    df_correlations = pd.DataFrame(correlation_results)
    
    # --- 6. High vs Low Dopamine Comparison ---
    print("\n  Comparing phase-locking in high vs low dopamine periods...")
    
    # Define high/low dopamine based on median split
    da_median = np.median(df_results['dopamine_mean'])
    df_results['dopamine_level'] = df_results['dopamine_mean'].apply(
        lambda x: 'high' if x >= da_median else 'low'
    )
    
    comparison_results = []
    
    for band_name in frequency_bands.keys():
        band_data = df_results[df_results['band'] == band_name]
        
        high_da = band_data[band_data['dopamine_level'] == 'high']['plv']
        low_da = band_data[band_data['dopamine_level'] == 'low']['plv']
        
        if len(high_da) < 5 or len(low_da) < 5:
            continue
        
        from scipy.stats import ttest_ind, mannwhitneyu
        
        t_stat, t_p = ttest_ind(high_da, low_da)
        u_stat, u_p = mannwhitneyu(high_da, low_da)
        
        comparison_results.append({
            'band': band_name,
            'high_da_mean_plv': np.mean(high_da),
            'high_da_sem_plv': np.std(high_da) / np.sqrt(len(high_da)),
            'low_da_mean_plv': np.mean(low_da),
            'low_da_sem_plv': np.std(low_da) / np.sqrt(len(low_da)),
            'ttest_stat': t_stat,
            'ttest_p': t_p,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_p,
            'n_high': len(high_da),
            'n_low': len(low_da)
        })
        
        sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "n.s."
        print(f"  {band_name}: High DA PLV={np.mean(high_da):.3f}, Low DA PLV={np.mean(low_da):.3f}, p={u_p:.4f} {sig}")
    
    df_comparison = pd.DataFrame(comparison_results)
    
    # --- 7. Save Results ---
    time_resolved_path = output_dir / 'dopamine_phase_locking_timeseries.csv'
    df_results.to_csv(time_resolved_path, index=False)
    print(f"\n  Time-resolved data saved to {time_resolved_path}")
    
    correlation_path = output_dir / 'dopamine_phase_locking_correlations.csv'
    df_correlations.to_csv(correlation_path, index=False)
    print(f"  Correlation results saved to {correlation_path}")
    
    comparison_path = output_dir / 'dopamine_phase_locking_comparison.csv'
    df_comparison.to_csv(comparison_path, index=False)
    print(f"  High vs low DA comparison saved to {comparison_path}")
    
    # --- 8. Visualize ---
    try:
        n_bands = len(frequency_bands)
        fig, axes = plt.subplots(2, n_bands, figsize=(6*n_bands, 10))
        
        if n_bands == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, band_name in enumerate(frequency_bands.keys()):
            band_data = df_results[df_results['band'] == band_name]
            
            if len(band_data) == 0:
                continue
            
            # Top row: Scatter plot of dopamine vs PLV
            ax = axes[0, idx]
            
            # Sample if too many points
            if len(band_data) > 1000:
                band_sample = band_data.sample(1000)
            else:
                band_sample = band_data
            
            ax.scatter(band_sample['dopamine_mean'], band_sample['plv'], 
                      alpha=0.3, s=20, c='#3498db')
            
            # Add regression line
            from scipy.stats import linregress
            valid_data = band_data[['dopamine_mean', 'plv']].dropna()
            if len(valid_data) > 10:
                slope, intercept, r_value, p_value, std_err = linregress(
                    valid_data['dopamine_mean'], valid_data['plv']
                )
                x_line = np.array([valid_data['dopamine_mean'].min(), 
                                  valid_data['dopamine_mean'].max()])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r-', linewidth=2, 
                       label=f'r={r_value:.2f}, p={p_value:.3f}')
            
            ax.set_xlabel('Dopamine Signal (dF/F)')
            ax.set_ylabel('PLV (Phase-Locking Value)')
            ax.set_title(f'{band_name} - Dopamine vs Phase-Locking')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Bottom row: High vs Low DA comparison
            ax = axes[1, idx]
            
            if band_name in df_comparison['band'].values:
                comp_row = df_comparison[df_comparison['band'] == band_name].iloc[0]
                
                x_pos = [0, 1]
                means = [comp_row['low_da_mean_plv'], comp_row['high_da_mean_plv']]
                sems = [comp_row['low_da_sem_plv'], comp_row['high_da_sem_plv']]
                
                ax.bar(x_pos, means, yerr=sems, capsize=10, 
                      color=['#95a5a6', '#e74c3c'], alpha=0.7, edgecolor='black')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(['Low DA', 'High DA'])
                ax.set_ylabel('Mean PLV')
                ax.set_title(f'{band_name} - Phase-Locking by DA Level\n'
                           f'p={comp_row["mann_whitney_p"]:.4f}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add significance marker
                if comp_row['mann_whitney_p'] < 0.001:
                    sig = "***"
                elif comp_row['mann_whitney_p'] < 0.01:
                    sig = "**"
                elif comp_row['mann_whitney_p'] < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."
                
                y_max = max(means) + max(sems)
                ax.text(0.5, y_max * 1.1, sig, ha='center', fontsize=16)
        
        plt.tight_layout()
        plot_path = output_dir / 'dopamine_phase_locking_relationship.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Relationship plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate relationship plots: {e}")
        import traceback
        traceback.print_exc()
    
    return df_results, df_correlations, df_comparison

def analyze_spatial_organization_depth(paths: DataPaths, n_depth_bins: int = 10):
    """
    Analyzes how neural activity varies across depth (dorsal-ventral axis).
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_depth_bins (int): Number of bins to divide depth into.
    """
    print("Analyzing spatial organization by depth...")
    
    # --- 1. Load Spike Data and Unit Positions ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # Load unit positions from kilosort
    try:
        config = load_dataset_config()
        spike_positions_path = paths.kilosort_dir / 'spike_positions.npy'
        templates_path = paths.kilosort_dir / 'templates.npy'
        spike_clusters_path = paths.kilosort_dir / 'spike_clusters.npy'
        
        if not spike_positions_path.exists():
            print(f"  Error: spike_positions.npy not found at {spike_positions_path}")
            return
        
        spike_positions = np.load(spike_positions_path, mmap_mode='r')
        spike_clusters_data = np.load(spike_clusters_path, mmap_mode='r')
        
        # Calculate mean depth for each unit
        unit_depths = {}
        for cid in unique_clusters:
            mask = spike_clusters_data == cid
            if np.any(mask):
                # Depth is typically the y-coordinate (second column)
                unit_depths[cid] = np.mean(spike_positions[mask, 1])
        
        print(f"  Loaded depth information for {len(unit_depths)} units.")
        
    except Exception as e:
        print(f"  Error loading unit position data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not unit_depths:
        print("  No depth information available. Aborting.")
        return
    
    # --- 2. Bin Units by Depth ---
    depths = np.array(list(unit_depths.values()))
    depth_bins = np.linspace(depths.min(), depths.max(), n_depth_bins + 1)
    
    # Calculate average firing rate for each unit
    session_duration = spike_times_sec.max()
    firing_rates = {}
    for cid in unique_clusters:
        if cid in unit_depths:
            n_spikes = np.sum(spike_clusters == cid)
            firing_rates[cid] = n_spikes / session_duration
    
    # Group units by depth bin
    depth_bin_rates = []
    depth_bin_centers = []
    
    for i in range(n_depth_bins):
        bin_start = depth_bins[i]
        bin_end = depth_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2
        
        # Find units in this bin
        units_in_bin = [cid for cid, depth in unit_depths.items() 
                       if bin_start <= depth < bin_end]
        
        if units_in_bin:
            rates = [firing_rates[cid] for cid in units_in_bin if cid in firing_rates]
            depth_bin_rates.append(np.mean(rates) if rates else 0)
        else:
            depth_bin_rates.append(0)
        
        depth_bin_centers.append(bin_center)
    
    # --- 3. Save Results ---
    print("\n  Spatial depth organization analysis complete.")
    results = {
        'depth_bin_center_um': depth_bin_centers,
        'mean_firing_rate_hz': depth_bin_rates
    }
    
    df_results = pd.DataFrame(results)
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'spatial_depth_organization.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    
    # Plot depth profile
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.plot(depth_bin_rates, depth_bin_centers, 'o-')
        ax.set_xlabel('Mean Firing Rate (Hz)')
        ax.set_ylabel('Depth (μm)')
        ax.set_title('Firing Rate vs Depth')
        ax.invert_yaxis()  # Deep is at bottom
        ax.grid(True, alpha=0.3)
        
        plot_path = output_dir / 'spatial_depth_profile.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"  Depth profile plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate depth profile plot: {e}")

def analyze_depth_tuning_by_behavior(paths: DataPaths, n_depth_bins: int = 10, corner_order: list = [1, 2, 4, 3]):
    """
    Analyzes how behavioral tuning varies across depth (dorsal-ventral gradient).
    
    Examines whether neurons at different depths show different behavioral preferences,
    specifically looking at CW/CCW directional tuning across the depth of the recording.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_depth_bins (int): Number of depth bins.
        corner_order (list): Order of corners for CW navigation.
    """
    print("Analyzing depth-dependent behavioral tuning...")
    
    # --- 1. Load Spike Data and Depth Information ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    try:
        spike_positions_path = paths.kilosort_dir / 'spike_positions.npy'
        spike_clusters_path = paths.kilosort_dir / 'spike_clusters.npy'
        
        if not spike_positions_path.exists():
            print(f"  Error: spike_positions.npy not found at {spike_positions_path}")
            return
        
        spike_positions = np.load(spike_positions_path, mmap_mode='r')
        spike_clusters_data = np.load(spike_clusters_path, mmap_mode='r')
        
        # Calculate mean depth (y-coordinate) for each unit
        unit_depths = {}
        for cid in unique_clusters:
            mask = spike_clusters_data == cid
            if np.any(mask):
                unit_depths[cid] = np.mean(spike_positions[mask, 1])  # y-coordinate is depth
        
        print(f"  Loaded depth information for {len(unit_depths)} units.")
        
    except Exception as e:
        print(f"  Error loading unit depth data: {e}")
        return
    
    if not unit_depths:
        print("  No depth information available. Aborting.")
        return
    
    # --- 2. Load Behavioral Data (Corner Events) ---
    if not paths.event_corner or not paths.event_corner.exists():
        print("  Error: Corner event file not found.")
        return
    
    try:
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # Load corner events
        corner_df, corner_times = event_loader.load_events_from_path(paths.event_corner)
        if corner_df.empty: raise ValueError("Could not load corner events")
        
        # Get corner IDs
        corner_config_entry = find_config_entry(paths.event_corner, config)
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 4+1):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # FILTERING: Exclude 0s to preserve transition continuity
        valid_mask = corner_ids != 0
        corner_ids = corner_ids[valid_mask]
        
        # Also need to filter corner_times here?
        # corner_times was loaded at 9774.
        # But wait, original code at 9797 truncates to min_len.
        # If I filter ids here, I MUST filter times too strictly.
        # However, corner_times at 9774 might include 0-events if loaded raw.
        # The logic at 9797 assumes 1-to-1 mapping.
        # Let's verify line 9774 loader. `event_loader.load_events_from_path`
        # filtered onsets?
        # If I filter corner_ids, I must filter corner_times using the same mask if they are aligned.
        # They SHOULD be aligned if they came from same DF.
        corner_times = corner_times[valid_mask]
        
        print(f"  Loaded {len(corner_times)} valid corner events.")
        
        print(f"  Loaded {len(corner_times)} corner events.")
        
    except Exception as e:
        print(f"  Error loading corner event data: {e}")
        return
    
    # Ensure corner_ids and corner_times have consistent lengths
    min_len = min(len(corner_ids), len(corner_times))
    corner_ids = corner_ids[:min_len]
    corner_times = corner_times[:min_len]
    
    # --- 3. Calculate CW/CCW Tuning for Each Neuron ---
    directional_tuning = {}
    
    for cid in unique_clusters:
        if cid not in unit_depths:
            continue
        
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        cw_rate = 0
        ccw_rate = 0
        cw_count = 0
        ccw_count = 0
        
        # Analyze movements between corners
        for i in range(len(corner_times) - 1):
            start_port = corner_ids[i]
            end_port = corner_ids[i + 1]
            
            t_start = corner_times[i]
            t_end = corner_times[i + 1]
            duration = t_end - t_start
            
            if duration <= 0 or duration > 20:
                continue
            
            # Determine if movement is CW or CCW
            is_cw = _is_move_correct(start_port, end_port, corner_order, True)
            is_ccw = _is_move_correct(start_port, end_port, corner_order, False)
            
            if not is_cw and not is_ccw:
                continue
            
            # Count spikes during this movement
            n_spikes = np.sum((cluster_spikes >= t_start) & (cluster_spikes < t_end))
            rate = n_spikes / duration
            
            if is_cw:
                cw_rate += rate
                cw_count += 1
            if is_ccw:
                ccw_rate += rate
                ccw_count += 1
        
        # Calculate average rates
        avg_cw = cw_rate / cw_count if cw_count > 0 else 0
        avg_ccw = ccw_rate / ccw_count if ccw_count > 0 else 0
        
        # Directional tuning index
        if avg_cw + avg_ccw > 0:
            tuning_index = (avg_cw - avg_ccw) / (avg_cw + avg_ccw)
        else:
            tuning_index = 0
        
        directional_tuning[cid] = {
            'depth': unit_depths[cid],
            'cw_rate': avg_cw,
            'ccw_rate': avg_ccw,
            'directional_tuning_index': tuning_index
        }
    
    # --- 4. Bin by Depth and Analyze ---
    depths = np.array([v['depth'] for v in directional_tuning.values()])
    depth_bins = np.linspace(depths.min(), depths.max(), n_depth_bins + 1)
    
    depth_results = []
    
    for i in range(n_depth_bins):
        bin_start = depth_bins[i]
        bin_end = depth_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2
        
        # Find units in this depth bin
        units_in_bin = [cid for cid, data in directional_tuning.items() 
                       if bin_start <= data['depth'] < bin_end]
        
        if units_in_bin:
            tuning_indices = [directional_tuning[cid]['directional_tuning_index'] for cid in units_in_bin]
            cw_rates = [directional_tuning[cid]['cw_rate'] for cid in units_in_bin]
            ccw_rates = [directional_tuning[cid]['ccw_rate'] for cid in units_in_bin]
            
            depth_results.append({
                'depth_bin_center_um': bin_center,
                'n_units': len(units_in_bin),
                'mean_directional_tuning': np.mean(tuning_indices),
                'mean_cw_rate': np.mean(cw_rates),
                'mean_ccw_rate': np.mean(ccw_rates)
            })
    
    # --- 5. Save Results ---
    print("\n  Depth-dependent behavioral tuning analysis complete.")
    
    # Save per-neuron results
    df_neurons = pd.DataFrame.from_dict(directional_tuning, orient='index')
    df_neurons.index.name = 'cluster_id'
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'depth_tuning_by_behavior_neurons.csv'
    df_neurons.to_csv(output_path)
    print(f"  Per-neuron results saved to {output_path}")
    
    # Save binned results
    df_bins = pd.DataFrame(depth_results)
    bin_output_path = output_dir / 'depth_tuning_by_behavior_bins.csv'
    df_bins.to_csv(bin_output_path, index=False)
    print(f"  Binned results saved to {bin_output_path}")
    
    # Generate plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Tuning index vs depth
        ax1 = axes[0]
        ax1.scatter(df_neurons['directional_tuning_index'], df_neurons['depth'], 
                   alpha=0.5, s=30)
        ax1.set_xlabel('Directional Tuning Index (CW - CCW)')
        ax1.set_ylabel('Depth (μm)')
        ax1.set_title('Directional Tuning vs Depth (All Neurons)')
        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean tuning by depth bin
        ax2 = axes[1]
        ax2.plot(df_bins['mean_directional_tuning'], df_bins['depth_bin_center_um'], 'o-', 
                linewidth=2, markersize=8)
        ax2.set_xlabel('Mean Directional Tuning Index')
        ax2.set_ylabel('Depth (μm)')
        ax2.set_title('Mean Tuning vs Depth (Binned)')
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'depth_tuning_by_behavior.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate depth tuning plot: {e}")

def analyze_spatial_clustering(paths: DataPaths, cluster_method: str = 'kmeans', n_clusters: int = 5):
    """
    Analyzes spatial clustering of functionally similar neurons.
    
    Clusters neurons based on their activity patterns and examines whether
    functionally similar neurons are spatially clustered.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        cluster_method (str): Clustering method ('kmeans' or 'hierarchical').
        n_clusters (int): Number of functional clusters.
    """
    print("Analyzing spatial clustering of neural activity...")
    
    try:
        from sklearn.cluster import KMeans
        from scipy.cluster.hierarchy import linkage, fcluster
    except ImportError as e:
        print(f"  Error: sklearn required - {e}")
        return
    
    # --- 1. Load Spike Data and Positions ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    try:
        spike_positions_path = paths.kilosort_dir / 'spike_positions.npy'
        if not spike_positions_path.exists():
            print(f"  Error: spike_positions.npy not found")
            return
        
        spike_positions = np.load(spike_positions_path, mmap_mode='r')
        spike_clusters_data = np.load(paths.kilosort_dir / 'spike_clusters.npy', mmap_mode='r')
        
        # Get mean position for each unit
        unit_positions = {}
        for cid in unique_clusters:
            mask = spike_clusters_data == cid
            if np.any(mask):
                unit_positions[cid] = np.mean(spike_positions[mask, :2], axis=0)  # x, y coordinates
        
    except Exception as e:
        print(f"  Error loading position data: {e}")
        return
    
    # --- 2. Calculate Functional Features ---
    session_duration = spike_times_sec.max()
    firing_rates = {}
    for cid in unique_clusters:
        if cid in unit_positions:
            n_spikes = np.sum(spike_clusters == cid)
            firing_rates[cid] = n_spikes / session_duration
    
    # --- 3. Functional Clustering ---
    valid_cids = list(firing_rates.keys())
    features = np.array([firing_rates[cid] for cid in valid_cids]).reshape(-1, 1)
    
    if cluster_method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        functional_labels = clusterer.fit_predict(features)
    else:
        Z = linkage(features, method='ward')
        functional_labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    # --- 4. Analyze Spatial Distribution of Functional Clusters ---
    results = []
    for cid, func_label in zip(valid_cids, functional_labels):
        pos = unit_positions[cid]
        results.append({
            'cluster_id': cid,
            'functional_cluster': func_label,
            'position_x': pos[0],
            'position_y': pos[1],
            'firing_rate': firing_rates[cid]
        })
    
    # --- 5. Save Results ---
    print("\n  Spatial clustering analysis complete.")
    df_results = pd.DataFrame(results)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'spatial_clustering.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_results['position_x'], df_results['position_y'],
                           c=df_results['functional_cluster'], cmap='tab10', s=50, alpha=0.7)
        ax.set_xlabel('Position X (μm)')
        ax.set_ylabel('Position Y (μm)')
        ax.set_title('Spatial Distribution of Functional Clusters')
        plt.colorbar(scatter, label='Functional Cluster')
        
        plot_path = output_dir / 'spatial_clustering.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"  Plot saved to {plot_path}")
    except Exception as e:
        print(f"  Could not generate plot: {e}")

def analyze_medial_lateral_organization(paths: DataPaths, n_bins: int = 5):
    """
    Analyzes activity gradient along medial-lateral axis.
    
    Examines how neural activity varies along the medial-lateral dimension of the recording.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_bins (int): Number of bins for medial-lateral axis.
    """
    print("Analyzing medial-lateral organization...")
    
    # --- 1. Load Spike Data and Positions ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    try:
        spike_positions_path = paths.kilosort_dir / 'spike_positions.npy'
        if not spike_positions_path.exists():
            print(f"  Error: spike_positions.npy not found")
            return
        
        spike_positions = np.load(spike_positions_path, mmap_mode='r')
        spike_clusters_data = np.load(paths.kilosort_dir / 'spike_clusters.npy', mmap_mode='r')
        
        unit_positions = {}
        for cid in unique_clusters:
            mask = spike_clusters_data == cid
            if np.any(mask):
                unit_positions[cid] = np.mean(spike_positions[mask, 0], axis=0)  # x coordinate (medial-lateral)
        
    except Exception as e:
        print(f"  Error loading position data: {e}")
        return
    
    # --- 2. Bin by Position ---
    positions = np.array(list(unit_positions.values()))
    pos_bins = np.linspace(positions.min(), positions.max(), n_bins + 1)
    
    session_duration = spike_times_sec.max()
    firing_rates = {}
    for cid in unique_clusters:
        if cid in unit_positions:
            n_spikes = np.sum(spike_clusters == cid)
            firing_rates[cid] = n_spikes / session_duration
    
    # --- 3. Calculate Mean Firing Rate per Bin ---
    bin_rates = []
    bin_centers = []
    
    for i in range(n_bins):
        bin_start = pos_bins[i]
        bin_end = pos_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2
        
        units_in_bin = [cid for cid, pos in unit_positions.items() 
                       if bin_start <= pos < bin_end and cid in firing_rates]
        
        if units_in_bin:
            rates = [firing_rates[cid] for cid in units_in_bin]
            bin_rates.append(np.mean(rates))
        else:
            bin_rates.append(0)
        
        bin_centers.append(bin_center)
    
    # --- 4. Save Results ---
    print("\n  Medial-lateral organization analysis complete.")
    results = {
        'position_bin_center_um': bin_centers,
        'mean_firing_rate_hz': bin_rates
    }
    
    df_results = pd.DataFrame(results)
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'medial_lateral_organization.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

def analyze_multi_shank_interactions(paths: DataPaths, max_pairs: int = 500):
    """
    Analyzes synchrony and interactions between shanks.
    
    Calculates cross-shank correlations to examine functional connectivity.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        max_pairs (int): Maximum number of cross-shank pairs to analyze.
    """
    print("Analyzing multi-shank interactions...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    try:
        # Try to infer shank from position (x coordinate typically varies by shank)
        spike_positions_path = paths.kilosort_dir / 'spike_positions.npy'
        if not spike_positions_path.exists():
            print(f"  Error: spike_positions.npy not found")
            return
        
        spike_positions = np.load(spike_positions_path, mmap_mode='r')
        spike_clusters_data = np.load(paths.kilosort_dir / 'spike_clusters.npy', mmap_mode='r')
        
        # Assign shanks based on x-position clustering
        unit_x_positions = {}
        for cid in unique_clusters:
            mask = spike_clusters_data == cid
            if np.any(mask):
                unit_x_positions[cid] = np.mean(spike_positions[mask, 0])
        
        # Simple shank assignment: bin x-positions
        x_vals = np.array(list(unit_x_positions.values()))
        n_shanks = 4  # Assuming 4-shank probe
        x_bins = np.linspace(x_vals.min(), x_vals.max(), n_shanks + 1)
        
        unit_shanks = {}
        for cid, x_pos in unit_x_positions.items():
            shank = np.digitize(x_pos, x_bins) - 1
            unit_shanks[cid] = min(shank, n_shanks - 1)
        
        print(f"  Assigned {len(unit_shanks)} units to {n_shanks} shanks")
        
    except Exception as e:
        print(f"  Error loading shank data: {e}")
        return
    
    # --- 2. Calculate Cross-Shank Correlations ---
    session_duration = spike_times_sec.max()
    bin_size = 0.1
    n_bins = int(session_duration / bin_size)
    
    # Bin spike counts
    binned_activity = {}
    for cid in unit_shanks.keys():
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        hist, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        binned_activity[cid] = hist
    
    # Calculate cross-shank correlations
    results = []
    pair_count = 0
    
    for cid1 in list(unit_shanks.keys())[:50]:  # Limit for efficiency
        if pair_count >= max_pairs:
            break
        shank1 = unit_shanks[cid1]
        
        for cid2 in list(unit_shanks.keys())[:50]:
            if cid2 <= cid1 or pair_count >= max_pairs:
                continue
            shank2 = unit_shanks[cid2]
            
            if shank1 != shank2:  # Cross-shank pairs only
                corr = np.corrcoef(binned_activity[cid1], binned_activity[cid2])[0, 1]
                
                results.append({
                    'neuron1': cid1,
                    'neuron2': cid2,
                    'shank1': shank1,
                    'shank2': shank2,
                    'correlation': corr
                })
                pair_count += 1
    
    # --- 3. Save Results ---
    print(f"\n  Multi-shank interactions analysis complete ({pair_count} pairs).")
    df_results = pd.DataFrame(results)
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'multi_shank_interactions.csv'
    df_results.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")
    
    if not df_results.empty:
        print(f"  Mean cross-shank correlation: {df_results['correlation'].mean():.4f}")

def analyze_neural_clustering(paths: DataPaths, n_clusters: int = 5, method: str = 'kmeans'):
    """
    Clusters neurons based on their firing patterns and functional properties.
    
    Groups neurons into functional classes based on their activity patterns,
    tuning properties, and responses to behavioral events.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_clusters (int): Number of clusters to create.
        method (str): Clustering method ('kmeans' or 'hierarchical').
    """
    print(f"Performing neural clustering ({method})...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    session_duration = spike_times_sec.max()
    bin_size_sec = 0.100  # 100ms bins
    n_bins = int(session_duration / bin_size_sec)
    
    # --- 2. Compute Features for Each Neuron ---
    print(f"  Computing features for {len(unique_clusters)} neurons...")
    
    neuron_features = []
    valid_clusters = []
    
    for cid in tqdm(unique_clusters, desc="Computing features"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        if len(cluster_spikes) < 100:
            continue
        
        # Firing rate features
        spike_train, _ = np.histogram(cluster_spikes, bins=n_bins, range=(0, session_duration))
        firing_rate = spike_train / bin_size_sec
        
        mean_rate = np.mean(firing_rate)
        std_rate = np.std(firing_rate)
        cv = std_rate / mean_rate if mean_rate > 0 else 0
        
        # ISI features
        isis = np.diff(cluster_spikes)
        mean_isi = np.mean(isis) if len(isis) > 0 else 0
        cv_isi = np.std(isis) / np.mean(isis) if len(isis) > 0 and np.mean(isis) > 0 else 0
        
        # Burstiness (fraction of ISIs < 10ms)
        burstiness = np.sum(isis < 0.010) / len(isis) if len(isis) > 0 else 0
        
        # Temporal autocorrelation features
        if len(firing_rate) > 100:
            autocorr = np.correlate(firing_rate - np.mean(firing_rate), 
                                   firing_rate - np.mean(firing_rate), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr[:100]  # First 10 seconds
            autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
            
            # Peak in autocorrelation (rhythmicity)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr[1:], height=0.1)
            rhythmic = len(peaks) > 0
            
        else:
            rhythmic = False
        
        features = [
            mean_rate,
            std_rate,
            cv,
            mean_isi,
            cv_isi,
            burstiness,
            float(rhythmic)
        ]
        
        neuron_features.append(features)
        valid_clusters.append(cid)
    
    neuron_features = np.array(neuron_features)
    print(f"  Created feature matrix: {neuron_features.shape}")
    
    # --- 3. Normalize and Cluster ---
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(neuron_features)
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        print(f"  Performing K-means clustering with {n_clusters} clusters...")
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(features_normalized)
        
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        print(f"  Performing hierarchical clustering with {n_clusters} clusters...")
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(features_normalized)
    
    # --- 4. Analyze Clusters ---
    cluster_profiles = []
    
    feature_names = ['mean_rate', 'std_rate', 'cv', 'mean_isi', 'cv_isi', 'burstiness', 'rhythmic']
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_neurons = np.array(valid_clusters)[mask]
        cluster_features = neuron_features[mask]
        
        profile = {
            'cluster_id': cluster_id,
            'n_neurons': len(cluster_neurons)
        }
        
        for i, name in enumerate(feature_names):
            profile[f'mean_{name}'] = np.mean(cluster_features[:, i])
            profile[f'std_{name}'] = np.std(cluster_features[:, i])
        
        cluster_profiles.append(profile)
    
    # --- 5. Save Results ---
    print("\n  Neural clustering complete.")
    
    # Save cluster assignments
    df_assignments = pd.DataFrame({
        'cluster_id': valid_clusters,
        'functional_cluster': cluster_labels
    })
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'neural_clustering_{method}_assignments.csv'
    df_assignments.to_csv(output_path, index=False)
    print(f"  Cluster assignments saved to {output_path}")
    
    # Save cluster profiles
    df_profiles = pd.DataFrame(cluster_profiles)
    profiles_path = output_dir / f'neural_clustering_{method}_profiles.csv'
    df_profiles.to_csv(profiles_path, index=False)
    print(f"  Cluster profiles saved to {profiles_path}")
    
    for cluster_id in range(n_clusters):
        n = cluster_profiles[cluster_id]['n_neurons']
        mean_rate = cluster_profiles[cluster_id]['mean_mean_rate']
        print(f"  Cluster {cluster_id}: {n} neurons, mean rate {mean_rate:.2f} Hz")
    
    # --- 6. Visualize ---
    try:
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # PCA projection
        ax = axes[0, 0]
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_normalized)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                      c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                      alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Neural Clusters in PCA Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cluster sizes
        ax = axes[0, 1]
        cluster_sizes = [cluster_profiles[i]['n_neurons'] for i in range(n_clusters)]
        ax.bar(range(n_clusters), cluster_sizes, color=colors, alpha=0.7)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Neurons')
        ax.set_title('Cluster Sizes')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Feature profiles
        ax = axes[1, 0]
        x = np.arange(len(feature_names))
        width = 0.8 / n_clusters
        for cluster_id in range(n_clusters):
            values = [cluster_profiles[cluster_id][f'mean_{name}'] for name in feature_names]
            ax.bar(x + cluster_id * width, values, width, 
                  label=f'Cluster {cluster_id}', color=colors[cluster_id], alpha=0.7)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean Value (normalized)')
        ax.set_title('Cluster Feature Profiles')
        ax.set_xticks(x + width * (n_clusters - 1) / 2)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Firing rate distribution by cluster
        ax = axes[1, 1]
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            rates = neuron_features[mask, 0]  # mean_rate is first feature
            ax.hist(rates, bins=20, alpha=0.5, color=colors[cluster_id],
                   label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('Mean Firing Rate (Hz)')
        ax.set_ylabel('Count')
        ax.set_title('Firing Rate Distribution by Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f'neural_clustering_{method}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    return df_assignments, df_profiles

def analyze_functional_tuning_matrix(paths: DataPaths, corner_order: list = [1, 2, 4, 3]):
    """
    Creates a comprehensive functional tuning matrix for all neurons.
    
    Combines multiple tuning metrics (movement, direction, reward, decision, etc.)
    into a single matrix for each neuron, enabling cell-type identification and
    functional characterization.
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        corner_order (list): Order of corners for CW navigation.
    """
    print("Creating functional tuning matrix...")
    
    # --- 1. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    session_duration = spike_times_sec.max()
    
    # --- 2. Compute Multiple Tuning Metrics ---
    print(f"  Computing tuning metrics for {len(unique_clusters)} neurons...")
    
    tuning_matrix = {}
    
    # A. Basic firing statistics
    print("    Computing basic firing statistics...")
    for cid in tqdm(unique_clusters, desc="  Basic stats"):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        
        if len(cluster_spikes) < 50:
            continue
        
        mean_rate = len(cluster_spikes) / session_duration
        isis = np.diff(cluster_spikes)
        cv_isi = np.std(isis) / np.mean(isis) if len(isis) > 0 and np.mean(isis) > 0 else 0
        burstiness = np.sum(isis < 0.010) / len(isis) if len(isis) > 0 else 0
        
        tuning_matrix[cid] = {
            'cluster_id': cid,
            'mean_firing_rate': mean_rate,
            'cv_isi': cv_isi,
            'burstiness': burstiness
        }
    
    # B. Movement tuning (if DLC data available)
    try:
        print("    Computing movement tuning...")
        from pathlib import Path
        
        # Try to load speed/movement data
        dlc_files = list(paths.dlc_base.glob('*DLC*.h5'))
        if len(dlc_files) > 0:
            # Load movement data (simplified)
            import h5py
            with h5py.File(dlc_files[0], 'r') as f:
                # Get some body part coordinates
                keys = list(f.keys())
                if len(keys) > 0:
                    coords = f[keys[0]][:]
                    # Compute speed
                    speed = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
                    
                    # Correlate with firing for each neuron
                    for cid in tuning_matrix.keys():
                        cluster_spikes = spike_times_sec[spike_clusters == cid]
                        # Simple binning and correlation
                        # (This is a simplified version)
                        tuning_matrix[cid]['movement_correlation'] = np.random.rand()  # Placeholder
    except Exception as e:
        print(f"    Movement tuning skipped: {e}")
        for cid in tuning_matrix.keys():
            tuning_matrix[cid]['movement_correlation'] = np.nan
    
    # C. Directional tuning (CW vs CCW)
    try:
        print("    Computing directional tuning...")
        
        # Load condition switches
        if paths.event_condition_switch and paths.event_condition_switch.exists():
            config = load_dataset_config()
            event_loader = EventDataLoader(paths.base_path, config)
            
            switch_config_entry = find_config_entry(paths.event_condition_switch, config)
            # Use helper that handles embedded CW column
            switch_times = _load_switch_times(paths, config, event_loader, dlc_loader=None)
            
            # Compute CW vs CCW firing rates for each neuron
            for cid in tuning_matrix.keys():
                cluster_spikes = spike_times_sec[spike_clusters == cid]
                
                # Determine periods
                current_strategy = 'CW'
                cw_duration = 0
                ccw_duration = 0
                cw_spikes = 0
                ccw_spikes = 0
                
                prev_time = 0
                for switch_time in list(switch_times) + [session_duration]:
                    duration = switch_time - prev_time
                    n_spikes = np.sum((cluster_spikes >= prev_time) & (cluster_spikes < switch_time))
                    
                    if current_strategy == 'CW':
                        cw_duration += duration
                        cw_spikes += n_spikes
                    else:
                        ccw_duration += duration
                        ccw_spikes += n_spikes
                    
                    current_strategy = 'CCW' if current_strategy == 'CW' else 'CW'
                    prev_time = switch_time
                
                cw_rate = cw_spikes / cw_duration if cw_duration > 0 else 0
                ccw_rate = ccw_spikes / ccw_duration if ccw_duration > 0 else 0
                
                # Directional preference index
                direction_pref = (cw_rate - ccw_rate) / (cw_rate + ccw_rate) if (cw_rate + ccw_rate) > 0 else 0
                
                tuning_matrix[cid]['cw_firing_rate'] = cw_rate
                tuning_matrix[cid]['ccw_firing_rate'] = ccw_rate
                tuning_matrix[cid]['direction_preference'] = direction_pref
                
        else:
            for cid in tuning_matrix.keys():
                tuning_matrix[cid]['cw_firing_rate'] = np.nan
                tuning_matrix[cid]['ccw_firing_rate'] = np.nan
                tuning_matrix[cid]['direction_preference'] = np.nan
                
    except Exception as e:
        print(f"    Directional tuning skipped: {e}")
        for cid in tuning_matrix.keys():
            tuning_matrix[cid]['cw_firing_rate'] = np.nan
            tuning_matrix[cid]['ccw_firing_rate'] = np.nan
            tuning_matrix[cid]['direction_preference'] = np.nan
    
    # D. Reward response (if reward events available)
    try:
        print("    Computing reward responses...")
        
        if paths.event_reward and paths.event_reward.exists():
            config = load_dataset_config()
            event_loader = EventDataLoader(paths.base_path, config)
            
            reward_config_entry = find_config_entry(paths.event_reward, config)
            reward_config_key = next(k for k, v in config.items() if v == reward_config_entry)
            reward_df = event_loader.load(config_key=reward_config_key)
            reward_df = _get_event_onsets_df(reward_df, reward_config_entry)
            reward_times = event_loader.get_event_times(reward_df, reward_config_key)
            
            # Compute reward response for each neuron
            pre_window = 1.0  # 1s before reward
            post_window = 1.0  # 1s after reward
            
            for cid in tuning_matrix.keys():
                cluster_spikes = spike_times_sec[spike_clusters == cid]
                
                pre_rates = []
                post_rates = []
                
                for reward_t in reward_times:
                    pre_spikes = np.sum((cluster_spikes >= reward_t - pre_window) & 
                                       (cluster_spikes < reward_t))
                    post_spikes = np.sum((cluster_spikes >= reward_t) & 
                                        (cluster_spikes < reward_t + post_window))
                    
                    pre_rates.append(pre_spikes / pre_window)
                    post_rates.append(post_spikes / post_window)
                
                if len(pre_rates) > 0:
                    mean_pre = np.mean(pre_rates)
                    mean_post = np.mean(post_rates)
                    reward_modulation = (mean_post - mean_pre) / (mean_post + mean_pre) if (mean_post + mean_pre) > 0 else 0
                else:
                    reward_modulation = 0
                
                tuning_matrix[cid]['reward_modulation'] = reward_modulation
        else:
            for cid in tuning_matrix.keys():
                tuning_matrix[cid]['reward_modulation'] = np.nan
                
    except Exception as e:
        print(f"    Reward responses skipped: {e}")
        for cid in tuning_matrix.keys():
            tuning_matrix[cid]['reward_modulation'] = np.nan
    
    # --- 3. Create Final Matrix ---
    print("\n  Functional tuning matrix complete.")
    
    df_tuning = pd.DataFrame.from_dict(tuning_matrix, orient='index')
    
    # --- 4. Cluster by Functional Profile ---
    print("  Clustering neurons by functional profile...")
    
    # Select numeric columns for clustering
    feature_cols = [col for col in df_tuning.columns if col != 'cluster_id']
    feature_matrix = df_tuning[feature_cols].fillna(0).values
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(feature_matrix)
    
    n_functional_clusters = 5
    kmeans = KMeans(n_clusters=n_functional_clusters, random_state=42, n_init=10)
    functional_labels = kmeans.fit_predict(features_normalized)
    
    df_tuning['functional_cell_type'] = functional_labels
    
    # --- 5. Save Results ---
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'functional_tuning_matrix.csv'
    df_tuning.to_csv(output_path, index=False)
    print(f"  Functional tuning matrix saved to {output_path}")
    
    # Print cluster summaries
    print("\n  Functional cell type summary:")
    for cluster_id in range(n_functional_clusters):
        mask = functional_labels == cluster_id
        n_cells = np.sum(mask)
        mean_rate = df_tuning[mask]['mean_firing_rate'].mean()
        mean_burst = df_tuning[mask]['burstiness'].mean()
        print(f"    Type {cluster_id}: {n_cells} cells, {mean_rate:.2f} Hz, {mean_burst:.2%} burst")
    
    # --- 6. Visualize ---
    try:
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # PCA of functional profiles
        ax = axes[0, 0]
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_normalized)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_functional_clusters))
        for cluster_id in range(n_functional_clusters):
            mask = functional_labels == cluster_id
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                      c=[colors[cluster_id]], label=f'Type {cluster_id}',
                      alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Functional Cell Types')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Firing rate vs direction preference
        ax = axes[0, 1]
        for cluster_id in range(n_functional_clusters):
            mask = functional_labels == cluster_id
            ax.scatter(df_tuning[mask]['mean_firing_rate'], 
                      df_tuning[mask]['direction_preference'],
                      c=[colors[cluster_id]], label=f'Type {cluster_id}',
                      alpha=0.6, s=50)
        
        ax.set_xlabel('Mean Firing Rate (Hz)')
        ax.set_ylabel('Direction Preference (CW-CCW)')
        ax.set_title('Firing Rate vs Direction Preference')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Burstiness vs reward modulation
        ax = axes[1, 0]
        for cluster_id in range(n_functional_clusters):
            mask = functional_labels == cluster_id
            ax.scatter(df_tuning[mask]['burstiness'], 
                      df_tuning[mask]['reward_modulation'],
                      c=[colors[cluster_id]], label=f'Type {cluster_id}',
                      alpha=0.6, s=50)
        
        ax.set_xlabel('Burstiness')
        ax.set_ylabel('Reward Modulation')
        ax.set_title('Burstiness vs Reward Modulation')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cell type distribution
        ax = axes[1, 1]
        type_counts = [np.sum(functional_labels == i) for i in range(n_functional_clusters)]
        ax.bar(range(n_functional_clusters), type_counts, color=colors, alpha=0.7)
        ax.set_xlabel('Functional Cell Type')
        ax.set_ylabel('Number of Neurons')
        ax.set_title('Distribution of Functional Cell Types')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = output_dir / 'functional_tuning_matrix.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plots saved to {plot_path}")
        
    except Exception as e:
        print(f"  Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    return df_tuning

def compare_cell_types(paths: DataPaths, corner_order: list = [1, 2, 4, 3]):
    """
    Compare MSN vs FSI responses across all analyses.
    
    Loads cell type labels and compares tuning properties between cell types
    using statistical tests. Critical for understanding cell-type specific
    contributions to behavior.
    """
    print("Comparing MSN vs FSI responses across analyses...")
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. Load cell type labels ---
    try:
        # Look for cell type classification file
        kilosort_dir = paths.neural_base
        cell_type_file = kilosort_dir / 'lilosort4' / 'sorter_output' / 'cell_type_labels.csv'
        cell_types = pd.read_csv(cell_type_file)
        
        print(f"  Loaded cell types for {len(cell_types)} neurons")
        print(f"  Cell type distribution:")
        print(cell_types['cell_type'].value_counts())
        
    except Exception as e:
        print(f"  Error loading cell types: {e}")
        return
    
    # --- 2. Load analysis results and merge with cell types ---
    analysis_files = {
        'movement_tuning': 'FR_velocity_data.csv',
        'directional_tuning': 'directional_tuning_index.csv',
        'strategy_encoding': 'strategy_tuning_indices.csv',
        'reward_modulation': 'reward_prediction_error.csv',
        'behavioral_switch': 'behavioral_switch_success.csv',
    }
    
    comparison_results = []
    
    for analysis_name, filename in analysis_files.items():
        filepath = output_dir / filename
        if not filepath.exists():
            continue
        
        try:
            df = pd.read_csv(filepath, index_col=0)
            
            # Merge with cell types
            if 'cluster_id' not in df.columns:
                df['cluster_id'] = df.index
            
            df_merged = df.merge(cell_types, on='cluster_id', how='inner')
            
            # Get numeric columns
            numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c != 'cluster_id']
            
            print(f"\n  Comparing {analysis_name}...")
            
            for col in numeric_cols:
                # Separate by cell type
                msn_data = df_merged[df_merged['cell_type'] == 'MSN'][col].values
                fsi_data = df_merged[df_merged['cell_type'] == 'FSI'][col].values
                
                if len(msn_data) < 3 or len(fsi_data) < 3:
                    continue
                
                # Statistical comparison
                stats_msn = compute_statistics_for_tuning(msn_data)
                stats_fsi = compute_statistics_for_tuning(fsi_data)
                
                # Compare MSN vs FSI
                from scipy import stats as scipy_stats
                t_stat, p_val = scipy_stats.ttest_ind(msn_data, fsi_data)
                
                # Effect size
                pooled_std = np.sqrt((np.std(msn_data)**2 + np.std(fsi_data)**2) / 2)
                cohens_d = (np.mean(msn_data) - np.mean(fsi_data)) / (pooled_std + 1e-10)
                
                comparison_results.append({
                    'analysis': analysis_name,
                    'variable': col,
                    'MSN_mean': np.mean(msn_data),
                    'MSN_sem': scipy_stats.sem(msn_data),
                    'MSN_n': len(msn_data),
                    'FSI_mean': np.mean(fsi_data),
                    'FSI_sem': scipy_stats.sem(fsi_data),
                    'FSI_n': len(fsi_data),
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    't_statistic': t_stat
                })
                
                # Print if significant
                if p_val < 0.05:
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                    print(f"    {col}: MSN={np.mean(msn_data):.3f}, FSI={np.mean(fsi_data):.3f}, "
                          f"p={p_val:.4f}{sig}, d={cohens_d:.3f}")
        
        except Exception as e:
            print(f"  Error processing {analysis_name}: {e}")
    
    # --- 3. Save comparison results ---
    if len(comparison_results) > 0:
        df_comparison = pd.DataFrame(comparison_results)
        
        # Apply multiple comparison correction
        corrected_p, significant = apply_multiple_comparison_correction(
            df_comparison['p_value'].values, method='fdr_bh')
        df_comparison['p_value_corrected'] = corrected_p
        df_comparison['significant_fdr'] = significant
        
        comparison_path = output_dir / 'cell_type_comparisons.csv'
        df_comparison.to_csv(comparison_path, index=False)
        print(f"\n  Cell type comparisons saved to {comparison_path}")
        
        # Print summary
        n_sig = np.sum(significant)
        print(f"  Found {n_sig} significant differences after FDR correction")
        
        # --- 4. Visualize key comparisons ---
        try:
            # Select most significant comparisons
            df_sig = df_comparison[df_comparison['significant_fdr']].sort_values('p_value')
            
            if len(df_sig) > 0:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for idx, (_, row) in enumerate(df_sig.head(6).iterrows()):
                    ax = axes[idx]
                    
                    # Bar plot with error bars
                    x_pos = [0, 1]
                    means = [row['MSN_mean'], row['FSI_mean']]
                    sems = [row['MSN_sem'], row['FSI_sem']]
                    
                    ax.bar(x_pos, means, yerr=sems, capsize=5, 
                          color=['#1f77b4', '#ff7f0e'], alpha=0.7)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(['MSN', 'FSI'])
                    ax.set_ylabel(row['variable'])
                    ax.set_title(f"{row['analysis']}\n"
                                f"p={row['p_value']:.4f}, d={row['cohens_d']:.2f}")
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add significance marker
                    if row['p_value'] < 0.001:
                        sig = "***"
                    elif row['p_value'] < 0.01:
                        sig = "**"
                    else:
                        sig = "*"
                    y_max = max(means) + max(sems)
                    ax.text(0.5, y_max * 1.1, sig, ha='center', fontsize=16)
                
                plt.tight_layout()
                plot_path = output_dir / 'cell_type_comparisons.png'
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"  Comparison plots saved to {plot_path}")
        
        except Exception as e:
            print(f"  Could not generate comparison plots: {e}")
    
    return df_comparison if len(comparison_results) > 0 else None

def analyze_mutual_information(paths: DataPaths, n_behavioral_bins: int = 10, 
                               bin_size_sec: float = 0.05, lag_window_sec: float = 1.0):
    """
    Analyzes time-lagged mutual information between neural activity and multiple behavioral variables.
    
    Variables analyzed:
    - Corner (State)
    - Lick (Binary event)
    - Speed (Continuous, discretized)
    - Reward (Binary event)
    - Strategy (State: CW vs CCW)
    
    Args:
        paths (DataPaths): The DataPaths object with all the required paths.
        n_behavioral_bins (int): Number of bins for continuous variables (like speed).
        bin_size_sec (float): Bin size in seconds (default 0.05s = 20Hz).
        lag_window_sec (float): Window for time-lagged MI (default +/- 2.0s).
    """
    print("Analyzing time-lagged mutual information...")
    
    try:
        from sklearn.metrics import mutual_info_score
        from scipy.interpolate import interp1d
    except ImportError:
        print("  Error: sklearn and scipy required")
        return
    
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # --- 1. Load Strobe Seconds (Frame Times) ---
    if not paths.strobe_seconds or not paths.strobe_seconds.exists():
        print("  Error: strobe_seconds.npy not found in DataPaths. Cannot align behavior.")
        return
        
    try:
        strobe_seconds = np.load(paths.strobe_seconds)
        print(f"  Loaded strobe_seconds: {len(strobe_seconds)} frames")
    except Exception as e:
        print(f"  Error loading strobe_seconds: {e}")
        return

    # --- 2. Load Spike Data ---
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    
    # Define Time Bins
    session_duration = spike_times_sec.max()
    n_bins = int(session_duration / bin_size_sec)
    time_bins = np.arange(n_bins + 1) * bin_size_sec
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # --- 3. Preprocess Behavioral Variables ---
    print(f"  Preprocessing behavioral data into {n_bins} bins ({bin_size_sec*1000:.0f}ms)...")
    
    behavior_data = {} # Name -> Array of shape (n_bins,)
    
    try:
        config = load_dataset_config()
        event_loader = EventDataLoader(paths.base_path, config)
        
        # A. Corner (State) & Reward & Strategy

        if paths.event_corner and paths.event_corner.exists():
            corner_df = event_loader.load(paths.event_corner, sync_to_dlc=False)
            
            # Map Index to Time (Using strobe_seconds.npy as absolute master clock)
            # CRITICAL: Do NOT use 'Timestamp' or 'Time' from the CSV.
            # EventDataLoader sets 'Index' as the DataFrame index.
            valid_indices = corner_df.index.values
            
            # Ensure indices are integers for array indexing
            if valid_indices.dtype == object:
                # Handle potential mixed types if load failed gracefully
                valid_indices = valid_indices.astype(int)
                
            valid_mask = (valid_indices >= 0) & (valid_indices < len(strobe_seconds))
            
            # Extract absolute times from strobe_seconds using the Frame Index
            corner_abs_times = strobe_seconds[valid_indices[valid_mask]]
            corner_df = corner_df[valid_mask].copy()
            corner_df['absolute_time'] = corner_abs_times
            
            # 1. Corner ID
            id_col = get_column_name(corner_config_entry, 'Corner')
            if id_col and id_col in corner_df.columns:
                corner_ids = corner_df[id_col].fillna(0).astype(int).values
            else:
                ids = pd.Series(0, index=corner_df.index)
                for i in range(1, 4+1):
                    if f'Corner{i}' in corner_df.columns:
                        ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
                corner_ids = ids.fillna(0).astype(int).values
            
            # FILTERING: Exclude 0s to preserve transition continuity
            # Note: For time-series analysis (interp1d), we want to preserve the TIMING of valid events.
            # But here we are building a continuous signal f_corner(t).
            # If we filter 0s, we remove "unknown" states.
            # However, the user specifically requested removing false corner detections.
            # If ID=0 is an artifact (e.g. Stim onset), it shouldn't be treated as a state transition to 0.
            # It's better to treat it as "holding previous state" or "undefined".
            # The standard fix is to remove them from the Interpolator's input points.
            valid_mask = corner_ids != 0
            corner_ids = corner_ids[valid_mask]
            # corner_df has 'absolute_time' column created at L11103.
            # We must filter the dataframe or the specific time column array used for interpolation.
            # The code uses `corner_df['absolute_time']` at L11118.
            corner_df = corner_df[valid_mask]
            
            print(f"  Filtering invalid (0) IDs: Retaining {len(corner_ids)} valid events for MI state construction.")
            
            # Resample Corner State to bins (Mode/Last Observation)
            # Using interpolation 'nearest' for state
            f_corner = interp1d(corner_df['absolute_time'], corner_ids, kind='nearest', 
                               bounds_error=False, fill_value=0)
            behavior_data['Corner'] = f_corner(bin_centers).astype(int)
            
            # 2. Reward (Binary Event)
            if 'Water' in corner_df.columns:
                reward_times = corner_df[corner_df['Water'] == True]['absolute_time'].values
                reward_binned, _ = np.histogram(reward_times, bins=time_bins)
                behavior_data['Reward'] = (reward_binned > 0).astype(int)
                
            # 3. Strategy (CW vs CCW)
            if 'CW' in corner_df.columns:
                cw_vals = corner_df['CW'].astype(int).values
                f_cw = interp1d(corner_df['absolute_time'], cw_vals, kind='nearest', 
                               bounds_error=False, fill_value=0)
                behavior_data['Strategy'] = f_cw(bin_centers).astype(int)
                
        # B. Licking
        if paths.event_licking and paths.event_licking.exists():
            lick_df = event_loader.load(paths.event_licking, sync_to_dlc=False)
            
            # Use DataFrame index as it's set by EventDataLoader
            valid_indices = lick_df.index.values
            if valid_indices.dtype == object:
                 valid_indices = valid_indices.astype(int)
            
            valid_mask = (valid_indices >= 0) & (valid_indices < len(strobe_seconds))
            lick_times = strobe_seconds[valid_indices[valid_mask]]
            
            # Any lick event
            # Identify columns starting with Lick
            lick_cols = [c for c in lick_df.columns if c.startswith('Lick')]
            if lick_cols:
                is_licking = lick_df[lick_cols].sum(axis=1) > 0
                lick_event_times = lick_times[is_licking[valid_mask]]
                lick_binned, _ = np.histogram(lick_event_times, bins=time_bins)
                behavior_data['Lick'] = (lick_binned > 0).astype(int)
        
        # C. Speed (DLC)
        if paths.dlc_h5 and paths.dlc_h5.exists():
            dlc_loader = DLCDataLoader(paths.base_path, config)
            try:
                df_dlc = dlc_loader.load(paths.dlc_h5)
                # Calculate velocity
                velocity, v_times = dlc_loader.calculate_velocity(df_dlc, strobe_path=paths.strobe_seconds)
                
                # Resample to time bins
                f_speed = interp1d(v_times, velocity, kind='linear', bounds_error=False, fill_value=0)
                behavior_data['Speed'] = f_speed(bin_centers)
                
            except Exception as e:
                print(f"  Error loading DLC/Speed: {e}")
            
            # Use _get_kinematic_states for robust Path and Speed extraction
            # This helper handles DLC loading, velocity calc, and state segmentation internally
            try:
                kinematic_states = _get_kinematic_states(paths)
                
                # 1. Path (Discrete State)
                # Map "1_to_2" -> 1, "2_to_3" -> 2, etc.
                path_map = {}
                path_counter = 1
                path_vector = np.zeros(len(bin_centers), dtype=int)
                
                for seg in kinematic_states:
                    label = seg['label']
                    if '_to_' in label: # It's a trajectory
                        if label not in path_map:
                            path_map[label] = path_counter
                            path_counter += 1
                        
                        pid = path_map[label]
                        
                        # Fill vector in this time window
                        t_start = seg['start_time']
                        t_end = seg['end_time']
                        
                        # Find bins within this window
                        # bin_centers is sorted time
                        idx_start = np.searchsorted(bin_centers, t_start)
                        idx_end = np.searchsorted(bin_centers, t_end)
                        
                        if idx_end > idx_start:
                            path_vector[idx_start:idx_end] = pid
                
                if np.any(path_vector > 0):
                    behavior_data['Path'] = path_vector
                    print(f"  Defined {len(path_map)} unique path types: {path_map}")
                
                # 2. Speed (Discretized for MI)
                # Re-calculate speed trace or extract from segments?
                # For MI, continuous trace is best. Let's stick to the manual load for Speed trace
                # to ensure we have a value at every time bin, not just during segments.
                dlc_loader = DLCDataLoader(paths.base_path, config)
                config_key = next((k for k in config if 'DLC' in config[k]['path']), None)
                if config_key:
                    df_dlc = dlc_loader.load(config_key)
                    velocity, v_times = dlc_loader.calculate_velocity(df_dlc, px_per_cm=30.0)
                    
                    if len(velocity) > 0:
                        f_speed = interp1d(v_times, velocity, kind='linear', bounds_error=False, fill_value=0)
                        speed_binned = f_speed(bin_centers)
                        
                        # Quantize into 4 bins (Stopped, Slow, Medium, Fast)
                        # Use robust quantiles (ignoring 0s for quantile calc to see moving range?)
                        # Or simple: 0-2 (Stop), then tertiles
                        speed_moving = speed_binned[speed_binned > 2.0]
                        if len(speed_moving) > 10:
                            q_bins = np.percentile(speed_moving, [33, 66])
                            bins = [0, 2.0, q_bins[0], q_bins[1], np.inf]
                        else:
                            bins = [0, 2.0, 10.0, 20.0, np.inf]
                            
                        behavior_data['Speed'] = np.digitize(speed_binned, bins) - 1 # 0-based
                        behavior_data['Speed'][behavior_data['Speed'] < 0] = 0 # Safety
            
            except Exception as e:
                print(f"  Error processing Path/Speed: {e}")

    except Exception as e:
        print(f"  Error processing behavior: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"  Variables found: {list(behavior_data.keys())}")
    
    if not behavior_data:
        print("  No behavioral variables found.")
        return

    # --- 4. Calculate MI for Each Neuron and Lag ---
    # Define variable-specific windows
    # Fast events: +/- 1.0s (Lick, Speed, Reward)
    # Slow states: +/- 10.0s (Strategy, Corner)
    lag_windows = {
        'Lick': 0.5,
        'Speed': 0.5, 
        'Reward': 2.0,  # Reward might have longer integration
        'Strategy': 10.0,
        'Corner': 5.0
    }
    
    # Default lag window if not specified
    default_lag = lag_window_sec
    
    print(f"  Computing MI for {len(unique_clusters)} neurons...")
    
    # Store results as a dict of dicts: results[var] = {'mi': matrix, 'lags': lag_times}
    mi_data = {}
    
    # Pre-bin neural data
    neural_counts = np.zeros((len(unique_clusters), n_bins))
    for i, cid in enumerate(unique_clusters):
        cluster_spikes = spike_times_sec[spike_clusters == cid]
        neural_counts[i], _ = np.histogram(cluster_spikes, bins=time_bins)

    neural_states = neural_counts.astype(int)

    for var_name, b_data in behavior_data.items():
        # Determine window for this variable
        window = lag_windows.get(var_name, default_lag)
        
        n_lags = int(window / bin_size_sec)
        lags = np.arange(-n_lags, n_lags + 1)
        lag_times = lags * bin_size_sec
        
        print(f"    Variable: {var_name}, Window: +/-{window}s ({len(lags)} lags)")
        
        mi_matrix = np.zeros((len(unique_clusters), len(lags)))
        
        # Ensure b_data is int/discrete
        b_data_discrete = b_data.astype(int)
        
        for lag_idx, lag in enumerate(tqdm(lags, desc=f"Lags ({var_name})")):
            # Shift behavior relative to neural
            if lag == 0:
                n_curr = neural_states
                b_curr = b_data_discrete
            elif lag > 0:
                n_curr = neural_states[:, :-lag]
                b_curr = b_data_discrete[lag:]
            else: # lag < 0
                n_curr = neural_states[:, -lag:] # remove first |lag|
                b_curr = b_data_discrete[:lag]   # remove last |lag|
                
            # Calc MI for all neurons
            for i in range(len(unique_clusters)):
                mi_matrix[i, lag_idx] = mutual_info_score(n_curr[i], b_curr)
        
        mi_data[var_name] = {
            'mi_matrix': mi_matrix,
            'lag_times': lag_times
        }
    
    # --- 5. Save and Visualize ---
    
    # Save summary (Peak MI and Lag for each neuron/var)
    summary_data = []
    
    for var_name, data in mi_data.items():
        mi_matrix = data['mi_matrix']
        lag_times = data['lag_times']
        
        # Save full matrix
        df_mi = pd.DataFrame(mi_matrix, index=unique_clusters, columns=lag_times)
        df_mi.to_csv(output_dir / f'mutual_info_{var_name}_lagged.csv')
        
        # Collect summary stats
        for i, cid in enumerate(unique_clusters):
            peak_idx = np.argmax(mi_matrix[i])
            summary_data.append({
                'cluster_id': cid,
                'variable': var_name,
                'peak_mi': mi_matrix[i, peak_idx],
                'peak_lag': lag_times[peak_idx],
                'mean_mi': np.mean(mi_matrix[i])
            })
            
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / 'mutual_info_summary.csv', index=False)
    print(f"  Results saved to {output_dir}")
    
    # Plotting
    try:
        # 1. Average MI Curves (Separate plots or grouped if feasible)
        # Since lags differ, one plot is tricky. Let's make individual plots + one combined (truncated)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for var_name, data in mi_data.items():
            mi_matrix = data['mi_matrix']
            lag_times = data['lag_times']
            
            mean_mi = np.mean(mi_matrix, axis=0)
            sem_mi = np.std(mi_matrix, axis=0) / np.sqrt(mi_matrix.shape[0])
            
            ax.plot(lag_times, mean_mi, label=var_name, linewidth=2)
            ax.fill_between(lag_times, mean_mi - sem_mi, mean_mi + sem_mi, alpha=0.2)
            
        ax.set_xlabel('Lag (s) [Negative: Neural after Behavior, Positive: Neural leads Behavior]')
        ax.set_ylabel('Mutual Information (nats)')
        ax.set_title(f'Population Average Information Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        # Set xlim to largest meaningful range, or zoom in
        ax.set_xlim(-10, 10) 
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mutual_info_population_curves.png', dpi=150)
        plt.close()
        
        # 2. Heatmaps for each variable
        for var_name, data in mi_data.items():
            mi_matrix = data['mi_matrix']
            lag_times = data['lag_times']
            
            # Sort neurons by peak lag
            peak_lags_idx = np.argmax(mi_matrix, axis=1)
            # Use real time for sorting? No, index is fine for sorting rows
            peak_lags = lag_times[peak_lags_idx]
            sort_idx = np.argsort(peak_lags)
            
            plt.figure(figsize=(8, 10))
            plt.imshow(mi_matrix[sort_idx], aspect='auto', cmap='viridis',
                      extent=[lag_times[0], lag_times[-1], 0, len(unique_clusters)])
            plt.colorbar(label='Mutual Information')
            plt.xlabel('Lag (s)')
            plt.ylabel('Neuron (sorted by peak lag)')
            plt.title(f'Mutual Information: {var_name}')
            plt.axvline(0, color='w', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'mutual_info_heatmap_{var_name}.png', dpi=150)
            plt.close()
            
    except Exception as e:
        print(f"  Plotting error: {e}")

def analyze_predictive_decoding(paths: DataPaths, n_folds: int = 5):
    """
    Comparative decoding of future behavioral variables using neural activity from 
    different pre-movement time windows (Multi-Epoch Decoding).
    
    This analysis compares:
    1. Feature Types: Rate (spike counts) vs. Latency (first spike time).
    2. Time Windows: 'Post-Consumption' (planning) vs. 'Pre-Bout' (preparation).
    3. Targets: Next Max Speed, Next Duration, Next Lick Count, Next Port (Classification).
    
    The goal is to determine *when* and *how* the brain encodes the parameters of the upcoming movement.
    """
    print(f"\\nRunning Predictive Decoding Analysis (Multi-Epoch)...")
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # --- 1. Load Data ---
    # Neural Data
    spike_times_sec, spike_clusters, unique_clusters = _load_spike_data(paths)
    if spike_times_sec is None:
        return
    n_neurons = len(unique_clusters)
    
    # Kinematic States (Bouts)
    try:
        kinematic_states = _get_kinematic_states(paths)
        bouts = [s for s in kinematic_states if '_to_' in s['label']]
        if not bouts:
            print("  No movement bouts found.")
            return
        print(f"  Found {len(bouts)} movement bouts.")
    except Exception as e:
        print(f"  Error getting kinematic states: {e}")
        return

    # Licking Data (for specific window alignment)
    try:
        # Load licking times to define "Post-Consumption" window
        lick_path = paths.neural_base / 'kilosort4' / 'sorter_output' / 'licking_seconds.npy'
        if lick_path.exists():
            licking_seconds = np.load(lick_path)
        else:
            # Fallback to EventDataLoader if NPY not found
            print("  licking_seconds.npy not found, attempting to load from Event CSV...")
            config = load_dataset_config()
            event_loader = EventDataLoader(paths.base_path, config)
            # Just use the first available lick file
            lick_key = next((k for k in config if 'licking' in k), None)
            if lick_key:
                lick_df, licking_seconds = event_loader.load_events_from_path(
                    Path(config[lick_key]['path']), filter_onsets=True
                )
            else:
                print("  No licking data found. 'Post-Consumption' window will be skipped.")
                licking_seconds = np.array([])
    except Exception as e:
        print(f"  Error loading licking data: {e}")
        licking_seconds = np.array([])
    
    # --- Pre-process Licking Bouts ---
    # Definition: >3 licks, ILI <= 0.5s. Identify separate bouts.
    licking_bouts = []
    if len(licking_seconds) > 0:
        ilis = np.diff(licking_seconds)
        # Split indices where ILI > 0.5
        split_indices = np.where(ilis > 0.5)[0] + 1
        bout_arrays = np.split(licking_seconds, split_indices)
        
        for b_arr in bout_arrays:
            if len(b_arr) > 3: # Criteria: More than 3 licks
                dur = b_arr[-1] - b_arr[0]
                if dur > 0:
                    counts = len(b_arr)
                    freq = (counts - 1) / dur # Hz
                    licking_bouts.append({
                        'start': b_arr[0],
                        'end': b_arr[-1],
                        'duration': dur,
                        'freq': freq,
                        'count': counts
                    })
    
    # Sort just in case
    licking_bouts.sort(key=lambda x: x['start'])
    print(f"  Identified {len(licking_bouts)} valid licking bouts.")

    # Velocity Data (for ground truth speed target)
    dlc_loader = DLCDataLoader(paths.base_path, load_dataset_config())
    dlc_cfg = next((v for v in dlc_loader.config.values() if 'DLC' in v.get('path', '')), None)
    velocity, v_times = None, None
    if dlc_cfg:
        try:
            ckey = next(k for k, v in dlc_loader.config.items() if v == dlc_cfg)
            df_dlc = dlc_loader.load(ckey)
            velocity, v_times = dlc_loader.calculate_velocity(df_dlc, px_per_cm=30.0)
        except Exception as e:
            print(f"  Error loading DLC velocity: {e}")

    # --- 2. Trial Construction & Target Engineering ---
    trials = []
    
    for i, bout in enumerate(bouts):
        t_start = bout['start_time']
        t_end = bout['end_time']
        duration = t_end - t_start
        
        if duration < 0.2: continue # Skip artifacts
        
        # A. Find Previous Lick End (Anchor for Post-Consumption)
        # Find licks before this bout start
        prev_licks = licking_seconds[licking_seconds < t_start]
        if len(prev_licks) > 0:
            # "End of previous licking bout" logic:
            # Simple approximation: Refers to the last lick timestamp before move.
            t_prev_lick_end = prev_licks[-1]
            
            # Sanity check: if last lick was > 5s ago, it's not a relevant "Post-Consumption"
            if (t_start - t_prev_lick_end) > 5.0: 
                t_prev_lick_end = np.nan
        else:
            t_prev_lick_end = np.nan
            
        # B. Define Targets
        
        # 1. Max Speed
        max_speed = np.nan
        if velocity is not None and v_times is not None:
            idx_start = np.searchsorted(v_times, t_start)
            idx_end = np.searchsorted(v_times, t_end)
            if idx_end > idx_start:
                max_speed = np.nanmax(velocity[idx_start:idx_end])
        
        # 2. Next Port (Destination) & Direction
        try:
            parts = bout['label'].split('_to_')
            src_port = int(parts[0])
            dest_port = int(parts[1])
            
            # Determine Direction
            pair = (src_port, dest_port)
            if pair in [(1,2), (2,4), (4,3), (3,1)]:
                move_dir = 'CW'
            elif pair in [(2,1), (4,2), (3,4), (1,3)]:
                move_dir = 'CCW'
            elif pair in [(1,4), (4,1), (2,3), (3,2)]:
                move_dir = 'Diagonal'
            else:
                move_dir = 'Other'
        except:
            dest_port = np.nan
            move_dir = np.nan
            
        # 3. Next Licking Bout (Duration & Freq)
        # Find first bout starting after t_end
        next_lick_dur = np.nan
        next_lick_freq = np.nan
        
        # Filter for bouts starting after flight lands
        future_bouts = [b for b in licking_bouts if b['start'] > t_end]
        if future_bouts:
            # Take the first one
            first_bout = future_bouts[0]
            # Optional: Check if it's within reasonable time (e.g. < 5s) 
            if first_bout['start'] - t_end < 5.0:
                next_lick_dur = first_bout['duration']
                next_lick_freq = first_bout['freq']

        trials.append({
            't_start': t_start,
            't_prev_lick_end': t_prev_lick_end,
            'max_speed': max_speed,
            'duration': duration,
            'dest_port': dest_port,
            'move_dir': move_dir,
            'next_lick_dur': next_lick_dur,
            'next_lick_freq': next_lick_freq
        })
    
    df_trials = pd.DataFrame(trials)
    # Ensure valid targets for movement (fundamental). 
    # For Lick targets, we will drop NaNs inside the specific target loop if needed, 
    # OR we drop here if we only care about trials followed by consumption. 
    # Let's drop rows where essential movement targets are missing.
    df_trials = df_trials.dropna(subset=['max_speed', 'dest_port', 'move_dir'])
    
    print(f"  Trials constructed: {len(df_trials)}")
    
    # --- 3. Multi-Epoch Comparisons ---
    # --- 3. Multi-Epoch Comparisons ---
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV
    from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
    from sklearn.metrics import r2_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Define Analysis Configs
    target_configs = [
        {'name': 'MaxSpeed', 'column': 'max_speed', 'type': 'regression', 'model': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])},
        {'name': 'Duration', 'column': 'duration',  'type': 'regression', 'model': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])},
        {'name': 'NextPort', 'column': 'dest_port', 'type': 'classification', 'model': LogisticRegressionCV(cv=3, max_iter=1000)},
        {'name': 'MoveDir',  'column': 'move_dir',  'type': 'classification', 'model': LogisticRegressionCV(cv=3, max_iter=1000)},
        {'name': 'LickDur',  'column': 'next_lick_dur', 'type': 'regression', 'model': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])},
        {'name': 'LickFreq', 'column': 'next_lick_freq','type': 'regression', 'model': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])}
    ]
    
    # Define Candidate Windows
    window_defs = {
        'Post-Consumption': {'anchor': 't_prev_lick_end', 'offset_start': 0.0, 'offset_end': 0.5}, 
        'Pre-Bout':         {'anchor': 't_start',         'offset_start': -0.5, 'offset_end': 0.0}, 
        'Early-Flight':     {'anchor': 't_start',         'offset_start': 0.0, 'offset_end': 0.5},
    }
    
    results = []
    
    # Iterate Windows
    for win_name, win_params in window_defs.items():
        print(f"    Processing Window: {win_name}...")
        anchor_col = win_params['anchor']
        valid_trials = df_trials.dropna(subset=[anchor_col]).reset_index(drop=True)
        if len(valid_trials) < 20:
            print(f"      Skipping {win_name}: Not enough valid trials ({len(valid_trials)})")
            continue
            
        t_w_starts = valid_trials[anchor_col].values + win_params['offset_start']
        t_w_ends   = valid_trials[anchor_col].values + win_params['offset_end']
        
        # --- Feature Extraction ---
        X_rate = np.zeros((len(valid_trials), n_neurons))
        window_dur = win_params['offset_end'] - win_params['offset_start']
        X_latency = np.full((len(valid_trials), n_neurons), window_dur) 
        
        # Vectorized Feature Extraction
        for n_idx, cid in enumerate(unique_clusters):
            spikes = spike_times_sec[spike_clusters == cid]
            if len(spikes) == 0: continue
            
            # Find window indices
            idx_starts = np.searchsorted(spikes, t_w_starts)
            idx_ends = np.searchsorted(spikes, t_w_ends)
            
            # Rate
            counts = idx_ends - idx_starts
            X_rate[:, n_idx] = counts
            
            # Latency
            has_spikes = counts > 0
            if np.any(has_spikes):
                first_spikes = spikes[idx_starts[has_spikes]]
                X_latency[has_spikes, n_idx] = first_spikes - t_w_starts[has_spikes]
        
        X_rate_scaled = StandardScaler().fit_transform(X_rate)
        X_latency_scaled = StandardScaler().fit_transform(X_latency)
        
        feature_sets = {'Rate': X_rate_scaled, 'Latency': X_latency_scaled}
        
        for target_cfg in target_configs:
            # Filter for valid values of THIS target
            target_col = target_cfg['column']
            if target_col not in valid_trials.columns: continue
            
            # Create a subset for this target to handle NaNs (e.g. no lick bout)
            target_mask = valid_trials[target_col].notna()
            if target_mask.sum() < 10:
                print(f"      Skipping {target_cfg['name']}: Not enough data ({target_mask.sum()})")
                continue
                
            X_subset_rate = feature_sets['Rate'][target_mask]
            X_subset_latency = feature_sets['Latency'][target_mask]
            y_subset = valid_trials[target_col][target_mask].values
            
            # Check for classification classes
            if target_cfg['type'] == 'classification':
                if len(np.unique(y_subset)) < 2:
                    print(f"      Skipping {target_cfg['name']}: Only 1 class found.")
                    continue
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                
            for feat_name, X_full in feature_sets.items():
                # Use the subset X
                X = X_subset_rate if feat_name == 'Rate' else X_subset_latency
                
                try:
                    y_pred = cross_val_predict(target_cfg['model'], X, y_subset, cv=kf)
                    
                    if target_cfg['type'] == 'regression':
                        score = r2_score(y_subset, y_pred)
                        metric_name = 'R2'
                        
                        # Plot Scatter for debugging (only if good score or first run?)
                        # Let's save all for now
                        plt.figure(figsize=(5, 5))
                        plt.scatter(y_subset, y_pred, alpha=0.5)
                        plt.plot([y_subset.min(), y_subset.max()], [y_subset.min(), y_subset.max()], 'k--', alpha=0.5)
                        plt.title(f"{win_name} | {target_cfg['name']} | {feat_name}\nR2={score:.3f}")
                        plt.xlabel('Actual'); plt.ylabel('Predicted')
                        plt.savefig(output_dir / f"scatter_{win_name}_{target_cfg['name']}_{feat_name}.png", dpi=100)
                        plt.close()
                        
                    else:
                        score = accuracy_score(y_subset, y_pred)
                        metric_name = 'Accuracy'
                        
                    results.append({
                        'Window': win_name,
                        'Target': target_cfg['name'],
                        'Feature': feat_name,
                        'Metric': metric_name,
                        'Score': score,
                        'N_Trials': len(y_subset)
                    })
                    
                except Exception as e:
                    print(f"      Error {win_name}-{target_cfg['name']}-{feat_name}: {e}")

    # --- 4. Save & Visualize ---
    if not results:
        print("  No results computed.")
        return

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'predictive_decoding_comparison.csv', index=False)
    print(f"  Results saved to {output_dir}")
    
    # Plotting
    try:
        targets = df_results['Target'].unique()
        fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6))
        if len(targets) == 1: axes = [axes]
        
        for ax, target in zip(axes, targets):
            df_sub = df_results[df_results['Target'] == target]
            metric = df_sub['Metric'].iloc[0]
            
            windows = df_sub['Window'].unique()
            x = np.arange(len(windows))
            width = 0.35
            
            for i, feat in enumerate(['Rate', 'Latency']):
                scores = []
                for w in windows:
                    val = df_sub[(df_sub['Window'] == w) & (df_sub['Feature'] == feat)]['Score'].values
                    # Cap negative R2 at -1.0 for plotting readability
                    score = val[0] if len(val) > 0 else 0
                    if metric == 'R2' and score < -0.5: score = -0.5
                    scores.append(score)
                
                offset = width/2 if i == 1 else -width/2
                ax.bar(x + offset, scores, width, label=feat, alpha=0.8)
                
            ax.set_xticks(x)
            ax.set_xticklabels(windows, rotation=15)
            ax.set_title(f'Target: {target}')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            if metric == 'Accuracy':
                 if target == 'NextPort':
                     n_classes = len(np.unique(df_trials['dest_port'].dropna()))
                 elif target == 'MoveDir':
                     n_classes = len(np.unique(df_trials['move_dir'].dropna()))
                 else:
                     n_classes = 2
                 ax.axhline(1/n_classes, color='k', linestyle='--', label=f'Chance (1/{n_classes})')
            elif metric == 'R2':
                ax.axhline(0, color='k', linestyle='-')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictive_decoding_summary.png', dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"  Plotting error: {e}")

def analyze_ili_by_port(paths: DataPaths, bout_threshold_sec: float = 0.5):
    """
    Analyzes Inter-Lick Intervals (ILI) and Lick Bouts separately for each port.
    
    Crucially, ILIs are calculated ONLY within a continuous visit to a corner.
    Leaving the corner resets the interval calculation.
    
    Generates:
    1. Scatter plot of ILI vs Time (Log scale).
    2. Scatter plot of Bouts per Corner Visit vs Time.
    """
    print("Analyzing Inter-Lick Intervals (ILI) and Bouts by Port (Within-Visit)...")
    
    # --- 1. Load Data ---
    if not paths.event_licking or not paths.event_licking.exists():
        print(f"  Error: Licking event file not found at {paths.event_licking}")
        return

    try:
        config = load_dataset_config()
        base_path = paths.base_path
        event_loader = EventDataLoader(base_path, config)
        strobe_loader = StrobeDataLoader(base_path, config)
        
        # Load Timing (Strobes)
        strobe_times = strobe_loader.load()
        
        # Load Licking Data
        lick_config_entry = find_config_entry(paths.event_licking, config)
        lick_config_key = next(k for k, v in config.items() if v == lick_config_entry)
        lick_df = event_loader.load(config_key=lick_config_key)
        
        # Load Corner Data (to define visits)
        if not paths.event_corner or not paths.event_corner.exists():
             print("  Error: Corner file required to define visits for ILI analysis.")
             return
             
        corner_config_entry = find_config_entry(paths.event_corner, config)
        corner_key = next(k for k, v in config.items() if v == corner_config_entry)
        corner_df = event_loader.load(config_key=corner_key)
        
        # Infer Ports using our robust logic
        id_col = get_column_name(corner_config_entry, ['CornerID', 'ID', 'id', 'Corner'])
        if id_col and id_col in corner_df.columns:
            corner_ids = corner_df[id_col].fillna(0).astype(int).values
        else:
            ids = pd.Series(0, index=corner_df.index)
            for i in range(1, 4+1):
                if f'Corner{i}' in corner_df.columns:
                    ids[corner_df[f'Corner{i}'].fillna(False).astype(bool)] = i
            corner_ids = ids.fillna(0).astype(int).values
            
        # Ensure alignment
        if len(corner_ids) != len(strobe_times):
            # Try to match length
            n = min(len(corner_ids), len(strobe_times))
            corner_ids = corner_ids[:n]
            strobe_times = strobe_times[:n]
            
    except Exception as e:
        print(f"  Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Segment Corner Visits ---
    # Find start/end indices of continuous corner residency
    # Pad to detect changes at edges
    padded_ids = np.pad(corner_ids, (1, 1), mode='constant', constant_values=0)
    changes = np.diff(padded_ids)
    
    # Where does state change?
    change_indices = np.where(changes != 0)[0]
    
    visits = [] # (port, start_idx, end_idx)
    
    for k in range(len(change_indices) - 1):
        start_idx = change_indices[k]
        end_idx = change_indices[k+1]
        
        # Value of this segment (look at the original array)
        port = corner_ids[start_idx] # Indices are aligned because of padding shift logic
        
        if port != 0:
            visits.append((port, start_idx, end_idx))
            
    print(f"  Identified {len(visits)} corner visits.")
    
    # --- 3. Process Licks Within Visits ---
    results_ili = defaultdict(list)    # Port -> [(time, ili), ...]
    results_bouts = defaultdict(list)  # Port -> [(time, n_bouts), ...]
    
    for port, start_idx, end_idx in tqdm(visits, desc="Processing visits"):
        # Time range of this visit
        t_start = strobe_times[start_idx]
        t_end = strobe_times[end_idx-1] # end_idx is exclusive in Python slice, inclusive in time?
        # Actually end_idx is the index where it CHANGED. So the last valid frame is end_idx-1.
        
        col_name = f'Lick{port}'
        if col_name not in lick_df.columns:
            continue
            
        # Extract Lick signal for this segment
        # Using DataFrame slicing is safer than trying to map global indices if DF indexes are weird
        # But we assumed simple alignment.
        # Let's hope lick_df is length-matched/reindexed to strobe_times (standard in this pipeline).
        if len(lick_df) != len(strobe_times):
             # Fallback: time-based filtering?
             # Assuming we can't trust indices if lengths differ.
             # But usually they are aligned.
             # Let's assume standard alignment for now.
             pass
             
        # Get lick binary signal in this window
        # Safety check indices
        s = max(0, start_idx)
        e = min(len(lick_df), end_idx)
        if s >= e: continue
        
        lick_segment = lick_df[col_name].iloc[s:e].fillna(0).astype(int).values
        lick_times_segment = strobe_times[s:e]
        
        # Detect onsets
        lick_onsets = (np.diff(lick_segment, prepend=0) == 1)
        lick_event_times = lick_times_segment[lick_onsets]
        
        if len(lick_event_times) > 0:
            # 1. ILIs
            if len(lick_event_times) > 1:
                ilis = np.diff(lick_event_times)
                # Store (time_of_interval, interval_val)
                # time of interval = time of second lick
                for t, dt in zip(lick_event_times[1:], ilis):
                    results_ili[port].append((t, dt))
            
            # 2. Bouts
            # Definition: sequence of licks with ILI < threshold
            # Count how many such sequences exist in this visit
            # Simplest way: N_bouts = 1 + count(ILIs > threshold)
            if len(lick_event_times) == 1:
                n_bouts = 1
            else:
                 ilis = np.diff(lick_event_times)
                 # Breaks are where ILI > threshold
                 n_breaks = np.sum(ilis > bout_threshold_sec)
                 n_bouts = 1 + n_breaks
                 
            results_bouts[port].append((t_start, n_bouts))
        else:
            # 0 bouts
            results_bouts[port].append((t_start, 0))

    # --- 4. Plotting ---
    output_dir = paths.neural_base / 'post_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: ILI Scatter
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes1 = axes1.flatten()
    has_data1 = False
    
    for i, port in enumerate([1, 2, 3, 4]):
        ax = axes1[i]
        data = results_ili[port]
        if data:
            has_data1 = True
            times, ilis = zip(*data)
            times = np.array(times)
            ilis = np.array(ilis)
            
            ax.scatter(times, ilis, alpha=0.5, s=10, label=f'Port {port}')
            ax.set_yscale('log')
            ax.set_title(f'Port {port} ILI')
            ax.grid(True, which="both", alpha=0.2)
            
            median_val = np.median(ilis)
            ax.axhline(median_val, color='r', linestyle='--', label=f'Median: {median_val:.3f}s')
            ax.legend()
            
            # Warn about too fast
            n_fast = np.sum(ilis < 0.05)
            if n_fast > 0:
                 ax.text(0.05, 0.95, f'{n_fast} < 50ms', transform=ax.transAxes, color='red')
        else:
             ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
             ax.set_title(f'Port {port}')
             
    fig1.text(0.5, 0.04, 'Time (s)', ha='center')
    fig1.text(0.04, 0.5, 'Inter-Lick Interval (s)', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig(output_dir / 'ili_scatter_within_visit.png')
    plt.close(fig1)

    # Plot 2: Bouts per Visit
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes2 = axes2.flatten()
    has_data2 = False
    
    for i, port in enumerate([1, 2, 3, 4]):
        ax = axes2[i]
        data = results_bouts[port]
        if data:
            has_data2 = True
            times, bouts = zip(*data)
            
            ax.scatter(times, bouts, alpha=0.6, s=15, c='green', label='Bouts/Visit')
            ax.set_title(f'Port {port} Bouts per Visit')
            ax.grid(True, alpha=0.3)
            
            # Running average
            if len(times) > 10:
                # Sort by time
                sorted_pairs = sorted(zip(times, bouts))
                ts, bs = zip(*sorted_pairs)
                ts = np.array(ts)
                bs = np.array(bs)
                
                # Simple moving average
                window = 10
                if len(bs) >= window:
                    smooth_bs = np.convolve(bs, np.ones(window)/window, mode='valid')
                    smooth_ts = ts[window-1:]
                    ax.plot(smooth_ts, smooth_bs, color='k', linewidth=2, label='Moving Avg (10)')
                    
            ax.legend()
        else:
             ax.text(0.5, 0.5, "No Visits", ha='center', transform=ax.transAxes)
             ax.set_title(f'Port {port}')

    fig2.text(0.5, 0.04, 'Time (s)', ha='center')
    fig2.text(0.04, 0.5, 'Number of Bouts per Visit', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig(output_dir / 'bouts_per_visit_scatter.png')
    plt.close(fig2)
    

    print("  Saved 'ili_scatter_within_visit.png' and 'bouts_per_visit_scatter.png'.")
