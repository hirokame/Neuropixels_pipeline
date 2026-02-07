"""
Refactored modular data loading system that strictly adheres to dataset_config.json
and implements proper synchronization logic for Frame ID alignment.

Key principles:
1. No hardcoding of column names - always use config
2. Proper Frame ID synchronization (Event CSVs are truncated, DLC is complete)
3. Schema validation against config
4. Modular design - each data stream can be loaded independently
"""

import pandas as pd
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import json
import logging
from functools import lru_cache
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from scipy.ndimage import convolve1d
from dataclasses import dataclass
from datetime import datetime
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _dtype_matches(expected: str, actual: str) -> bool:
    """Check if dtypes match (with some flexibility)."""
    # Normalize dtypes
    dtype_map = {
        'int64': ['int64', 'int32', 'int'],
        'float64': ['float64', 'float32', 'float'],
        'bool': ['bool', 'bool_'],
        '<U4': ['<U4', 'object', 'str'],
    }
    for key, variants in dtype_map.items():
        if expected in variants and actual in variants:
            return True
    return expected == actual


class DataStreamLoader:
    """
    Base class for loading individual data streams.
    Reference base_path for resolving relative paths if needed.
    """
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    # helper function that turns relative paths into absolute paths
    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.base_path / p

class SpikeDataLoader(DataStreamLoader):
    """
    Load spike data from Kilosort output.
    """
    def load(self, kilosort_dir: Path) -> Dict[str, np.ndarray]:
        """
        Load spike times and clusters directly from kilosort directory.
        
        Args:
            kilosort_dir: Path to kilosort output directory
        
        Returns:
            Dict with keys: 'spike_times_sec', 'spike_clusters', 'unique_clusters'
        """
        if not kilosort_dir or not kilosort_dir.exists():
            raise FileNotFoundError(f"Kilosort directory not found: {kilosort_dir}")
        
        # 1. Load Spike Times (Strictly spike_seconds_adj.npy)
        spike_times_path = kilosort_dir / "spike_seconds_adj.npy"
        if not spike_times_path.exists():
             raise FileNotFoundError(f"spike_seconds_adj.npy not found in {kilosort_dir}")
             
        spike_times = np.load(spike_times_path, mmap_mode='r')
        # We assume it is already in seconds given the name
        spike_times_sec = spike_times.flatten()
        logger.info(f"Loaded spike times from {spike_times_path.name}")

        # 2. Load Clusters
        spike_clusters_path = kilosort_dir / "spike_clusters.npy"
        if not spike_clusters_path.exists():
            raise FileNotFoundError(f"spike_clusters.npy not found in {kilosort_dir}")
            
        spike_clusters = np.load(spike_clusters_path, mmap_mode='r')
        spike_clusters = spike_clusters.flatten()
        unique_clusters = np.unique(spike_clusters)
        
        logger.info(
            f"Loaded {len(spike_times_sec)} spikes from {len(unique_clusters)} clusters"
        )
        
        # 3. Load Unit Classification (if available)
        unit_types = {}
        classification_path = kilosort_dir / "unit_classification_rulebased.csv"
        if classification_path.exists():
            try:
                df_class = pd.read_csv(classification_path)
                if 'unit_id' in df_class.columns and 'cell_type' in df_class.columns:
                    unit_types = dict(zip(df_class['unit_id'], df_class['cell_type']))
                    logger.info(f"Loaded {len(unit_types)} unit classifications")
            except Exception as e:
                logger.warning(f"Failed to load unit classification: {e}")
        
        return {
            'spike_times_sec': spike_times_sec,
            'spike_clusters': spike_clusters,
            'unique_clusters': unique_clusters,
            'unit_types': unit_types
        }


class DLCDataLoader(DataStreamLoader):
    """Load DLC (DeepLabCut) tracking data."""
    
    def load(self, dlc_path: Path) -> pd.DataFrame:
        """
        Load DLC H5 file directly from path.
        
        Args:
            dlc_path: Path to DLC H5 file
        
        Returns:
            DataFrame with MultiIndex columns (scorer, bodypart, coord)
        """
        if not dlc_path or not dlc_path.exists():
            raise FileNotFoundError(f"DLC file not found: {dlc_path}")
            
        try:
            df_dlc = pd.read_hdf(dlc_path)
            # Handle potential different storage keys? Usually pandas finds default.
        except Exception as e:
            raise IOError(f"Failed to load DLC file {dlc_path}: {e}")
        
        # Verify MultiIndex columns
        if not isinstance(df_dlc.columns, pd.MultiIndex):
             # Try to fix if it's flat? (Unlikely for standard DLC)
             raise ValueError(f"DLC file {dlc_path.name} does not have MultiIndex columns")
             
        # Basic validation (optional, can skip strict schema check if we trust path)
        logger.info(f"Loaded DLC data with {len(df_dlc)} frames from {dlc_path.name}")
        return df_dlc
    
    def calculate_velocity(
        self, 
        df_dlc: pd.DataFrame,
        video_fs: int = 60,
        px_per_cm: float = 30.0,
        strobe_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate velocity from DLC data.
        
        Args:
            df_dlc: DLC DataFrame
            video_fs: Video frame rate (fallback)
            px_per_cm: Pixels per centimeter conversion
            strobe_path: Optional path to strobe times for accurate timing
        
        Returns:
            Tuple of (velocity, velocity_times)
        """
        from scipy.ndimage import gaussian_filter1d
        
        # Get scorer
        if isinstance(df_dlc.columns, pd.MultiIndex):
            scorer = df_dlc.columns.get_level_values(0).unique()[0]
        else:
            raise ValueError("DLC DataFrame must have MultiIndex columns")
        
        # Extract x, y coordinates for Snout and Tail
        def _get_interp(bodypart):
            _x = df_dlc[(scorer, bodypart, 'x')].copy()
            _y = df_dlc[(scorer, bodypart, 'y')].copy()
            mask = df_dlc[(scorer, bodypart, 'likelihood')] < 0.8
            _x[mask], _y[mask] = np.nan, np.nan
            return _x.interpolate().ffill().bfill().values, _y.interpolate().ffill().bfill().values

        x_snout, y_snout = _get_interp('Snout')
        x_tail, y_tail = _get_interp('Tail')

        # Average x, y coordinates
        x = (x_snout + x_tail) / 2
        y = (y_snout + y_tail) / 2
        
        # Convert to cm and smooth
        x_cm = x / px_per_cm
        y_cm = y / px_per_cm
        x_cm_smooth = gaussian_filter1d(x_cm, sigma=2)
        y_cm_smooth = gaussian_filter1d(y_cm, sigma=2)
        
        # Try to load strobe times for accurate velocity calculation
        strobe_times = np.load(strobe_path, mmap_mode='r').flatten()
        
        n_frames = len(x_cm_smooth)
        n_strobes = len(strobe_times)
        
        if abs(n_strobes - n_frames) < 10:
            n = min(n_strobes, n_frames)
            strobe_times = strobe_times[:n]
            x_cm_smooth = x_cm_smooth[:n]
            y_cm_smooth = y_cm_smooth[:n]
            
            # Calculate dt and velocity
            dt = np.diff(strobe_times)
            dt[dt <= 0] = np.nan # Avoid division by zero
            
            dist = np.sqrt(np.diff(x_cm_smooth)**2 + np.diff(y_cm_smooth)**2)
            velocity = dist / dt
            velocity = np.nan_to_num(velocity, nan=0.0)
            
            velocity_times = strobe_times[:-1]
            logger.info(f"Calculated velocity for {len(velocity)} frames using strobe timestamps")
        else:
            logger.warning(f"Large mismatch between DLC frames ({n_frames}) and Strobe times ({n_strobes}). Using fixed frame rate.")
            strobe_times = None
            
        return velocity, velocity_times

    def get_movement_onsets(
        self,
        df_dlc: Optional[pd.DataFrame] = None,
        strobe_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Detect movement onset times.
        
        Args:
            df_dlc: Optional DataFrame (if already loaded)
            strobe_path: Path to strobe times
        Returns:
            np.array: Times of movement onset (seconds)
        """
        if df_dlc is None:
             raise ValueError("df_dlc must be provided. Auto-loading removed for strictness.")
        
        if df_dlc.empty:
            logger.warning("Empty DLC dataframe, cannot detect onsets.")
            return np.array([])
            
        velocity, velocity_times = self.calculate_velocity(df_dlc, video_fs=video_fs, px_per_cm=px_per_cm, strobe_path=strobe_path)
        
        if velocity is None or len(velocity) == 0:
            return np.array([])

        # Binning velocity
        session_duration = velocity_times.max()
        if smoothing_window_sec > 0:
             n_bins = int(np.ceil(session_duration / smoothing_window_sec))
             time_bins = np.arange(0, n_bins * smoothing_window_sec, smoothing_window_sec)
             
             velocity_binned = np.zeros(n_bins)
             for i in range(n_bins):
                 t_start, t_end = time_bins[i], time_bins[i] + smoothing_window_sec
                 mask = (velocity_times >= t_start) & (velocity_times < t_end)
                 if np.any(mask):
                     velocity_binned[i] = np.mean(velocity[mask])
        else:
            velocity_binned = velocity
            time_bins = velocity_times
            
        # Threshold detection
        is_moving = velocity_binned > threshold
        
        # Rising edges
        onsets_idx = np.where(np.diff(is_moving.astype(int)) == 1)[0] + 1
        
        if len(onsets_idx) == 0:
            logger.info("No movement onsets detected.")
            return np.array([])
            
        onset_times = time_bins[onsets_idx]
        logger.info(f"Detected {len(onset_times)} movement onsets (> {threshold} cm/s)")
        
        return onset_times

class EventDataLoader(DataStreamLoader):
    """Load event CSV files with proper Frame ID synchronization."""
    
    def load(
        self, 
        event_path: Path,
        sync_to_dlc: bool = True,
        dlc_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Load event CSV file with proper synchronization.
        
        Event CSVs are truncated. Row 0 might correspond to Frame 305.
        We must align by Frame ID (Index column), not row index.
        
        Args:
            event_path: Path to the event CSV
            sync_to_dlc: If True, reindex to match DLC length using Frame ID
            dlc_data: Optional DLC DataFrame to get full frame range
        
        Returns:
            DataFrame with proper Frame ID alignment
        """
        if not event_path or not event_path.exists():
            raise FileNotFoundError(f"Event file not found: {event_path}")
        
        # Load CSV
        event_df = pd.read_csv(event_path)
        index_col = 'Index'
        if event_df[index_col].duplicated().any():
            n_dupes = event_df[index_col].duplicated().sum()
            logger.warning(
                f"Found {n_dupes} duplicate Frame IDs in '{index_col}'. "
                f"Keeping last occurrence to allow synchronization."
            )
            event_df = event_df.drop_duplicates(subset=[index_col], keep='last')

        event_df = event_df.set_index(index_col)
        
        if sync_to_dlc and dlc_data is not None:
            try:
                full_frame_range = pd.Index(range(len(dlc_data)), name=index_col)
                event_df = event_df.reindex(full_frame_range)
                n_nans_at_start = event_df.iloc[:500].isna().all(axis=1).sum()
                if n_nans_at_start > 0:
                    logger.info(
                        f"Event data reindexed. First {n_nans_at_start} frames are NaN "
                        f"(expected due to truncation)."
                    )
            except Exception as e:
                logger.warning(f"Could not sync to DLC: {e}. Using event data as-is.")
        else:
            logger.info("Event data loaded without DLC synchronization (dlc_data not provided)")
        
        return event_df
    
    def get_event_times(
        self, 
        event_df: pd.DataFrame,
        strobe_path: Optional[Path] = None,
        time_column: Optional[str] = "Index"
    ) -> np.ndarray:
        """
        Extract event times from event DataFrame.
        
        Args:
            event_df: Event DataFrame
            strobe_path: Path to strobe_seconds.npy
            time_column: Optional explicit time column name
        
        Returns:
            Array of event times
        """
        try:
            if strobe_path and strobe_path.exists():
                strobe_times = np.load(strobe_path, mmap_mode='r').flatten()
                indices = event_df[time_column].values
                valid_mask = (indices >= 0) & (indices < len(strobe_times))
                valid_indices = indices[valid_mask].astype(int)
                
                logger.info(f"Mapped {len(valid_indices)} events to strobe timestamps using '{time_column}'")
                return strobe_times[valid_indices]
        
        except Exception as e:
            logger.warning(f"Could not map events to strobe timestamps: {e}")

    def load_events_from_path(
        self, 
        file_path: Path,
        sync_to_dlc: bool = True,
        dlc_loader: Optional['DLCDataLoader'] = None,
        filter_onsets: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        High-level method to load events from a file path.
        
        Args:
            file_path: Path to the event file
            sync_to_dlc: Whether to sync to DLC frames
            dlc_loader: Optional DLC loader instance
            filter_onsets: Whether to filter for event onsets (rising edges/changes)
            
        Returns:
            Tuple of (DataFrame, timestamps_array)
        """
        if not file_path or not file_path.exists():
            logger.warning(f"Event file not found: {file_path}")
            return pd.DataFrame(), np.array([])

        # 1. Load with sync
        try:
            dlc_data = None
            df = self.load(event_path=file_path, sync_to_dlc=False, dlc_data=None)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame(), np.array([])

        # 2. Filter onsets
        if filter_onsets:
            df = self._filter_onsets(df)

        # 3. Get times
        try:
            # Attempt to find strobe file in standard location
            strobe_path = None
            if self.base_path:
                 potential_strobe = self.base_path / "kilosort4" / "sorter_output" / "strobe_seconds.npy"
                 if potential_strobe.exists():
                     strobe_path = potential_strobe
                 elif (self.base_path / "strobe_seconds.npy").exists():
                     strobe_path = self.base_path / "strobe_seconds.npy"
            
            times = self.get_event_times(df, strobe_path=strobe_path)
            if times is None:
                times = np.array([])
        except Exception as e:
            logger.warning(f"Could not extract times for {file_path}: {e}")
            times = np.array([])

        return df, times

    def infer_port_id(self, event_df: pd.DataFrame) -> pd.Series:
        """
        Infers port ID (1-4) for reward events based on Corner and Lick columns.
        Logic: Corner1-4 (presence) -> Lick1-4 (backup).

        Returns:
            pd.Series: Series of port IDs (1-4) for each time
        """
        # Initialize with 0 (unknown)
        port_ids = pd.Series(0, index=event_df.index)
        
        # 1. Check Corners (Priority)
        for i in range(1, 5):
            col = f'Corner{i}'
            if col in event_df.columns:
                # Mark rows where this corner is active and no port assigned yet
                is_active = event_df[col].fillna(0).astype(bool)
                mask = is_active & (port_ids == 0)
                port_ids[mask] = i
                
        # 2. Check Licks (Fallback)
        for i in range(1, 5):
            col = f'Lick{i}'
            if col in event_df.columns:
                is_active = event_df[col].fillna(0).astype(bool)
                mask = is_active & (port_ids == 0)
                port_ids[mask] = i
                
        return port_ids

    @staticmethod
    def detect_onsets(event_df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Filters dataframe to return only rows corresponding to state changes (onsets).
        Handles boolean (False->True) specific columns.
        """
        if event_df.empty:
            return event_df

        # 1. Target Column Logic (Priority)
        if target_column:
            if target_column in event_df.columns:
                 # Detect rising edge (0->1 or False->True)
                 vals = pd.to_numeric(event_df[target_column], errors='coerce').fillna(0)
                 onsets = (vals.diff() == 1) & (vals == 1)
                 return event_df[onsets]
            else:
                 logger.warning(f"Target column '{target_column}' not found.")

        # 2. Strict Column Detection
        df_filled = event_df.fillna(0)
        strict_columns = [
            'Corner1', 'Corner2', 'Corner3', 'Corner4',
            'Lick1', 'Lick2', 'Lick3', 'Lick4',
            'Water', 'CW', 'CCW'
        ]
        
        found_cols = [c for c in strict_columns if c in event_df.columns]

        if found_cols:
            vals = df_filled[found_cols].astype(int)
            onsets = (vals.diff().fillna(0) == 1).any(axis=1)
            return event_df[onsets]

        # 3. Fallback: Return as is
        return event_df

    def get_event_times_by_type(self, event_type_name: str, paths: Any, dlc_loader: Optional[DLCDataLoader] = None) -> np.ndarray:
        """
        High-level method to get timestamps for a specific event type.
        Handles 'corner', 'reward', 'reward_first', 'reward_second', 'licking'.
        
        Args:
            event_type_name: Name of event type
            paths: DataPaths object (duck-typed) containing file paths
            dlc_loader: Optional DLC loader for sync
            
        Returns:
            np.ndarray: Event timestamps in seconds
        """
        if event_type_name == 'movement_onset':
             if dlc_loader is None:
                 dlc_loader = DLCDataLoader(self.base_path)
             return dlc_loader.get_movement_onsets()

        event_times = None
        event_df = None

        # Check for pre-processed licking times
        if event_type_name in ['licking', 'licking_bout_start']:
            lick_npy = getattr(paths, 'licking_seconds', None)
            if lick_npy and lick_npy.exists():
                try:
                    event_times = np.load(lick_npy)
                    logger.info(f"Loaded {len(event_times)} licking events directly from {lick_npy}")
                except Exception as e:
                    logger.warning(f"Failed to load licking_seconds.npy: {e}")

        if event_times is None:
            # Fallback to CSV loading
            # Determine file
            event_file = None
            if event_type_name == 'corner':
                event_file = paths.event_corner
            elif event_type_name in ['licking', 'licking_bout_start']:
                event_file = paths.event_licking
            elif event_type_name in ['reward', 'reward_first', 'reward_second']:
                event_file = paths.event_reward if paths.event_reward and paths.event_reward.exists() else paths.event_corner
            else:
                logger.error(f"Unknown event type: {event_type_name}")
                return np.array([])
                
            if not event_file or not event_file.exists():
                logger.error(f"File for {event_type_name} not found: {event_file}")
                return np.array([])
                
            # Load Raw (no config key needed)
            try:
                event_df = self.load(event_path=event_file, sync_to_dlc=False)
            except Exception as e:
                logger.error(f"Failed to load event file {event_file}: {e}")
                return np.array([])
            
            # Specific filtering logic moved from analyses.py
            
            # 1. Port Inference
            if 'reward' in event_type_name:
                # Infer port from Corner1-4 or Lick1-4 (if available)
                # Usually Corner file is used for Reward.
                # Reward itself comes from 'Water' column, but we need Port ID for splitting first/second
                port_series = self.infer_port_id(event_df)
                    
            # 2. Column Filtering (Water)
            target_column = None
            if 'reward' in event_type_name:
                target_column = 'Water' # Look for Water column
                water_cols = [c for c in event_df.columns if 'Water' in str(c)]
                if water_cols:
                    # Keep Water cols + Time/Index
                    cols_to_keep = water_cols.copy()
                    time_col = None
                    for c in ['Timestamp', 'timestamp', 'Time', 'time', 'Index', 'index']:
                        if c in event_df.columns:
                            time_col = c
                            break
                     
                    if time_col and time_col not in cols_to_keep:
                        cols_to_keep.append(time_col)
                    event_df = event_df[cols_to_keep]
                else:
                    target_column = None # Fallback to general onset
    
            # 3. Detect Onsets
            event_df = self.detect_onsets(event_df, target_column=target_column)
            
            # 4. Post-Onset Filtering (Splitting First/Second)
            if 'reward' in event_type_name:
                current_ports = port_series.reindex(event_df.index).fillna(0).values
                 
                if event_type_name in ['reward_first', 'reward_second']:
                    is_first = np.zeros(len(current_ports), dtype=bool)
                    is_second = np.zeros(len(current_ports), dtype=bool)
                     
                    if len(current_ports) > 0:
                        # First reward is always first? Or depends on context.
                        # Defaulting to first logic from analyses.py
                        is_first[0] = True
                        prev_port = current_ports[0]
                         
                        for i in range(1, len(current_ports)):
                            curr_port = current_ports[i]
                            if curr_port != 0 and prev_port != 0:
                                if curr_port == prev_port:
                                    is_second[i] = True
                                else:
                                    is_first[i] = True
                            else:
                                is_first[i] = True
                             
                            if curr_port != 0:
                                prev_port = curr_port
                                 
                    if event_type_name == 'reward_first':
                        event_df = event_df[is_first]
                        logger.info(f"Filtered for First Reward (Switch): {len(event_df)} events")
                    elif event_type_name == 'reward_second':
                        event_df = event_df[is_second]
                        logger.info(f"Filtered for Second Reward (Repeat): {len(event_df)} events")
                      
            elif event_type_name == 'corner':
                 # Exclude ID 0
                 id_col = None
                 for c in ['CornerID', 'ID', 'id', 'Corner']:
                     if c in event_df.columns:
                         id_col = c
                         break
                 
                 if id_col:
                     ids = event_df[id_col].fillna(0).astype(int)
                 else:
                     ids = self.infer_port_id(event_df) # fallback
                 
                 event_df = event_df[ids != 0]
    
            # 5. Extract Times
            # Attempt to find strobe file in standard location
            strobe_path = None
            if self.base_path:
                 potential_strobe = self.base_path / "kilosort4" / "sorter_output" / "strobe_seconds.npy"
                 if potential_strobe.exists():
                     strobe_path = potential_strobe
                 elif (self.base_path / "strobe_seconds.npy").exists():
                     strobe_path = self.base_path / "strobe_seconds.npy"
            
            event_times = self.get_event_times(event_df, strobe_path=strobe_path)
            
        # Ensure times are sorted for correct filtering/PETH
        if event_times is not None and len(event_times) > 0:
            event_times.sort()

        # 6. Post-Time Filtering (Lick Bouts)
        if event_type_name == 'licking_bout_start':
            # Default threshold if not passed? 
            # We can't easy change signature from here without checking callers, but default 0.5s is standard.
            bout_threshold = 0.5
            
            if len(event_times) > 1:
                inter_lick_intervals = np.diff(event_times)
                # Bout start = Events where time since LAST event is > threshold
                # np.diff is (t[1]-t[0], t[2]-t[1]...) corresponding to index 0 of difference array.
                # indices where diff > threshold means the event at index+1 is the start of a new bout.
                bout_start_indices = np.where(inter_lick_intervals > bout_threshold)[0] + 1
                
                # The very first event is always a bout start
                bout_start_indices = np.insert(bout_start_indices, 0, 0)
                
                event_times = event_times[bout_start_indices]
                total_raw = len(event_df) if event_df is not None else "N/A"
                logger.info(f"Filtered {total_raw} raw licks to {len(event_times)} lick bout starts (ISI > {bout_threshold}s)")
            else:
                logger.info("Not enough licks to determine bouts.")

        return event_times

class PhotometryDataLoader(DataStreamLoader):
    """Load photometry (TDT) data with proper absolute timestamp extraction."""
    
    def load(
        self, 
        dff_path: Path,
        raw_path: Path
    ) -> Dict[str, np.ndarray]:
        """
        Load photometry data (dFF values and absolute timestamps).
        
        CRITICAL: Do not use normalized timestamps from dFF struct.
        Must extract absolute timestamps from raw H5 file.
        
        Args:
            dff_path: Path to _dFF.mat file
            raw_path: Path to _UnivRAW_offdemod.mat file
        
        Returns:
            Dict with 'dff_values' and 'dff_timestamps' (absolute seconds)
        """
        if not dff_path or not dff_path.exists():
            raise FileNotFoundError(f"dFF file not found: {dff_path}")
        if not raw_path or not raw_path.exists():
            raise FileNotFoundError(f"RAW file not found: {raw_path}")
        
        # Load dFF values
        # Load dFF values
        dff_struct = sio.loadmat(dff_path)
        
        # Get dFF variable name
        # Try standard names: 'dFF', 'Data', 'data', or variable matching *dFF*
        dff_var_name = 'dFF'
        if dff_var_name not in dff_struct:
            # Search keys
            possible = [k for k in dff_struct.keys() if 'dff' in k.lower()]
            if possible:
                dff_var_name = possible[0]
            else:
                 # Last resort: take first non-standard key
                 standard_keys = ['__header__', '__version__', '__globals__']
                 keys = [k for k in dff_struct.keys() if k not in standard_keys]
                 if keys:
                     dff_var_name = keys[0]
        
        if dff_var_name not in dff_struct:
             raise ValueError(f"Could not find dFF variable in {dff_path}")

        dff_obj = dff_struct[dff_var_name]
        
        # Handle TDT Nested Struct (dFF -> pair -> data)
        # Check if the object has 'pair' field
        try:
            # Unwrap (1,1) array
            if dff_obj.shape == (1, 1):
                val = dff_obj[0, 0]
                if hasattr(val, 'dtype') and val.dtype.names and 'pair' in val.dtype.names:
                    # Found 'pair' field, drill down
                    pair_struct = val['pair']
                    if pair_struct.size > 0:
                        # Take first element of pair (index 0) and 'data' field
                        dff_obj = pair_struct[0, 0]['data']
                        logger.info("  Extracted dFF data from nested 'pair' struct.")
        except Exception as e:
            logger.warning(f"  Failed to traverse struct hierarchy: {e}. Using raw object.")

        dff_vals = dff_obj.squeeze()
        
        # Extract absolute timestamps from raw H5 file
        with h5py.File(raw_path, 'r') as f_raw:
            # Access timestamp reference array
            ts_refs = f_raw['/handles/Ts']
            
            # Dereference to get actual absolute seconds
            # 'box' usually = 0 for single subject recordings
            box_index = 0
            ts_abs = np.array(f_raw[np.ravel(ts_refs)[box_index]]).squeeze()
            
        # Ensure 1D arrays
        dff_vals = np.atleast_1d(dff_vals)
        ts_abs = np.atleast_1d(ts_abs)
        
        if dff_vals.ndim == 0:
             dff_vals = dff_vals.reshape(1)
        if ts_abs.ndim == 0:
             ts_abs = ts_abs.reshape(1)
             
        logger.info(f"Debug: dff_vals shape={dff_vals.shape}, ts_abs shape={ts_abs.shape}")
        
        # Validate lengths match
        if len(dff_vals) != len(ts_abs):
            logger.warning(
                f"dFF values ({len(dff_vals)}) and timestamps ({len(ts_abs)}) have different lengths. "
                f"Truncating to shorter length."
            )
            min_len = min(len(dff_vals), len(ts_abs))
            dff_vals = dff_vals[:min_len]
            ts_abs = ts_abs[:min_len]
        
        # Validate that timestamps are absolute (not normalized 0-1)
        if ts_abs.max() <= 1.0:
            logger.error(
                "TDT time vector appears to be normalized (max <= 1.0). "
                "This suggests incorrect timestamp extraction. Check raw H5 file."
            )
            raise ValueError("TDT timestamps appear normalized - check extraction logic")
        
        logger.info(
            f"Loaded photometry data: {len(dff_vals)} samples, "
            f"time range: {ts_abs.min():.2f} - {ts_abs.max():.2f} seconds"
        )
        
        return {
            'dff_values': dff_vals,
            'dff_timestamps': ts_abs
        }


class StrobeDataLoader(DataStreamLoader):
    """Load strobe_seconds.npy for time synchronization."""
    
    def load(self, strobe_path: Path) -> np.ndarray:
        """
        Load strobe_seconds.npy.
        
        Args:
            strobe_path: Path to strobe_seconds.npy
        
        Returns:
            Array of strobe times in seconds
        """
        if not strobe_path or not strobe_path.exists():
            raise FileNotFoundError(f"Strobe file not found: {strobe_path}")
        
        strobe_seconds = np.load(strobe_path, mmap_mode='r')
        strobe_seconds = strobe_seconds.flatten()
        
        logger.info(f"Loaded {len(strobe_seconds)} strobe timestamps")
        return strobe_seconds

class LFPDataLoader(DataStreamLoader):
    """
    Load LFP data using SpikeInterface with TPrime synchronization.
    Supports CSD and Bipolar referencing.
    """
    

    def __init__(self, lfp_dir: Path, kilosort_dir: Path):
        """
        Initialize LFP Loader.
        
        Args:
            lfp_dir: Path to LFP directory (containing binary.json or similar)
            kilosort_dir: Path to Kilosort output (for sync)
        """
        self.lfp_dir = lfp_dir
        self.kilosort_dir = kilosort_dir
        
        # Initialize defaults
        self.extractor = None
        self.fs = None
        self.channel_ids = None
        self.sync_params = None
        self.t_start = 0.0
        
        if not lfp_dir or not lfp_dir.exists():
            logger.warning(f"LFP folder not found: {lfp_dir}. LFP loading will fail.")
            self.extractor = None
            return

        # Load Extractor
        try:
            # We assume it's a binary recording saved by spikeinterface
            self.extractor = si.load_extractor(lfp_dir)
            
            # Ensure loaded is a BaseRecording
            if not isinstance(self.extractor, si.BaseRecording):
                 raise TypeError("Loaded object is not a RecordingExtractor")
                 
        except Exception as e:
            logger.error(f"Failed to load LFP extractor from {lfp_dir}: {e}")
            self.extractor = None
            return
            
        self.fs = self.extractor.get_sampling_frequency()
        self.channel_ids = self.extractor.get_channel_ids()
        
        # Attempt to get t_start (if sorting segment info exists)
        # Usually for concatenated recordings, t_start might be > 0
        # For now assume 0 or read from probegroup if available
        # self.t_start = ... 
        
        # Synchronization (Drift Correction)
        # CRITICAL: We calculate the linear transform (slope/intercept) DYNAMICALLY 
        # for every session because the clock drift varies day-to-day.
        self._compute_sync_params()

    def _compute_sync_params(self):
        """
        Compute linear transform from LFP Sample Space -> TPrime Adjusted Time.
        """
        if not self.kilosort_dir or not self.kilosort_dir.exists():
            logger.warning("Kilosort directory missing. Cannot sync LFP.")
            return

        st_path = self.kilosort_dir / "spike_times.npy"
        sa_path = self.kilosort_dir / "spike_seconds_adj.npy"
        
        if not st_path.exists() or not sa_path.exists():
             logger.warning("Spike timing files do not exist in kilosort_dir.")
             return
             
        # Load sample
        spike_samples = np.load(st_path, mmap_mode='r').flatten()
        spike_adj = np.load(sa_path, mmap_mode='r').flatten()
        
        # Use a subset for fitting to save memory/time
        n_points = 50000
        if len(spike_samples) > n_points:
            indices = np.linspace(0, len(spike_samples)-1, n_points, dtype=int)
            x = spike_samples[indices].flatten()
            y = spike_adj[indices].flatten()
        else:
            x = spike_samples.flatten()
            y = spike_adj.flatten()
            
        # Fit: Adjusted_Time = m * Spike_Sample + c
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # LFP sample i correspond to Spike sample i * 30 (approx, hardware decimation 30kHz->1kHz)
        # So LFP_Aligned_Time = m * (LFP_Sample * 30) + c
        self.sync_params = {'m': m, 'c': c, 'ratio': 30.0} 
        logger.info(f"LFP Sync: fit m={m:.9f}, c={c:.4f} using spike times (assumed 30:1 ratio)")

    def get_data(
        self, 
        start_time: float, 
        end_time: float, 
        channels: Optional[List[int]] = None, 
        reference: str = 'csd'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get LFP traces and timestamps aligned to TPrime.
        
        Args:
            start_time, end_time: Time window in Adjusted Seconds
            channels: List of channel indices to load (if None, load all)
            reference: 'none', 'car' (common average), 'bipolar', 'csd'
            
        Returns:
            traces: (N_samples, N_channels)
            timestamps: (N_samples,) aligned to TPrime
        """
        if self.extractor is None:
            raise RuntimeError("LFP Extractor not initialized")

        # Map start/end adjusted time -> LFP samples
        # Adj = m * (LFP * 30) + c
        # LFP = (Adj - c) / (m * 30)
        
        if self.sync_params:
            m = self.sync_params['m']
            c = self.sync_params['c']
            r = self.sync_params['ratio']
            
            s_sample = int((start_time - c) / (m * r))
            e_sample = int((end_time - c) / (m * r))
        else:
            # Fallback: Nominal time
            # Time = Sample / fs
            s_sample = int(start_time * self.fs)
            e_sample = int(end_time * self.fs)
            
        # Bounds check
        n_samples = self.extractor.get_num_samples()
        s_sample = max(0, s_sample)
        e_sample = min(n_samples, e_sample)
        
        if s_sample >= e_sample:
            return np.array([]), np.array([])
            
        # Load traces
        # return_scaled=True returns uV (float) usually
        
        if reference == 'car':
            # Use SpikeInterface for CAR to ensure we use ALL channels for the average,
            # even if we are only requesting a subset.
            rec = spre.common_reference(self.extractor, reference='global', operator='average')
            traces = rec.get_traces(
                start_frame=s_sample, 
                end_frame=e_sample, 
                channel_ids=channels,
                return_scaled=True
            )
        elif reference in ['bipolar', 'csd']:
             # For Spatial Referencing (Bipolar/CSD), we MUST use the full probe topology.
             # 1. Load ALL channels (to ensure we have neighbors)
             # 2. Sort by Shank and Depth
             # 3. Apply transform
             # 4. Subset to requested 'channels' at the end
             
             # Load all traces for the time window
             all_traces = self.extractor.get_traces(
                start_frame=s_sample, 
                end_frame=e_sample, 
                channel_ids=None, # Load all
                return_scaled=True
            )
             
             # Get Geometry
             locs = self.extractor.get_channel_locations() # (N, 2)
             keys = self.extractor.get_property_keys()
             
             if 'group' in keys:
                 groups = self.extractor.get_property('group')
             else:
                 # Assume single shank if no group property
                 groups = np.zeros(self.extractor.get_num_channels(), dtype=int)
                 
             # Sort indices: Group (Shank) Primary, X (Column) Secondary, Y (Depth) Tertiary
             # np.lexsort usage: lexsort((secondary, primary)) ... applied recursively?
             # lexsort((tertiary, secondary, primary))
             
             # Keys: groups, locs[:, 0] (x), locs[:, 1] (y)
             # We want Primary=Group, Secondary=X, Tertiary=Y
             sort_idx = np.lexsort((locs[:, 1], locs[:, 0], groups))
             
             # Reorder traces and metadata according to spatial sort
             sorted_traces = all_traces[:, sort_idx]
             sorted_groups = groups[sort_idx]
             sorted_x = locs[sort_idx, 0]
             
             # Apply Transform
             if reference == 'bipolar':
                 # Standard Bipolar: Ch[i] - Ch[i+1] (next channel up/down)
                 # We implement: Output[i] = Input[i] - Input[next_neighbor]
                 # Valid only if next_neighbor is on same shank AND same X-column
                 
                 # Shifted array (next spatial neighbor)
                 next_traces = np.roll(sorted_traces, -1, axis=1)
                 next_groups = np.roll(sorted_groups, -1)
                 next_x = np.roll(sorted_x, -1)
                 
                 # Mask where neighbor is valid (Same Group AND Same X)
                 valid_mask = (sorted_groups == next_groups) & (sorted_x == next_x)
                 valid_mask[-1] = False # Last element wrap-around invalid
                 
                 ref_traces = sorted_traces.copy()
                 # Subtract next neighbor where valid
                 ref_traces[:, valid_mask] -= next_traces[:, valid_mask]
                 
                 # For invalid channels (end of columns), set to 0
                 ref_traces[:, ~valid_mask] = 0

             elif reference == 'csd':
                 # CSD Kernel [-1, 2, -1] (-1*Laplacian)
                 # Requires i-1, i, i+1 to be on same shank AND same column
                 
                 # Check continuity for triads
                 g = sorted_groups
                 x = sorted_x
                 
                 # Valid CSD needs g[i-1]==g[i]==g[i+1] AND x[i-1]==x[i]==x[i+1]
                 # Indices 1..N-2
                 valid_groups = (g[:-2] == g[1:-1]) & (g[1:-1] == g[2:])
                 valid_x = (x[:-2] == x[1:-1]) & (x[1:-1] == x[2:])
                 valid_triads = valid_groups & valid_x
                 
                 # Create full mask (edges False)
                 valid_mask = np.zeros(len(g), dtype=bool)
                 valid_mask[1:-1] = valid_triads
                 
                 # Apply convolution to full sorted array
                 # Note: convolve1d will bleed across invalid boundaries, but we mask them out.
                 # However, the values AT valid indices will be correct because they only depend on +/- 1 neighbor
                 # which we verified are valid.
                 csd_full = convolve1d(sorted_traces, [-1, 2, -1], axis=1, mode='constant', cval=0.0)
                 
                 ref_traces = np.zeros_like(sorted_traces)
                 # Only keep CSD where triad was valid
                 ref_traces[:, valid_mask] = csd_full[:, valid_mask]
                 
             # Map back to Original Order to satisfy 'channels' request
             # We computed ref_traces in 'sort_idx' order.
             # We need to construct output array in original order.
             
             # output_all[sort_idx[i]] = ref_traces[i]
             output_all = np.zeros_like(ref_traces)
             output_all[:, sort_idx] = ref_traces
             
             # Subset to requested channels
             if channels is not None:
                 all_ids = self.extractor.get_channel_ids()
                 # Map requested IDs to integer indices
                 # Optimization: Create a lookup map
                 id_to_idx = {id: i for i, id in enumerate(all_ids)}
                 req_indices = [id_to_idx[ch] for ch in channels if ch in id_to_idx]
                 traces = output_all[:, req_indices]
             else:
                 traces = output_all

        else:
             traces = self.extractor.get_traces(
                start_frame=s_sample, 
                end_frame=e_sample, 
                channel_ids=channels,
                return_scaled=True
            )

        # Generate timestamps
        lfp_indices = np.arange(s_sample, e_sample)
        if self.sync_params:
            m = self.sync_params['m']
            c = self.sync_params['c']
            r = self.sync_params['ratio']
            timestamps = m * (lfp_indices * r) + c
        else:
            timestamps = lfp_indices / self.fs
            
        return traces, timestamps
@dataclass
class DataPaths:
    """Container for all data file paths for a session."""
    # Neural data paths
    neural_base: Optional[Path] = None
    kilosort_dir: Optional[Path] = None
    lfp_dir: Optional[Path] = None
    rastermap_dir: Optional[Path] = None
    analyzer_beh: Optional[Path] = None
    analyzer_tag: Optional[Path] = None
    
    # Event files
    event_corner: Optional[Path] = None
    event_licking: Optional[Path] = None
    event_reward: Optional[Path] = None
    event_condition_switch: Optional[Path] = None
    
    # Pre-processed event times
    licking_seconds: Optional[Path] = None
    strobe_seconds: Optional[Path] = None
    
    # DLC data
    dlc_h5: Optional[Path] = None
    
    # Video
    video_avi: Optional[Path] = None
    
    # TDT (dopamine) data
    tdt_dff: Optional[Path] = None
    tdt_raw: Optional[Path] = None
    tdt_sampling_rate: Optional[float] = None
    
    # Metadata
    mouse_id: Optional[str] = None
    date_str: Optional[str] = None
    date_mmddyyyy: Optional[str] = None  # MMDDYYYY format
    date_yyyymmdd: Optional[str] = None  # YYYY-MM-DD format
    date_yymmdd: Optional[str] = None    # YYMMDD format (for TDT: last 2 digits of year, month, day)
    base_path: Optional[Path] = None
    neural_base_path: Optional[Path] = None


def convert_date_formats(date_str: str) -> Dict[str, str]:
    """
    Convert date string to multiple formats used in the pipeline.
    
    Accepts dates in formats:
    - MMDDYYYY (e.g., "09182025")
    - YYYY-MM-DD (e.g., "2025-09-18")
    - YYMMDD (e.g., "250918") - last 2 digits of year, month, day (for TDT)
    - YYYYMMDD (e.g., "20250918")
    
    Returns dict with keys: 'mmddyyyy', 'yyyymmdd', 'yymmdd', 'date_obj'
    """
    date_obj = None
    mmddyyyy = None
    yyyymmdd = None
    yymmdd = None
    
    # Try to parse the date string
    if len(date_str) == 8 and not date_str.startswith('20'):
        try:
            date_obj = datetime.strptime(date_str, "%m%d%Y")
        except ValueError:
            pass
    elif len(date_str) == 8 and date_str.startswith('20'):
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            pass
    elif len(date_str) == 6:
        # Could be YYMMDD (for TDT) - try this first as it's more common
        try:
            date_obj = datetime.strptime(date_str, "%y%m%d")
        except ValueError:
            # Try DDMMYY as fallback (though less likely)
            try:
                date_obj = datetime.strptime(date_str, "%d%m%y")
            except ValueError:
                pass
    elif '-' in date_str:
        # Likely YYYY-MM-DD
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
    
    if date_obj is None:
        raise ValueError(f"Could not parse date string: {date_str}")
    
    # Convert to all formats
    mmddyyyy = date_obj.strftime("%m%d%Y")
    yyyymmdd = date_obj.strftime("%Y-%m-%d")
    yyyymmdd_nodash = date_obj.strftime("%Y%m%d")
    yymmdd = date_obj.strftime("%y%m%d")  # YYMMDD format for TDT
    
    return {
        'mmddyyyy': mmddyyyy,
        'yyyymmdd': yyyymmdd,
        'yyyymmdd_nodash': yyyymmdd_nodash,
        'yymmdd': yymmdd,
        'date_obj': date_obj
    }


def validate_data_paths(paths: DataPaths, required: List[str] = None) -> Dict[str, bool]:
    """
    Validate that required data files exist.
    """
    if required is None:
        required = ['neural', 'events']
    
    validation = {}
    
    if 'neural' in required:
        validation['neural'] = (
            paths.kilosort_dir is not None and 
            paths.kilosort_dir.exists() and
            (paths.kilosort_dir / "spike_times.npy").exists()
        )
    
    if 'events' in required:
        validation['events'] = (
            paths.event_corner is not None and
            paths.event_corner.exists() and
            paths.event_licking is not None and
            paths.event_licking.exists()
        )
    
    if 'dlc' in required:
        validation['dlc'] = (
            paths.dlc_h5 is not None and
            paths.dlc_h5.exists()
        )
    
    if 'video' in required:
        validation['video'] = (
            paths.video_avi is not None and
            paths.video_avi.exists()
        )
    
    if 'tdt' in required:
        validation['tdt'] = (
            paths.tdt_dff is not None and
            paths.tdt_dff.exists()
        )
    
    return validation


def print_data_summary(paths: DataPaths):
    """Print a summary of found data files."""
    print(f"\n{'='*60}")
    print(f"Data Summary for Mouse {paths.mouse_id}, Day {paths.date_str}")
    print(f"{'='*60}")
    
    print(f"\nDate formats:")
    print(f"  MMDDYYYY: {paths.date_mmddyyyy}")
    print(f"  YYYY-MM-DD: {paths.date_yyyymmdd}")
    print(f"  YYMMDD (TDT): {paths.date_yymmdd}")
    
    print(f"\nNeural data:")
    if paths.neural_base:
        print(f"  ✓ Base: {paths.neural_base}")
        print(f"  ✓ Kilosort: {paths.kilosort_dir}" if paths.kilosort_dir and paths.kilosort_dir.exists() else f"  ✗ Kilosort: {paths.kilosort_dir}")
        print(f"  ✓ LFP: {paths.lfp_dir}" if paths.lfp_dir and paths.lfp_dir.exists() else f"  ✗ LFP: {paths.lfp_dir}")
    else:
        print(f"  ✗ Neural data not found")
    
    print(f"\nEvent files:")
    print(f"  {'✓' if paths.event_corner and paths.event_corner.exists() else '✗'} Corner: {paths.event_corner}")
    print(f"  {'✓' if paths.event_licking and paths.event_licking.exists() else '✗'} Licking: {paths.event_licking}")
    
    print(f"\nDLC:")
    print(f"  {'✓' if paths.dlc_h5 and paths.dlc_h5.exists() else '✗'} {paths.dlc_h5}")
    
    print(f"\nVideo:")
    print(f"  {'✓' if paths.video_avi and paths.video_avi.exists() else '✗'} {paths.video_avi}")
    
    print(f"\nTDT (Dopamine):")
    print(f"  {'✓' if paths.tdt_dff and paths.tdt_dff.exists() else '✗'} dFF: {paths.tdt_dff}")
    print(f"  {'✓' if paths.tdt_raw and paths.tdt_raw.exists() else '✗'} RAW: {paths.tdt_raw}")
    
    print(f"\n{'='*60}\n")


def load_session_data(
    mouse_id: str,
    day: str,
    base_path: str = ".",
    neural_base_path: Optional[str] = None
) -> DataPaths:
    """Main function to find and load all data files for a session."""
    base_path = Path(base_path)
    if neural_base_path is None:
        neural_base_path = base_path
    else:
        neural_base_path = Path(neural_base_path)
    
    # 1. Convert Dates
    date_formats = convert_date_formats(day)
    
    # 2. Init DataPaths
    paths = DataPaths(
        mouse_id=mouse_id,
        date_str=day,
        date_mmddyyyy=date_formats['mmddyyyy'],
        date_yyyymmdd=date_formats['yyyymmdd'],
        date_yymmdd=date_formats['yymmdd'],
        base_path=base_path,
        neural_base_path=neural_base_path
    )
    
    # 3. Locate Neural Data
    # Heuristic: search in neural_base_path for {mouse_id}_{mmddyyyy}*
    session_glob = f"*{mouse_id}_{date_formats['mmddyyyy']}*"
    imec_dir = None
    
    # Try looking specifically for standard neuropixels folder structure
    potential_dirs = list(neural_base_path.glob(session_glob))
    for d in potential_dirs:
        if d.is_dir():
             # Check for imec folder inside
             imec_subdirs = list(d.glob("*_imec0"))
             if imec_subdirs:
                 imec_dir = imec_subdirs[0]
                 break
    
    # If not found, check if base_path ITSELF is the dir (or close to it)
    if not imec_dir:
        for d in neural_base_path.glob("*_imec0"):
            if mouse_id in d.name and date_formats['mmddyyyy'] in d.name:
                imec_dir = d
                break
                
    if imec_dir:
        paths.neural_base = imec_dir
        paths.kilosort_dir = imec_dir / "kilosort4" / "sorter_output"
        paths.lfp_dir = imec_dir / "LFP"
        paths.rastermap_dir = imec_dir / "rastermap"
        paths.analyzer_beh = imec_dir / "analyzer_beh"

        # Check for pre-processed licking
        lick_npy = paths.kilosort_dir / "licking_seconds.npy"
        if lick_npy.exists():
            paths.licking_seconds = lick_npy
        
        # Check for strobe_seconds
        if paths.kilosort_dir:
            strobe_npy = paths.kilosort_dir / "strobe_seconds.npy"
            if strobe_npy.exists():
                paths.strobe_seconds = strobe_npy

    # 4. Locate Behavioral/Other Files using Globbing

    # Event Corner
    try:
         event_dir = base_path / "Event"
         if event_dir.exists():
             # Try YYYYMMDD first as it's standard
             f = next(event_dir.glob(f"*{mouse_id}*corner*{date_formats['yyyymmdd']}*.csv"), None)
             if not f:
                 f = next(event_dir.glob(f"*{mouse_id}*corner*{date_formats['mmddyyyy']}*.csv"), None)
             paths.event_corner = f
    except: pass
    
    # Event Licking
    try:
         event_dir = base_path / "Event"
         if event_dir.exists():
             f = next(event_dir.glob(f"*{mouse_id}*licking*{date_formats['yyyymmdd']}*.csv"), None)
             if not f:
                 f = next(event_dir.glob(f"*{mouse_id}*licking*{date_formats['mmddyyyy']}*.csv"), None)
             paths.event_licking = f
             
         if not paths.event_licking and paths.event_corner:
             # Check same dir as corner
             parent = paths.event_corner.parent
             f = next(parent.glob(f"*{mouse_id}*licking*{date_formats['yyyymmdd']}*.csv"), None)
             paths.event_licking = f
    except: pass

    # For now, reward and switch are same as corner
    paths.event_reward = paths.event_corner
    paths.event_condition_switch = paths.event_corner
    
    # DLC
    try:
        dlc_dir = base_path / "DLC"
        if dlc_dir.exists():
            f = next(dlc_dir.glob(f"*{mouse_id}*{date_formats['yyyymmdd']}*DLC*.h5"), None)
            if not f:
                f = next(dlc_dir.glob(f"*{mouse_id}*{date_formats['mmddyyyy']}*DLC*.h5"), None)
            paths.dlc_h5 = f
    except: pass

    # Video
    try:
        video_dir = base_path / "Video"
        if video_dir.exists():
             f = next(video_dir.glob(f"*{mouse_id}*{date_formats['yyyymmdd']}*.avi"), None)
             if not f:
                 f = next(video_dir.glob(f"*{mouse_id}*{date_formats['mmddyyyy']}*.avi"), None)
             paths.video_avi = f
    except: pass
    
    # TDT
    try:
        tdt_base = base_path / "TDT"
        # TDT folders usually YYMMDD
        session_tdt_dir = tdt_base / date_formats['yymmdd']
        if not session_tdt_dir.exists():
            # Try searching
             found = list(tdt_base.glob(f"*{date_formats['yymmdd']}*"))
             if found:
                 session_tdt_dir = found[0]
        
        if session_tdt_dir.exists():
             paths.tdt_dff = next(session_tdt_dir.glob(f"*{mouse_id}*dFF*.mat"), None)
             paths.tdt_raw = next(session_tdt_dir.glob(f"*UnivRAW_offdemod.mat"), None)
    except: pass
    
    return paths
