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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_dataset_config() -> Dict[str, Any]:
    """Load dataset_config.json from the project root."""
    config_path = Path(__file__).resolve().parent.parent / 'dataset_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"dataset_config.json not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded dataset_config.json with {len(config)} entries")
    return config


def find_config_entry(file_path: Path, config: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Find the configuration entry for a given file path.
    
    Args:
        file_path: Path to the file
        config: Optional pre-loaded config dict (for efficiency)
    
    Returns:
        Config entry dict or None if not found
    """
    if config is None:
        config = load_dataset_config()
    
    if not file_path or not file_path.exists():
        return None
    
    # Normalize path for matching
    file_path_str = str(file_path.as_posix())
    
    for key, value in config.items():
        config_path_str = value.get('path', '').replace('\\', '/')
        if config_path_str and file_path_str.endswith(config_path_str):
            return value
    
    return None


def get_column_name(config: Dict[str, Any], potential_names: List[str]) -> Optional[str]:
    """
    Get the correct column name from config based on potential names.
    
    Args:
        config: Config entry dict
        potential_names: List of potential column name variations
    
    Returns:
        Actual column name from config or None
    """
    if not config or 'columns' not in config:
        return None
    
    # Create case-insensitive mapping
    config_cols = {c['name'].lower(): c['name'] for c in config['columns']}
    
    for name in potential_names:
        if name.lower() in config_cols:
            return config_cols[name.lower()]
    
    return None


def validate_schema(file_path: Path, loaded_data: Any, config: Dict[str, Any]) -> bool:
    """
    Validate that loaded data matches the schema defined in config.
    
    Args:
        file_path: Path to the file
        loaded_data: The loaded data (DataFrame, array, etc.)
        config: Config entry dict
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if 'columns' in config:
        # Validate CSV/TSV columns
        if isinstance(loaded_data, pd.DataFrame):
            expected_cols = {c['name'] for c in config['columns']}
            actual_cols = set(loaded_data.columns)
            
            missing_cols = expected_cols - actual_cols
            if missing_cols:
                error_msg = (
                    f"Schema validation failed for {file_path}:\n"
                    f"Missing columns: {missing_cols}\n"
                    f"Expected: {expected_cols}\n"
                    f"Actual: {actual_cols}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate dtypes if specified
            for col_def in config['columns']:
                col_name = col_def['name']
                if col_name in loaded_data.columns and 'dtype' in col_def:
                    expected_dtype = col_def['dtype']
                    actual_dtype = str(loaded_data[col_name].dtype)
                    # Allow some flexibility in dtype matching
                    if not _dtype_matches(expected_dtype, actual_dtype):
                        logger.warning(
                            f"Column {col_name} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
                        )
    
    if 'dtype' in config and isinstance(loaded_data, np.ndarray):
        expected_dtype = config['dtype']
        actual_dtype = str(loaded_data.dtype)
        if not _dtype_matches(expected_dtype, actual_dtype):
            logger.warning(
                f"Array dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
            )
    
    if 'shape' in config and isinstance(loaded_data, np.ndarray):
        expected_shape = tuple(config['shape'])
        actual_shape = loaded_data.shape
        # Allow flexible shape matching (e.g., if one dimension is variable)
        if len(expected_shape) != len(actual_shape):
            error_msg = (
                f"Schema validation failed for {file_path}:\n"
                f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    return True


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
    Base class for loading individual data streams with proper validation.
    """
    
    def __init__(self, base_path: Path, config: Optional[Dict] = None):
        self.base_path = Path(base_path)
        self.config = config if config is not None else load_dataset_config()
    
    def _get_file_path(self, config_key: str) -> Optional[Path]:
        """Get full file path from config key."""
        if config_key not in self.config:
            return None
        
        config_entry = self.config[config_key]
        path_str = config_entry.get('path', '').replace('\\', '/')
        if not path_str:
            return None
        
        # Handle absolute vs relative paths
        if Path(path_str).is_absolute():
            return Path(path_str)
        else:
            return self.base_path / path_str


class SpikeDataLoader(DataStreamLoader):
    """Load spike data from Kilosort output."""
    
    def load(self, config_key: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load spike times and clusters.
        
        Args:
            config_key: Optional config key for spike_times. If None, searches config.
        
        Returns:
            Dict with keys: 'spike_times_sec', 'spike_clusters', 'unique_clusters'
        """
        # Find spike_times config entry
        if config_key is None:
            spike_times_key = next(
                (k for k, v in self.config.items() 
                 if 'spike_seconds_adj.npy' in v.get('path', '')), 
                None
            )
        else:
            spike_times_key = config_key
        
        if not spike_times_key:
            raise ValueError("No spike time file (spike_seconds_adj.npy) found in dataset_config.json")
        
        spike_times_path = self._get_file_path(spike_times_key)
        if not spike_times_path or not spike_times_path.exists():
            raise FileNotFoundError(f"Spike file not found: {spike_times_path}")
        
        # Load spike_times
        spike_times_config = self.config[spike_times_key]
        spike_times = np.load(spike_times_path, mmap_mode='r')
        
        # Validate schema
        validate_schema(spike_times_path, spike_times, spike_times_config)
        
        sampling_rate = 30000.0
        
        # Determine if conversion is needed
        # If filename contains 'seconds' or dtype is float, assume it's already in seconds
        is_seconds = 'seconds' in str(spike_times_path).lower() or spike_times.dtype.kind == 'f'
        
        if is_seconds:
            spike_times_sec = spike_times.flatten()
            logger.info(f"Loaded spike times from {spike_times_path.name} (units: seconds)")
        else:
            # Convert to seconds (assuming 30kHz from SpikeGLX)
            spike_times_sec = spike_times.flatten() / sampling_rate
            logger.info(f"Loaded spike times from {spike_times_path.name} (units: samples, converted to seconds with fs={sampling_rate})")
        
        # Find and load spike_clusters
        kilosort_dir = spike_times_path.parent
        spike_clusters_path = kilosort_dir / "spike_clusters.npy"
        
        if not spike_clusters_path.exists():
            raise FileNotFoundError(f"spike_clusters.npy not found in {kilosort_dir}")
        
        # Find config for spike_clusters
        spike_clusters_key = next(
            (k for k, v in self.config.items() 
             if 'spike_clusters.npy' in v.get('path', '')), 
            None
        )
        
        if spike_clusters_key:
            spike_clusters_config = self.config[spike_clusters_key]
            spike_clusters = np.load(spike_clusters_path, mmap_mode='r')
            validate_schema(spike_clusters_path, spike_clusters, spike_clusters_config)
        else:
            spike_clusters = np.load(spike_clusters_path, mmap_mode='r')
        
        spike_clusters = spike_clusters.flatten()
        unique_clusters = np.unique(spike_clusters)
        
        logger.info(
            f"Loaded {len(spike_times_sec)} spikes from {len(unique_clusters)} clusters"
        )
        
        # Load unit classification if available
        unit_types = {}
        try:
            classification_path = kilosort_dir / "unit_classification_rulebased.csv"
            if classification_path.exists():
                df_class = pd.read_csv(classification_path)
                # Ensure columns exist
                if 'unit_id' in df_class.columns and 'cell_type' in df_class.columns:
                    unit_types = dict(zip(df_class['unit_id'], df_class['cell_type']))
                    logger.info(f"Loaded {len(unit_types)} unit classifications from {classification_path.name}")
                else:
                    logger.warning(f"Classification file {classification_path.name} missing required columns")
            else:
                 # Try to look in nested folders if not found in root (some versions have different structure)
                 # But based on user path, it is in sorter_output
                 pass
        except Exception as e:
            logger.warning(f"Could not load unit classification: {e}")

        return {
            'spike_times_sec': spike_times_sec,
            'spike_clusters': spike_clusters,
            'unique_clusters': unique_clusters,
            'sampling_rate': sampling_rate,
            'unit_types': unit_types
        }


class DLCDataLoader(DataStreamLoader):
    """Load DLC (DeepLabCut) tracking data."""
    
    def load(self, config_key: Optional[str] = None) -> pd.DataFrame:
        """
        Load DLC H5 file.
        
        Args:
            config_key: Optional config key. If None, searches for DLC H5 file.
        
        Returns:
            DataFrame with MultiIndex columns (scorer, bodypart, coord)
        """
        if config_key is None:
            dlc_key = next(
                (k for k, v in self.config.items() 
                 if 'DLC' in v.get('path', '') and v.get('path', '').endswith('.h5')), 
                None
            )
        else:
            dlc_key = config_key
        
        if not dlc_key:
            raise ValueError("DLC H5 file not found in dataset_config.json")
        
        dlc_path = self._get_file_path(dlc_key)
        if not dlc_path or not dlc_path.exists():
            raise FileNotFoundError(f"DLC file not found: {dlc_path}")
        
        # Load DLC data
        dlc_config = self.config[dlc_key]
        df_dlc = pd.read_hdf(dlc_path)
        
        # Validate header structure if specified in config
        if 'header_structure' in dlc_config:
            expected_structure = dlc_config['header_structure']
            # Check that MultiIndex columns match expected structure
            if isinstance(df_dlc.columns, pd.MultiIndex):
                level_0 = df_dlc.columns.get_level_values(0).unique()
                level_1 = df_dlc.columns.get_level_values(1).unique()
                level_2 = df_dlc.columns.get_level_values(2).unique()
                
                if 'level_0_scorer' in expected_structure:
                    expected_scorer = expected_structure['level_0_scorer'][0]
                    if expected_scorer not in level_0:
                        logger.warning(
                            f"Expected scorer '{expected_scorer}' not found. Found: {level_0}"
                        )
                
                if 'level_1_bodyparts' in expected_structure:
                    expected_bodyparts = set(expected_structure['level_1_bodyparts'])
                    actual_bodyparts = set(level_1)
                    if not expected_bodyparts.issubset(actual_bodyparts):
                        logger.warning(
                            f"Some expected bodyparts missing. Expected: {expected_bodyparts}, "
                            f"Found: {actual_bodyparts}"
                        )
        
        logger.info(f"Loaded DLC data with {len(df_dlc)} frames")
        return df_dlc
    
    def calculate_velocity(
        self, 
        df_dlc: pd.DataFrame,
        bodypart: Optional[str] = None,
        video_fs: int = 60,
        px_per_cm: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate velocity from DLC data.
        
        Args:
            df_dlc: DLC DataFrame
            bodypart: Bodypart to track (if None, uses config or first available)
            video_fs: Video frame rate
            px_per_cm: Pixels per centimeter conversion
        
        Returns:
            Tuple of (velocity, velocity_times)
        """
        from scipy.ndimage import gaussian_filter1d
        
        # Get scorer
        if isinstance(df_dlc.columns, pd.MultiIndex):
            scorer = df_dlc.columns.get_level_values(0).unique()[0]
        else:
            raise ValueError("DLC DataFrame must have MultiIndex columns")
        
        # Determine bodypart
        if bodypart is None:
            # Try to get from config
            dlc_key = next(
                (k for k, v in self.config.items() 
                 if 'DLC' in v.get('path', '') and v.get('path', '').endswith('.h5')), 
                None
            )
            if dlc_key and 'header_structure' in self.config[dlc_key]:
                bodyparts = self.config[dlc_key]['header_structure'].get('level_1_bodyparts', [])
                # Prefer 'body', 'torso', or 'bodycenter'
                for bp in ['body', 'torso', 'bodycenter']:
                    if bp in bodyparts:
                        bodypart = bp
                        break
                if bodypart is None and bodyparts:
                    bodypart = bodyparts[0]
        
        if bodypart is None:
            available_bodyparts = df_dlc.columns.get_level_values(1).unique()
            bodypart = available_bodyparts[0]
            logger.warning(f"No bodypart specified, using first available: {bodypart}")
        
        # Extract x, y coordinates
        x = df_dlc[(scorer, bodypart, 'x')].values
        y = df_dlc[(scorer, bodypart, 'y')].values
        
        # Convert to cm and smooth
        x_cm = x / px_per_cm
        y_cm = y / px_per_cm
        x_cm_smooth = gaussian_filter1d(x_cm, sigma=2)
        y_cm_smooth = gaussian_filter1d(y_cm, sigma=2)
        
        # Try to load strobe times for accurate velocity calculation
        strobe_times = None
        try:
            strobe_loader = StrobeDataLoader(self.base_path, self.config)
            strobe_times = strobe_loader.load()
        except Exception:
            pass
        
        if strobe_times is not None:
            # Validate lengths and handle mismatches
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

        if strobe_times is None:
            # Calculate velocity using fixed frame rate
            velocity = np.sqrt(np.diff(x_cm_smooth)**2 + np.diff(y_cm_smooth)**2) * video_fs
            velocity_times = np.arange(len(velocity)) / video_fs
            logger.info(f"Calculated velocity for {len(velocity)} frames using fixed frame rate {video_fs}")
            
        return velocity, velocity_times

    def get_movement_onsets(
        self,
        df_dlc: Optional[pd.DataFrame] = None,
        config_key: Optional[str] = None,
        video_fs: int = 60,
        px_per_cm: float = 30.0,
        smoothing_window_sec: float = 0.1,
        threshold: float = 2.0
    ) -> np.ndarray:
        """
        Detect movement onset times.
        
        Args:
            df_dlc: Optional DataFrame (if already loaded)
            config_key: Optional config key (to load if df_dlc is None)
            video_fs: Video sampling rate
            px_per_cm: Pixels per cm
            smoothing_window_sec: Window for binning/smoothing velocity (seconds)
            threshold: Velocity threshold (cm/s)
            
        Returns:
            np.array: Times of movement onset (seconds)
        """
        if df_dlc is None:
            if config_key:
                df_dlc = self.load(config_key)
            else:
                df_dlc = self.load() # Default load
        
        if df_dlc.empty:
            logger.warning("Empty DLC dataframe, cannot detect onsets.")
            return np.array([])
            
        velocity, velocity_times = self.calculate_velocity(df_dlc, video_fs=video_fs, px_per_cm=px_per_cm)
        
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
        config_key: Optional[str] = None,
        sync_to_dlc: bool = True,
        dlc_loader: Optional[DLCDataLoader] = None
    ) -> pd.DataFrame:
        """
        Load event CSV file with proper synchronization.
        
        CRITICAL: Event CSVs are truncated. Row 0 might correspond to Frame 305.
        We must align by Frame ID (Index column), not row index.
        
        Args:
            config_key: Config key for the event file
            sync_to_dlc: If True, reindex to match DLC length using Frame ID
            dlc_loader: Optional DLCDataLoader to get full frame range
        
        Returns:
            DataFrame with proper Frame ID alignment
        """
        if config_key is None:
            raise ValueError("config_key must be provided for event files")
        
        event_path = self._get_file_path(config_key)
        if not event_path or not event_path.exists():
            raise FileNotFoundError(f"Event file not found: {event_path}")
        
        event_config = self.config[config_key]
        
        # Determine if file has header
        header = 0 if 'columns' in event_config else None
        
        # Load CSV
        event_df = pd.read_csv(event_path, header=header)
        
        # Validate schema
        validate_schema(event_path, event_df, event_config)
        
        # CRITICAL SYNCHRONIZATION LOGIC
        # Event CSVs are truncated - we must align by Frame ID (Index column)
        index_col = get_column_name(event_config, ['Index', 'index', 'Frame', 'frame'])
        
        if index_col and index_col in event_df.columns:
            # Check for duplicates in Index column before setting as index
            if event_df[index_col].duplicated().any():
                n_dupes = event_df[index_col].duplicated().sum()
                logger.warning(
                    f"Found {n_dupes} duplicate Frame IDs in '{index_col}'. "
                    f"Keeping last occurrence to allow synchronization."
                )
                event_df = event_df.drop_duplicates(subset=[index_col], keep='last')

            # Set Index column as index
            event_df = event_df.set_index(index_col)
            
            if sync_to_dlc and dlc_loader:
                # Get DLC data to determine full frame range
                try:
                    dlc_df = dlc_loader.load()
                    full_frame_range = pd.Index(range(len(dlc_df)), name=index_col)
                    
                    # Reindex to full frame range (missing frames will be NaN)
                    event_df = event_df.reindex(full_frame_range)
                    
                    # Check for expected NaN padding at start (truncation)
                    n_nans_at_start = event_df.iloc[:500].isna().all(axis=1).sum()
                    if n_nans_at_start > 0:
                        logger.info(
                            f"Event data reindexed. First {n_nans_at_start} frames are NaN "
                            f"(expected due to truncation)."
                        )
                except Exception as e:
                    logger.warning(f"Could not sync to DLC: {e}. Using event data as-is.")
            else:
                logger.info("Event data loaded without DLC synchronization")
        else:
            logger.warning(
                f"No 'Index' column found in {event_path}. "
                f"Cannot perform Frame ID synchronization."
            )
        
        return event_df
    
    def get_event_times(
        self, 
        event_df: pd.DataFrame,
        config_key: str,
        time_column: Optional[str] = "Index"
    ) -> np.ndarray:
        """
        Extract event times from event DataFrame.
        
        Args:
            event_df: Event DataFrame
            config_key: Config key for the event file
            time_column: Optional explicit time column name
        
        Returns:
            Array of event times
        """
        event_config = self.config[config_key]
        
        # Try to load strobe times for accurate timing mapping
        try:
            # Attempt to load strobe_seconds.npy using StrobeDataLoader
            strobe_loader = StrobeDataLoader(self.base_path, self.config)
            strobe_times = strobe_loader.load()
            
            # Try to use Index/Frame ID to map to strobe times
            index_col = get_column_name(event_config, ['Index', 'index', 'Frame', 'frame'])
            
            if index_col:
                indices = None
                if event_df.index.name == index_col:
                    indices = event_df.index.values
                elif index_col in event_df.columns:
                    indices = event_df[index_col].values
                
                if indices is not None and pd.api.types.is_numeric_dtype(indices):
                    # Handle potential NaNs in indices
                    if np.issubdtype(indices.dtype, np.floating):
                        indices = indices[~np.isnan(indices)].astype(int)
                    
                    # Check bounds
                    valid_mask = (indices >= 0) & (indices < len(strobe_times))
                    if np.any(valid_mask):
                        valid_indices = indices[valid_mask]
                        logger.info(f"Mapped {len(valid_indices)} events to strobe timestamps using '{index_col}'")
                        return strobe_times[valid_indices]
        except Exception:
            # Strobe data might not be available or configured, proceed with fallback
            pass
        
        if time_column and time_column in event_df.columns:
            event_times = event_df[time_column].dropna().values
        elif time_column and time_column == event_df.index.name:
            event_times = event_df.index.dropna().values
        elif len(event_df.columns) > 0:
            # Fallback to first column
            event_times = event_df.iloc[:, 0].dropna().values
            logger.warning(f"Using first column as time: {event_df.columns[0]}")
        else:
            raise ValueError(f"Could not determine time column for {config_key}")
        
        return event_times

    def load_events_from_path(
        self, 
        file_path: Path,
        sync_to_dlc: bool = True,
        dlc_loader: Optional['DLCDataLoader'] = None,
        filter_onsets: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        High-level method to load events from a file path.
        
        Handles:
        1. Config lookup
        2. Loading with synchronization
        3. Onset filtering (optional)
        4. Timestamp extraction
        
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

        # 1. Find config entry and key
        config_entry = find_config_entry(file_path, self.config)
        if not config_entry:
            # Fallback: try to load as generic CSV if no config found?
            # For now, require config as per strict rules
            logger.warning(f"No config entry found for {file_path}")
            return pd.DataFrame(), np.array([])
            
        config_key = next((k for k, v in self.config.items() if v == config_entry), None)
        if not config_key:
            return pd.DataFrame(), np.array([])

        # 2. Load with sync
        try:
            df = self.load(config_key=config_key, sync_to_dlc=sync_to_dlc, dlc_loader=dlc_loader)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame(), np.array([])

        # 3. Filter onsets
        if filter_onsets:
            df = self._filter_onsets(df, config_entry)

        # 4. Get times
        try:
            times = self.get_event_times(df, config_key)
        except Exception as e:
            logger.warning(f"Could not extract times for {file_path}: {e}")
            times = np.array([])

        return df, times

    def infer_port_id(self, event_df: pd.DataFrame) -> pd.Series:
        """
        Infers port ID (1-4) for reward events based on Corner and Lick columns.
        Logic: Corner1-4 (presence) -> Lick1-4 (backup).
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

    def detect_onsets(self, event_df: pd.DataFrame, config_entry: Dict[str, Any], target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Filters dataframe to return only rows corresponding to state changes (onsets).
        Handles boolean (False->True) and ID changes.
        """
        if event_df.empty:
            return event_df

        # 1. Target Column Logic (Priority)
        if target_column:
            matched_col = None
            if target_column in event_df.columns:
                matched_col = target_column
            else:
                matches = [c for c in event_df.columns if target_column in str(c)]
                if matches:
                    matched_col = matches[0]
            
            if matched_col:
                # Detect rising edge (0->1 or False->True)
                # Handle varying types
                vals = pd.to_numeric(event_df[matched_col], errors='coerce').fillna(0)
                onsets = (vals.diff() == 1) & (vals == 1)
                
                # Special case: First row might be 1 but diff is NaN. 
                # If we assume start of file is not an onset unless 0->1, then diff capture is correct (NaN != 1).
                # But if we treat first row 1 as onset? Typically no.
                
                return event_df[onsets]
            else:
                logger.warning(f"Target column '{target_column}' not found.")
        
        # 2. ID Change Logic (General)
        id_col = get_column_name(config_entry, ['CornerID', 'ID', 'id', 'Corner', 'Port', 'port', 'reward_type', 'Type'])
        
        if id_col and id_col in event_df.columns:
            ids = event_df[id_col]
            prev_ids = ids.shift(1)
            
            if pd.api.types.is_numeric_dtype(ids):
                valid_curr = ids.notna() & (ids != 0)
                changed = ids != prev_ids
                valid_prev = prev_ids.notna() 
                onsets = valid_curr & changed & valid_prev
                return event_df[onsets]
            else:
                valid_curr = ids.notna() & (ids != '')
                changed = ids != prev_ids
                valid_prev = prev_ids.notna()
                onsets = valid_curr & changed & valid_prev
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
                 dlc_loader = DLCDataLoader(self.base_path, self.config)
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
            # Determine file and config
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
                
            config_entry = find_config_entry(event_file, self.config)
            config_key = next((k for k, v in self.config.items() if v == config_entry), None)
            
            if not config_key:
                return np.array([])
                
            # Load Raw
            event_df = self.load(config_key=config_key, sync_to_dlc=(dlc_loader is not None), dlc_loader=dlc_loader)
            
            # Specific filtering logic moved from analyses.py
            
            # 1. Port Inference & Reward Splitting PRE-ONSET logic
            # Some filtering works better on the full trace, some on onsets.
            # "Infer ports BEFORE filtering columns (needed for reward splitting)" - analyses.py
            
            port_series = None
            if 'reward' in event_type_name:
                port_col = get_column_name(config_entry, ['port_id', 'PortID', 'Port', 'id'])
                if port_col and port_col in event_df.columns:
                    port_series = event_df[port_col]
                else:
                    port_series = self.infer_port_id(event_df)
                    
            # 2. Column Filtering (Water)
            target_column = None
            if 'reward' in event_type_name:
                 target_column = 'Water' # Look for Water column
                 water_cols = [c for c in event_df.columns if 'Water' in str(c)]
                 if water_cols:
                     # Keep Water cols + Time/Index
                     cols_to_keep = water_cols.copy()
                     time_col = get_column_name(config_entry, ['Timestamp', 'timestamp', 'Time', 'time', 'Index', 'index'])
                     if time_col and time_col in event_df.columns and time_col not in cols_to_keep:
                         cols_to_keep.append(time_col)
                     
                     # IMPORTANT: If we generated port_series, we need to align it or keep it
                     # If we index event_df, port_series indices must match.
                     event_df = event_df[cols_to_keep]
                 else:
                     target_column = None # Fallback to general onset
    
            # 3. Detect Onsets
            event_df = self.detect_onsets(event_df, config_entry, target_column=target_column)
            
            # 4. Post-Onset Filtering (Splitting First/Second)
            if 'reward' in event_type_name:
                 # Re-align port series to onset dataframe
                 # port_series has same index as original event_df
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
                 id_col = get_column_name(config_entry, ['CornerID', 'ID', 'id', 'Corner'])
                 if id_col and id_col in event_df.columns:
                     ids = event_df[id_col].fillna(0).astype(int)
                 else:
                     ids = self.infer_port_id(event_df) # fallback
                 
                 event_df = event_df[ids != 0]
    
            # 5. Extract Times
            event_times = self.get_event_times(event_df, config_key)
            
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

    def _filter_onsets(self, event_df: pd.DataFrame, config_entry: Dict[str, Any]) -> pd.DataFrame:
        """
        Filters logic transferred from analyses.py to keep loader self-contained.
        Detects rising edges (Boolean) or ID changes.
        """
        if event_df.empty:
            return event_df

        # Fill NaNs with 0/False to handle reindexed data
        df_filled = event_df.fillna(0)

        # 1. Boolean columns
        bool_cols = []
        if config_entry and 'columns' in config_entry:
            bool_cols = [c['name'] for c in config_entry['columns'] 
                         if c.get('dtype') == 'bool' and c['name'] in event_df.columns]
        
        # Fallback detection
        if not bool_cols:
            keywords = ['Corner', 'Lick', 'Beam', 'Stim', 'Water', 'Reward']
            for c in event_df.columns:
                if any(k in str(c) for k in keywords):
                    if pd.api.types.is_bool_dtype(event_df[c]) or \
                       set(df_filled[c].unique()).issubset({0, 1, 0.0, 1.0, False, True}):
                        bool_cols.append(c)

        if bool_cols:
            vals = df_filled[bool_cols].astype(int)
            # diff() == 1 means 0 -> 1 transition
            onsets = (vals.diff().fillna(0) == 1).any(axis=1)
            return event_df[onsets]
        
        # 2. ID columns
        id_col = get_column_name(config_entry, ['CornerID', 'ID', 'id', 'Corner', 'Port', 'port', 'reward_type', 'Type'])
        if id_col and id_col in event_df.columns:
            if pd.api.types.is_numeric_dtype(event_df[id_col]):
                ids = df_filled[id_col]
                onsets = (ids != ids.shift(1).fillna(0)) & (ids != 0)
                return event_df[onsets]
            else:
                vals = event_df[id_col].fillna('')
                onsets = (vals != vals.shift(1).fillna('')) & (vals != '')
                return event_df[onsets]

        return event_df


class PhotometryDataLoader(DataStreamLoader):
    """Load photometry (TDT) data with proper absolute timestamp extraction."""
    
    def load(self, dff_config_key: str, raw_config_key: str) -> Dict[str, np.ndarray]:
        """
        Load photometry data (dFF values and absolute timestamps).
        
        CRITICAL: Do not use normalized timestamps from dFF struct.
        Must extract absolute timestamps from raw H5 file.
        
        Args:
            dff_config_key: Config key for _dFF.mat file
            raw_config_key: Config key for _UnivRAW_offdemod.mat file
        
        Returns:
            Dict with 'dff_values' and 'dff_timestamps' (absolute seconds)
        """
        dff_path = self._get_file_path(dff_config_key)
        raw_path = self._get_file_path(raw_config_key)
        
        if not dff_path or not dff_path.exists():
            raise FileNotFoundError(f"dFF file not found: {dff_path}")
        if not raw_path or not raw_path.exists():
            raise FileNotFoundError(f"RAW file not found: {raw_path}")
        
        # Load dFF values
        dff_config = self.config[dff_config_key]
        dff_struct = sio.loadmat(dff_path)
        
        # Get dFF variable name from config
        if 'variables' in dff_config and 'dFF' in dff_config['variables']:
            dff_var_name = dff_config['variables']['dFF']['name']
        else:
            dff_var_name = 'dFF'
        
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
    
    def load(self, config_key: Optional[str] = None) -> np.ndarray:
        """
        Load strobe_seconds.npy.
        
        This is complete (not truncated) - Row 0 corresponds to Frame 0.
        
        Args:
            config_key: Optional config key. If None, searches for strobe_seconds.
        
        Returns:
            Array of strobe times in seconds
        """
        if config_key is None:
            strobe_key = next(
                (k for k, v in self.config.items() 
                 if 'strobe_seconds.npy' in v.get('path', '')), 
                None
            )
        else:
            strobe_key = config_key
        
        if not strobe_key:
            raise ValueError("strobe_seconds.npy not found in dataset_config.json")
        
        strobe_path = self._get_file_path(strobe_key)
        if not strobe_path or not strobe_path.exists():
            raise FileNotFoundError(f"Strobe file not found: {strobe_path}")
        
        strobe_config = self.config[strobe_key]
        strobe_seconds = np.load(strobe_path, mmap_mode='r')
        
        validate_schema(strobe_path, strobe_seconds, strobe_config)
        
        strobe_seconds = strobe_seconds.flatten()
        
        logger.info(f"Loaded {len(strobe_seconds)} strobe timestamps")
        return strobe_seconds

class LFPDataLoader(DataStreamLoader):
    """
    Load LFP data using SpikeInterface with TPrime synchronization.
    Supports CSD and Bipolar referencing.
    """
    
    def __init__(self, base_path: Path, config: Optional[Dict] = None):
        super().__init__(base_path, config)
        self.extractor = None
        self.channel_ids = None
        self.fs = None
        self.sync_params = None  # {m, c, ratio}
        self.t_start = 0.0
        
        self._init_extractor()
        
    def _init_extractor(self):
        """Initialize SpikeInterface extractor and synchronization parameters."""
        # Finds LFP folder via binary.json config
        binary_key = next((k for k, v in self.config.items() if 'binary.json' in v.get('path', '')), None)
        if not binary_key:
            logger.warning("No 'binary.json' config entry found for LFP. Using generic folder search.")
            # Fallback search
            lfp_candidates = list(self.base_path.rglob("LFP/binary.json"))
            if lfp_candidates:
                lfp_config_path = lfp_candidates[0]
                lfp_folder = lfp_config_path.parent
            else:
                lfp_folder = None
        else:
            lfp_config_path = self._get_file_path(binary_key)
            if lfp_config_path:
                lfp_folder = lfp_config_path.parent
            else:
                lfp_folder = None
            
        if not lfp_folder or not lfp_folder.exists():
             logger.warning(f"LFP folder not found: {lfp_folder}. LFP loading will fail.")
             return

        # Load Extractor
        try:
            # We assume it's a binary recording saved by spikeinterface
            self.extractor = si.load_extractor(lfp_folder)
            
            # Ensure loaded is a BaseRecording
            if not isinstance(self.extractor, si.BaseRecording):
                 raise TypeError("Loaded object is not a RecordingExtractor")
                 
        except Exception as e:
            logger.error(f"Failed to load LFP extractor from {lfp_folder}: {e}")
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
        
        This loads spike_times.npy (raw) and spike_seconds_adj.npy (corrected)
        and fits a robust linear regression. This accounts for:
        1. Clock drift
        2. Hardware decimation (30kHz -> 1kHz)
        3. Start time offsets
        """
        # Find spike times files
        spike_times_key = next((k for k, v in self.config.items() if 'spike_times.npy' in v.get('path', '')), None)
        spike_adj_key = next((k for k, v in self.config.items() if 'spike_seconds_adj.npy' in v.get('path', '')), None)
        
        if not spike_times_key or not spike_adj_key:
            logger.warning("Missing spike timing files for synchronization. Using nominal FS.")
            return

        st_path = self._get_file_path(spike_times_key)
        sa_path = self._get_file_path(spike_adj_key)
        
        if not st_path or not st_path.exists() or not sa_path or not sa_path.exists():
             logger.warning("Spike timing files do not exist.")
             return
             
        # Load sample
        spike_samples = np.load(st_path, mmap_mode='r')
        spike_adj = np.load(sa_path, mmap_mode='r')
        
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
