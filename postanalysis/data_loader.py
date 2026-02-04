import pandas as pd
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import warnings

class DataInspector:
    """
    Robust inspector that captures 'evidence' about file structure 
    rather than trying to perfectly parse everything.
    """

    @staticmethod
    def get_file_info(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"status": "missing"}
        
        info = {
            "status": "found",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower()
        }

        # 1. Capture Raw Text Snippet (Crucial for irregular headers)
        # We read the first 1KB of text to let the Agent "see" the header structure.
        if info['extension'] in ['.csv', '.tsv', '.txt']:
            try:
                with open(path, 'r', errors='replace') as f:
                    # Read first 15 lines as a raw list
                    head_lines = [f.readline().strip() for _ in range(15)]
                    # Remove empty strings from end of read
                    info['raw_snippet'] = [line for line in head_lines if line]
            except Exception as e:
                info['snippet_error'] = str(e)

        # 2. Attempt Structured Inspection (Heuristics)
        try:
            if info['extension'] in ['.csv', '.tsv']:
                DataInspector._inspect_tabular(path, info)
            elif info['extension'] == '.h5':
                DataInspector._inspect_h5(path, info)
            elif info['extension'] == '.mat':
                DataInspector._inspect_mat(path, info)
            elif info['extension'] == '.npy':
                DataInspector._inspect_npy(path, info)
        except Exception as e:
            info['inspection_error'] = str(e)
            
        return info

    @staticmethod
    def _inspect_tabular(path: Path, info: Dict):
        """
        Smart check for DLC (3 headers) vs Standard (1 header) vs Matrix (0 header).
        """
        sep = '\t' if path.suffix == '.tsv' else ','
        
        # Check 1: Is it DeepLabCut? (DLC usually has 'scorer' in first row)
        if 'raw_snippet' in info and len(info['raw_snippet']) > 0:
            first_line = info['raw_snippet'][0].lower()
            if 'scorer' in first_line:
                info['detected_format'] = 'DeepLabCut'
                # Load with multi-index to get bodyparts
                df = pd.read_csv(path, sep=sep, header=[0,1,2], nrows=3)
                info['columns'] = list(df.columns.get_level_values(1).unique()) # Bodyparts
                return

        # Check 2: Try standard load
        try:
            df = pd.read_csv(path, sep=sep, nrows=5)
            # Heuristic: If all columns are numbers (0, 1, 2...), it might be headerless
            if all(isinstance(c, int) or (isinstance(c, str) and c.isdigit()) for c in df.columns):
                info['detected_format'] = 'headerless_matrix'
                info['shape_guess'] = f"N rows x {df.shape[1]} cols"
                info['note'] = "Columns appear to be indices. Row number likely corresponds to Unit ID or Trial ID."
            else:
                info['detected_format'] = 'standard_csv'
                info['columns'] = list(df.columns)
        except:
            info['detected_format'] = 'complex_text'

    @staticmethod
    def _inspect_h5(path: Path, info: Dict):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            info['keys'] = keys
            
            # Check for DLC H5 format
            if 'df_with_missing' in keys:
                info['detected_format'] = 'DeepLabCut_H5'
                # Usually complex to parse fully without pandas, but keys are enough hint
            
    @staticmethod
    def _inspect_npy(path: Path, info: Dict):
        arr = np.load(path, mmap_mode='r')
        info['shape'] = arr.shape
        info['dtype'] = str(arr.dtype)
        info['note'] = "Binary numpy array. If this is spike data, dim 0 is usually events/spikes."

    @staticmethod
    def _inspect_mat(path: Path, info: Dict):
        try:
            mat = sio.loadmat(path, whos_only=True)
            info['variables'] = [x[0] for x in mat]
            info['detected_format'] = 'matlab_standard'
        except NotImplementedError:
            info['detected_format'] = 'matlab_v7.3_hdf5'
            with h5py.File(path, 'r') as f:
                info['variables'] = list(f.keys())

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

    def generate_manifest(self, save_path: Optional[Path] = None) -> Dict:
        """
        Creates a 'Data Passport' for the AI Agent.
        """
        manifest = {
            "mouse_id": self.mouse_id,
            "date": self.date_str,
            "files": {}
        }

        # Inspect every Path object in the dataclass
        for field, value in self.__dict__.items():
            if isinstance(value, Path) and value.exists() and value.is_file():
                print(f"Inspecting {field}...")
                manifest["files"][field] = DataInspector.get_file_info(value)
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(manifest, f, indent=4)
        
        return manifest
    
    def __post_init__(self):
        """Validate that critical paths exist."""
        if self.neural_base and not self.neural_base.exists():
            warnings.warn(f"Neural base path does not exist: {self.neural_base}")
        if self.event_corner and not self.event_corner.exists():
            warnings.warn(f"Event corner file does not exist: {self.event_corner}")
        if self.event_licking and not self.event_licking.exists():
            warnings.warn(f"Event licking file does not exist: {self.event_licking}")
        if self.event_reward and not self.event_reward.exists():
            warnings.warn(f"Event reward file does not exist: {self.event_reward}")
        if self.event_condition_switch and not self.event_condition_switch.exists():
            warnings.warn(f"Event condition switch file does not exist: {self.event_condition_switch}")


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


def find_neural_data(
    base_path: Path,
    mouse_id: str,
    date_mmddyyyy: str
) -> Optional[Path]:
    """
    Find neural data directory.
    
    Expected pattern: {base_path}/{mouse_id}_{date_mmddyyyy}_g0/{mouse_id}_{date_mmddyyyy}_g0_imec0/
    
    Args:
        base_path: Base path for neural data (e.g., "Z:/Koji/Neuropixels")
        mouse_id: Mouse ID (e.g., "1818")
        date_mmddyyyy: Date in MMDDYYYY format (e.g., "09182025")
    
    Returns:
        Path to imec0 directory or None if not found
    """
    session_name_part = f"{mouse_id}_{date_mmddyyyy}"
    
    for session_dir in base_path.glob(f"*{session_name_part}*"):
        if session_dir.is_dir():
            for imec_dir in session_dir.glob("*_imec0"):
                if imec_dir.is_dir():
                    return imec_dir
    
    return None


def find_event_files(
    base_path: Path,
    mouse_id: str,
    date_yyyymmdd: str
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find event CSV files (corner and licking).
    
    Args:
        base_path: Base path for data (e.g., "Z:/Koji/Neuropixels")
        mouse_id: Mouse ID (e.g., "1818")
        date_yyyymmdd: Date in YYYY-MM-DD format (e.g., "2025-09-18")
    
    Returns:
        Tuple of (corner_file, licking_file) paths, or (None, None) if not found
    """
    event_dir = base_path / "Event"
    print(f"  Searching for event files in: {event_dir}")
    if not event_dir.exists():
        event_dir = base_path # Fallback to base path
        print("Event directory not found, using base path instead.")

    corner_file = next(event_dir.glob(f"*{mouse_id}*corner*{date_yyyymmdd}*.csv"), None)
    licking_file = next(event_dir.glob(f"*{mouse_id}*licking*{date_yyyymmdd}*.csv"), None)
    
    # Fallback: Check dataset_config.json if files not found via glob
    if not corner_file or not licking_file:
        try:
            config_path = Path(__file__).resolve().parent.parent / 'dataset_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                for key, entry in config.items():
                    path_str = entry.get('path', '').replace('\\', '/')
                    # Check if path matches mouse and date
                    if mouse_id in path_str and date_yyyymmdd in path_str:
                        full_path = base_path / path_str
                        
                        if not corner_file and 'corner' in path_str.lower() and full_path.exists():
                            corner_file = full_path
                        if not licking_file and 'licking' in path_str.lower() and full_path.exists():
                            licking_file = full_path
        except Exception as e:
            print(f"  Warning: Error searching dataset_config.json: {e}")

    if not corner_file:
        print(f"  Warning: Could not find corner file in {event_dir} matching *{mouse_id}*corner*{date_yyyymmdd}*")
    if not licking_file:
        print(f"  Warning: Could not find licking file in {event_dir} matching *{mouse_id}*licking*{date_yyyymmdd}*")

    return corner_file, licking_file


def find_dlc_file(
    base_path: Path,
    mouse_id: str,
    date_yyyymmdd: str
) -> Optional[Path]:
    """
    Find DLC H5 file.
    
    Args:
        base_path: Base path for data
        mouse_id: Mouse ID
        date_yyyymmdd: Date in YYYY-MM-DD format
    
    Returns:
        Path to DLC H5 file or None if not found
    """
    dlc_dir = base_path / "DLC"
    print(f"  Searching for DLC files in: {dlc_dir}")
    if not dlc_dir.exists():
        dlc_dir = base_path
        print("DLC directory not found, using base path instead.")
    
    dlc_file = next(dlc_dir.glob(f"*{mouse_id}*{date_yyyymmdd}*DLC*.h5"), None)

    # Fallback: Check dataset_config.json if file not found via glob
    if not dlc_file:
        try:
            config_path = Path(__file__).resolve().parent.parent / 'dataset_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                for key, entry in config.items():
                    path_str = entry.get('path', '').replace('\\', '/')
                    if mouse_id in path_str and date_yyyymmdd in path_str and 'DLC' in path_str and path_str.endswith('.h5'):
                        full_path = base_path / path_str
                        if full_path.exists():
                            dlc_file = full_path
                            break
        except Exception as e:
            print(f"  Warning: Error searching dataset_config.json for DLC: {e}")

    return dlc_file


def find_video_file(
    base_path: Path,
    mouse_id: str,
    date_yyyymmdd: str
) -> Optional[Path]:
    """
    Find video AVI file.
    
    Args:
        base_path: Base path for data
        mouse_id: Mouse ID
        date_yyyymmdd: Date in YYYY-MM-DD format
    
    Returns:
        Path to video AVI file or None if not found
    """
    video_dir = base_path / "Video"
    if not video_dir.exists():
        video_dir = base_path
        if not video_dir.exists():
            return None

    return next(video_dir.glob(f"*{mouse_id}*{date_yyyymmdd}*.avi"), None)


def find_tdt_files(
    base_path: Path,
    mouse_id: str,
    date_yymmdd: str
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find TDT files (dFF and RAW).
    
    Args:
        base_path: Base path for data
        mouse_id: Mouse ID
        date_yymmdd: Date in YYMMDD format (e.g., "250918" = 2025-09-18)
    
    Returns:
        Tuple of (dff_file, raw_file) paths, or (None, None) if not found
    """
    tdt_dir = base_path / "TDT" / date_yymmdd
    if not tdt_dir.exists():
        tdt_dir = base_path / "TDT" # Fallback
        if not tdt_dir.exists():
            return None, None
    
    dff_file = next(tdt_dir.glob(f"*{mouse_id}*{date_yymmdd}*dFF*.mat"), None)
    raw_file = next(tdt_dir.glob(f"*Session_Mouse-{date_yymmdd}-*_UnivRAW_offdemod.mat"), None)
    
    if raw_file is None:
        raw_file = next(tdt_dir.glob(f"*UnivRAW_offdemod.mat"), None)
    
    return dff_file, raw_file


def load_session_data(
    mouse_id: str,
    day: str,
    base_path: str = ".",
    neural_base_path: Optional[str] = None
) -> DataPaths:
    """
    Main function to find and load all data files for a session.
    
    Args:
        mouse_id: Mouse ID (e.g., "1818")
        day: Date string in any format (MMDDYYYY, YYYY-MM-DD, YYMMDD, or YYYYMMDD)
        base_path: Base path for all data (default: ".")
        neural_base_path: Optional separate path for neural data (defaults to base_path)
    
    Returns:
        DataPaths object containing all found file paths
    
    Example:
        >>> paths = load_session_data("1818", "09182025", base_path="../DemoData")
        >>> print(paths.kilosort_dir)
        >>> print(paths.event_corner)
    """
    base_path = Path(base_path)
    print(f"  Searching for data in base path: {base_path.resolve()}")
    if neural_base_path is None:
        neural_base_path = base_path
    else:
        neural_base_path = Path(neural_base_path)
    
    # Convert date to all formats
    date_formats = convert_date_formats(day)
    
    # Find neural data
    imec_dir = find_neural_data(
        neural_base_path,
        mouse_id,
        date_formats['mmddyyyy']
    )
    
    # Initialize DataPaths
    paths = DataPaths(
        mouse_id=mouse_id,
        date_str=day,
        date_mmddyyyy=date_formats['mmddyyyy'],
        date_yyyymmdd=date_formats['yyyymmdd'],
        date_yymmdd=date_formats['yymmdd'],
        base_path=base_path,
        neural_base_path=neural_base_path
    )
    
    if imec_dir:
        paths.neural_base = imec_dir
        paths.kilosort_dir = imec_dir / "kilosort4" / "sorter_output"
        paths.lfp_dir = imec_dir / "LFP"
        paths.rastermap_dir = imec_dir / "rastermap"
        paths.analyzer_beh = imec_dir / "analyzer_beh"
        paths.analyzer_tag = imec_dir / "analyzer_tag"
        
        # Check for pre-processed licking times
        lick_npy = paths.kilosort_dir / "licking_seconds.npy"
        if lick_npy.exists():
            paths.licking_seconds = lick_npy
    
    # Find event files
    corner_file, licking_file = find_event_files(
        base_path,
        mouse_id,
        date_formats['yyyymmdd']
    )
    paths.event_corner = corner_file
    paths.event_licking = licking_file
    # Reward and condition switch events are stored in the corner file
    paths.event_reward = corner_file
    paths.event_condition_switch = corner_file
    
    # Find DLC file
    paths.dlc_h5 = find_dlc_file(
        base_path,
        mouse_id,
        date_formats['yyyymmdd']
    )
    
    # Find video file
    paths.video_avi = find_video_file(
        base_path,
        mouse_id,
        date_formats['yyyymmdd']
    )
    
    # Find TDT files
    dff_file, raw_file = find_tdt_files(
        base_path,
        mouse_id,
        date_formats['yymmdd']
    )
    paths.tdt_dff = dff_file
    paths.tdt_raw = raw_file

    if raw_file and raw_file.exists():
        try:
            import scipy.io as sio
            import h5py
            try:
                mat_contents = sio.loadmat(raw_file)
                if 'orig_Fs' in mat_contents:
                    paths.tdt_sampling_rate = float(mat_contents['orig_Fs'])
            except NotImplementedError:
                with h5py.File(raw_file, 'r') as f:
                    if 'orig_Fs' in f:
                        paths.tdt_sampling_rate = float(f['orig_Fs'][0, 0])
        except Exception as e:
            warnings.warn(f"Could not load TDT sampling rate from {raw_file}: {e}")
    
    return paths


def validate_data_paths(paths: DataPaths, required: List[str] = None) -> Dict[str, bool]:
    """
    Validate that required data files exist.
    
    Args:
        paths: DataPaths object
        required: List of required data types. Options:
            - 'neural': Neural data (kilosort)
            - 'events': Event files
            - 'dlc': DLC file
            - 'video': Video file
            - 'tdt': TDT files
    
    Returns:
        Dictionary mapping data type to existence status
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
        print(f"  Γ£ô Base: {paths.neural_base}")
        print(f"  Γ£ô Kilosort: {paths.kilosort_dir}" if paths.kilosort_dir and paths.kilosort_dir.exists() else f"  Γ£ù Kilosort: {paths.kilosort_dir}")
        print(f"  Γ£ô LFP: {paths.lfp_dir}" if paths.lfp_dir and paths.lfp_dir.exists() else f"  Γ£ù LFP: {paths.lfp_dir}")
        print(f"  Γ£ô Analyzer (beh): {paths.analyzer_beh}" if paths.analyzer_beh and paths.analyzer_beh.exists() else f"  Γ£ù Analyzer (beh): {paths.analyzer_beh}")
    else:
        print(f"  Γ£ù Neural data not found")
    
    print(f"\nEvent files:")
    print(f"  {'Γ£ô' if paths.event_corner and paths.event_corner.exists() else 'Γ£ù'} Corner: {paths.event_corner}")
    print(f"  {'Γ£ô' if paths.event_licking and paths.event_licking.exists() else 'Γ£ù'} Licking: {paths.event_licking}")
    print(f"  {'Γ£ô' if paths.event_reward and paths.event_reward.exists() else 'Γ£ù'} Reward: {paths.event_reward}")
    print(f"  {'Γ£ô' if paths.event_condition_switch and paths.event_condition_switch.exists() else 'Γ£ù'} Condition Switch: {paths.event_condition_switch}")
    
    print(f"\nDLC:")
    print(f"  {'Γ£ô' if paths.dlc_h5 and paths.dlc_h5.exists() else 'Γ£ù'} {paths.dlc_h5}")
    
    print(f"\nVideo:")
    print(f"  {'Γ£ô' if paths.video_avi and paths.video_avi.exists() else 'Γ£ù'} {paths.video_avi}")
    
    print(f"\nTDT (Dopamine):")
    print(f"  {'Γ£ô' if paths.tdt_dff and paths.tdt_dff.exists() else 'Γ£ù'} dFF: {paths.tdt_dff}")
    print(f"  {'Γ£ô' if paths.tdt_raw and paths.tdt_raw.exists() else 'Γ£ù'} RAW: {paths.tdt_raw}")
    
    print(f"\n{'='*60}\n")
