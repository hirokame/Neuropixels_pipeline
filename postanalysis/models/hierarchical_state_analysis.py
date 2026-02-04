"""
Analysis script for Hierarchical Neuromotor Model.

This module handles the data loading, preprocessing, and execution of the 
HierarchicalNeuromotorModel defined in hierarchical_state_dynamics.py.

It bridges the gap between raw pipeline data (DLC .h5, TDT .mat) and the 
abstract model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List, Union

# Import your custom modules
from ..data_loader import DataPaths, load_session_data
from .hierarchical_state_dynamics import HierarchicalNeuromotorModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class HierarchicalStateAnalysis:
    """
    Runner class to apply Hierarchical HMM to a specific session.
    """
    
    def __init__(self, data_paths: DataPaths):
        """
        Initialize with session paths.
        """
        self.paths = data_paths
        self.model = None
        self.results = {}
        
        # Hyperparameters
        self.sampling_rate = 60.0  # DLC is 60Hz
        self.strobe_times = None   # Exact timestamps for DLC frames
        
    def load_and_preprocess(self):
        """
        Load DLC, Dopamine, and Neural data, aligning them to common timebase.
        """
        print(f"Loading data for {self.paths.mouse_id}...")
        
        # 1. Load DLC (Micro-scale data)
        if self.paths.dlc_h5 and self.paths.dlc_h5.exists():
            # Loading complex multi-index H5
            self.dlc_df = pd.read_hdf(self.paths.dlc_h5)
            print(f"  Loaded DLC: {self.dlc_df.shape}")
        else:
            raise FileNotFoundError("DLC file required for hierarchical analysis")
            
        # 2. Load Dopamine (Context signal)
        if self.paths.tdt_dff and self.paths.tdt_dff.exists():
            # Assuming standard structure from your TDT_dFF_stage2.m output
            import scipy.io as sio
            mat = sio.loadmat(self.paths.tdt_dff)
            self.dopamine_dff = mat['dFF_array'][0]  # Adjust key as needed
            # Downsample DA to match DLC rate (30Hz) if necessary
            # Placeholder: assuming already aligned or handled here
        else:
            print("  Warning: TDT dFF not found, using synthetic placeholder.")
            self.dopamine_dff = np.zeros(len(self.dlc_df))
            
        # 2a. Load Strobe Times (Exact DLC timing)
        if self.paths.kilosort_dir and (self.paths.kilosort_dir / "strobe_seconds.npy").exists():
            print(f"  Loading strobe_seconds.npy for precise alignment...")
            self.strobe_times = np.load(self.paths.kilosort_dir / "strobe_seconds.npy")
            # Ensure length matches DLC
            if len(self.strobe_times) != len(self.dlc_df):
                print(f"  Warning: Strobe times ({len(self.strobe_times)}) != DLC frames ({len(self.dlc_df)}). Truncating to min.")
                min_len = min(len(self.strobe_times), len(self.dlc_df))
                self.strobe_times = self.strobe_times[:min_len]
                self.dlc_df = self.dlc_df.iloc[:min_len]
        else:
            print("  Warning: strobe_seconds.npy not found. Assuming constant 60Hz.")
            self.strobe_times = np.arange(len(self.dlc_df)) / self.sampling_rate

        # 3. Load Neural Features (PCA of Firing Rates)
        self.neural_features = self._compute_neural_features(len(self.dlc_df), self.strobe_times)
        
    def _compute_neural_features(self, target_length: int, time_vector: np.ndarray) -> np.ndarray:
        """
        Compute neural features (e.g. PC1 of population rate) for the HMM.
        """
        print("  Computing neural features...")
        
        # Method A: Load from Kilosort Spike Times
        # Priority: spike_seconds_adj.npy (Absolute seconds) > spike_seconds.npy
        # Strict Mode: Do NOT use spike_times.npy (indices)
        
        st = None
        
        ks_dir = self.paths.kilosort_dir
        if ks_dir:
            print(f"    Loading spike_seconds_adj.npy (Absolute seconds)...")
            st = np.load(ks_dir / "spike_seconds_adj.npy")
        
        if st is not None:
            try:
                sc = np.load(ks_dir / "spike_clusters.npy")
                
                # Flatten if needed
                st = st.flatten()
                sc = sc.flatten()
                
                # Get unique clusters
                item_ids = np.unique(sc)
                n_units = len(item_ids)
                
                if n_units > 0:
                    print(f"    Found {n_units} units in Kilosort output.")
                    
                    # Create bins matching DLC frames using strobe times
                    # We want one bin per DLC frame.
                    # Use edges: [t0, t1, t2, ...] 
                    # Note: histogram needs n_bins + 1 edges or n_bins (if using range)
                    # We'll use the strobe times as definition of 'bins'.
                    # Ideal: Bin i covers [strobe[i], strobe[i+1]).
                    
                    if len(time_vector) > 1:
                        # Append one estimated end-point to define the last bin
                        dt = np.mean(np.diff(time_vector))
                        bins = np.concatenate([time_vector, [time_vector[-1] + dt]])
                    else:
                        bins = np.arange(0, target_length + 1) / self.sampling_rate
                    
                    # Spikes are already in seconds (Strict Mode)
                    st_sec = st
                    
                    # Bin spikes for each unit
                    spike_counts = np.zeros((target_length, n_units))
                    
                    # Optimize binning? For now, simple loop is robust enough for <1000 units
                    # or use np.histogram logic
                    for i, unit_id in enumerate(item_ids):
                        unit_spikes = st_sec[sc == unit_id]
                        counts, _ = np.histogram(unit_spikes, bins=bins)
                        # Truncate or pad if necessary (histogram should match target_length exactly)
                        spike_counts[:, i] = counts
                    
                    # Smooth/Z-score?
                    # HMM usually likes Gaussian features. 
                    # Let's smooth slightly with a kernel? 
                    # Typically neural trajectories are smoothed.
                    from scipy.ndimage import gaussian_filter1d
                    rates = gaussian_filter1d(spike_counts.astype(float), sigma=2, axis=0) # ~66ms smoothing
                    
                    # PCA
                    if n_units >= 1:
                        pca = PCA(n_components=1)
                        pc1 = pca.fit_transform(StandardScaler().fit_transform(rates))
                        print(f"    Extracted Neural PC1 (Explains {pca.explained_variance_ratio_[0]:.2%} of var)")
                        return pc1.flatten()
            except Exception as e:
                print(f"    Error processing Kilosort data: {e}")

        # Method B: Fallback to Rastermap X_binned
        if self.paths.rastermap_dir and (self.paths.rastermap_dir / "X_binned.npy").exists():
             try:
                 print("    Falling back to Rastermap X_binned...")
                 X = np.load(self.paths.rastermap_dir / "X_binned.npy")
                 # X is usually (n_neurons, n_timebins)
                 if X.shape[0] < X.shape[1]: 
                     X = X.T # Make (Time, Neurons)
                     
                 # Resize to match DLC
                 if len(X) != target_length:
                     from scipy.ndimage import zoom
                     # Resample time axis
                     zoom_factor = target_length / len(X)
                     X = zoom(X, (zoom_factor, 1), order=1)
                     
                 pca = PCA(n_components=1)
                 pc1 = pca.fit_transform(StandardScaler().fit_transform(X))
                 return pc1.flatten()
             except Exception as e:
                 print(f"    Error processing Rastermap data: {e}")

        # Method C: Zeros
        print("    Warning: No neural data found. Using zeros.")
        return np.zeros(target_length) 
        
    def run_analysis(self, 
                     n_micro_states: int = 8, 
                     n_macro_states: int = 3,
                     use_vame: bool = False):
        """
        Instantiate and fit the model.
        """
        print(f"Initializing HNM (Micro={n_micro_states}, Macro={n_macro_states})...")
        self.model = HierarchicalNeuromotorModel(
            n_micro_states=n_micro_states,
            n_macro_states=n_macro_states,
            sampling_rate=self.sampling_rate
        )
        
        # 1. Fit Micro-Dynamics (Movement Motifs)
        print("  Fitting Micro-Dynamics (Kinematics)...")
        self.model.fit_micro_dynamics(self.dlc_df, use_vame_labels=use_vame)
        
        # 2. Fit Macro-Dynamics (Behavioral Modes)
        # We need to ensure DA and Neural data match DLC length
        min_len = min(len(self.dopamine_dff), len(self.dlc_df))
        da_aligned = self.dopamine_dff[:min_len]
        neural_aligned = self.neural_features[:min_len]
        
        print("  Fitting Macro-Dynamics (HMM)...")
        self.model.fit_macro_dynamics(da_aligned, neural_aligned)
        
        self.results['micro_labels'] = self.model.micro_labels
        self.results['macro_labels'] = self.model.macro_labels_upsampled
        
    def plot_results(self, save_dir: Optional[Path] = None):
        """
        Generate and save hierarchy visualization.
        """
        if self.model is None:
            print("Run analysis first.")
            return

        # Use the model's internal plotting function
        min_len = len(self.model.macro_labels_upsampled)
        da_plot = self.dopamine_dff[:min_len]
        
        fig = self.model.plot_hierarchy(da_plot)
        
        if save_dir:
            save_path = save_dir / f"hierarchy_analysis_{self.paths.mouse_id}.png"
            fig.savefig(save_path)
            print(f"Saved plot to {save_path}")
            
        return fig

# Example usage block (if run as script)
if __name__ == "__main__":
    # Example: User would run this on a specific mouse
    mouse = "1818"
    date = "20250918"
    base_path = "Z:/Koji/Neuropixels"
    
    # 1. Load Paths
    paths = load_session_data(mouse, date, base_path)
    
    # 2. Initialize Analysis
    analyzer = HierarchicalStateAnalysis(paths)
    
    # 3. Run
    analyzer.load_and_preprocess()
    analyzer.run_analysis(n_micro_states=10, n_macro_states=3)
    
    # 4. Plot
    analyzer.plot_results(save_dir=Path("./plots"))