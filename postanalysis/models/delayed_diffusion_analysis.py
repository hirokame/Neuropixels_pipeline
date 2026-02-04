"""
Complete Analysis Script for Delayed Diffusion Model with Real Data Integration.

This script demonstrates how to:
1. Load real Neuropixels spike data
2. Simulate LFP propagation using the delayed diffusion model
3. Compare simulated LFP with actual recorded LFP
4. Validate model predictions against experimental data
5. Analyze resonant frequencies during behavior

Author: Neuropixels DA Pipeline Team
Date: 2026-01
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import json
from typing import Dict, Optional, Tuple, List
import warnings

# Import the delayed diffusion model
from diffusion_eq_model import DelayedDiffusionModel
import spikeinterface.core as si


class DelayedDiffusionDataIntegration:
    """
    Integration class for loading real data and validating delayed diffusion model.
    """
    
    def __init__(self, data_root: str):
        """
        Initialize with path to data root directory.
        
        Args:
            data_root: Path to the data directory
        """
        self.data_root = Path(data_root)
        self.spike_data = None
        self.lfp_data = None
        self.channel_positions = None
        self.behavioral_events = None
        
    def load_spike_data(self, session_path: str) -> Dict:
        """
        Load spike times, clusters, and positions from Kilosort output.
        
        Args:
            session_path: Path to session directory (e.g., '1818_09182025_g0/1818_09182025_g0_imec0')
            
        Returns:
            Dictionary with spike data
        """
        session_dir = self.data_root / session_path / 'kilosort4' / 'sorter_output'
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        print(f"Loading spike data from {session_dir}...")
        
        # Load spike data
        spike_times = np.load(session_dir / 'spike_times.npy', mmap_mode='r')
        spike_clusters = np.load(session_dir / 'spike_clusters.npy', mmap_mode='r')
        spike_positions = np.load(session_dir / 'spike_positions.npy', mmap_mode='r')
        
        # Convert spike times to seconds if needed
        # Strict timestamp loading
        # 1. spike_seconds_adj.npy (Absolute adjusted seconds)
        # 2. spike_seconds.npy (Standard seconds)
        # 3. FAIL (Do not assume 30kHz)
        
        spike_seconds_adj_path = session_dir / 'spike_seconds_adj.npy'
        
        if spike_seconds_adj_path.exists():
            print(f"  Loading spike_seconds_adj.npy (Absolute seconds)...")
            spike_seconds = np.load(spike_seconds_adj_path, mmap_mode='r')
        else:
            raise FileNotFoundError(
                f"CRITICAL: No pre-computed spike seconds file found in {session_dir}.\n"
                "Expected 'spike_seconds_adj.npy'.\n"
                "Computation from spike indices via 30kHz assumption is strictly forbidden to ensure accuracy."
            )
        
        self.spike_data = {
            'spike_seconds': spike_seconds,
            'spike_clusters': spike_clusters,
            'spike_positions': spike_positions,
            'n_spikes': len(spike_seconds),
            'n_clusters': len(np.unique(spike_clusters))
        }
        
        print(f"  Loaded {self.spike_data['n_spikes']:,} spikes from {self.spike_data['n_clusters']} clusters")
        print(f"  Recording duration: {spike_seconds[-1]:.2f} seconds")
        
        return self.spike_data
    
    def load_channel_positions(self, session_path: str) -> np.ndarray:
        """
        Load channel positions from Kilosort output.
        
        Args:
            session_path: Path to session directory
            
        Returns:
            Array of channel positions (n_channels, 2) in micrometers
        """
        session_dir = self.data_root / session_path / 'kilosort4' / 'sorter_output'
        
        channel_pos_path = session_dir / 'channel_positions.npy'
        if channel_pos_path.exists():
            self.channel_positions = np.load(channel_pos_path)
            print(f"Loaded {len(self.channel_positions)} channel positions")
            return self.channel_positions
        else:
            warnings.warn(f"Channel positions not found at {channel_pos_path}")
            return None
    
    def load_behavioral_events(self, session_path: str) -> Dict:
        """
        Load behavioral event times (reward, licking, etc.).
        
        Args:
            session_path: Path to session directory
            
        Returns:
            Dictionary with behavioral events
        """
        session_dir = self.data_root / session_path / 'kilosort4' / 'sorter_output'
        
        events = {}
        
        # Load reward times
        reward_path = session_dir / 'reward_seconds.npy'
        if reward_path.exists():
            events['reward_times'] = np.load(reward_path)
            print(f"  Loaded {len(events['reward_times'])} reward events")
        
        # Load licking times
        licking_path = session_dir / 'licking_seconds.npy'
        if licking_path.exists():
            events['licking_times'] = np.load(licking_path)
            print(f"  Loaded {len(events['licking_times'])} licking events")
        
        self.behavioral_events = events
        return events
    
    def load_lfp_data(self, session_path: str, channels: Optional[List[int]] = None) -> Dict:
        """
        Load LFP data from extracted LFP directory.
        
        Args:
            session_path: Path to session directory
            channels: Optional list of channels to load (default: load all)
            
        Returns:
            Dictionary with LFP data
        """
        lfp_dir = self.data_root / session_path / 'LFP'
        
        if not lfp_dir.exists():
            warnings.warn(f"LFP directory not found: {lfp_dir}")
            return None
        
        try:
            print(f"Loading LFP data using SpikeInterface from {lfp_dir}")
            recording = si.load_extractor(str(lfp_dir))
            
            # Get basic metadata
            n_channels = recording.get_num_channels()
            fs = recording.get_sampling_frequency()
            dtype = recording.get_dtype()
            locations = recording.get_channel_locations()
            
            self.lfp_data = {
                'recording_object': recording,
                'binary_path': lfp_dir, # Keep for ref
                'n_channels': n_channels,
                'dtype': dtype,
                'positions': locations,
                'sampling_rate': fs
            }
            
            print(f"  LFP data loaded: {n_channels} channels at {fs} Hz")
            return self.lfp_data
            
        except Exception as e:
            warnings.warn(f"Failed to load LFP with SpikeInterface: {e}")
            return None
    
    def extract_spike_subset(self, time_window: Tuple[float, float], 
                            spatial_window: Optional[Tuple[float, float, float, float]] = None,
                            cluster_subset: Optional[np.ndarray] = None) -> Dict:
        """
        Extract a subset of spikes for a specific time and/or spatial window.
        
        Args:
            time_window: (start_time, end_time) in seconds
            spatial_window: Optional (x_min, x_max, y_min, y_max) in micrometers
            cluster_subset: Optional array of cluster IDs to include
            
        Returns:
            Dictionary with filtered spike data
        """
        if self.spike_data is None:
            raise ValueError("Spike data not loaded. Call load_spike_data() first.")
        
        # Time filtering
        time_mask = (self.spike_data['spike_seconds'] >= time_window[0]) & \
                    (self.spike_data['spike_seconds'] < time_window[1])
        
        # Spatial filtering
        if spatial_window is not None:
            x_min, x_max, y_min, y_max = spatial_window
            positions = self.spike_data['spike_positions']
            spatial_mask = (positions[:, 0] >= x_min) & (positions[:, 0] < x_max) & \
                          (positions[:, 1] >= y_min) & (positions[:, 1] < y_max)
            time_mask = time_mask & spatial_mask
        
        # Cluster filtering
        if cluster_subset is not None:
            cluster_mask = np.isin(self.spike_data['spike_clusters'], cluster_subset)
            time_mask = time_mask & cluster_mask
        
        # Extract subset
        subset = {
            'spike_times': self.spike_data['spike_times'][time_mask],
            'spike_seconds': self.spike_data['spike_seconds'][time_mask],
            'spike_clusters': self.spike_data['spike_clusters'][time_mask],
            'spike_positions': self.spike_data['spike_positions'][time_mask],
            'n_spikes': int(np.sum(time_mask))
        }
        
        print(f"Extracted {subset['n_spikes']:,} spikes in time window {time_window}")
        
        return subset
    
    def compute_spectral_coherence(self, signal1: np.ndarray, signal2: np.ndarray,
                                   fs: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral coherence between two signals.
        
        Args:
            signal1: First signal
            signal2: Second signal
            fs: Sampling frequency in Hz
            
        Returns:
            frequencies, coherence
        """
        freqs, coherence = signal.coherence(signal1, signal2, fs=fs, nperseg=1024)
        return freqs, coherence
    
    def compute_spatial_correlation(self, simulated_lfp: np.ndarray, 
                                   real_lfp: np.ndarray,
                                   positions: np.ndarray) -> Dict:
        """
        Compute spatial correlation between simulated and real LFP.
        
        Args:
            simulated_lfp: Simulated LFP at different positions (n_positions, n_timepoints)
            real_lfp: Real LFP at same positions
            positions: Spatial positions (n_positions, 2)
            
        Returns:
            Dictionary with correlation metrics
        """
        # Compute correlation at each position
        correlations = np.zeros(len(positions))
        for i in range(len(positions)):
            if len(simulated_lfp[i]) > 0 and len(real_lfp[i]) > 0:
                correlations[i] = np.corrcoef(simulated_lfp[i], real_lfp[i])[0, 1]
        
        return {
            'correlations': correlations,
            'mean_correlation': np.nanmean(correlations),
            'std_correlation': np.nanstd(correlations),
            'positions': positions
        }

    def validate_anisotropy(self, snapshots: np.ndarray, model_dx: float) -> float:
        """
        Validate Model 2 Anisotropy: Calculate spread ratio along Y vs X.
        
        Args:
            snapshots: LFP snapshots (n_snaps, nx, ny)
            model_dx: grid resolution in um
            
        Returns:
            anisotropy_ratio: sigma_y / sigma_x
        """
        # Collapse time: mean absolute map
        mean_map = np.mean(np.abs(snapshots), axis=0) # (nx, ny)
        
        # Marginal distributions
        profile_x = np.mean(mean_map, axis=1) # Average over y, profile along x
        profile_y = np.mean(mean_map, axis=0) # Average over x, profile along y
        
        # Fit Gaussian to estimate width (sigma)
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
            
        try:
            x_axis = np.arange(len(profile_x)) * model_dx
            popt_x, _ = curve_fit(gaussian, x_axis, profile_x, p0=[np.max(profile_x), np.mean(x_axis), 500.0])
            sigma_x = abs(popt_x[2])
            
            y_axis = np.arange(len(profile_y)) * model_dx
            popt_y, _ = curve_fit(gaussian, y_axis, profile_y, p0=[np.max(profile_y), np.mean(y_axis), 1500.0])
            sigma_y = abs(popt_y[2])
            
            ratio = sigma_y / sigma_x
            return ratio, sigma_x, sigma_y
        except Exception as e:
            print(f"  Anisotropy fitting failed: {e}")
            return np.nan, np.nan, np.nan

    def validate_spatial_decay(self, snapshots: np.ndarray, model_dx: float) -> Tuple[float, float]:
        """
        Validate Model 3 Spatial Decay: Fit exponential to radial profile.
        V(r) ~ exp(-r/lambda)
        """
        # Use max projection to capture peak amplitude at each location
        max_map = np.max(np.abs(snapshots), axis=0)
        
        # Find peak location
        idx = np.unravel_index(np.argmax(max_map), max_map.shape)
        center_x, center_y = idx[0] * model_dx, idx[1] * model_dx
        
        # Gather (r, V) points
        r_vals = []
        v_vals = []
        nx, ny = max_map.shape
        
        # Subsample for speed
        step = max(1, int(nx / 50))
        for i in range(0, nx, step):
            for j in range(0, ny, step):
                x, y = i * model_dx, j * model_dx
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                r_vals.append(r)
                v_vals.append(max_map[i, j])
                
        r_vals = np.array(r_vals)
        v_vals = np.array(v_vals)
        
        # Fit exponential decay: V = A * exp(-r/lambda) + B
        def exp_decay(r, A, lam, B):
            return A * np.exp(-r / lam) + B
            
        try:
            # Initial guess: A=max, lam=200, B=0
            popt, _ = curve_fit(exp_decay, r_vals, v_vals, p0=[np.max(v_vals), 200.0, 0.0], bounds=([0, 10, -np.inf], [np.inf, 2000, np.inf]))
            lambda_um = popt[1]
            return lambda_um, (r_vals, v_vals, exp_decay(r_vals, *popt))
        except Exception as e:
            print(f"  Spatial decay fitting failed: {e}")
            return np.nan, (None, None, None)


def analyze_delayed_diffusion_with_real_data(data_root: str, 
                                             session_path: str,
                                             time_window: Tuple[float, float],
                                             output_dir: str = 'model_outputs'):
    """
    Complete analysis pipeline for delayed diffusion model with real data.
    
    Args:
        data_root: Path to data root directory
        session_path: Relative path to session
        time_window: Time window to analyze (start, end) in seconds
        output_dir: Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*80)
    print("Delayed Diffusion Model: Real Data Analysis")
    print("="*80)
    
    # Initialize data loader
    data_loader = DelayedDiffusionDataIntegration(data_root)
    
    # Load data
    spike_data = None
    channel_positions = None
    behavioral_events = None
    lfp_data = None
    
    try:
        spike_data = data_loader.load_spike_data(session_path)
        channel_positions = data_loader.load_channel_positions(session_path)
        behavioral_events = data_loader.load_behavioral_events(session_path)
        lfp_data = data_loader.load_lfp_data(session_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Continuing with simulation using synthetic data...")
    
    # Extract spike subset for analysis window
    if spike_data is not None:
        spike_subset = data_loader.extract_spike_subset(time_window)
    else:
        # Generate synthetic spike data for demonstration
        print("\nGenerating synthetic spike data for demonstration...")
        n_spikes = 5000
        # Generate random spikes reflecting NP 2.0 geometry
        # x: 0-800um (spanning 4 shanks), y: 0-3840um (probe length)
        random_x = np.random.uniform(0, 800, n_spikes)
        random_y = np.random.uniform(0, 3840, n_spikes)
        spike_subset = {
            'spike_seconds': np.sort(np.random.uniform(time_window[0], time_window[1], n_spikes)),
            'spike_positions': np.column_stack((random_x, random_y)),
            'n_spikes': n_spikes
        }
    
    # Initialize delayed diffusion model
    print("\n" + "="*80)
    print("Initializing Delayed Diffusion Model")
    print("="*80)
    
    # Estimate spatial grid size from probe dimensions
    # User Specification:
    # - 4 shanks spaced 250 μm apart (X: 0, 250, 500, 750 μm)
    # - Recording depth: ~0.8 mm (800 μm) vertical band
    # - "0.8mm * 0.8mm square box"
    # - High density vertical spacing (16 μm)
    
    # We need finer resolution than 100μm to capture the vertical structure
    resolution_um = 20.0 
    
    # Grid Coverage:
    # X: Need to cover 0 to 750 μm + margins -> Let's do 1200 μm width
    # Y: Need to cover 0 to 800 μm + margins -> Let's do 1200 μm height
    
    grid_size_x = int(1500 / resolution_um)  # 75 grid points
    grid_size_y = int(1500 / resolution_um)  # 75 grid points
    
    # Calculate active ROI from spike data
    if spike_subset['n_spikes'] > 0:
        y_min = np.min(spike_subset['spike_positions'][:, 1])
        y_max = np.max(spike_subset['spike_positions'][:, 1])
        data_y_center = (y_min + y_max) / 2
    else:
        # Default fallback
        data_y_center = 1000.0
        
    print(f"  Spike Y-range: {y_min:.1f} to {y_max:.1f} (Center: {data_y_center:.1f})")

    # Define simulation grid height (fixed ROI size)
    # 1500um height covers the ~800um recording range comfortably
    grid_height_um = grid_size_y * resolution_um 
    
    # Calculate coordinate offset to center grid on data
    # Real Y = Model Y + y_origin_offset
    # Model Y = Real Y - y_origin_offset
    y_origin_offset = data_y_center - (grid_height_um / 2)
    print(f"  Grid origin offset: {y_origin_offset:.1f} μm")

    model = DelayedDiffusionModel(
        spatial_grid_size=(grid_size_x, grid_size_y),
        spatial_resolution_um=resolution_um, 
        time_step_ms=0.05,  # Reduced time step for stability with 20um grid
        diffusion_coefficient=1.0,
        conduction_delay_ms=5.0,
        anisotropy_ratio=3.0  # Faster along shank direction
    )
    
    # Run spike-driven LFP propagation simulation
    print("\n" + "="*80)
    print("Simulating LFP Propagation from Spike Data")
    print("="*80)
    
    duration = time_window[1] - time_window[0]
    
    # Define recording positions (sample channels along probe)
    # Transform spike positions to model coordinates
    model_spike_positions = spike_subset['spike_positions'].copy()
    model_spike_positions[:, 1] -= y_origin_offset
    
    # Define recording positions
    if channel_positions is not None:
        # Use actual channel positions
        recording_positions = channel_positions[::10][:20]  # Sample every 10th channel
        # Transform custom recording positions
        recording_positions[:, 1] -= y_origin_offset
    else:
        # Use synthetic positions along probe (NP 2.0 4-shank geometry)
        # We generate them centered on the data_y_center we found
        
        # Shanks at x = 250, 500, 750, 1000 microns (relative to model grid)
        # Note: X is usually 0-based in common conventions, if actual data has large X, we should offset that too.
        # For now, we assume X matches the model's 0-1500 range or is small.
        # But let's check spike X too:
        if spike_subset['n_spikes'] > 0:
             x_min = np.min(spike_subset['spike_positions'][:, 0])
             # If spikes are far shifted in X, shift model X too? 
             # For simpler logic, we keep X fixed (assuming 0-1000 range) or apply simple offset if needed.
             # User said "but x axis is quite sparse that we only have 4 sample point", probably 0, 250...
             pass

        x_offset = 250
        y_offset_model = 250 # Center within model grid vertically (relative to bottom 0)
        # Actually, let's just span the center of the grid
        
        shank_pitch = 250 # microns
        shanks_relative_x = [0, 250, 500, 750]
        
        # Vertical coverage: 96 channels over ~800 microns
        # Generate points centered in the grid vertically (model Y ~ 750)
        grid_center_y = grid_height_um / 2
        y_range_model = np.linspace(grid_center_y - 400, grid_center_y + 400, 20)
        
        recording_positions = []
        for x_rel in shanks_relative_x:
            for y in y_range_model:
                recording_positions.append([x_offset + x_rel, y])
                
        recording_positions = np.array(recording_positions)
    
    # Run simulation
    # Ensure spike positions are also within or near this grid
    # If using synthetic spikes, regenerate them to match this new small ROI if needed
    if spike_data is None:
         # Update synthetic spikes to focus on the centered ROI (model coords)
         n_spikes = 5000
         random_x = np.random.uniform(200, 1050, n_spikes)
         random_y = np.random.uniform(grid_height_um/2 - 400, grid_height_um/2 + 400, n_spikes)
         model_spike_positions = np.column_stack((random_x, random_y))
         
         # Also update the original 'spike_subset' for consistency in return values
         # (restoring absolute coordinates)
         spike_subset['spike_positions'] = model_spike_positions.copy()
         spike_subset['spike_positions'][:, 1] += y_origin_offset
    
    # Scale spike amplitude to preserve energy density on finer grid
    # Reference resolution was 100um, Amplitude 0.1
    # New resolution is 20um. Scaling factor = (20/100)^2 = 0.04
    scaled_amplitude = 0.1 * (model.dx / 100.0)**2
         
    results = model.simulate_spike_driven_propagation(
        spike_times=spike_subset['spike_seconds'] - time_window[0],  # Relative to window start
        spike_positions=model_spike_positions, # Use transformed positions
        duration_sec=duration,
        recording_positions=recording_positions,
        spike_amplitude=scaled_amplitude
    )
    
    # Perform Validation Analyses
    print("\n" + "="*80)
    print("Running Validation Tests (Tests 2, 3, 4)")
    print("="*80)
    
    # Test 2: Anisotropy
    anisotropy_ratio, sigma_x, sigma_y = data_loader.validate_anisotropy(results['V_snapshots'], model.dx)
    print(f"  Test 2 Anisotropy: Ratio = {anisotropy_ratio:.2f} (Sigma Y: {sigma_y:.1f}um / Sigma X: {sigma_x:.1f}um)")
    
    # Test 3: Spatial Decay
    lambda_um, decay_data = data_loader.validate_spatial_decay(results['V_snapshots'], model.dx)
    print(f"  Test 3 Spatial Decay: Lambda = {lambda_um:.1f} μm (Space Constant)")
    
    # Test 4: Real LFP Coherence
    coherence_stats = {}
    real_lfp_traces = None
    
    if lfp_data is not None:
        print("  Test 4: Extracting Real LFP for coherence comparison...")
        try:
            recording = lfp_data['recording_object']
            fs_lfp = recording.get_sampling_frequency()
            
            # Calculate time indices
            start_frame = int(time_window[0] * fs_lfp)
            end_frame = int(time_window[1] * fs_lfp)
            
            # Channel Mapping: Find nearest real channels to our simulated positions
            # Simulated positions are in 'recording_positions' (which might be transformed/centered)
            # Important: recording_positions contains model coords. But to find real channels,
            # we need the REAL coordinates.
            # If we used real `channel_positions` for the simulation, we should map those back.
            # Reconstruct real coordinates of the simulation sites:
            sim_sites_real_y = recording_positions[:, 1] + y_origin_offset
            # X coordinates are assumed matched or relative. 
            # If recording_positions came from channel_positions, they are consistent.
            
            real_locs = recording.get_channel_locations()
            
            nearest_ch_ids = []
            valid_indices = [] # Indices in simulation array that we found matches for
            
            print(f"  Mapping {len(recording_positions)} simulation sites to {recording.get_num_channels()} real channels...")
            
            for i, (sim_x, sim_y) in enumerate(zip(recording_positions[:, 0], sim_sites_real_y)):
                # dist = sqrt((x1-x2)^2 + (y1-y2)^2)
                dists = np.sqrt((real_locs[:, 0] - sim_x)**2 + (real_locs[:, 1] - sim_y)**2)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                
                if min_dist < 50.0: # Only accept if within 50um (fairly close)
                    nearest_ch_ids.append(recording.channel_ids[min_idx])
                    valid_indices.append(i)
            
            if len(nearest_ch_ids) > 0:
                print(f"  Found {len(nearest_ch_ids)} matching channels.")
                
                # Load traces
                traces = recording.get_traces(channel_ids=nearest_ch_ids, 
                                            start_frame=start_frame, 
                                            end_frame=end_frame,
                                            return_scaled=True)
                
                # Handle NaNs (common in raw recordings for bad channels/segments)
                if np.isnan(traces).any():
                    print(f"  WARNING: Found NaNs in LFP traces. Replacing with 0.")
                    traces = np.nan_to_num(traces, nan=0.0)

                print(f"  Extracted traces shape: {traces.shape}")
                print(f"  Trace stats - Mean: {np.mean(traces):.4f}, Std: {np.std(traces):.4f}, Max: {np.max(traces):.4f}")
                
                if np.std(traces) == 0:
                    print("  WARNING: Extracted LFP traces are flat (all same value)!")
                
                # Traces shape: (n_samples, n_channels)
                # Ensure length matches simulation (resample if fs differs)
                # Model fs = 1000/dt = 20000 Hz (dt=0.05ms)
                # LFP fs is usually 2500 Hz.
                # For coherence in Theta (4-12Hz) or Beta (15-30Hz), we need good frequency resolution.
                # High fs (20000) with default nperseg (256) gives bin size ~78Hz.
                # We MUST downsample or increase nperseg. Downsampling is more efficient.
                
                target_fs = 1000.0
                step = int((1000.0/model.dt) / target_fs) # e.g. 20000 / 1000 = 20
                if step < 1: step = 1
                
                sim_signals_subset = results['lfp_signals'][::step, valid_indices]
                actual_sim_fs = (1000.0/model.dt) / step
                
                # Resample real LFP to match this downsampled rate
                # real traces: n_samples at recording.get_sampling_frequency()
                # we want: same number of samples as sim_signals_subset
                target_len = sim_signals_subset.shape[0]
                real_resampled = signal.resample(traces, target_len, axis=0) # Resample along time
                
                # Compute Coherence
                # Use nperseg = fs to get ~1Hz resolution
                nperseg = int(actual_sim_fs) # 1 sec window
                if nperseg > target_len: nperseg = target_len
                
                f_coh, Cxy = signal.coherence(sim_signals_subset[:, 0], real_resampled[:, 0], fs=actual_sim_fs, nperseg=nperseg)
                
                # Band coherence
                theta_mask = (f_coh >= 4) & (f_coh <= 12)
                beta_mask = (f_coh >= 15) & (f_coh <= 30)
                
                # Handle empty masks or NaNs
                if np.any(theta_mask):
                    coherence_stats['theta'] = np.nanmean(Cxy[theta_mask])
                else:
                    coherence_stats['theta'] = 0.0
                    
                if np.any(beta_mask):
                    coherence_stats['beta'] = np.nanmean(Cxy[beta_mask])
                else:
                    coherence_stats['beta'] = 0.0
                
                print(f"  Real Data Coherence (fs={actual_sim_fs:.1f}Hz) - Theta: {coherence_stats['theta']:.2f}, Beta: {coherence_stats['beta']:.2f}")
            else:
                 print("  No close channel matches found between simulation sites and LFP probe geometry.")

        except Exception as e:
            print(f"  Failed to load real LFP: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Test 4 Skipped: Real LFP data not available.")

    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # Define extent for absolute coordinates
    x_extent = [0, model.nx*model.dx] # Assuming X is local/relative 0-1500
    y_extent = [y_origin_offset, y_origin_offset + model.ny*model.dx]
    full_extent = x_extent + y_extent 

    # 1. LFP propagation snapshots
    fig1 = model.plot_propagation_snapshots(results, 
                                           save_path=str(output_path / 'lfp_propagation.png'),
                                           extent=full_extent)
    print(f"  Saved: {output_path / 'lfp_propagation.png'}")
    
    # 2. LFP timeseries (Comparison if real data exists)
    fig2, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    channel_indices = [0, 5, 10, 15, 19]
    
    for i, ch_idx in enumerate(channel_indices):
        if ch_idx < results['lfp_signals'].shape[1]:
            # Plot Simulated
            t_vec = results['time']
            axes[i].plot(t_vec, results['lfp_signals'][:, ch_idx], 'r-', linewidth=1, label='Simulated LFP', alpha=0.8)
            
            # Plot Real if available
            if real_lfp_traces is not None and ch_idx < real_lfp_traces.shape[1]:
                # Normalize real trace to match scale roughly for visual comparison
                real_trace = real_lfp_traces[:, ch_idx]
                real_trace = real_trace - np.mean(real_trace)
                sim_std = np.std(results['lfp_signals'][:, ch_idx])
                real_std = np.std(real_trace)
                if real_std > 0:
                     real_trace = real_trace * (sim_std / real_std)
                
                # Resample real trace to match simulation time base if needed
                # For now assume fs same or just plot against its own time
                t_real = np.linspace(0, duration, len(real_trace))
                axes[i].plot(t_real, real_trace, 'k-', linewidth=0.5, label='Real LFP (Experiment)', alpha=0.4)
            
            axes[i].set_ylabel(f'Ch {ch_idx}\n(mV)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(loc='upper right', fontsize=9)
            
            # Mark behavioral events
            if behavioral_events and 'reward_times' in behavioral_events:
                reward_times_in_window = behavioral_events['reward_times'][
                    (behavioral_events['reward_times'] >= time_window[0]) & 
                    (behavioral_events['reward_times'] < time_window[1])
                ] - time_window[0]
                for rt in reward_times_in_window:
                    axes[i].axvline(rt, color='blue', alpha=0.3, linestyle='--', linewidth=1)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig2.suptitle('Model Validation: Simulated vs Real LFP Traces', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'lfp_timeseries_validation.png', dpi=150)
    print(f"  Saved: {output_path / 'lfp_timeseries_validation.png'}")
    plt.close(fig2)
    
    # 3. Spectral analysis
    print("\n" + "="*80)
    print("Spectral Analysis of Simulated LFP")
    print("="*80)
    
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Power spectral density for each channel
    for ch_idx in range(min(4, results['lfp_signals'].shape[1])):
        lfp_signal = results['lfp_signals'][:, ch_idx]
        
        # Compute PSD
        freqs, psd = signal.welch(lfp_signal, fs=1000.0/model.dt, nperseg=1024)
        
        ax = axes[ch_idx // 2, ch_idx % 2]
        ax.semilogy(freqs, psd, 'k-', linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('PSD (mV²/Hz)', fontsize=11)
        ax.set_title(f'Channel {ch_idx}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])
        
        # Mark key frequency bands
        ax.axvspan(4, 8, alpha=0.2, color='blue', label='Theta')
        ax.axvspan(13, 30, alpha=0.2, color='green', label='Beta')
        ax.axvspan(30, 80, alpha=0.2, color='red', label='Gamma')
        ax.legend(fontsize=9)
        
        # Find and annotate peak frequency
        peak_freq, _ = model.estimate_resonant_frequency(lfp_signal, 1000.0/model.dt)
        ax.axvline(peak_freq, color='red', linestyle='--', linewidth=2, 
                  label=f'Peak: {peak_freq:.1f} Hz')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'lfp_spectral_analysis.png', dpi=150)
    print(f"  Saved: {output_path / 'lfp_spectral_analysis.png'}")
    plt.close(fig3)
    
    # 4. Delay optimization analysis
    print("\n" + "="*80)
    print("Delay Optimization for Different Frequency Bands")
    print("="*80)
    
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    target_freqs = [8.0, 20.0, 40.0, 70.0]  # Theta, Beta, Low Gamma, High Gamma
    freq_names = ['Theta (8 Hz)', 'Beta (20 Hz)', 'Low Gamma (40 Hz)', 'High Gamma (70 Hz)']
    
    for idx, (target_freq, freq_name) in enumerate(zip(target_freqs, freq_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Test different delays
        test_delays = np.linspace(1, 20, 15)
        
        print(f"\n  Optimizing for {freq_name}...")
        optimal_delay, powers = model.optimize_delay_for_frequency(
            target_freq_hz=target_freq,
            test_delays_ms=test_delays,
            n_steps=500
        )
        
        # Plot
        ax.plot(test_delays, powers, 'o-', linewidth=2, markersize=8, color='navy')
        ax.axvline(optimal_delay, color='red', linestyle='--', linewidth=2,
                  label=f'Optimal: {optimal_delay:.2f} ms')
        ax.set_xlabel('Conduction Delay (ms)', fontsize=11)
        ax.set_ylabel('Power at Target Frequency', fontsize=11)
        ax.set_title(freq_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Theoretical optimal delay (quarter-cycle)
        theoretical_delay = 1000.0 / (4 * target_freq)
        ax.axvline(theoretical_delay, color='green', linestyle=':', linewidth=2,
                  label=f'Theoretical: {theoretical_delay:.2f} ms')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'delay_optimization.png', dpi=150)
    print(f"  Saved: {output_path / 'delay_optimization.png'}")
    plt.close(fig4)
    
    # 5. Spatial propagation analysis
    print("\n" + "="*80)
    print("Spatial Propagation Analysis")
    print("="*80)
    
    fig5, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Select middle snapshot for analysis
    mid_idx = len(results['V_snapshots']) // 2
    snapshot = results['V_snapshots'][mid_idx]
    
    # a) Spatial heatmap
    im1 = axes[0].imshow(snapshot.T, origin='lower', cmap='RdBu_r',
                        extent=full_extent,
                        aspect='auto')
    axes[0].set_xlabel('x (μm)', fontsize=11)
    axes[0].set_ylabel('y (μm) [Absolute]', fontsize=11)
    axes[0].set_title('LFP Spatial Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='V (mV)')
    
    # b) Spatial gradient (propagation direction)
    grad_y, grad_x = np.gradient(snapshot, model.dx)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    im2 = axes[1].imshow(grad_magnitude.T, origin='lower', cmap='viridis',
                        extent=full_extent,
                        aspect='auto')
    axes[1].set_xlabel('x (μm)', fontsize=11)
    axes[1].set_ylabel('y (μm) [Absolute]', fontsize=11)
    axes[1].set_title('Propagation Strength', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='|∇V| (mV/μm)')
    
    # c) Spatial profile along probe (y-axis)
    y_profile = np.mean(snapshot, axis=0)
    # Create y-axis for plotting (absolute coordinates)
    y_axis_abs = np.arange(len(y_profile)) * model.dx + y_origin_offset
    axes[2].plot(y_axis_abs, y_profile, 'k-', linewidth=2)
    axes[2].set_xlabel('y (μm) [Absolute]', fontsize=11)
    axes[2].set_ylabel('Mean LFP (mV)', fontsize=11)
    axes[2].set_title('Spatial Profile Along Probe', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'spatial_propagation.png', dpi=150)
    print(f"  Saved: {output_path / 'spatial_propagation.png'}")
    plt.close(fig5)
    
    # Create Validation Figure (Tests 2 & 3)
    fig_val, axes_val = plt.subplots(1, 2, figsize=(12, 5))
    
    # Anisotropy Plot
    if not np.isnan(anisotropy_ratio):
        # We don't have the profile variables here easily without recomputing or returning them
        # Let's just visualize the middle snapshot with aspect ratio to hint at it
        pass
        
    # Spatial Decay Plot
    if decay_data[0] is not None:
        r_vals, v_vals, fit_vals = decay_data
        axes_val[0].scatter(r_vals, v_vals, s=1, color='gray', alpha=0.3, label='Data points')
        sorted_idx = np.argsort(r_vals)
        axes_val[0].plot(r_vals[sorted_idx], fit_vals[sorted_idx], 'r-', linewidth=2, label=f'Fit (λ={lambda_um:.1f}μm)')
        axes_val[0].set_xlabel('Distance from Peak (μm)')
        axes_val[0].set_ylabel('LFP Amplitude (mV)')
        axes_val[0].set_title('Test 3: Spatial Decay')
        axes_val[0].legend()
        axes_val[0].grid(True, alpha=0.3)
        
    # Anisotropy Text
    axes_val[1].text(0.5, 0.5, f"Test 2: Anisotropy\n\nRatio = {anisotropy_ratio:.2f}\nTarget ~ 3.0", 
                    ha='center', va='center', fontsize=16)
    axes_val[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'validation_metrics.png')
    print(f"  Saved: {output_path / 'validation_metrics.png'}")
    plt.close(fig_val)
    
    # Generate summary report
    print("\n" + "="*80)
    print("Analysis Summary")
    print("="*80)
    
    summary = {
        'session': str(session_path),
        'time_window': time_window,
        'coordinate_offset_y': float(y_origin_offset),
        'n_spikes_analyzed': int(spike_subset['n_spikes']),
        'duration_sec': float(duration),
        'n_recording_sites': int(results['lfp_signals'].shape[1]),
        'validation_metrics': {
            'anisotropy_ratio': float(anisotropy_ratio) if not np.isnan(anisotropy_ratio) else None,
            'spatial_decay_lambda_um': float(lambda_um) if not np.isnan(lambda_um) else None,
            'coherence_theta': float(coherence_stats.get('theta', np.nan)) if coherence_stats else None,
            'coherence_beta': float(coherence_stats.get('beta', np.nan)) if coherence_stats else None
        },
        'model_parameters': {
            'spatial_resolution_um': model.dx,
            'time_step_ms': model.dt * 1000,
            'diffusion_coefficient': model.D,
            'conduction_delay_ms': model.tau * 1000,
            'anisotropy_ratio': model.anisotropy
        }
    }
    
    # Compute dominant frequencies
    summary['dominant_frequencies'] = {}
    for ch_idx in range(min(5, results['lfp_signals'].shape[1])):
        lfp_signal = results['lfp_signals'][:, ch_idx]
        peak_freq, _ = model.estimate_resonant_frequency(lfp_signal, 1000.0/model.dt)
        summary['dominant_frequencies'][f'channel_{ch_idx}'] = float(peak_freq)
    
    # Save summary
    summary_path = output_path / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis summary to: {summary_path}")
    
    print("\n" + "="*80)
    print("Key Findings:")
    print("="*80)
    print(f"1. Analyzed {spike_subset['n_spikes']:,} spikes over {duration:.2f} seconds")
    print(f"2. Simulated LFP propagation at {results['lfp_signals'].shape[1]} recording sites")
    print(f"3. Model parameters:")
    print(f"   - Conduction delay: {model.tau*1000:.2f} ms")
    print(f"   - Diffusion coefficient: {model.D:.3f}")
    print(f"   - Anisotropy ratio: {model.anisotropy:.2f}")
    print(f"4. Dominant frequencies:")
    for ch_name, freq in summary['dominant_frequencies'].items():
        print(f"   - {ch_name}: {freq:.2f} Hz")
    print(f"\nAll outputs saved to: {output_path}")
    print("="*80)
    
    return summary, results


def main():
    """
    Main function to run the complete delayed diffusion analysis.
    """
    print("\n" + "="*80)
    print("Delayed Diffusion Model: Complete Analysis")
    print("Aligning Model Predictions with Real Neuropixels Data")
    print("="*80 + "\n")
    
    # Configuration
    # If actual data is available, use this path structure
    data_root = Path('E:/Neuropixels/Python/DemoData')
    session_path = Path('1818_09182025_g0/1818_09182025_g0_imec0')
    
    # Analyze a shorter window for testing stability
    time_window = (100.0, 105.0)  # seconds
    # Note: 200s is too long for 0.05ms timestep (4 million steps)
    
    output_dir = Path('model_outputs/delayed_diffusion_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run complete analysis
    try:
        summary, results = analyze_delayed_diffusion_with_real_data(
            data_root=data_root,
            session_path=session_path,
            time_window=time_window,
            output_dir=str(output_dir)
        )
        
        print("\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        print("\nNote: This script expects Neuropixels data in the standard Kilosort format.")
        print("If data is not available, it will generate synthetic data for demonstration.")
        raise
    
    return summary, results


if __name__ == '__main__':
    summary, results = main()
