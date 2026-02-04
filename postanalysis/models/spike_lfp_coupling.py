"""
Spike-LFP Coupling Model.

Implements Chapter 2.2: Coupled Spike-LFP Models with phase filtering.

Key Features:
    - Forward model: How spikes generate LFP (current source density)
    - Feedback model: How LFP phase affects spike generation
    - Phase-dependent spike probability
"""

import numpy as np
from typing import Dict


class SpikeLFPCouplingModel:
    """
    Models bidirectional coupling between spikes and LFP.
    
    Forward: spikes -> LFP via spatial kernel
    Feedback: LFP phase -> spike probability
    """
    
    def __init__(self,
                 spatial_kernel_um: float = 100.0,
                 temporal_kernel_ms: float = 10.0,
                 sampling_rate_hz: float = 1000.0):
        """
        Initialize spike-LFP coupling model.
        
        Args:
            spatial_kernel_um: Spatial extent of spike contribution to LFP
            temporal_kernel_ms: Temporal extent of spike contribution
            sampling_rate_hz: Sampling rate
        """
        self.spatial_sigma = spatial_kernel_um
        self.temporal_sigma = temporal_kernel_ms / 1000.0
        self.fs = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        
        print(f"Initialized SpikeLFPCouplingModel")
        print(f"  Spatial kernel: {spatial_kernel_um}μm")
        print(f"  Temporal kernel: {temporal_kernel_ms}ms")
    
    def generate_lfp_from_spikes(self,
                                 spike_times: np.ndarray,
                                 spike_positions: np.ndarray,
                                 recording_position: np.ndarray,
                                 duration_sec: float) -> Dict:
        """
        Generate LFP from spike trains using forward model.
        
        LFP = Σ_spikes K(r, t) where K is spatiotemporal kernel.
        
        Args:
            spike_times: Array of spike times (seconds)
            spike_positions: Spike positions (n_spikes, 2) in μm
            recording_position: LFP recording position (2,) in μm
            duration_sec: Total duration
            
        Returns:
            Dictionary with 'time', 'lfp_signal', 'contributions'
        """
        n_samples = int(duration_sec * self.fs)
        time = np.arange(n_samples) * self.dt
        lfp_signal = np.zeros(n_samples)
        
        # Temporal kernel (alpha function)
        t_kernel = np.arange(0, 5*self.temporal_sigma, self.dt)
        temporal_kernel = (t_kernel / self.temporal_sigma) * np.exp(-t_kernel / self.temporal_sigma)
        temporal_kernel /= np.max(temporal_kernel)
        
        for spike_t, spike_pos in zip(spike_times, spike_positions):
            # Spatial distance
            dist = np.linalg.norm(spike_pos - recording_position)
            spatial_weight = np.exp(-dist**2 / (2 * self.spatial_sigma**2))
            
            # Add contribution to LFP
            spike_idx = int(spike_t * self.fs)
            if spike_idx < n_samples:
                contrib_length = min(len(temporal_kernel), n_samples - spike_idx)
                lfp_signal[spike_idx:spike_idx+contrib_length] += (
                    spatial_weight * temporal_kernel[:contrib_length]
                )
        
        return {
            'time': time,
            'lfp_signal': lfp_signal,
            'recording_position': recording_position
        }
    
    def compute_phase_dependent_firing_probability(self,
                                                   lfp_phase: float,
                                                   preferred_phase: float,
                                                   phase_locking_strength: float) -> float:
        """
        Compute firing probability modulated by LFP phase.
        
        P(spike|phase) ∝ exp(κ * cos(phase - preferred_phase))
        
        Args:
            lfp_phase: Current LFP phase (radians)
            preferred_phase: Neuron's preferred phase (radians)
            phase_locking_strength: Strength of phase-locking (κ)
            
        Returns:
            Firing probability (0 to 1, relative)
        """
        phase_diff = lfp_phase - preferred_phase
        probability = np.exp(phase_locking_strength * np.cos(phase_diff))
        return probability
    
    def compute_spike_triggered_average(self,
                                      spike_times: np.ndarray,
                                      lfp_signal: np.ndarray,
                                      sampling_rate: float,
                                      window_sec: float = 0.1) -> Dict:
        """
        Compute spike-triggered average (STA) LFP.
        
        Classic measure of spike-LFP coupling. Averages LFP trace
        in a window around each spike time.
        
        Args:
            spike_times: Array of spike times (seconds)
            lfp_signal: LFP signal (samples)
            sampling_rate: Sampling rate in Hz
            window_sec: Window size around spike (±window_sec)
            
        Returns:
            Dictionary with 'time_lags', 'sta_lfp', 'n_spikes'
        """
        window_samples = int(window_sec * sampling_rate)
        time_lags = np.arange(-window_samples, window_samples + 1) / sampling_rate
        
        sta_windows = []
        
        for spike_t in spike_times:
            spike_idx = int(spike_t * sampling_rate)
            
            # Check if window is within signal bounds
            start_idx = spike_idx - window_samples
            end_idx = spike_idx + window_samples + 1
            
            if start_idx >= 0 and end_idx < len(lfp_signal):
                window = lfp_signal[start_idx:end_idx]
                sta_windows.append(window)
        
        if len(sta_windows) == 0:
            return {
                'time_lags': time_lags,
                'sta_lfp': np.zeros(len(time_lags)),
                'n_spikes': 0,
                'std_lfp': np.zeros(len(time_lags))
            }
        
        sta_windows = np.array(sta_windows)
        sta_lfp = np.mean(sta_windows, axis=0)
        std_lfp = np.std(sta_windows, axis=0)
        
        return {
            'time_lags': time_lags,
            'sta_lfp': sta_lfp,
            'n_spikes': len(sta_windows),
            'std_lfp': std_lfp
        }
    
    def compute_phase_locking(self,
                            spike_times: np.ndarray,
                            lfp_signal: np.ndarray,
                            sampling_rate: float,
                            target_frequency_hz: float = 8.0) -> Dict:
        """
        Compute phase locking between spikes and LFP.
        
        Extracts LFP phase at each spike time and computes phase distribution
        to validate preferred phase locking.
        
        Args:
            spike_times: Array of spike times (seconds)
            lfp_signal: LFP signal (samples)
            sampling_rate: Sampling rate in Hz
            target_frequency_hz: Frequency to filter LFP
            
        Returns:
            Dictionary with 'spike_phases', 'preferred_phase', 'phase_locking_value'
        """
        from scipy.signal import hilbert, butter, filtfilt
        
        # Bandpass filter around target frequency
        lowcut = target_frequency_hz * 0.8
        highcut = target_frequency_hz * 1.2
        nyq = 0.5 * sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        
        if high >= 1.0:
            high = 0.99
        
        try:
            b, a = butter(3, [low, high], btype='band')
            filtered_lfp = filtfilt(b, a, lfp_signal)
        except:
            print(f"Warning: Could not filter LFP at {target_frequency_hz}Hz")
            filtered_lfp = lfp_signal
        
        # Compute instantaneous phase
        analytic_signal = hilbert(filtered_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        
        # Extract phase at each spike time
        spike_phases = []
        for spike_t in spike_times:
            spike_idx = int(spike_t * sampling_rate)
            if 0 <= spike_idx < len(instantaneous_phase):
                spike_phases.append(instantaneous_phase[spike_idx])
        
        spike_phases = np.array(spike_phases)
        
        if len(spike_phases) == 0:
            return {
                'spike_phases': spike_phases,
                'preferred_phase': 0.0,
                'phase_locking_value': 0.0,
                'rayleigh_p': 1.0
            }
        
        # Compute preferred phase (circular mean)
        preferred_phase = np.angle(np.mean(np.exp(1j * spike_phases)))
        
        # Compute phase locking value (PLV)
        plv = np.abs(np.mean(np.exp(1j * spike_phases)))
        
        # Rayleigh test for non-uniformity
        from scipy.stats import circstd
        n = len(spike_phases)
        R = n * plv
        rayleigh_p = np.exp(-R**2 / n)
        
        return {
            'spike_phases': spike_phases,
            'preferred_phase': preferred_phase,
            'phase_locking_value': plv,
            'rayleigh_p': rayleigh_p,
            'target_frequency': target_frequency_hz
        }
    
    def plot_spike_triggered_average(self, sta_results: Dict, save_path: str = None):
        """
        Plot spike-triggered average LFP.
        
        Args:
            sta_results: Output from compute_spike_triggered_average
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_lags = sta_results['time_lags'] * 1000  # Convert to ms
        sta_lfp = sta_results['sta_lfp']
        std_lfp = sta_results['std_lfp']
        n_spikes = sta_results['n_spikes']
        
        # Plot STA with confidence interval
        ax.plot(time_lags, sta_lfp, 'b-', linewidth=2, label='STA-LFP')
        ax.fill_between(time_lags, sta_lfp - std_lfp, sta_lfp + std_lfp,
                       alpha=0.3, color='blue', label='±1 SD')
        
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Spike Time')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Time from Spike (ms)', fontsize=12)
        ax.set_ylabel('LFP Amplitude (mV)', fontsize=12)
        ax.set_title(f'Spike-Triggered Average LFP (n={n_spikes} spikes)', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved STA plot to {save_path}")
        
        return fig
    
    def plot_phase_polar_histogram(self, phase_locking_results: Dict, save_path: str = None):
        """
        Plot polar histogram of spike phases.
        
        Shows distribution of LFP phases at spike times to visualize
        preferred phase locking.
        
        Args:
            phase_locking_results: Output from compute_phase_locking
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        
        spike_phases = phase_locking_results['spike_phases']
        preferred_phase = phase_locking_results['preferred_phase']
        plv = phase_locking_results['phase_locking_value']
        rayleigh_p = phase_locking_results['rayleigh_p']
        target_freq = phase_locking_results['target_frequency']
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Create histogram bins
        n_bins = 36  # 10 degree bins
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        # Compute histogram
        hist, bin_edges = np.histogram(spike_phases, bins=bins)
        
        # Plot bars
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 2 * np.pi / n_bins
        bars = ax.bar(bin_centers, hist, width=width, alpha=0.7, 
                     edgecolor='black', linewidth=1)
        
        # Color bars by angle
        colors = plt.cm.hsv(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        # Plot preferred phase as arrow
        max_hist = np.max(hist)
        ax.arrow(preferred_phase, 0, 0, max_hist * 1.2, 
                width=0.1, head_width=0.3, head_length=max_hist * 0.15,
                fc='red', ec='darkred', linewidth=2, length_includes_head=True,
                zorder=10)
        
        # Add annotations
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'Phase Locking at {target_freq}Hz\n' + 
                    f'PLV = {plv:.3f}, p = {rayleigh_p:.4f}\n' +
                    f'Preferred Phase = {np.degrees(preferred_phase):.1f}°',
                    fontweight='bold', fontsize=14, pad=20)
        
        # Add phase labels
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
        
        # Add significance annotation
        if rayleigh_p < 0.001:
            sig_text = '*** p < 0.001\nHighly Significant'
        elif rayleigh_p < 0.01:
            sig_text = '** p < 0.01\nSignificant'
        elif rayleigh_p < 0.05:
            sig_text = '* p < 0.05\nSignificant'
        else:
            sig_text = 'n.s.\nNot Significant'
        
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
               ha='left', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved phase polar histogram to {save_path}")
        
        return fig

