"""
Phase-Coherence Gating Function Model.

Implements Chapter 2.2-2.3: Coupled Spike-LFP Models with Phase Filtering.

Based on Pouzzner's insight that the Basal Ganglia functions as a "router" that 
selects specific oscillatory patterns and returns them phase-coherently to the thalamus.

Key Concepts:
    1. FSI-mediated feedforward inhibition forces MSN spike timing to entrain to LFP phases
    2. MSNs and FSIs act as "Phase Filters" rather than simple inhibitory units
    3. Phase selection process gates information based on temporal structure
    4. Reward-related patterns are selectively phase-locked and propagated

Mathematical Framework:
    - MSN firing probability: P(spike) = f(LFP_phase, DA_level)
    - FSI entrainment: FSI spikes locked to LFP phase
    - Phase-dependent gating: Information passed only at specific phases
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import laplace
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt


class PhaseCoherenceGatingModel:
    """
    Implements phase-coherence gating where neurons act as phase filters.
    
    FSIs entrain to LFP oscillations and gate MSN firing to specific phases,
    implementing Pouzzner's "router" concept for selective information flow.
    """
    
    def __init__(self,
                 n_msns: int = 100,
                 n_fsis: int = 20,
                 target_frequency_hz: float = 8.0,
                 sampling_rate_hz: float = 1000.0,
                 phase_preference_std: float = 0.5):
        """
        Initialize phase-coherence gating model.
        
        Args:
            n_msns: Number of medium spiny neurons
            n_fsis: Number of fast-spiking interneurons
            target_frequency_hz: Target frequency for phase-locking (e.g., theta ~8Hz)
            sampling_rate_hz: Sampling rate for simulation
            phase_preference_std: Standard deviation of phase preference (radians)
        """
        self.n_msns = n_msns
        self.n_fsis = n_fsis
        self.target_freq = target_frequency_hz
        self.fs = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        
        # Phase preferences for each neuron (radians, 0 to 2π)
        # MSNs: Distributed across phases
        self.msn_phase_preferences = np.random.uniform(0, 2*np.pi, n_msns)
        # FSIs: More tightly clustered (they entrain strongly)
        self.fsi_phase_preferences = np.random.normal(np.pi/2, 0.3, n_fsis) % (2*np.pi)
        
        # Phase selectivity (how strongly neurons prefer their phase)
        self.msn_phase_selectivity = np.random.uniform(0.5, 1.5, n_msns)
        self.fsi_phase_selectivity = np.random.uniform(1.0, 2.0, n_fsis)  # FSIs more selective
        
        # Baseline firing rates (Hz)
        self.msn_baseline_rate = np.random.uniform(1.0, 5.0, n_msns)
        self.fsi_baseline_rate = np.random.uniform(10.0, 30.0, n_fsis)  # FSIs fire faster
        
        # Synaptic connectivity: FSI -> MSN inhibition
        # Each MSN receives input from subset of FSIs
        self.fsi_to_msn_weights = self._initialize_fsi_msn_connectivity()
        
        # Reward modulation (will be set during simulation)
        self.reward_signal = 0.0
        
        print(f"Initialized PhaseCoherenceGatingModel:")
        print(f"  {n_msns} MSNs, {n_fsis} FSIs")
        print(f"  Target frequency: {target_frequency_hz} Hz")
        print(f"  Phase preference std: {phase_preference_std:.2f} rad")
    
    def _initialize_fsi_msn_connectivity(self, connection_prob: float = 0.3) -> np.ndarray:
        """
        Initialize FSI -> MSN inhibitory connectivity.
        
        Args:
            connection_prob: Probability of connection between FSI and MSN
            
        Returns:
            Connectivity matrix (n_msns x n_fsis) with synaptic weights
        """
        # Random sparse connectivity
        connections = np.random.rand(self.n_msns, self.n_fsis) < connection_prob
        
        # Weight strengths (inhibitory, so negative)
        weights = -np.random.uniform(0.5, 2.0, (self.n_msns, self.n_fsis))
        weights *= connections  # Zero out non-connected pairs
        
        return weights
    
    def extract_lfp_phase(self, lfp_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract instantaneous phase and amplitude from LFP signal.
        
        Uses Hilbert transform to get analytic signal.
        
        Args:
            lfp_signal: LFP time series
            
        Returns:
            phase: Instantaneous phase (radians, -π to π)
            amplitude: Instantaneous amplitude envelope
        """
        # Bandpass filter around target frequency
        nyquist = self.fs / 2
        low = max(0.5, self.target_freq - 2) / nyquist
        high = min(nyquist - 0.5, self.target_freq + 2) / nyquist
        b, a = butter(3, [low, high], btype='band')
        lfp_filtered = filtfilt(b, a, lfp_signal)
        
        # Hilbert transform
        analytic_signal = hilbert(lfp_filtered)
        phase = np.angle(analytic_signal)  # -π to π
        amplitude = np.abs(analytic_signal)
        
        return phase, amplitude
    
    def compute_phase_modulated_firing_rate(self,
                                            current_phase: float,
                                            neuron_type: str,
                                            neuron_idx: int,
                                            reward_modulation: float = 1.0) -> float:
        """
        Compute firing rate modulated by LFP phase.
        
        Implements phase-dependent gating: neurons fire preferentially at their
        preferred phase.
        
        Args:
            current_phase: Current LFP phase (radians)
            neuron_type: 'msn' or 'fsi'
            neuron_idx: Index of specific neuron
            reward_modulation: Dopamine/reward modulation factor (0 to ~2)
            
        Returns:
            instantaneous_rate: Phase-modulated firing rate (Hz)
        """
        if neuron_type == 'msn':
            baseline = self.msn_baseline_rate[neuron_idx]
            preferred_phase = self.msn_phase_preferences[neuron_idx]
            selectivity = self.msn_phase_selectivity[neuron_idx]
        elif neuron_type == 'fsi':
            baseline = self.fsi_baseline_rate[neuron_idx]
            preferred_phase = self.fsi_phase_preferences[neuron_idx]
            selectivity = self.fsi_phase_selectivity[neuron_idx]
        else:
            raise ValueError("neuron_type must be 'msn' or 'fsi'")
        
        # Phase difference from preferred
        phase_diff = np.abs(((current_phase - preferred_phase + np.pi) % (2*np.pi)) - np.pi)
        
        # Von Mises-like phase modulation (concentration around preferred phase)
        phase_modulation = np.exp(selectivity * (np.cos(phase_diff) - 1))
        
        # Firing rate = baseline * phase_modulation * reward_modulation
        rate = baseline * phase_modulation * reward_modulation
        
        return rate
    
    def simulate_phase_gated_network(self,
                                     lfp_signal: np.ndarray,
                                     reward_times: Optional[np.ndarray] = None,
                                     reward_window_sec: float = 0.5) -> Dict:
        """
        Simulate network with phase-coherence gating.
        
        FSIs entrain to LFP phase and gate MSN activity through inhibition.
        Reward signals enhance phase-locking at reward-associated phases.
        
        Args:
            lfp_signal: LFP time series to drive the network
            reward_times: Times of reward delivery (seconds)
            reward_window_sec: Duration of reward modulation effect
            
        Returns:
            results: Dict with spike times, phases, and phase-locking metrics
        """
        duration = len(lfp_signal) * self.dt
        n_steps = len(lfp_signal)
        
        # Extract LFP phase
        lfp_phase, lfp_amplitude = self.extract_lfp_phase(lfp_signal)
        
        # Initialize spike arrays
        msn_spikes = [[] for _ in range(self.n_msns)]
        fsi_spikes = [[] for _ in range(self.n_fsis)]
        
        # Reward signal over time
        if reward_times is None:
            reward_times = np.array([])
        reward_signal = np.zeros(n_steps)
        for rt in reward_times:
            start_idx = int(rt * self.fs)
            end_idx = int((rt + reward_window_sec) * self.fs)
            reward_signal[start_idx:end_idx] = 1.5  # 50% increase
        
        print(f"Simulating phase-gated network for {duration:.2f}s...")
        
        # Simulate each time step
        for step in range(n_steps):
            current_time = step * self.dt
            current_phase = lfp_phase[step]
            current_reward = reward_signal[step]
            
            # 1. FSI activity (entrained to LFP phase)
            fsi_active = np.zeros(self.n_fsis, dtype=bool)
            for fsi_idx in range(self.n_fsis):
                fsi_rate = self.compute_phase_modulated_firing_rate(
                    current_phase, 'fsi', fsi_idx, reward_modulation=1.0
                )
                
                # Poisson spiking
                if np.random.rand() < fsi_rate * self.dt:
                    fsi_spikes[fsi_idx].append(current_time)
                    fsi_active[fsi_idx] = True
            
            # 2. MSN activity (gated by FSI inhibition and phase)
            for msn_idx in range(self.n_msns):
                # Base phase-modulated rate
                msn_rate = self.compute_phase_modulated_firing_rate(
                    current_phase, 'msn', msn_idx, reward_modulation=current_reward
                )
                
                # FSI inhibition (active FSIs suppress connected MSNs)
                fsi_inhibition = np.sum(self.fsi_to_msn_weights[msn_idx] * fsi_active)
                # Inhibition reduces firing probability
                inhibited_rate = msn_rate * np.exp(fsi_inhibition * 0.1)  # Scaled inhibition
                
                # Poisson spiking
                if np.random.rand() < inhibited_rate * self.dt:
                    msn_spikes[msn_idx].append(current_time)
        
        # Convert to arrays
        msn_spike_times = [np.array(spikes) for spikes in msn_spikes]
        fsi_spike_times = [np.array(spikes) for spikes in fsi_spikes]
        
        # Calculate phase-locking statistics
        msn_phase_locking = self._compute_phase_locking_stats(
            msn_spike_times, lfp_phase, self.fs
        )
        fsi_phase_locking = self._compute_phase_locking_stats(
            fsi_spike_times, lfp_phase, self.fs
        )
        
        print(f"  Simulation complete.")
        print(f"  Mean MSN phase-locking strength: {msn_phase_locking['mean_plv']:.3f}")
        print(f"  Mean FSI phase-locking strength: {fsi_phase_locking['mean_plv']:.3f}")
        
        return {
            'msn_spike_times': msn_spike_times,
            'fsi_spike_times': fsi_spike_times,
            'lfp_phase': lfp_phase,
            'lfp_amplitude': lfp_amplitude,
            'time': np.arange(n_steps) * self.dt,
            'reward_signal': reward_signal,
            'msn_phase_locking': msn_phase_locking,
            'fsi_phase_locking': fsi_phase_locking
        }
    
    def _compute_phase_locking_stats(self,
                                    spike_times_list: List[np.ndarray],
                                    lfp_phase: np.ndarray,
                                    sampling_rate: float) -> Dict:
        """
        Compute phase-locking value (PLV) for each neuron.
        
        PLV measures consistency of spike timing relative to LFP phase.
        PLV = 1: perfect phase-locking, PLV = 0: no phase-locking.
        
        Args:
            spike_times_list: List of spike time arrays for each neuron
            lfp_phase: LFP phase time series
            sampling_rate: Sampling rate in Hz
            
        Returns:
            stats: Dict with PLVs and preferred phases for each neuron
        """
        n_neurons = len(spike_times_list)
        plvs = np.zeros(n_neurons)
        preferred_phases = np.zeros(n_neurons)
        
        for neuron_idx, spike_times in enumerate(spike_times_list):
            if len(spike_times) < 5:  # Need minimum spikes
                plvs[neuron_idx] = 0
                preferred_phases[neuron_idx] = 0
                continue
            
            # Get phases at spike times
            spike_indices = (spike_times * sampling_rate).astype(int)
            spike_indices = spike_indices[spike_indices < len(lfp_phase)]
            
            if len(spike_indices) == 0:
                continue
            
            spike_phases = lfp_phase[spike_indices]
            
            # Compute PLV (mean resultant length of phase vectors)
            phase_vectors = np.exp(1j * spike_phases)
            mean_vector = np.mean(phase_vectors)
            plvs[neuron_idx] = np.abs(mean_vector)
            preferred_phases[neuron_idx] = np.angle(mean_vector)
        
        return {
            'plvs': plvs,
            'preferred_phases': preferred_phases,
            'mean_plv': np.mean(plvs),
            'std_plv': np.std(plvs)
        }
    
    def plot_phase_locking_analysis(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize phase-locking results.
        
        Args:
            results: Output from simulate_phase_gated_network
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. Raster plot with LFP phase
        ax = axes[0, 0]
        for neuron_idx, spike_times in enumerate(results['msn_spike_times'][:20]):  # First 20 MSNs
            ax.scatter(spike_times, np.ones_like(spike_times) * neuron_idx, 
                      c='blue', s=1, alpha=0.6)
        ax.set_ylabel('MSN #')
        ax.set_xlabel('Time (s)')
        ax.set_title('MSN Raster Plot')
        
        # 2. Phase-locking values
        ax = axes[0, 1]
        msn_plvs = results['msn_phase_locking']['plvs']
        fsi_plvs = results['fsi_phase_locking']['plvs']
        ax.hist(msn_plvs, bins=20, alpha=0.5, label='MSN', color='blue')
        ax.hist(fsi_plvs, bins=20, alpha=0.5, label='FSI', color='red')
        ax.set_xlabel('Phase-Locking Value (PLV)')
        ax.set_ylabel('Count')
        ax.set_title('Phase-Locking Strength')
        ax.legend()
        
        # 3. Preferred phases (circular plot)
        ax = axes[0, 2]
        msn_phases = results['msn_phase_locking']['preferred_phases']
        fsi_phases = results['fsi_phase_locking']['preferred_phases']
        ax.scatter(msn_phases, msn_plvs, c='blue', alpha=0.5, label='MSN', s=20)
        ax.scatter(fsi_phases, fsi_plvs, c='red', alpha=0.5, label='FSI', s=20)
        ax.set_xlabel('Preferred Phase (rad)')
        ax.set_ylabel('PLV')
        ax.set_title('Preferred Phase vs PLV')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(np.pi, color='gray', linestyle='--', alpha=0.3)
        ax.legend()
        
        # 4. LFP and reward signal
        ax = axes[1, 0]
        time = results['time']
        ax.plot(time, results['lfp_phase'], 'k-', alpha=0.5, label='LFP Phase')
        ax2 = ax.twinx()
        ax2.plot(time, results['reward_signal'], 'g-', alpha=0.7, label='Reward')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (rad)', color='k')
        ax2.set_ylabel('Reward Signal', color='g')
        ax.set_title('LFP Phase and Reward')
        
        # 5. Phase histogram (circular histogram)
        ax = axes[1, 1]
        # Collect all MSN spike phases
        all_spike_phases = []
        for spike_times in results['msn_spike_times']:
            if len(spike_times) > 0:
                spike_indices = (spike_times * self.fs).astype(int)
                spike_indices = spike_indices[spike_indices < len(results['lfp_phase'])]
                all_spike_phases.extend(results['lfp_phase'][spike_indices])
        
        if len(all_spike_phases) > 0:
            ax.hist(all_spike_phases, bins=36, range=(-np.pi, np.pi), 
                   color='blue', alpha=0.7)
            ax.set_xlabel('LFP Phase (rad)')
            ax.set_ylabel('Spike Count')
            ax.set_title('MSN Spike Phase Distribution')
            ax.axvline(0, color='red', linestyle='--', label='Peak')
            ax.axvline(np.pi, color='red', linestyle='--')
        
        # 6. FSI gating effectiveness
        ax = axes[1, 2]
        # Calculate correlation between FSI activity and MSN suppression
        # This is a simplified metric
        ax.text(0.5, 0.5, f"Mean MSN PLV: {results['msn_phase_locking']['mean_plv']:.3f}\n"
                          f"Mean FSI PLV: {results['fsi_phase_locking']['mean_plv']:.3f}\n"
                          f"FSI Gating: {'Strong' if results['fsi_phase_locking']['mean_plv'] > 0.3 else 'Weak'}",
               ha='center', va='center', fontsize=12,
               transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('Phase-Gating Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved phase-locking visualization to {save_path}")
        
        return fig


class PhaseFieldModel:
    """
    Continuum neural field model with phase-coherence dynamics.
    
    Implements Chapter 2.5: Phase-Field Models.
    Models striatum as continuous field u(x,t) with local interactions
    and traveling waves of activity.
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (100, 100),
                 spatial_resolution_um: float = 10.0,
                 time_step_ms: float = 0.1):
        """
        Initialize phase-field model.
        
        Args:
            grid_size: Spatial grid size (nx, ny)
            spatial_resolution_um: Spatial resolution in micrometers
            time_step_ms: Time step in milliseconds
        """
        self.nx, self.ny = grid_size
        self.dx = spatial_resolution_um
        self.dt = time_step_ms / 1000.0
        
        # Activity field u(x,t)
        self.u = np.zeros((self.nx, self.ny))
        
        # Phase field φ(x,t) - represents oscillation phase
        self.phi = np.random.uniform(0, 2*np.pi, (self.nx, self.ny))
        
        print(f"Initialized PhaseFieldModel: {self.nx}x{self.ny} grid")
    
    def step(self, coupling_strength: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance phase-field model by one time step.
        
        Implements: ∂u/∂t = f(u) + coupling * ∇²u
                    ∂φ/∂t = ω + phase_coupling
        
        Args:
            coupling_strength: Strength of spatial coupling
            
        Returns:
            Updated (u, phi) fields
        """
        # Activity dynamics (simple excitable medium)
        du_dt = self.u * (1 - self.u) * (self.u - 0.3) + coupling_strength * laplace(self.u)
        
        # Phase dynamics with spatial coupling
        dphi_dt = 2 * np.pi * 8.0 + coupling_strength * laplace(np.sin(self.phi))
        
        self.u += self.dt * du_dt
        self.phi += self.dt * dphi_dt
        self.phi = self.phi % (2 * np.pi)
        
        return self.u.copy(), self.phi.copy()
