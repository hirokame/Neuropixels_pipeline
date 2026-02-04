"""
Striatal Microcircuit Model with Dopamine-Driven Resonance Tuning.

Implements Chapter 2.3: MSN-FSI Interactions with dopamine modulation.

Based on Pouzzner's insight that reinforcement learning optimizes "useful time 
alignments" and their associated "resonant frequencies".

Key Features:
    - MSN-FSI feedforward inhibition
    - Dopamine plastically modifies synaptic time constants
    - Dopamine tunes resonant frequencies for task-relevant synchronization
    - D1 vs D2 MSN differential responses
"""

import numpy as np
from typing import Dict, Optional, List
import matplotlib.pyplot as plt


class StrialMicrocircuitModel:
    """
    Striatal microcircuit with MSNs, FSIs, and dopamine modulation.
    
    Implements dopamine-driven tuning of resonant frequencies through
    plasticity of synaptic time constants and delays.
    """
    
    def __init__(self,
                 n_d1_msns: int = 50,
                 n_d2_msns: int = 50,
                 n_fsis: int = 20,
                 base_synaptic_delay_ms: float = 2.0,
                 sampling_rate_hz: float = 1000.0):
        """
        Initialize striatal microcircuit model.
        
        Args:
            n_d1_msns: Number of D1-type MSNs (direct pathway)
            n_d2_msns: Number of D2-type MSNs (indirect pathway)
            n_fsis: Number of fast-spiking interneurons
            base_synaptic_delay_ms: Baseline synaptic delay
            sampling_rate_hz: Simulation sampling rate
        """
        self.n_d1 = n_d1_msns
        self.n_d2 = n_d2_msns
        self.n_msns = n_d1_msns + n_d2_msns
        self.n_fsis = n_fsis
        self.fs = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        
        # Synaptic delays (ms) - plastically modified by dopamine
        self.fsi_msn_delays = np.full((self.n_msns, self.n_fsis), base_synaptic_delay_ms)
        
        # Synaptic time constants (ms) - also modified by dopamine
        self.synaptic_tau = np.full((self.n_msns, self.n_fsis), 5.0)  # GABA time constant
        
        # Dopamine sensitivity (D1 vs D2)
        self.da_sensitivity = np.concatenate([
            np.ones(n_d1_msns) * 1.5,    # D1: excited by DA
            np.ones(n_d2_msns) * 0.5     # D2: inhibited by DA
        ])
        
        # Synaptic weights
        self.fsi_msn_weights = self._initialize_connectivity()
        
        # Resonant frequencies (Hz) - tuned by dopamine
        self.resonant_frequencies = np.random.uniform(6, 12, self.n_msns)  # Theta-beta range
        
        print(f"Initialized StrialMicrocircuitModel:")
        print(f"  {n_d1_msns} D1-MSNs, {n_d2_msns} D2-MSNs, {n_fsis} FSIs")
        print(f"  Base synaptic delay: {base_synaptic_delay_ms}ms")
    
    def _initialize_connectivity(self) -> np.ndarray:
        """Initialize FSI -> MSN connectivity."""
        connection_prob = 0.4
        connections = np.random.rand(self.n_msns, self.n_fsis) < connection_prob
        weights = -np.random.uniform(0.5, 2.0, (self.n_msns, self.n_fsis))
        weights *= connections
        return weights
    
    def apply_dopamine_plasticity(self,
                                  dopamine_level: float,
                                  reward_outcome: bool,
                                  target_frequency_hz: float,
                                  learning_rate: float = 0.01):
        """
        Apply dopamine-driven plasticity to tune circuit to target frequency.
        
        This implements Pouzzner's concept that dopamine tunes synaptic delays
        and time constants to optimize resonance at task-relevant frequencies.
        
        Args:
            dopamine_level: Current DA level (0 to 2, baseline=1)
            reward_outcome: Whether trial was rewarded
            target_frequency_hz: Target frequency for current strategy (CW/CCW)
            learning_rate: Plasticity learning rate
        """
        if not reward_outcome:
            return  # No plasticity without reward
        
        # Dopamine prediction error drives plasticity
        da_error = dopamine_level - 1.0  # Positive = better than expected
        
        # Target delay for resonance at target frequency
        # Optimal delay ≈ 1/(4*f) for quarter-cycle offset
        target_delay_ms = 1000.0 / (4.0 * target_frequency_hz)
        
        # Update delays toward target (scaled by DA error and neuron sensitivity)
        for msn_idx in range(self.n_msns):
            da_mod = da_error * self.da_sensitivity[msn_idx]
            
            # Move delays toward target proportional to DA signal
            current_delay = np.mean(self.fsi_msn_delays[msn_idx])
            delay_error = target_delay_ms - current_delay
            
            self.fsi_msn_delays[msn_idx] += learning_rate * da_mod * delay_error
            
            # Clip delays to reasonable range
            self.fsi_msn_delays[msn_idx] = np.clip(
                self.fsi_msn_delays[msn_idx], 0.5, 20.0
            )
            
            # Update resonant frequency estimate
            self.resonant_frequencies[msn_idx] = 1000.0 / (4.0 * np.mean(self.fsi_msn_delays[msn_idx]))
        
        print(f"  Applied DA plasticity: DA={dopamine_level:.2f}, target_freq={target_frequency_hz:.1f}Hz")
        print(f"    Mean resonant freq: {np.mean(self.resonant_frequencies):.2f}Hz")
    
    def simulate_trial(self,
                      duration_sec: float,
                      input_rate_hz: float = 10.0,
                      dopamine_level: float = 1.0) -> Dict:
        """
        Simulate a single trial of microcircuit activity.
        
        Args:
            duration_sec: Trial duration
            input_rate_hz: Input drive to MSNs
            dopamine_level: Dopamine modulation level
            
        Returns:
            Dictionary with spike times and activity
        """
        n_steps = int(duration_sec * self.fs)
        
        # Generate input spikes (cortical drive)
        msn_spikes = [[] for _ in range(self.n_msns)]
        fsi_spikes = [[] for _ in range(self.n_fsis)]
        
        # FSIs fire at higher baseline rate
        fsi_base_rate = 20.0
        
        for step in range(n_steps):
            t = step * self.dt
            
            # FSI activity
            for fsi_idx in range(self.n_fsis):
                if np.random.rand() < fsi_base_rate * self.dt:
                    fsi_spikes[fsi_idx].append(t)
            
            # MSN activity (modulated by DA and FSI inhibition)
            for msn_idx in range(self.n_msns):
                # Base input drive modulated by DA
                da_modulated_rate = input_rate_hz * (1.0 + 0.5 * (dopamine_level - 1.0) * self.da_sensitivity[msn_idx])
                
                # Check for FSI inhibition (with delays)
                inhibited = False
                for fsi_idx in range(self.n_fsis):
                    if len(fsi_spikes[fsi_idx]) > 0:
                        last_fsi_spike = fsi_spikes[fsi_idx][-1]
                        delay = self.fsi_msn_delays[msn_idx, fsi_idx] / 1000.0
                        if (t - last_fsi_spike) < delay + 0.01:  # Within inhibitory window
                            inhibited = True
                            break
                
                if not inhibited:
                    if np.random.rand() < da_modulated_rate * self.dt:
                        msn_spikes[msn_idx].append(t)
        
        return {
            'msn_spikes': [np.array(s) for s in msn_spikes],
            'fsi_spikes': [np.array(s) for s in fsi_spikes],
            'duration': duration_sec,
            'resonant_frequencies': self.resonant_frequencies.copy()
        }
    
    def plot_resonance_tuning(self, 
                             trial_history: List[Dict],
                             save_path: Optional[str] = None):
        """
        Visualize how resonant frequencies are tuned over trials.
        
        Args:
            trial_history: List of trial results
            save_path: Optional path to save figure
        """
        n_trials = len(trial_history)
        
        # Extract resonant frequencies over time
        resonant_freqs_over_time = np.array([
            trial['resonant_frequencies'] for trial in trial_history
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Resonant frequency evolution
        ax = axes[0, 0]
        mean_freq = np.mean(resonant_freqs_over_time, axis=1)
        std_freq = np.std(resonant_freqs_over_time, axis=1)
        trials = np.arange(n_trials)
        ax.plot(trials, mean_freq, 'b-', linewidth=2)
        ax.fill_between(trials, mean_freq - std_freq, mean_freq + std_freq, 
                        alpha=0.3, color='blue')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Mean Resonant Frequency (Hz)')
        ax.set_title('Dopamine-Driven Frequency Tuning')
        ax.grid(True, alpha=0.3)
        
        # 2. D1 vs D2 resonance
        ax = axes[0, 1]
        d1_freq = np.mean(resonant_freqs_over_time[:, :self.n_d1], axis=1)
        d2_freq = np.mean(resonant_freqs_over_time[:, self.n_d1:], axis=1)
        ax.plot(trials, d1_freq, 'r-', label='D1-MSN', linewidth=2)
        ax.plot(trials, d2_freq, 'b-', label='D2-MSN', linewidth=2)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Mean Resonant Frequency (Hz)')
        ax.set_title('D1 vs D2 Frequency Tuning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Final frequency distribution
        ax = axes[1, 0]
        final_freqs = resonant_freqs_over_time[-1]
        ax.hist(final_freqs[:self.n_d1], bins=15, alpha=0.5, 
               label='D1-MSN', color='red')
        ax.hist(final_freqs[self.n_d1:], bins=15, alpha=0.5, 
               label='D2-MSN', color='blue')
        ax.set_xlabel('Resonant Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.set_title('Final Frequency Distribution')
        ax.legend()
        
        # 4. Summary text
        ax = axes[1, 1]
        summary_text = f"Dopamine-Driven Resonance Tuning\n\n"
        summary_text += f"Initial mean freq: {mean_freq[0]:.2f} Hz\n"
        summary_text += f"Final mean freq: {mean_freq[-1]:.2f} Hz\n"
        summary_text += f"Freq shift: {mean_freq[-1] - mean_freq[0]:.2f} Hz\n\n"
        summary_text += f"D1-MSN final: {d1_freq[-1]:.2f} Hz\n"
        summary_text += f"D2-MSN final: {d2_freq[-1]:.2f} Hz\n"
        ax.text(0.1, 0.5, summary_text, fontsize=12, 
               verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved resonance tuning plot to {save_path}")
        
        return fig
    
    def plot_population_raster(self, trial_results: Dict, save_path: Optional[str] = None):
        """
        Plot population raster with color-coded neuron types.
        
        Shows spike times for D1-MSNs (red), D2-MSNs (blue), and FSIs (green)
        to visualize assembly formation and synchronization patterns.
        
        Args:
            trial_results: Output from simulate_trial
            save_path: Optional path to save figure
        """
        msn_spikes = trial_results['msn_spikes']
        fsi_spikes = trial_results['fsi_spikes']
        duration = trial_results['duration']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        y_offset = 0
        
        # Plot D1-MSN spikes (red)
        for i in range(self.n_d1):
            if len(msn_spikes[i]) > 0:
                ax.scatter(msn_spikes[i], np.ones(len(msn_spikes[i])) * y_offset,
                          c='red', s=2, marker='|', alpha=0.8)
            y_offset += 1
        
        d1_end = y_offset
        
        # Plot D2-MSN spikes (blue)
        for i in range(self.n_d1, self.n_msns):
            if len(msn_spikes[i]) > 0:
                ax.scatter(msn_spikes[i], np.ones(len(msn_spikes[i])) * y_offset,
                          c='blue', s=2, marker='|', alpha=0.8)
            y_offset += 1
        
        d2_end = y_offset
        
        # Plot FSI spikes (green)
        for i in range(self.n_fsis):
            if len(fsi_spikes[i]) > 0:
                ax.scatter(fsi_spikes[i], np.ones(len(fsi_spikes[i])) * y_offset,
                          c='green', s=2, marker='|', alpha=0.8)
            y_offset += 1
        
        # Add separators and labels
        ax.axhline(d1_end, color='black', linestyle='--', alpha=0.5)
        ax.axhline(d2_end, color='black', linestyle='--', alpha=0.5)
        
        ax.text(-0.02, (0 + d1_end) / 2, 'D1-MSN', transform=ax.get_yaxis_transform(),
               ha='right', va='center', fontsize=12, color='red', fontweight='bold')
        ax.text(-0.02, (d1_end + d2_end) / 2, 'D2-MSN', transform=ax.get_yaxis_transform(),
               ha='right', va='center', fontsize=12, color='blue', fontweight='bold')
        ax.text(-0.02, (d2_end + y_offset) / 2, 'FSI', transform=ax.get_yaxis_transform(),
               ha='right', va='center', fontsize=12, color='green', fontweight='bold')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Neuron ID', fontsize=12)
        ax.set_title('Population Raster Plot (Color-coded by Type)', fontweight='bold', fontsize=14)
        ax.set_xlim([0, duration])
        ax.set_ylim([-1, y_offset + 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='D1-MSN'),
                          Patch(facecolor='blue', label='D2-MSN'),
                          Patch(facecolor='green', label='FSI')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved population raster to {save_path}")
        
        return fig
    
    def plot_delay_matrix_evolution(self, initial_delays: np.ndarray, 
                                   final_delays: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Visualize delay matrix evolution during learning.
        
        Shows MSN x FSI delay matrix before and after dopamine-driven plasticity
        to reveal structured patterns (e.g., stripes) or global shifts.
        
        Args:
            initial_delays: Initial delay matrix (saved before learning)
            final_delays: Final delay matrix (current state)
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Plot initial delays
        ax1 = axes[0]
        im1 = ax1.imshow(initial_delays, cmap='viridis', aspect='auto')
        ax1.set_xlabel('FSI Index', fontsize=11)
        ax1.set_ylabel('MSN Index', fontsize=11)
        ax1.set_title('Initial Delay Matrix', fontweight='bold')
        ax1.axhline(self.n_d1, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(self.n_fsis * 0.95, self.n_d1 * 0.5, 'D1', ha='right', va='center',
                color='red', fontweight='bold', fontsize=10)
        ax1.text(self.n_fsis * 0.95, self.n_d1 + self.n_d2 * 0.5, 'D2', ha='right', va='center',
                color='blue', fontweight='bold', fontsize=10)
        plt.colorbar(im1, ax=ax1, label='Delay (ms)')
        
        # Plot final delays
        ax2 = axes[1]
        im2 = ax2.imshow(final_delays, cmap='viridis', aspect='auto')
        ax2.set_xlabel('FSI Index', fontsize=11)
        ax2.set_ylabel('MSN Index', fontsize=11)
        ax2.set_title('Final Delay Matrix (After Learning)', fontweight='bold')
        ax2.axhline(self.n_d1, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(self.n_fsis * 0.95, self.n_d1 * 0.5, 'D1', ha='right', va='center',
                color='red', fontweight='bold', fontsize=10)
        ax2.text(self.n_fsis * 0.95, self.n_d1 + self.n_d2 * 0.5, 'D2', ha='right', va='center',
                color='blue', fontweight='bold', fontsize=10)
        plt.colorbar(im2, ax=ax2, label='Delay (ms)')
        
        # Plot change (difference)
        ax3 = axes[2]
        delay_change = final_delays - initial_delays
        im3 = ax3.imshow(delay_change, cmap='RdBu_r', aspect='auto',
                        vmin=-np.max(np.abs(delay_change)), vmax=np.max(np.abs(delay_change)))
        ax3.set_xlabel('FSI Index', fontsize=11)
        ax3.set_ylabel('MSN Index', fontsize=11)
        ax3.set_title('Delay Change (Final - Initial)', fontweight='bold')
        ax3.axhline(self.n_d1, color='black', linestyle='--', alpha=0.7, linewidth=2)
        plt.colorbar(im3, ax=ax3, label='Δ Delay (ms)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved delay matrix evolution to {save_path}")
        
        return fig
    
    def compute_d1_d2_balance_ratio(self, trial_history: List[Dict], window_sec: float = 0.5):
        """
        Compute rolling D1/D2 firing rate balance over trial duration.
        
        Args:
            trial_history: List of trial results
            window_sec: Rolling window size in seconds
            
        Returns:
            Dictionary with time points and D1/D2 ratios
        """
        # Concatenate all trials
        all_d1_spikes = []
        all_d2_spikes = []
        total_duration = 0
        
        for trial in trial_history:
            duration = trial['duration']
            
            # Collect D1 spikes
            for i in range(self.n_d1):
                spikes = trial['msn_spikes'][i] + total_duration
                all_d1_spikes.extend(spikes)
            
            # Collect D2 spikes
            for i in range(self.n_d1, self.n_msns):
                spikes = trial['msn_spikes'][i] + total_duration
                all_d2_spikes.extend(spikes)
            
            total_duration += duration
        
        all_d1_spikes = np.array(all_d1_spikes)
        all_d2_spikes = np.array(all_d2_spikes)
        
        # Compute rolling window firing rates
        time_points = np.arange(0, total_duration, 0.1)  # 100ms resolution
        d1_rates = np.zeros(len(time_points))
        d2_rates = np.zeros(len(time_points))
        
        for i, t in enumerate(time_points):
            # Count spikes in window
            d1_count = np.sum((all_d1_spikes >= t) & (all_d1_spikes < t + window_sec))
            d2_count = np.sum((all_d2_spikes >= t) & (all_d2_spikes < t + window_sec))
            
            d1_rates[i] = d1_count / (window_sec * self.n_d1)
            d2_rates[i] = d2_count / (window_sec * self.n_d2)
        
        # Compute ratio (with smoothing to avoid division by zero)
        epsilon = 0.1
        balance_ratio = d1_rates / (d2_rates + epsilon)
        
        return {
            'time': time_points,
            'd1_rates': d1_rates,
            'd2_rates': d2_rates,
            'balance_ratio': balance_ratio
        }
    
    def plot_d1_d2_balance(self, trial_history: List[Dict], save_path: Optional[str] = None):
        """
        Plot D1/D2 balance ratio over time.
        
        Args:
            trial_history: List of trial results
            save_path: Optional path to save figure
        """
        balance_data = self.compute_d1_d2_balance_ratio(trial_history)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Panel 1: Individual firing rates
        ax1 = axes[0]
        ax1.plot(balance_data['time'], balance_data['d1_rates'], 'r-', 
                label='D1-MSN Rate', linewidth=1.5, alpha=0.8)
        ax1.plot(balance_data['time'], balance_data['d2_rates'], 'b-', 
                label='D2-MSN Rate', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Firing Rate (Hz)', fontsize=11)
        ax1.set_title('D1 vs D2 Firing Rates Over Time', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Balance ratio
        ax2 = axes[1]
        ax2.plot(balance_data['time'], balance_data['balance_ratio'], 'k-', linewidth=2)
        ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Equal Balance')
        ax2.fill_between(balance_data['time'], 0, balance_data['balance_ratio'],
                        where=(balance_data['balance_ratio'] > 1), alpha=0.3, color='red',
                        label='D1 Dominant')
        ax2.fill_between(balance_data['time'], balance_data['balance_ratio'], 10,
                        where=(balance_data['balance_ratio'] < 1), alpha=0.3, color='blue',
                        label='D2 Dominant')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('D1/D2 Ratio', fontsize=11)
        ax2.set_title('D1/D2 Balance Ratio (Action Selection)', fontweight='bold')
        ax2.set_ylim([0, 5])
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved D1/D2 balance plot to {save_path}")
        
        return fig
    
    def apply_stdp_delay_plasticity(self,
                                   msn_spikes: List[np.ndarray],
                                   fsi_spikes: List[np.ndarray],
                                   dopamine_level: float,
                                   learning_rate: float = 0.01):
        """
        Apply STDP-based delay plasticity (Hebbian rule).
        
        This is more biologically plausible than the teleological rule.
        If spike arrives before postsynaptic resonance peak, increase delay.
        If after, decrease delay. Achieves tuning without "calculating" the target.
        
        Args:
            msn_spikes: List of MSN spike times
            fsi_spikes: List of FSI spike times
            dopamine_level: Current DA level (modulates plasticity)
            learning_rate: STDP learning rate
        """
        if dopamine_level < 0.5:  # Only apply plasticity with sufficient DA
            return
        
        # For each MSN-FSI connection
        for msn_idx in range(self.n_msns):
            if len(msn_spikes[msn_idx]) == 0:
                continue
            
            for fsi_idx in range(self.n_fsis):
                if len(fsi_spikes[fsi_idx]) == 0:
                    continue
                
                if self.fsi_msn_weights[msn_idx, fsi_idx] == 0:
                    continue  # No connection
                
                # Find spike pairs
                for msn_spike_time in msn_spikes[msn_idx]:
                    # Find nearest FSI spike
                    fsi_times = fsi_spikes[fsi_idx]
                    time_diffs = fsi_times - msn_spike_time
                    
                    # Only consider FSI spikes before MSN spike
                    valid_fsi_spikes = time_diffs[time_diffs < 0]
                    
                    if len(valid_fsi_spikes) > 0:
                        # Get most recent FSI spike
                        delta_t = valid_fsi_spikes[-1]  # Negative value
                        actual_delay = -delta_t * 1000  # Convert to ms
                        
                        # STDP rule: If delay is shorter than current, increase it
                        # If delay is longer, decrease it
                        current_delay = self.fsi_msn_delays[msn_idx, fsi_idx]
                        delay_error = actual_delay - current_delay
                        
                        # Update with DA modulation
                        da_mod = dopamine_level * self.da_sensitivity[msn_idx]
                        self.fsi_msn_delays[msn_idx, fsi_idx] += (
                            learning_rate * da_mod * np.sign(delay_error) * 0.5
                        )
                        
                        # Clip delays
                        self.fsi_msn_delays[msn_idx, fsi_idx] = np.clip(
                            self.fsi_msn_delays[msn_idx, fsi_idx], 0.5, 20.0
                        )
