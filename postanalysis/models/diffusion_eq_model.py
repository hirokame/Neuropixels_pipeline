"""
Delayed Diffusion Equation Model for LFP Propagation.

Implements Chapter 2.1: Diffusion Equation Models with conduction delays.

Based on Pouzzner's insight that long and varied conduction delays within the 
cortical-basal ganglia-thalamic loop facilitate resonance at specific frequencies.

Mathematical Model:
    ∂V/∂t = D∇²V(t - τ) + S(x,t)
    
Where:
    - V(x,t): LFP voltage at position x and time t
    - D: Diffusion coefficient (spatial decay rate)
    - τ: Conduction delay
    - S(x,t): Source term from spike locations and times
    - ∇²: Laplacian operator (spatial second derivative)

Key Features:
    1. Delayed diffusion term models temporal lag in signal propagation
    2. Anisotropic diffusion with different rates along vs across shanks
    3. Source terms derived from actual spike data
    4. Estimates optimal delay τ for resonance at task-relevant frequencies
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


class DelayedDiffusionModel:
    """
    Implements delayed diffusion equation for LFP propagation.
    
    This model extends classical diffusion by incorporating conduction delays,
    which Pouzzner proposes are critical for phase-adjustment and synchronization.
    """
    
    def __init__(self, 
                 spatial_grid_size: Tuple[int, int] = (100, 100),
                 spatial_resolution_um: float = 10.0,
                 time_step_ms: float = 0.1,
                 diffusion_coefficient: float = 1.0,
                 conduction_delay_ms: float = 5.0,
                 anisotropy_ratio: float = 2.0):
        """
        Initialize the delayed diffusion model.
        
        Args:
            spatial_grid_size: Size of spatial grid (x, y) in grid points
            spatial_resolution_um: Spatial resolution in micrometers per grid point
            time_step_ms: Time step for simulation in milliseconds
            diffusion_coefficient: Diffusion coefficient D (spatial decay rate)
            conduction_delay_ms: Conduction delay τ in milliseconds
            anisotropy_ratio: Ratio of diffusion along vs across shanks
        """
        self.nx, self.ny = spatial_grid_size
        self.dx = spatial_resolution_um  # μm
        self.dt = time_step_ms / 1000.0  # Convert to seconds
        self.D = diffusion_coefficient
        self.tau = conduction_delay_ms / 1000.0  # Convert to seconds
        self.anisotropy = anisotropy_ratio
        
        # Initialize voltage field
        self.V = np.zeros((self.nx, self.ny))
        
        # History buffer for delayed terms (stores past states)
        self.delay_steps = int(self.tau / self.dt)
        self.V_history = np.zeros((self.delay_steps + 1, self.nx, self.ny))
        self.time_index = 0
        
        # Anisotropic diffusion tensor (different rates in x vs y)
        self.D_tensor = np.array([[self.D, 0], 
                                  [0, self.D / self.anisotropy]])
        
        print(f"Initialized DelayedDiffusionModel:")
        print(f"  Grid: {self.nx}x{self.ny}, Resolution: {self.dx}μm")
        print(f"  Time step: {self.dt*1000:.3f}ms, Delay: {self.tau*1000:.1f}ms ({self.delay_steps} steps)")
        print(f"  Diffusion coefficient: {self.D}, Anisotropy: {self.anisotropy}")
    
    def compute_anisotropic_laplacian(self, V: np.ndarray) -> np.ndarray:
        """
        Compute anisotropic Laplacian with different diffusion rates.
        
        Implements: ∇·(D_tensor ∇V) where D_tensor is anisotropic.
        This models different propagation rates along vs across shanks.
        
        Args:
            V: Voltage field (nx x ny)
            
        Returns:
            Anisotropic Laplacian of V
        """
        # Compute gradients
        dVdx = np.gradient(V, self.dx, axis=0)
        dVdy = np.gradient(V, self.dx, axis=1)
        
        # Apply diffusion tensor
        flux_x = self.D_tensor[0, 0] * dVdx
        flux_y = self.D_tensor[1, 1] * dVdy
        
        # Compute divergence of flux
        laplacian = np.gradient(flux_x, self.dx, axis=0) + np.gradient(flux_y, self.dx, axis=1)
        
        return laplacian
    
    def add_spike_source(self, x_pos: float, y_pos: float, amplitude: float = 1.0):
        """
        Add a spike source at specified location.
        
        Source term S(x,t) represents current injection from spiking activity.
        
        Args:
            x_pos: X position in micrometers
            y_pos: Y position in micrometers
            amplitude: Source amplitude (current injection strength)
        """
        # Convert to grid coordinates
        i = int(x_pos / self.dx)
        j = int(y_pos / self.dx)
        
        # Bounds checking
        if 0 <= i < self.nx and 0 <= j < self.ny:
            # Add Gaussian source (spatially extended spike)
            sigma = 2.0  # Grid points
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < self.nx and 0 <= jj < self.ny:
                        weight = np.exp(-(di**2 + dj**2) / (2 * sigma**2))
                        self.V[ii, jj] += amplitude * weight
    
    def step(self, source_term: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advance simulation by one time step using delayed diffusion.
        
        Implements: ∂V/∂t = D∇²V(t - τ) + S(x,t)
        
        Args:
            source_term: Optional external source S(x,t) at current time
            
        Returns:
            Updated voltage field V(t+dt)
        """
        # Get delayed voltage field V(t - τ)
        delay_idx = self.delay_steps
        V_delayed = self.V_history[delay_idx]
        
        # Compute anisotropic Laplacian of delayed field
        laplacian = self.compute_anisotropic_laplacian(V_delayed)
        
        # Add source term if provided
        if source_term is None:
            source_term = np.zeros_like(self.V)
        
        # Update voltage: Forward Euler integration
        # dV/dt = D∇²V(t-τ) + S
        self.V += self.dt * (laplacian + source_term)
        
        # Update history buffer (circular buffer)
        self.V_history = np.roll(self.V_history, 1, axis=0)
        self.V_history[0] = self.V.copy()
        
        self.time_index += 1
        
        return self.V.copy()
    
    def estimate_resonant_frequency(self, V_timeseries: np.ndarray, 
                                    sampling_rate_hz: float) -> Tuple[float, np.ndarray]:
        """
        Estimate resonant frequency from LFP power spectrum.
        
        Pouzzner's theory: Delays tune the circuit to specific frequencies
        that are optimal for task-relevant synchronization.
        
        Args:
            V_timeseries: Time series of voltage at a location (n_timepoints,)
            sampling_rate_hz: Sampling rate in Hz
            
        Returns:
            peak_freq: Dominant frequency in Hz
            power_spectrum: Power spectrum (freq, power) arrays
        """
        # Compute power spectrum
        fft = np.fft.rfft(V_timeseries)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(V_timeseries), 1.0 / sampling_rate_hz)
        
        # Find peak frequency (excluding DC)
        peak_idx = np.argmax(power[1:]) + 1
        peak_freq = freqs[peak_idx]
        
        return peak_freq, (freqs, power)
    
    def optimize_delay_for_frequency(self, 
                                    target_freq_hz: float,
                                    test_delays_ms: np.ndarray,
                                    n_steps: int = 1000) -> Tuple[float, np.ndarray]:
        """
        Find optimal conduction delay to maximize power at target frequency.
        
        This implements Pouzzner's concept that delays are tuned to facilitate
        resonance at task-relevant frequencies.
        
        Args:
            target_freq_hz: Target resonant frequency (e.g., theta ~8Hz, beta ~20Hz)
            test_delays_ms: Array of delays to test in milliseconds
            n_steps: Number of simulation steps per delay
            
        Returns:
            optimal_delay_ms: Delay that maximizes power at target frequency
            power_at_target: Power at target frequency for each tested delay
        """
        print(f"Optimizing delay for target frequency: {target_freq_hz:.1f} Hz")
        
        original_tau = self.tau
        original_delay_steps = self.delay_steps
        
        powers = []
        
        for delay_ms in test_delays_ms:
            # Reset model with new delay
            self.tau = delay_ms / 1000.0
            self.delay_steps = int(self.tau / self.dt)
            self.V = np.zeros((self.nx, self.ny))
            self.V_history = np.zeros((self.delay_steps + 1, self.nx, self.ny))
            
            # Run simulation with periodic input at target frequency
            center_x, center_y = self.nx // 2, self.ny // 2
            timeseries = []
            
            for step in range(n_steps):
                # Periodic source at target frequency
                t = step * self.dt
                source = np.zeros_like(self.V)
                amplitude = np.sin(2 * np.pi * target_freq_hz * t)
                self.add_spike_source(center_x * self.dx, center_y * self.dx, amplitude)
                
                self.step(source)
                timeseries.append(self.V[center_x, center_y])
            
            # Measure power at target frequency
            timeseries = np.array(timeseries)
            _, (freqs, power_spec) = self.estimate_resonant_frequency(
                timeseries, 1.0 / self.dt
            )
            
            # Interpolate to get power at exact target frequency
            power_interp = interp1d(freqs, power_spec, kind='cubic', fill_value='extrapolate')
            power_at_target = power_interp(target_freq_hz)
            powers.append(power_at_target)
        
        powers = np.array(powers)
        optimal_idx = np.argmax(powers)
        optimal_delay_ms = test_delays_ms[optimal_idx]
        
        # Restore original parameters
        self.tau = original_tau
        self.delay_steps = original_delay_steps
        
        print(f"  Optimal delay: {optimal_delay_ms:.2f}ms (power: {powers[optimal_idx]:.2e})")
        
        return optimal_delay_ms, powers
    
    def simulate_spike_driven_propagation(self,
                                         spike_times: np.ndarray,
                                         spike_positions: np.ndarray,
                                         duration_sec: float,
                                         recording_positions: Optional[np.ndarray] = None,
                                         spike_amplitude: float = 0.1) -> Dict:
        """
        Simulate LFP propagation driven by actual spike data.
        
        Args:
            spike_times: Array of spike times in seconds
            spike_positions: Array of spike positions (n_spikes, 2) in micrometers
            duration_sec: Total simulation duration
            recording_positions: Positions to record LFP (n_channels, 2) in μm
            spike_amplitude: Amplitude of spike source injection
            
        Returns:
            results: Dictionary with 'time', 'lfp_signals', 'V_snapshots'
        """
        n_steps = int(duration_sec / self.dt)
        
        # Default recording positions (center and corners)
        if recording_positions is None:
            recording_positions = np.array([
                [self.nx * self.dx / 2, self.ny * self.dx / 2],  # Center
                [self.nx * self.dx * 0.25, self.ny * self.dx * 0.25],  # Corners
                [self.nx * self.dx * 0.75, self.ny * self.dx * 0.75],
            ])
        
        n_channels = len(recording_positions)
        lfp_signals = np.zeros((n_steps, n_channels))
        time_array = np.arange(n_steps) * self.dt
        
        # Store snapshots at regular intervals
        snapshot_interval = max(1, n_steps // 10)
        snapshots = []
        snapshot_times = []
        
        print(f"Simulating spike-driven LFP propagation for {duration_sec}s...")
        
        for step in range(n_steps):
            current_time = step * self.dt
            
            # Find spikes in current time window
            spike_mask = (spike_times >= current_time - self.dt/2) & \
                        (spike_times < current_time + self.dt/2)
            current_spikes = spike_positions[spike_mask]
            
            # Add spike sources
            for spike_pos in current_spikes:
                self.add_spike_source(spike_pos[0], spike_pos[1], amplitude=spike_amplitude)
            
            # Advance simulation
            self.step()
            
            # Record LFP at specified positions
            for ch_idx, pos in enumerate(recording_positions):
                i = int(pos[0] / self.dx)
                j = int(pos[1] / self.dx)
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    lfp_signals[step, ch_idx] = self.V[i, j]
            
            # Store snapshots
            if step % snapshot_interval == 0:
                snapshots.append(self.V.copy())
                snapshot_times.append(current_time)
        
        print(f"  Simulation complete. Recorded {n_channels} channels.")
        
        return {
            'time': time_array,
            'lfp_signals': lfp_signals,
            'recording_positions': recording_positions,
            'V_snapshots': np.array(snapshots),
            'snapshot_times': np.array(snapshot_times)
        }
    
    def plot_propagation_snapshots(self, results: Dict, save_path: Optional[str] = None,
                                  extent: Optional[Tuple[float, float, float, float]] = None):
        """
        Visualize LFP propagation over time.
        
        Args:
            results: Output from simulate_spike_driven_propagation
            save_path: Optional path to save figure
            extent: Optional (min_x, max_x, min_y, max_y) for axes labels
        """
        snapshots = results['V_snapshots']
        snapshot_times = results['snapshot_times']
        
        n_snapshots = len(snapshots)
        ncols = min(5, n_snapshots)
        nrows = (n_snapshots + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
        if n_snapshots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        vmin, vmax = snapshots.min(), snapshots.max()
        
        # Determine extent
        if extent is None:
            extent = [0, self.nx*self.dx, 0, self.ny*self.dx]
        
        for idx, (snapshot, t) in enumerate(zip(snapshots, snapshot_times)):
            ax = axes[idx]
            im = ax.imshow(snapshot.T, origin='lower', cmap='RdBu_r',
                          vmin=vmin, vmax=vmax, extent=extent)
            ax.set_title(f't = {t*1000:.1f}ms')
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
            plt.colorbar(im, ax=ax, label='V (mV)')
        
        # Hide unused subplots
        for idx in range(n_snapshots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved propagation visualization to {save_path}")
        
        return fig
    
    def plot_isopotential_contours(self, results: Dict, save_path: Optional[str] = None,
                                   extent: Optional[Tuple[float, float, float, float]] = None):
        """
        Plot iso-potential contours to visualize anisotropy.
        
        If propagation is truly elliptical (anisotropic), contours should be ellipses
        aligned with the shank direction, not circles.
        
        Args:
            results: Output from simulate_spike_driven_propagation
            save_path: Optional path to save figure
            extent: Optional (min_x, max_x, min_y, max_y) for axes labels
        """
        snapshots = results['V_snapshots']
        snapshot_times = results['snapshot_times']
        
        # Select representative snapshots (peak activity)
        n_snapshots = min(6, len(snapshots))
        indices = np.linspace(0, len(snapshots)-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        if extent is None:
            extent = [0, self.nx*self.dx, 0, self.ny*self.dx]
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            snapshot = snapshots[idx]
            
            # Plot heatmap background
            im = ax.imshow(snapshot.T, origin='lower', cmap='RdBu_r',
                          extent=extent, alpha=0.3)
            
            # Plot contour lines at fixed voltage levels
            max_val = np.max(np.abs(snapshot))
            levels = [0.8*max_val, 0.6*max_val, 0.4*max_val, 0.2*max_val]
            
            X = np.linspace(extent[0], extent[1], snapshot.shape[0])
            Y = np.linspace(extent[2], extent[3], snapshot.shape[1])
            XX, YY = np.meshgrid(X, Y)
            
            contours = ax.contour(XX, YY, snapshot.T, levels=levels, 
                                 colors='black', linewidths=2)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
            
            ax.set_title(f't = {snapshot_times[idx]*1000:.1f}ms', fontweight='bold')
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
            ax.set_aspect('equal')
            
            # Add annotation about expected anisotropy
            if i == 0:
                ax.text(0.05, 0.95, 'Expected: Elliptical\ncontours if anisotropic', 
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved isopotential contours to {save_path}")
        
        return fig
    
    def plot_phase_distance_analysis(self, results: Dict, spike_positions: np.ndarray,
                                    target_frequency_hz: float = 8.0,
                                    save_path: Optional[str] = None):
        """
        Plot phase vs distance to analyze traveling wave behavior.
        
        For delayed diffusion/traveling waves, phase lag should accumulate with distance
        linearly. The slope gives the phase conduction velocity.
        
        Args:
            results: Output from simulate_spike_driven_propagation
            spike_positions: Original spike positions (n_spikes, 2)
            target_frequency_hz: Frequency band to analyze
            save_path: Optional path to save figure
        """
        from scipy.signal import hilbert
        
        lfp_signals = results['lfp_signals']
        recording_positions = results['recording_positions']
        time = results['time']
        fs = 1.0 / self.dt
        
        # Find spike source center (centroid)
        spike_center = np.mean(spike_positions, axis=0)
        
        # Compute distance from spike source for each recording site
        distances = np.array([np.linalg.norm(pos - spike_center) 
                             for pos in recording_positions])
        
        # Bandpass filter around target frequency
        from scipy.signal import butter, filtfilt
        
        lowcut = target_frequency_hz * 0.8
        highcut = target_frequency_hz * 1.2
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        if high >= 1.0:
            high = 0.99
        
        try:
            b, a = butter(3, [low, high], btype='band')
        except:
            print(f"Warning: Could not design filter for {target_frequency_hz}Hz")
            return None
        
        # Compute phase for each channel
        phases = np.zeros(len(recording_positions))
        
        for i in range(len(recording_positions)):
            signal = lfp_signals[:, i]
            
            if np.std(signal) > 0:
                # Filter signal
                filtered = filtfilt(b, a, signal)
                
                # Compute instantaneous phase using Hilbert transform
                analytic_signal = hilbert(filtered)
                instantaneous_phase = np.angle(analytic_signal)
                
                # Take median phase (or phase at peak activity)
                peak_idx = np.argmax(np.abs(filtered))
                phases[i] = instantaneous_phase[peak_idx]
        
        # Plot phase vs distance
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Phase vs Distance scatter with fit
        ax1 = axes[0]
        scatter = ax1.scatter(distances, phases, c=distances, cmap='viridis', s=100, edgecolors='black')
        ax1.set_xlabel('Distance from Spike Source (μm)', fontsize=12)
        ax1.set_ylabel(f'Phase at {target_frequency_hz}Hz (radians)', fontsize=12)
        ax1.set_title('Phase-Distance Plot (Traveling Wave Analysis)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Distance (μm)')
        
        # Fit linear regression
        if len(distances) > 2:
            z = np.polyfit(distances, phases, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(distances.min(), distances.max(), 100)
            ax1.plot(x_fit, p(x_fit), 'r--', linewidth=2, label=f'Fit: slope={z[0]:.4f} rad/μm')
            
            # Compute phase velocity
            phase_velocity_um_per_s = 2 * np.pi * target_frequency_hz / np.abs(z[0])
            ax1.text(0.05, 0.95, f'Phase velocity: {phase_velocity_um_per_s:.1f} μm/s\n' + 
                    f'= {phase_velocity_um_per_s/1000:.2f} mm/s',
                    transform=ax1.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax1.legend()
        
        # Panel 2: Unwrapped phase for clarity
        ax2 = axes[1]
        unwrapped_phases = np.unwrap(phases)
        scatter2 = ax2.scatter(distances, unwrapped_phases, c=distances, cmap='viridis', 
                              s=100, edgecolors='black')
        ax2.set_xlabel('Distance from Spike Source (μm)', fontsize=12)
        ax2.set_ylabel(f'Unwrapped Phase at {target_frequency_hz}Hz', fontsize=12)
        ax2.set_title('Unwrapped Phase (Linear=Traveling Wave)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Distance (μm)')
        
        # Fit unwrapped
        if len(distances) > 2:
            z_unw = np.polyfit(distances, unwrapped_phases, 1)
            p_unw = np.poly1d(z_unw)
            ax2.plot(x_fit, p_unw(x_fit), 'r--', linewidth=2, 
                    label=f'Slope={z_unw[0]:.4f} rad/μm')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved phase-distance analysis to {save_path}")
        
        return fig
    
    def plot_frequency_delay_heatmap(self, test_frequencies: np.ndarray = None,
                                   test_delays: np.ndarray = None,
                                   n_steps: int = 500,
                                   save_path: Optional[str] = None):
        """
        Plot frequency-delay optimization heatmap.
        
        Shows whether a single delay can support resonance at multiple harmonic
        frequencies simultaneously, or if the optimum is a sharp peak.
        
        Args:
            test_frequencies: Array of frequencies to test (Hz)
            test_delays: Array of delays to test (ms)
            n_steps: Number of simulation steps
            save_path: Optional path to save figure
        """
        if test_frequencies is None:
            test_frequencies = np.linspace(4, 80, 20)  # 4-80 Hz
        
        if test_delays is None:
            test_delays = np.linspace(1, 20, 20)  # 1-20 ms
        
        # Create power matrix
        power_matrix = np.zeros((len(test_frequencies), len(test_delays)))
        
        print(f"Computing frequency-delay heatmap ({len(test_frequencies)}x{len(test_delays)} grid)...")
        
        for i, freq in enumerate(test_frequencies):
            for j, delay_ms in enumerate(test_delays):
                # Temporarily set delay
                original_delay = self.tau
                original_delay_steps = self.delay_steps
                
                self.tau = delay_ms / 1000.0
                self.delay_steps = int(self.tau / self.dt)
                self.V_history = np.zeros((self.delay_steps + 1, self.nx, self.ny))
                
                # Run short simulation with oscillating source
                V_center_history = []
                center_x, center_y = self.nx // 2, self.ny // 2
                
                for step in range(n_steps):
                    t = step * self.dt
                    # Oscillating source at target frequency
                    source_amplitude = np.sin(2 * np.pi * freq * t)
                    source = np.zeros((self.nx, self.ny))
                    source[center_x, center_y] = source_amplitude
                    
                    self.step(source)
                    V_center_history.append(self.V[center_x, center_y])
                
                # Compute power at target frequency
                from scipy.signal import welch
                fs = 1.0 / self.dt
                f, psd = welch(V_center_history, fs=fs, nperseg=min(256, len(V_center_history)))
                
                # Find power at target frequency
                freq_idx = np.argmin(np.abs(f - freq))
                power_matrix[i, j] = psd[freq_idx]
                
                # Restore original delay
                self.tau = original_delay
                self.delay_steps = original_delay_steps
                self.V_history = np.zeros((self.delay_steps + 1, self.nx, self.ny))
                self.V = np.zeros((self.nx, self.ny))
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(power_matrix, aspect='auto', origin='lower', cmap='hot',
                      extent=[test_delays[0], test_delays[-1], 
                             test_frequencies[0], test_frequencies[-1]])
        
        ax.set_xlabel('Conduction Delay τ (ms)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title('Frequency-Delay Optimization Heatmap\n(Bright = High Resonance Power)', 
                    fontweight='bold', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power at Target Frequency', fontsize=11)
        
        # Add theoretical optimal delay line (quarter-cycle)
        theoretical_delays = 1000.0 / (4.0 * test_frequencies)
        ax.plot(theoretical_delays, test_frequencies, 'c--', linewidth=2, 
               label='Theoretical τ = 1/(4f)')
        ax.legend(fontsize=11)
        
        # Add harmonic annotations
        base_freq = 8.0  # Theta
        for n in [1, 2, 3, 4]:
            harmonic_freq = n * base_freq
            if harmonic_freq <= test_frequencies[-1]:
                ax.axhline(harmonic_freq, color='cyan', linestyle=':', alpha=0.5, linewidth=1)
                ax.text(test_delays[-1]*0.95, harmonic_freq, f'{n}×{base_freq}Hz', 
                       color='cyan', fontsize=8, va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved frequency-delay heatmap to {save_path}")
        
        return fig
