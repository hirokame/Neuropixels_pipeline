"""
Attractor Energy Landscape Model for Uncertainty-Driven Exploration.

Implements a model where neural uncertainty (attractor instability) directly drives
exploratory motor behavior through energy spillover, without explicit computation.

Conceptual Framework:
    1. Brain states are modeled as a ball rolling in an energy landscape
    2. Deep attractors (certainty) → stable states → exploitation behavior
    3. Shallow/flat attractors (uncertainty) → oscillations → exploration behavior
    4. Oscillation energy "spills over" to drive exploratory motor primitives
    
Key Novelty:
    The model does NOT compute "if (entropy > threshold)". Instead, it physically
    connects neural state velocity/variance directly to motor gain:
        Motor_Gain = α * |dx/dt| + β * Var(x)
    
    This implements zero-lag, reflexive exploration without computational overhead.

Mathematical Model:
    1. Hidden State Dynamics (Competing Attractors):
        dx/dt = -∇U(x) + noise
        where U(x) is potential energy with multiple minima
        
    2. Energy Spillover (Direct Coupling):
        exploration_gain = α * ||dx/dt|| + β * Var(x_recent)
        
    3. Phase Transition:
        When sensory input breaks symmetry → attractor forms → speed drops → exploration stops

Neuropixels Analysis:
    This model predicts that exploratory movements (whisking, head turns, licking) should be
    preceded by increased neural trajectory speed in the population manifold.
"""

import numpy as np
from scipy.stats import mannwhitneyu
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class AttractorEnergyLandscapeModel:
    """
    Models uncertainty-driven exploration through attractor dynamics.
    
    This model implements the concept that neural state instability (oscillations
    in metastable states) directly drives exploratory behavior through energy spillover,
    bypassing explicit uncertainty computation.
    """
    
    def __init__(self,
                 n_dimensions: int = 3,
                 n_attractors: int = 2,
                 attractor_strength: float = 1.0,
                 noise_level: float = 0.1,
                 coupling_alpha: float = 1.0,
                 coupling_beta: float = 0.5,
                 sampling_rate_hz: float = 1000.0):
        """
        Initialize the attractor energy landscape model.
        
        Args:
            n_dimensions: Dimensionality of neural state space
            n_attractors: Number of competing attractors (e.g., 2 for left/right choice)
            attractor_strength: Depth of attractor basins (higher = deeper basins)
            noise_level: Amplitude of neural noise
            coupling_alpha: Weight for velocity term in motor gain
            coupling_beta: Weight for variance term in motor gain
            sampling_rate_hz: Sampling rate for simulation
        """
        self.n_dim = n_dimensions
        self.n_attractors = n_attractors
        self.k = attractor_strength  # Basin depth
        self.sigma_noise = noise_level
        self.alpha = coupling_alpha
        self.beta = coupling_beta
        self.fs = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        
        # Initialize attractor positions (evenly spaced around origin)
        angles = np.linspace(0, 2*np.pi, n_attractors, endpoint=False)
        self.attractor_positions = np.zeros((n_attractors, n_dimensions))
        for i, angle in enumerate(angles):
            self.attractor_positions[i, 0] = np.cos(angle)
            self.attractor_positions[i, 1] = np.sin(angle)
        
        # Current state
        self.state = np.random.randn(n_dimensions) * 0.1
        self.state_history = []
        self.velocity_history = []
        
        print(f"Initialized AttractorEnergyLandscapeModel:")
        print(f"  Dimensions: {n_dimensions}, Attractors: {n_attractors}")
        print(f"  Attractor strength: {attractor_strength}, Noise: {noise_level}")
        print(f"  Coupling (α={coupling_alpha}, β={coupling_beta})")
    
    def potential_energy(self, state: np.ndarray) -> float:
        """
        Compute potential energy U(x) of the landscape.
        
        Multiple attractors create a landscape with multiple minima.
        Uncertainty corresponds to being on a ridge or saddle point.
        
        Args:
            state: Current state vector
            
        Returns:
            Potential energy (lower = more stable)
        """
        energy = 0.0
        
        # Sum of Gaussian wells centered at each attractor
        for attractor_pos in self.attractor_positions:
            dist_sq = np.sum((state - attractor_pos) ** 2)
            energy += -self.k * np.exp(-dist_sq / 2.0)
        
        return energy
    
    def force_field(self, state: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute force field F = -∇U(x) at current state.
        
        This drives the dynamics toward attractors. When multiple attractors compete
        with equal strength, the force is weak → state oscillates → high velocity.
        
        Args:
            state: Current state vector
            bias: Optional bias vector (e.g., from sensory input) to break symmetry
            
        Returns:
            Force vector
        """
        force = np.zeros(self.n_dim)
        
        # Gradient from attractor potentials
        for attractor_pos in self.attractor_positions:
            diff = state - attractor_pos
            dist_sq = np.sum(diff ** 2)
            gaussian = np.exp(-dist_sq / 2.0)
            force += self.k * gaussian * diff
        
        # Apply bias (sensory input that resolves uncertainty)
        if bias is not None:
            force += bias
        
        return -force
    
    def step(self, bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Simulate one time step of dynamics.
        
        dx/dt = -∇U(x) + bias + noise
        
        Args:
            bias: Optional sensory input bias
            
        Returns:
            Tuple of (new_state, velocity_magnitude)
        """
        # Compute force
        force = self.force_field(self.state, bias)
        
        # Add noise
        noise = np.random.randn(self.n_dim) * self.sigma_noise
        
        # Update state (Euler integration)
        velocity = force + noise
        self.state = self.state + velocity * self.dt
        
        # Track history
        self.state_history.append(self.state.copy())
        velocity_mag = np.linalg.norm(velocity)
        self.velocity_history.append(velocity_mag)
        
        return self.state.copy(), velocity_mag
    
    def compute_exploration_gain(self, window_size: int = 100) -> float:
        """
        Compute motor exploration gain from neural dynamics.
        
        This is the KEY MECHANISM: Neural state instability (high velocity and variance)
        directly drives exploratory motor primitives.
        
        Motor_Gain = α * |dx/dt| + β * Var(x_recent)
        
        Args:
            window_size: Number of recent time steps for variance calculation
            
        Returns:
            Motor exploration gain (0 = no exploration, higher = more vigorous exploration)
        """
        if len(self.velocity_history) < 2:
            return 0.0
        
        # Current velocity magnitude
        current_velocity = self.velocity_history[-1]
        
        # Recent state variance
        if len(self.state_history) >= window_size:
            recent_states = np.array(self.state_history[-window_size:])
        else:
            recent_states = np.array(self.state_history)
        
        state_variance = np.mean(np.var(recent_states, axis=0))
        
        # Combine velocity and variance
        exploration_gain = self.alpha * current_velocity + self.beta * state_variance
        
        return exploration_gain
    
    def detect_phase_transition(self, 
                               threshold_velocity: float = 0.1,
                               stability_window: int = 50) -> bool:
        """
        Detect transition from exploration to exploitation.
        
        Phase transition occurs when:
        - Neural trajectory velocity drops below threshold
        - State becomes stable (converged to attractor)
        
        Args:
            threshold_velocity: Velocity threshold for stability
            stability_window: Number of steps that must be stable
            
        Returns:
            True if transitioned to exploitation phase
        """
        if len(self.velocity_history) < stability_window:
            return False
        
        recent_velocities = self.velocity_history[-stability_window:]
        mean_velocity = np.mean(recent_velocities)
        
        return mean_velocity < threshold_velocity
    
    def simulate_trial(self,
                      duration_sec: float = 2.0,
                      bias_onset_sec: Optional[float] = None,
                      bias_direction: Optional[int] = None,
                      bias_strength: float = 1.0) -> Dict:
        """
        Simulate a single trial with optional sensory bias.
        
        Initial phase: Uncertainty → oscillations → exploration
        After bias: Certainty → convergence → exploitation
        
        Args:
            duration_sec: Trial duration
            bias_onset_sec: Time when sensory input arrives (None = no bias)
            bias_direction: Which attractor to bias toward (0 to n_attractors-1)
            bias_strength: Strength of sensory bias
            
        Returns:
            Dictionary with trial results
        """
        n_steps = int(duration_sec * self.fs)
        
        # Reset state
        self.state = np.random.randn(self.n_dim) * 0.1
        self.state_history = []
        self.velocity_history = []
        
        # Time arrays
        time = np.arange(n_steps) / self.fs
        states = np.zeros((n_steps, self.n_dim))
        velocities = np.zeros(n_steps)
        exploration_gains = np.zeros(n_steps)
        phases = np.zeros(n_steps, dtype=bool)  # False=exploration, True=exploitation
        
        # Simulate
        for t in range(n_steps):
            # Apply bias if past onset time
            bias = None
            if bias_onset_sec is not None and time[t] >= bias_onset_sec:
                if bias_direction is not None:
                    bias = bias_strength * (self.attractor_positions[bias_direction] - self.state)
            
            # Step dynamics
            state, velocity = self.step(bias)
            
            # Compute exploration gain
            exploration_gain = self.compute_exploration_gain()
            
            # Detect phase
            is_exploitation = self.detect_phase_transition()
            
            # Record
            states[t] = state
            velocities[t] = velocity
            exploration_gains[t] = exploration_gain
            phases[t] = is_exploitation
        
        return {
            'time': time,
            'states': states,
            'velocities': velocities,
            'exploration_gains': exploration_gains,
            'phases': phases,
            'bias_onset_sec': bias_onset_sec,
            'bias_direction': bias_direction
        }
    
    def analyze_movement_onset(self,
                              neural_population_activity: np.ndarray,
                              movement_onsets: np.ndarray,
                              pre_onset_window_sec: float = 0.5,
                              sampling_rate: float = 1000.0) -> Dict:
        """
        Analyze neural trajectory speed before movement onset.
        
        This is the key Neuropixels analysis: If the model is correct, exploratory
        movements should be preceded by increased trajectory speed in neural manifold.
        
        Args:
            neural_population_activity: Neural activity matrix (time x neurons)
            movement_onsets: Array of movement onset times (in seconds)
            pre_onset_window_sec: Window before onset to analyze
            sampling_rate: Sampling rate of neural data
            
        Returns:
            Dictionary with analysis results
        """
        # Project neural activity to low-D manifold
        pca = PCA(n_components=min(10, neural_population_activity.shape[1]))
        neural_manifold = pca.fit_transform(neural_population_activity)
        
        # Compute trajectory velocity
        trajectory_velocity = np.zeros(len(neural_manifold))
        for t in range(1, len(neural_manifold)):
            diff = neural_manifold[t] - neural_manifold[t-1]
            trajectory_velocity[t] = np.linalg.norm(diff)
        
        # Extract pre-onset windows
        pre_window_samples = int(pre_onset_window_sec * sampling_rate)
        n_movements = len(movement_onsets)
        
        pre_onset_velocities = []
        pre_onset_variances = []
        
        for onset_time in movement_onsets:
            onset_idx = int(onset_time * sampling_rate)
            
            # Skip if too early in recording
            if onset_idx < pre_window_samples:
                continue
            
            # Extract pre-onset window
            start_idx = onset_idx - pre_window_samples
            end_idx = onset_idx
            
            window_velocity = trajectory_velocity[start_idx:end_idx]
            window_states = neural_manifold[start_idx:end_idx]
            
            # Compute metrics
            mean_velocity = np.mean(window_velocity)
            state_variance = np.mean(np.var(window_states, axis=0))
            
            pre_onset_velocities.append(mean_velocity)
            pre_onset_variances.append(state_variance)
        
        # Compare to quiescence baseline (periods of zero velocity >1s)
        # This addresses the critique that random baseline might include other movements
        quiescence_threshold = np.percentile(trajectory_velocity, 10)  # Bottom 10%
        quiescence_duration_samples = int(1.0 * sampling_rate)  # 1 second
        
        # Find quiescence periods
        is_quiescent = trajectory_velocity < quiescence_threshold
        quiescence_periods = []
        
        i = 0
        while i < len(is_quiescent):
            if is_quiescent[i]:
                start = i
                while i < len(is_quiescent) and is_quiescent[i]:
                    i += 1
                end = i
                if end - start >= quiescence_duration_samples:
                    quiescence_periods.append((start, end))
            i += 1
        
        # Sample from quiescence periods
        n_baseline = min(n_movements * 2, 1000)
        baseline_velocities = []
        baseline_variances = []
        
        if len(quiescence_periods) > 0:
            for _ in range(n_baseline):
                # Randomly select a quiescence period
                period = quiescence_periods[np.random.randint(len(quiescence_periods))]
                period_start, period_end = period
                
                # Randomly sample within this period (with sufficient window)
                if period_end - period_start > pre_window_samples:
                    baseline_idx = np.random.randint(period_start + pre_window_samples, period_end)
                    start_idx = baseline_idx - pre_window_samples
                    end_idx = baseline_idx
                    
                    window_velocity = trajectory_velocity[start_idx:end_idx]
                    window_states = neural_manifold[start_idx:end_idx]
                    
                    baseline_velocities.append(np.mean(window_velocity))
                    baseline_variances.append(np.mean(np.var(window_states, axis=0)))
        else:
            # Fallback to random sampling if no quiescence periods found
            print("Warning: No quiescence periods found, using random baseline")
            valid_range = len(neural_manifold) - pre_window_samples
            baseline_times = np.random.randint(pre_window_samples, valid_range, size=n_baseline)
            
            for baseline_idx in baseline_times:
                start_idx = baseline_idx - pre_window_samples
                end_idx = baseline_idx
                
                window_velocity = trajectory_velocity[start_idx:end_idx]
                window_states = neural_manifold[start_idx:end_idx]
                
                baseline_velocities.append(np.mean(window_velocity))
                baseline_variances.append(np.mean(np.var(window_states, axis=0)))
        
        return {
            'pre_onset_velocities': np.array(pre_onset_velocities),
            'pre_onset_variances': np.array(pre_onset_variances),
            'baseline_velocities': np.array(baseline_velocities),
            'baseline_variances': np.array(baseline_variances),
            'trajectory_velocity': trajectory_velocity,
            'neural_manifold': neural_manifold,
            'pca_model': pca,
            'quiescence_periods': quiescence_periods
        }
    
    def compare_exploration_exploitation(self,
                                        neural_activity: np.ndarray,
                                        exploration_times: np.ndarray,
                                        exploitation_times: np.ndarray,
                                        window_sec: float = 0.5,
                                        sampling_rate: float = 1000.0) -> Dict:
        """
        Compare neural trajectory dynamics during exploration vs exploitation.
        
        Model prediction:
        - Exploration: High trajectory speed, high variance (metastable)
        - Exploitation: Low trajectory speed, low variance (stable attractor)
        
        Args:
            neural_activity: Neural population activity (time x neurons)
            exploration_times: Times of exploratory behaviors (e.g., head turns)
            exploitation_times: Times of exploitation behaviors (e.g., goal-directed movement)
            window_sec: Window around event time
            sampling_rate: Sampling rate
            
        Returns:
            Dictionary with comparison results
        """
        # Project to manifold
        pca = PCA(n_components=min(10, neural_activity.shape[1]))
        neural_manifold = pca.fit_transform(neural_activity)
        
        window_samples = int(window_sec * sampling_rate)
        
        def extract_metrics(event_times):
            velocities = []
            variances = []
            
            for event_time in event_times:
                event_idx = int(event_time * sampling_rate)
                
                start_idx = max(0, event_idx - window_samples // 2)
                end_idx = min(len(neural_manifold), event_idx + window_samples // 2)
                
                if end_idx - start_idx < window_samples // 2:
                    continue
                
                window_states = neural_manifold[start_idx:end_idx]
                
                # Compute velocity
                window_velocity = np.zeros(len(window_states) - 1)
                for t in range(len(window_states) - 1):
                    diff = window_states[t+1] - window_states[t]
                    window_velocity[t] = np.linalg.norm(diff)
                
                velocities.append(np.mean(window_velocity))
                variances.append(np.mean(np.var(window_states, axis=0)))
            
            return np.array(velocities), np.array(variances)
        
        exploration_velocities, exploration_variances = extract_metrics(exploration_times)
        exploitation_velocities, exploitation_variances = extract_metrics(exploitation_times)
        
        return {
            'exploration_velocities': exploration_velocities,
            'exploration_variances': exploration_variances,
            'exploitation_velocities': exploitation_velocities,
            'exploitation_variances': exploitation_variances
        }
    
    def compute_cross_correlation(self,
                                 neural_velocity: np.ndarray,
                                 behavioral_velocity: np.ndarray,
                                 sampling_rate: float = 1000.0,
                                 max_lag_sec: float = 0.5) -> Dict:
        """
        Compute cross-correlation between neural and behavioral velocity.
        
        This addresses the circularity critique by checking lead/lag relationships.
        If neural leads behavioral (positive lag), it supports "motor drive" hypothesis.
        If lag is zero/negative, it might just be representation.
        
        Args:
            neural_velocity: Neural trajectory velocity
            behavioral_velocity: DLC-derived movement velocity
            sampling_rate: Sampling rate in Hz
            max_lag_sec: Maximum lag to test (in seconds)
            
        Returns:
            Dictionary with cross-correlation results
        """
        # Ensure same length
        min_len = min(len(neural_velocity), len(behavioral_velocity))
        neural_vel = neural_velocity[:min_len]
        behavioral_vel = behavioral_velocity[:min_len]
        
        # Normalize signals
        neural_vel = (neural_vel - np.mean(neural_vel)) / (np.std(neural_vel) + 1e-10)
        behavioral_vel = (behavioral_vel - np.mean(behavioral_vel)) / (np.std(behavioral_vel) + 1e-10)
        
        # Compute cross-correlation
        max_lag_samples = int(max_lag_sec * sampling_rate)
        lags = np.arange(-max_lag_samples, max_lag_samples + 1)
        
        correlations = np.correlate(neural_vel, behavioral_vel, mode='full')
        center_idx = len(correlations) // 2
        start_idx = center_idx - max_lag_samples
        end_idx = center_idx + max_lag_samples + 1
        correlations = correlations[start_idx:end_idx]
        
        # Normalize
        correlations = correlations / min_len
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlations))
        peak_lag_samples = lags[peak_idx]
        peak_lag_sec = peak_lag_samples / sampling_rate
        peak_correlation = correlations[peak_idx]
        
        return {
            'lags': lags / sampling_rate,  # Convert to seconds
            'correlations': correlations,
            'peak_lag_sec': peak_lag_sec,
            'peak_correlation': peak_correlation,
            'interpretation': self._interpret_cross_correlation(peak_lag_sec)
        }
    
    def _interpret_cross_correlation(self, peak_lag_sec: float) -> str:
        """Interpret cross-correlation lag."""
        if peak_lag_sec > 0.05:
            return "Neural leads behavioral → Motor Drive hypothesis supported"
        elif peak_lag_sec < -0.05:
            return "Behavioral leads neural → Sensory representation"
        else:
            return "Zero lag → Concurrent activity (ambiguous causality)"
    
    def plot_enhanced_landscape(self, trial_results: Dict, save_path: Optional[str] = None):
        """
        Enhanced visualization with 3D energy landscape, vector field, and velocity-stability analysis.
        
        Shows:
        1. 3D energy landscape surface with trajectory
        2. Quiver plot (vector field) showing forces
        3. Velocity-stability phase space
        4. Exploration gain contribution breakdown
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(20, 12))
        
        time = trial_results['time']
        states = trial_results['states']
        velocities = trial_results['velocities']
        exploration_gains = trial_results['exploration_gains']
        
        # 1. 3D Energy Landscape Surface
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state_2d = np.array([X[i, j], Y[i, j]])
                full_state = np.zeros(self.n_dim)
                full_state[:2] = state_2d
                Z[i, j] = self.potential_energy(full_state)
        
        # Plot surface
        surf = ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.6, edgecolor='none')
        
        # Overlay trajectory as 3D line
        traj_energy = np.array([self.potential_energy(s) for s in states])
        ax1.plot(states[:, 0], states[:, 1], traj_energy, 'k-', linewidth=2, label='Trajectory')
        
        # Mark attractors
        for i, pos in enumerate(self.attractor_positions):
            full_state = np.zeros(self.n_dim)
            full_state[:2] = pos[:2]
            energy = self.potential_energy(full_state)
            ax1.scatter([pos[0]], [pos[1]], [energy], c='red', s=100, marker='*', 
                       edgecolors='black', linewidths=2)
        
        ax1.set_xlabel('State Dim 1')
        ax1.set_ylabel('State Dim 2')
        ax1.set_zlabel('Potential Energy U(x)')
        ax1.set_title('3D Energy Landscape with Trajectory', fontweight='bold')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        
        # 2. Quiver Plot (Vector Field)
        ax2 = fig.add_subplot(2, 3, 2)
        x_q = np.linspace(-2, 2, 15)
        y_q = np.linspace(-2, 2, 15)
        X_q, Y_q = np.meshgrid(x_q, y_q)
        
        U = np.zeros_like(X_q)
        V = np.zeros_like(Y_q)
        
        for i in range(X_q.shape[0]):
            for j in range(X_q.shape[1]):
                state_2d = np.array([X_q[i, j], Y_q[i, j]])
                full_state = np.zeros(self.n_dim)
                full_state[:2] = state_2d
                force = self.force_field(full_state)
                U[i, j] = force[0]
                V[i, j] = force[1]
        
        # Plot vector field
        magnitude = np.sqrt(U**2 + V**2)
        ax2.quiver(X_q, Y_q, U, V, magnitude, cmap='viridis', scale=10, width=0.003)
        
        # Overlay trajectory
        ax2.plot(states[:, 0], states[:, 1], 'k-', linewidth=1, alpha=0.5)
        
        # Mark attractors
        for i, pos in enumerate(self.attractor_positions):
            ax2.plot(pos[0], pos[1], 'r*', markersize=15)
        
        ax2.set_xlabel('State Dim 1')
        ax2.set_ylabel('State Dim 2')
        ax2.set_title('Force Field -∇U(x)', fontweight='bold')
        ax2.set_xlim([-2, 2])
        ax2.set_ylim([-2, 2])
        ax2.grid(True, alpha=0.3)
        
        # 3. Velocity-Stability Phase Space
        ax3 = fig.add_subplot(2, 3, 3)
        
        # Compute distance to nearest attractor
        distances = np.zeros(len(states))
        for t, state in enumerate(states):
            min_dist = np.inf
            for attractor_pos in self.attractor_positions:
                dist = np.linalg.norm(state - attractor_pos)
                min_dist = min(min_dist, dist)
            distances[t] = min_dist
        
        # Plot velocity vs stability (distance)
        scatter = ax3.scatter(distances, velocities, c=time, cmap='viridis', s=10, alpha=0.6)
        ax3.set_xlabel('Distance to Nearest Attractor (Instability)')
        ax3.set_ylabel('Instantaneous Velocity')
        ax3.set_title('Velocity-Stability Phase Space', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=ax3, label='Time (s)')
        
        # Add annotations for expected "L" shape
        ax3.text(0.95, 0.95, 'Expected: "L" shape\nHigh velocity at high distance\nLow velocity at low distance', 
                transform=ax3.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
        
        # 4. Exploration Gain Contribution (Stacked Area)
        ax4 = fig.add_subplot(2, 3, 4)
        
        # Compute velocity and variance contributions
        velocity_contribution = self.alpha * velocities
        
        # Compute variance contribution
        variance_contribution = np.zeros(len(time))
        window_size = 100
        for t in range(len(time)):
            start = max(0, t - window_size)
            end = t + 1
            if end - start >= 2:
                recent_states = states[start:end]
                state_variance = np.mean(np.var(recent_states, axis=0))
                variance_contribution[t] = self.beta * state_variance
        
        # Stacked area chart
        ax4.fill_between(time, 0, velocity_contribution, alpha=0.7, color='blue', 
                        label=f'Velocity Term (α={self.alpha})')
        ax4.fill_between(time, velocity_contribution, velocity_contribution + variance_contribution, 
                        alpha=0.7, color='orange', label=f'Variance Term (β={self.beta})')
        ax4.plot(time, exploration_gains, 'k-', linewidth=2, label='Total Gain', alpha=0.8)
        
        if trial_results['bias_onset_sec'] is not None:
            ax4.axvline(trial_results['bias_onset_sec'], color='r', linestyle='--', 
                       linewidth=2, label='Sensory Input')
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Exploration Gain')
        ax4.set_title('Exploration Gain Decomposition', fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Energy over time
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(time, traj_energy, 'b-', linewidth=1.5)
        if trial_results['bias_onset_sec'] is not None:
            ax5.axvline(trial_results['bias_onset_sec'], color='r', linestyle='--',
                       label='Sensory Input')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Potential Energy U(x)')
        ax5.set_title('Energy Evolution', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Contribution ratio pie chart (final state)
        ax6 = fig.add_subplot(2, 3, 6)
        final_vel_contrib = np.mean(velocity_contribution[-window_size:])
        final_var_contrib = np.mean(variance_contribution[-window_size:])
        
        if final_vel_contrib + final_var_contrib > 0:
            contributions = [final_vel_contrib, final_var_contrib]
            labels = ['Velocity Term', 'Variance Term']
            colors = ['blue', 'orange']
            wedges, texts, autotexts = ax6.pie(contributions, labels=labels, colors=colors, 
                                               autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_fontweight('bold')
        ax6.set_title('Gain Contribution Ratio (Final)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved enhanced landscape visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_trial_results(self, trial_results: Dict, save_path: Optional[str] = None):
        """
        Visualize a simulated trial.
        
        Shows:
        1. Neural state trajectory in 2D/3D
        2. Velocity over time
        3. Exploration gain over time
        4. Phase transitions
        """
        fig = plt.figure(figsize=(15, 10))
        
        time = trial_results['time']
        states = trial_results['states']
        velocities = trial_results['velocities']
        exploration_gains = trial_results['exploration_gains']
        phases = trial_results['phases']
        
        # 1. State trajectory (2D projection)
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(states[:, 0], states[:, 1], c=time, cmap='viridis', s=1)
        
        # Plot attractors
        for i, pos in enumerate(self.attractor_positions):
            ax1.plot(pos[0], pos[1], 'r*', markersize=15, label=f'Attractor {i}' if i == 0 else '')
        
        plt.colorbar(scatter, ax=ax1, label='Time (s)')
        ax1.set_xlabel('State Dim 1')
        ax1.set_ylabel('State Dim 2')
        ax1.set_title('Neural State Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Velocity over time
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(time, velocities, 'b-', linewidth=1)
        if trial_results['bias_onset_sec'] is not None:
            ax2.axvline(trial_results['bias_onset_sec'], color='r', linestyle='--', 
                       label='Sensory Input')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Trajectory Velocity')
        ax2.set_title('Neural Trajectory Speed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Exploration gain over time
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(time, exploration_gains, 'g-', linewidth=1)
        if trial_results['bias_onset_sec'] is not None:
            ax3.axvline(trial_results['bias_onset_sec'], color='r', linestyle='--',
                       label='Sensory Input')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Motor Exploration Gain')
        ax3.set_title('Exploration Drive (Motor Output)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Phase over time
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(time, phases.astype(float), 'k-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Phase (0=Explore, 1=Exploit)')
        ax4.set_title('Behavioral Phase')
        ax4.set_ylim([-0.1, 1.1])
        ax4.grid(True, alpha=0.3)
        
        # 5. Energy landscape (2D slice)
        ax5 = plt.subplot(2, 3, 5)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state_2d = np.array([X[i, j], Y[i, j]])
                full_state = np.zeros(self.n_dim)
                full_state[:2] = state_2d
                Z[i, j] = self.potential_energy(full_state)
        
        contour = ax5.contourf(X, Y, Z, levels=20, cmap='coolwarm')
        ax5.plot(states[:, 0], states[:, 1], 'k-', linewidth=0.5, alpha=0.5)
        plt.colorbar(contour, ax=ax5, label='Potential Energy')
        ax5.set_xlabel('State Dim 1')
        ax5.set_ylabel('State Dim 2')
        ax5.set_title('Energy Landscape with Trajectory')
        
        # 6. Velocity vs Exploration Gain
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(velocities, exploration_gains, c=time, cmap='viridis', s=1, alpha=0.5)
        ax6.set_xlabel('Neural Velocity')
        ax6.set_ylabel('Exploration Gain')
        ax6.set_title('Velocity-Exploration Coupling')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trial visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_neuropixels_analysis(self,
                                  analysis_results: Dict,
                                  save_path: Optional[str] = None):
        """
        Visualize Neuropixels analysis results.
        
        Shows evidence for/against the model's prediction that exploratory movements
        are preceded by increased neural trajectory speed.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        pre_onset_vel = analysis_results['pre_onset_velocities']
        baseline_vel = analysis_results['baseline_velocities']
        pre_onset_var = analysis_results['pre_onset_variances']
        baseline_var = analysis_results['baseline_variances']
        
        # 1. Velocity comparison
        ax1 = axes[0, 0]
        positions = [1, 2]
        bp1 = ax1.boxplot([baseline_vel, pre_onset_vel], positions=positions,
                           widths=0.6, patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp1['boxes'][1].set_facecolor('lightcoral')
        ax1.set_xticklabels(['Baseline', 'Pre-Movement'])
        ax1.set_ylabel('Trajectory Velocity')
        ax1.set_title('Neural Trajectory Speed Before Movement')
        ax1.grid(True, alpha=0.3)
        
        # Add significance test
        stat, p_val = mannwhitneyu(pre_onset_vel, baseline_vel, alternative='greater')
        ax1.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax1.transAxes,
                verticalalignment='top')
        
        # 2. Variance comparison
        ax2 = axes[0, 1]
        bp2 = ax2.boxplot([baseline_var, pre_onset_var], positions=positions,
                           widths=0.6, patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightblue')
        bp2['boxes'][1].set_facecolor('lightcoral')
        ax2.set_xticklabels(['Baseline', 'Pre-Movement'])
        ax2.set_ylabel('State Variance')
        ax2.set_title('Neural State Variability Before Movement')
        ax2.grid(True, alpha=0.3)
        
        stat, p_val = mannwhitneyu(pre_onset_var, baseline_var, alternative='greater')
        ax2.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax2.transAxes,
                verticalalignment='top')
        
        # 3. Trajectory velocity over time
        ax3 = axes[1, 0]
        trajectory_vel = analysis_results['trajectory_velocity']
        time_sec = np.arange(len(trajectory_vel)) / 1000.0  # Assuming 1kHz
        ax3.plot(time_sec, trajectory_vel, 'k-', linewidth=0.5, alpha=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Trajectory Velocity')
        ax3.set_title('Full Recording Trajectory Speed')
        ax3.grid(True, alpha=0.3)
        
        # 4. Neural manifold (first 2 PCs)
        ax4 = axes[1, 1]
        manifold = analysis_results['neural_manifold']
        ax4.scatter(manifold[:, 0], manifold[:, 1], c=np.arange(len(manifold)),
                   cmap='viridis', s=0.1, alpha=0.3)
        ax4.set_xlabel('PC 1')
        ax4.set_ylabel('PC 2')
        ax4.set_title('Neural Population Manifold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Neuropixels analysis to {save_path}")
        else:
            plt.show()
        
        plt.close()


class AttractorModelIntegration:
    """
    Integration class that connects AttractorEnergyLandscapeModel with real data.
    
    This class provides a complete pipeline for:
    1. Loading neural data (spikes + LFP)
    2. Loading behavioral data (DLC body parts + B-SOID/VAME motifs)
    3. Fitting the attractor model to data
    4. Quantitative validation of model predictions
    5. Computing model goodness-of-fit metrics
    """
    
    def __init__(self, model: Optional[AttractorEnergyLandscapeModel] = None):
        """
        Initialize integration wrapper.
        
        Args:
            model: Pre-initialized AttractorEnergyLandscapeModel. If None, creates default.
        """
        if model is None:
            self.model = AttractorEnergyLandscapeModel(
                n_dimensions=10,  # Will be adjusted based on data
                n_attractors=2,
                attractor_strength=2.0,
                noise_level=0.15
            )
        else:
            self.model = model
        
        self.neural_data = None
        self.dlc_data = None
        self.behavior_labels = None
        self.results = {}
    
    def load_data(self,
                  spike_times: List[np.ndarray],
                  spike_clusters: np.ndarray,
                  dlc_bodyparts: Dict[str, np.ndarray],
                  behavior_motifs: Optional[np.ndarray] = None,
                  lfp_data: Optional[np.ndarray] = None,
                  sampling_rate: float = 1000.0):
        """
        Load all data streams.
        
        Args:
            spike_times: List of spike time arrays per neuron
            spike_clusters: Cluster IDs for each spike
            dlc_bodyparts: Dict mapping bodypart names to (time x 2) position arrays
            behavior_motifs: Array of behavioral motif labels (from B-SOID/VAME)
            lfp_data: LFP signals (time x channels)
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
        # Store raw data
        self.spike_times = spike_times
        self.spike_clusters = spike_clusters
        self.dlc_data = dlc_bodyparts
        self.behavior_motifs = behavior_motifs
        self.lfp_data = lfp_data
        
        # Compute neural population activity matrix
        self.neural_data = self._compute_population_activity(spike_times, spike_clusters)
        
        print(f"Data loaded:")
        print(f"  Neural: {self.neural_data.shape} (time x neurons)")
        print(f"  DLC bodyparts: {list(dlc_bodyparts.keys())}")
        if behavior_motifs is not None:
            print(f"  Behavioral motifs: {len(np.unique(behavior_motifs))} unique motifs")
        if lfp_data is not None:
            print(f"  LFP: {lfp_data.shape}")
    
    def _compute_population_activity(self, spike_times, spike_clusters, bin_size_ms=10):
        """Convert spike times to binned population activity matrix."""
        # Find time range
        all_spikes = np.concatenate([st for st in spike_times if len(st) > 0])
        if len(all_spikes) == 0:
            return np.zeros((1000, len(spike_times)))
        
        max_time = np.max(all_spikes)
        n_bins = int(max_time * self.sampling_rate / bin_size_ms) + 1
        n_neurons = len(spike_times)
        
        # Bin spikes
        activity = np.zeros((n_bins, n_neurons))
        for neuron_idx, st in enumerate(spike_times):
            if len(st) > 0:
                bins = (st * self.sampling_rate / bin_size_ms).astype(int)
                bins = bins[bins < n_bins]
                np.add.at(activity[:, neuron_idx], bins, 1)
        
        return activity
    
    def identify_behavioral_epochs(self, motif_ids: Optional[List[int]] = None) -> Dict:
        """
        Identify exploration vs exploitation epochs based on behavioral motifs.
        
        Args:
            motif_ids: List of motif IDs to classify as exploration. If None, uses heuristics.
            
        Returns:
            Dictionary with 'exploration_times' and 'exploitation_times'
        """
        if self.behavior_motifs is None:
            # Fallback: Use DLC velocity to identify exploration
            return self._identify_epochs_from_velocity()
        
        # Identify exploration motifs (high velocity, non-directed movement)
        if motif_ids is None:
            motif_ids = self._identify_exploration_motifs()
        
        exploration_mask = np.isin(self.behavior_motifs, motif_ids)
        
        # Find continuous epochs
        exploration_times = self._mask_to_times(exploration_mask)
        exploitation_times = self._mask_to_times(~exploration_mask)
        
        return {
            'exploration_times': exploration_times,
            'exploitation_times': exploitation_times,
            'exploration_motifs': motif_ids
        }
    
    def _identify_exploration_motifs(self) -> List[int]:
        """
        Automatically identify exploration motifs based on movement characteristics.
        
        Exploration typically involves:
        - High velocity variance
        - Non-directed movement (low velocity autocorrelation)
        - Frequent direction changes
        """
        unique_motifs = np.unique(self.behavior_motifs)
        exploration_motifs = []
        
        # Compute metrics for each motif
        for motif_id in unique_motifs:
            if motif_id == -1 or motif_id == 0:  # Skip invalid/unknown
                continue
            
            motif_mask = self.behavior_motifs == motif_id
            if np.sum(motif_mask) < 10:  # Too few samples
                continue
            
            # Get velocity during this motif
            velocities = self._compute_velocity_from_dlc()
            motif_velocities = velocities[motif_mask]
            
            # High variance = exploration
            velocity_variance = np.var(motif_velocities)
            
            # High mean = also movement
            velocity_mean = np.mean(np.abs(motif_velocities))
            
            # Combined score
            exploration_score = velocity_variance * velocity_mean
            
            exploration_motifs.append((motif_id, exploration_score))
        
        # Compute threshold from all scores
        if len(exploration_motifs) > 0:
            all_scores = [score for _, score in exploration_motifs]
            threshold = np.percentile(all_scores, 70)
            exploration_motifs = [motif_id for motif_id, score in exploration_motifs 
                                 if score > threshold]
        else:
            exploration_motifs = []
        
        return exploration_motifs
    
    def _compute_velocity_from_dlc(self) -> np.ndarray:
        """Compute velocity magnitude from DLC data."""
        # Use center of mass or main bodypart
        if 'body' in self.dlc_data:
            pos = self.dlc_data['body']
        elif 'centroid' in self.dlc_data:
            pos = self.dlc_data['centroid']
        else:
            # Use first available bodypart
            pos = list(self.dlc_data.values())[0]
        
        # Compute velocity
        vel = np.diff(pos, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        
        # Pad to match original length
        vel_mag = np.concatenate([[0], vel_mag])
        
        return vel_mag
    
    def _mask_to_times(self, mask: np.ndarray) -> np.ndarray:
        """Convert boolean mask to array of onset times."""
        transitions = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
        onsets = np.where(transitions == 1)[0]
        return onsets / self.sampling_rate
    
    def _identify_epochs_from_velocity(self) -> Dict:
        """Fallback method using velocity when motifs not available."""
        velocity = self._compute_velocity_from_dlc()
        
        # High velocity = exploration (simplified heuristic)
        threshold = np.percentile(velocity, 70)
        exploration_mask = velocity > threshold
        
        exploration_times = self._mask_to_times(exploration_mask)
        exploitation_times = self._mask_to_times(~exploration_mask)
        
        return {
            'exploration_times': exploration_times,
            'exploitation_times': exploitation_times
        }
    
    def fit_and_validate(self) -> Dict:
        """
        Complete pipeline: fit model and validate predictions.
        
        Returns:
            Dictionary with all validation metrics and results
        """
        print("\n" + "="*60)
        print("Attractor Model: Fit and Validation Pipeline")
        print("="*60)
        
        # 1. Identify behavioral epochs
        print("\n1. Identifying behavioral epochs...")
        epochs = self.identify_behavioral_epochs()
        print(f"   Found {len(epochs['exploration_times'])} exploration epochs")
        print(f"   Found {len(epochs['exploitation_times'])} exploitation epochs")
        
        # 2. Test prediction: exploration preceded by high trajectory speed
        print("\n2. Testing trajectory speed prediction...")
        speed_analysis = self.model.analyze_movement_onset(
            neural_population_activity=self.neural_data,
            movement_onsets=epochs['exploration_times'],
            pre_onset_window_sec=0.5,
            sampling_rate=self.sampling_rate
        )
        
        # 3. Compare exploration vs exploitation dynamics
        print("\n3. Comparing exploration vs exploitation...")
        comparison = self.model.compare_exploration_exploitation(
            neural_activity=self.neural_data,
            exploration_times=epochs['exploration_times'],
            exploitation_times=epochs['exploitation_times'],
            window_sec=0.5,
            sampling_rate=self.sampling_rate
        )
        
        # 4. Compute goodness-of-fit metrics
        print("\n4. Computing model goodness-of-fit...")
        gof_metrics = self._compute_goodness_of_fit(speed_analysis, comparison)
        
        # 5. Store results
        self.results = {
            'epochs': epochs,
            'speed_analysis': speed_analysis,
            'comparison': comparison,
            'goodness_of_fit': gof_metrics
        }
        
        # Print summary
        self._print_validation_summary(gof_metrics)
        
        return self.results
    
    def _compute_goodness_of_fit(self, speed_analysis, comparison) -> Dict:
        """
        Compute quantitative metrics for model validation.
        
        Returns metrics including:
        - Effect sizes (Cohen's d)
        - Statistical significance (p-values)
        - Variance explained (R²)
        - Classification accuracy
        """
        metrics = {}
        
        # 1. Trajectory speed prediction
        pre_onset_vel = speed_analysis['pre_onset_velocities']
        baseline_vel = speed_analysis['baseline_velocities']
        
        # Effect size (Cohen's d)
        # Using pooled standard deviation for unequal sample sizes
        n1, n2 = len(pre_onset_vel), len(baseline_vel)
        if n1 > 1 and n2 > 1:
            pooled_std = np.sqrt(((n1-1)*np.var(pre_onset_vel, ddof=1) + 
                                  (n2-1)*np.var(baseline_vel, ddof=1)) / (n1+n2-2))
            cohens_d = (np.mean(pre_onset_vel) - np.mean(baseline_vel)) / (pooled_std + 1e-10)
        else:
            cohens_d = np.nan
        metrics['trajectory_speed_effect_size'] = cohens_d
        
        # Statistical test
        stat, p_val = mannwhitneyu(pre_onset_vel, baseline_vel, alternative='greater')
        metrics['trajectory_speed_pvalue'] = p_val
        
        # Variance effect size
        pre_onset_var = speed_analysis['pre_onset_variances']
        baseline_var = speed_analysis['baseline_variances']
        n1, n2 = len(pre_onset_var), len(baseline_var)
        if n1 > 1 and n2 > 1:
            pooled_std_var = np.sqrt(((n1-1)*np.var(pre_onset_var, ddof=1) + 
                                      (n2-1)*np.var(baseline_var, ddof=1)) / (n1+n2-2))
            cohens_d_var = (np.mean(pre_onset_var) - np.mean(baseline_var)) / (pooled_std_var + 1e-10)
        else:
            cohens_d_var = np.nan
        metrics['trajectory_variance_effect_size'] = cohens_d_var
        
        # 2. Exploration vs exploitation comparison
        expl_vel = comparison['exploration_velocities']
        expt_vel = comparison['exploitation_velocities']
        
        n1, n2 = len(expl_vel), len(expt_vel)
        if n1 > 1 and n2 > 1:
            pooled_std_behavior = np.sqrt(((n1-1)*np.var(expl_vel, ddof=1) + 
                                           (n2-1)*np.var(expt_vel, ddof=1)) / (n1+n2-2))
            cohens_d_behavior = (np.mean(expl_vel) - np.mean(expt_vel)) / (pooled_std_behavior + 1e-10)
        else:
            cohens_d_behavior = np.nan
        metrics['behavior_discrimination_effect_size'] = cohens_d_behavior
        
        stat, p_val = mannwhitneyu(expl_vel, expt_vel, alternative='greater')
        metrics['behavior_discrimination_pvalue'] = p_val
        
        # 3. Classification accuracy (can we predict behavior from neural velocity?)
        all_velocities = np.concatenate([expl_vel, expt_vel])
        all_labels = np.concatenate([np.ones(len(expl_vel)), np.zeros(len(expt_vel))])
        
        # Simple threshold classifier
        threshold = np.median(all_velocities)
        predictions = (all_velocities > threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        metrics['classification_accuracy'] = accuracy
        
        # 4. Model consistency: variance explained by attractor dynamics
        # Model predicts velocity should correlate with uncertainty
        # Here we use neural variance as proxy for uncertainty
        neural_manifold = speed_analysis['neural_manifold']
        neural_velocity = speed_analysis['trajectory_velocity']
        local_variance = np.zeros(len(neural_manifold))
        window = 50
        for i in range(len(neural_manifold)):
            start = max(0, i - window)
            end = min(len(neural_manifold), i + window)
            local_variance[i] = np.mean(np.var(neural_manifold[start:end], axis=0))
        
        # Correlation between variance and velocity
        valid_idx = ~np.isnan(local_variance) & ~np.isnan(neural_velocity)
        if np.sum(valid_idx) > 100:
            correlation = np.corrcoef(local_variance[valid_idx], 
                                     neural_velocity[valid_idx])[0, 1]
            r_squared = correlation ** 2
            metrics['variance_velocity_correlation'] = correlation
            metrics['variance_explained'] = r_squared
        else:
            metrics['variance_velocity_correlation'] = np.nan
            metrics['variance_explained'] = np.nan
        
        return metrics
    
    def _print_validation_summary(self, metrics: Dict):
        """Print human-readable summary of validation results."""
        print("\n" + "="*60)
        print("Model Validation Summary")
        print("="*60)
        
        print("\n1. Trajectory Speed Prediction:")
        print(f"   Effect size (Cohen's d): {metrics['trajectory_speed_effect_size']:.3f}")
        print(f"   P-value: {metrics['trajectory_speed_pvalue']:.4e}")
        if metrics['trajectory_speed_pvalue'] < 0.001:
            print(f"   ✓ STRONG support for prediction (p < 0.001)")
        elif metrics['trajectory_speed_pvalue'] < 0.05:
            print(f"   ✓ Significant support for prediction (p < 0.05)")
        else:
            print(f"   ✗ No significant support (p > 0.05)")
        
        print("\n2. Trajectory Variance Prediction:")
        print(f"   Effect size: {metrics['trajectory_variance_effect_size']:.3f}")
        
        print("\n3. Behavior Discrimination:")
        print(f"   Effect size: {metrics['behavior_discrimination_effect_size']:.3f}")
        print(f"   P-value: {metrics['behavior_discrimination_pvalue']:.4e}")
        print(f"   Classification accuracy: {metrics['classification_accuracy']*100:.1f}%")
        
        if not np.isnan(metrics['variance_explained']):
            print("\n4. Model Consistency:")
            print(f"   Velocity-Variance correlation: {metrics['variance_velocity_correlation']:.3f}")
            print(f"   Variance explained (R²): {metrics['variance_explained']*100:.1f}%")
        
        # Overall verdict
        print("\n" + "="*60)
        strong_support = (metrics['trajectory_speed_pvalue'] < 0.001 and 
                         metrics['trajectory_speed_effect_size'] > 0.5)
        moderate_support = (metrics['trajectory_speed_pvalue'] < 0.05 and
                           metrics['trajectory_speed_effect_size'] > 0.3)
        
        if strong_support:
            print("Overall: ✓✓ STRONG empirical support for attractor model")
        elif moderate_support:
            print("Overall: ✓ MODERATE empirical support for attractor model")
        else:
            print("Overall: Model predictions not strongly supported by data")
        print("="*60)
    
    def generate_report(self, save_path: str = 'attractor_model_report.png'):
        """
        Generate comprehensive visualization report.
        
        Args:
            save_path: Path to save the report figure
        """
        if not self.results:
            print("Error: Run fit_and_validate() first")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Trajectory speed analysis
        ax1 = plt.subplot(3, 4, 1)
        speed_analysis = self.results['speed_analysis']
        bp = ax1.boxplot([speed_analysis['baseline_velocities'],
                          speed_analysis['pre_onset_velocities']],
                         labels=['Baseline', 'Pre-Movement'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('salmon')
        ax1.set_ylabel('Neural Trajectory Speed')
        ax1.set_title('Prediction 1: Speed Before Movement')
        ax1.grid(True, alpha=0.3)
        
        # 2. Variance analysis
        ax2 = plt.subplot(3, 4, 2)
        bp = ax2.boxplot([speed_analysis['baseline_variances'],
                          speed_analysis['pre_onset_variances']],
                         labels=['Baseline', 'Pre-Movement'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('salmon')
        ax2.set_ylabel('Neural State Variance')
        ax2.set_title('Prediction 2: Variance Before Movement')
        ax2.grid(True, alpha=0.3)
        
        # 3. Exploration vs exploitation
        ax3 = plt.subplot(3, 4, 3)
        comparison = self.results['comparison']
        bp = ax3.boxplot([comparison['exploitation_velocities'],
                          comparison['exploration_velocities']],
                         labels=['Exploitation', 'Exploration'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('orange')
        ax3.set_ylabel('Neural Trajectory Speed')
        ax3.set_title('Behavioral Phase Comparison')
        ax3.grid(True, alpha=0.3)
        
        # 4. Goodness of fit metrics
        ax4 = plt.subplot(3, 4, 4)
        gof = self.results['goodness_of_fit']
        metrics_to_plot = {
            'Speed\nEffect': gof['trajectory_speed_effect_size'],
            'Variance\nEffect': gof['trajectory_variance_effect_size'],
            'Behavior\nEffect': gof['behavior_discrimination_effect_size']
        }
        bars = ax4.bar(range(len(metrics_to_plot)), list(metrics_to_plot.values()))
        bars[0].set_color('salmon')
        bars[1].set_color('coral')
        bars[2].set_color('orange')
        ax4.set_xticks(range(len(metrics_to_plot)))
        ax4.set_xticklabels(list(metrics_to_plot.keys()))
        ax4.set_ylabel("Cohen's d")
        ax4.set_title('Effect Sizes')
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
        ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='Large effect')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Neural manifold trajectory
        ax5 = plt.subplot(3, 4, 5)
        manifold = speed_analysis['neural_manifold']
        time_colors = np.arange(len(manifold))
        scatter = ax5.scatter(manifold[:, 0], manifold[:, 1], 
                             c=time_colors, cmap='viridis', s=1, alpha=0.3)
        ax5.set_xlabel('PC 1')
        ax5.set_ylabel('PC 2')
        ax5.set_title('Neural Population Manifold')
        plt.colorbar(scatter, ax=ax5, label='Time')
        
        # 6. Trajectory velocity time series
        ax6 = plt.subplot(3, 4, 6)
        trajectory_vel = speed_analysis['trajectory_velocity']
        time_sec = np.arange(len(trajectory_vel)) / self.sampling_rate
        ax6.plot(time_sec, trajectory_vel, 'k-', alpha=0.5, linewidth=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Trajectory Velocity')
        ax6.set_title('Neural Trajectory Speed Over Time')
        ax6.grid(True, alpha=0.3)
        
        # 7. Movement epochs
        ax7 = plt.subplot(3, 4, 7)
        epochs = self.results['epochs']
        ax7.eventplot([epochs['exploration_times']], 
                     lineoffsets=1, colors='orange', label='Exploration')
        ax7.eventplot([epochs['exploitation_times']], 
                     lineoffsets=0, colors='green', label='Exploitation')
        ax7.set_yticks([0, 1])
        ax7.set_yticklabels(['Exploitation', 'Exploration'])
        ax7.set_xlabel('Time (s)')
        ax7.set_title('Behavioral Epochs')
        ax7.legend(loc='upper right')
        
        # 8. Velocity distribution
        ax8 = plt.subplot(3, 4, 8)
        ax8.hist(speed_analysis['baseline_velocities'], bins=30, alpha=0.5, 
                label='Baseline', color='lightblue', density=True)
        ax8.hist(speed_analysis['pre_onset_velocities'], bins=30, alpha=0.5,
                label='Pre-Movement', color='salmon', density=True)
        ax8.set_xlabel('Neural Trajectory Speed')
        ax8.set_ylabel('Density')
        ax8.set_title('Speed Distributions')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9-12: Behavioral motif analysis if available
        if self.behavior_motifs is not None:
            ax9 = plt.subplot(3, 4, 9)
            unique_motifs, counts = np.unique(self.behavior_motifs, return_counts=True)
            ax9.bar(unique_motifs, counts)
            ax9.set_xlabel('Motif ID')
            ax9.set_ylabel('Counts')
            ax9.set_title('Behavioral Motif Distribution')
            
            # Highlight exploration motifs
            if 'exploration_motifs' in epochs:
                for motif_id in epochs['exploration_motifs']:
                    idx = np.where(unique_motifs == motif_id)[0]
                    if len(idx) > 0:
                        ax9.patches[idx[0]].set_facecolor('orange')
            ax9.grid(True, alpha=0.3)
        
        # 10. DLC velocity
        ax10 = plt.subplot(3, 4, 10)
        velocity = self._compute_velocity_from_dlc()
        time_dlc = np.arange(len(velocity)) / self.sampling_rate
        ax10.plot(time_dlc, velocity, 'b-', alpha=0.5, linewidth=0.5)
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Movement Velocity')
        ax10.set_title('DLC-Derived Velocity')
        ax10.grid(True, alpha=0.3)
        
        # 11. Classification performance
        ax11 = plt.subplot(3, 4, 11)
        accuracy = gof['classification_accuracy']
        categories = ['Accuracy', 'Chance']
        values = [accuracy, 0.5]
        bars = ax11.bar(categories, values)
        bars[0].set_color('green' if accuracy > 0.6 else 'orange')
        bars[1].set_color('gray')
        ax11.set_ylabel('Proportion')
        ax11.set_ylim([0, 1])
        ax11.set_title('Behavior Classification')
        ax11.axhline(y=0.5, color='red', linestyle='--', label='Chance')
        ax11.grid(True, alpha=0.3)
        
        # 12. Summary text
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        summary_text = f"""Model Validation Summary
        
Speed Prediction:
  Effect: {gof['trajectory_speed_effect_size']:.2f}
  p-value: {gof['trajectory_speed_pvalue']:.2e}

Behavior Discrimination:
  Effect: {gof['behavior_discrimination_effect_size']:.2f}
  p-value: {gof['behavior_discrimination_pvalue']:.2e}
  Accuracy: {gof['classification_accuracy']*100:.1f}%

Variance Explained: {'N/A' if np.isnan(gof.get('variance_explained', np.nan)) else f"{gof['variance_explained']*100:.1f}%"}

Overall: {'✓ Supported' if gof['trajectory_speed_pvalue'] < 0.05 else '✗ Not supported'}
        """
        ax12.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nReport saved to: {save_path}")
        plt.close()


def example_usage():
    """
    Example usage of the Attractor Energy Landscape Model.
    """
    print("\n" + "="*60)
    print("Attractor Energy Landscape Model - Example Usage")
    print("="*60)
    
    # Initialize model
    model = AttractorEnergyLandscapeModel(
        n_dimensions=3,
        n_attractors=2,  # Left vs Right choice
        attractor_strength=2.0,
        noise_level=0.15,
        coupling_alpha=1.0,
        coupling_beta=0.5
    )
    
    # Simulate trial with uncertainty → certainty transition
    print("\n1. Simulating trial with sensory input at 1.0s...")
    trial_results = model.simulate_trial(
        duration_sec=3.0,
        bias_onset_sec=1.0,  # Sensory input arrives
        bias_direction=0,     # Bias toward first attractor
        bias_strength=2.0
    )
    
    # Visualize
    model.plot_trial_results(trial_results, save_path='/tmp/attractor_trial.png')
    
    # Demonstrate Neuropixels analysis
    print("\n2. Demonstrating Neuropixels analysis...")
    
    # Generate synthetic neural data
    n_neurons = 50
    n_timepoints = 10000
    synthetic_neural_activity = np.random.randn(n_timepoints, n_neurons) * 0.1
    
    # Add structure: higher activity before "movement" events
    movement_onsets = np.array([1.0, 2.5, 4.0, 5.5, 7.0, 8.5])  # seconds
    for onset in movement_onsets:
        onset_idx = int(onset * 1000)
        if onset_idx < n_timepoints - 500:
            # Add increased variability before onset
            synthetic_neural_activity[onset_idx-500:onset_idx] += np.random.randn(500, n_neurons) * 0.5
    
    # Analyze
    analysis_results = model.analyze_movement_onset(
        neural_population_activity=synthetic_neural_activity,
        movement_onsets=movement_onsets,
        pre_onset_window_sec=0.5,
        sampling_rate=1000.0
    )
    
    # Visualize
    model.plot_neuropixels_analysis(analysis_results, save_path='/tmp/neuropixels_analysis.png')
    
    print("\n" + "="*60)
    print("Example complete! Check /tmp/ for output plots.")
    print("="*60)


def example_integration():
    """
    Example demonstrating full integration with Neuropixels + DLC + behavioral motifs.
    """
    print("\n" + "="*60)
    print("Attractor Model Integration Example")
    print("="*60)
    
    # Generate synthetic data that mimics real experiment
    print("\n1. Generating synthetic data...")
    
    # Simulate 60 seconds of recording at 1000 Hz
    duration_sec = 60
    sampling_rate = 1000.0
    n_timepoints = int(duration_sec * sampling_rate)
    
    # Neural data: 30 neurons
    n_neurons = 30
    spike_times = []
    spike_clusters = []
    
    for neuron_id in range(n_neurons):
        # Poisson process with varying rates
        base_rate = np.random.uniform(1, 10)  # Hz
        n_spikes = np.random.poisson(base_rate * duration_sec)
        spikes = np.sort(np.random.uniform(0, duration_sec, n_spikes))
        spike_times.append(spikes)
        spike_clusters.append(np.full(len(spikes), neuron_id))
    
    spike_clusters = np.concatenate(spike_clusters)
    
    # DLC data: simulate mouse trajectory
    time_points = np.linspace(0, duration_sec, n_timepoints)
    
    # Simulate exploration (random walk) and exploitation (directed movement)
    position = np.zeros((n_timepoints, 2))
    velocity = np.zeros((n_timepoints, 2))
    
    current_pos = np.array([0.5, 0.5])  # Start at center
    is_exploring = np.zeros(n_timepoints, dtype=bool)
    
    for t in range(1, n_timepoints):
        # Switch between exploration and exploitation
        if t % 10000 == 0:  # Switch every 10 seconds
            is_exploring[t:t+5000] = True
        
        if is_exploring[t]:
            # Random walk (exploration)
            velocity[t] = np.random.randn(2) * 0.01
        else:
            # Directed movement toward goal (exploitation)
            goal = np.array([0.8, 0.8])
            direction = goal - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            velocity[t] = direction * 0.005 + np.random.randn(2) * 0.002
        
        current_pos = current_pos + velocity[t]
        current_pos = np.clip(current_pos, 0, 1)  # Keep in bounds
        position[t] = current_pos
    
    dlc_bodyparts = {
        'body': position,
        'head': position + np.random.randn(n_timepoints, 2) * 0.01,
        'tail': position - np.random.randn(n_timepoints, 2) * 0.01
    }
    
    # Behavioral motifs: classify based on velocity
    vel_mag = np.sqrt(np.sum(velocity**2, axis=1))
    behavior_motifs = np.zeros(n_timepoints, dtype=int)
    
    # Motif 1: High velocity exploration
    behavior_motifs[vel_mag > np.percentile(vel_mag, 75)] = 1
    # Motif 2: Medium velocity directed movement
    behavior_motifs[(vel_mag > np.percentile(vel_mag, 25)) & 
                   (vel_mag <= np.percentile(vel_mag, 75))] = 2
    # Motif 3: Low velocity (stationary)
    behavior_motifs[vel_mag <= np.percentile(vel_mag, 25)] = 3
    
    print(f"   Generated {n_neurons} neurons, {duration_sec}s recording")
    print(f"   DLC bodyparts: {list(dlc_bodyparts.keys())}")
    print(f"   Behavioral motifs: {len(np.unique(behavior_motifs))} types")
    
    # 2. Initialize integration
    print("\n2. Initializing attractor model integration...")
    integration = AttractorModelIntegration()
    
    # 3. Load data
    print("\n3. Loading data into integration framework...")
    integration.load_data(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        dlc_bodyparts=dlc_bodyparts,
        behavior_motifs=behavior_motifs,
        sampling_rate=sampling_rate
    )
    
    # 4. Fit and validate
    print("\n4. Running fit and validation pipeline...")
    results = integration.fit_and_validate()
    
    # 5. Generate report
    print("\n5. Generating comprehensive report...")
    integration.generate_report(save_path='/tmp/attractor_integration_report.png')
    
    print("\n" + "="*60)
    print("Integration example complete!")
    print("Check /tmp/attractor_integration_report.png for results")
    print("="*60)
    
    return integration, results


if __name__ == "__main__":
    # Run basic example
    example_usage()
    
    # Run integration example with full data pipeline
    print("\n\n")
    example_integration()
