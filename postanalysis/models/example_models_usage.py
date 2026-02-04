"""
Example usage of theoretical models for striatal LFP propagation.

This script demonstrates how to use all models including:
1. Delayed Diffusion Model - LFP propagation with conduction delays
2. Phase Coherence Gating Model - FSI-MSN phase filtering
3. Spike-LFP Coupling Model - Bidirectional spike-LFP interactions
4. Striatal Microcircuit Model - Dopamine-driven resonance tuning
5. Phase Field Model - Continuum neural field dynamics
6. Attractor Energy Landscape Model - Uncertainty-driven exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import models
from postanalysis.models import (
    DelayedDiffusionModel,
    PhaseCoherenceGatingModel,
    SpikeLFPCouplingModel,
    StrialMicrocircuitModel,
    PhaseFieldModel,
    AttractorEnergyLandscapeModel
)


def example_1_delayed_diffusion():
    """
    Example 1: Delayed diffusion with conduction delays.
    
    Demonstrates how conduction delays facilitate resonance at specific frequencies.
    """
    print("\n" + "="*60)
    print("Example 1: Delayed Diffusion Model")
    print("="*60)
    
    # Initialize model
    model = DelayedDiffusionModel(
        spatial_grid_size=(80, 80),
        spatial_resolution_um=10.0,
        diffusion_coefficient=1.0,
        conduction_delay_ms=5.0,
        anisotropy_ratio=2.0  # Faster along shanks
    )
    
    # Generate synthetic spike data
    n_spikes = 500
    spike_times = np.sort(np.random.uniform(0, 5.0, n_spikes))
    spike_positions = np.random.uniform(0, 800, (n_spikes, 2))
    
    # Simulate spike-driven LFP propagation
    results = model.simulate_spike_driven_propagation(
        spike_times=spike_times,
        spike_positions=spike_positions,
        duration_sec=5.0
    )
    
    # Visualize propagation
    output_dir = Path('model_outputs')
    output_dir.mkdir(exist_ok=True)
    model.plot_propagation_snapshots(results, save_path=output_dir / 'delayed_diffusion_propagation.png')
    
    # Optimize delay for theta frequency (8 Hz)
    print("\nOptimizing conduction delay for theta resonance...")
    optimal_delay, powers = model.optimize_delay_for_frequency(
        target_freq_hz=8.0,
        test_delays_ms=np.linspace(2, 15, 10),
        n_steps=500
    )
    
    # Plot delay optimization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.linspace(2, 15, 10), powers, 'o-', linewidth=2, markersize=8)
    ax.axvline(optimal_delay, color='red', linestyle='--', label=f'Optimal: {optimal_delay:.2f}ms')
    ax.set_xlabel('Conduction Delay (ms)')
    ax.set_ylabel('Power at 8 Hz')
    ax.set_title('Delay Optimization for Theta Resonance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'delay_optimization.png', dpi=150)
    print(f"  Saved delay optimization plot")
    
    return results


def example_2_phase_coherence_gating():
    """
    Example 2: Phase-coherence gating with FSI-MSN interactions.
    
    Demonstrates how FSIs gate MSN activity to specific LFP phases.
    """
    print("\n" + "="*60)
    print("Example 2: Phase-Coherence Gating Model")
    print("="*60)
    
    # Initialize model
    model = PhaseCoherenceGatingModel(
        n_msns=100,
        n_fsis=20,
        target_frequency_hz=8.0  # Theta band
    )
    
    # Generate synthetic LFP signal (theta oscillation with noise)
    duration = 10.0  # seconds
    fs = 1000.0  # Hz
    t = np.arange(0, duration, 1/fs)
    lfp_signal = np.sin(2 * np.pi * 8.0 * t) + 0.3 * np.random.randn(len(t))
    
    # Simulate reward times
    reward_times = np.array([2.0, 4.5, 7.0])
    
    # Simulate network
    results = model.simulate_phase_gated_network(
        lfp_signal=lfp_signal,
        reward_times=reward_times,
        reward_window_sec=0.5
    )
    
    # Visualize
    output_dir = Path('model_outputs')
    model.plot_phase_locking_analysis(results, save_path=output_dir / 'phase_gating_analysis.png')
    
    return results


def example_3_spike_lfp_coupling():
    """
    Example 3: Bidirectional spike-LFP coupling.
    
    Demonstrates forward model (spikes -> LFP) and feedback (LFP phase -> spikes).
    """
    print("\n" + "="*60)
    print("Example 3: Spike-LFP Coupling Model")
    print("="*60)
    
    model = SpikeLFPCouplingModel(
        spatial_kernel_um=100.0,
        temporal_kernel_ms=10.0
    )
    
    # Generate synthetic spike data
    n_spikes = 200
    spike_times = np.sort(np.random.uniform(0, 5.0, n_spikes))
    spike_positions = np.random.uniform(0, 1000, (n_spikes, 2))
    
    # Generate LFP from spikes (forward model)
    recording_pos = np.array([500, 500])
    lfp_results = model.generate_lfp_from_spikes(
        spike_times=spike_times,
        spike_positions=spike_positions,
        recording_position=recording_pos,
        duration_sec=5.0
    )
    
    # Plot generated LFP
    output_dir = Path('model_outputs')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lfp_results['time'], lfp_results['lfp_signal'], 'k-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (mV)')
    ax.set_title('LFP Generated from Spike Trains')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'spike_generated_lfp.png', dpi=150)
    print(f"  Saved spike-generated LFP plot")
    
    return lfp_results


def example_4_dopamine_resonance_tuning():
    """
    Example 4: Dopamine-driven resonance tuning in striatal microcircuit.
    
    Demonstrates how dopamine tunes synaptic delays to optimize resonance
    at task-relevant frequencies.
    """
    print("\n" + "="*60)
    print("Example 4: Dopamine-Driven Resonance Tuning")
    print("="*60)
    
    model = StrialMicrocircuitModel(
        n_d1_msns=50,
        n_d2_msns=50,
        n_fsis=20,
        base_synaptic_delay_ms=5.0
    )
    
    # Simulate learning task (CW vs CCW navigation)
    n_trials = 50
    trial_history = []
    
    # Different target frequencies for CW vs CCW
    # CW: theta (8 Hz), CCW: beta (20 Hz)
    target_freqs = []
    dopamine_levels = []
    rewards = []
    
    for trial in range(n_trials):
        # Alternate between CW and CCW
        is_cw = (trial % 2 == 0)
        target_freq = 8.0 if is_cw else 20.0
        target_freqs.append(target_freq)
        
        # Simulate trial outcome (learning improves over time)
        success_prob = min(0.9, 0.4 + trial * 0.01)
        rewarded = np.random.rand() < success_prob
        rewards.append(rewarded)
        
        # Dopamine level depends on reward
        da_level = 1.5 if rewarded else 0.8
        dopamine_levels.append(da_level)
        
        # Run trial
        result = model.simulate_trial(
            duration_sec=2.0,
            input_rate_hz=10.0,
            dopamine_level=da_level
        )
        trial_history.append(result)
        
        # Apply dopamine-driven plasticity
        if rewarded:
            model.apply_dopamine_plasticity(
                dopamine_level=da_level,
                reward_outcome=rewarded,
                target_frequency_hz=target_freq,
                learning_rate=0.02
            )
    
    # Visualize resonance tuning over trials
    output_dir = Path('model_outputs')
    model.plot_resonance_tuning(trial_history, save_path=output_dir / 'resonance_tuning.png')
    
    return trial_history


def example_5_phase_field():
    """
    Example 5: Phase-field model for traveling waves.
    
    Demonstrates continuum neural field dynamics.
    """
    print("\n" + "="*60)
    print("Example 5: Phase-Field Model")
    print("="*60)
    
    model = PhaseFieldModel(
        grid_size=(100, 100),
        spatial_resolution_um=10.0
    )
    
    # Run simulation
    n_steps = 100
    snapshots = []
    
    for step in range(n_steps):
        u, phi = model.step(coupling_strength=0.1)
        if step % 10 == 0:
            snapshots.append((u.copy(), phi.copy()))
    
    # Visualize
    output_dir = Path('model_outputs')
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for idx, (u, phi) in enumerate(snapshots):
        ax = axes[0, idx]
        im = ax.imshow(u.T, origin='lower', cmap='viridis')
        ax.set_title(f'Activity t={idx*10}')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1, idx]
        im = ax.imshow(phi.T, origin='lower', cmap='hsv', vmin=0, vmax=2*np.pi)
        ax.set_title(f'Phase t={idx*10}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_field_dynamics.png', dpi=150)
    print(f"  Saved phase-field dynamics plot")
    
    return snapshots


def example_6_attractor_energy_landscape():
    """
    Example 6: Attractor energy landscape model.
    
    Demonstrates uncertainty-driven exploration through energy spillover.
    """
    print("\n" + "="*60)
    print("Example 6: Attractor Energy Landscape Model")
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
    print("\n  Simulating trial with sensory input at 1.0s...")
    trial_results = model.simulate_trial(
        duration_sec=3.0,
        bias_onset_sec=1.0,  # Sensory input arrives
        bias_direction=0,     # Bias toward first attractor
        bias_strength=2.0
    )
    
    # Visualize
    output_dir = Path('model_outputs')
    output_dir.mkdir(exist_ok=True)
    model.plot_trial_results(trial_results, 
                            save_path=str(output_dir / 'attractor_trial.png'))
    print(f"  Saved trial visualization")
    
    # Key findings
    exploration_gain_before = np.mean(trial_results['exploration_gains'][:1000])
    exploration_gain_after = np.mean(trial_results['exploration_gains'][2000:])
    velocity_before = np.mean(trial_results['velocities'][:1000])
    velocity_after = np.mean(trial_results['velocities'][2000:])
    
    print(f"\n  Key Results:")
    print(f"    Pre-sensory input:")
    print(f"      Exploration gain: {exploration_gain_before:.3f}")
    print(f"      Neural velocity: {velocity_before:.3f}")
    print(f"    Post-sensory input:")
    print(f"      Exploration gain: {exploration_gain_after:.3f}")
    print(f"      Neural velocity: {velocity_after:.3f}")
    print(f"    Reduction: {(1 - exploration_gain_after/exploration_gain_before)*100:.1f}%")
    
    return trial_results


def main():
    """
    Run all examples demonstrating the theoretical models.
    """
    print("="*60)
    print("Theoretical Models for Striatal LFP Propagation")
    print("Chapter 2: Enhanced with Pouzzner's Theory")
    print("="*60)
    
    # Run examples
    example_1_delayed_diffusion()
    example_2_phase_coherence_gating()
    example_3_spike_lfp_coupling()
    example_4_dopamine_resonance_tuning()
    example_5_phase_field()
    example_6_attractor_energy_landscape()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("Results saved to model_outputs/")
    print("="*60)


if __name__ == '__main__':
    main()
