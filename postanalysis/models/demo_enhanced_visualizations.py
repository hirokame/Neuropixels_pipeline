"""
Demo script showcasing enhanced visualization features for all models.

This script demonstrates the new visualization capabilities added to:
1. Attractor Energy Landscape Model
2. Delayed Diffusion Model
3. Striatal Microcircuit Model
4. Spike-LFP Coupling Model

Usage:
    python demo_enhanced_visualizations.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import all models
from attractor_energy_landscape import AttractorEnergyLandscapeModel
from diffusion_eq_model import DelayedDiffusionModel
from striatal_microcircuit import StrialMicrocircuitModel
from spike_lfp_coupling import SpikeLFPCouplingModel


def demo_attractor_model():
    """Demonstrate enhanced attractor model visualizations."""
    print("\n" + "="*80)
    print("DEMO 1: Attractor Energy Landscape Model - Enhanced Visualizations")
    print("="*80)
    
    output_dir = Path('/tmp/attractor_demo')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = AttractorEnergyLandscapeModel(
        n_dimensions=3,
        n_attractors=2,
        attractor_strength=2.0,
        noise_level=0.15,
        coupling_alpha=1.0,
        coupling_beta=0.5
    )
    
    # Simulate trial
    print("\n1. Simulating trial with sensory bias...")
    trial_results = model.simulate_trial(
        duration_sec=3.0,
        bias_onset_sec=1.0,
        bias_direction=0,
        bias_strength=2.0
    )
    
    # Original visualization
    print("\n2. Generating original trial visualization...")
    model.plot_trial_results(trial_results, 
                            save_path=str(output_dir / 'original_trial.png'))
    
    # NEW: Enhanced landscape visualization
    print("\n3. Generating ENHANCED visualization with:")
    print("   - 3D energy landscape surface")
    print("   - Vector field (force gradients)")
    print("   - Velocity-stability phase space")
    print("   - Exploration gain decomposition")
    model.plot_enhanced_landscape(trial_results,
                                 save_path=str(output_dir / 'enhanced_landscape.png'))
    
    # NEW: Cross-correlation analysis
    print("\n4. Demonstrating cross-correlation analysis...")
    # Generate synthetic behavioral velocity
    neural_velocity = trial_results['velocities']
    # Simulate behavioral velocity that lags neural velocity
    behavioral_velocity = np.concatenate([np.zeros(50), neural_velocity[:-50]])
    
    cross_corr = model.compute_cross_correlation(
        neural_velocity=neural_velocity,
        behavioral_velocity=behavioral_velocity,
        sampling_rate=1000.0,
        max_lag_sec=0.2
    )
    
    # Plot cross-correlation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cross_corr['lags'], cross_corr['correlations'], 'b-', linewidth=2)
    ax.axvline(cross_corr['peak_lag_sec'], color='r', linestyle='--', linewidth=2,
              label=f"Peak lag: {cross_corr['peak_lag_sec']*1000:.1f}ms")
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Lag (seconds)', fontsize=12)
    ax.set_ylabel('Cross-Correlation', fontsize=12)
    ax.set_title('Neural vs Behavioral Velocity Cross-Correlation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, cross_corr['interpretation'], transform=ax.transAxes,
           va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_correlation.png', dpi=150)
    plt.close()
    
    print(f"\n✓ Attractor model demo complete! Outputs saved to {output_dir}")


def demo_diffusion_model():
    """Demonstrate enhanced diffusion model visualizations."""
    print("\n" + "="*80)
    print("DEMO 2: Delayed Diffusion Model - Enhanced Visualizations")
    print("="*80)
    
    output_dir = Path('/tmp/diffusion_demo')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = DelayedDiffusionModel(
        spatial_grid_size=(80, 80),
        spatial_resolution_um=20.0,
        time_step_ms=0.1,
        diffusion_coefficient=1.0,
        conduction_delay_ms=5.0,
        anisotropy_ratio=3.0
    )
    
    # Generate synthetic spike data
    print("\n1. Generating synthetic spike data...")
    n_spikes = 200
    spike_times = np.sort(np.random.uniform(0, 0.5, n_spikes))
    spike_positions = np.random.uniform(400, 1200, (n_spikes, 2))
    
    # Recording positions
    recording_positions = []
    for x in [500, 750, 1000]:
        for y in [600, 800, 1000]:
            recording_positions.append([x, y])
    recording_positions = np.array(recording_positions)
    
    # Simulate
    print("\n2. Simulating LFP propagation...")
    results = model.simulate_spike_driven_propagation(
        spike_times=spike_times,
        spike_positions=spike_positions,
        duration_sec=0.5,
        recording_positions=recording_positions,
        spike_amplitude=0.05
    )
    
    # NEW: Iso-potential contours
    print("\n3. Generating iso-potential contours (anisotropy visualization)...")
    model.plot_isopotential_contours(results,
                                    save_path=str(output_dir / 'isopotential_contours.png'))
    
    # NEW: Phase-distance plot
    print("\n4. Generating phase-distance plot (traveling wave analysis)...")
    model.plot_phase_distance_analysis(results,
                                      spike_positions=spike_positions,
                                      target_frequency_hz=8.0,
                                      save_path=str(output_dir / 'phase_distance.png'))
    
    # NEW: Frequency-delay heatmap
    print("\n5. Generating frequency-delay optimization heatmap...")
    print("   (This may take a minute...)")
    model.plot_frequency_delay_heatmap(
        test_frequencies=np.linspace(4, 40, 10),
        test_delays=np.linspace(2, 15, 10),
        n_steps=200,
        save_path=str(output_dir / 'frequency_delay_heatmap.png')
    )
    
    print(f"\n✓ Diffusion model demo complete! Outputs saved to {output_dir}")


def demo_striatal_model():
    """Demonstrate enhanced striatal microcircuit visualizations."""
    print("\n" + "="*80)
    print("DEMO 3: Striatal Microcircuit Model - Enhanced Visualizations")
    print("="*80)
    
    output_dir = Path('/tmp/striatal_demo')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = StrialMicrocircuitModel(
        n_d1_msns=30,
        n_d2_msns=30,
        n_fsis=15,
        base_synaptic_delay_ms=3.0,
        sampling_rate_hz=1000.0
    )
    
    # Save initial delays
    initial_delays = model.fsi_msn_delays.copy()
    
    # Simulate multiple trials with learning
    print("\n1. Simulating trials with dopamine-driven learning...")
    trial_history = []
    target_frequencies = [8.0, 10.0, 12.0, 10.0, 8.0]  # Vary target frequency
    
    for trial_idx in range(5):
        print(f"   Trial {trial_idx + 1}/5...")
        
        # Simulate trial
        trial_results = model.simulate_trial(
            duration_sec=2.0,
            input_rate_hz=15.0,
            dopamine_level=1.5
        )
        trial_history.append(trial_results)
        
        # Apply plasticity
        target_freq = target_frequencies[trial_idx]
        model.apply_dopamine_plasticity(
            dopamine_level=1.5,
            reward_outcome=True,
            target_frequency_hz=target_freq,
            learning_rate=0.05
        )
        
        # Also apply STDP
        model.apply_stdp_delay_plasticity(
            msn_spikes=trial_results['msn_spikes'],
            fsi_spikes=trial_results['fsi_spikes'],
            dopamine_level=1.5,
            learning_rate=0.01
        )
    
    # NEW: Population raster plot
    print("\n2. Generating population raster plot...")
    model.plot_population_raster(trial_history[0],
                                save_path=str(output_dir / 'population_raster.png'))
    
    # NEW: Delay matrix evolution
    print("\n3. Generating delay matrix evolution visualization...")
    final_delays = model.fsi_msn_delays.copy()
    model.plot_delay_matrix_evolution(initial_delays, final_delays,
                                     save_path=str(output_dir / 'delay_matrix_evolution.png'))
    
    # NEW: D1/D2 balance ratio
    print("\n4. Generating D1/D2 balance ratio plot...")
    model.plot_d1_d2_balance(trial_history,
                            save_path=str(output_dir / 'd1_d2_balance.png'))
    
    # Original: Resonance tuning
    print("\n5. Generating resonance tuning plot...")
    model.plot_resonance_tuning(trial_history,
                               save_path=str(output_dir / 'resonance_tuning.png'))
    
    print(f"\n✓ Striatal model demo complete! Outputs saved to {output_dir}")


def demo_spike_lfp_coupling():
    """Demonstrate enhanced spike-LFP coupling visualizations."""
    print("\n" + "="*80)
    print("DEMO 4: Spike-LFP Coupling Model - Enhanced Visualizations")
    print("="*80)
    
    output_dir = Path('/tmp/spike_lfp_demo')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = SpikeLFPCouplingModel(
        spatial_kernel_um=150.0,
        temporal_kernel_ms=15.0,
        sampling_rate_hz=1000.0
    )
    
    # Generate synthetic data
    print("\n1. Generating synthetic spike train and LFP...")
    duration_sec = 10.0
    n_spikes = 500
    
    # Generate spikes with preferred phase locking at 8 Hz
    target_freq = 8.0
    preferred_phase = np.pi / 4  # 45 degrees
    
    spike_times = []
    t = 0
    while t < duration_sec:
        # Phase at current time
        phase = 2 * np.pi * target_freq * t
        # Probability modulated by phase
        prob = 0.02 * (1 + 0.5 * np.cos(phase - preferred_phase))
        if np.random.rand() < prob:
            spike_times.append(t)
        t += 0.001  # 1ms steps
    
    spike_times = np.array(spike_times)
    n_spikes = len(spike_times)
    
    # Generate spike positions
    recording_pos = np.array([500.0, 500.0])
    spike_positions = recording_pos + np.random.randn(n_spikes, 2) * 100
    
    # Generate LFP from spikes
    lfp_results = model.generate_lfp_from_spikes(
        spike_times=spike_times,
        spike_positions=spike_positions,
        recording_position=recording_pos,
        duration_sec=duration_sec
    )
    
    # NEW: Spike-triggered average
    print("\n2. Computing and plotting spike-triggered average...")
    sta_results = model.compute_spike_triggered_average(
        spike_times=spike_times,
        lfp_signal=lfp_results['lfp_signal'],
        sampling_rate=1000.0,
        window_sec=0.1
    )
    
    model.plot_spike_triggered_average(sta_results,
                                      save_path=str(output_dir / 'spike_triggered_average.png'))
    
    # NEW: Phase locking analysis
    print("\n3. Computing phase locking...")
    phase_results = model.compute_phase_locking(
        spike_times=spike_times,
        lfp_signal=lfp_results['lfp_signal'],
        sampling_rate=1000.0,
        target_frequency_hz=target_freq
    )
    
    print(f"   Phase Locking Value: {phase_results['phase_locking_value']:.3f}")
    print(f"   Preferred Phase: {np.degrees(phase_results['preferred_phase']):.1f}°")
    print(f"   Rayleigh p-value: {phase_results['rayleigh_p']:.4f}")
    
    # NEW: Phase polar histogram
    print("\n4. Generating phase polar histogram...")
    model.plot_phase_polar_histogram(phase_results,
                                    save_path=str(output_dir / 'phase_polar_histogram.png'))
    
    print(f"\n✓ Spike-LFP coupling demo complete! Outputs saved to {output_dir}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("ENHANCED VISUALIZATIONS DEMO")
    print("Demonstrating new visualization features for all four models")
    print("="*80)
    
    try:
        # Demo 1: Attractor model
        demo_attractor_model()
        
        # Demo 2: Diffusion model
        demo_diffusion_model()
        
        # Demo 3: Striatal microcircuit
        demo_striatal_model()
        
        # Demo 4: Spike-LFP coupling
        demo_spike_lfp_coupling()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nOutput directories:")
        print("  /tmp/attractor_demo/")
        print("  /tmp/diffusion_demo/")
        print("  /tmp/striatal_demo/")
        print("  /tmp/spike_lfp_demo/")
        print("\nNew visualization features demonstrated:")
        print("\n1. Attractor Model:")
        print("   ✓ 3D energy landscape with trajectory")
        print("   ✓ Vector field (force gradients)")
        print("   ✓ Velocity-stability phase space")
        print("   ✓ Exploration gain decomposition")
        print("   ✓ Cross-correlation analysis")
        print("\n2. Diffusion Model:")
        print("   ✓ Iso-potential contours (anisotropy)")
        print("   ✓ Phase-distance plot (traveling waves)")
        print("   ✓ Frequency-delay optimization heatmap")
        print("\n3. Striatal Microcircuit:")
        print("   ✓ Population raster (color-coded)")
        print("   ✓ Delay matrix evolution")
        print("   ✓ D1/D2 balance ratio")
        print("   ✓ STDP-based plasticity")
        print("\n4. Spike-LFP Coupling:")
        print("   ✓ Spike-triggered average")
        print("   ✓ Phase polar histogram")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
