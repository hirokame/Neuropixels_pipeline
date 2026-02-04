"""
Quick Start Guide: Running the Delayed Diffusion Model

This script demonstrates the simplest way to use the delayed diffusion model
to analyze striatal LFP propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from postanalysis.models.diffusion_eq_model import DelayedDiffusionModel


def quick_start_example():
    """
    Quick start example: Generate and visualize LFP propagation.
    """
    print("="*80)
    print("Delayed Diffusion Model - Quick Start Example")
    print("="*80)
    
    # Step 1: Create the model
    print("\n1. Initializing model...")
    model = DelayedDiffusionModel(
        spatial_grid_size=(50, 100),      # 50x100 grid
        spatial_resolution_um=50.0,       # 50μm per point
        time_step_ms=0.5,                 # 0.5ms time steps
        diffusion_coefficient=1.0,        # Moderate diffusion
        conduction_delay_ms=5.0,          # 5ms delay
        anisotropy_ratio=2.0              # 2x faster along probe
    )
    print("   ✓ Model initialized")
    
    # Step 2: Generate synthetic spike data
    print("\n2. Generating synthetic spike data...")
    n_spikes = 1000
    duration = 5.0  # seconds
    
    spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
    spike_positions = np.random.uniform(
        [0, 0], 
        [model.nx * model.dx, model.ny * model.dx], 
        (n_spikes, 2)
    )
    print(f"   ✓ Generated {n_spikes} spikes over {duration}s")
    
    # Step 3: Run simulation
    print("\n3. Running LFP propagation simulation...")
    results = model.simulate_spike_driven_propagation(
        spike_times=spike_times,
        spike_positions=spike_positions,
        duration_sec=duration
    )
    print("   ✓ Simulation complete")
    print(f"   - Recorded {results['lfp_signals'].shape[1]} channels")
    print(f"   - Generated {len(results['V_snapshots'])} spatial snapshots")
    
    # Step 4: Visualize results
    print("\n4. Creating visualizations...")
    output_dir = Path('quick_start_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Propagation snapshots
    fig1 = model.plot_propagation_snapshots(
        results, 
        save_path=str(output_dir / 'propagation.png')
    )
    print(f"   ✓ Saved: {output_dir / 'propagation.png'}")
    
    # Plot 2: LFP timeseries
    fig2, ax = plt.subplots(figsize=(12, 6))
    for ch in range(min(5, results['lfp_signals'].shape[1])):
        ax.plot(results['time'], results['lfp_signals'][:, ch] + ch*0.5, 
               label=f'Channel {ch}', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('LFP (mV)', fontsize=12)
    ax.set_title('Simulated LFP at Multiple Sites', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'lfp_traces.png', dpi=150)
    plt.close(fig2)
    print(f"   ✓ Saved: {output_dir / 'lfp_traces.png'}")
    
    # Plot 3: Frequency analysis
    print("\n5. Analyzing frequencies...")
    ch_idx = 0
    lfp_signal = results['lfp_signals'][:, ch_idx]
    peak_freq, (freqs, power) = model.estimate_resonant_frequency(
        lfp_signal, 
        1000.0 / model.dt
    )
    
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(freqs, power, 'k-', linewidth=1.5)
    ax.axvline(peak_freq, color='red', linestyle='--', linewidth=2,
              label=f'Peak: {peak_freq:.1f} Hz')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title(f'Power Spectrum (Channel {ch_idx})', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'power_spectrum.png', dpi=150)
    plt.close(fig3)
    print(f"   ✓ Saved: {output_dir / 'power_spectrum.png'}")
    print(f"   - Dominant frequency: {peak_freq:.2f} Hz")
    
    # Step 6: Test delay optimization
    print("\n6. Optimizing delay for theta band (8 Hz)...")
    test_delays = np.linspace(2, 15, 8)
    optimal_delay, powers = model.optimize_delay_for_frequency(
        target_freq_hz=8.0,
        test_delays_ms=test_delays,
        n_steps=300
    )
    
    fig4, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_delays, powers, 'o-', linewidth=2, markersize=10, color='navy')
    ax.axvline(optimal_delay, color='red', linestyle='--', linewidth=2,
              label=f'Optimal: {optimal_delay:.2f} ms')
    ax.set_xlabel('Conduction Delay (ms)', fontsize=12)
    ax.set_ylabel('Power at 8 Hz', fontsize=12)
    ax.set_title('Delay Optimization for Theta Band', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'delay_optimization.png', dpi=150)
    plt.close(fig4)
    print(f"   ✓ Saved: {output_dir / 'delay_optimization.png'}")
    print(f"   - Optimal delay: {optimal_delay:.2f} ms")
    
    # Summary
    print("\n" + "="*80)
    print("Quick Start Complete!")
    print("="*80)
    print(f"\nGenerated outputs in: {output_dir}")
    print("\nNext steps:")
    print("1. Try different model parameters (spatial_grid_size, diffusion_coefficient)")
    print("2. Load real Neuropixels data using delayed_diffusion_analysis.py")
    print("3. Compare simulated LFP with actual recordings")
    print("4. Test different frequency bands (theta, beta, gamma)")
    print("\nFor more details, see: DELAYED_DIFFUSION_README.md")
    print("="*80)


if __name__ == '__main__':
    quick_start_example()
