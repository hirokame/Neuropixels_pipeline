import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths and Parameters
lfp_path = Path(r"e:\Neuropixels\Python\DemoData\1818_09182025_g0\1818_09182025_g0_imec0\LFP\traces_cached_seg0.raw")
output_dir = Path(r"e:\Neuropixels\Python\lfp_plots")
output_dir.mkdir(exist_ok=True, parents=True)

n_channels = 384 # Note: 384 channels, not 385 like AP raw (usually sync is removed or it's just probe channels)
fs = 1000.0
dtype = 'int16'
# Scaling from binary.json/meta
uV_per_bit = 3.02734375

duration_sec = 10 # 2 seconds for LFP to see rhythms
n_windows = 10

# Calculate file limits
n_bytes_per_sample = 2 * n_channels
file_size = lfp_path.stat().st_size
total_samples = file_size // n_bytes_per_sample
total_duration = total_samples / fs

window_starts = np.linspace(0, total_duration - duration_sec - 10, n_windows)
channels_to_plot = [50, 100, 150, 200, 250]
n_samples = int(duration_sec * fs)

print(f"Reading from {lfp_path}")
print(f"Sampling Rate: {fs} Hz, Duration: {total_duration:.1f}s")

data_map = np.memmap(lfp_path, dtype=dtype, mode='r', order='C')
data_map = data_map.reshape((total_samples, n_channels))

for i, start_time in enumerate(window_starts):
    start_sample = int(start_time * fs)
    end_sample = start_sample + n_samples
    
    print(f"Plotting LFP {i+1}/{n_windows}: t={start_time:.1f}s")
    
    chunk = data_map[start_sample:end_sample, :]
    
    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(16, 12), sharex=True)
    time_axis = np.arange(n_samples) / fs
    
    for j, ch in enumerate(channels_to_plot):
        ax = axes[j]
        # Extract and scale
        trace_uV = chunk[:, ch] * uV_per_bit
        
        ax.plot(time_axis, trace_uV, linewidth=0.8, color='#004488', label='LFP')
        
        ax.set_ylabel(f"Ch {ch}\n(uV)", rotation=0, labelpad=40, va='center')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    axes[-1].set_xlabel("Time (s) from window start")
    fig.suptitle(f"LFP Cache Traces (1000Hz) @ t={start_time:.2f}s", y=0.92)
    
    out_path = output_dir / f"lfp_plot_{i+1:02d}_t{int(start_time)}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

print("LFP plotting complete.")
