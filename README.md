# Neuropixels + Dopamine Photometry Pipeline

A comprehensive pipeline for processing Neuropixels electrophysiology data (spikes and LFP) and dopamine photometry (TDT) data from mice performing a CW/CCW reward-seeking task in a four-corner chamber.

## Experimental Setup

**Task**: Mice perform a clockwise (CW) or counterclockwise (CCW) navigation task in a four-corner chamber:
- Mice must collect rewards from four corner ports
- After poking the water port, mice can get a second reward by continuing to lick
- Task condition (CW/CCW) switches after 12-30 consecutive rewards
- Mice must adapt their navigation strategy when conditions switch

**Data Types**:
- **Neuropixels**: Spike and LFP data from medial to lateral striatum (4-shank probe)
- **Dopamine Photometry**: Local dopamine release sensor data (TDT system)
- **Behavioral Events**: Camera strobe timing, water delivery, licking (IR sensor), task condition
- **Video Tracking**: DLC-analyzed body part coordinates (60 Hz)

## Repository Structure

```
.
├── run_full_pipeline.py          # Main automated pipeline script
├── Preprocessing/                # Preprocessing notebooks
│   ├── SpikeSorting.ipynb        # Kilosort spike sorting
│   ├── Cellclassify.ipynb        # Cell type classification
│   ├── LFP_extraction_new.ipynb  # LFP extraction
│   └── savewaveform_info.ipynb   # Waveform metrics export
├── pipeline/                     # Core processing modules
│   ├── spikeinterface_waveform_extraction/  # Waveform extraction
│   └── Unitmatch/                # Cross-session unit matching
├── Matlab4TDT/                   # TDT dopamine photometry processing
├── Behavior_analyze.ipynb        # Behavioral analysis
├── DA_analysis.ipynb             # Dopamine analysis
├── Event_DLC_DA_alignment.ipynb  # Multi-modal data alignment
├── SpikeLFP_analysis.ipynb       # Spike-LFP analyses
├── SeqNMF/                       # Sequence NMF analysis
├── Replay_HMM_analysis.ipynb     # HMM replay analysis
├── plot_functions/               # Visualization utilities
└── postanalysis/                 # Post-analysis ideas and plans
```

## Automated Pipeline Flow

The main pipeline (`run_full_pipeline.py`) automates the entire workflow:

### 1. Data Preprocessing
- **CatGT**: Concatenates multiple files, delay adjustment, edge detection for TPrime
- **Channel Map**: `SGLXMetaToCoords.m` creates Kilosort channel map from probe metadata
- **Kilosort4**: Spike sorting with built-in highpass filtering

### 2. Digital I/O Extraction
- Extracts behavioral events from NIDQ digital lines:
  - Camera strobe (line 0)
  - Licking sensor (line 1)
  - Reward delivery (line 2)
  - Trial start (line 4)
  - Stimulation (line 7, if tagging session)
- Converts spike times to seconds for alignment

### 3. Temporal Alignment
- **TPrime**: Aligns spike times across streams (NIDQ ↔ IMEC)
- **Spike Masking**: Removes spikes near stimulation artifacts (±1ms tolerance)

### 4. Quality Control
- **BombCell**: Automated curation and quality metrics
  - Classifies units as Good (1), MUA (2), Non-somatic (3), or Noise (0)
  - Outputs to `kilosort4qMetrics/`

### 5. Waveform Extraction
- **SpikeInterface**: Extracts waveforms for behavioral and tagging periods
  - Creates `analyzer_beh` (full behavioral session)
  - Creates `analyzer_tag` (tagging period only, if g1 session exists)
  - Computes convolution similarity to filter artifacts

### 6. Metrics Export
- Template metrics (peak-to-valley, half-width, etc.)
- Auto-correlograms (ACC)
- ISI histograms
- Unit locations (center of mass)
- Waveform arrays (average and median)

### 7. Cell Classification
- Rule-based classification using waveform metrics:
  - **FSI** (Fast-spiking interneurons): PTV < 0.42ms, HFW < 0.42ms
  - **MSN** (Medium spiny neurons): 0.42ms ≤ PTV < 1.5ms, HFW < 0.75ms
  - **Other**: All remaining units

### 8. LFP Processing
- Extracts LFP from raw AP data:
  - Phase shift correction (if available)
  - Resample to 1000 Hz
  - Bandpass filter (0.5-300 Hz)
  - Common median reference
- Generates sanity check plots (heatmaps, PSD, traces)

### 9. Rastermap Analysis
- Runs Rastermap on quality-filtered unit subsets:
  - All good/MUA units
  - MSN + FSI only
  - MSN only
- Generates sorted activity heatmaps and spatial probe plots

## Manual Workflows

### Dopamine Photometry (TDT)
1. Run `Matlab4TDT/TDT_main.m` or `TDT_test.m`
   - Calls `TDT_demod.m` for demodulation
   - Calls `TDT_dFF_stage2.m` for ΔF/F calculation
2. Output: `*_dFF.mat` files with dopamine signals and timestamps

### Cross-Session Unit Matching
1. Extract waveforms using `pipeline/Unitmatch/for_RawWaveform_extraction/`
2. Run UnitMatch MATLAB pipeline
3. Output: `Unitmatch.mat` with `UniqueIDConversion` struct

### Behavioral Analysis
- `Behavior_analyze.ipynb`: Analyzes DLC tracking, event timing, navigation patterns
- `Event_DLC_DA_alignment.ipynb`: Aligns all data streams to common timebase

## Usage

### Running the Full Pipeline

```bash
python run_full_pipeline.py \
    --data-root /path/to/data \
    --sessions session1 session2 \
    --steps all \
    --matlab matlab \
    --python-bin python
```

### Running Specific Steps

```bash
# Only spike sorting and quality control
python run_full_pipeline.py \
    --data-root /path/to/data \
    --steps kilosort,bombcell,spikeinterface \
    --matlab matlab

# Only LFP extraction
python run_full_pipeline.py \
    --data-root /path/to/data \
    --steps lfp_csd
```

### Key Parameters

- `--tagging`: Enable tagging mode (extracts stimulation line, creates analyzer_tag)
- `--stim-protocol-cleanup`: Remove debris stimulation pulses before protocol start
- `--rastermap-bin-ms`: Bin size for Rastermap (default: 50ms)
- `--digital-lines`: Custom digital lines to extract (default: 0,1,2,4,7 for tagging)

## Data Organization

Each session folder should contain:
```
session_name_g0/
├── session_name_g0_imec0/
│   ├── *.ap.bin                    # Raw AP data
│   ├── *.ap.meta                   # Metadata
│   ├── kilosort4/                  # Kilosort output
│   │   ├── spike_times.npy
│   │   ├── spike_clusters.npy
│   │   ├── template_metrics.csv
│   │   └── waveform_*.npy
│   ├── kilosort4qMetrics/          # BombCell output
│   ├── analyzer_beh/               # SpikeInterface analyzer (behavioral)
│   ├── analyzer_tag/               # SpikeInterface analyzer (tagging, if g1 exists)
│   └── LFP/                        # Extracted LFP data
├── *.nidq.bin                      # NIDQ digital I/O
└── *.nidq.xa_0_500.txt            # TPrime sync file
```

## Dependencies

- **Python**: spikeinterface, numpy, pandas, matplotlib, scipy
- **MATLAB**: BombCell, SGLXMetaToCoords, TDT processing scripts
- **External Tools**: CatGT, Kilosort4, TPrime, Rastermap

## Post-Analysis Modules

The `postanalysis/` directory provides comprehensive analysis tools for neural, behavioral, and photometry data:

### Core Analyses (`postanalysis/analyses.py`)

#### Event-Aligned Neural Activity
- **`calculate_event_tuning()`**: Spike PETH for behavioral events (reward, licking, corner visits)
- **`calculate_lfp_peth()`**: LFP power PETH across frequency bands (theta, beta, gamma)
  - Filters LFP into frequency bands
  - Computes power envelope using Hilbert transform
  - Aligns to behavioral events with customizable time windows
- **`calculate_dopamine_peth()`**: Dopamine signal PETH for behavioral events
  - Event-aligned dopamine release dynamics
  - Baseline normalization and visualization

#### Movement and Kinematics
- **`calculate_movement_tuning()`**: Velocity, acceleration, and turning analysis
- Movement onset PETH and kinematic state segmentation

#### Strategic Encoding
- **`analyze_behavioral_switch_response()`**: Neural responses during strategy switches (CW/CCW)
- **`analyze_directional_tuning()`**: Directional preference and trajectory encoding
- **`analyze_strategy_encoding()`**: Port-to-port trajectory analysis

#### Spike-LFP Coupling
- **`analyze_spike_phase_locking()`**: Comprehensive phase-locking analysis
  - Calculates phase-locking metrics (PLV, PPC, Rayleigh test)
  - Identifies beta-locked and gamma-locked neurons
  - Determines preferred phase angles for each neuron
  - Spatial distribution analysis of phase-locked cells
  - Visualization of phase preferences and locking strength
  
#### Dopamine-Neural Interactions
- **`analyze_dopamine_phase_locking_relationship()`**: Dopamine-phase locking correlation
  - Time-resolved phase-locking during high vs low dopamine periods
  - Correlation analysis between dopamine levels and locking strength
  - Statistical comparison of phase-locking across dopamine states
  - Visualization of dopamine modulation effects

#### Population-Level Analyses
- **`analyze_population_statistics()`**: Statistical testing across all analyses
- **`compare_cell_types()`**: MSN vs FSI comparison with FDR correction
- **`analyze_decoding_performance()`**: Machine learning decoding of behavior from neural activity
- **`analyze_neural_clustering()`**: Functional clustering of neurons

#### Advanced Models
See `postanalysis/models/` for theoretical frameworks:
- Spike-LFP coupling models
- Diffusion equation modeling of LFP propagation
- Phase coherence analysis
- Attractor landscape analysis
- Striatal microcircuit models

### Usage Example

```python
from postanalysis.data_loader import DataPaths
from postanalysis.analyses import (
    calculate_lfp_peth,
    calculate_dopamine_peth,
    analyze_spike_phase_locking,
    analyze_dopamine_phase_locking_relationship
)

# Setup paths
paths = DataPaths(
    base_path="/path/to/session",
    neural_base="/path/to/session/imec0",
    event_corner="/path/to/corner_events.csv",
    event_reward="/path/to/reward_events.csv"
)

# LFP frequency band analysis
lfp_peth_reward = calculate_lfp_peth(
    paths, 
    event_file_type='reward',
    frequency_bands={'beta': (13, 30), 'gamma': (30, 80)},
    time_window_ms=2000
)

# Dopamine event response
da_peth_reward = calculate_dopamine_peth(
    paths,
    event_file_type='reward',
    time_window_ms=2000
)

# Phase-locking analysis
phase_locking = analyze_spike_phase_locking(
    paths,
    frequency_bands={'beta': (13, 30), 'gamma': (30, 80)},
    min_spikes=100
)

# Dopamine-phase locking relationship
da_phase_relationship = analyze_dopamine_phase_locking_relationship(
    paths,
    frequency_bands={'beta': (13, 30), 'gamma': (30, 80)},
    time_window_sec=5.0
)
```

### Output Files

All analyses save results to `{neural_base}/post_analysis/`:
- **CSV files**: Quantitative metrics (firing rates, tuning curves, phase-locking values)
- **PNG files**: Visualizations (heatmaps, PETHs, spatial distributions)
- **Summary reports**: Statistical summaries and publication-ready figures

Key output files:
- `LFP_PETH_{event}_data.csv`: LFP power aligned to events
- `Dopamine_PETH_{event}_data.csv`: Dopamine signal aligned to events
- `spike_phase_locking_data.csv`: Phase-locking metrics per neuron
- `spike_phase_locking_spatial.png`: Spatial distribution of locked cells
- `dopamine_phase_locking_correlations.csv`: DA-phase relationships
- `dopamine_phase_locking_comparison.csv`: High vs low DA comparison

## Next Steps

Additional advanced analyses in development:
- B-SOID/VAME behavior labeling
- DPAD joint neural-behavioral modeling
- Cross-session tracking with UnitMatch
- Replay detection and sequence analysis
