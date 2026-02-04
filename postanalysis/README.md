# Post-Analysis Pipeline

This folder contains the foundational data loading infrastructure and analysis functions for post-processing Neuropixels + behavioral data.

## Quick Start

### Basic Usage

```python
from postanalysis import load_session_data, print_data_summary

# Load all data files for a session
paths = load_session_data(
    mouse_id="1818",
    day="09182025",  # Can be MMDDYYYY, YYYY-MM-DD, YYMMDD, or YYYYMMDD
    base_path=r"Z:\Koji\Neuropixels"
)

# Print summary of found files
print_data_summary(paths)

# Access specific paths
spike_times = paths.kilosort_dir / "spike_times.npy"
events = paths.event_corner
```

### Command-Line Usage

```bash
# Run a specific analysis
python postanalysis/run_post_analysis.py \
    --mouse 1818 \
    --day 09182025 \
    --analysis peth

# Run all analyses
python postanalysis/run_post_analysis.py \
    --mouse 1818 \
    --day 2025-09-18 \
    --analysis all \
    --validate \
    --summary
```

## Module Structure

### `data_loader_refactored.py`

Modular data loading system that strictly adheres to `dataset_config.json`:

- **Proper Frame ID synchronization**: Correctly aligns truncated Event CSVs with complete DLC/Neural data using Frame ID alignment.
- **Modular Data Streams**: Independent loaders for Spikes, DLC, Events, Photometry (TDT), and Strobe timestamps.
- **Robust TDT Extraction**: Extracts absolute timestamps directly from raw H5 handles to avoid normalization errors.

### `analyses.py`

High-precision analysis implementations:

- **Kinematic State Segmentation**: segregates trials into active movement bouts (trajectories) vs stationary periods.
- **Behavioral Switch Helpers**: `_load_switch_times` (fixes embedded switch detection) and `_get_behavioral_switch_points` (aligns to Decision/Success).

### `run_post_analysis.py`

Command-line interface for running the analysis suite.

---

## Behavioral Alignment & Kinematics

The pipeline prioritizes behavioral relevance over raw timestamps:

1. **Environmental vs Behavioral Switch**: Rule changes are detected in behavioral events, and analyses are aligned to the **Decision** (departure from previous port) of the first correct trial.
2. **Kinematic Filters**: Neural firing rates are calculated during clean movement trajectories, identified by velocity thresholds and port-to-port spatial labels.
3. **Synchronization**: All data streams are aligned to the global strobe-timestamp clock (30Hz/60Hz) to ensure sub-frame precision across modalities.

## Available Analyses

The following analyses are implemented and refactored:

### Tuning & PETHs
- `calculate_event_tuning`
  - **Description**: Calculates Peri-Event Time Histograms (PETH) aligned to specific behavioral events.
  - **Input**: Spike times, Event CSV (e.g., Reward, Corner).
  - **Output**: `PETH_{event_type}_data.csv` (binned firing rates), `PETH_{event_type}_heatmap.png` (PETH plots).
- `calculate_movement_tuning`
  - **Description**: Computes neural tuning curves relative to movement velocity.
  - **Input**: Spike times, DLC position data.
  - **Output**: `movement_tuning.csv` (tuning curves), `movement_tuning.png` (tuning plots).
- `calculate_lfp_peth`
  - **Description**: Calculates LFP power PETHs in specific frequency bands (beta, gamma).
  - **Input**: LFP binary data, Event CSV.
  - **Output**: `lfp_power_peth_{band}_{event}.csv`, `lfp_power_peth_{band}_{event}.png`.
- `calculate_dopamine_peth`
  - **Description**: Aligns dopamine photometry signals to behavioral events.
  - **Input**: Photometry data (.mat), Event CSV.
  - **Output**: `dopamine_peth_{event}.csv`, `dopamine_peth_{event}.png`.
- `analyze_spatial_rate_maps`
  - **Description**: Analyzes spatial rate maps.
  - **Input**: Spike times, DLC position data.
  - **Output**: `ratemap/rate_map_cluster_{uid}.png`.

### Behavioral Encoding
- `analyze_behavioral_switch_response`
  - **Description**: Analyzes neural activity aligned to rule switches (Decision/Success).
  - **Input**: Spike times, Corner events, Switch info.
  - **Output**: `behavioral_switch_response.csv`, `behavioral_switch_response.png`.
- `analyze_port_to_port_trajectories`
  - **Description**: Examines activity during specific port-to-port navigation sequences.
  - **Input**: Spike times, Corner events (for trajectory definition).
  - **Output**: `port_to_port_trajectories.csv`, `port_to_port_trajectories.png`.
- `analyze_strategy_encoding`
  - **Description**: Quantifies neuronal preference for specific strategies (CW vs CCW).
  - **Input**: Spike times, Corner events (trajectory context).
  - **Output**: `strategy_encoding.csv`, `strategy_encoding.png`.
- `analyze_directional_tuning`
  - **Description**: Calculates preferred direction vectors (CW vs CCW) for neurons.
  - **Input**: Spike times, Corner events (movements).
  - **Output**: `directional_tuning.csv`, `directional_tuning.png`.
- `analyze_context_dependent_encoding`
  - **Description**: Tests if the same port visit is encoded differently in CW vs CCW contexts.
  - **Input**: Spike times, Corner events.
  - **Output**: `context_encoding.csv`, `context_encoding.png`.
- `analyze_trajectory_consistency`
  - **Description**: Measures neural consistency across repeated traversals of the same path.
  - **Input**: Spike times, Corner events.
  - **Output**: `trajectory_consistency.csv`, `trajectory_consistency.png`.

### Reward & Error Processing
- `analyze_reward_prediction_error`
  - **Description**: Identifies RPE signals by comparing expected vs. unexpected rewards.
  - **Input**: Spike times, Reward events (typed).
  - **Output**: `reward_prediction_error.csv`, `reward_prediction_error.png`.
- `analyze_reward_magnitude_encoding`
  - **Description**: Tests encoding of reward value (e.g., 1st vs 2nd reward at a port).
  - **Input**: Spike times, Reward events.
  - **Output**: `reward_magnitude_encoding.csv`, `reward_magnitude_encoding.png`.
- `analyze_reward_omission`
  - **Description**: Analyzes neural responses when an expected reward is omitted.
  - **Input**: Spike times, Reward omission events.
  - **Output**: `reward_omission.csv`, `reward_omission.png`.
- `analyze_reward_history`
  - **Description**: Examines how previous trial outcomes affect current firing.
  - **Input**: Spike times, Reward/Corner events.
  - **Output**: `reward_history.csv`, `reward_history.png`.
- `analyze_perseveration_signals`
  - **Description**: Identifies neural signatures of perseverative errors after checking switches.
  - **Input**: Spike times, Corner/Switch events.
  - **Output**: `perseveration_analysis.csv`, `perseveration_analysis.png`.
- `analyze_error_detection`
  - **Description**: Compares neural activity between correct choices and errors.
  - **Input**: Spike times, Corner events.
  - **Output**: `error_detection.csv`, `error_detection.png`.
- `analyze_decision_confidence`
  - **Description**: Correlates firing rates with decision confidence (proxied by reaction time).
  - **Input**: Spike times, Corner events (reaction times).
  - **Output**: `decision_confidence.csv`, `decision_confidence.png`.
- `analyze_decision_accumulation`
  - **Description**: Looks for ramping activity indicative of evidence accumulation.
  - **Input**: Spike times, Corner events.
  - **Output**: `decision_accumulation.csv`, `decision_accumulation.png`.
- `analyze_choice_prediction`
  - **Description**: Decodes upcoming port choices from pre-decision activity.
  - **Input**: Spike times, Corner events.
  - **Output**: `choice_prediction.csv`, `choice_prediction.png` (Confusion Matrix).

### Learning & Plasticity
- `analyze_reversal_learning_dynamics`
  - **Description**: Tracks population adaptation dynamics across multiple reversals.
  - **Input**: Spike times, Switch/Corner events.
  - **Output**: `reversal_learning_dynamics.csv`, `reversal_learning_dynamics.png`.
- `analyze_pre_switch_activity`
  - **Description**: Examines neural activity patterns before rule switches.
  - **Input**: Spike times, Switch/Corner events.
  - **Output**: `pre_switch_activity.csv`, `pre_switch_activity.png`.
- `analyze_post_switch_adaptation`
  - **Description**: Compares early vs. late adaptation phases after a rule switch.
  - **Input**: Spike times, Switch events.
  - **Output**: `post_switch_adaptation.csv`, `post_switch_adaptation.png`.
- `analyze_learning_curves`
  - **Description**: Correlates neural changes with behavioral learning curves.
  - **Input**: Spike times, Corner events (performance metrics).
  - **Output**: `learning_curves.csv`, `learning_curves.png`.
- `analyze_navigation_efficiency`
  - **Description**: Correlates neural activity with navigation efficiency metrics.
  - **Input**: Spike times, Corner events/DLC.
  - **Output**: `navigation_efficiency.csv`, `navigation_efficiency.png`.

### Population Dynamics
- `analyze_population_manifolds`
  - **Description**: Projects population activity into low-dimensional space (PCA/UMAP).
  - **Input**: Spike times (Population matrix).
  - **Output**: `population_manifold_{method}.csv` (embeddings), `population_manifold_{method}.png`.
- `analyze_population_trajectories_by_direction`
  - **Description**: Compares state-space trajectories between CW and CCW movements.
  - **Input**: Spike times, Corner events.
  - **Output**: `population_trajectories_direction.csv`, `population_trajectories_direction.png`.
- `analyze_dimensionality_reduction`
  - **Description**: Computes intrinsic dimensionality of the population (PCA/ICA).
  - **Input**: Spike times.
  - **Output**: `dimensionality_reduction_{method}.csv`.
- `analyze_phase_space_trajectories`
  - **Description**: Analyzes flow fields and fixed points in phase space.
  - **Input**: Spike times.
  - **Output**: `phase_space_trajectories.csv`, `phase_space_trajectories.png`.
- `analyze_ica_decomposition`
  - **Description**: Decomposes population activity into independent components.
  - **Input**: Spike times.
  - **Output**: `ica_decomposition.csv`, `ica_decomposition.png`.
- `analyze_decoding_performance`
  - **Description**: Decodes various behavioral variables from population activity.
  - **Input**: Spike times, Behavioral labels (Strategy, Port).
  - **Output**: `decoding_performance.csv`, `decoding_performance.png`.
- `analyze_population_statistics`
  - **Description**: Aggregates statistical summaries across all analyses.
  - **Input**: Results from other analyses.
  - **Output**: `population_statistics.csv`.
- `generate_publication_summary`
  - **Description**: Generates a final summary report and key figures.
  - **Input**: All analysis outputs.
  - **Output**: `publication_summary.md`, `summary_figures/`.

### Temporal Structure & Oscillations
- `analyze_lfp_movement_power`
  - **Description**: correlated LFP power changes with movement epochs.
  - **Input**: LFP, DLC velocity.
  - **Output**: `lfp_movement_power.csv`, `lfp_movement_power.png`.
- `analyze_theta_oscillations`
  - **Description**: Characterizes theta band power and frequency during navigation.
  - **Input**: LFP, DLC velocity.
  - **Output**: `theta_oscillations.csv`, `theta_oscillations.png`.
- `analyze_phase_amplitude_coupling`
  - **Description**: Computes PAC between low-frequency phase and high-frequency amplitude.
  - **Input**: LFP.
  - **Output**: `phase_amplitude_coupling.csv`, `phase_amplitude_coupling.png` (Commodulogram).
- `analyze_cross_frequency_coupling`
  - **Description**: Analysis of cross-frequency coupling interactions.
  - **Input**: LFP.
  - **Output**: `cross_frequency_coupling.csv`, `cross_frequency_coupling.png`.
- `analyze_spike_phase_locking`
  - **Description**: Calculates locking of single-unit spikes to LFP phases.
  - **Input**: Spike times, LFP.
  - **Output**: `spike_phase_locking.csv`, `spike_phase_locking.png` (Polar plots).
- `analyze_neural_synchrony`
  - **Description**: pairing cross-correlations to measure synchrony.
  - **Input**: Spike times.
  - **Output**: `neural_synchrony.csv`, `neural_synchrony.png`.
- `analyze_temporal_autocorrelation`
  - **Description**: Autocorrelation analysis for rhythmicity and refractoriness.
  - **Input**: Spike times.
  - **Output**: `temporal_autocorrelation.csv`, `temporal_autocorrelation.png`.
- `analyze_cross_correlation_pairs`
  - **Description**: Detailed pairwise cross-correlation analysis.
  - **Input**: Spike times.
  - **Output**: `cross_correlation_pairs.csv`, `cross_correlation_pairs.png`.
- `analyze_bursting_behavior`
  - **Description**: Detects burst firing modes in single units.
  - **Input**: Spike times.
  - **Output**: `bursting_behavior.csv`, `bursting_behavior.png`.
- `analyze_isi_distribution`
  - **Description**: Computes ISI distributions and CV statistics.
  - **Input**: Spike times.
  - **Output**: `isi_distribution.csv`, `isi_distribution.png`.
- `analyze_spike_pattern_motifs`
  - **Description**: Finds repeating sequential firing patterns (SeqNMF).
  - **Input**: Spike times.
  - **Output**: `spike_pattern_motifs.csv`, `spike_pattern_motifs.png`.
- `analyze_wave_propagation`
  - **Description**: Detects traveling waves across the electrode array.
  - **Input**: Spike times, Probe geometry.
  - **Output**: `wave_propagation.csv`, `wave_propagation.png`.
- `analyze_multi_scale_temporal_encoding`
  - **Description**: information encoding analysis at multiple time scales.
  - **Input**: Spike times.
  - **Output**: `multi_scale_encoding.csv`.
- `analyze_rank_order_coding`
  - **Description**: Investigates sequence coding via spike rank order.
  - **Input**: Spike times.
  - **Output**: `rank_order_coding.csv`.
- `analyze_temporal_clustering`
  - **Description**: Clusters neurons based on temporal profiles.
  - **Input**: Spike times.
  - **Output**: `temporal_clustering.csv`, `temporal_clustering.png`.

### Spatial & Dopamine Interactions
- `analyze_dopamine_spike_coupling`
  - **Description**: Cross-correlation between dopamine transients and spikes.
  - **Input**: Dopamine data, Spike times.
  - **Output**: `dopamine_spike_coupling.csv`, `dopamine_spike_coupling.png`.
- `analyze_dopamine_triggered_firing`
  - **Description**: PETH of spikes aligned to dopamine peaks.
  - **Input**: Dopamine data, Spike times.
  - **Output**: `dopamine_triggered_firing.csv`, `dopamine_triggered_firing.png`.
- `analyze_dopamine_modulation_index`
  - **Description**: Statistics of dopamine modulation for each neuron.
  - **Input**: Dopamine data, Spike times.
  - **Output**: `dopamine_modulation.csv`.
- `analyze_dopamine_lfp_coupling`
  - **Description**: Relationship between dopamine and LFP power.
  - **Input**: Dopamine data, LFP.
  - **Output**: `dopamine_lfp_coupling.csv`, `dopamine_lfp_coupling.png`.
- `analyze_dopamine_phase_locking_relationship`
  - **Description**: How dopamine levels affect spike-LFP phase locking.
  - **Input**: Dopamine, LFP, Spike times.
  - **Output**: `dopamine_phase_locking.csv`.
- `analyze_spatial_organization_depth`
  - **Description**: Activity profiles along the probe depth.
  - **Input**: Spike times, Channel positions.
  - **Output**: `spatial_organization_depth.csv`, `spatial_organization_depth.png`.
- `analyze_depth_tuning_by_behavior`
  - **Description**: Behavioral tuning maps vs. cortical depth.
  - **Input**: Spike times, Channel positions, Behavior.
  - **Output**: `depth_tuning_by_behavior.csv`, `depth_tuning_by_behavior.png`.
- `analyze_spatial_clustering`
  - **Description**: Spatial clustering of functional cell types.
  - **Input**: Spike times, Channel positions.
  - **Output**: `spatial_clustering.csv`, `spatial_clustering.png`.
- `analyze_medial_lateral_organization`
  - **Description**: Functional gradients ML axis (if valid).
  - **Input**: Spike times, Probe geometry.
  - **Output**: `medial_lateral_organization.csv`.
- `analyze_multi_shank_interactions`
  - **Description**: Interaction metrics between different probe shanks.
  - **Input**: Spike times, Probe geometry.
  - **Output**: `multi_shank_interactions.csv`.
- `analyze_neural_clustering`
  - **Description**: Unsupervised clustering of neurons by functional properties.
  - **Input**: Feature matrix (from other analyses).
  - **Output**: `neural_clustering_assignments.csv` (Classes), `neural_clustering_profiles.csv`.
- `analyze_functional_tuning_matrix`
  - **Description**: Aggregates all tuning metrics into a master matrix.
  - **Input**: All tuning results.
  - **Output**: `functional_tuning_matrix.csv`.
- `compare_cell_types`
  - **Description**: Statistical comparison of metrics between cell types (e.g. MSN vs FSI).
  - **Input**: Functional Tuning Matrix.
  - **Output**: `cell_type_comparisons.csv`, `cell_type_comparisons.png`.
- `analyze_mutual_information`
  - **Description**: Information theoretic analysis of behavior encoding.
  - **Input**: Spike times, Behavioral variables.
  - **Output**: `mutual_info_summary.csv`, `mutual_info_{event}_lagged.csv`, `mutual_info_heatmap_{event}.png`, `mutual_info_population_curves.png`.
- `predictive_decoding`
  - **Description**: Decoding of behavior from neural activity.
  - **Input**: Spike times, DLC, Event data.
  - **Output**: `predictive_decoding_comparison.csv`, `predictive_decoding_summary.png`.

## Adding New Analyses

To add a new analysis:

1. Create a function in `run_post_analysis.py`:
   ```python
   def run_my_analysis(paths: DataPaths):
       """Run my custom analysis."""
       # Load data using paths
       spike_times = np.load(paths.kilosort_dir / "spike_times.npy")
       # ... do analysis
   ```

2. Add to `ANALYSIS_FUNCTIONS` dictionary:
   ```python
   ANALYSIS_FUNCTIONS = {
       ...
       'my_analysis': run_my_analysis,
   }
   ```

3. Run with:
   ```bash
   python postanalysis/run_post_analysis.py --mouse 1818 --day 09182025 --analysis my_analysis
   ```

## Data Validation

Before running analyses, you can validate that required data exists:

```python
from postanalysis import load_session_data, validate_data_paths

paths = load_session_data("1818", "09182025")
validation = validate_data_paths(paths, required=['neural', 'events', 'dlc', 'tdt'])

for data_type, exists in validation.items():
    print(f"{data_type}: {'✓' if exists else '✗'}")
```

## Next Steps

See `post_analysis_ideas.md` for comprehensive analysis ideas and implementation priorities.

