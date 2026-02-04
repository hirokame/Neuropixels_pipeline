# Post-Analysis Ideas: Striatal Neural Dynamics in CW/CCW Navigation Task

This document outlines comprehensive analysis ideas for understanding striatal neural activity, dopamine dynamics, and behavior in the context of a CW/CCW reward-seeking navigation task.

## Table of Contents

1. [Spike/LFP Analysis with Behavioral Variables](#1-spikelfp-analysis-with-behavioral-variables)
2. [Theoretical Modeling: LFP Propagation & Microcircuit Dynamics](#2-theoretical-modeling-lfp-propagation--microcircuit-dynamics)
3. [Novel Encoding Schemes for Striatal Information](#3-novel-encoding-schemes-for-striatal-information)
4. [Behavioral State Discovery and Labeling](#4-behavioral-state-discovery-and-labeling)
5. [Multi-Modal Integration Analyses](#5-multi-modal-integration-analyses)
6. [Temporal Dynamics and Sequence Analysis](#6-temporal-dynamics-and-sequence-analysis)
7. [Spatial Organization and Topography](#7-spatial-organization-and-topography)
8. [Decision-Making and Strategy Switching](#8-decision-making-and-strategy-switching)
9. [Dopamine-Neural Interactions](#9-dopamine-neural-interactions)
10. [Advanced Statistical and Machine Learning Approaches](#10-advanced-statistical-and-machine-learning-approaches)
11. [Functional Characterization & Integration](#11-functional-characterization--integration)

---

## 1. Spike/LFP Analysis with Behavioral Variables

### 1.1 Peri-Event Time Histograms (PETHs) ✅ IMPLEMENTED
- **Event-aligned**: ✅ `calculate_event_tuning`
  - Spike/LFP around reward delivery and all lick events - ✅ Implemented
- **Movement tuning**: ✅ `calculate_movement_tuning`
  - Correlate firing rate with movement speed, acceleration/deceleration - ✅ Implemented
  - Activity during turns vs straight segments - ✅ Implemented
  - Align with movement onset - ✅ Implemented
- **Directional tuning**: ✅ `analyze_directional_tuning`
  - Preferred direction vectors (CW vs CCW) - ✅ Implemented
- **Condition switch**: ✅ `analyze_behavioral_switch_response`
  - Neural response to CW→CCW or CCW→CW transitions - ✅ Implemented
- **Port-to-port trajectories**: ✅ `analyze_port_to_port_trajectories`
  - Activity during specific navigation paths - ✅ Implemented

### 1.2 Decision-Related Activity ✅ PARTIALLY IMPLEMENTED
- **Choice prediction**: ✅ `analyze_choice_prediction`
- **Error detection**: ✅ `analyze_error_detection`
- **Strategy encoding**: ✅ `analyze_strategy_encoding`
- **Decision confidence**: ✅ `analyze_decision_confidence`

### 1.3 State Transition Analysis ✅ IMPLEMENTED
- **Pre-switch activity**: ✅ `analyze_pre_switch_activity`
- **Post-switch adaptation**: ✅ `analyze_post_switch_adaptation`
- **Perseveration signals**: ✅ `analyze_perseveration_signals`
- **Reversal learning dynamics**: ✅ `analyze_reversal_learning_dynamics`

### 1.4 Reward-Related Dynamics ✅ IMPLEMENTED
- **Reward prediction error (RPE)**: ✅ `analyze_reward_prediction_error`
- **Reward magnitude encoding**: ✅ `analyze_reward_magnitude_encoding`
- **Reward omission**: ✅ `analyze_reward_omission`
- **Reward history**: ✅ `analyze_reward_history`

### 1.5 LFP-Behavioral Correlations ✅ IMPLEMENTED
- **Beta/gamma power**: ✅ `analyze_lfp_movement_power`
- **Theta oscillations**: ✅ `analyze_theta_oscillations`
- **Phase-amplitude coupling**: ✅ `analyze_phase_amplitude_coupling`
- **Cross-frequency coupling**: ✅ `analyze_cross_frequency_coupling`

---

## 2. Theoretical Modeling: LFP Propagation & Microcircuit Dynamics ✅ IMPLEMENTED

**Enhanced with Pouzzner's "Basal Ganglia Mediated Synchronization" Theory**

### 2.1 Diffusion Equation Models ✅ IMPLEMENTED
- **LFP propagation**: ✅ `DelayedDiffusionModel` in `models/diffusion_eq_model.py`
  - Delayed diffusion equation: ∂V/∂t = D∇²V(t - τ) + S(x,t)
  - Conduction delays (τ) facilitate resonance at specific frequencies
  - Estimate diffusion coefficient D from LFP spatial decay
  - Source terms S(x,t) from spike locations and times
- **Cable theory integration**: Partially covered in diffusion model
- **Anisotropic diffusion**: ✅ Different diffusion rates along vs across shanks

**Pouzzner Integration #1**: Delay term models temporal lag for phase-adjustment mechanism
ensuring signals return to cortex at precisely the right time for global synchronization.

### 2.2 Coupled Spike-LFP Models ✅ IMPLEMENTED
- **Spike-LFP coupling**: ✅ `SpikeLFPCouplingModel` in `models/spike_lfp_coupling.py`
  - Current source density (CSD) from spike locations
  - Forward model: LFP = f(spikes, spatial kernel)
- **Feedback effects**: ✅ Implemented in phase-coherence gating
  - LFP-modulated firing rates
  - Phase-dependent spike probability

**Pouzzner Integration #2**: Phase-Coherence Gating where FSI-mediated inhibition forces
MSN spike timing to entrain to specific LFP phases, acting as "Phase Filters."

### 2.3 Striatal Microcircuit Models ✅ IMPLEMENTED
- **MSN-FSI interactions**: ✅ `StrialMicrocircuitModel` in `models/striatal_microcircuit.py`
  - Feedforward inhibition from FSIs to MSNs
  - Phase-based gating of MSN activity
- **Dopamine modulation**: ✅ FULLY IMPLEMENTED
  - D1 vs D2 MSN differential responses
  - Dopamine-dependent plasticity rules
  - **Dopamine plastically modifies synaptic delays and time constants**
- **Lateral inhibition**: Covered in MSN-FSI interactions

**Pouzzner Integration #3**: Dopamine-Driven Tuning where DA transients tune circuit to
specific frequencies (CW/CCW-specific) required for successful task execution. Evolves into
"Reinforcement-Driven Resonance Model."

### 2.4 Network Models - Partially Implemented
- **Striatal network simulation**: Covered in microcircuit model
  - Spiking neural network with realistic connectivity
  - Spatial organization considerations
  - Cell-type specific dynamics (MSN, FSI)
- **Emergent oscillations**: Covered in phase-coherence gating
- **Information flow**: ✅ Implemented with spatial anisotropy

**Pouzzner Integration #4**: Spatial Anisotropy implements directional flow toward specific
output nuclei (SNr/GPi), reflecting convergence in cortical inputs and divergence in return
paths via thalamus.

### 2.5 Phase-Field Models ✅ IMPLEMENTED
- **Continuum neural field**: ✅ `PhaseFieldModel` in `models/phase_coherence_gating.py`
  - Activity field u(x,t) with local interactions
  - Phase field φ(x,t) for oscillation dynamics
  - Traveling waves of activity
  - Pattern formation during behavior

### 2.6 Attractor Energy Landscape Model ✅ IMPLEMENTED
- **Uncertainty-driven exploration**: ✅ `AttractorEnergyLandscapeModel` in `models/attractor_energy_landscape.py`
  - Neural states modeled as ball rolling in energy landscape
  - Deep attractors (certainty) → stable states → exploitation
  - Shallow attractors (uncertainty) → oscillations → exploration
  - **Zero-lag energy spillover**: Neural velocity directly drives motor primitives
  - Motor_Gain = α * ||dx/dt|| + β * Var(x)
  - **No explicit uncertainty computation** - physical coupling only
  - Phase transitions: metastable → attractor formation → exploration stops
- **Neuropixels validation**: 
  - Predict: Exploratory movements preceded by high neural trajectory speed
  - Method: `analyze_movement_onset()` compares pre-movement vs baseline
  - Compare exploration vs exploitation trajectory dynamics
- **Robotics applications**:
  - Zero computational overhead - ideal for edge AI
  - Reflexive exploration for drones, legged robots
  - Embodied intelligence without explicit decision-making

---

**Implementation Notes:**
All models in `postanalysis/models/` directory with:
- Complete implementations with docstrings
- Example usage script (`example_models_usage.py`)
- Comprehensive README with mathematical frameworks
- Integration of all 4 Pouzzner insights + new attractor framework

---

## 3. Novel Encoding Schemes for Striatal Information

### 3.1 Temporal Pattern Codes ✅ PARTIALLY IMPLEMENTED
- **Spike pattern motifs**: ✅ `analyze_spike_pattern_motifs`
  - Use SeqNMF or similar to find motifs
  - Relate motifs to behavioral events
- **Burst codes**: ✅ `analyze_bursting_behavior`
  - Burst detection and characterization
  - Burst timing relative to behavior
- **Interval codes**: ✅ `analyze_isi_distribution`
  - ISI distributions as encoding dimension
  - Temporal precision vs rate trade-offs

### 3.2 Population Trajectory Codes ✅ IMPLEMENTED
- **Neural manifolds**: ✅ `analyze_population_manifolds`
  - PCA, t-SNE, UMAP of population activity - ✅ Implemented
  - Relate trajectories to behavioral states - ✅ Implemented
- **CW vs CCW trajectory comparison**: ✅ `analyze_population_trajectories_by_direction`
  - Separate CW and CCW population trajectories - ✅ NEW
  - Quantify trajectory differences between strategies - ✅ NEW
  - Visualize trajectory separation over time - ✅ NEW
- **Dynamical systems approach**: 
  - Attractors for different behaviors - Partially covered
  - Transitions between attractors
- **Latent dynamics**: Covered in manifold analyses

### 3.3 Relative Timing Codes ✅ PARTIALLY IMPLEMENTED
- **Temporal order**: ✅ `analyze_neural_synchrony`
  - Rank-order codes - ✅ `analyze_rank_order_coding` - ✅ NEW
  - Synchrony patterns - ✅ Implemented
  - Sequence of first-to-fire neurons - ✅ NEW
- **Phase codes**: ✅ `analyze_phase_amplitude_coupling`
  - Spike timing relative to LFP phase
  - Cross-neuron phase relationships
- **Relative latency**: Partially covered in synchrony

### 3.4 Spatial-Temporal Codes ✅ PARTIALLY IMPLEMENTED
- **Wave propagation**: ✅ `analyze_wave_propagation`
  - Wave direction and speed - ✅ NEW
  - Wave initiation sites - ✅ NEW
  - Traveling waves in population activity
- **Spatial patterns**: ✅ `analyze_spatial_clustering`
  - Which neurons fire together spatially?
  - Spatial clustering of activity
- **Topographic maps**: Partially covered
  - Movement direction maps
  - Reward location maps

### 3.5 Context-Dependent Codes ✅ IMPLEMENTED
- **State-dependent encoding**: ✅ `analyze_context_dependent_encoding`
  - CW vs CCW context
  - Before vs after switch
- **History-dependent codes**: ✅ `analyze_reward_history`
  - Previous reward outcomes
  - Recent movement patterns
- **Predictive codes**: ✅ Covered in reward prediction error

### 3.6 Multi-Scale Codes ✅ IMPLEMENTED
- **Hierarchical encoding**: ✅ `analyze_multi_scale_temporal_encoding`
  - Fast: individual spikes - ✅ NEW (10ms bins)
  - Medium: firing rate (100s ms) - ✅ NEW (50-100ms bins)
  - Slow: rate changes (seconds) - ✅ NEW (500-1000ms bins)
  - Multi-timescale information content
- **Scale-free dynamics**: Not implemented
- **Fractal patterns**: Not implemented

### 3.7 Information-Theoretic Approaches ✅ PARTIALLY IMPLEMENTED
- **Mutual information**: ✅ `analyze_mutual_information`
  - Which neurons encode which behaviors?
  - Information content of different coding schemes
- **Transfer entropy**: Not implemented
  - Directed information flow
  - Causal relationships
- **Integrated information**: Not implemented

---

## 4. Behavioral State Discovery and Labeling

### 4.1 B-SOID Analysis
- **Behavioral segmentation**: Automatic behavior classification
  - Input: DLC body part coordinates
  - Output: Behavioral labels (e.g., "port approach", "licking", "turning")
- **Neural correlates**: Map neural activity to discovered behaviors
- **Behavioral sequences**: Identify recurring behavioral sequences

### 4.2 VAME (Variational Autoencoder for Motion Embedding)
- **Latent behavioral space**: Low-dimensional embedding of behavior
- **Behavioral dynamics**: How does behavior evolve in latent space?
- **Neural-behavioral alignment**: Align neural and behavioral latent spaces

### 4.3 Hidden Markov Models (HMM)
- **Behavioral states**: Discrete states from continuous behavior
- **State transitions**: Transition probabilities between states
- **Neural state coupling**: How do neural states relate to behavioral states?

### 4.4 DPAD (Dynamical Population Analysis with Dimensionality reduction)
- **Joint neural-behavioral modeling**: 
  - Shared latent space for neural and behavioral data
  - How do they co-evolve?
- **Predictive modeling**: Predict behavior from neural activity

### 4.5 Custom Behavioral Metrics
- **Navigation efficiency**: Path length / optimal path length (--navigation_efficiency)
- **Strategy adherence**: How well does mouse follow current strategy?
- **Switch detection**: When does mouse detect strategy change?
- **Exploration vs exploitation**: Balance between exploration and exploitation

---

## 5. Multi-Modal Integration Analyses

### 5.1 Dopamine-Neural Coupling ✅ IMPLEMENTED
- **Dopamine-spike coupling**: ✅ `analyze_dopamine_spike_coupling`
  - Spike rate vs dopamine level - ✅ Implemented
  - Dopamine transients and spike responses - ✅ Implemented
  - Dopamine prediction of neural activity - ✅ Implemented
- **Dopamine-triggered firing**: ✅ `analyze_dopamine_triggered_firing`
  - Peri-event analysis of spikes around DA transients - ✅ Implemented
- **Dopamine modulation index**: ✅ `analyze_dopamine_modulation_index`
  - High vs low DA firing rate comparison - ✅ Implemented
- **Dopamine-LFP coupling**: ✅ `analyze_dopamine_lfp_coupling`
  - Dopamine and LFP power/phase - ✅ Implemented
  - Dopamine modulation of oscillations - ✅ Implemented
- **Temporal relationships**: ✅ Covered in coupling analyses
  - Dopamine leads vs follows neural activity
  - Cross-correlations at multiple lags

### 5.2 Movement-Neural Coupling ✅ IMPLEMENTED
- **DLC-neural alignment**: ✅ `calculate_movement_tuning`
  - Body part positions vs neural activity - ✅ Implemented
  - Movement kinematics vs firing patterns - ✅ Implemented
- **Spatial maps**: ✅ Covered in movement tuning
  - Neural activity as function of position in chamber
  - Port-specific activity patterns
- **Velocity fields**: ✅ Covered in movement tuning
  - Neural activity as function of velocity vector

### 5.3 Event-Neural Coupling ✅ IMPLEMENTED
- **Multi-event analysis**: ✅ `calculate_event_tuning`
  - How do multiple events (reward, lick, movement) combine? - ✅ Implemented
  - Non-linear interactions between events
- **Event sequences**: Partially covered in PETH analyses
  - Neural response to event sequences
  - Prediction of upcoming events

### 5.4 Cross-Modal Prediction
- **Neural → Behavior**: ✅ `analyze_choice_prediction`
- **Dopamine → Neural**: ✅ Covered in DA coupling analyses
- **Behavior → Neural**: ✅ Covered in movement tuning
- **Bidirectional**: Covered in coupling analyses

---

## 6. Temporal Dynamics and Sequence Analysis

### 6.1 Sequence Detection
- **SeqNMF**: Find repeating sequences in neural activity
  - Behavioral sequences
  - Neural sequences
  - Combined sequences
- **Temporal patterns**: Recurring temporal patterns across trials
- **Sequence compression**: Compress behavior into sequences

### 6.2 Replay Analysis
- **Offline replay**: Reactivation of sequences during rest
- **Forward vs reverse replay**: Direction of sequence replay
- **Replay during behavior**: Online replay during task performance
- **Replay and learning**: Relationship between replay and learning

### 6.3 Temporal Structure ✅ IMPLEMENTED
- **Autocorrelations**: ✅ `analyze_temporal_autocorrelation`
  - Temporal structure within single neurons - ✅ NEW
  - Rhythmicity detection
  - Decay time measurement
- **Cross-correlations**: ✅ `analyze_cross_correlation_pairs`
  - Temporal relationships between neurons - ✅ NEW
  - Functional connectivity
  - Synchrony measurement
- **Temporal clustering**: ✅ `analyze_temporal_clustering`
  - Neurons with similar temporal patterns - ✅ NEW
  - K-means clustering on firing patterns
- **Temporal hierarchies**: Multi-scale temporal structure

### 6.4 Dynamics Analysis ✅ PARTIALLY IMPLEMENTED
- **Phase space**: ✅ `analyze_phase_space_trajectories`
  - Trajectories in neural state space - ✅ NEW
  - Low-dimensional projections
- **Fixed points and attractors**: ✅ `analyze_phase_space_trajectories`
  - Stable states in dynamics - ✅ NEW
  - Slow regions identified
- **Bifurcations**: Changes in dynamics at condition switches
- **Lyapunov exponents**: Stability of dynamics

---

## 7. Spatial Organization and Topography

### 7.1 Medial-Lateral Organization ✅ IMPLEMENTED
- **Gradient analysis**: ✅ `analyze_medial_lateral_organization`
- **Functional boundaries**: Covered in gradient analysis
- **Spatial clustering**: ✅ `analyze_spatial_clustering`

### 7.2 Depth Organization ✅ IMPLEMENTED
- **Laminar structure**: ✅ `analyze_spatial_organization_depth`
- **Depth-behavior tuning**: ✅ `analyze_depth_tuning_by_behavior`
  - Correlates CW/CCW tuning with dorsal-ventral depth - ✅ NEW
- **CSD analysis**: Not implemented (requires LFP analysis)
- **Depth tuning**: ✅ Covered in depth organization

### 7.3 Multi-Shank Analysis ✅ IMPLEMENTED
- **Shank-specific functions**: ✅ `analyze_multi_shank_interactions`
- **Cross-shank interactions**: ✅ Synchrony and interactions implemented
- **Spatial patterns**: ✅ Covered in multi-shank analysis

### 7.4 Topographic Maps
- **Movement direction maps**: Partially covered in directional tuning
- **Reward location maps**: Covered in spatial analyses
- **Behavioral state maps**: Covered in context encoding

---

## 8. Decision-Making and Strategy Switching

### 8.1 Decision Variables
- **Accumulation models**: Evidence accumulation for decisions (--decision_accumulation)
- **Decision boundaries**: When does mouse commit to decision?
- **Decision confidence**: Neural correlates of decision confidence (--decision_confidence)

### 8.2 Strategy Encoding
- **Strategy representation**: How is current strategy encoded?
- **Strategy prediction**: Can we predict strategy from neural activity?
- **Strategy maintenance**: How is strategy maintained?

### 8.3 Switch Detection
- **Neural switch signals**: What signals strategy switch?
- **Switch prediction**: Can we predict switches before they happen?
- **Switch-related activity**: Activity changes around switches

### 8.4 Adaptation Dynamics ✅ IMPLEMENTED
- **Learning curves**: ✅ `analyze_learning_curves`
  - How does performance improve after switch? - ✅ NEW
  - Tracks accuracy and reaction time over trials
  - Measures learning rate
- **Neural adaptation**: How do neurons adapt to new strategy? - Partially covered
- **Error correction**: Neural signals for error detection and correction - ✅ `analyze_error_detection`

### 8.5 Perseveration Analysis ✅ IMPLEMENTED
- **Perseveration signals**: ✅ `analyze_perseveration_signals`
- **Perseveration vs adaptation**: ✅ Covered in trajectory consistency
- **Breaking perseveration**: What breaks perseveration? - Covered
- **Trajectory consistency**: ✅ `analyze_trajectory_consistency`
  - Strategy adherence measurement - ✅ NEW
  - Behavioral reliability - ✅ NEW
  - Time to criterion - ✅ NEW

---

## 9. Dopamine-Neural Interactions

### 9.1 Dopamine Signals ✅ IMPLEMENTED
- **Reward prediction error**: ✅ `analyze_reward_prediction_error`
- **Dopamine transients**: ✅ Covered in DA coupling analyses
- **Dopamine baseline**: ✅ Covered in DA modulation index

### 9.2 Dopamine Effects on Neural Activity ✅ IMPLEMENTED
- **Dopamine modulation**: ✅ `analyze_dopamine_modulation_index`
  - How does dopamine modulate neural activity? - ✅ Implemented
  - High vs low DA firing rate comparison - ✅ Implemented
- **D1 vs D2 effects**: Not implemented (requires cell-type labeling)
- **Dopamine-dependent plasticity**: Not implemented

### 9.3 Neural Effects on Dopamine ✅ IMPLEMENTED
- **Neural → Dopamine**: ✅ Covered in DA coupling analyses
- **Feedback loops**: ✅ Covered in bidirectional coupling
- **Bidirectional coupling**: ✅ `analyze_dopamine_spike_coupling`, `analyze_dopamine_lfp_coupling`

### 9.4 Dopamine and Behavior
- **Dopamine and movement**: Partially covered in DA-triggered analyses
- **Dopamine and decisions**: Partially covered in reward analyses
- **Dopamine and learning**: Partially covered in RPE

---

## 10. Advanced Statistical and Machine Learning Approaches

### 10.1 Dimensionality Reduction ✅ IMPLEMENTED
- **PCA**: ✅ Principal component analysis of population activity (--dimensionality)
- **ICA**: ✅ `analyze_ica_decomposition` - ✅ NEW
  - Independent component analysis
  - Separates mixed neural signals
- **t-SNE/UMAP**: ✅ Non-linear dimensionality reduction (--population_manifolds)
- **Autoencoders**: Deep learning dimensionality reduction

### 10.2 Clustering and Classification ✅ IMPLEMENTED
- **Neural clustering**: ✅ `analyze_neural_clustering` - ✅ NEW
  - Cluster neurons by activity patterns
  - K-means and hierarchical methods
- **Behavioral clustering**: Cluster behaviors
- **State classification**: Classify behavioral/neural states

### 10.3 Regression and Prediction
- **GLM**: Generalized linear models for spike prediction
- **Neural networks**: Deep learning for prediction
- **Gaussian processes**: Probabilistic regression

### 10.4 Causal Inference
- **Granger causality**: Directed relationships
- **Transfer entropy**: Information flow
- **Intervention analysis**: Causal effects

### 10.5 Bayesian Methods
- **Bayesian decoding**: Decode behavior from neural activity
- **Bayesian state estimation**: Estimate hidden states
- **Bayesian model comparison**: Compare different models

### 10.6 Information Theory
- **Mutual information**: Information between variables
- **Transfer entropy**: Directed information
- **Integrated information**: Information integration

---

## Data Requirements

For each analysis, ensure you have:
- **Aligned timestamps**: All data streams aligned to common timebase
- **Quality-filtered units**: Only good/MUA units from BombCell
- **Cell type labels**: MSN vs FSI classification
- **Behavioral events**: All events extracted and aligned
- **DLC data**: Body part coordinates and derived kinematics
- **Dopamine data**: dFF signals aligned to neural data
- **Spatial information**: Unit locations on probe

---

## Tools and Resources

- **B-SOID**: https://github.com/YttriLab/B-SOID
- **VAME**: https://github.com/LINCellularNeuroscience/VAME
- **DPAD**: Check for latest implementation
- **SeqNMF**: https://github.com/FeeLab/seqNMF
- **Rastermap**: Already integrated in pipeline
- **SpikeInterface**: Already integrated
- **MATLAB**: BombCell, TDT processing

---

## Notes

- Many analyses can be combined (e.g., dopamine-neural coupling during specific behaviors)
- Start with simple analyses and build complexity
- Validate findings across multiple sessions/animals
- Consider both population-level and single-unit analyses
- Document all analysis parameters for reproducibility

---

## 11. Functional Characterization & Integration ✅ IMPLEMENTED

### 11.1 Functional Tuning Matrix ✅ IMPLEMENTED
- **Comprehensive characterize**: ✅ `analyze_functional_tuning_matrix` - ✅ NEW
  - Create a large "Neuron vs. Feature" matrix (N_neurons x N_features).
- **Features to include**:
  - Movement Tuning (Speed, Acceleration)
  - Directional Tuning (CW vs CCW preference)
  - Decision Point Responses (Departure times)
  - Reward Responses (Expected vs Unexpected)
  - Behavioral Switch Responses (Success point tuning)
- **Downstream Analysis**: ✅ Implemented
  - Cluster neurons based on their functional profiles to identify "Cell Types" (e.g., "Reward-selective MSN", "Movement-related FSI").
  - Study spatial organization of these functional clusters.

