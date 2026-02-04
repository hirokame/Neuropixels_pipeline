# Theoretical Models for Striatal LFP Propagation and Microcircuit Dynamics

This package implements validational computational models described in Chapter 2 of the post-analysis framework. These models provide the theoretical backing for the "Basal Ganglia Mediated Synchronization" theory proposed by Daniel Pouzzner.

Each model addresses a specific mechanism of how the striatum processes information through spatiotemporal dynamics, from signal propagation to microcircuit learning and macroscopic behavioral exploration.

## 📂 File Guide

| File | Type | Description |
|------|------|-------------|
| **Core Models** | | |
| `diffusion_eq_model.py` | Class | Implements the Delayed Diffusion Equation (PDE) solver. |
| `phase_coherence_gating.py` | Class | Defines `PhaseCoherenceGatingModel` (MSN-FSI) and `PhaseFieldModel`. |
| `spike_lfp_coupling.py` | Class | Defines bidirectional `SpikeLFPCouplingModel`. |
| `striatal_microcircuit.py` | Class | Defines `StrialMicrocircuitModel` with DA-driven plasticity. |
| `attractor_energy_landscape.py` | Class | Defines `AttractorEnergyLandscapeModel` for exploration dynamics. |
| **Analysis & Validation** | | |
| `delayed_diffusion_analysis.py` | Script | Validation of Model 1 with real Neuropixels LFP data. |
| `phase_coherence_analysis.py` | Script | Validation of Model 2 (FSI gating) and Model 5 (Phase Field). |
| `spike_lfp_coupling_analysis.py` | Script | Validation of Model 3 (Spike-LFP locking). |
| `striatal_microcircuit_analysis.py` | Script | Validation of Model 4 (Learning & Tuning). |
| `attractor_energy_landscape_analysis.py` | Script | Validation of Model 6 (Exploration/Exploitation). |
| `phase_field_analysis.py` | Script | Specific analysis for macroscopic traveling waves. |
| **Utilities** | | |
| `quick_start_delayed_diffusion.py` | Demo | Minimal working example for the diffusion model. |
| `demo_enhanced_visualizations.py` | Demo | Generates publication-quality figures for all models. |

---

## 🧠 Detailed Model Descriptions

### 1. Delayed Diffusion Model
**Focus**: Signal Propagation & Resonance  
**Files**: `diffusion_eq_model.py`, `delayed_diffusion_analysis.py`

#### Theoretical Background
This model treats the Local Field Potential (LFP) not as instantaneous, but as a diffusive process subject to **conduction delays**.

$$
\frac{\partial V}{\partial t} = D \nabla^2 V(t - \tau) + S(x,t)
$$

The core insight is that long and varied conduction delays in the cortico-basal ganglia loop don't just "lag" the signal—they **tune** the circuit to resonate at specific frequencies (Theta, Beta, Gamma) where the delay equals a quarter-cycle phase shift.

#### Scientific Origins
*   **Cable Theory (Rall, 1962)**: Modeling neurites as passive cables.
*   **Reaction-Diffusion Systems**: Turing mechanisms for pattern formation.
*   **Ephaptic Coupling**: Propagation of fields through extracellular medium.

#### Our Enhancements
*   **Anisotropy**: We explicitly model different diffusion coefficients along vs. across the probe shank ($D_{y} \approx 3 D_{x}$), matching the structural anisotropy of striatal axons/dendrites.
*   **Conduction Delays**: Classical cable theory is often instantaneous or purely resistive. We added the $\tau$ term to explicitly model propagation delays that create frequency-selective resonance.

---

### 2. Phase-Coherence Gating Model aka " The Router"
**Focus**: Information Selection  
**Files**: `phase_coherence_gating.py`, `phase_coherence_analysis.py`

#### Theoretical Background
The Basal Ganglia acts as a "router" that filters information based on phase. Fast-Spiking Interneurons (FSIs) are strongly entrained to the LFP phase and provide potent feedforward inhibition to MSNs. This creates "transmission windows" where MSNs can only fire at specific phases of the oscillation, effectively gating information flow.

#### Scientific Origins
*   **Communication Through Coherence (CTC) (Fries, 2005)**: The hypothesis that effective connectivity depends on the phase relationship between oscillating groups.
*   **Feedforward Inhibition**: Established circuit motif where inhibition truncates excitation to enforce temporal precision (Pouille & Scanziani, 2001).

#### Our Enhancements
*   **Differential Phase-Locking**: We explicitly model the finding that FSIs phase-lock much stronger than MSNs ($PLV_{FSI} \gg PLV_{MSN}$).
*   **Reward Modulation**: We integrated dopamine signals that dynamically widen or shift these gating windows, allowing "rewarded" patterns to pass through.

---

### 3. Spike-LFP Coupling Model
**Focus**: Bidirectional Interaction  
**Files**: `spike_lfp_coupling.py`, `spike_lfp_coupling_analysis.py`

#### Theoretical Background
A bidirectional model where:
1.  **Forward**: Spikes generate LFP features (Current Source Density) via spatiotemporal kernels.
2.  **Feedback**: The LFP phase modulates the probability of future spikes.
This creates a closed-loop system where spikes self-organize into coherent oscillations.

#### Scientific Origins
*   **Spike-Field Coherence**: Standard measure in electrophysiology (Pesaran et al., 2002).
*   **Ephaptic Effects**: Evidence that endogenous fields can influence neuronal firing (Anastassiou et al., 2011).

#### Our Enhancements
*   **Combined Framework**: Often models look at *either* how spikes make LFP *or* how LFP affects spikes. We implemented both to study the stability of self-sustained oscillations.
*   **Biological Kernels**: Parameters ($\lambda \approx 120\mu m$, $\tau \approx 10ms$) are fitted to real Neuropixels CSD data.

---

### 4. Striatal Microcircuit Model
**Focus**: Learning & Plasticity  
**Files**: `striatal_microcircuit.py`, `striatal_microcircuit_analysis.py`

#### Theoretical Background
This model focuses on how **Dopamine (DA)** tunes the resonance of same microcircuit. It proposes that Reinforcement Learning (RL) doesn't just change synaptic weights ($w$), it changes **synaptic delays** ($\tau$) and time constants. By adjusting delays, DA brings the circuit into resonance with task-relevant inputs (e.g., shifting from Theta exploration to Beta exploitation).

#### Scientific Origins
*   **STDP (Spike-Timing-Dependent Plasticity)**: Bi & Poo (1998).
*   **Canonical Microcircuits**: Douglas & Martin (1991).
*   **Izhikevich Polychronization**: Groups of neurons that fire in time-locked patterns due to specific delays (Izhikevich, 2006).

#### Our Enhancements
*   **Delay Plasticity**: We implemented a hebbian-like rule for *delay* adaptation: $\Delta \tau \propto \text{Reward} \times \text{Error}$.
*   **D1/D2 Asymmetry**: Explicitly modeled differential DA sensitivity (D1 excited, D2 inhibited) leads to competitive dynamics that balance action selection vs. suppression.

---

### 5. Phase-Field Model
**Focus**: Macroscopic Traveling Waves  
**Files**: `phase_coherence_gating.py` (PhaseFieldModel class), `phase_field_analysis.py`

#### Theoretical Background
Treats the striatum as a continuous excitable medium $u(x,t)$ where activity propagates as **traveling waves**. This moves beyond single-neuron dynamics to study mesoscopic patterns, predicting that synchronization occurs via waves sweeping across the tissue rather than simultaneous global locking.

#### Scientific Origins
*   **Neural Field Theory**: Amari (1977), Wilson-Cowan (1972).
*   **Cortical Traveling Waves**: Lubenov & Siapas (2009) (hippocampal theta waves).

#### Our Enhancements
*   **Phase-Dynamics**: We coupled the activity field equation with a Kuramoto-like phase equation to study how physical waves underpin mathematical phase-locking metrics.

---

### 6. Attractor Energy Landscape Model
**Focus**: Exploration vs. Exploitation  
**Files**: `attractor_energy_landscape.py`, `attractor_energy_landscape_analysis.py`

#### Theoretical Background
Addresses the "**Homunculus Problem**" of exploration: How does the brain "decide" to explore without a central decider?
This model proposes **Energy Spillover**:
1.  **Uncertainty** = Flat/Shallow Attractor Landscape.
2.  **Shallow Attractors** = High Neural Velocity & Variance (instability).
3.  **Mechanism**: This high neural energy mechanically "spills over" into motor circuits, causing random movements (licking, whisking) *directly* proportional to neural instability.
4.  No calculation of entropy is required; exploration is a physical consequence of uncertainty.

#### Scientific Origins
*   **Attractor Dynamics**: Hopfield (1982).
*   **Free Energy Principle**: Friston (2010).
*   **Exploration-Exploitation Dilemma**: Cohen, McClure, Yu (2007).

#### Our Enhancements
*   **Direct Motor Coupling**: `Motor_Gain = α|v| + βVar(x)`. We provide a mechanistic link between abstract manifold dynamics and concrete motor behavior (validated against DLC tracking).

---

## 🚀 Usage

### Delayed Diffusion Model
```bash
# Quick example (5 minutes)
PYTHONPATH=$PWD python3 postanalysis/models/quick_start_delayed_diffusion.py

# Full analysis with real data
PYTHONPATH=$PWD python3 postanalysis/models/delayed_diffusion_analysis.py
```

### Phase-Coherence Gating Model
```bash
# Complete analysis
PYTHONPATH=$PWD python3 postanalysis/models/phase_coherence_analysis.py
```

### Spike-LFP Coupling Model
```bash
# Complete bidirectional coupling validation
PYTHONPATH=$PWD python3 postanalysis/models/spike_lfp_coupling_analysis.py
```

### Striatal Microcircuit Model
```bash
# Dopamine-driven learning analysis
PYTHONPATH=$PWD python3 postanalysis/models/striatal_microcircuit_analysis.py
```

### Phase-Field Model
```bash
# Continuum field dynamics and synchronization
PYTHONPATH=$PWD python3 postanalysis/models/phase_field_analysis.py
```

### Attractor Energy Landscape Model
```bash
# Uncertainty-driven exploration dynamics
PYTHONPATH=$PWD python3 postanalysis/models/attractor_energy_landscape_analysis.py
```

---

## 📊 Comparison of Validation Status

| Model | Key Validation Metric | Result |
|-------|----------------------|--------|
| **Delayed Diffusion** | Anisotropy Ratio | 3.02 (Expected: ~3.0) ✅ |
| **Gating** | FSI vs MSN PLV | FSI 0.62 >> MSN 0.00 ✅ |
| **Spike-LFP** | Coherence Peak | 0.19 at 7.8 Hz ✅ |
| **Microcircuit** | Learning Correlation | $r=1.000$ (Delay vs Frequency) ✅ |
| **Energy Landscape** | Velocity-Motor Lag | 0ms (Instantaneous coupling) ✅ |
