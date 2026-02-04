# Future Analysis Roadmap

1. Behavioral Motif Analysis
   - **Goal**: Identify discrete behavioral motifs from continuous tracking data and map them to neural activity.
   - **Roadmap**:
     - Implement B-SOID or VAME to unsupervisedly cluster DLC data into behavioral motifs (e.g., "grooming", "turning", "rearing").
     - Train Hidden Markov Models (HMM) or DPAD to model transitions between behavioral motifs.
     - Correlate discovered states with distinct neural population vectors.

2. Neural Replay and Sequence Analysis
   - **Goal**: Detect offline reactivation of task-related activity patterns (replay) and their role in learning/decision making.
   - **Roadmap**:
     - Identify replay events in LFP/spikes data during rest/inter-trial intervals.
     - quantifying "replay" by correlating spontaneous activity sequences with task-evoked templates (e.g. using rank-order or Bayesian decoding).
     - Compare forward vs. reverse replay frequency during learning vs. expert performance.

3. Temporal Dynamics on Neural Activity Manifold
   - **Goal**: Characterize the stability and non-linear dynamics of striatal population activity.
   - **Roadmap**:
     - Compute Lyapunov exponents to measure sensitivity to initial conditions (chaos vs stability).
     - Detect bifurcation points in neural dynamics that correspond to behavioral strategy switches.
     - Analyze "metastable" states where activity dwells before transitioning (attractor ruins).

4. Dopamine-Dependent Plasticity & Cell-Type Specifics
   - **Goal**: Infer plasticity rules and differentiate D1/D2 pathway contributions without genetic labeling.
   - **Roadmap**:
     - Classify putative D1 vs D2 neurons based on firing response to dopamine transients (excitation vs inhibition).
     - Quantify "learning rules" by correlating spike-timing-dependent plasticity (STDP) windows with dopamine arrival times.
     - Compare plasticity rates between putative D1 and D2 populations during reversal learning.

5. Causal and Information-Theoretic Analysis
   - **Goal**: Determine directionality of information flow and causal links between neurons/variables.
   - **Roadmap**:
     - Compute Transfer Entropy or Granger Causality between neuron pairs or between LFP bands and spikes.
     - Estimate "Integrated Information" (Phi) to measure network complexity/integration.
     - Use causal intervention models (Do-calculus on graphical models) to predict effects of hypothetical perturbations.

6. Generative Modeling (Autoencoders & GPs)
   - **Goal**: Learn low-dimensional latent representations that generate the observed high-dimensional neural data.
   - **Roadmap**:
     - Train Variational Autoencoders (VAEs) or Sequential VAEs (LFADS) to denoise single-trial firing rates.
     - Use Gaussian Process Factor Analysis (GPFA) to extract smooth latent trajectories.
     - Perform "latent arithmetic" to see if behavioral variables (speed, choice) correspond to specific axes in the latent space.

7. 

