"""
Hierarchical Neuromotor Model (HNM).

Models behavior as a hierarchy of timescales:
1. Micro-States (Sub-second): Movement motifs (gait, turns) derived from DLC/VAME.
2. Macro-States (Supra-second): Behavioral modes (Engaged, Exploratory, Disengaged).

Logic:
    - Low-level motor primitives are conditioned by the high-level state.
    - Transitions between high-level states are modulated by Dopamine and Reward history.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
    print("Warning: hmmlearn not found. Macro-state analysis will default to GMM (no temporal dynamics).")
    print("To fix: pip install hmmlearn")

class HierarchicalNeuromotorModel:
    """
    Hierarchical model linking sub-second motor motifs to longer behavioral states.
    """
    
    def __init__(self, 
                 n_micro_states: int = 10,   # Number of movement motifs (e.g., VAME clusters)
                 n_macro_states: int = 3,    # Number of behavioral modes (e.g., Engaged, Explore, Idle)
                 sampling_rate: float = 30.0): # DLC sampling rate
        
        self.n_micro = n_micro_states
        self.n_macro = n_macro_states
        self.fs = sampling_rate
        
        # Micro-scale model (GMM or placeholder for VAME)
        self.micro_model = GaussianMixture(n_components=n_micro_states, covariance_type='full')
        
        # Macro-scale model (HMM)
        # Observations: [Dopamine, RewardRate, MicroStateEntropy, Velocity]
        if hmm is not None:
            self.macro_model = hmm.GaussianHMM(n_components=n_macro_states, covariance_type="diag", n_iter=100)
        else:
            self.macro_model = None

        self.micro_labels = None
        self.macro_labels = None
        
    def extract_kinematic_features(self, dlc_df: pd.DataFrame) -> np.ndarray:
        """
        Extract 'Eigen-movements' from DLC data using PCA.
        
        Process:
        1. Identify all X/Y columns.
        2. Impute NaNs (linear interpolation).
        3. Perform PCA on the posture matrix (Time x BodyParts*2).
        4. Compute velocity of the top PCs (eigen-movements).
        """
        print("    Extracting kinematic features via PCA...")
        
        # 1. Gather all coordinate data
        coords = []
        
        if isinstance(dlc_df.columns, pd.MultiIndex):
            # DLC standard: (scorer, bodypart, coords)
            # We want all 'x' and 'y' columns
            for col in dlc_df.columns:
                if col[-1] in ['x', 'y']:
                    coords.append(dlc_df[col].values)
        else:
            # Flat CSV: assume valid columns are numerical or end in x/y
            # Heuristic: Take all columns as coords
            coords = [dlc_df[c].values for c in dlc_df.columns if dlc_df[c].dtype in [float, np.float32, np.float64]]
            
        if not coords:
            raise ValueError("No coordinate columns found in DLC dataframe.")
            
        X_pose = np.column_stack(coords) # (Time, n_bodyparts*2)
        
        # 2. Impute NaNs
        # Simple forward/backward fill or interpolation
        df_pose = pd.DataFrame(X_pose)
        df_pose = df_pose.interpolate(method='linear', limit_direction='both')
        X_pose = df_pose.values
        
        # 3. PCA on Posture (Eigen-poses)
        # We want to capture the main modes of body configuration
        n_components = min(5, X_pose.shape[1])
        pca = PCA(n_components=n_components)
        X_eigen = pca.fit_transform(StandardScaler().fit_transform(X_pose))
        print(f"      PCA: Top {n_components} components explain {np.sum(pca.explained_variance_ratio_):.2%} of pose variance.")
        
        # 4. Compute Derivatives (Eigen-movements)
        # We care about how these postures CHANGE (velocity)
        velocities = np.gradient(X_eigen, axis=0)
        
        # Also include total energy (speed of all parts)? 
        # For now, just the velocities of the eigen-poses is a good 'movement state' descriptor.
        
        # Stack Eigen-Pose and Eigen-Velocity? 
        # Often just velocity is better for 'motifs' (static pose shouldn't define a dynamic motif ideally, 
        # but sometimes posture matters e.g. rearing vs sitting).
        # Let's use both: Abstract Pose + Abstract Velocity
        X_features = np.column_stack([X_eigen, velocities])
        
        # Handle start/end artifacts from gradient
        X_features = np.nan_to_num(X_features)
        
        return StandardScaler().fit_transform(X_features)

    def fit_micro_dynamics(self, 
                          dlc_data: Union[pd.DataFrame, np.ndarray], 
                          use_vame_labels: bool = False):
        """
        Step 1: Define the sub-second 'alphabet' of movement.
        If VAME data is provided, use it directly. Otherwise, cluster kinematics.
        """
        if use_vame_labels and isinstance(dlc_data, np.ndarray):
            # User provided pre-calculated VAME motifs
            self.micro_labels = dlc_data
            self.n_micro = len(np.unique(dlc_data))
            print(f"Loaded {self.n_micro} VAME motifs.")
            return

        print("Extracting kinematic features...")
        if isinstance(dlc_data, pd.DataFrame):
            X = self.extract_kinematic_features(dlc_data)
        else:
            X = dlc_data

        print(f"Clustering micro-states (n={self.n_micro}) via GMM...")
        self.micro_labels = self.micro_model.fit_predict(X)
        
    def fit_macro_dynamics(self, 
                          dopamine_signal: np.ndarray, 
                          neural_features: np.ndarray,
                          window_size_sec: float = 2.0):
        """
        Step 2: Define longer behavioral states based on context.
        
        Inputs:
            dopamine_signal: Aligned DA vector (e.g., dFF)
            neural_features: Aligned neural features (e.g., LFP power, PC1 of pop rate)
            window_size_sec: Window to aggregate micro-states
        """
        if self.micro_labels is None:
            raise ValueError("Must fit micro-dynamics first.")

        # 1. Aggregate Micro-States into "Motif Distributions" (Bag of Words)
        window_samples = int(window_size_sec * self.fs)
        n_samples = len(dopamine_signal)
        n_windows = n_samples // window_samples
        
        macro_features = []
        
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            
            # A. Micro-state statistics (Distribution of motifs in this window)
            window_motifs = self.micro_labels[start:end]
            motif_counts = np.bincount(window_motifs, minlength=self.n_micro)
            motif_dist = motif_counts / np.sum(motif_counts)
            
            # B. Entropy (Is behavior stereotyped or chaotic?)
            entropy = -np.sum(motif_dist * np.log(motif_dist + 1e-9))
            
            # C. Contextual signals (Mean DA, Neural variance)
            mean_da = np.mean(dopamine_signal[start:end])
            neural_var = np.var(neural_features[start:end])
            
            # Feature vector for Macro-HMM
            # [Mean_DA, Entropy, Neural_Var, ...Motif_Distribution...]
            feat_vec = np.concatenate([[mean_da, entropy, neural_var], motif_dist])
            macro_features.append(feat_vec)
            
        X_macro = np.vstack(macro_features)
        X_macro = StandardScaler().fit_transform(X_macro)
        
        # 2. Fit HMM on these "behavioral windows"
        if self.macro_model:
            print(f"Fitting Macro-HMM (n={self.n_macro}) on behavioral windows...")
            self.macro_labels = self.macro_model.fit_predict(X_macro)
        else:
            print("HMM not available, using GMM for macro states.")
            self.macro_labels = GaussianMixture(n_components=self.n_macro).fit_predict(X_macro)
            
        # Upsample macro labels back to original resolution for plotting
        self.macro_labels_upsampled = np.repeat(self.macro_labels, window_samples)
        
        # Trim if rounding caused length mismatch
        target_len = len(dopamine_signal)
        if len(self.macro_labels_upsampled) > target_len:
            self.macro_labels_upsampled = self.macro_labels_upsampled[:target_len]
        elif len(self.macro_labels_upsampled) < target_len:
            pad = np.full(target_len - len(self.macro_labels_upsampled), self.macro_labels_upsampled[-1])
            self.macro_labels_upsampled = np.concatenate([self.macro_labels_upsampled, pad])

    def plot_hierarchy(self, dopamine: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize the interaction between Dopamine, Macro-States, and Micro-States.
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        t = np.arange(len(dopamine)) / self.fs
        
        # 1. Macro State (Background color) & Dopamine
        ax = axes[0]
        # Create colored background for states
        ax.imshow(self.macro_labels_upsampled[np.newaxis, :], aspect='auto', 
                  extent=[t[0], t[-1], np.min(dopamine), np.max(dopamine)], 
                  cmap='Pastel1', alpha=0.5)
        ax.plot(t, dopamine, 'k', linewidth=1, label='Dopamine (dFF)')
        ax.set_ylabel('DA & Macro State')
        ax.legend(loc='upper right')
        ax.set_title('Level 2: Behavioral Modes (Influenced by Dopamine)')
        
        # 2. Micro State Raster
        ax = axes[1]
        # Scatter plot of micro states
        ax.scatter(t, self.micro_labels, c=self.micro_labels, cmap='tab20', s=1, marker='|')
        ax.set_ylabel('Micro State ID')
        ax.set_ylim(0, self.n_micro)
        ax.set_title('Level 1: Sub-second Motor Motifs (e.g., Gait, Turn)')
        
        # 3. State Transition Probability (Dynamic)
        # Calculate how "stable" the micro-states are in a rolling window
        ax = axes[2]
        rolling_stability = (self.micro_labels[:-1] == self.micro_labels[1:]).astype(float)
        # Smooth it
        kernel = np.ones(int(self.fs)) / self.fs
        stability_smooth = np.convolve(rolling_stability, kernel, mode='same')
        
        # Pad to match length
        stability_plot = np.pad(stability_smooth, (0, 1), 'edge')
        
        ax.plot(t, stability_plot, 'b-')
        ax.set_ylabel('Motor Stability')
        ax.set_xlabel('Time (s)')
        ax.set_title('Interaction: Motor Stability (High during Engagement?)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            
        return fig