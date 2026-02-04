"""
Theoretical models for LFP propagation and microcircuit dynamics.

This package contains implementations of computational models described in Chapter 2
of the post-analysis ideas, enhanced with insights from Pouzzner's "Basal Ganglia 
Mediated Synchronization" theory.

Modules:
    diffusion_eq_model: Delayed diffusion equation with conduction delays
    phase_coherence_gating: Phase-coherence gating function and phase-field models
    spike_lfp_coupling: Coupled spike-LFP models with phase filtering
    striatal_microcircuit: MSN-FSI interactions with dopamine-driven resonance tuning
    attractor_energy_landscape: Uncertainty-driven exploration via attractor dynamics
"""

from .diffusion_eq_model import DelayedDiffusionModel
from .phase_coherence_gating import PhaseCoherenceGatingModel, PhaseFieldModel
from .spike_lfp_coupling import SpikeLFPCouplingModel
from .striatal_microcircuit import StrialMicrocircuitModel
from .attractor_energy_landscape import AttractorEnergyLandscapeModel, AttractorModelIntegration
from .hierarchical_state_analysis import HierarchicalStateAnalysis

__all__ = [
    'DelayedDiffusionModel',
    'PhaseCoherenceGatingModel',
    'PhaseFieldModel',
    'SpikeLFPCouplingModel',
    'StrialMicrocircuitModel',
    'AttractorEnergyLandscapeModel',
    'AttractorModelIntegration',
    'HierarchicalStateAnalysis'
]
