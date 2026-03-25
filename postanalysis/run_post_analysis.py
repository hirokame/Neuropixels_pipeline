"""
Main post-analysis pipeline runner.

This script provides a command-line interface to run various post-analysis
functions on loaded data. It uses the data_loader module to find and load
all necessary data files.

Usage:
    python run_post_analysis.py --mouse 1818 --day 09182025 --analysis peth
    python run_post_analysis.py --mouse 1818 --day 2025-09-18 --analysis movement_tuning
"""

import argparse
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path to import data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import (
    load_session_data,
    validate_data_paths,
    print_data_summary,
    DataPaths
)
from analyses_population import (
    calculate_event_tuning, 
    calculate_movement_tuning,
    calculate_lfp_peth,
    calculate_dopamine_peth,

    analyze_behavioral_switch_response,
    analyze_port_to_port_trajectories,
    analyze_strategy_encoding,
    analyze_directional_tuning,
    analyze_context_dependent_encoding,
    analyze_trajectory_consistency,
    analyze_spatial_rate_maps,

    analyze_outcome_encoding,
    analyze_reward_magnitude_encoding,
    analyze_reward_history,
    analyze_history_dependence_glm,
    predict_reaction_time_multimodal,
    analyze_choice_prediction,
    analyze_population_statistics,
    analyze_population_manifolds,
    analyze_population_trajectories_by_direction,
    analyze_phase_space_trajectories,
    analyze_ica_decomposition)
from analyses_lfp import(
    analyze_lfp_movement_power,
    # analyze_theta_oscillations,
    # analyze_phase_amplitude_coupling,
    # analyze_cross_frequency_coupling,
    # analyze_spike_phase_locking,
    # analyze_temporal_autocorrelation,
    # analyze_cross_correlation_pairs,
    # analyze_bursting_behavior,
    # analyze_isi_distribution,
    # analyze_spike_pattern_motifs,
    # analyze_wave_propagations,
    # analyze_multi_scale_temporal_encoding,
    # analyze_rank_order_coding,
    # analyze_temporal_clustering,

    # analyze_dopamine_spike_coupling,
    # analyze_dopamine_triggered_firing,
    # analyze_dopamine_modulation_index,
    # analyze_dopamine_lfp_coupling,
    # analyze_dopamine_phase_locking_relationship,
    # analyze_spatial_organization_depth,
    # analyze_depth_tuning_by_behavior,
    # analyze_spatial_clustering,
    # analyze_medial_lateral_organization,
    # analyze_multi_shank_interactions,
    # analyze_neural_clustering,
    # analyze_functional_tuning_matrix,
    # compare_cell_types,
    # analyze_mutual_information,
    # analyze_predictive_decoding,
    # analyze_ili_by_port,
)
from analyses_openfield import analyze_of_tier1_behavior

def run_event_tuning_analysis(paths: DataPaths):
    """Run peri-event time histogram analysis."""
    print("Running PETH analysis...")
    calculate_event_tuning(paths, event_file_type='reward')
    calculate_event_tuning(paths, event_file_type='reward_first')
    calculate_event_tuning(paths, event_file_type='reward_second')
    calculate_event_tuning(paths, event_file_type='licking')
    calculate_event_tuning(paths, event_file_type='licking_bout_start')

def run_lfp_peth_analysis(paths: DataPaths):
    """Run LFP power PETH analysis."""
    print("Running LFP PETH analysis...")
    calculate_lfp_peth(paths, event_file_type='reward_first')
    calculate_lfp_peth(paths, event_file_type='reward_second')
    calculate_lfp_peth(paths, event_file_type='licking_bout_start')
    calculate_lfp_peth(paths, event_file_type='movement_onset')

def run_dopamine_peth_analysis(paths: DataPaths):
    """Run dopamine PETH analysis."""
    print("Running Dopamine PETH analysis...")
    calculate_dopamine_peth(paths, event_file_type='reward_first')
    calculate_dopamine_peth(paths, event_file_type='reward_second')
    calculate_dopamine_peth(paths, event_file_type='licking_bout_start')
    calculate_dopamine_peth(paths, event_file_type='movement_onset')

def run_movement_tuning(paths: DataPaths):
    """Run movement-velocity tuning analysis."""
    print("Running movement tuning analysis...")
    calculate_movement_tuning(paths)

def run_behavioral_switch_analysis(paths: DataPaths):
    """Run behavioral switch analysis."""
    print("Running behavioral switch analysis...")
    analyze_behavioral_switch_response(paths)

def run_port_to_port_analysis(paths: DataPaths):
    """Run port-to-port trajectory analysis."""
    print("Running port-to-port trajectory analysis...")
    analyze_port_to_port_trajectories(paths)

def run_strategy_encoding_analysis(paths: DataPaths):
    """Run strategy encoding analysis."""
    print("Running strategy encoding analysis...")
    analyze_strategy_encoding(paths)

def run_directional_tuning_analysis(paths: DataPaths):
    """Run directional tuning analysis."""
    print("Running directional tuning analysis...")
    analyze_directional_tuning(paths)

def run_context_dependent_encoding_analysis(paths: DataPaths):
    """Run context-dependent encoding analysis."""
    print("Running context-dependent encoding analysis...")
    analyze_context_dependent_encoding(paths)

def run_trajectory_consistency_analysis(paths: DataPaths):
    """Run trajectory consistency analysis."""
    print("Running trajectory consistency analysis...")
    analyze_trajectory_consistency(paths)

def run_spatial_rate_maps_analysis(paths: DataPaths):
    """Run spatial rate maps analysis."""
    print("Running spatial rate maps analysis...")
    analyze_spatial_rate_maps(paths)

def run_pre_switch_activity_analysis(paths: DataPaths):
    """Run pre-switch activity analysis."""
    print("Running pre-switch activity analysis...")
    analyze_pre_switch_activity(paths)

def run_outcome_encoding_analysis(paths: DataPaths):
    """Run outcome encoding analysis (replacing RPE)."""
    print("Running Outcome Encoding analysis (Reward vs Error/Omission)...")
    analyze_outcome_encoding(paths)

def run_reward_magnitude_analysis(paths: DataPaths):
    """Run reward magnitude encoding analysis."""
    print("Running reward magnitude analysis...")
    analyze_reward_magnitude_encoding(paths)

def run_reward_history_analysis(paths: DataPaths):
    """Run reward history analysis."""
    print("Running reward history analysis...")
    analyze_reward_history(paths)

def run_history_glm_analysis(paths: DataPaths):
    """Run history dependence GLM analysis."""
    print("Running history dependence GLM analysis...")
    analyze_history_dependence_glm(paths, n_back=5)

def run_predict_reaction_time_multimodal(paths: DataPaths):
    """Run multimodal reaction time prediction analysis."""
    print("Running multimodal reaction time prediction analysis...")
    predict_reaction_time_multimodal(paths)

def run_choice_prediction_analysis(paths: DataPaths):
    """Run choice prediction analysis."""
    print("Running choice prediction analysis...")
    analyze_choice_prediction(paths)

def run_population_statistics_analysis(paths: DataPaths):
    """Run population statistics across all analyses."""
    print("Running population statistics analysis...")
    analyze_population_statistics(paths)

def run_population_manifolds_analysis(paths: DataPaths):
    """Run population manifolds analysis."""
    print("Running population manifolds analysis...")
    analyze_population_manifolds(paths)

def run_population_trajectories_by_direction_analysis(paths: DataPaths):
    """Run population trajectory analysis comparing CW vs CCW."""
    print("Running population trajectory analysis by direction...")
    analyze_population_trajectories_by_direction(paths, method='pca')

def run_phase_space_trajectories_analysis(paths: DataPaths):
    """Run phase space trajectories analysis."""
    print("Running phase space trajectories analysis...")
    analyze_phase_space_trajectories(paths)

def run_ica_decomposition_analysis(paths: DataPaths):
    """Run ICA decomposition analysis."""
    print("Running ICA decomposition analysis...")
    analyze_ica_decomposition(paths)

def run_lfp_power_analysis(paths: DataPaths):
    """Run LFP power analysis around movement."""
    print("Running LFP power analysis...")
    analyze_lfp_movement_power(paths)

# def run_theta_analysis(paths: DataPaths):
#     """Run LFP theta oscillation analysis."""
#     print("Running theta oscillation analysis...")
#     analyze_theta_oscillations(paths)

# def run_pac_analysis(paths: DataPaths):
#     """Run LFP phase-amplitude coupling analysis."""
#     print("Running PAC analysis...")
#     analyze_phase_amplitude_coupling(paths)

# def run_cfc_analysis(paths: DataPaths):
#     """Run LFP cross-frequency coupling analysis."""
#     print("Running CFC analysis...")
#     analyze_cross_frequency_coupling(paths)

# def run_spike_phase_locking_analysis(paths: DataPaths):
#     """Run spike-LFP phase locking analysis."""
#     print("Running spike-LFP phase locking analysis...")
#     analyze_spike_phase_locking(paths)

# def run_temporal_autocorrelation_analysis(paths: DataPaths):
#     """Run temporal autocorrelation analysis."""
#     print("Running temporal autocorrelation analysis...")
#     analyze_temporal_autocorrelation(paths)

# def run_cross_correlation_pairs_analysis(paths: DataPaths):
#     """Run cross-correlation analysis between neuron pairs."""
#     print("Running cross-correlation pairs analysis...")
#     analyze_cross_correlation_pairs(paths)

# def run_burst_analysis(paths: DataPaths):
#     """Run analysis of single-unit bursting properties."""
#     print("Running burst analysis...")
#     analyze_bursting_behavior(paths)

# def run_isi_analysis(paths: DataPaths):
#     """Run analysis of inter-spike interval distributions."""
#     print("Running ISI analysis...")
#     analyze_isi_distribution(paths)

# def run_spike_motif_analysis(paths: DataPaths):
#     """Run spike pattern motif (SeqNMF) analysis."""
#     print("Running spike motif analysis...")
#     analyze_spike_pattern_motifs(paths)

# def run_wave_propagation_analysis(paths: DataPaths):
#     """Run wave propagation analysis."""
#     print("Running wave propagation analysis...")
#     analyze_wave_propagations(paths)

# def run_multi_scale_temporal_encoding_analysis(paths: DataPaths):
#     """Run multi-scale temporal encoding analysis."""
#     print("Running multi-scale temporal encoding analysis...")
#     analyze_multi_scale_temporal_encoding(paths)

# def run_rank_order_coding_analysis(paths: DataPaths):
#     """Run rank-order coding analysis."""
#     print("Running rank-order coding analysis...")
#     analyze_rank_order_coding(paths)

# def run_temporal_clustering_analysis(paths: DataPaths):
#     """Run temporal clustering analysis."""
#     print("Running temporal clustering analysis...")
#     analyze_temporal_clustering(paths)

# def run_dopamine_spike_coupling_analysis(paths: DataPaths):
#     """Run dopamine-spike coupling analysis."""
#     print("Running dopamine-spike coupling analysis...")
#     analyze_dopamine_spike_coupling(paths)
# def run_dopamine_triggered_firing_analysis(paths: DataPaths):
#     """Run dopamine-triggered average neural firing analysis."""
#     print("Running dopamine-triggered firing analysis...")
#     analyze_dopamine_triggered_firing(paths)

# def run_dopamine_modulation_index_analysis(paths: DataPaths):
#     """Run dopamine modulation index analysis."""
#     print("Running dopamine modulation index analysis...")
#     analyze_dopamine_modulation_index(paths)

# def run_dopamine_lfp_coupling_analysis(paths: DataPaths):
#     """Run dopamine-LFP coupling analysis."""
#     print("Running dopamine-LFP coupling analysis...")
#     analyze_dopamine_lfp_coupling(paths)

# def run_dopamine_phase_locking_relationship_analysis(paths: DataPaths):
#     """Run analysis of dopamine modulation of phase locking."""
#     print("Running dopamine-phase locking relationship analysis...")
#     analyze_dopamine_phase_locking_relationship(paths)



# def run_spatial_depth_analysis(paths: DataPaths):
#     """Run spatial organization by depth analysis."""
#     print("Running spatial depth analysis...")
#     analyze_spatial_organization_depth(paths)

# def run_depth_tuning_by_behavior_analysis(paths: DataPaths):
#     """Run depth-dependent behavioral tuning analysis."""
#     print("Running depth tuning by behavior analysis...")
#     analyze_depth_tuning_by_behavior(paths)

# def run_spatial_clustering_analysis(paths: DataPaths):
#     """Run spatial clustering analysis."""
#     print("Running spatial clustering analysis...")
#     analyze_spatial_clustering(paths)

# def run_medial_lateral_organization_analysis(paths: DataPaths):
#     """Run medial-lateral organization analysis."""
#     print("Running medial-lateral organization analysis...")
#     analyze_medial_lateral_organization(paths)

# def run_multi_shank_interactions_analysis(paths: DataPaths):
#     """Run multi-shank interactions analysis."""
#     print("Running multi-shank interactions analysis...")
#     analyze_multi_shank_interactions(paths)

# def run_neural_clustering_analysis(paths: DataPaths):
#     """Run neural clustering analysis."""
#     print("Running neural clustering analysis...")
#     analyze_neural_clustering(paths)

# def run_functional_tuning_matrix_analysis(paths: DataPaths):
#     """Run functional tuning matrix analysis."""
#     print("Running functional tuning matrix analysis...")
#     analyze_functional_tuning_matrix(paths)

# def run_cell_type_comparison_analysis(paths: DataPaths):
#     """Run MSN vs FSI comparison across all analyses."""
#     print("Running cell type comparison analysis...")
#     compare_cell_types(paths)

# def run_mutual_information_analysis(paths: DataPaths):
#     """Run mutual information analysis."""
#     print("Running mutual information analysis...")
#     analyze_mutual_information(paths)

# def run_predictive_decoding_analysis(paths: DataPaths):
#     """Run predictive decoding analysis."""
#     print("Running predictive decoding analysis...")
#     analyze_predictive_decoding(paths)

# def run_analyze_ili_by_port(paths: DataPaths):
#     """Run analyze_ili_by_port analysis."""
#     print("Running analyze_ili_by_port analysis...")
#     analyze_ili_by_port(paths)

# Map analysis names to functions
ANALYSIS_FUNCTIONS = {
    'event_tuning': run_event_tuning_analysis,
    'movement_tuning': run_movement_tuning,
    'lfp_peth': run_lfp_peth_analysis,
    'dopamine_peth': run_dopamine_peth_analysis,
    'behavioral_switch': run_behavioral_switch_analysis,
    'port_to_port': run_port_to_port_analysis,
    'strategy_encoding': run_strategy_encoding_analysis,
    'directional_tuning': run_directional_tuning_analysis,
    'context_encoding': run_context_dependent_encoding_analysis,
    'trajectory_consistency': run_trajectory_consistency_analysis,
    'spatial_rate_maps': run_spatial_rate_maps_analysis,
    'pre_switch': run_pre_switch_activity_analysis,
    'outcome_encoding': run_outcome_encoding_analysis,
    'reward_magnitude': run_reward_magnitude_analysis,
    'reward_history': run_reward_history_analysis,
    'history_glm': run_history_glm_analysis,
    'predict_rt': run_predict_reaction_time_multimodal,
    'choice_prediction': run_choice_prediction_analysis,
    'population_stats': run_population_statistics_analysis,
    'population_manifolds': run_population_manifolds_analysis,
    'population_trajectories_direction': run_population_trajectories_by_direction_analysis,
    'phase_space': run_phase_space_trajectories_analysis,
    'ica': run_ica_decomposition_analysis,
    'lfp_power': run_lfp_power_analysis,
    # 'theta_osc': run_theta_analysis,
    # 'pac': run_pac_analysis,
    # 'cfc': run_cfc_analysis,
    # 'spike_phase_locking': run_spike_phase_locking_analysis,
    # 'temporal_autocorrelation': run_temporal_autocorrelation_analysis,
    # 'cross_correlation_pairs': run_cross_correlation_pairs_analysis,
    # 'bursts': run_burst_analysis,
    # 'isi': run_isi_analysis,
    # 'spike_motifs': run_spike_motif_analysis,
    # 'wave_propagations': run_wave_propagation_analysis,
    # 'multi_scale_temporal': run_multi_scale_temporal_encoding_analysis,
    # 'rank_order_coding': run_rank_order_coding_analysis,
    # 'temporal_clustering': run_temporal_clustering_analysis,
    # 'dopamine_coupling': run_dopamine_spike_coupling_analysis,
    # 'dopamine_triggered': run_dopamine_triggered_firing_analysis,
    # 'dopamine_modulation': run_dopamine_modulation_index_analysis,
    # 'dopamine_lfp': run_dopamine_lfp_coupling_analysis,
    # 'dopamine_phase_locking_relationship': run_dopamine_phase_locking_relationship_analysis,
    # 'spatial_depth': run_spatial_depth_analysis,
    # 'depth_behavior_tuning': run_depth_tuning_by_behavior_analysis,
    # 'spatial_clustering': run_spatial_clustering_analysis,
    # 'medial_lateral': run_medial_lateral_organization_analysis,
    # 'multi_shank': run_multi_shank_interactions_analysis,
    # 'neural_clustering': run_neural_clustering_analysis,
    # 'functional_tuning': run_functional_tuning_matrix_analysis,
    # 'cell_type_comparison': run_cell_type_comparison_analysis,
    # 'mutual_information': run_mutual_information_analysis,
    # 'predictive_decoding': run_predictive_decoding_analysis,
    # 'ili_by_port': analyze_ili_by_port,
    'of_tier1_behavior': analyze_of_tier1_behavior,
    'all': None
}


def main():
    parser = argparse.ArgumentParser(
        description="Run post-analysis on Neuropixels + behavioral data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_post_analysis.py --mouse 1818 --day 09182025 --analysis event_tuning
  python run_post_analysis.py --mouse 1818 --day 2025-09-18 --analysis all
        """
    )
    
    parser.add_argument(
        '--mouse',
        required=True,
        help='Mouse ID (e.g., "1818")'
    )
    
    parser.add_argument(
        '--day',
        required=True,
        help='Date in any format (MMDDYYYY, YYYY-MM-DD, YYMMDD, or YYYYMMDD)'
    )
    
    parser.add_argument(
        '--task',
        default=None,
        help='Task type (e.g., "openfield", "y_maze")'
    )
    
    parser.add_argument(
        '--analysis',
        required=True,
        help='Analysis to run (comma-separated list, or "all")'
    )
    
    parser.add_argument(
        '--base-path',
        default="E:/Neuropixels/Python/DemoData",
        help='Base path for all data (default: E:/Neuropixels/Python/DemoData)'
    )
    
    parser.add_argument(
        '--neural-base-path',
        default=None,
        help='Optional separate path for neural data (defaults to base-path)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate that required data files exist before running analysis'
    )
    
    parser.add_argument(
        '--required-data',
        nargs='+',
        default=['neural', 'events'],
        choices=['neural', 'events', 'dlc', 'video', 'tdt'],
        help='Required data types for validation (default: neural events)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print data summary before running analysis'
    )

    parser.add_argument(
        '--start-from',
        default=None,
        choices=list(ANALYSIS_FUNCTIONS.keys()),
        help='Resume analysis from this function (only applies when analysis="all")'
    )
    
    args = parser.parse_args()
    
    # Validate base path exists
    if not Path(args.base_path).exists():
        print(f"Error: Base path '{args.base_path}' does not exist.")
        sys.exit(1)

    # Load data
    print(f"Loading data for mouse {args.mouse}, day {args.day}...")
    paths = load_session_data(
        mouse_id=args.mouse,
        day=args.day,
        base_path=args.base_path,
        neural_base_path=args.neural_base_path
    )
    
    if not hasattr(paths, 'base_path'):
        try:
            paths.base_path = args.base_path
        except AttributeError:
            print("Warning: Could not set base_path on DataPaths object (it might be immutable).")

    if not hasattr(paths, 'neural_base_path'):
        try:
            paths.neural_base_path = args.neural_base_path if args.neural_base_path else args.base_path
        except AttributeError:
            print("Warning: Could not set neural_base_path on DataPaths object (it might be immutable).")

    # Print summary if requested
    if args.summary:
        print_data_summary(paths)
    
    # Validate if requested
    if args.validate:
        validation = validate_data_paths(paths, required=args.required_data)
        all_valid = all(validation.values())
        
        if not all_valid:
            print("\nΓÜá∩╕Å  Validation failed:")
            for data_type, exists in validation.items():
                status = "Γ£ô" if exists else "Γ£ù"
                print(f"  {status} {data_type}")
            print("\nSome required data files are missing. Continuing anyway...\n")
        else:
            print("\nΓ£ô All required data files found.\n")
    
    # Determine which analysis dictionary to use based on the task
    if args.task and args.task.lower() == 'openfield':
        try:
            from analyses_openfield import ANALYSIS_FUNCTIONS_OPENFIELD
            current_analysis_functions = ANALYSIS_FUNCTIONS_OPENFIELD
        except ImportError:
            print("Error: Could not import ANALYSIS_FUNCTIONS_OPENFIELD from analyses_openfield")
            sys.exit(1)
    else:
        current_analysis_functions = ANALYSIS_FUNCTIONS

    # Validate start_from argument if using 'all'
    if args.analysis == 'all' and args.start_from and args.start_from not in current_analysis_functions:
        print(f"Error: --start-from '{args.start_from}' not found in the selected task's analysis functions.")
        sys.exit(1)

    # Run analysis
    analyses_to_run = []
    if args.analysis == 'all':
        analyses_to_run = ['all']
    else:
        # Split by comma and strip whitespace
        requested = [a.strip() for a in args.analysis.split(',')]
        for req in requested:
            if req in current_analysis_functions:
                analyses_to_run.append(req)
            else:
                print(f"Error: Analysis '{req}' not implemented for task '{args.task}'")
                sys.exit(1)

    if 'all' in analyses_to_run:
        print(f"Running all available analyses for task '{args.task}'...\n")
        
        skip = True if args.start_from else False
        
        for name, func in current_analysis_functions.items():
            # Handle resume logic
            if skip:
                if name == args.start_from:
                    skip = False
                    print(f"Resuming from: {name}")
                else:
                    continue
            
            if name != 'all' and func is not None:
                print(f"\n{'='*60}")
                print(f"Analysis: {name}")
                print(f"{'='*60}")
                try:
                    func(paths)
                except Exception as e:
                    print(f"  Γ£ù Error: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Run specific list of analyses
        for analysis_name in analyses_to_run:
            func = current_analysis_functions[analysis_name]
            
            print(f"\n{'='*60}")
            print(f"Running analysis: {analysis_name}")
            print(f"{'='*60}\n")
            try:
                func(paths)
            except Exception as e:
                print(f"Γ£ù Error running analysis: {analysis_name}: {e}")
                import traceback
                traceback.print_exc()
                # Don't exit, try next analysis

    print("\nAnalysis complete!")
if __name__ == "__main__":
    main()

