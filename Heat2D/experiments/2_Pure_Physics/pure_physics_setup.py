"""
Experiment Goal: 2_Pure_Physics
Description: PINN training using Pure Physics approach (No internal data loss).
"""
import os

def get_pure_physics_config():
    """
    Returns the configuration for the Pure Physics-Informed experiment.
    
    Strategy:
    - Data Loss (Internal): Disabled (weight=0.0).
    - Warmup: Disabled (0 epochs), physics is active from the start.
    - Physics Weight: Standard or boosted.
    - Directory: 'Results/pure_physics'.
    """
    return {
        'loss_weights': {'data': 0.0, 'bc': 1.0, 'physics': 1.0}, # Boost physics weight slightly since it's the main driver
        'warmup_epochs': 0,
        'results_subdir': 'pure_physics',
        'model_suffix': '_pure_physics'
    }
