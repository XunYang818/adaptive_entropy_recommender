"""Namespaced helpers for the adaptive method."""
from .baseline_model import SimulationConfig, InformationDynamicsSimulator, save_result

ADAPTIVE_DEFAULTS = {
    "history_window": 100,
    "target_entropy": 0.75,
    "epsilon_min": 0.02,
    "epsilon_max": 0.20,
    "gain": 0.40,
}

def make_adaptive_config(**kwargs):
    values = dict(ADAPTIVE_DEFAULTS)
    values.update(kwargs)
    values["method"] = "adaptive"
    return SimulationConfig(**values)
