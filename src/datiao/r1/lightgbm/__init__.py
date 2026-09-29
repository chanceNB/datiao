"""R1 fixed LightGBM multi-label baseline."""

def run_baseline(*args, **kwargs):
    from .train import run_baseline as implementation
    return implementation(*args, **kwargs)

def reload_lightgbm_run(*args, **kwargs):
    from .io import reload_lightgbm_run as implementation
    return implementation(*args, **kwargs)

__all__ = ["run_baseline", "reload_lightgbm_run"]
