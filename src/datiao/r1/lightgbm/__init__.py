"""R1 fixed LightGBM multi-label baseline."""

def run_baseline(*args, **kwargs):
    from .train import run_baseline as implementation
    return implementation(*args, **kwargs)

__all__ = ["run_baseline"]
