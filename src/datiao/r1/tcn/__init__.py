"""Fixed causal Pen TCN baseline."""

def run_baseline(*args, **kwargs):
    from .io import run_baseline as implementation
    return implementation(*args, **kwargs)

def reload_tcn_run(*args, **kwargs):
    from .io import reload_tcn_run as implementation
    return implementation(*args, **kwargs)
