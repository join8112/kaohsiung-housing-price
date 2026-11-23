import numpy as np
import pandas as pd

from khp.policy import wild_cluster_bootstrap


def _panel(effect: float, seed: int = 0) -> pd.DataFrame:
    """30 clusters with a cluster-level shock; half the clusters are treated."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(30), 40)
    treated = (g < 15).astype(int)
    y = effect * treated + rng.normal(0, 0.5, 30)[g] * 0.3 + rng.normal(0, 1, g.size)
    return pd.DataFrame({"y": y, "treated": treated, "g": g})


def test_wild_cluster_bootstrap_detects_large_effect():
    out = wild_cluster_bootstrap("y ~ treated", _panel(effect=1.0), "treated", "g", reps=199)
    assert out["clusters"] == 30
    assert out["p_value"] < 0.05


def test_wild_cluster_bootstrap_no_effect_is_not_significant():
    out = wild_cluster_bootstrap("y ~ treated", _panel(effect=0.0, seed=3), "treated", "g", reps=199)
    assert 0.0 < out["p_value"] <= 1.0
    assert out["p_value"] > 0.05
