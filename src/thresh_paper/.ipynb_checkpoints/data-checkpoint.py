# src/calib_paper/data.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid


def make_toy_dataset(
    n_centers: int = 10,
    n_per_center: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Generate a toy dataset with:
      - center ID
      - age
      - smoking status
      - model risk score (probs)
      - binary outcome golds ~ Bernoulli(probs)
    """

    rng = np.random.default_rng(random_state)
    n = n_centers * n_per_center

    centers = np.repeat(np.arange(n_centers), n_per_center)
    age = rng.normal(60, 5, size=n)
    smoker = rng.binomial(1, 0.4, size=n)

    center_effects = rng.normal(0.0, 0.5, size=n_centers)
    c_shift = center_effects[centers]

    logit_risk = -6.0 + 0.05 * (age - 60) + 0.8 * smoker + c_shift
    probs = sigmoid(logit_risk)

    golds = rng.binomial(1, probs)

    df = pd.DataFrame(
        {
            "center": centers,
            "age": age,
            "smoker": smoker,
            "probs": probs,
            "golds": golds,
        }
    )

    df["center"] = df["center"].astype(int) + 1

    return df
