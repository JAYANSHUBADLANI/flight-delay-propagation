"""Predict the next departure's delay from what was known two hours ahead.

Two things are being measured here, and the second matters more than the first:

1. whether an as-of-cutoff model beats the obvious rules, and
2. **how much better a leaky model would look**, which is the number that says
   why the cutoff exists at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import CATEGORICAL_FEATURES, FEATURES, TARGET

# Available at the cutoff only if the inbound had already landed. The leaky
# variant swaps these in unconditionally, which is the mistake being quantified.
LEAKY_FEATURES = FEATURES + ["prev_arr_delay"]


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("category")
    for col in out.columns:
        if col not in CATEGORICAL_FEATURES and out[col].dtype == bool:
            out[col] = out[col].astype(float)
    return out


def fit_model(train: pd.DataFrame, features=None, seed: int = 0):
    features = features or FEATURES
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=seed,
    )
    model.fit(_prepare(train[features]), train[TARGET])
    return model


def scores(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "n": int(len(y_true)),
    }


def baseline_mean(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Predict the training mean for everything. The floor any model must clear."""
    return scores(test[TARGET], np.full(len(test), train[TARGET].mean()))


def baseline_carry_forward(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """The rule an operations team would use without a model at all.

    Pass the inbound delay straight through when the aircraft has already landed,
    and fall back to the training mean when it has not. This is the honest
    non-model comparison, and it is available at the cutoff.
    """
    known = test["known_inbound_arr_delay"]
    pred = known.fillna(train[TARGET].mean()).clip(lower=0)
    return scores(test[TARGET], pred)


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    honest = fit_model(train)
    leaky = fit_model(train, features=LEAKY_FEATURES)

    return {
        "predict_training_mean": baseline_mean(train, test),
        "carry_inbound_delay_forward": baseline_carry_forward(train, test),
        "model_as_of_cutoff": scores(test[TARGET],
                                     honest.predict(_prepare(test[FEATURES]))),
        "model_with_leakage": scores(test[TARGET],
                                     leaky.predict(_prepare(test[LEAKY_FEATURES]))),
    }
