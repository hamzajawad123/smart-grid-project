"""Official Colab selection record. Do not invent replacement scores."""

from __future__ import annotations

# Walk-forward RandomizedSearch + mean ensemble, from notebooks/02_smart_grid_training.ipynb
COLAB_TEST_METRICS = {
    "ensemble_mean": {"wape": 0.028887, "mae": 129.721, "rmse": 185.937, "peak_mae": 287.251},
    "lightgbm": {"wape": 0.029758, "mae": 133.632, "rmse": 191.432, "peak_mae": 289.697},
    "xgboost": {"wape": 0.030117, "mae": 135.242, "rmse": 190.538, "peak_mae": 290.585},
    "seasonal_naive_24": {"wape": 0.070727, "mae": 317.608, "rmse": 436.407, "peak_mae": 506.122},
    "seasonal_naive_168": {"wape": 0.111709, "mae": 501.642, "rmse": 712.153, "peak_mae": 989.992},
}

WINNER = "ensemble_mean"
TRAIN_P90_MW = 5710.0
