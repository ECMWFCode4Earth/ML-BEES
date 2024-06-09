# ------------------------------------------------------------------
# Script to define evaluation metrics
# ------------------------------------------------------------------

import numpy as np
from sklearn.metrics import r2_score

def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculated the r-squared score between 2 arrays of values

    Args:
        y_pred (np.ndarray): Predicted values
        y_true (np.ndarray): Target values

    Returns:
         (float) r-squared score
    """
    return r2_score(y_pred.flatten(), y_true.flatten())

def calc_R2(pred, target):
    target_hat = np.mean(target)
    residuals_sum = np.sum((target - pred) ** 2)
    total_sum = np.sum((target - target_hat) ** 2)
    R2 = 1 - (residuals_sum / total_sum)
    return R2