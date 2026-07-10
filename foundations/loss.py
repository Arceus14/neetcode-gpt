import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        L = 0
        epsilon = 1e-7
        for yi, pi in zip(y_true, y_pred):
            L += (yi * np.log(pi + epsilon)) + \
                ((1-yi) * np.log(1-pi + epsilon))
        L *= -1/len(y_true)
        return round(L, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        L = 0
        epsilon = 1e-7

        for act_class, pred_class in zip(y_true, y_pred):
            for yi, pi in zip(act_class, pred_class):

                L += (yi*np.log(pi + epsilon))
        L *= -1/len(y_true)
        return round(L, 4)















