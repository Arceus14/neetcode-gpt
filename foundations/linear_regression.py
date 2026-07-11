import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        r =np.dot(X, weights)
        r = [round(y, 5) for y in r]
        return r

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        e = 0
        e += sum((yi - pi)**2 for yi, pi in zip(ground_truth, model_prediction))
        e /= len(ground_truth)
        return round(e[0], 5)





