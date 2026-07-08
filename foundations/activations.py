import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        e = np.e
        for i, x in enumerate(z):
            z[i] = 1 / (1 + e**(-x))
        return np.round(z, 5)


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        for i, x in enumerate(z):
            z[i] = max(0, x)
        return z
