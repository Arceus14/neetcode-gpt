import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        MAX = np.max(z)
        SUM = np.sum(np.exp(z-MAX))
        return np.round(np.exp(z-MAX)/SUM, 4)
