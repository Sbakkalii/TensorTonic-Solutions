import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    element = x - np.max(x, axis=-1, keepdims=True)
    element_x = np.exp(element)
    softmax_x = element_x / np.sum(element_x, axis=-1, keepdims=True)
    return softmax_x