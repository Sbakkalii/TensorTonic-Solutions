import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    relu = np.maximum(0, x)
    return relu
