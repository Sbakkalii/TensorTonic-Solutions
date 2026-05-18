import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    vals = []
    for x in x:
        if x > 0:
            elu = x
        else:
            elu = alpha * (np.exp(x) - 1)
        vals.append(elu)
    return vals