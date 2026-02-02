import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    N = X.shape[0]
    indices = np.arange(N)
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    for start_idx in range(0, N, batch_size):
        if drop_last and start_idx+batch_size > N:
            continue
        end_idx = min(start_idx+batch_size, N)
        batch_idx = indices[start_idx:end_idx]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        yield X_batch, y_batch