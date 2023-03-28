import numpy as np


def batch_diagonal(A):
    N = A.shape[1]
    A = np.expand_dims(A, axis=1)
    return A * np.eye(N)


def batch_data(X, batch_size):
    N, _ = X.shape
    X_batch = []
    for k in range(N // batch_size):
        X_batch.append(X[k * batch_size:(k + 1) * batch_size])
    if (N // batch_size) * batch_size != len(X):
        X_batch.append(X[(N // batch_size) * batch_size:])
    return X_batch
