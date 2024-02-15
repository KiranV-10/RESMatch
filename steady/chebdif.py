import numpy as np

def chebdif(N, M):
    I = np.eye(N)  # Identity matrix
    L = np.eye(N, dtype=bool)  # Logical identity matrix

    n1 = N // 2
    n2 = (N + 1) // 2  # Indices used for flipping trick

    k = np.arange(0, N).reshape(N, 1)  # Column vector
    th = k * np.pi / (N - 1)

    x = np.sin(np.pi * np.arange(N - 1, -N, -2) / (2 * (N - 1)))

    T = np.tile(th / 2, (1, N))
    DX = 2 * np.sin(T + T.T) * np.sin(T - T.T)  # Trigonometric identity
    DX = np.vstack([DX[:n1, :], -np.flipud(np.fliplr(DX[:n2, :]))])  # Flipping trick
    DX[L] = 1  # Put 1's on the main diagonal of DX

    C = np.tile((-1) ** k, (1, N))
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] /= 2
    C[:, -1] /= 2

    Z = 1 / DX
    Z[L] = 0  # Zeroes on the diagonal

    D = np.eye(N)
    DM = np.zeros((N, N, M))

    for ell in range(1, M + 1):
        D = ell * Z * (C * np.tile(np.diag(D), (N, 1)) - D)  # Off-diagonals
        D[L] = -np.sum(D, axis=1)  # Correct main diagonal of D
        DM[:, :, ell - 1] = D

    return x, DM
