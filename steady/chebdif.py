import numpy as np

def chebdif(N, M):
    # Identity matrix and logical identity matrix
    I = np.eye(N)
    L = np.eye(N, dtype=bool)

    # Indices for flipping trick
    n1, n2 = N // 2, (N + 1) // 2

    # Compute theta vector
    k = np.arange(N).reshape(-1, 1)
    th = k * np.pi / (N - 1)

    # Compute Chebyshev points
    x = np.sin(np.pi * np.arange(N - 1, -N - 1, -2) / (2 * (N - 1)))

    # Trigonometric identity using broadcasting
    T = np.tile(th / 2, (1, N))
    DX = 2 * np.sin(T + T.T) * np.sin(T - T.T)

    # Flipping trick
    DX[:n1, :] = DX[:n1, :] - np.flipud(np.fliplr(DX[n1:, :]))
    DX[n1:, :] = -np.flipud(np.fliplr(DX[:n1, :]))

    # Ensure the main diagonal is filled with ones to avoid division by zero
    np.fill_diagonal(DX, 1)

    # Toeplitz matrix for coefficients
    c = np.tile((-1) ** k, (1, N))
    c[0, :] = (c[0, :] * 2) 
    c[-1, :] = (c[-1, :] * 2) 
    c[:, 0] = (c[:, 0] * 0.5)
    c[:, -1] = ( c[:, -1] * 0.5)

    # Initialize differentiation matrices
    Z = 1.0 / DX
    np.fill_diagonal(Z, 0)  # Clear the diagonal to ensure no division by zero errors
    D = np.eye(N)
    DM = np.zeros((N, N, M))

    for ell in range(1, M + 1):
        D = ell * Z * (c * np.tile(np.diag(D), (N, 1)) - D)
        np.fill_diagonal(D, -D.sum(axis=1))
        DM[:, :, ell - 1] = D

    return x, DM

# Example usage
N = 10  # Size of differentiation matrix
M = 2   # Number of derivatives required
x, DM = chebdif(N, M)
print("Chebyshev nodes (x):", x)
for ell in range(M):
    print(f"D{ell + 1} matrix:\n", DM[:, :, ell])
