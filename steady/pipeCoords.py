import numpy as np
from chebdif import chebdif
from clencurt import clencurt

def pipeCoords(n, N):
    """
    Set up the coordinate system, differentiation, and integration matrices
    for pipe geometry.
    """
    # Call the chebdif function to get Chebyshev points and differentiation matrices
    x, DM = chebdif(2*N, 2)
    # Use slicing to focus on the first half (similar to MATLAB's 1:N for half)
    half = slice(0, N)
    r = x[half]

    D1 = DM[:, :, 0]  # First differentiation matrix
    D2 = DM[:, :, 1]  # Second differentiation matrix

    # Split D1 and D2 into odd and even components
    s = (-1)**(n % 2)
    D1E = D1[half, half] + s * D1[half, 2*N - np.arange(1, N+1)]
    D1O = D1[half, half] - s * D1[half, 2*N - np.arange(1, N+1)]
    D2E = D2[half, half] + s * D2[half, 2*N - np.arange(1, N+1)]
    D2O = D2[half, half] - s * D2[half, 2*N - np.arange(1, N+1)]

    # Call the clencurt function to get integration weights
    _, dr = clencurt(2*N - 1)
    dr = dr[half]

    return r, dr, D1E, D1O, D2E, D2O

# Example usage, assuming n and N are defined
n, N = 1, 10
r, dr, D1E, D1O, D2E, D2O = pipeCoords(n, N)
print("Radial coordinate (r):", r)
print("Integration weights (dr):", dr)
# D1E, D1O, D2E, D2O matrices are now available for further use
