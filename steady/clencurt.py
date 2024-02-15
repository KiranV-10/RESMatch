import numpy as np

print("running ....")
def clencurt(N):
    # Generate theta values
    theta = np.pi * np.arange(0, N + 1) / N
    # Compute Chebyshev points
    x = np.cos(theta)
    
    # Initialize weights
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    
    # Different computation for even and odd N
    if N % 2 == 0:
        w[0] = 1 / (N**2 - 1)
        w[N] = w[0]
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = 1 / N**2
        w[N] = w[0]
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    
    # Assign weights
    w[ii] = 2 * v / N
    
    return x, w

# Example usage
N = 10
x, w = clencurt(N)
print("Nodes (x):", x)
print("Weights (w):", w)
