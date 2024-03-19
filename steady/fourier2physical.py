import numpy as np

def fourier2physical(u, k, x, n, theta, omega=None, t=None):
    """
    Convert Fourier modes to physical coordinates.
    
    Parameters:
    - u: Input (complex) mode shape (array-like).
    - k: Streamwise wavenumber.
    - x: Streamwise coordinates (array-like).
    - n: Azimuthal wavenumber.
    - theta: Azimuthal coordinates (array-like).
    - omega (optional): Frequency.
    - t (optional): Time.
    
    Returns:
    - U: The Fourier-transformed velocity field in physical space.
    """
    # Handling the optional frequency (omega) and time (t)
    exp_omegat = np.exp(-1j * omega * t) if omega is not None and t is not None else 1
    
    # Initialize the output array U
    U = np.zeros((len(u), len(theta), len(x)), dtype=complex)
    
    # Using broadcasting to fill in the values of U for each combination of x and theta
    for ix, x_val in enumerate(x):
        for it, theta_val in enumerate(theta):
            U[:, it, ix] = u * np.exp(1j * k * x_val + 1j * n * theta_val) * exp_omegat
    
    # Taking the real part and squeezing the array to remove any single-dimensional entries
    U = np.real(np.squeeze(U))
    
    return U

# Example usage
u = np.array([1+1j, 2+2j])  # Example complex mode shapes
k = 1  # Streamwise wavenumber
x = np.linspace(0, 10, 100)  # Streamwise coordinates
n = 1  # Azimuthal wavenumber
theta = np.linspace(0, 2*np.pi, 100)  # Azimuthal coordinates

# Without frequency and time
U = fourier2physical(u, k, x, n, theta)

# With frequency (omega) and time (t), if needed
omega = 1
t = 1
U_with_omega_t = fourier2physical(u, k, x, n, theta, omega, t)

print("U shape:", U.shape)
print("U with omega and t shape:", U_with_omega_t.shape)
