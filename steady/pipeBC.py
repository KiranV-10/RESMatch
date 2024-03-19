import numpy as np

def pipeBC(LHS, RHS, yP, N, *args):
    """
    Impose boundary conditions for Navier-Stokes Resolvent.

    Parameters:
    - LHS: Left-hand side matrix.
    - RHS: Right-hand side matrix.
    - yP: Coordinates (in plus units).
    - N: Number of grid points in r.
    - args: Optional arguments for boundary conditions. Empty for no slip,
            (yPD, AD) for opposition control.

    Returns:
    - H: Resolvent matrix reflecting the imposed boundary conditions.
    """
    # Set specified rows in LHS and RHS to zero
    LHS[1:N:3*N, :] = 0
    RHS[1:N:3*N, :] = 0

    # No slip for u and w
    LHS[0, 0] = 1  # u(y=0) = 0
    LHS[2*N, 2*N] = 1  # w(y=0) = 0

    if len(args) > 1:
        # Opposition control
        yPD, AD = args
        ci = np.where(yP > yPD)[0][0]  # Find index where yP > yPD
        LHS[N, N] = 1
        LHS[N, N+ci] = AD * 1
        # v(y=0) + AD*v(y=yD) = 0
    else:
        # No slip
        LHS[N, N] = 1  # v(y=0) = 0

    # Compute resolvent
    H = np.linalg.solve(LHS, RHS)

    return H

# Example usage
# You would need to define LHS, RHS, yP, N, and optionally yPD and AD according to your specific problem setup.
# N = 10  # Example setup
# yP = np.linspace(0, 1, N)  # Example yP
# LHS = np.random.randn(3*N, 3*N)  # Example LHS matrix
# RHS = np.random.randn(3*N, 3*N)  # Example RHS matrix
# H = pipeBC(LHS, RHS, yP, N)  # No slip condition
# For opposition control, you would call it as follows:
# H_opposition = pipeBC(LHS, RHS, yP, N, yPD, AD)  # with yPD and AD specified
