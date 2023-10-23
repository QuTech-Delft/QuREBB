import numpy as np
import qutip as qt

# Define standard basis states for a two-level system
up = qt.basis(2, 0)  # dark state
down = qt.basis(2, 1)  # bright state

# Define superposition states in the X and Y basis
x = (up + down).unit()  # X basis state
x_min = (up - down).unit()  # -X basis state
y = (up + 1j * down).unit()  # Y basis state
y_min = (up - 1j * down).unit()  # -Y basis state

# Convert the defined kets to density matrices
up_dm = qt.ket2dm(up)
down_dm = qt.ket2dm(down)
x_dm = qt.ket2dm(x)
x_min_dm = qt.ket2dm(x_min)
y_dm = qt.ket2dm(y)
y_min_dm = qt.ket2dm(y_min)

# Define the identity operator for a two-level system
eye = qt.qeye(2)

# Alias for the tensor product function
tp = qt.tensor


def alpha_ket(alpha):
    """
    Create a superposition state with population ratio of alpha to (1 - alpha).

    Parameters:
    ----------
    alpha : float
        Population of down state

    Returns:
    --------
    Qobj
        Resulting superposition state.
    """
    return np.sqrt(alpha) * down + np.sqrt(1 - alpha) * up


def alpha_dm(alpha):
    """
    Create a density matrix of superposition state with population ratio of alpha to (1 - alpha).

    Parameters:
    ----------
    alpha : float
        Population of down state

    Returns:
    --------
    Qobj
        Density matrix of superposition state.
    """
    return qt.ket2dm(alpha_ket(alpha))


def vacuum(dim=2):
    """
    Create the vacuum state.

    Parameters:
    ----------
    dim : int, optional
        Dimension of the Hilbert space. Default is 2.
    """
    return qt.basis(dim, 0)


def vacuum_dm(dim=2):
    """
    Create a density matrix of vacuum state.

    Parameters:
    ----------
    dim : int, optional
        Dimension of the Hilbert space. Default is 2.
    """
    return qt.ket2dm(vacuum(dim=dim))


def photon(dim=2):
    """
    Create the single-photon state.

    Parameters:
    ----------
    dim : int, optional
        Dimension of the Hilbert space. Default is 2.
    """
    return qt.basis(dim, 1)


def photon_dm(dim=2):
    """
    Create a density matrix of single-photon state.

    Parameters:
    ----------
    dim : int, optional
        Dimension of the Hilbert space. Default is 2.
    """
    return qt.ket2dm(photon(dim=dim))
