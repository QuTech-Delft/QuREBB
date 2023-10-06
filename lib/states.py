import numpy as np
import qutip as qt

up = qt.basis(2, 0)  # dark state
down = qt.basis(2, 1)  # bright state
x = (up + down).unit()
x_min = (up - down).unit()
y = (up + 1j * down).unit()
y_min = (up - 1j * down).unit()

up_dm = qt.ket2dm(up)
down_dm = qt.ket2dm(down)
x_dm = qt.ket2dm(x)
x_min_dm = qt.ket2dm(x_min)
y_dm = qt.ket2dm(y)
y_min_dm = qt.ket2dm(y_min)

eye = qt.qeye(2)
tp = qt.tensor


def alpha_ket(alpha):
    return np.sqrt(alpha) * down + np.sqrt(1 - alpha) * up


def alpha_dm(alpha):
    return qt.ket2dm(alpha_ket(alpha))


def vacuum(dim=2):
    return qt.basis(dim, 0)


def vacuum_dm(dim=2):
    return qt.ket2dm(vacuum(dim=dim))


def photon(dim=2):
    return qt.basis(dim, 1)


def photon_dm(dim=2):
    return qt.ket2dm(photon(dim=dim))
