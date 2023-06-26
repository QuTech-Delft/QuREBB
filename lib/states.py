import numpy as np
import qutip as qt

up          = qt.basis(2,0) # dark state
down        = qt.basis(2,1) # bright state
x           = (up + down).unit()
x_min       = (up - down).unit()
y           = (up + 1j*down).unit()
y_min       = (up - 1j*down).unit()

up_dm       = qt.ket2dm(up)
down_dm     = qt.ket2dm(down)
x_dm        = qt.ket2dm(x)
x_min_dm    = qt.ket2dm(x_min)
y_dm        = qt.ket2dm(y)
y_min_dm    = qt.ket2dm(y_min)

vacuum      = qt.basis(2,0)
photon      = qt.basis(2,1)
vacuum_2    = qt.basis(3,0)
photon_2    = qt.basis(3,1)
photon2_2   = qt.basis(3,2)
vacuum_dm   = qt.ket2dm(vacuum)
photon_dm   = qt.ket2dm(photon)
vacuum_2_dm = qt.ket2dm(vacuum_2)
photon_2_dm = qt.ket2dm(photon_2)

eye         = qt.qeye(2)
tp          = qt.tensor

def alpha_ket(alpha):
    return np.sqrt(alpha) * down + np.sqrt(1 - alpha) * up

def alpha_dm(alpha):
    return qt.ket2dm(alpha_ket(alpha))

def vacuum_dim(dim):
    return qt.basis(dim, 0)

def photon_dim(dim):
    return qt.basis(dim, 1)

def wcs_dim(dim, alpha):
    return qt.coherent(dim, alpha)

def create_dim(dim):
    return qt.create(dim)

def displace_dim(dim, alpha):
    return qt.displace(dim, alpha)