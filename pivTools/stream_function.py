#---------- Forematters---------------------------------------------
import numpy as np
from scipy import integrate
#-------------------------------------------------------------------


def stream(x, y, u, v):
    '''
    Calculate stream function of velocity field, based on irrotational assumption
    Inputs:
    x, y - coordinates of the plane in meshgrid form
    u, v - velocity components in x, y direction
    Outputs:
    psi - value of stream function on meshgrid
    '''
    dy = y[1,0] - y[0,0]

    if dy<0:
        x = np.flipud(x)
        y = np.flipud(y)
        u = np.flipud(u)
        v = np.flipud(v)


    psi = integrate.cumtrapz(u, y[:,0], axis =0 , initial = u[0,0])\
           -integrate.cumtrapz(v[0, :], x[0,:],  initial = v[0,0])

    if dy<0:
        psi = np.flipud(psi)

    return psi
