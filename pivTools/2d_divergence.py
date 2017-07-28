#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def divergence(u, v, x, y):
    '''
    Calculate in-plane divergence of velocity field
    Inputs:
    x, y - coordinates of the plane in meshgrid form
    u, v - velocity components in x, y direction
    Outputs:
    div - in-plane divergence
    '''
    dx = abs(x[0,0]-x[0,1])
    dy = abs(y[0,0]-y[1,0])
    [dudx, dudy] = np.gradient(u, dx, dy)
    [dvdx, dvdy] = np.gradient(v, dx, dy)
    div = dudx + dvdy
    return div
