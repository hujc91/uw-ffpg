#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def inner_scaling(u, y, dudyw, mu, rho):
    '''
    Compute the dimensionless velocity profile of wall-bounded shear flow based on inner-scaling parameters

    Inputs:
    u - velocity of the profile(s)
    y - coordinates of the profile(s)
    dudyw - wall-normal velocity gradient evaluated at the wall
    (In case of multiple profiles, each profile should be stored in a column.)

    mu - dynamic viscosity of the fluid
    rho - density of the fluid

    Outputs:
    up - dimensionless velocity scaled by inner-scaling velocity scale (u+)
    yp - dimensionless coordiates scaled by inner-scaling length scale (y+)
    '''

    # Wall shear velocity
    utau = np.sqrt(mu*dudyw/rho)
    # Wall scaling
    dv = mu/rho/utau
    # u+ and y+
    up = u/utau; yp = y/dv

    return up, yp
