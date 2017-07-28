#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def outer_scaling(u, y, U, Y):
    '''
    Compute the dimensionless velocity profile based on outer-scaling parameters

    Inputs:
    u - velocity of the profile(s)
    y - coordinates of the profile(s)
    (In case of multiple profiles, each profile should be stored in a column.)

    U - velocity scale in outer-scaling
    Y - length scale in outer-scaling

    Outputs:
    uo - dimensionless velocity scaled by outer-scaling velocity scale
    yo - dimensionless coordiates scaled by outer-scaling length scale
    '''
    uo = u/U
    yo = y/Y

    return uo, yo
