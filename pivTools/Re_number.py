#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def Re_number(*args):
    '''
    Compute Reynolds number

    Inputs:
    args[0] - characteristic velocity scale
    args[1] - density of the fluid
    args[2] - characteristic length scale
    args[3] - dynamic viscosity of the fluid
    args[4] - virtual origin for turbulent flow (if applicable)

    Outputs:
    Re - calculated Reynolds number
    '''

    # Check inputs
    nargin = len([*args])
    if nargin ==4:
        u = args[0]; rho = args[1]; L = args[2]; mu = args[3]; x0_virt = 0
    elif nargin == 5:
        u = args[0]; rho = args[1]; L = args[2]; mu = args[3]; x0_virt = args[4]
    else:
        print('Wrong number of inputs!')

    Re = rho*u*(L-x0_virt)/mu

    return Re
