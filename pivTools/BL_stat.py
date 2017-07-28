#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def BL_stat(*args):
    '''
    Compute boundary layer (BL) statistics in wall-bounded shear flow

    Inputs:
    args[0] - streamwise velocity in meshgrid form
    args[1] - wall-normal coordinates in meshgrid form
    args[2] - free stream velocity (if applicable)

    Outputs:
    delta - BL geometric thickness
    deltaS - BL displacement thickness
    theta - BL momentum thickness
    H - BL shape factor
    umax - maximum streamwise velocity in the profile at a given wall location
    ymax - wall-normal location of umax
    yhalf - jet half-width at a given wall location (if applicable, e.g. in wall-jet)
    dudyw - streamwise velocity gradient evaluated at the wall
    '''
    # Check inputs
    nargin = len([*args])
    if nargin ==2:
        u = args[0]; yy = args[1]; Ucrit = []
    elif nargin == 3:
        u = args[0]; yy = args[1]; Ucrit = 0.99* args[2]
    else:
        print('Wrong number of inputs!')

    # Format the storage of the coordinate system:
    # Increase in index indicating increase in y coordinate
    dy = yy[1,0] - yy[0,0]
    if dy<0:
        yy = np.flipud(yy); u = np.flipud(u)

    # Initialize output variables
    delta = np.zeros([1, np.shape(u)[1]])
    deltaS = np.zeros([1, np.shape(u)[1]])
    theta = np.zeros([1, np.shape(u)[1]])
    H = np.zeros([1, np.shape(u)[1]])
    umax = np.zeros([1, np.shape(u)[1]])
    ymax = np.zeros([1, np.shape(u)[1]])
    yhalf = np.zeros([1, np.shape(u)[1]])
    dudyw = np.zeros([1, np.shape(u)[1]])


    # Loop through velocity profiles at each downstream locations
    for kk in range(0, np.shape(u)[1]):
        # Streamwise velocity profile at the current location
        uT =  u[:, kk]
        y =  yy[:, kk]

        ## Local maximum velocity and wall-normal location
        umax[0, kk] = np.max(uT)
        ymax[0, kk] = y[np.argmax(uT)]
        # Local critical velocity: 0.99 of the free stream|| 0.99 of the local maximum
        if nargin <3:
            Ucrit = 0.99*np.max(uT)
        # Wall-normal location where u just beyond the critical velocity U
        ycrit = np.min(y[np.where(uT >= Ucrit)])
        idxcrit = np.where(y==ycrit)
        idxcrit = np.int(idxcrit[0])
        ucrit = uT[idxcrit]
        # Linear interpolation to find the location of Ucrit
        intpcrit = (ucrit - Ucrit)/(ucrit - uT[idxcrit-1])

        ## BL geometric thickness
        delta[0, kk] = ycrit - intpcrit*(ycrit - y[idxcrit-1])


        ## BL displacement thickness
        deltaS[0, kk] = np.trapz( (1-uT[0:idxcrit]/Ucrit), x = y[0:idxcrit] )

        ## BL momentum thickness
        theta[0, kk] = np.trapz( ((1-uT[0:idxcrit]/Ucrit)*uT[0:idxcrit]/Ucrit), x = y[0:idxcrit] )

        ## Shape factor
        if theta[0, kk] == 0:
            H[0, kk] = np.nan
        else:
            H[0, kk] = deltaS[0, kk]/theta[0, kk]

        ## Jet half-width
        Uhalf = umax[0,kk]/2
        idxhalf = np.min( np.where((uT<=Uhalf)&(y>ymax[0,kk]) == True) )
        idxhalf = np.int(idxhalf)

        if np.size(idxhalf) == 0:
            yhalf[0,kk] = np.nan
        else:
            uhalf = uT[idxhalf]
            intphalf = (uhalf - Uhalf)/(uhalf - uT[idxhalf - 1])
            yhalf[0, kk] = y[idxhalf] + intphalf*( y[idxhalf] - y[idxhalf-1] )

        ## Velocity gradient at the wall
        p = np.polyfit( np.array([0,y[0], y[1], y[2]]), np.array([0,uT[0], uT[1], uT[2]]), 1)
        dudyw[0, kk] = p[0]


    return  delta, deltaS, theta, H, umax, ymax, yhalf, dudyw
