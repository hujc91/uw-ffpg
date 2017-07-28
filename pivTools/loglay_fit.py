#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def loglay_fit(up, yp, ypthresL, ypthresH):
    '''
    Curve fit for velocity profiles in the log-law layer of a wall-bounded shear flow
    u+ = a*log(y+) + b (eq.1)

    Inputs:
    up - dimensionless velocity scaled by inner-scaling velocity scale (u+)
    yp - dimensionless coordiates scaled by inner-scaling length scale (y+)
    ypthresL - lower bound of the log-law range (typical value range: [20,35])
    ypthresH - upper bound of the log-law range (typical value range: [50,80])

    Outputs:
    u_grwrt - curve fit coefficient (a) in eq.1
    u_intcp - curve fit interception (b) in eq.1

    Note:
    For fully developed turbulent flow over a flat surface:
    a ~= 2.43
    b ~ = 5.2
    '''


    # yplus index
    idxloglay = np.where((yp>=ypthresL)&(yp<=ypthresH)==True)

    # Take natural logarithm of u and y
    ufit = up[idxloglay]
    yfit = np.log(yp[idxloglay])

    # Estimate the slope for least square regression
    idxlsq0 = np.int(np.max(np.argmax(ufit)))
    idxlsq1 = np.int(np.min(np.argmin(ufit)))
    idxlsq2 = np.int(np.size(ufit)/2)

    du = ufit[idxlsq0]-ufit[idxlsq1]
    dy = yfit[idxlsq0]-yfit[idxlsq1]
    dudy = du/dy

    A = np.vstack([yfit, dudy*np.ones(len(yfit))]).T
    u_grwrt, u_intcp = np.linalg.lstsq(A, ufit)[0]

    # Correction to the interception
    u_offset0 = ufit[idxlsq0] - (u_grwrt*yfit[idxlsq0]+u_intcp)
    u_offset1 = ufit[idxlsq1] - (u_grwrt*yfit[idxlsq1]+u_intcp)
    u_offset2 = ufit[idxlsq2] - (u_grwrt*yfit[idxlsq2]+u_intcp)
    u_intcp = u_intcp + (u_offset0 + u_offset1 + u_offset2)/3

    return u_grwrt,u_intcp
