#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def power_fit(x, y):
    '''
    Compute the power law between Re_x and Re_delta in wall bounded shear flow
    Re_delta = a*Re_x^b (eq.1)

    Inputs:
    x - Reynolds number based on streamwise locations (Re_x)
    y - Reynolds number based on local boundary layer geometric thickness (Re_delta)

    Outputs:
    pf_expon - exponent (b) in eq.1
    pf_coeff - coefficient (a) in eq.1
    '''

    # Take natural logarithm of x and y
    xfit = np.log(x)
    yfit = np.log(y)

    # Estimate the slope for least square regression
    idxlsq0 = np.int(np.max(np.argmax(yfit)))
    idxlsq1 = np.int(np.min(np.argmin(yfit)))
    idxlsq2 = np.int(np.size(yfit)/2)

    dy = yfit[idxlsq0]-yfit[idxlsq1]
    dx = xfit[idxlsq0]-xfit[idxlsq1]
    dydx = dy/dx

    A = np.vstack([xfit, dydx*np.ones(len(xfit))]).T
    y_grwrt, y_intcp = np.linalg.lstsq(A, yfit)[0]

    # Correction to the interception
    y_offset0 = yfit[idxlsq0] - (y_grwrt*xfit[idxlsq0]+y_intcp)
    y_offset1 = yfit[idxlsq1] - (y_grwrt*xfit[idxlsq1]+y_intcp)
    y_offset2 = yfit[idxlsq2] - (y_grwrt*xfit[idxlsq2]+y_intcp)
    y_intcp = y_intcp + (y_offset0 + y_offset1 + y_offset2)/3

    # Convert the linear fit to power-law fit
    pf_expon = y_grwrt
    pf_coeff = np.exp(y_intcp)

    return pf_expon, pf_coeff
