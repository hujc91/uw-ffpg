#---------- Forematters---------------------------------------------
from scipy.interpolate import griddata
import numpy as np
#-------------------------------------------------------------------

def stitch(x1, y1, u1, v1, x2, y2, u2, v2, blend):
    '''
    Stitch two vector fields with overlapping regions.

    Inputs:
    x1, y1 - coordinates of the first fields in meshgrid form
    u1, v1 - scaler values of the first field
    x2, y2 - coordinates of the second fields in meshgrid form
    u2, v2 - scaler values of the second field
    blend - stitching method in the overlapping region: 'none','average', 'cubic' or 'cosine'

    Outputs:
    x, y - coordinates of the stitched field in meshgrid form
    u, v - scaler values of the stitched field

    Notes:
    - PIV mask region must takes the value of zeros

    Disclaimer:
    This Py-code is translated from the Matlab-code shared on FMRL SharePoint, written by J. McClure
    '''
    # -------- Create 2D field for stitched image ------------------
    dx = np.max((np.abs(x1[0,0]-x1[0,1]), np.abs(x2[0,0]-x2[0,1])));
    dy = np.max((np.abs(y1[0,0]-y1[1,0]), np.abs(y2[0,0]-y2[1,0])));

    xmin = np.min((x1.min(), x2.min()))
    xmax = np.max((x1.max(), x2.max()))
    xmax = xmax + dx*((xmax-xmin)/dx-np.floor((xmax-xmin)/dx))

    ymin = np.min((y1.min(), y2.min()))
    ymax = np.max((y1.max(), y2.max()))
    ymax = ymax + dy*((ymax-ymin)/dy-np.floor((ymax-ymin)/dy))

    [x, y] = np.meshgrid(np.linspace(xmin,xmax, num = np.int(np.round(np.abs(xmax-xmin)/dx+1))),\
                         np.linspace(ymin,ymax, num = np.int(np.round(np.abs(xmax-xmin)/dx+1))) )
    # -------- Region of overlap -----------------------------------
    # Right boundary
    Ab = x[0,:]-np.min((x1.max(), x2.max()))
    Ab[Ab<0] = 10e8
    xov2 = np.argmin(np.abs(Ab)) -1
    # Left boundary
    Ac = x[0,:]-np.max((x1.min(), x2.min()))
    Ac[Ac>0] = 10e8
    xov1 = np.argmin(np.abs(Ac)) +1
    # Bottom boundary
    Ad = y[:,0]-np.max((y1.min(), y2.min()))
    Ad[Ad>0] = 10e8
    yov1 = np.argmin(np.abs(Ad))
    # Top boundary
    Ae = y[:,0]-np.min((y1.max(), y2.max()))
    Ae[Ae<0] = 10e8
    yov2 = np.argmin(np.abs(Ae))


    #-------- Interpolated region in both FOV -----------------------
    x11 = np.argmin( np.abs( x[0,:]-x1.min() ) )
    x12 = np.argmin( np.abs( x[0,:]-x1.max() ) )
    y11 = np.argmin( np.abs( y[:,0]-y1.min() ) )
    y12 = np.argmin( np.abs( y[:,0]-y1.max() ) )

    Af = x[0,:]-x2.min()
    Af[Af<0] = 10e8

    x21 = np.argmin( np.abs(Af) )
    x22 = np.argmin( np.abs( x[0,:]-x2.max() ) )
    y21 = np.argmin( np.abs( y[:,0]-y2.min() ) )
    y22 = np.argmin( np.abs( y[:,0]-y2.max() ) )

    #-------- Fill the interpolated velocity space ------------------
    u = np.zeros(np.shape(x))
    v = np.zeros(np.shape(x))

    # Reshape x1, y1, u1, v1, x2, y2, u2, v2 for interpolation
    x1_intp = np.reshape(x1, x1.size); y1_intp = np.reshape(y1, y1.size)
    u1_intp = np.reshape(u1, u1.size); v1_intp = np.reshape(v1, v1.size)

    x2_intp = np.reshape(x2, x2.size); y2_intp = np.reshape(y2, y2.size)
    u2_intp = np.reshape(u2, u2.size); v2_intp = np.reshape(v2, v2.size)

    # Interpolate velocity measurements of each FOV onto the new mesh
     # The new mesh extends each field of view to the outmost x and y
    u[y11:y12, x11:x12] = griddata((x1_intp, y1_intp), u1_intp, (x[y11:y12, x11:x12], y[y11:y12, x11:x12]), method = 'cubic')
    v[y11:y12, x11:x12] = griddata((x1_intp, y1_intp), v1_intp, (x[y11:y12, x11:x12], y[y11:y12, x11:x12]), method = 'cubic')

    u[y21:y22, x21:x22] = griddata((x2_intp, y2_intp), u2_intp, (x[y21:y22, x21:x22], y[y21:y22, x21:x22]), method = 'cubic')
    v[y21:y22, x21:x22] = griddata((x2_intp, y2_intp), v2_intp, (x[y21:y22, x21:x22], y[y21:y22, x21:x22]), method = 'cubic')

    # Interpolate velocity measurements of each FOV onto the overlapping region on the new mesh
    uov1 = griddata((x1_intp, y1_intp), u1_intp, (x[yov1:yov2,xov1:xov2], y[yov1:yov2,xov1:xov2]), method = 'cubic')
    vov1 = griddata((x1_intp, y1_intp), v1_intp, (x[yov1:yov2,xov1:xov2], y[yov1:yov2,xov1:xov2]), method = 'cubic')

    uov2 = griddata((x2_intp, y2_intp), u2_intp, (x[yov1:yov2,xov1:xov2], y[yov1:yov2,xov1:xov2]), method = 'cubic')
    vov2 = griddata((x2_intp, y2_intp), v2_intp, (x[yov1:yov2,xov1:xov2], y[yov1:yov2,xov1:xov2]), method = 'cubic')


    #------- Select blending function for overlapping region ------------------
    ## Generate weighting functions


    # No blending
    if blend == 'none':
        wgt1b = np.ones((1, np.abs(xov2-xov1)))
        wgt2b = np.zeros((1, np.abs(xov2-xov1)))
    # Simple average
    elif blend == 'average':
        wgt1b = 0.5*np.ones((1, np.abs(xov2-xov1)))
        wgt2b = 0.5*np.ones((1, np.abs(xov2-xov1)))
    # Linear weight
    elif blend == 'cubic':
        wgt1b = np.linspace(1, 0, num = (np.abs(xov2-xov1)))
        wgt2b = np.linspace(0, 1, num = (np.abs(xov2-xov1)))
    # Cosine weight
    elif blend == 'cosine':
        wgt1b = -0.5*np.cos(np.linspace(0, np.pi, num = (np.abs(xov2-xov1)))) + 0.5
        wgt2b =  0.5*np.cos(np.linspace(0, np.pi, num = (np.abs(xov2-xov1)))) + 0.5
    # Invalid input
    else:
        wgt1b = np.zeros((1, np.abs(xov2-xov1)))
        wgt2b = np.zeros((1, np.abs(xov2-xov1)))
        print('Invalid blending method!')

    wgt1 = np.matlib.repmat(wgt1b, (np.abs(yov2-yov1)), 1)
    wgt2 = np.matlib.repmat(wgt2b, (np.abs(yov2-yov1)), 1)


    # Blending
    uov1c = uov1*wgt2
    uov2c = uov2*wgt1

    vov1c = vov1*wgt2
    vov2c = vov2*wgt1

    u[yov1:yov2, xov1:xov2] = uov1c + uov2c
    v[yov1:yov2, xov1:xov2] = vov1c + vov2c
    return x, y, u, v
