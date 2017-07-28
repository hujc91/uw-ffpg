#---------- Forematters---------------------------------------------
import numpy as np
#-------------------------------------------------------------------

def stat(u, v, x, y):
    '''
    Calculate time-averaged velocity, velocity fluctuation mean square, velocity covariance,
    turbulent kinetic energy and turbulence production

    Inputs:
    x, y - coordinates in meshgrid form
    u, v - time series velocity components in 3D array

    Outputs:
    uavg, vavg - time-averaged velocity components
    Vavg - time-averaged velocity magnitude
    uu, vv - velocity magnitude mean square
    uv - velocity covariance
    tke - turbulent kinetic energy
    genT - turbulence production
    '''
    uavg = np.nanmean(u, axis=2); vavg = np.nanmean(v, axis = 2); Vavg = np.sqrt(uavg**2+vavg**2);
    uu = np.nanstd(u, 2)**2; vv = np.nanstd(v, 2)**2

    uavg3d = uavg[:,:,np.newaxis]
    vavg3d = vavg[:,:,np.newaxis]

    uv = np.nanmean(((u-uavg3d)*(v-vavg3d)),axis = 2)

    [dudx, dudy] = np.gradient(uavg, hx, hy)
    [dvdx, dvdy] = np.gradient(vavg, hx, hy)

    tke = 0.5*(uu+vv)
    genT = uu*dudx+uv*(dudy+dvdx)+vv*dvdy

    return uavg, vavg, Vavg, uu, vv, uv, tke, genT
