import numpy as np

def POD(u, v):
    '''
    Compute the proper orthogonal decomposition (POD) of 2-component vector fields based on singular value decomposition (SVD)
    Inputs:
    u, v - velocity components
           Snapshots at one time instant are stored in axis = 0&1
           Time series in axis = 2

    Outputs:
    psi_u, psi_v - Spatial modes for u, v
                   Each mode is stored in 2D matrix axis = 0&1
                   Mode series in axis = 2
    E - Modal energy (in percentage)
    CumE - Cumulative modal energy (in percentage)
    Rsvd - Temporal coefficients
           Temporal coefficients for each mode at N snapshots are stored in columns
    '''
    # Format the velocity field snapshots into column vectors
    uavg = np.nanmean(u,2); uavg3d = uavg[:,:,np.newaxis]
    ufluc = u-uavg3d
    vavg = np.nanmean(v,2); vavg3d = vavg[:,:,np.newaxis]
    vfluc = v-vavg3d


    ursh = ufluc.reshape((np.size(x), v.shape[2]))
    vrsh = vfluc.reshape((np.size(x), v.shape[2]))

    Ufluc = np.concatenate((ursh, vrsh), axis = 0)

    ###############################################################
    # Singular value decomposition (Core code)
    [Lsvd, ssvd, Rsvd] = np.linalg.svd(Ufluc, full_matrices = False)
    ###############################################################


    # Format the modal energy for output
    E = ssvd**2/np.sum(ssvd**2)   # Modal energy (in percentage)
    CumE = np.cumsum(E)           # Cumulative modal energy (in percentage)

    # Format the POD modes/temporal coefficients for output
    Ssvd = np.diag(ssvd)          # Singular value in diagonal matrix
    Qsvd = np.dot(Lsvd, Ssvd)/ssvd[np.newaxis,:]
    ## Spatial mode $\Phi_k(x)$ in POD is the $k_{th}$ column of matrix Qsvd (each column needs normalizing by ssvd)
    ## Temporal coefficient $a_k(t)$ in POD is the $k_{th}$ row of matrix Rsvd
    psiu = Qsvd[0:np.size(uavg), :]
    psi_u = psiu.reshape(np.shape(u))
    psiv = Qsvd[np.size(uavg):,:]
    psi_v = psiv.reshape(np.shape(v))

    return psi_u, psi_v, E, CumE, Rsvd
