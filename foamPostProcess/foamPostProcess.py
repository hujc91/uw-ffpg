import numpy as np

def loadVector(fPath):
    '''
    Extract vector fields from openFOAM format files
    fName - file pa
    '''
    f = open(fPath).readlines()

    for num, line in enumerate(f):
        if line.partition(' ')[0] == 'internalField':
            dataLine = num
            break

    dataNum = int(f[dataLine+1].partition(' ')[0])

    vec = np.zeros((dataNum,3))

    for num, line in enumerate(f[dataLine+3:dataLine+3+dataNum]):
        vecTXT = line.split(' ')

        vec[num,0] = float(vecTXT[0].strip('('))
        vec[num,1] = float(vecTXT[1])
        vec[num,2] = float(vecTXT[2].strip(')\n'))

    return vec

def loadScalar(fPath):
    '''
    Extract scaler fields from openFOAM format file
    '''
    f = open(fPath).readlines()

    for num, line in enumerate(f):
        if line.partition(' ')[0] == 'internalField':
            dataLine = num
            break

    dataNum = int(f[dataLine+1].partition(' ')[0])

    scr = np.zeros(dataNum)

    for num, line in enumerate(f[dataLine+3:dataLine+3+dataNum]):
        scr[num] = float(line.partition(' ')[0])

    return scr

def calKineticEnergy(UU, dL, ax=0):
    '''
    Calculate circultaion from numerical data loaded from OpenFOAM files
        UU - velocity fields in 4 columns format [uu,vv,ww,tt]
        dL - element length
        ax - integrate about array axis
    '''

    dA  = dL**2
    UU2 = uu**2 + vv**2
    KE  = 0.5*np.sum(UU2, axis=ax)*dA

    return KE

def calEnstrophy(ww, dL, ax=0):
    '''
    Calculate circultaion from numerical data
        ww - vorticity field
        dL  - element length
        ax - integrate about array axis
    '''
    dA  = dL**2
    ww2 = ww**2
    Ens = 0.5*np.sum(ww2, axis=ax)*dA

    return Ens
