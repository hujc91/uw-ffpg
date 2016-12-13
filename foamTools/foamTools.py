import numpy as np

def loadVector(fPath):
    '''
    Extract vector internal field from openFOAM format file
    fPath - OpenFOAM file path
    vec   - Vector field output, 3 columns corresponds to (x,y,z) velocity
            components, row number corresponds cell number in OpenFOAM
    '''
    f = open(fPath, 'r').readlines()

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
    Extract scaler internal fields from openFOAM format file
    fPath - OpenFOAM file path
    vec   - Scalar field output, single column, row number corresponds cell
            number in OpenFOAM
    '''
    f = open(fPath, 'r').readlines()

    for num, line in enumerate(f):
        if line.partition(' ')[0] == 'internalField':
            dataLine = num
            break

    dataNum = int(f[dataLine+1].partition(' ')[0])

    scr = np.zeros(dataNum)

    for num, line in enumerate(f[dataLine+3:dataLine+3+dataNum]):
        scr[num] = float(line.partition(' ')[0])

    return scr

def writeVector(fPath, tPath, uu, vv, ww):
    '''
    Write vector internal fields into openFOAM format file using tPath template
    fPath - write to file path
    tPath - template file path
    uu    - x component
    vv    - y component
    ww    - z component
    '''
    fTemplate = open(tPath, 'r')
    fWrite    = open(fPath, 'w')

    for line in fTemplate.readlines():
        fWrite.write(line)
        if line.partition(' ')[0] == 'internalField':
            N = len(uu)
            fWrite.write('{}\n'.format(N))
            fWrite.write('(\n')
            for i in range(N):
                fWrite.write('({:e} {:e} {:e})\n'.format(uu[i],vv[i],ww[i]))
            fWrite.write(')\n')
            fWrite.write(';\n')

def writeScalar(fPath, tPath, ss):
    '''
    Write vector internal fields into openFOAM format file using tPath template
    fPath - write to file path
    tPath - template file path
    ss    - scalar field
    '''
    fTemplate = open(tPath, 'r')
    fWrite    = open(fPath, 'w')

    for line in fTemplate.readlines():
        fWrite.write(line)
        if line.partition(' ')[0] == 'internalField':
            N = len(ss)
            fWrite.write('{}\n'.format(N))
            fWrite.write('(\n')
            for i in range(N):
                fWrite.write('{:e}\n'.format(ss[i]))
            fWrite.write(')\n')
            fWrite.write(';\n')
