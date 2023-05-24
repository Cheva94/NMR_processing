import readbruker as rb
import numpy as np
import scipy.fft as FT


def readFID(fileDir):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    # read in the bruker formatted data
    dic, rawdata = rb.read(fileDir)
    
    SW = dic["acqus"]["SW_h"] # Hz
    nS = dic["acqus"]["NS"]
    RDT = dic["acqus"]["DE"] # us
    RG = dic["acqus"]["RG"] # dB
    
    RD = dic["acqus"]["D"][1] # s
    p90 = dic["acqus"]["P"][1] # us
    att = dic["acqus"]["PL"][1] # dB
    
    filter = 69 # Puntos que no sirven tema de filtro digital
    SGL = rawdata[filter:]
    SGL[0] = 2*SGL[0] # arreglo lo que el bruker rompe
    nP = len(SGL) # Cantidad total de puntos que me quedan en la FID
    t = np.array([x * 1E6 / SW for x in range(nP)]) # eje temporal en microsegundos

    return t, SGL, nP, SW, nS, RDT, RG, att, RD, p90


def PhCorrFID(SGL):
    '''
    Corrección de fase.
    '''

    maxVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        SGL_ph = SGL * np.exp(1j * tita)
        maxVal[i] = np.max(SGL_ph.real[0:30])
    SGL *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))
    
    return SGL


def spectrum(SGL, nP, SW):
    '''
    Creación del espectro.
    '''

    # Preparación del espectro
    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=1/SW)) # Hz scale
    CS = freq / 300 # ppm for Bruker scale
    spec = np.flip(FT.fftshift(FT.fft(SGL, n = zf)))

    return CS, spec

def writeFID(t, SGL, nP, CS, spec, Out):

    with open(f'{Out}_td.csv', 'w') as f:
        f.write("t [ms]\tRe[FID]\tIm[FID] \n")
        for i in range(nP):
            f.write(f'{t[i]:.6f}\t{SGL.real[i]:.6f}\t{SGL.imag[i]:.6f} \n')

    # Preparación del espectro
    mask = (CS>-5)&(CS<5)
    CS = CS[mask]
    spec = spec[mask]

    with open(f'{Out}_fd.csv', 'w') as f:
        f.write("CS [ppm]\tRe[spec]\tIm[spec] \n")
        for i in range(len(CS)):
            f.write(f'{CS[i]:.6f}\t{spec.real[i]:.6f}\t{spec.imag[i]:.6f} \n')