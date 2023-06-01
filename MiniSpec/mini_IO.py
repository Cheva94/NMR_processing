import numpy as np
import pandas as pd
import scipy.fft as FT

# FID related functions


def readFID(File):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    data = pd.read_csv(File, header = None, sep='\t').to_numpy()

    t = data[:, 0] # In ms
    DW = t[1] - t[0] # Dwell time
    nP = len(t) # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    SGL = Re + Im * 1j

    pAcq = pd.read_csv(File.split(".txt")[0]+'_acqs.txt', header = None,  sep='\t')
    nS, RDT, RG, att, RD, p90 = pAcq.iloc[0, 1], pAcq.iloc[1, 1], pAcq.iloc[2, 1], pAcq.iloc[3, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1]

    return t, SGL, nP, DW, nS, RDT, RG, att, RD, p90


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


def NormFID(SGL, RG):
    '''
    Normalización por ganancia.
    '''

    norm = 1 / (6.32589E-4 * np.exp(RG/9) - 0.0854)
    return SGL * norm


def specFID(SGL, nP, DW):
    '''
    Creación del espectro.
    '''

    # Preparación del espectro
    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
    CS = freq / 20 # ppm for Minispec scale
    spec = np.flip(FT.fftshift(FT.fft(SGL, n = zf)))

    return CS, spec


def writeFID(t, SGL, nP, CS, spec, Out, mlim):

    with open(f'{Out}_td.csv', 'w') as f:
        f.write("t [ms]\tRe[FID]\tIm[FID] \n")
        for i in range(nP):
            f.write(f'{t[i]:.6f}\t{SGL.real[i]:.6f}\t{SGL.imag[i]:.6f} \n')

    mask = (CS>-mlim)&(CS<mlim)
    CS = CS[mask]
    spec = spec[mask]

    with open(f'{Out}_fd.csv', 'w') as f:
        f.write("CS [ppm]\tRe[spec]\tIm[spec] \n")
        for i in range(len(CS)):
            f.write(f'{CS[i]:.6f}\t{spec[i].real:.6f}\t{spec[i].imag:.6f} \n')