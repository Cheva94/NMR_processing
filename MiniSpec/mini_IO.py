import numpy as np
import pandas as pd
import scipy.fft as FT


################################################################################
###################### FID related functions ###################################
################################################################################


def readFID(path):
    '''
    Reads signal file.
    '''

    data = pd.read_csv(path, header = None, sep='\t').to_numpy()

    t = data[:, 0] # In ms
    DW = t[1] - t[0] # Dwell time
    nP = len(t) # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    SGL = Re + Im * 1j

    return t, SGL, nP, DW


def readFIDparams(path):
    '''
    Reads acquisition parameters.
    '''

    pAcq = pd.read_csv(path.split(".txt")[0]+'_acqs.txt', 
                       header = None,  sep='\t')
    
    RDT, att, RG = pAcq.iloc[1, 1], pAcq.iloc[3, 1], pAcq.iloc[2, 1]
    nS, RD, p90 = pAcq.iloc[0, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1]

    return int(nS), int(RDT*1E3), int(RG), int(att), float(RD), float(p90)


def PhCorrFID(SGL):
    '''
    Phase correction.
    '''

    maxVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        SGL_ph = SGL * np.exp(1j * tita)
        maxVal[i] = np.max(SGL_ph[2].real)
    SGL *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))

    return SGL


def NormFID(SGL, RG):
    '''
    Normalization by receiver gain.
    '''

    norm = 1 / (6.32589E-4 * np.exp(RG/9) - 0.0854)

    return SGL * norm


def specFID(SGL, nP, DW):
    '''
    Generates spectrum.
    '''

    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
    CS = freq / 20 # ppm for Minispec scale
    spec = np.flip(FT.fftshift(FT.fft(SGL, n = zf)))

    return CS, spec


def writeFID(t, SGL, nP, CS, spec, Out, ppm):
    '''
    Saves processed data.
    '''

    with open(f'{Out}_TimeDom.csv', 'w') as f:
        f.write("t [ms]\tRe[FID]\t\tIm[FID]\n")
        for i in range(nP):
            f.write(f'{t[i]:.6f}\t{SGL.real[i]:.6f}\t{SGL.imag[i]:.6f} \n')

    mask = (CS>-ppm)&(CS<ppm)
    CS = CS[mask]
    spec = spec[mask]

    with open(f'{Out}_FreqDom.csv', 'w') as f:
        f.write("CS [ppm]\tRe[spec]\tIm[spec]\n")
        for i in range(len(CS)):
            f.write(f'{CS[i]:.6f}\t{spec[i].real:.6f}\t{spec[i].imag:.6f} \n')


################################################################################
################################################################################
################################################################################