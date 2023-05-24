import readbruker as rb
import numpy as np
import pandas as pd
import scipy.fft as FT

# FID related functions


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
        maxVal[i] = np.max(SGL_ph[0:30].real)
    SGL *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))
    
    return SGL


def specFID(SGL, nP, SW):
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

    with open(f'{Out}FID_td.csv', 'w') as f:
        f.write("t [ms]\tRe[FID]\tIm[FID] \n")
        for i in range(nP):
            f.write(f'{t[i]:.6f}\t{SGL[i].real:.6f}\t{SGL[i].imag:.6f} \n')

    mask = (CS>-5)&(CS<5)
    CS = CS[mask]
    spec = spec[mask]

    with open(f'{Out}FID_fd.csv', 'w') as f:
        f.write("CS [ppm]\tRe[spec]\tIm[spec] \n")
        for i in range(len(CS)):
            f.write(f'{CS[i]:.6f}\t{spec[i].real:.6f}\t{spec[i].imag:.6f} \n')


# DQ related functions


def readDQ(fileDir):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    # read in the bruker formatted data
    dic, rawdata = rb.read(fileDir)

    vd = []
    vdlist = pd.read_csv(f'{fileDir}/vdlist', header=None).to_numpy()
    for k in range(len(vdlist)):
        vd.append(vdlist[k][0].split('u')[0])
    vd = np.array(vd).astype(float) # us

    SW = dic["acqus"]["SW_h"] # Hz
    nS = dic["acqus"]["NS"]
    RDT = dic["acqus"]["DE"] # us
    RG = dic["acqus"]["RG"] # dB
    
    RD = dic["acqus"]["D"][1] # s
    evol = dic["acqus"]["D"][2] # s
    zFilter = dic["acqus"]["D"][10] # s
    p90 = dic["acqus"]["P"][1] # us
    att = dic["acqus"]["PL"][1] # dB
    DQfilter = None

    filter = 69 # Puntos que no sirven tema de filtro digital
    SGL = rawdata[:, filter:]
    SGL[:, 0] = 2*SGL[:, 0] # arreglo lo que el bruker rompe
    nP = len(SGL[0, :]) # Cantidad total de puntos que me quedan en la FID
    t = np.array([x * 1E6 / SW for x in range(nP)]) # eje temporal en microsegundos

    return t, SGL, nP, SW, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, DQfilter


def PhCorrDQ(SGL, lenvd):
    '''
    Corrección de fase.
    '''

    maxVal = {}
    for k in range(lenvd):
        for i in range(360):
            tita = np.deg2rad(i)
            SGL_ph = SGL[k, :] * np.exp(1j * tita)
            maxVal[i] = np.max(SGL_ph[0:30].real)
        SGL[k, :] *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))
    
    return SGL


def specDQ(SGL, nP, SW, lenvd):
    '''
    Creación del espectro.
    '''

    # Preparación del espectro
    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=1/SW)) # Hz scale
    CS = freq / 300 # ppm for Bruker scale
    spec = []

    for k in range(lenvd):
        spec.append(np.flip(FT.fftshift(FT.fft(SGL[k, :], n = zf))))
    spec = np.array(spec)

    return CS, spec


def writeDQ_verbose(t, SGL, nP, CS, spec, Out, lenvd):

    mask = (CS>-5)&(CS<5)
    CS = CS[mask]
    spec = spec[:, mask]

    for k in range(lenvd):
        with open(f'{Out}FID_td_{k}.csv', 'w') as f:
            f.write("t [ms]\tRe[FID]\tIm[FID] \n")
            for i in range(nP):
                f.write(f'{t[i]:.6f}\t{SGL[k, i].real:.6f}\t{SGL[k, i].imag:.6f} \n')

        with open(f'{Out}FID_fd_{k}.csv', 'w') as f:
            f.write("CS [ppm]\tRe[spec]\tIm[spec] \n")
            for i in range(len(CS)):
                f.write(f'{CS[i]:.6f}\t{spec[k, i].real:.6f}\t{spec[k, i].imag:.6f} \n')


def writeDQ(vd, fid00, fidPts, fidPtsSD, pArea, Out):

    with open(f'{Out}DQ_bu.csv', 'w') as f:
            f.write("vd [us]\tFID00\tFID Pts\tFID Pts (SD)\tPeak Area\n")
            for i in range(len(vd)):
                f.write(f'{vd[i]:.1f}\t{fid00[i]:.6f}\t{fidPts[i]:.6f}\t{fidPtsSD[i]:.6f}\t{pArea[i]:.6f}\n')