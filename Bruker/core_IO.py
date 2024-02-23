import core_readbruker as rb
import numpy as np
import pandas as pd
import scipy.fft as FT
from scipy.optimize import curve_fit

def r_square(x, y, f, popt):
    '''
    Pearson's coefficient.
    '''

    residuals = y - f(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return 1 - ss_res / ss_tot


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
        maxVal[i] = np.max(SGL_ph[0].real)
    SGL *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))
    
    return SGL


def specFID(SGL, nP, SW):
    '''
    Creación del espectro.
    '''

    # Preparación del espectro
    zf = FT.next_fast_len(4 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=1/SW)) # Hz scale
    CS = freq / 300 # ppm for Bruker scale
    spec = np.flip(FT.fftshift(FT.fft(SGL, n = zf)))

    return CS, spec


def writeFID(t, SGL, nP, CS, spec, Out, mlim):

    with open(f'{Out}FID_td.csv', 'w') as f:
        f.write("t [ms]\tRe[FID]\tIm[FID]\n")
        for i in range(nP):
            f.write(f'{t[i]:.6f}\t{SGL[i].real:.6f}\t{SGL[i].imag:.6f}\n')

    mask = (CS>-2*mlim)&(CS<2*mlim)
    CS = CS[mask]
    spec = spec[mask]

    with open(f'{Out}FID_fd.csv', 'w') as f:
        f.write("CS [ppm]\tRe[spec]\tIm[spec]\n")
        for i in range(len(CS)):
            f.write(f'{CS[i]:.6f}\t{spec[i].real:.6f}\t{spec[i].imag:.6f}\n')

def writeFID_acq(nS, RDT, RG, att, RD, p90, Out):

    with open(f'{Out}acq_param.csv', 'w') as f:
        f.write("Parámetros de adquisición:\n")
        f.write(f"\tCantidad de scans\t{nS:.0f}\n")
        f.write(f"\tTiempo entre scans\t{RD:.4f}\ts\n")
        f.write(f"\tTiempo muerto\t{RDT}\tus\n")
        f.write(f"\tGanancia\t{RG:.1f}\tdB\n")
        f.write(f"\tAtenuación\t{att:.0f}\tdB\n")
        f.write(f"\tAncho del pulso de 90\t{p90}\tus")


# Nutation related functions


def readNutac(fileDir):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    # read in the bruker formatted data
    dic, rawdata = rb.read(fileDir)

    vp = []
    vplist = pd.read_csv(f'{fileDir}/vplist', header=None).to_numpy()
    for k in range(len(vplist)):
        vp.append(vplist[k][0].split('u')[0])
    vp = np.array(vp).astype(float) # us

    nS = dic["acqus"]["NS"]
    RDT = dic["acqus"]["DE"] # us
    RG = dic["acqus"]["RG"] # dB
    
    RD = dic["acqus"]["D"][1] # s
    att = dic["acqus"]["PL"][1] # dB

    filter = 69 # Puntos que no sirven tema de filtro digital
    SGL = rawdata[:, filter:]

    return vp, SGL, nS, RDT, RG, att, RD


def writeNutac_acq(nS, RDT, RG, att, RD, vp, Out, lenvp):
    
    with open(f'{Out}acq_param.csv', 'w') as f:
        f.write("Barrido de pulsos:\n")
        f.write(f"\tRango de tiempo variable\t{vp[0]:.2f}\tus\t{vp[-1]:.2f}\tus\n")
        f.write(f"\tCantidad total de puntos\t{lenvp}\n")

        f.write("Otros parámetros de adquisición:\n")
        f.write(f"\tCantidad de scans\t{nS:.0f}\n")
        f.write(f"\tTiempo entre scans\t{RD:.4f}\ts\n")
        f.write(f"\tTiempo muerto\t{RDT}\tus\n")
        f.write(f"\tGanancia\t{RG:.1f}\tdB\n")
        f.write(f"\tAtenuación\t{att:.0f}\tdB\n")


def writeNutac(vp, fid00, fidPts, Out):

    with open(f'{Out}Nutac.csv', 'w') as f:
        f.write("vp [us]\tFID00\tFID Pts\n")
        for i in range(len(vp)):
            f.write(f'{vp[i]:.1f}\t{fid00[i]:.6f}\t{fidPts[i]:.6f}\n')


# CPMG related functions
            

def readCPMG(fileDir):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    # read in the bruker formatted data
    dic, rawdata = rb.read(fileDir)

    nS = dic["acqus"]["NS"]
    RDT = dic["acqus"]["DE"] # us
    RG = dic["acqus"]["RG"] # dB
    
    RD = dic["acqus"]["D"][1] # s
    att = dic["acqus"]["PL"][1] # dB

    p90 = dic["acqus"]["P"][1] # us
    p180 = dic["acqus"]["P"][2] # us

    nEcho = dic["acqus"]["TD"]
    nEcho = int(nEcho/2)
    tEcho = dic["acqus"]["D"][6] # s
    tEcho *= 2 * 1000 # ms

    t = np.linspace(tEcho, tEcho*nEcho, nEcho) # eje temporal en ms
    SGL = rawdata[:nEcho]

    return t, SGL, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho


def initKernel1D(nP, t, T2min, T2max):
    '''
    Initialize variables for Laplace transform.
    '''

    nBin = 150
    S0 = np.ones(nBin)
    T2 = np.logspace(T2min, T2max, nBin)
    K = np.zeros((nP, nBin))

    for i in range(nP):
        K[i, :] = np.exp(-t[i] / T2)

    return S0, T2, K


def exp_1(t, M0, T2):
    return M0 * np.exp(- t / T2)


def expFit_1(t, Z):
    '''
    Monoexponential fitting of CPMG decay.
    '''

    try:
        popt, pcov = curve_fit(exp_1, t, Z, bounds=(0, np.inf), p0=[Z[0], 100])
        perr = np.sqrt(np.diag(pcov))

        Mag_1 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', '']
        T2_1 = [fr'T2 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', '']
        
        r2 = r_square(t, Z, exp_1, popt)

        return Mag_1, T2_1, r2
    except RuntimeError:
        print("No se encontraron valores óptimos para el ajuste monoexponencial")
        return [0, 0], [0, 0], 0


def exp_2(t, M0_1, T2_1, M0_2, T2_2):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2)


def expFit_2(t, Z):
    '''
    Biexponential fitting of CPMG decay.
    '''

    try:
        popt, pcov = curve_fit(exp_2, t, Z, bounds=(0, np.inf), 
                           p0=[Z[0], 100, Z[0]/4, 50])
        perr = np.sqrt(np.diag(pcov))

        Mag_2 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', 
                    fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})']
        T2_2 = [fr'T2 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', 
                fr'T2 = ({popt[3]:.2f}$\pm${perr[3]:.2f}) ms']

        r2 = r_square(t, Z, exp_2, popt)

        return Mag_2, T2_2, r2
    except RuntimeError:
        print("No se encontraron valores óptimos para el ajuste biexponencial")
        return [0, 0], [0, 0], 0


def NLI_FISTA_1D(K, Z, alpha, S):
    '''
    Numeric Laplace inversion, based on FISTA.
    '''

    Z = np.reshape(Z, (len(Z), 1))
    S = np.reshape(S, (len(S), 1))

    KTK = K.T @ K
    KTZ = K.T @ Z
    ZZT = np.trace(Z @ Z.T)

    invL = 1 / (np.trace(KTK) + alpha)
    factor = 1 - alpha * invL

    Y = S
    tstep = 1
    lastRes = np.inf

    for iter in range(100000):
        term2 = KTZ - KTK @ Y
        Snew = factor * Y + invL * term2
        Snew[Snew<0] = 0

        tnew = 0.5 * (1 + np.sqrt(1 + 4 * tstep**2))
        tRatio = (tstep - 1) / tnew
        Y = Snew + tRatio * (Snew - S)
        tstep = tnew
        S = Snew

        if iter % 500 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            ObjFunc = ZZT - 2 * np.trace(S.T @ KTZ) + np.trace(S.T @ KTK @ S) + TikhTerm

            Res = np.abs(ObjFunc - lastRes) / ObjFunc
            lastRes = ObjFunc
            print(f'\t# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S[:, 0], iter


def fitLapMag_1D(t, T2, S, nP):
    '''
    Fits decay from T2 distribution.
    '''

    # t = range(nP)
    d = range(len(T2))
    M = []
    for i in range(nP):
        m = 0
        for j in d:
            m += S[j] * np.exp(- t[i] / T2[j])
        M.append(m)

    return M


def writeCPMG_acq(nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, Out):

    with open(f'{Out}acq_param.csv', 'w') as f:
        f.write("Parámetros de adquisición:\n")
        f.write(f"\tCantidad de scans\t{nS:.0f}\n")
        f.write(f"\tTiempo entre scans\t{RD:.4f}\ts\n")
        f.write(f"\tTiempo muerto\t{RDT}\tus\n")
        f.write(f"\tGanancia\t{RG:.1f}\tdB\n")
        f.write(f"\tAtenuación\t{att:.0f}\tdB\n")
        f.write(f"\tAncho del pulso de 90\t{p90}\tus")
        f.write(f"\tAncho del pulso de 180\t{p180}\tus")
        f.write(f"\tTiempo de eco\t{tEcho}\tms")
        f.write(f"\tNúmero de ecos\t{nEcho}")


def writeCPMG(t, Z, MLaplace, T2, S, Out):

    with open(f'{Out}CPMG_td.csv', 'w') as f:
        f.write("t [ms]\t\tDecay\t\tFit (NLI)\n")
        for i in range(len(t)):
            f.write(f'{t[i]:.6f}\t{Z[i]:.6f}\t{MLaplace[i]:.6f}\n')

    cumT2 = np.cumsum(S)
    with open(f'{Out}CPMG_rd.csv', 'w') as f:
        f.write("T2 [ms]\t\tDistribution\tCumulative\n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}\t{S[i]:.6f}\t{cumT2[i]:.6f}\n')


# DQ related functions


def readDQ(fileDir):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    # read in the bruker formatted data
    dic, rawdata = rb.read(fileDir)

    vd = []
    vdlist = pd.read_csv(f'{fileDir}/vdlist', header=None).to_numpy().T[0]
    for k in range(len(vdlist)):
        vd.append(vdlist[k].split('u')[0])
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
    DQfilter = dic["acqus"]["D"][5] # s
    DQfilterzFil = dic["acqus"]["D"][8] # s

    filter = 69 # Puntos que no sirven tema de filtro digital
    SGL = rawdata[:, filter:]
    nP = len(SGL[0, :]) # Cantidad total de puntos que me quedan en la FID
    t = np.array([x * 1E6 / SW for x in range(nP)]) # eje temporal en microsegundos

    return t, SGL, nP, SW, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, DQfilter, DQfilterzFil


def PhCorrDQ(SGL, lenvd, phasecorr):
    '''
    Corrección de fase.
    '''
    
    # # >>>> Corrige cada FID individualmente.
    # maxVal = {}
    # for k in range(lenvd):
    #     for i in range(360):
    #         tita = np.deg2rad(i)
    #         SGL_ph = SGL[k, :] * np.exp(1j * tita)
    #         maxVal[i] = np.max(SGL_ph[0].real)
    #     SGL[k, :] *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))
    # # <<<< Corrige cada FID individualmente.

    # >>>> Corrige en función de la FID más intensa
    if phasecorr == None:
        fidsint = []
        for k in range(lenvd):
            fidsint.append(SGL[k, 0])

        fid_idx = fidsint.index(np.max(fidsint[1:]))
        maxVal = {}
        for i in range(360):
            tita = np.deg2rad(i)
            SGL_ph = SGL[fid_idx, :] * np.exp(1j * tita)
            maxVal[i] = np.max(SGL_ph[0].real)
        phasecorr = max(maxVal, key=maxVal.get)

        for k in range(lenvd):
            SGL[k, :] *= np.exp(1j * np.deg2rad(phasecorr))
    
    else:
        for k in range(lenvd):
            SGL[k, :] *= np.exp(1j * np.deg2rad(phasecorr))
    # <<<< Corrige en función de la FID más intensa
    return SGL, phasecorr


def specDQ(SGL, nP, SW, lenvd):
    '''
    Creación del espectro.
    '''

    # Preparación del espectro
    zf = FT.next_fast_len(4 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=1/SW)) # Hz scale
    CS = freq / 300 # ppm for Bruker scale
    spec = []

    for k in range(lenvd):
        spec.append(np.flip(FT.fftshift(FT.fft(SGL[k, :], n = zf))))
    spec = np.array(spec)

    return CS, spec


def writeDQ_acq(nS, RDT, RG, att, RD, evol, zFilter, p90, vd, DQfilter, DQfilterzFil, Out, lenvd):

    with open(f'{Out}acq_param.csv', 'w') as f:
        f.write("DQ-filter:\n")
        f.write(f"\tFiltro utilizado\t{DQfilter:.6f}\ts\n")
        f.write(f"\tz-Filter del filtro utilizado\t{DQfilterzFil:.6f}\ts\n\n")
        
        f.write("DQ:\n")
        f.write(f"\tRango de tiempo variable\t{vd[0]:.2f}\tus\t{vd[-1]:.2f}\tus\n")
        f.write(f"\tCantidad total de puntos\t{lenvd}\n")
        f.write(f"\tTiempo de evolución entre bloques\t{evol:.6f}\ts\n")
        f.write(f"\tz-Filter preadquisición\t{zFilter:.6f}\ts\n\n")

        f.write("Otros parámetros de adquisición:\n")
        f.write(f"\tCantidad de scans\t{nS:.0f}\n")
        f.write(f"\tTiempo entre scans\t{RD:.4f}\ts\n")
        f.write(f"\tTiempo muerto\t{RDT}\tus\n")
        f.write(f"\tGanancia\t{RG:.1f}\tdB\n")
        f.write(f"\tAtenuación\t{att:.0f}\tdB\n")
        f.write(f"\tAncho del pulso de 90\t{p90}\tus")

def writeDQ_verbose(t, SGL, nP, CS, spec, Out, lenvd, mlim):

    print('Progress:')

    mask = (CS>-2*mlim)&(CS<2*mlim)
    CS = CS[mask]
    spec = spec[:, mask]

    for k in range(lenvd):
        with open(f'{Out}FID_td_{k}.csv', 'w') as f:
            f.write("t [ms]\tRe[FID]\tIm[FID]\n")
            for i in range(nP):
                f.write(f'{t[i]:.6f}\t{SGL[k, i].real:.6f}\t{SGL[k, i].imag:.6f}\n')

        with open(f'{Out}FID_fd_{k}.csv', 'w') as f:
            f.write("CS [ppm]\tRe[spec]\tIm[spec]\n")
            for i in range(len(CS)):
                f.write(f'{CS[i]:.6f}\t{spec[k, i].real:.6f}\t{spec[k, i].imag:.6f}\n')
        
        if k % 5 == 0:
            print(f'\t\t{(k+1)*100/lenvd:.0f} %')
        
        elif k == (lenvd-1):
            print(f'\t\t100 %')


def writeDQ(vd, fid00, fidPts, fidPtsSD, pArea, Out):

    with open(f'{Out}DQ_bu.csv', 'w') as f:
            f.write("vd [us]\tFID00\tFID Pts\tFID Pts (SD)\tPeak Area\n")
            for i in range(len(vd)):
                f.write(f'{vd[i]:.1f}\t{fid00[i]:.6f}\t{fidPts[i]:.6f}\t{fidPtsSD[i]:.6f}\t{pArea[i]:.6f}\n')


def readDQLaplace(path):
    '''
    Reads signal file.
    '''

    data = pd.read_csv(path, sep='\t').to_numpy()

    vd_us = data[:, 0] # In us
    bu = data[:, 2] # Tomo el promedio de primeros puntos

    return vd_us, bu

def initKernelDQ(nP, vdFit, DipMin, DipMax):
    '''
    Initialize variables for Laplace transform.
    '''

    nBin = 150
    S0 = np.ones(nBin)
    # Dip = np.logspace(DipMin, DipMax, nBin) # separación logarítmica
    Dip = np.linspace(DipMin, DipMax, nBin) # separación lineal

    K = np.zeros((nP, nBin))

    for i in range(nP):
        K[i, :] = 1 - np.exp(-(vdFit[i] * Dip)**2)

    return S0, Dip, K

def fitLapMag_Dip(vd, Dip, S, nP):
    '''
    Fits decay from Dip distribution.
    '''

    d = range(len(Dip))
    M = []
    for i in range(nP):
        m = 0
        for j in d:
            m += S[j] * (1 - np.exp(- vd[i]**2 * Dip[j]**2))
        M.append(m)

    return M


def writeDQLap(vd, Z, MLaplace, Dip, S, root):
    '''
    Saves processed data.
    '''

    with open(f'{root}_TimeDom.csv', 'w') as f:
        f.write("t [ms]\t\tDecay\t\tFit (NLI)\n")
        for i in range(len(MLaplace)):
            f.write(f'{vd[i]:.6f}\t{Z[i]:.6f}\t{MLaplace[i]:.6f}\n')

    cumDip = np.cumsum(S)
    with open(f'{root}_RatesDom.csv', 'w') as f:
        f.write("Dip [ms]\t\tDistribution\tCumulative\n")
        for i in range(len(Dip)):
            f.write(f'{Dip[i]:.6f}\t{S[i]:.6f}\t{cumDip[i]:.6f}\n')


# SR-CPMG related functions
            
def readSRCPMG(fileDir):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    # read in the bruker formatted data
    dic, rawdata = rb.read(fileDir)

    tau1 = []
    vdlist = pd.read_csv(f'{fileDir}/vdlist', header=None).to_numpy().T[0]
    for k in range(len(vdlist)):
        if len(vdlist[k].split('u')) == 2:
            tau1.append(float(vdlist[k].split('u')[0])*0.001)
        elif len(vdlist[k].split('m')) == 2:
            tau1.append(float(vdlist[k].split('m')[0]))
        elif len(vdlist[k].split('s')) == 2:
            tau1.append(float(vdlist[k].split('s')[0])*1000)
    tau1 = np.array(tau1).astype(float) # ms

    nS = dic["acqus"]["NS"]
    RDT = dic["acqus"]["DE"] # us
    RG = dic["acqus"]["RG"] # dB

    RD = dic["acqus"]["D"][1] # s
    att = dic["acqus"]["PL"][1] # dB

    p90 = dic["acqus"]["P"][1] # us
    p180 = dic["acqus"]["P"][2] # us

    nEcho = dic["acqus"]["TD"]
    nEcho = int(nEcho/2)
    tEcho = dic["acqus"]["D"][6] # s
    tEcho *= 2 * 1000 # ms

    tau2 = np.linspace(tEcho, tEcho*nEcho, nEcho) # eje temporal en ms
    SGL = rawdata[:, :nEcho]

    return tau1, tau2, SGL, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho


def initKernel2D(lenIndir, nEcho, tau1, tau2, T1min, T1max, T2min, T2max):
    '''
    Initialize variables for Laplace transform.
    '''

    nBinx = nBiny = 150
    S0 = np.ones((nBinx, nBiny))
    T1 = np.logspace(T1min, T1max, nBinx)
    T2 = np.logspace(T2min, T2max, nBiny)

    K1 = np.zeros((lenIndir, nBinx))
    for i in range(lenIndir):
        K1[i, :] = 1 - np.exp(-tau1[i] / T1)

    K2 = np.zeros((nEcho, nBiny))
    for i in range(nEcho):
        K2[i, :] = np.exp(-tau2[i] / T2)

    return S0, T1, T2, K1, K2


def PhCorr2D(SGL, lenIndir, nEcho):
    '''
    Phase correction, based on the last measurement.
    '''

    Z = []

    signal_Last = SGL[-1, :]
    maxVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal_Last * np.exp(1j * tita)
        maxVal[i] = np.max(signal_ph[0:30].real)

    tita = np.deg2rad(max(maxVal, key=maxVal.get))

    for k in range(lenIndir):
        signal_k = SGL[k, :] * np.exp(1j * tita)
        signal_k = signal_k.real
        Z.append(signal_k)

    return np.array(Z)

def NLI_FISTA_2D(K1, K2, Z, alpha, S):
    '''
    Numeric Laplace inversion, based on FISTA.
    '''

    K1TK1 = K1.T @ K1
    K2TK2 = K2.T @ K2
    K1TZK2 = K1.T @ Z @ K2
    ZZT = np.trace(Z @ Z.T)

    invL = 1 / (np.trace(K1TK1) * np.trace(K2TK2) + alpha)
    factor = 1 - alpha * invL

    Y = S
    tstep = 1
    lastRes = np.inf

    for iter in range(100000):
        term2 = K1TZK2 - K1TK1 @ Y @ K2TK2
        Snew = factor * Y + invL * term2
        Snew[Snew<0] = 0

        tnew = 0.5 * (1 + np.sqrt(1 + 4 * tstep**2))
        tRatio = (tstep - 1) / tnew
        Y = Snew + tRatio * (Snew - S)
        tstep = tnew
        S = Snew

        if iter % 1000 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            ObjFunc = ZZT - 2 * np.trace(S.T @ K1TZK2) + np.trace(S.T @ K1TK1 @ S @ K2TK2) + TikhTerm

            Res = np.abs(ObjFunc - lastRes) / ObjFunc
            lastRes = ObjFunc
            print(f'\t# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S, iter

def fitLapMag_2D(tau1, tau2, T1, T2, S):
    '''
    Fits decay from T1 and T2 distributions.
    '''

    print(f'\tFitting T1 projection from 2D-Laplace in time domain...')

    t1 = range(len(tau1))
    d1 = range(len(T1))
    S1 = np.sum(S, axis=1)
    M1 = []

    for i in t1:
        m1 = 0
        for j in d1:
            m1 += S1[j] * (1 - np.exp(-tau1[i] / T1[j]))
        M1.append(m1)

    print(f'\tFitting T2 projection from 2D-Laplace in time domain...')

    t2 = range(len(tau2))
    d2 = range(len(T2))
    S2 = np.sum(S, axis=0)
    M2 = []

    for i in t2:
        m2 = 0
        for j in d2:
            m2 += S2[j] * np.exp(-tau2[i] / T2[j])
        M2.append(m2)

    return np.array(M1), np.array(M2)

def writeSRCPMG_acq(nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, Out):

    with open(f'{Out}acq_param.csv', 'w') as f:
        f.write("Parámetros de adquisición:\n")
        f.write(f"\tCantidad de scans\t{nS:.0f}\n")
        f.write(f"\tTiempo entre scans\t{RD:.4f}\ts\n")
        f.write(f"\tTiempo muerto\t{RDT}\tus\n")
        f.write(f"\tGanancia\t{RG:.1f}\tdB\n")
        f.write(f"\tAtenuación\t{att:.0f}\tdB\n")
        f.write(f"\tAncho del pulso de 90\t{p90}\tus")
        f.write(f"\tAncho del pulso de 180\t{p180}\tus")
        f.write(f"\tTiempo de eco\t{tEcho}\tms")
        f.write(f"\tNúmero de ecos\t{nEcho}")


def writeSRCPMG(t, Z, MLaplace, T2, S, Out):

    with open(f'{Out}CPMG_td.csv', 'w') as f:
        f.write("t [ms]\t\tDecay\t\tFit (NLI)\n")
        for i in range(len(t)):
            f.write(f'{t[i]:.6f}\t{Z[i]:.6f}\t{MLaplace[i]:.6f}\n')

    cumT2 = np.cumsum(S)
    with open(f'{Out}CPMG_rd.csv', 'w') as f:
        f.write("T2 [ms]\t\tDistribution\tCumulative\n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}\t{S[i]:.6f}\t{cumT2[i]:.6f}\n')


def writeSRCPMG(T1, T2, S, Out):
    '''
    Saves processed data.
    '''
    
    np.savetxt(f"{Out}SR-CPMG_RatesDom_Full2D.csv", S, delimiter='\t')

    projT1 = np.sum(S, axis=1)
    cumT1 = np.cumsum(projT1)
    with open(f'{Out}SR-CPMG_RatesDom_ProjectionT1.csv', 'w') as f:
        f.write("T1 [ms]\t\tDistribution\tCumulative\n")
        for i in range(len(T1)):
            f.write(f'{T1[i]:.6f}\t{projT1[i]:.6f}\t{cumT1[i]:.6f}\n')

    projT2 = np.sum(S, axis=0)
    cumT2 = np.cumsum(projT2)
    with open(f'{Out}SR-CPMG_RatesDom_ProjectionT2.csv', 'w') as f:
        f.write("T2 [ms]\t\tDistribution\tCumulative\n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}\t{projT2[i]:.6f}\t{cumT2[i]:.6f}\n')