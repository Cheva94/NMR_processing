import numpy as np
import pandas as pd
import scipy.fft as FT
from scipy.optimize import curve_fit

################################################################################
###################### Common functions ########################################
################################################################################


def r_square(x, y, f, popt):
    '''
    Pearson's coefficient.
    '''

    residuals = y - f(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return 1 - ss_res / ss_tot


def read1Dsgl(path):
    '''
    Reads signal file.
    '''

    data = pd.read_csv(path, header = None, delim_whitespace=True).to_numpy()

    t = data[:, 0] # In ms
    DW = t[1] - t[0] # Dwell time
    nP = len(t) # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    SGL = Re + Im * 1j # Complex signal

    return t, SGL, nP, DW


def read1Dparams(root):
    '''
    Reads acquisition parameters.
    '''

    pAcq = pd.read_csv(root+'_acqs.txt', header = None,  sep='\t')
    
    RDT, att, RG = pAcq.iloc[1, 1], pAcq.iloc[3, 1], pAcq.iloc[2, 1]
    RDT = int(RDT*1E3)
    att = int(att)
    RG = int(RG)

    nS, RD, p90 = pAcq.iloc[0, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1]
    nS = int(nS)
    RD = float(RD)
    p90 = float(p90)

    with open(root+'_acqs.txt', "rbU") as f:
        lines = sum(1 for _ in f)

    if lines > 7:
        p180, tEcho, nEcho = pAcq.iloc[6, 1], pAcq.iloc[7, 1], pAcq.iloc[8, 1]
        p180 = float(p180)
        tEcho = int(tEcho)
        nEcho = int(nEcho)
    else:
        p180, tEcho, nEcho = None, None, None

    return RDT, att, RG, nS, RD, p90, p180, tEcho, nEcho


def PhCorr1D(SGL):
    '''
    Phase correction.
    '''

    maxVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        SGL_ph = SGL * np.exp(1j * tita)
        maxVal[i] = np.max(SGL_ph[0:30].real)
    SGL *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))

    return SGL


def NormRG(SGL, RG):
    '''
    Normalization by receiver gain.
    '''

    norm = 1 / (6.32589E-4 * np.exp(RG/9) - 0.0854)

    return SGL * norm


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


################################################################################
###################### FID related functions ###################################
################################################################################


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

    mask = (CS>-2*ppm)&(CS<2*ppm)
    CS = CS[mask]
    spec = spec[mask]

    with open(f'{Out}_FreqDom.csv', 'w') as f:
        f.write("CS [ppm]\tRe[spec]\tIm[spec]\n")
        for i in range(len(CS)):
            f.write(f'{CS[i]:.6f}\t{spec[i].real:.6f}\t{spec[i].imag:.6f} \n')


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


################################################################################
###################### CPMG related functions ##################################
################################################################################


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

    popt, pcov = curve_fit(exp_1, t, Z, bounds=(0, np.inf), p0=[70, 2000])
    perr = np.sqrt(np.diag(pcov))

    Mag_1 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', '']
    T2_1 = [fr'T2 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', '']
    
    r2 = r_square(t, Z, exp_1, popt)

    return Mag_1, T2_1, r2


def exp_2(t, M0_1, T2_1, M0_2, T2_2):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2)


def expFit_2(t, Z):
    '''
    Biexponential fitting of CPMG decay.
    '''

    popt, pcov = curve_fit(exp_2, t, Z, bounds=(0, np.inf), 
                           p0=[70, 2000, 30, 1000])
    perr = np.sqrt(np.diag(pcov))

    Mag_2 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', 
             fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})']
    T2_2 = [fr'T2 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', 
            fr'T2 = ({popt[3]:.2f}$\pm${perr[3]:.2f}) ms']

    r2 = r_square(t, Z, exp_2, popt)

    return Mag_2, T2_2, r2


def NLI_FISTA(K, Z, alpha, S):
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


def fitLapMag(t, T2, S, nP):
    '''
    Fits decay from T2 distribution.
    '''

    t = range(nP)
    d = range(len(T2))
    M = []
    for i in t:
        m = 0
        for j in d:
            m += S[j] * np.exp(- t[i] / T2[j])
        M.append(m)

    return M


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------