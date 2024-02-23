import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['ytick.minor.width'] = 5

plt.rcParams["legend.loc"] = 'best'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'

plt.rcParams["figure.autolayout"] = True

# Colores
Verde = '#08a189' # = 08A189 = 8 161 137    (principal)
Naranja = '#fa6210' # = FA6210 = 250 98 16    (secundario)
Morado = '#7f03fc' # =  = 127 3 252   (terciario)
Amarillo = '#f7f307' # =  = 247 243 7   (cuaternario)
Gris = '#626262' # =  = 98 98 98 (alternativo)
Negro = '#000000' # =  = 0 0 0 (base)

# FID related funcitons


def freq2CS(freq):
    return freq / 300


def CS2freq(CS):
    return CS * 300


def FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out, mlim):
    '''
    Grafica resultados de la FID.
    '''
    
    fig, axs = plt.subplots(2, 2, figsize=(50, 20), gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(rf'nS = {nS:.0f} | RDT = {RDT} $\mu$s | RG = {RG:.1f} dB | Atten = {att:.0f} dB | RD = {RD:.4f} s | p90 = {p90} $\mu$s', fontsize='medium')

    # Promedio de los primeros 10 puntos de la FID
    points = 10
    fid0Arr = SGL[:points].real
    fid0 = sum(fid0Arr) / points
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

    # Plot de la parte real de la FID
    m = np.floor(np.log10(np.max(SGL.real)))
    axs[0,0].scatter(t, SGL.real, label = fr'$M_R (0)$ = {SGL[0].real * 10**-m:.4f} E{m:.0f}', color=Naranja)
    axs[0,0].plot(t[:points], SGL[:points].real, lw = 10, label = fr'$M_R ({points})$ = ({fid0 * 10**-m:.4f} $\pm$ {fid0_SD * 10**-m:.4f}) E{m:.0f}', color=Verde, zorder=-10)
    axs[0,0].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[0,0].set_xlabel(r't [$\mu$s]')
    axs[0,0].set_ylabel('FID (real part)')
    axs[0,0].legend()
    axs[0,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Inset del comienzo de la parte real de la FID
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:40], SGL[0:40].real, color=Naranja)
    axins1.plot(t[:points], SGL[:points].real, color=Verde)
    axins1.ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Plot de la parte imaginaria de la FID
    axs[1,0].scatter(t, SGL.imag, color=Morado)
    axs[1,0].plot(t[:points], SGL[:points].imag, lw = 10, color=Amarillo, zorder=50)
    axs[1,0].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[1,0].set_xlabel(r't [$\mu$s]')
    axs[1,0].set_ylabel('FID (imag. part)')
    m = np.floor(np.log10(np.max(SGL.imag)))
    axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Preparación del espectro
    mask = (CS>-mlim)&(CS<mlim)
    max_peak = np.max(spec[mask].real)
    specNorm = spec / max_peak
    area_peak = np.sum(spec[mask].real)
    peaks, _ = find_peaks(specNorm[mask].real, height=0.9)
    peaksx, peaksy = CS[mask][peaks], specNorm[mask][peaks].real
    
    # Plot de la parte real del espectro, zoom en el pico
    m = np.floor(np.log10(np.max(area_peak)))
    axs[0,1].plot(CS, specNorm.real, color=Naranja)
    axs[0,1].fill_between(CS[mask], 0, specNorm[mask].real, label = fr'Peak area = {area_peak * 10**-m:.4f} E{m:.0f}', alpha = 0.25, color="teal")
    axs[0,1].plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
    axs[0,1].annotate(f'{peaksx[0]:.4f} ppm', xy = (peaksx[0], peaksy[0] + 0.07), fontsize=30, ha='center') 
    axs[0,1].set_xlim(-2*mlim, 2*mlim)
    axs[0,1].set_ylim(-0.05, 1.2)
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,1].set_xlabel(r'$\delta$ [ppm]')
    axs[0,1].axvline(x=0, color=Gris, ls=':', lw=2)
    axs[0,1].axhline(y=0, color=Gris, ls=':', lw=2)
    axs[0,1].legend(loc='upper right')
    axs[0,1].set_ylabel('Norm. Spec. (real part)')
    secax = axs[0,1].secondary_xaxis('top', functions=(CS2freq, freq2CS))
    secax.set_xlabel(r'$\omega$ [Hz]')
    axs[0,1].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[0,1].grid(which='both', color='gray', alpha=1, zorder=-5)

    # Inset del espectro completo
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
    axins2.tick_params(labelleft=False)
    axins2.plot(CS, specNorm.real, color=Naranja)

    # Plot de la parte imaginaria del espectro
    axs[1,1].scatter(CS, specNorm.imag, color=Morado)
    axs[1,1].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[1,1].axvline(x=0, color=Gris, ls=':', lw=4)
    axs[1,1].set_xlim(-2*mlim, 2*mlim)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].set_ylabel('Norm. Spec. (imag. part)')
    secax = axs[1,1].secondary_xaxis('top', functions=(CS2freq, freq2CS))
    secax.set_xlabel(r'$\omega$ [Hz]')

    plt.savefig(f'{Out}FID')


# Nutation related funcitons


def Nutac(SGL, nS, RDT, RG, att, RD, vp, Out, lenvp):
    '''
    Grafica resultados de la nutación
    '''

    points = 10
    fid00, fidPts = [], []
    
    for k in range(lenvp):
        # Sólo el primer punto de la FID
        fid00.append(SGL[k, 0].real)

        # Promedio de los primeros 10 puntos de la FID
        fid0Arr = SGL[k, :points].real
        fid0 = sum(fid0Arr) / points
        fidPts.append(fid0)
    
    fid00 = np.array(fid00)
    fidPts = np.array(fidPts)

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    acqgral = rf'RDT = {RDT} $\mu$s | RG = {RG:.1f} dB | Atten = {att:.0f} dB | nS = {nS:.0f} | RD = {RD:.4f} s'
    fig.suptitle(acqgral, fontsize='small')
        
    # Plot del primer punto de la FID
    axs[0].set_title('Primer punto de cada FID')
    axs[0].scatter(vp, fid00, label=rf'Max = {vp[fid00 == np.max(fid00)][0]} $\mu$s', color=Verde)
    axs[0].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[0].set_xlabel(r't [$\mu$s]')
    axs[0].legend()
    m = np.floor(np.log10(np.max(fid00)))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Plot del promedio de los primeros puntos de la FID
    axs[1].set_title('Primeros 10 puntos de cada FID')
    axs[1].scatter(vp, fidPts, label=rf'Max = {vp[fidPts == np.max(fidPts)][0]} $\mu$s', color=Verde)
    axs[1].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[1].set_xlabel(r't [$\mu$s]')
    axs[1].legend()
    m = np.floor(np.log10(np.max(fidPts)))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    plt.savefig(f'{Out}Nutac')

    return fid00, fidPts


# CPMG related functions

def CPMG(t, Z, T2, S, MLaplace, Out, alpha, T2min, T2max, params, dataFit, tEcho):
    '''
    Grafica resultados de la CPMG.
    '''

    # Remove edge effects from NLI on the plots
    S = S[2:-2]
    T2 = T2[2:-2]

    fig, axs = plt.subplots(2, 3, figsize=(50, 20), 
                            gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(params, fontsize='large')

    # CPMG: experimental and fit
    axs[0,0].scatter(t, Z, label='Exp', color=Naranja)
    axs[0,0].plot(t, MLaplace, label='NLI Fit', color=Verde)
    axs[0,0].set_xlabel(r'$\tau$ [ms]')
    axs[0,0].set_ylabel('CPMG')
    axs[0,0].legend()
    axs[0,0].axhline(0, c = 'k', lw = 4, ls = ':', zorder=-2)

    # CPMG: experimental and fit (inset: zoom at the beginning)
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:30], Z[0:30], color=Naranja)
    axins1.plot(t[0:30], MLaplace[0:30], color=Verde)

    # CPMG: experimental and fit (semilog)
    axs[0,1].scatter(t, Z, label='Exp', color=Naranja)
    axs[0,1].plot(t, MLaplace, label='NLI Fit', color=Verde)
    axs[0,1].set_yscale('log')
    axs[0,1].set_xlabel(r'$\tau$ [ms]')
    axs[0,1].set_ylabel('log(CPMG)')
    axs[0,1].legend()

    # CPMG: NLI residuals
    residuals = MLaplace-Z
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)
    R2Laplace = 1 - ss_res / ss_tot
    axs[1,0].set_title(fr'NLI: R$^2$ = {R2Laplace:.6f}')
    axs[1,0].scatter(t, residuals, color = Morado)
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,0].set_xlabel(r'$\tau$ [ms]')
    axs[1,0].axhline(0.1*np.max(Z), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(-0.1*np.max(Z), c = 'red', lw = 6, ls = '-')

    # T2 distribution
    Snorm = S / np.max(S)
    peaks, _ = find_peaks(Snorm,height=0.025, distance = 5)
    peaksx, peaksy = T2[peaks], Snorm[peaks]

    axs[0,2].fill_between([tEcho, 5 * tEcho], -0.02, 1.2, color=Gris, 
                          alpha=0.3, zorder=-2)
    axs[0,2].set_title(rf'$\alpha$ = {alpha}')
    axs[0,2].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,2].plot(T2, Snorm, label = 'Distrib.', color = Verde)
    for i in range(len(peaksx)):
            axs[0,2].plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, 
                          color='black')
            axs[0,2].annotate(f'{peaksx[i]:.2f}', 
                              xy = (peaksx[i], peaksy[i] + 0.07), 
                              fontsize=30, ha='center')
    axs[0,2].set_xlabel(r'$T_2$ [ms]')
    axs[0,2].set_xscale('log')
    axs[0,2].set_ylim(-0.02, 1.2)
    axs[0,2].set_xlim(10.0**T2min, 10.0**T2max)

    cumT2 = np.cumsum(S)
    cumT2norm = cumT2 / cumT2[-1]
    ax = axs[0,2].twinx()
    ax.plot(T2, cumT2norm, label = 'Cumul.', color = Naranja)
    ax.set_ylim(-0.02, 1.2)

    axs[1,1].axis('off')
    axs[1,2].axis('off')

    # Exponential fit resutls
    axs[1,1].annotate('>>> Monoexponential fit <<<', xy = (0.5, 1.00), 
                      fontsize=30, ha='center')
    axs[1,1].annotate(f'{dataFit[0,0]} --> {dataFit[1,0]}', xy = (0.5, 0.85), 
                      fontsize=30, ha='center')
    axs[1,1].annotate(f'{dataFit[0,2]}', xy = (0.5, 0.70), 
                      fontsize=30, ha='center')

    axs[1,1].annotate('>>> Biexponential fit <<<', xy = (0.5, 0.45), 
                      fontsize=30, ha='center')
    axs[1,1].annotate(f'{dataFit[2,0]} --> {dataFit[3,0]}', xy = (0.5, 0.30), 
                      fontsize=30, ha='center')
    axs[1,1].annotate(f'{dataFit[2,1]} --> {dataFit[3,1]}', xy = (0.5, 0.15), 
                      fontsize=30, ha='center')
    axs[1,1].annotate(f'{dataFit[2,2]}', xy = (0.5, 0.00), 
                      fontsize=30, ha='center')
    
    plt.savefig(f'{Out}CPMG')

def SRCPMG(tau1, tau2, Z, T1, T2, S, MLap_SR, MLap_CPMG, Out, 
           alpha, T1min, T1max, T2min, T2max, params, tEcho):
    '''
    Plots SR-CPMG results.
    '''
    
    # Counts negative signal points
    NegPts = 0
    for k in range(len(tau1)):
        if Z[k, 0] < 0.0000:
            NegPts += 1

    # Remove edge effects from NLI on the plots
    S = S[4:-9, 2:-2]
    T1 = T1[4:-9]
    T2 = T2[2:-2]

    fig, axs = plt.subplots(2, 4, figsize=(50, 20))
    fig.suptitle(params, fontsize='large')

    # SR: experimental and fit
    projOffset = Z[-1, 0] / MLap_SR[-1]
    MLap_SR *= projOffset

    axs[0,0].set_title(f'Neg. pts.: {NegPts}', fontsize='large')
    axs[0,0].scatter(tau1, Z[:, 0], label='Exp', color=Naranja)
    axs[0,0].plot(tau1, MLap_SR, label='NLI Fit', color=Verde)
    axs[0,0].set_xlabel(r'$\tau_1$ [ms]')
    axs[0,0].set_ylabel('SR')
    axs[0,0].legend()
    axs[0,0].axhline(0, c = 'k', lw = 4, ls = ':', zorder=-2)

    # SR: NLI residuals
    residuals = MLap_SR-Z[:, 0]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z[:, 0] - np.mean(Z[:, 0])) ** 2)
    R2_indir = 1 - ss_res / ss_tot

    axs[1,0].set_title(fr'NLI: R$^2$ = {R2_indir:.6f}')
    axs[1,0].scatter(tau1, residuals, color = Morado)
    axs[1,0].axhline(0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(-0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,0].set_xlabel(r'$\tau_1$ [ms]')
    
    # CPMG: experimental and fit
    projOffset = Z[-1, 0] / MLap_SR[-1]
    MLap_SR *= projOffset

    axs[0,1].set_title(f'Disc. pts.: 1', fontsize='large')
    axs[0,1].scatter(tau2, Z[-1, :], label='Exp', color=Naranja)
    axs[0,1].plot(tau2, MLap_CPMG, label='NLI Fit', color=Verde)
    axs[0,1].set_xlabel(r'$\tau_2$ [ms]')
    axs[0,1].set_ylabel('CPMG')
    axs[0,1].legend()
    axs[0,1].axhline(0, c = 'k', lw = 4, ls = ':', zorder=-2)

    # CPMG: experimental and fit (inset: zoom at the beginning)
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=5)
    axins2.scatter(tau2[0:30], Z[-1, :][0:30], color=Naranja)
    axins2.plot(tau2[0:30], MLap_CPMG[0:30], color=Verde)

    # CPMG: NLI residuals
    residuals = MLap_CPMG-Z[-1, :]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z[-1, :] - np.mean(Z[-1, :])) ** 2)
    R2_dir = 1 - ss_res / ss_tot

    axs[1,1].set_title(fr'NLI: R$^2$ = {R2_dir:.6f}')
    axs[1,1].scatter(tau2, residuals, color = Morado)
    axs[1,1].axhline(0.1*np.max(Z[-1, :]), c = 'red', lw = 6, ls = '-')
    axs[1,1].axhline(-0.1*np.max(Z[-1, :]), c = 'red', lw = 6, ls = '-')
    axs[1,1].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,1].set_xlabel(r'$\tau_2$ [ms]')

    # Projected T1 distribution
    projT1 = np.sum(S, axis=1)
    projT1 = projT1 / np.max(projT1)
    peaks1, _ = find_peaks(projT1, height=0.025, distance = 5)
    peaks1x, peaks1y = T1[peaks1], projT1[peaks1]
    
    axs[0,2].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,2].plot(T1, projT1, label = 'Distrib.', color = Verde)
    for i in range(len(peaks1x)):
        axs[0,2].plot(peaks1x[i], peaks1y[i] + 0.05, lw = 0, marker=11, 
                      color='black')
        axs[0,2].annotate(f'{peaks1x[i]:.2f}', 
                          xy = (peaks1x[i], peaks1y[i] + 0.07), 
                          fontsize=30, ha = 'center')
    axs[0,2].set_xlabel(r'$T_1$ [ms]')
    axs[0,2].set_xscale('log')
    axs[0,2].set_ylim(-0.02, 1.2)
    axs[0,2].set_xlim(10.0**T1min, 10.0**T1max)

    cumT1 = np.cumsum(projT1)
    cumT1 /= cumT1[-1]
    ax = axs[0,2].twinx()
    ax.plot(T1, cumT1, label = 'Cumul.', color = Naranja)
    ax.set_ylim(-0.02, 1.2)

    # Projected T2 distribution
    projT2 = np.sum(S, axis=0)
    projT2 = projT2 / np.max(projT2)
    peaks2, _ = find_peaks(projT2, height=0.025, distance = 5)
    peaks2x, peaks2y = T2[peaks2], projT2[peaks2]
    
    axs[0,3].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,3].plot(T2, projT2, label = 'Distrib.', color = Verde)
    for i in range(len(peaks2x)):
        axs[0,3].plot(peaks2x[i], peaks2y[i] + 0.05, lw = 0, marker=11, 
                      color='black')
        axs[0,3].annotate(f'{peaks2x[i]:.2f}', 
                          xy = (peaks2x[i], peaks2y[i] + 0.07), 
                          fontsize=30, ha = 'center')
    axs[0,3].set_xlabel(r'$T_2$ [ms]')
    axs[0,3].set_xscale('log')
    axs[0,3].set_ylim(-0.02, 1.2)
    axs[0,3].set_xlim(10.0**T2min, 10.0**T2max)
    axs[0,3].fill_between([tEcho, 5 * tEcho], -0.02, 1.2, color=Gris, 
                          alpha=0.3, zorder=-2)
    
    cumT2 = np.cumsum(projT2)
    cumT2 /= cumT2[-1]
    ax = axs[0,3].twinx()
    ax.plot(T2, cumT2, label = 'Cumul.', color = Naranja)
    ax.set_ylim(-0.02, 1.2)

    # T1-T2 map
    mini = np.max([T1min, T2min])
    maxi = np.min([T1max, T2max])

    axs[1,3].set_title(rf'$\alpha$ = {alpha}')
    axs[1,3].plot([10.0**mini, 10.0**maxi], [10.0**mini, 10.0**maxi], 
                  color='black', ls='-', alpha=0.7, zorder=-2, 
                  label = r'$T_1$ = $T_2$')
    for i in range(len(peaks2x)):
        axs[1,3].axvline(x=peaks2x[i], color='k', ls=':', lw=4)
    for i in range(len(peaks1x)):
        axs[1,3].axhline(y=peaks1x[i], color='k', ls=':', lw=4)
    axs[1,3].contour(T2, T1, S, 100, cmap='rainbow')
    axs[1,3].set_xlabel(r'$T_2$ [ms]')
    axs[1,3].set_ylabel(r'$T_1$ [ms]')
    axs[1,3].set_xlim(10.0**T2min, 10.0**T2max)
    axs[1,3].set_ylim(10.0**T1min, 10.0**T1max)
    axs[1,3].set_xscale('log')
    axs[1,3].set_yscale('log')
    axs[1,3].legend(loc='lower right')

    axs[1,2].axis('off')

    plt.savefig(f'{Out}SR-CPMG')

# DQ related funcitons


def DQ_bu(SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, DQfilterzFil, Out, lenvd, mlim):
    '''
    Grafica resultados de la DQ.
    '''

    points = 10
    fid00, fidPts, fidPtsSD, pArea = [], [], [], []
    mask = (CS>-mlim)&(CS<mlim)
    
    for k in range(lenvd):
        # Sólo el primer punto de la FID
        fid00.append(SGL[k, 0].real)

        # Promedio de los primeros 10 puntos de la FID
        fid0Arr = SGL[k, :points].real
        fid0 = sum(fid0Arr) / points
        fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

        fidPts.append(fid0)
        fidPtsSD.append(fid0_SD)

        # Áreas de los espectros
        pArea.append(np.sum(spec[k, mask].real))
    
    fid00 = np.array(fid00)
    fidPts = np.array(fidPts)
    fidPtsSD = np.array(fidPtsSD)
    pArea = np.array(pArea)

    fig, axs = plt.subplots(1, 3, figsize=(37.5, 10))
    acqgral = rf'RDT = {RDT} $\mu$s | RG = {RG:.1f} dB | Atten = {att:.0f} dB | p90 = {p90} $\mu$s | nS = {nS:.0f} | RD = {RD:.4f} s'
    if DQfilter == 0:
        acqdq = rf'No filter used | Evol = {evol:.6f} s | z-Filter = {zFilter:.6f} s'
    else:
        acqdq = rf'DQ-filter = {DQfilter:.6f} s | DQ-fil (z-fil) = {DQfilterzFil:.6f} s | Evol = {evol:.6f} s | z-Filter = {zFilter:.6f} s'
    fig.suptitle(acqgral+'\n'+acqdq, fontsize='large')
        
    # Plot del primer punto de la FID
    axs[0].set_title('Primer punto de cada FID')
    axs[0].scatter(vd, fid00, label=rf'Max = {vd[fid00 == np.max(fid00[1:])][0]} $\mu$s', color=Verde)
    axs[0].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[0].set_xlabel(r't [$\mu$s]')
    m = np.floor(np.log10(np.max(fid00)))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))
    axs[0].legend()

    # Plot del promedio de los primeros puntos de la FID
    axs[1].set_title('Primeros 10 puntos de cada FID')
    axs[1].scatter(vd, fidPts, label=rf'Max = {vd[fidPts == np.max(fidPts[1:])][0]} $\mu$s', color=Verde)
    axs[1].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[1].set_xlabel(r't [$\mu$s]')
    m = np.floor(np.log10(np.max(fidPts)))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(m,m))
    axs[1].legend()

    # Plot de las áreas de los espectros
    axs[2].set_title('Área de los picos de cada espectro')
    axs[2].scatter(vd, pArea, label=rf'Max = {vd[pArea == np.max(pArea[1:])][0]} $\mu$s', color=Verde)
    axs[2].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[2].set_xlabel(r't [$\mu$s]')
    m = np.floor(np.log10(np.max(pArea)))
    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(m,m))
    axs[2].legend()

    plt.savefig(f'{Out}DQ_bu')

    return fid00, fidPts, fidPtsSD, pArea


def DQ_verbose(t, SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, DQfilterzFil, Out, lenvd, mlim):
    
    print('Progress:')

    for k in range(lenvd):
        fig, axs = plt.subplots(2, 2, figsize=(50, 20), gridspec_kw={'height_ratios': [3,1]})

        acqgral = rf'RDT = {RDT} $\mu$s | RG = {RG:.1f} dB | Atten = {att:.0f} dB | p90 = {p90} $\mu$s | nS = {nS:.0f} | RD = {RD:.4f} s'
        if DQfilter == 0:
            acqdq = rf'vd = {vd[k]:.2f} $\mu$s | No filter used | Evol = {evol:.6f} s | z-Filter = {zFilter:.6f} s'
        else:
            acqdq = rf'vd = {vd[k]:.2f} $\mu$s | DQ-filter = {DQfilter:.6f} s | DQ-fil (z-fil) = {DQfilterzFil:.6f} s | Evol = {evol:.6f} s | z-Filter = {zFilter:.6f} s'

        fig.suptitle(acqgral+'\n'+acqdq, fontsize='large')
    
        # Promedio de los primeros 10 puntos de la FID
        points = 10
        fid0Arr = SGL[k, :points].real
        fid0 = sum(fid0Arr) / points
        fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

        # Plot de la parte real de la FID
        m = np.floor(np.log10(np.max(SGL.real)))
        axs[0,0].scatter(t, SGL[k, :].real, label = fr'$M_R (0)$ = {SGL[k, 0].real * 10**-m:.4f} E{m:.0f}', color=Naranja)
        axs[0,0].plot(t[:points], SGL[k, :points].real, lw = 10, label = fr'$M_R ({points})$ = ({fid0 * 10**-m:.4f} $\pm$ {fid0_SD * 10**-m:.4f}) E{m:.0f}', color=Verde, zorder=-10)
        axs[0,0].axhline(y=0, color=Gris, ls=':', lw=4)
        axs[0,0].set_xlabel(r't [$\mu$s]')
        axs[0,0].set_ylabel('FID (real part)')
        axs[0,0].legend()
        axs[0,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

        # Inset del comienzo de la parte real de la FID
        axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
        axins1.scatter(t[0:40], SGL[k, 0:40].real, color=Naranja)
        axins1.plot(t[:points], SGL[k, :points].real, color=Verde)

        # Plot de la parte imaginaria de la FID
        axs[1,0].scatter(t, SGL[k, :].imag, color=Morado)
        axs[1,0].plot(t[:points], SGL[k, :points].imag, lw = 10, color=Amarillo, zorder=50)
        axs[1,0].axhline(y=0, color=Gris, ls=':', lw=4)
        axs[1,0].set_xlabel(r't [$\mu$s]')
        axs[1,0].set_ylabel('FID (imag. part)')
        m = np.floor(np.log10(np.max(SGL[k, :].imag)))
        axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

        # Preparación del espectro
        mask = (CS>-mlim)&(CS<mlim)
        max_peak = np.max(spec[k, mask].real)
        specNorm = spec[k, :] / max_peak
        area_peak = np.sum(spec[k, mask].real)
        peaks, _ = find_peaks(specNorm[mask].real, height=0.9)
        peaksx, peaksy = CS[mask][peaks], specNorm[mask][peaks].real
        
        # Plot de la parte real del espectro, zoom en el pico
        m = np.floor(np.log10(np.max(area_peak)))
        axs[0,1].plot(CS, specNorm.real, color=Naranja)
        axs[0,1].fill_between(CS[mask], 0, specNorm[mask].real, label = fr'Peak area = {area_peak * 10**-m:.4f} E{m:.0f}', alpha = 0.25, color="teal")
        axs[0,1].plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
        axs[0,1].annotate(f'{peaksx[0]:.4f} ppm', xy = (peaksx[0], peaksy[0] + 0.07), fontsize=30, ha='center') 
        axs[0,1].set_xlim(-2*mlim, 2*mlim)
        axs[0,1].set_ylim(-0.05, 1.2)
        axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[0,1].set_xlabel(r'$\delta$ [ppm]')
        axs[0,1].axvline(x=0, color=Gris, ls=':', lw=2)
        axs[0,1].axhline(y=0, color=Gris, ls=':', lw=2)
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_ylabel('Norm. Spec. (real part)')
        secax = axs[0,1].secondary_xaxis('top', functions=(CS2freq, freq2CS))
        secax.set_xlabel(r'$\omega$ [Hz]')
        axs[0,1].yaxis.set_minor_locator(MultipleLocator(0.05))
        axs[0,1].grid(which='both', color='gray', alpha=1, zorder=-5)

        # Inset del espectro completo
        axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
        axins2.tick_params(labelleft=False)
        axins2.plot(CS, specNorm.real, color=Naranja)

        # Plot de la parte imaginaria del espectro
        axs[1,1].scatter(CS, specNorm.imag, color=Morado)
        axs[1,1].axhline(y=0, color=Gris, ls=':', lw=4)
        axs[1,1].axvline(x=0, color=Gris, ls=':', lw=4)
        axs[1,1].set_xlim(-2*mlim, 2*mlim)
        axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1,1].set_xlabel(r'$\delta$ [ppm]')
        axs[1,1].set_ylabel('Norm. Spec. (imag. part)')
        secax = axs[1,1].secondary_xaxis('top', functions=(CS2freq, freq2CS))
        secax.set_xlabel(r'$\omega$ [Hz]')

        plt.savefig(f'{Out}FID_{k}')
        
        if k % 5 == 0:
            print(f'\t\t{(k+1)*100/lenvd:.0f} %')
        
        elif k == (lenvd-1):
            print(f'\t\t100 %')


def DQLap(vd_us, bu, Dip, S, MLaplace, root, alpha, DipMin, DipMax, limSup):
    '''
    Plots DQ Laplace results.
    '''

    _, axs = plt.subplots(2, 3, figsize=(37.5, 20), 
                            gridspec_kw={'height_ratios': [3,1]})

    # Build-up: experimental and fit
    axs[0,0].scatter(vd_us, bu, label='Exp', color='coral')
    axs[0,0].plot(vd_us[:limSup], MLaplace, label='NLI Fit', color='teal')
    axs[0,0].set_xlabel('vd [us]')
    axs[0,0].legend()
    axs[0,0].axhline(0, c = 'k', lw = 4, ls = ':', zorder=-2)

    # Build-up: NLI residuals
    residuals = MLaplace-bu[:limSup]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((bu - np.mean(bu)) ** 2)
    R2Laplace = 1 - ss_res / ss_tot
    axs[1,0].set_title(fr'NLI: R$^2$ = {R2Laplace:.6f}')
    axs[1,0].scatter(vd_us[:limSup], residuals, color = 'blue')
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,0].set_xlabel('vd [us]')
    axs[1,0].axhline(0.1*np.max(bu), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(-0.1*np.max(bu), c = 'red', lw = 6, ls = '-')

    # Dip distribution
    Snorm = S / np.max(S)
    peaks, _ = find_peaks(Snorm,height=0.025, distance = 5)
    peaksx, peaksy = Dip[peaks], Snorm[peaks]

    axs[0,1].set_title(rf'$\alpha$ = {alpha}')
    axs[0,1].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,1].plot(Dip, Snorm, label = 'Distrib.', color = 'teal')
    for i in range(len(peaksx)):
            axs[0,1].plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, 
                          color='black')
            axs[0,1].annotate(f'{peaksx[i]:.2f}', 
                              xy = (peaksx[i], peaksy[i] + 0.07), 
                              fontsize=30, ha='center')
    axs[0,1].set_xlabel('D [kHz]')
    axs[0,1].set_ylim(-0.02, 1.2)

    # Constantes físicas
    mu0 = 1.256637E-6 # N/A^2
    gammaH = 267.5E6 # Hz/T
    hbar = 6.62607E-34 # Js
    Hzm3TokHzA3 = 1E27
    factConv = gammaH**2 * mu0 * hbar * Hzm3TokHzA3 / (4*np.pi) # kHz A^3
    radDist = np.cbrt(factConv/Dip)
    peaksx = radDist[peaks]

    # Radial distribution
    axs[0,2].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,2].plot(radDist, Snorm, label = 'Distrib.', color = 'teal')
    for i in range(len(peaksx)):
            axs[0,2].plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, 
                          color='black')
            axs[0,2].annotate(f'{peaksx[i]:.2f}', 
                              xy = (peaksx[i], peaksy[i] + 0.07), 
                              fontsize=30, ha='center')
    axs[0,2].set_xlabel('r [A]')
    axs[0,2].set_ylim(-0.02, 1.2)

    axs[1,1].axis('off')
    axs[1,2].axis('off')

    plt.savefig(f'{root}')