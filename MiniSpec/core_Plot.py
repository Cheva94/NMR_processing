import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

################################################################################
###################### Common plotting parameters ##############################
################################################################################

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

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


################################################################################
###################### FID related functions ###################################
################################################################################


def freq2CS(freq):
    return freq / 20


def CS2freq(CS):
    return CS * 20


def FID(t, SGL, CS, spec, root, ppm, params, pDrop):
    '''
    Plots FID results.
    '''

    fig, axs = plt.subplots(2, 2, figsize=(50, 20),
                            gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(params, fontsize='large')

    # Mean of the first `pts` FID points
    pts = 20
    fid0Arr = SGL[:pts].real
    fid0 = sum(fid0Arr) / pts
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / pts) ** 0.5

    # FID: real part (full)
    m = int(np.floor(np.log10(np.max(SGL.real))))
    axs[0,0].set_title(f'Discarded points: {pDrop}.', fontsize='large')
    axs[0,0].scatter(t, SGL.real, 
                     label = fr'$M_R (0)$ = {SGL[0].real * 10**-m:.4f} E{m}', 
                     color=Naranja)
    lbl = (fr'$M_R ({pts})$ = ({fid0 * 10**-m:.4f} $\pm$'
           fr'{fid0_SD * 10**-m:.4f}) E{m}')
    axs[0,0].plot(t[:pts], SGL[:pts].real, lw = 10, 
                  label = lbl, color=Verde)
    axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,0].set_xlabel('t [ms]')
    axs[0,0].set_ylabel('FID (Real)')
    axs[0,0].legend()
    axs[0,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # FID: real part (inset: zoom at the beginning)
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:40], SGL[0:40].real, color=Naranja)
    axins1.plot(t[:pts], SGL[:pts].real, color=Verde)

    # FID: imaginary part (full)
    axs[1,0].scatter(t, SGL.imag, color=Morado)
    axs[1,0].plot(t[:pts], SGL[:pts].imag, lw = 10, color=Amarillo, zorder=50)
    axs[1,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,0].set_xlabel('t [ms]')
    axs[1,0].set_ylabel('FID (Imag)')
    m = int(np.floor(np.log10(np.max(SGL.imag))))
    axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Spectrum normalization
    mask = (CS>-ppm)&(CS<ppm)
    max_peak = np.max(spec[mask].real)
    specNorm = spec / max_peak
    area_peak = np.sum(spec[mask].real)
    peaks, _ = find_peaks(specNorm[mask].real, height=0.9)
    peaksx, peaksy = CS[mask][peaks], specNorm[mask][peaks].real
    
    # Spectrum: real part (zoom around the peak)
    m = int(np.floor(np.log10(np.max(area_peak))))
    axs[0,1].plot(CS, specNorm.real, color=Naranja)
    axs[0,1].fill_between(CS[mask], 0, specNorm[mask].real, 
                          label = fr'Area = {area_peak * 10**-m:.4f} E{m}',
                          alpha = 0.25, color="teal")
    axs[0,1].plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
    axs[0,1].annotate(f'{peaksx[0]:.4f} ppm', 
                      xy = (peaksx[0], peaksy[0] + 0.07), 
                      fontsize=30, ha='center') 
    axs[0,1].set_xlim(-2*ppm, 2*ppm)
    axs[0,1].set_ylim(-0.05, 1.2)
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,1].set_xlabel(r'$\delta$ [ppm]')
    axs[0,1].axvline(x=0, color='k', ls=':', lw=2)
    axs[0,1].axhline(y=0, color='k', ls=':', lw=2)
    axs[0,1].legend(loc='upper right')
    axs[0,1].set_ylabel('Norm. Spec. (Real)')
    secax = axs[0,1].secondary_xaxis('top', functions=(CS2freq, freq2CS))
    secax.set_xlabel(r'$\omega$ [Hz]')
    axs[0,1].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[0,1].grid(which='both', color='gray', alpha=1, zorder=-5)

    # Spectrum: real part (inset: full spectrum)
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
    axins2.tick_params(labelleft=False)
    axins2.plot(CS, specNorm.real, color=Naranja)

    # Spectrum: imaginary part (zoom around the peak)
    axs[1,1].scatter(CS, specNorm.imag, color=Morado)
    axs[1,1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,1].axvline(x=0, color='k', ls=':', lw=4)
    axs[1,1].set_xlim(-2*ppm, 2*ppm)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].set_ylabel('Norm. Spec. (Imag)')
    secax = axs[1,1].secondary_xaxis('top', functions=(CS2freq, freq2CS))
    secax.set_xlabel(r'$\omega$ [Hz]')

    plt.savefig(f'{root}')


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


################################################################################
###################### CPMG related functions ##################################
################################################################################


def CPMG(t, Z, T2, S, MLaplace, root, alpha, T2min, T2max, params, dataFit, tEcho):
    '''
    Plots CPMG results.
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

    plt.savefig(f'{root}')


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


################################################################################
###################### SR-CPMG related functions ###############################
################################################################################


def SRCPMG(tau1, tau2, Z, T1, T2, S, MLap_SR, MLap_CPMG, root, alpha, 
           T1min, T1max, T2min, T2max, params, tEcho):
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

    plt.savefig(f'{root}')


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------