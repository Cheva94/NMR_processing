import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")


################################################################################
###################### Plotting parameters #####################################
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


################################################################################
################################################################################
################################################################################


################################################################################
###################### FID related functions ###################################
################################################################################


def freq2CS(freq):
    return freq / 20


def CS2freq(CS):
    return CS * 20


def FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out, ppm, pDrop):
    '''
    Plots FID results.
    '''

    fig, axs = plt.subplots(2, 2, figsize=(50, 20), gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(rf'RDT = {RDT} $\mu$s | Atten = {att} dB | RG = {RG} dB | nS = {nS} | RD = {RD:.2f} s | p90 = {p90} $\mu$s', fontsize='medium')

    # Mean of the first `pts` FID points
    pts = 20
    fid0Arr = SGL[:pts].real
    fid0 = sum(fid0Arr) / pts
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / pts) ** 0.5

    # Plot de la parte real de la FID
    m = int(np.floor(np.log10(np.max(SGL.real))))
    axs[0,0].set_title(f'Discarded points: {pDrop}.', fontsize='large')
    axs[0,0].scatter(t, SGL.real, label = fr'$M_R (0)$ = {SGL[0].real * 10**-m:.4f} E{m}', color='coral')
    axs[0,0].plot(t[:pts], SGL[:pts].real, lw = 10, label = fr'$M_R ({pts})$ = ({fid0 * 10**-m:.4f} $\pm$ {fid0_SD * 10**-m:.4f}) E{m}', color='teal')
    axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,0].set_xlabel('t [ms]')
    axs[0,0].set_ylabel('FID (Real)')
    axs[0,0].legend()
    axs[0,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Inset del comienzo de la parte real de la FID
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:40], SGL[0:40].real, color='coral')
    axins1.plot(t[:pts], SGL[:pts].real, color='teal')

    # Plot de la parte imaginaria de la FID
    axs[1,0].scatter(t, SGL.imag)
    axs[1,0].plot(t[:pts], SGL[:pts].imag, lw = 10, color='red')
    axs[1,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,0].set_xlabel('t [ms]')
    axs[1,0].set_ylabel('FID (Imag)')
    m = int(np.floor(np.log10(np.max(SGL.imag))))
    axs[1,0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Preparación del espectro
    mask = (CS>-ppm)&(CS<ppm)
    max_peak = np.max(spec[mask].real)
    specNorm = spec / max_peak
    area_peak = np.sum(spec[mask].real)
    peaks, _ = find_peaks(specNorm[mask].real, height=0.9)
    peaksx, peaksy = CS[mask][peaks], specNorm[mask][peaks].real
    
    # Plot de la parte real del espectro, zoom en el pico
    m = int(np.floor(np.log10(np.max(area_peak))))
    axs[0,1].plot(CS, specNorm.real, color='coral')
    axs[0,1].fill_between(CS[mask], 0, specNorm[mask].real, label = fr'Peak area = {area_peak * 10**-m:.4f} E{m}', alpha = 0.25, color="teal")
    axs[0,1].plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
    axs[0,1].annotate(f'{peaksx[0]:.4f} ppm', xy = (peaksx[0], peaksy[0] + 0.07), fontsize=30, ha='center') 
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

    # Inset del espectro completo
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
    axins2.tick_params(labelleft=False)
    axins2.plot(CS, specNorm.real, color='coral')

    # Plot de la parte imaginaria del espectro
    axs[1,1].scatter(CS, specNorm.imag)
    axs[1,1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,1].axvline(x=0, color='k', ls=':', lw=4)
    axs[1,1].set_xlim(-2*ppm, 2*ppm)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].set_ylabel('Norm. Spec. (Imag)')
    secax = axs[1,1].secondary_xaxis('top', functions=(CS2freq, freq2CS))
    secax.set_xlabel(r'$\omega$ [Hz]')

    plt.savefig(f'{Out}')


################################################################################
################################################################################
################################################################################