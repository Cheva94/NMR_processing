import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks

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


def FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out):
    '''
    Grafica resultados de la FID.
    '''
    
    fig, axs = plt.subplots(2, 2, figsize=(50, 20), gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(rf'nS = {nS:.0f}    |    RDT = {RDT} $\mu$s    |    RG = {RG:.1f} dB    |    Atten = {att:.0f} dB    |    RD = {RD:.4f} s    |    p90 = {p90} $\mu$s', fontsize='large')

    # Promedio de los primeros 10 puntos de la FID
    points = 20 #10
    fid0Arr = SGL.real[:points]
    fid0 = sum(fid0Arr) / points
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

    # Plot de la parte real de la FID
    axs[0,0].scatter(t, SGL.real, label = fr'$M_R (0)$ = {SGL[0].real:.0f}', color='coral')
    axs[0,0].plot(t[:points], SGL.real[:points], lw = 10, label = fr'$M_R ({points})$ = ({fid0:.0f} $\pm$ {fid0_SD:.0f})', color='teal')
    axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,0].set_xlabel('t [ms]')
    axs[0,0].set_ylabel('FID (real part)')
    axs[0,0].legend()

    # Inset del comienzo de la parte real de la FID
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:40], SGL[0:40].real, color='coral')
    axins1.plot(t[:points], SGL.real[:points], color='teal')

    # Plot de la parte imaginaria de la FID
    axs[1,0].scatter(t, SGL.imag)
    axs[1,0].plot(t[:points], SGL.imag[:points], lw = 10, color='red')
    axs[1,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,0].set_xlabel('t [ms]')
    axs[1,0].set_ylabel('FID (imag. part)')

    # Preparación del espectro
    mask = (CS>-5)&(CS<5)
    max_peak = np.max(spec.real[mask])
    spec /= max_peak
    area_peak = np.sum(spec.real[mask])
    peaks, _ = find_peaks(spec.real[mask], height=0.9)
    peaksx, peaksy = CS[mask][peaks], spec.real[mask][peaks]
    
    # Plot de la parte real del espectro, zoom en el pico
    axs[0,1].plot(CS, spec.real, color='coral')
    axs[0,1].fill_between(CS[mask], 0, spec.real[mask], label = fr'Peak area = {area_peak:.0f}', alpha = 0.25, color="teal")
    axs[0,1].plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
    axs[0,1].annotate(f'{peaksx[0]:.4f} ppm', xy = (peaksx[0], peaksy[0] + 0.07), fontsize=30, ha='center') 
    axs[0,1].set_xlim(-5, 5)
    axs[0,1].set_ylim(-0.05, 1.2)
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,1].set_xlabel(r'$\delta$ [ppm]')
    axs[0,1].axvline(x=0, color='k', ls=':', lw=2)
    axs[0,1].axhline(y=0, color='k', ls=':', lw=2)
    axs[0,1].legend(loc='upper right')
    axs[0,1].set_ylabel('Norm. Spec. (real part)')

    # Inset del espectro completo
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
    axins2.tick_params(labelleft=False)
    axins2.plot(CS, spec.real, color='coral')

    # Plot de la parte imaginaria del espectro
    axs[1,1].scatter(CS, spec.imag)
    axs[1,1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,1].axvline(x=0, color='k', ls=':', lw=4)
    axs[1,1].set_xlim(-5, 5)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].set_ylabel('Norm. Spec. (imag. part)')

    plt.savefig(f'{Out}')
