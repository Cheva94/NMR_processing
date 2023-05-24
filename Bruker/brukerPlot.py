import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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

def FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out):
    '''
    Grafica resultados de la FID.
    '''
    
    fig, axs = plt.subplots(2, 2, figsize=(50, 20), gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(rf'nS = {nS:.0f}    |    RDT = {RDT} $\mu$s    |    RG = {RG:.1f} dB    |    Atten = {att:.0f} dB    |    RD = {RD:.4f} s    |    p90 = {p90} $\mu$s', fontsize='medium')

    # Promedio de los primeros 10 puntos de la FID
    points = 10
    fid0Arr = SGL[:points].real
    fid0 = sum(fid0Arr) / points
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

    # Plot de la parte real de la FID
    axs[0,0].scatter(t, SGL.real, label = fr'$M_R (0)$ = {SGL[0].real:.0f}', color='coral')
    axs[0,0].plot(t[:points], SGL[:points].real, lw = 10, label = fr'$M_R ({points})$ = ({fid0:.0f} $\pm$ {fid0_SD:.0f})', color='teal')
    axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,0].set_xlabel(r't [$\mu$s]')
    axs[0,0].set_ylabel('FID (real part)')
    axs[0,0].legend()

    # Inset del comienzo de la parte real de la FID
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:40], SGL[0:40].real, color='coral')
    axins1.plot(t[:points], SGL[:points].real, color='teal')

    # Plot de la parte imaginaria de la FID
    axs[1,0].scatter(t, SGL.imag)
    axs[1,0].plot(t[:points], SGL[:points].imag, lw = 10, color='red')
    axs[1,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,0].set_xlabel(r't [$\mu$s]')
    axs[1,0].set_ylabel('FID (imag. part)')

    # Preparación del espectro
    mask = (CS>-5)&(CS<5)
    max_peak = np.max(spec[mask].real)
    specNorm = spec / max_peak
    area_peak = np.sum(spec[mask].real)
    peaks, _ = find_peaks(specNorm[mask].real, height=0.9)
    peaksx, peaksy = CS[mask][peaks], specNorm[mask][peaks].real
    
    # Plot de la parte real del espectro, zoom en el pico
    axs[0,1].plot(CS, specNorm.real, color='coral')
    axs[0,1].fill_between(CS[mask], 0, specNorm[mask].real, label = fr'Peak area = {area_peak:.0f}', alpha = 0.25, color="teal")
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
    axins2.plot(CS, specNorm.real, color='coral')

    # Plot de la parte imaginaria del espectro
    axs[1,1].scatter(CS, specNorm.imag)
    axs[1,1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,1].axvline(x=0, color='k', ls=':', lw=4)
    axs[1,1].set_xlim(-5, 5)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].set_ylabel('Norm. Spec. (imag. part)')

    plt.savefig(f'{Out}FID')


def DQ_bu(SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, Out, lenvd):
    '''
    Grafica resultados de la DQ.
    '''

    points = 10
    fid00, fidPts, fidPtsSD, pArea = [], [], [], []
    mask = (CS>-5)&(CS<5)
    
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

    if DQfilter == None:
            fig.suptitle(rf'nS = {nS:.0f}    |    RDT = {RDT} $\mu$s    |    RG = {RG:.1f} dB    |    Atten = {att:.0f} dB    |    p90 = {p90} $\mu$s  \
                 DQ-filter = None    |    RD = {RD:.4f} s    |    Evol = {evol:.6f} s    |    z-Filter = {zFilter:.6f} s', fontsize='medium')
    else:
        fig.suptitle(rf'nS = {nS:.0f}    |    RDT = {RDT} $\mu$s    |    RG = {RG:.1f} dB    |    Atten = {att:.0f} dB    |    p90 = {p90} $\mu$s  \
                DQ-filter = {DQfilter:.4f} us    |    RD = {RD:.4f} s    |    Evol = {evol:.6f} s    |    z-Filter = {zFilter:.6f} s', fontsize='medium')
    # Plot del primer punto de la FID
    axs[0].set_title('Primer punto de cada FID')
    axs[0].scatter(vd, fid00, label=rf'Max = {vd[fid00 == np.max(fid00)][0]} $\mu$s', color='tab:blue')
    axs[0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0].set_xlabel(r't [$\mu$s]')
    axs[0].legend()

    # Plot del promedio de los primeros puntos de la FID
    axs[1].set_title('Primeros 10 puntos de cada FID')
    axs[1].scatter(vd, fidPts, label=rf'Max = {vd[fidPts == np.max(fidPts)][0]} $\mu$s', color='tab:orange')
    axs[1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1].set_xlabel(r't [$\mu$s]')
    axs[1].legend()

    # Plot de las áreas de los espectros
    axs[2].set_title('Área de los picos de cada espectro')
    axs[2].scatter(vd, pArea, label=rf'Max = {vd[pArea == np.max(pArea)][0]} $\mu$s', color='tab:green')
    axs[2].axhline(y=0, color='k', ls=':', lw=4)
    axs[2].set_xlabel(r't [$\mu$s]')
    axs[2].legend()

    plt.savefig(f'{Out}DQ_bu')

    return fid00, fidPts, fidPtsSD, pArea


def DQ_verbose(t, SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, Out):
    
    for k in range(len(vd)):
        fig, axs = plt.subplots(2, 2, figsize=(50, 20), gridspec_kw={'height_ratios': [3,1]})
        if DQfilter == None:
            fig.suptitle(rf'vd = {vd[k]:.2f} $\mu$s    |    nS = {nS:.0f}    |    RDT = {RDT} $\mu$s    |    RG = {RG:.1f} dB    |    Atten = {att:.0f} dB    |    p90 = {p90} $\mu$s    |    DQ-filter = None    |    RD = {RD:.4f} s    |    Evol = {evol:.6f} s    |    z-Filter = {zFilter:.6f} s', fontsize='medium')
        else:
            fig.suptitle(rf'vd = {vd[k]:.2f} $\mu$s    |    nS = {nS:.0f}    |    RDT = {RDT} $\mu$s    |    RG = {RG:.1f} dB    |    Atten = {att:.0f} dB    |    p90 = {p90} $\mu$s    |    DQ-filter = {DQfilter:.4f} us    |    RD = {RD:.4f} s    |    Evol = {evol:.6f} s    |    z-Filter = {zFilter:.6f} s', fontsize='medium')

        # Promedio de los primeros 10 puntos de la FID
        points = 10
        fid0Arr = SGL[k, :points].real
        fid0 = sum(fid0Arr) / points
        fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

        # Plot de la parte real de la FID
        axs[0,0].scatter(t, SGL[k, :].real, label = fr'$M_R (0)$ = {SGL[k, 0].real:.0f}', color='coral')
        axs[0,0].plot(t[:points], SGL[k, :points].real, lw = 10, label = fr'$M_R ({points})$ = ({fid0:.0f} $\pm$ {fid0_SD:.0f})', color='teal')
        axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
        axs[0,0].set_xlabel(r't [$\mu$s]')
        axs[0,0].set_ylabel('FID (real part)')
        axs[0,0].legend()

        # Inset del comienzo de la parte real de la FID
        axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
        axins1.scatter(t[0:40], SGL[k, 0:40].real, color='coral')
        axins1.plot(t[:points], SGL[k, :points].real, color='teal')

        # Plot de la parte imaginaria de la FID
        axs[1,0].scatter(t, SGL[k, :].imag)
        axs[1,0].plot(t[:points], SGL[k, :points].imag, lw = 10, color='red')
        axs[1,0].axhline(y=0, color='k', ls=':', lw=4)
        axs[1,0].set_xlabel(r't [$\mu$s]')
        axs[1,0].set_ylabel('FID (imag. part)')

        # Preparación del espectro
        mask = (CS>-5)&(CS<5)
        max_peak = np.max(spec[k, mask].real)
        specNorm = spec[k, :] / max_peak
        area_peak = np.sum(spec[k, mask].real)
        peaks, _ = find_peaks(specNorm[mask].real, height=0.9)
        peaksx, peaksy = CS[mask][peaks], specNorm[mask][peaks].real
        
        # Plot de la parte real del espectro, zoom en el pico
        axs[0,1].plot(CS, specNorm.real, color='coral')
        axs[0,1].fill_between(CS[mask], 0, specNorm[mask].real, label = fr'Peak area = {area_peak:.0f}', alpha = 0.25, color="teal")
        axs[0,1].plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
        axs[0,1].annotate(f'{peaksx[0]:.4f} ppm', xy = (peaksx[0], peaksy[0] + 0.07), fontsize=30, ha='center') 
        axs[0,1].set_xlim(-10, 10)
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
        axins2.plot(CS, specNorm.real, color='coral')

        # Plot de la parte imaginaria del espectro
        axs[1,1].scatter(CS, specNorm.imag)
        axs[1,1].axhline(y=0, color='k', ls=':', lw=4)
        axs[1,1].axvline(x=0, color='k', ls=':', lw=4)
        axs[1,1].set_xlim(-10, 10)
        axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1,1].set_xlabel(r'$\delta$ [ppm]')
        axs[1,1].set_ylabel('Norm. Spec. (imag. part)')

        plt.savefig(f'{Out}FID_{k}')