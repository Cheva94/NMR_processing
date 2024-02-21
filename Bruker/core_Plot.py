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
    axs[0].scatter(vp, fid00, label=rf'Max = {vp[fid00 == np.max(fid00)][0]} $\mu$s', color='tab:blue')
    axs[0].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[0].set_xlabel(r't [$\mu$s]')
    axs[0].legend()
    m = np.floor(np.log10(np.max(fid00)))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    # Plot del promedio de los primeros puntos de la FID
    axs[1].set_title('Primeros 10 puntos de cada FID')
    axs[1].scatter(vp, fidPts, label=rf'Max = {vp[fidPts == np.max(fidPts)][0]} $\mu$s', color='tab:orange')
    axs[1].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[1].set_xlabel(r't [$\mu$s]')
    axs[1].legend()
    m = np.floor(np.log10(np.max(fidPts)))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(m,m))

    plt.savefig(f'{Out}Nutac')

    return fid00, fidPts

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
    axs[0].scatter(vd, fid00, label=rf'Max = {vd[fid00 == np.max(fid00[1:])][0]} $\mu$s', color='tab:blue')
    axs[0].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[0].set_xlabel(r't [$\mu$s]')
    m = np.floor(np.log10(np.max(fid00)))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(m,m))
    axs[0].legend()

    # Plot del promedio de los primeros puntos de la FID
    axs[1].set_title('Primeros 10 puntos de cada FID')
    axs[1].scatter(vd, fidPts, label=rf'Max = {vd[fidPts == np.max(fidPts[1:])][0]} $\mu$s', color='tab:orange')
    axs[1].axhline(y=0, color=Gris, ls=':', lw=4)
    axs[1].set_xlabel(r't [$\mu$s]')
    m = np.floor(np.log10(np.max(fidPts)))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(m,m))
    axs[1].legend()

    # Plot de las áreas de los espectros
    axs[2].set_title('Área de los picos de cada espectro')
    axs[2].scatter(vd, pArea, label=rf'Max = {vd[pArea == np.max(pArea[1:])][0]} $\mu$s', color='tab:green')
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