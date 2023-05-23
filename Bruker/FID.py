#!/usr/bin/python3.10
# -*- coding: utf-8 -*-

import argparse
from coreFID import *

def main():

    fileDir = args.input
    Out = fileDir+'/'+'test_FID_Bruker'

    t, sgl, nP, sw = FID_file(fileDir)

    sgl = PhCorr(sgl)

    CS, spec = espec(sgl, nP, sw)

    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3,1]})

    # Promedio de los primeros 10 puntos de la FID
    points = 20 #10
    fid0Arr = sgl.real[:points]
    fid0 = sum(fid0Arr) / points
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

    # Plot de la parte real de la FID
    axs[0,0].scatter(t, sgl.real, label='FID (real)', color='coral')
    axs[0,0].plot(t[:points], sgl.real[:points], lw = 10, label = fr'$M_R ({points})$ = ({fid0:.2f} $\pm$ {fid0_SD:.2f})', color='teal')
    axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,0].set_xlabel('t [ms]')
    axs[0,0].set_ylabel('FID')
    axs[0,0].legend()

    # Inset del comienzo de la parte real de la FID
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:40], sgl[0:40].real, color='coral')
    axins1.plot(t[:points], sgl.real[:points], color='teal')

    # Plot de la parte imaginaria de la FID
    axs[1,0].scatter(t, sgl.imag, label='FID (imag)')
    axs[1,0].plot(t[:points], sgl.imag[:points], lw = 10, color='red')
    axs[1,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,0].set_xlabel('t [ms]')
    axs[1,0].set_ylabel('FID')
    axs[1,0].legend()

    # mask = (CS>-0.05)&(CS<0.1)
    # max_peak = np.max(spec.real[mask])
    # spec /= max_peak
    # area_peak = np.sum(spec.real[mask])
    # peaks, _ = find_peaks(spec.real[mask], height=0.9)
    # peaksx, peaksy = CS[mask][peaks], spec.real[mask][peaks]
    
    # Plot de la parte real del espectro, zoom en el pico
    axs[0,1].plot(CS, spec.real, label='Spectrum (real)', color='coral')
    # axs[0,1].fill_between(CS[mask], 0, spec.real[mask], label = fr'Peak area = {area_peak:.0f}', alpha = 0.25, color="teal")
    # axs[0,1].plot(peaksx[0], peaksy[0] + 0.05, lw = 0, marker=11, color='black')
    # axs[0,1].annotate(f'{peaksx[0]:.4f}', xy = (peaksx[0], peaksy[0] + 0.07), fontsize=30, ha='center') 
    # axs[0,1].set_xlim(-0.2, 0.2)
    # axs[0,1].set_ylim(-0.05, 1.2)
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,1].set_xlabel(r'$\delta$ [ppm]')
    axs[0,1].axvline(x=0, color='k', ls=':', lw=2)
    axs[0,1].axhline(y=0, color='k', ls=':', lw=2)
    axs[0,1].legend(loc='upper right')

    # Inset del espectro completo
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
    axins2.tick_params(labelleft=False)
    axins2.plot(CS, spec.real, color='coral')

    # Plot de la parte imaginaria del espectro
    axs[1,1].scatter(CS, spec.imag, label='Spectrum (imag)')
    axs[1,1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,1].axvline(x=0, color='k', ls=':', lw=4)
    # axs[1,1].set_xlim(-0.1, 0.1)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].legend()

    plt.savefig(f'{Out}')

    # print('Writing output...')

    # with open(f'{Out}.csv', 'w') as f:
    #     f.write("t [ms]\tRe[FID]\tIm[FID] \n")
    #     for i in range(nP):
    #         f.write(f'{t[i]:.6f}\t{sgl.real[i]:.6f}\t{sgl.imag[i]:.6f} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID fileDir.")
# #    parser.add_argument('output', help = "Path for the output fileDirs.")
#     parser.add_argument('-back', '--background', help = "Path to de FID background fileDir.")
#     parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning the FID.", type = int, default=0)

    args = parser.parse_args()

    main()
