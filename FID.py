#!/usr/bin/python3.6

'''
    Description: corrects phase of FID and normalizes it considering the receiver
    gain. It may also normalize by mass of 1H when given. Then plots FID and
    transforms it to get spectrum in Hz and ppm. All the processed data will be
    saved in ouput files (.csv). It may substract the background when given.

    Notes: doesn't normalize the background by it mass yet (only by RG).

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreFID import *

def main():

    file = args.input
    m = args.mass
    back = args.background
    show = args.ShowPlot

    t, nP, DW, FID, nS, RD, RG = userfile(file)
    FID = PhCorrNorm(FID, RG, m)

    if back == None:
        fileRoot = file.split(".txt")[0]
        plot_FID(t, FID, nS, RD, fileRoot)
        out_FID(t, FID, fileRoot)

        freq, spec, max_peak = spectrum(FID, nP, DW)
        plot_spec(freq, spec, max_peak, nS, RD, fileRoot)
        out_spec(freq, spec, fileRoot)

    # else:
    #     t_B, nP_B, DW_B, back, nS_B, RD_B, RG_B = userfile(back)
    #     if nP != nP_B:
    #         print('Both files must have same number of points. Quitting job.')
    #         exit()
    #
    #     back = phase_correction(back)
    #     back = normalize(back, RG_B)
    #
    #     FID = back_subs(FID, back)
    #
    #     fileRoot = file.split(".txt")[0]+'-BackSub'
    #     plot_FID(t, FID, nS, RD, fileRoot)
    #     out_FID(t, FID, fileRoot)
    #
    #     freq, spec, max_peak = spectrum(FID, nP, DW)
    #     plot_spec(freq, spec, max_peak, nS, RD, fileRoot)
    #     out_spec(freq, spec, fileRoot)

    if show == 'on':
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Corrects phase of FID and normalizes it considering the receiver gain. It may also normalize by mass of 1H when given. Then plots FID and transforms it to get spectrum in Hz and ppm. All the processed data will be saved in ouput files (.csv). It may substract the background when given. \n\n Notes: doesn't normalize the background by it mass yet (only by RG).")

    parser.add_argument('input', help = "Path to the input file.")

    parser.add_argument('-m', '--mass', help = "Sample mass.", type = float, default = 1)

    parser.add_argument('-show', '--ShowPlot', help = "Show plots.", default = 'off')

    parser.add_argument('-back', '--background', help = "Substracts the file given to the input file. It is NOT assumed that the background is already processed.")

    args = parser.parse_args()

    main()
