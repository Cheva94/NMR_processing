#!/usr/bin/python3.6

'''
    Description: plots FID and its spectrum, with phase correction.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreFID import *

def main():

    Files = args.input_file

    for F in Files:

        t, Np, dw, FID, ns, rd, rg = userfile(F)

        FID = phase_correction(FID)

        freq, spec, max_peak = spectrum(FID, Np, dw)

        if args.FID:
            plot_FID(t, FID, ns, rd, rg, F)

        elif args.spectrum:
            if args.MiniSpec:
                plot_spec_mini(freq, spec, max_peak, ns, rd, rg, F)
            elif args.Bruker:
                plot_spec_bruker(freq, spec, max_peak, ns, rd, rg, F)
            else:
                plot_spec_freq(freq, spec, max_peak, ns, rd, rg, F)

        else:
            print('Must choose an option: -fid, -spec, -mini or -bruker. Use -h for guidance.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', help = "Path to the input files.", nargs = '+')

    parser.add_argument('-fid', '--FID', action = 'store_true', help = "Plots FID with phase correction.")

    parser.add_argument('-spec', '--spectrum', action = 'store_true', help = "Plots spectrum vs frequency, with phase correction.")

    parser.add_argument('-mini', '--MiniSpec', action = 'store_true', help = "Plots spectrum vs CS [ppm] (Minispec=20MHz), with phase correction.")

    parser.add_argument('-bruker', '--Bruker', action = 'store_true', help = "Plots spectrum vs CS [ppm] (Bruker=300MHz), with phase correction.")

    args = parser.parse_args()

    main()
