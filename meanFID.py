#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
from core.coremeanFID import *

def main():

    FileArr = args.input
    Out = args.output
    show = args.ShowPlot

    signalArr = []
    nF = len(FileArr)

    for F in FileArr:
        t, signal = FID_file(F)
        signal = PhCorrNorm(signal)
        signalArr.append(signal)

    signalArr = np.reshape(signalArr, (nF, len(t))).T
    signalMean = signalArr.mean(axis=1)

    with open(f'{Out}.txt', 'w') as f:
        for k in range(len(t)):
            f.write(f'{t[k]:.6f}    {signalMean.real[k]:.6f}    {signalMean.imag[k]:.6f} \n')

    plot(t, signalArr, signalMean, nF, Out)

    if show == 'on':
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-show', '--ShowPlot', help = "Show plots.", default = 'off')

    args = parser.parse_args()

    main()
