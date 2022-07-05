#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
from core.meanCPMG import *

def main():

    FileArr = args.input
    Out = args.output
    Labels = args.labels
    nini = args.niniValues
    nH = args.protonMoles

    nF = range(len(FileArr))

    for k in nF:
        data = pd.read_csv(FileArr[k], header = None, delim_whitespace = True, comment='#').to_numpy()
        Laplace = data[3:153, :3]
        T2 = Laplace[:, 0]
        Cumulative = Laplace[:, 2]

        Time = data[154:, :3]
        t = Time[:, 0]
        Decay = Time[:, 1]

        return t[nini:], Decay[nini:], T2, Cumulative



        t, Decay, T2, Cumulative = CPMG_file(F, nini)
        signalArr.append(Decay)
        cumulArr.append(Cumulative)

    signalArr = np.reshape(signalArr, (nF, len(t))).T
    cumulArr = np.reshape(cumulArr, (nF, len(T2))).T

    plot(t, signalArr, T2, cumulArr, nF, Out, Labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-nH', '--protonMoles', type = float, default = 1)
    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-L', '--labels', nargs='+')
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning the FID.", type = int, default=0)

    args = parser.parse_args()

    main()
