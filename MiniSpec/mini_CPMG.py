#!/usr/bin/python3.10

import argparse
import mini_IO as IO
import mini_Plot as graph


def main():

    print('Analysing CPMG raw data...')
    File = args.input
    Out = File.split(".txt")[0]
    alpha = args.alpha
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    nini = args.croppedValues






    t, SGL, nP, DW, nS, RDT, RG, att, RD, p90 = IO.readCPMG(File)
    SGL = IO.PhCorrCPMG(SGL)
    SGL = IO.NormCPMG(SGL, RG)
    CS, spec = IO.specCPMG(SGL, nP, DW)

    print('Writing CPMG processed data...')
    IO.writeCPMG(t, SGL, nP, CS, spec, Out)

    print('Plotting CPMG processed data...')
    graph.CPMG(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the CPMG file.")
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [0, 4])
    # parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=1)
    args = parser.parse_args()
    main()