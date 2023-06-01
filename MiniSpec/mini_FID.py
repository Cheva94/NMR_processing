#!/usr/bin/python3.10

import argparse
import mini_IO as IO
import mini_Plot as graph


def main():

    print('Analysing FID raw data...')
    File = args.input
    mlim = args.mlim
    Out = File.split(".txt")[0]

    t, SGL, nP, DW, nS, RDT, RG, att, RD, p90 = IO.readFID(File)
    SGL = IO.PhCorrFID(SGL)
    SGL = IO.NormFID(SGL, RG)
    CS, spec = IO.specFID(SGL, nP, DW)

    print('Writing FID processed data...')
    IO.writeFID(t, SGL, nP, CS, spec, Out, mlim)

    print('Plotting FID processed data...')
    graph.FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out, mlim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the FID fileDir.")
    parser.add_argument('mlim', help = "Mask limits to integrate spectrum.", type=float)
    args = parser.parse_args()
    main()