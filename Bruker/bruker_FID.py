#!/usr/bin/python3.10

import argparse
import bruker_IO as IO
import bruker_Plot as graph
import os

def main():

    print('Analysing FID raw data...')
    fileDir = args.input
    mlim = args.mlim
    Out = fileDir.split('/')[0]+'_procFID/'
    isExist = os.path.exists(Out)
    if not isExist:
        os.makedirs(Out)

    t, SGL, nP, SW, nS, RDT, RG, att, RD, p90 = IO.readFID(fileDir)
    SGL = IO.PhCorrFID(SGL)
    CS, spec = IO.specFID(SGL, nP, SW)

    print('Writing acquisition parameters...')
    IO.writeFID_acq(nS, RDT, RG, att, RD, p90, Out)

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