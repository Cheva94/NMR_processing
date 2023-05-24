#!/usr/bin/python3.10

import argparse
import brukerIO as IO
import brukerPlot as graph
import os

def main():

    print('Analysing FID raw data from DQ...')
    fileDir = args.input
    verbose = args.verbose
    Out = fileDir.split('/')[0]+'_procDQ/'
    isExist = os.path.exists(Out)
    if not isExist:
        os.makedirs(Out)

    t, SGL, nP, SW, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, DQfilter = IO.readDQ(fileDir)
    lenvd = len(vd)
    SGL = IO.PhCorrDQ(SGL, lenvd)
    CS, spec = IO.specDQ(SGL, nP, SW, lenvd)

    if verbose == True:
        # print('Writing FID processed data from DQ...')
        # IO.writeDQ_verbose(t, SGL, nP, CS, spec, Out, lenvd)

        print('Plotting FID processed data from DQ...')
        graph.DQ_verbose(t, SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, Out)

    print('Plotting build-up processed data...')
    fid00, fidPts, fidPtsSD, pArea = graph.DQ_bu(SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, Out, lenvd)

    print('Writing build-up results...')
    IO.writeDQ(vd, fid00, fidPts, fidPtsSD, pArea, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the DQ fileDir.")
    parser.add_argument('-v', '--verbose', help = "Write every single FID and spectrum.", type = bool, default=False)
    args = parser.parse_args()
    main()