#!/usr/bin/python3.10

import argparse
import brukerIO as IO
import brukerPlot as graph

def main():

    print('Analysing raw data...')
    fileDir = args.input
    verbose = args.verbose
    Out = fileDir+'/'+'proc_DQ'

    t, SGL, nP, SW, nS, RDT, RG, att, RD, evol, zFilter, p90, vd = IO.readDQ(fileDir)
    SGL = IO.PhCorrDQ(SGL)
    CS, spec = IO.specDQ(SGL, nP, SW)
    
    if verbose == True:
        print('Writing processed data...')
        IO.writeDQ_verbose(t, SGL, nP, CS, spec, Out)

    print('Plotting processed data...')
    fid00, fidPts, fidPtsSD, pArea = graph.DQ(t, SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, Out)

    print('Writing build-up results...')
    IO.writeDQ(vd, fid00, fidPts, fidPtsSD, pArea, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the FID fileDir.")
    parser.add_argument('-v', '--verbose', help = "Write every single FID and spectrum.", type = bool, default=False)
    args = parser.parse_args()
    main()