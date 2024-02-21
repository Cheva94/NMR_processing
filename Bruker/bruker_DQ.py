#!/usr/bin/python3.10

import argparse
import bruker_IO as IO
import bruker_Plot as graph
import os

def main():

    print('Analysing FID raw data from DQ...')
    fileDir = args.input
    mlim = args.mlim
    verbose = args.verbose
    Out = fileDir.split('/')[0]+'_procDQ/'
    isExist = os.path.exists(Out)
    if not isExist:
        os.makedirs(Out)
    phasecorr = args.phasecorr

    t, SGL, nP, SW, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, DQfilter, DQfilterzFil = IO.readDQ(fileDir)
    lenvd = len(vd)
    SGL, phasecorr = IO.PhCorrDQ(SGL, lenvd, phasecorr)
    print(f'La correción de fase se hizo con {phasecorr}°.')
    CS, spec = IO.specDQ(SGL, nP, SW, lenvd)

    print('Writing acquisition parameters...')
    IO.writeDQ_acq(nS, RDT, RG, att, RD, evol, zFilter, p90, vd, DQfilter, DQfilterzFil, Out, lenvd)

    if verbose == True:
        print('Writing FID processed data from DQ...')
        IO.writeDQ_verbose(t, SGL, nP, CS, spec, Out, lenvd)

        print('Plotting FID processed data from DQ...')
        graph.DQ_verbose(t, SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, DQfilterzFil, Out, lenvd, mlim)

    print('Plotting build-up processed data...')
    fid00, fidPts, fidPtsSD, pArea = graph.DQ_bu(SGL, nS, RDT, RG, att, RD, evol, zFilter, p90, vd, CS, spec, DQfilter, DQfilterzFil, Out, lenvd, mlim)

    print('Writing build-up results...')
    IO.writeDQ(vd, fid00, fidPts, fidPtsSD, pArea, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the DQ fileDir.")
    parser.add_argument('mlim', help = "Mask limits to integrate spectrum.", type=int)
    parser.add_argument('-v', '--verbose', help = "Write every single FID and spectrum.", type = bool, default=False)
    parser.add_argument('-ph', '--phasecorr', help = "Phase correction to use.", type = int, default=None)
    args = parser.parse_args()
    main()