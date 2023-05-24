#!/usr/bin/python3.10

import argparse
import bruker_IO as IO
import bruker_Plot as graph
import os

def main():

    print('Analysing nutation raw data...')
    fileDir = args.input
    Out = fileDir.split('/')[0]+'_procFID/'
    isExist = os.path.exists(Out)
    if not isExist:
        os.makedirs(Out)

    vp, SGL, nS, RDT, RG, att, RD = IO.readNutac(fileDir)
    lenvp = len(vp)
    # SGL = IO.PhCorrNutac(SGL, lenvp)

    print('Writing acquisition parameters...')
    IO.writeNutac_acq(nS, RDT, RG, att, RD, vp, Out, lenvp)

    print('Plotting nutation processed data...')
    fid00, fidPts = graph.Nutac(SGL, nS, RDT, RG, att, RD, vp, Out, lenvp)

    print('Writing nutation results...')
    IO.writeNutac(vp, fid00, fidPts, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the FID fileDir.")
    args = parser.parse_args()
    main()