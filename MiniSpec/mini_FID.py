#!/usr/bin/python3.10

import argparse
import mini_IO as IO
import mini_Plot as graph


def main():

    path = args.path
    root = path.split(".txt")[0]
    ppm = args.ppm

    print('Reading FID raw data...')
    t, SGL, nP, DW = IO.readFID(path)
    nS, RDT, RG, att, RD, p90 = IO.readFIDparams(path)

    print('Analysing FID raw data...')
    SGL = IO.PhCorrFID(SGL)
    SGL = IO.NormFID(SGL, RG)
    
    # Nomber of points to drop at the FID beginning.
    pDrop = SGL.real[0:30].argmax()
    t, SGL, nP = t[pDrop:], SGL[pDrop:], nP-pDrop

    CS, spec = IO.specFID(SGL, nP, DW)

    print('Writing FID processed data...')
    IO.writeFID(t, SGL, nP, CS, spec, root, ppm)

    print('Plotting FID processed data...')
    graph.FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, root, ppm, pDrop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = "Path to the FID signal.")
    parser.add_argument('-ppm', help = "Plus/minus limits (in ppm) to integrate spectrum.", type=float, default = 0.05)
    args = parser.parse_args()
    main()