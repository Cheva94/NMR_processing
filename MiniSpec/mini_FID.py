#!/usr/bin/python3.10

import argparse
import core_IO as IO
import core_Plot as graph


def main():

    path = args.path
    root = path.split(".txt")[0]
    ppm = args.ppm

    print('Reading FID raw data...')
    t, SGL, nP, DW = IO.read1Dsgl(path)
    nS, RDT, RG, att, RD, p90, _, _, _ = IO.read1Dparams(root)
    params = (rf'Acquisition: RDT = {RDT} $\mu$s | Atten = {att} dB | '
              rf'RG = {RG} dB | nS = {nS} | RD = {RD:.2f} s | '
              rf'p90 = {p90} $\mu$s')

    print('Analysing FID raw data...')
    SGL = IO.PhCorr1D(SGL)
    SGL = IO.NormRG1D(SGL, RG)
    
    # Nomber of points to drop at the FID beginning.
    pDrop = SGL.real[0:30].argmax()
    t, SGL, nP = t[pDrop:], SGL[pDrop:], nP-pDrop

    CS, spec = IO.specFID(SGL, nP, DW)

    print('Writing FID processed data...')
    IO.writeFID(t, SGL, nP, CS, spec, root, ppm)
    
    print('Plotting FID processed data...')
    graph.FID(t, SGL, CS, spec, root, ppm, params, pDrop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default = 'FID.txt', 
                        help = "Path to the FID signal.")
    parser.add_argument('-ppm', type=float, default = 0.05, 
                        help = "Plus/minus limits (in ppm) to integrate spec.")
    args = parser.parse_args()
    main()