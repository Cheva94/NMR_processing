#!/usr/bin/python3.10

import argparse
import brukerIO as IO
import brukerPlot as graph

def main():

    print('Analysing raw data...')

    fileDir = args.input
    Out = fileDir+'/'+'proc_FID'

    t, SGL, nP, SW, nS, RDT, RG, att, RD, p90 = IO.readFID(fileDir)
    SGL = IO.PhCorrFID(SGL)
    CS, spec = IO.spectrum(SGL, nP, SW)

    print('Writing processed data...')

    IO.writeFID(t, SGL, nP, CS, spec, Out)

    print('Plotting processed data...')

    graph.FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID fileDir.")
# #    parser.add_argument('output', help = "Path for the output fileDirs.")
#     parser.add_argument('-back', '--background', help = "Path to de FID background fileDir.")
#     parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning the FID.", type = int, default=0)

    args = parser.parse_args()

    main()
