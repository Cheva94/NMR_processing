#!/usr/bin/python3.10

import argparse
import bruker_IO as IO
import bruker_Plot as graph

def main():

    print('Analysing FID raw data...')
    fileDir = args.input
    Out = fileDir+'proc_FID'

    t, SGL, nP, SW, nS, RDT, RG, att, RD, p90 = IO.readFID(fileDir)
    SGL = IO.PhCorrFID(SGL)
    CS, spec = IO.specFID(SGL, nP, SW)

    print('Writing FID processed data...')
    IO.writeFID(t, SGL, nP, CS, spec, Out)

    print('Plotting FID processed data...')
    graph.FID(t, SGL, nS, RDT, RG, att, RD, p90, CS, spec, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the FID fileDir.")
    args = parser.parse_args()
    main()