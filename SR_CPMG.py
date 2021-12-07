#!/usr/bin/python3.6

'''
    Description:

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import argparse
import numpy as np
import pandas as pd
from core.coreSR_CPMG import *

def main():

    File = args.input
    alpha = args.alpha #5E0

    Nx, Ny = 100, 100 # Number of bins in relaxation time grids
    T1 = np.logspace(-3,4,Nx)
    T2 = np.logspace(-3,3,Ny)
    S = np.ones((Nx, Ny)) # Initial guess del espectro

    rawData = pd.read_csv(File, header = None, delim_whitespace = True).to_numpy()[:, 0]
    # Re = rawData[:, 0]
    # Im = rawData[:, 1]
    # rawDecay = Re + Im * 1j # Complex signal

    taus = File.split('.txt')[0]
    tau1 = pd.read_csv(f'{taus+"_t1.dat"}', header = None, delim_whitespace = True).to_numpy()
    tau2 = pd.read_csv(f'{taus+"_t2.dat"}', header = None, delim_whitespace = True).to_numpy()
    N1, N2 = len(tau1), len(tau2) # Number of data points in each dimension

    # Por acá haría falta la corrección de fase
    data = np.reshape(rawData, (N2, N1))

    Z = data.T
    K1 = 1 - np.exp(-tau1 / T1)
    K2 = np.exp(-tau2 / T2)

    S, resida = flint(K1, K2, Z, alpha, S) # Numeric Laplace inversion (NLI) con FISTA

    np.savetxt("RatesSpectrum.csv", S, delimiter=',')

    plot_map(T2, T1, S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser = argparse.ArgumentParser(description="Corrects phase of CPMG decay and normalizes it considering the receiver gain. It may also normalize by mass of 1H when given. Then fits it considering 1, 2 or 3 exponentials. Finally it plots the decay in normal and semilog scales with the fitting. All the processed data will be also saved in ouput files (.csv). It may substract the background when given. \n\n Notes: doesn't normalize the background by it mass yet (only by RG).")
    #
    parser.add_argument('input', help = "Path to the inputs file.")

    parser.add_argument('alpha', help = "Tikhonov regularization parameter.", type = float)
    #
    # parser.add_argument('exponential_fit', help = "Fits exponential decay. Must choose mono, bi or tri to fit with 1, 2 or 3 exponentials, respectively.")
    #
    # parser.add_argument('-back', '--background', help = "Substracts the file given to the input file. It is NOT assumed that the background is already processed.")
    #
    # parser.add_argument('-mH', '--proton_mass', help = "Mass of protons in the sample.", type = float)

    args = parser.parse_args()

    main()
