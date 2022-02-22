#!/usr/bin/python3.6

'''
    Description:

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import numpy as np
import pandas as pd
from random import random

def main():
    T11, T12 = 10, 100
    T21, T22 = 0.5, 20
    A1, A2 = 1, 1
    d1, d2 = 100, 10000

    tau1 = np.logspace(-3, 4, d1)
    tau2 = np.logspace(-3, 3, d2)

    R11, R12 = np.zeros((d1,1)), np.zeros((d1,1))
    R21, R22 = np.zeros((d2,1)), np.zeros((d2,1))
    M = np.zeros((d1, d2))

    for k in range(d1):
        R11[k] = A1 * (1 - np.exp(-tau1[k] / T11))
        R12[k] = A2 * (1 - np.exp(-tau1[k] / T12))

    for i in range(d2):
        R21[i] = A1 * np.exp(-tau2[i] / T21)
        R22[i] = A2 * np.exp(-tau2[i] / T22)

    M1 = R11 * R21.T
    M2 = R12 * R22.T
    M = M1 + M2

    noise = 0.03 * np.max(M)
    Mnoise = M + noise * (2*random() - 1)

    np.savetxt("synth.txt", M, delimiter='\t')
    np.savetxt("synth_t1.dat", tau1, delimiter='\t')
    np.savetxt("synth_t2.dat", tau2, delimiter='\t')
    np.savetxt("synth_noise.txt", Mnoise, delimiter='\t')
    np.savetxt("synth_noise_t1.dat", tau1, delimiter='\t')
    np.savetxt("synth_noise_t2.dat", tau2, delimiter='\t')

if __name__ == "__main__":
    main()
