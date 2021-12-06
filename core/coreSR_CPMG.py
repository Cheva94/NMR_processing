#!/usr/bin/python3.6

'''
    Description: core functions for SR_CPMG.py.

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
# from cycler import cycler
# import scipy.fft as FT

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange', 'mediumseagreen', 'm', 'y', 'k'])

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5

plt.rcParams["legend.loc"] = 'best'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 13.5
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linestyle"] = '-'

def flint(K1,K2,Z,alpha,S):
    '''
    Fast 2D NMR relaxation distribution estimation.

    Section 4: although the Lipshitz constant there does not have alpha added as it should have)
    '''


    return S, resida
