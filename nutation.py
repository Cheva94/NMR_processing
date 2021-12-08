
#!/usr/bin/python3.6
'''
    Description: corrects phase of FID and normalizes it considering the receiver
    gain. It may also normalize by mass of 1H when given. Then plots FID and
    transforms it to get spectrum in Hz and ppm. All the processed data will be
    saved in ouput files (.csv). It may substract the background when given.

    Notes: doesn't normalize the background by it mass yet (only by RG).

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange', 'mediumseagreen', 'k', 'm', 'y'])

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5

plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def main():

    Files = args.input
    t, M = [], []
    A = []

    for F in Files:
        data = pd.read_csv(F, header = None, delim_whitespace = True).to_numpy()
        for k in range(np.shape(data)[0]):
            t.append(data[k, 0])
            M.append(data[k, 1])

    t = np.array(t)
    M = np.array(M)
    S = np.argsort(t)

    t = t[S]
    M = M[S]

    with open('nutation.csv', 'w') as f:
        f.write("t [us], M \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {M[i]:.4f} \n')

    fig, ax = plt.subplots()

    ax.scatter(t, M)
    ax.minorticks_on()
    ax.grid(True, which='both')
    plt.axhline(0, color='black', zorder=-2)
    ax.set_xlabel('t [us]')
    ax.set_ylabel('M')

    plt.savefig('nutation')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the input file.", nargs = '+')

    args = parser.parse_args()

    main()
