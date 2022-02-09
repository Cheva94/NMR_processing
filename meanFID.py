#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
from core.coremeanFID import *

def main():

    FileArr = args.input
    Out = args.output
    show = args.ShowPlot

    for F in FileArr:
        t, signal, nP, DW = FID_file(F)
        signal = (PhCorr(signal))
        fig, ax = plt.subplots()

for k in range(nF):
    ax1.plot(t, signalArrRe[:, k], ls = '--')
    # ax2.plot(t, signalArr[:, k].imag, ls = '--')

ax1.set_xlabel('t [ms]')
ax1.set_ylabel(r'$M_R$')

# ax2.xaxis.tick_top()
# ax2.set_ylabel(r'$M_I$')

plt.savefig(f'{Out}')

# with open(f'{Out}.csv', 'w') as f:
#     f.write("t [ms], Re[FID], Im[FID] \n")
#     for i in range(len(t)):
#         f.write(f'{t[i]:.6f}, {signal.real[i]:.6f}, {signal.imag[i]:.6f} \n')




    # if show == 'on':
    #     plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-show', '--ShowPlot', help = "Show plots.", default = 'off')

    args = parser.parse_args()

    main()
