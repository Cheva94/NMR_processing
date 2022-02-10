#!/usr/bin/python3.6

'''
    Description: plots several FIDs together. They all must be already processed.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange',
                                        'mediumseagreen', 'm', 'y', 'k'])

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5

plt.rcParams["legend.loc"] = 'upper right'
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

def main():

    Files = args.input
    Labels = args.labels
    Saving = args.output

    fig, ax = plt.subplots()
    for i in range(len(Files)):
        data = read_csv(f'{Files[i]}').to_numpy()
        ax.plot(data[:, 0], data[:, 1], label=Labels[i])

    ax.set_xlabel('t [ms]')
    ax.set_ylabel(r'$M_R$')

    plt.savefig(f'{Saving}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Plots several FIDs together.')

    parser.add_argument('input', help = "Path to the input file.", nargs = '+')

    parser.add_argument('labels', help = "Labels to be used in the plot.", nargs = '+')

    parser.add_argument('output', help = "Path for the output file.")

    args = parser.parse_args()

    main()