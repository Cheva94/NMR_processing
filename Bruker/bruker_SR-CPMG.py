#!/usr/bin/python3.10

import argparse
import bruker_IO as IO
import bruker_Plot as graph
import os

def main():

    print('WIP')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the FID fileDir.")
    parser.add_argument('mlim', help = "Mask limits to integrate spectrum.", type=int)
    args = parser.parse_args()
    main()