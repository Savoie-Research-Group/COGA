#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
home=str(Path.home())
sys.path.append('{}/bin/CG_Crystal'.format(home))
sys.path.append('{}/bin/'.format(home))
from GA_plot import plot_ac

def main(argv):
    parser = argparse.ArgumentParser(description='''Enter description here                                                                                                                                           
    Input: 
                                                                                                                                                                                                      
    Output:                                                                                                                                                                                         
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('data_file', type=str, help='Name of data file to plot')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG_files/', help='Output. Default: ')
        
    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    plot_ac(args.data_file)


    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
