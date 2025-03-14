#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
home=str(Path.home())

def main(argv):
    parser = argparse.ArgumentParser(description='''Enter description here                                                                                                                                           
    Input: 
                                                                                                                                                                                                      
    Output:                                                                                                                                                                                         
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('input', type=str, help='Input Description')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG_files/', help='Output. Default: ')
        
    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below



    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
