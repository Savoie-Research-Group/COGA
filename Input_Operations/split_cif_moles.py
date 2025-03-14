#!/bin/env python                                                                                                                                                                                                                                                               
# Author: Dylan Fortney (ddfortne@purdue.edu)                                                                                                                                                                                                                                   

import os, sys, argparse
import numpy as np
sys.path.append('/home/ddfortne/bin/taffi_beta/Lib/')
sys.path.append('/home/ddfortne/bin/taffi_beta/Parsers')
from adjacency import Table_generator
from percolation_calc import find_subgraphs
sys.path.append('/home/ddfortne/bin/taffi/CG')
from file_operations import read_xyz, write_xyz

def main(argv):
    parser = argparse.ArgumentParser(description='''Separates cif file into constituent molecules, and outputs separate xyz files.                                                                                                                                          
    Input: crystal xyz file
                                                                                                                                                                                                     
    Output: Simulation files                                                                                                                                                                                        
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('crystal_xyz', type=str, help='Input taffi-style map file')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG_files/', help='The main output folder for output mapping files. Default: CG_files')

    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    elements, geometry, charge = read_xyz(args.crystal_xyz) # Read xyz file into lists for outputs
    Adj_mat = Table_generator(elements, geometry) # Pass elements and geometry into Table generator to find adjacency matrix.
    #print(Adj_mat)
    SG_list=find_subgraphs(Adj_mat)
    N_Mols=len(SG_list[:])
    print("Found "+ str(N_Mols) +" molecules in file...")
    Geo_np=np.array(geometry)
    ele_np=np.array(elements)
    #print(Geo_np[SG_list[1]])
    for i in range(N_Mols):
        write_xyz(Geo_np[SG_list[i]], 'NDI_{}.xyz'.format(str(i+1)), ele_np[SG_list[i]])

    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])

