#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
import subprocess as sp
from pathlib import Path
home=str(Path.home())
sys.path.append('{}/bin/CG_Crystal'.format(home))
from auto_genetic_martini import read_thermo

def main(argv):
    parser = argparse.ArgumentParser(description='''Enter description here                                                                                                                                           
    Input: number of parameters, name of species
                                                                                                                                                                                                      
    Output: rdfs in a variety of folders!                                                                                                                                                                                        
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('params_num', type=str, help='Number of parameters in model.')
    parser.add_argument('species', type=str, help='Species name.')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG_files/', help='Output. Default: ')
        
    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below

    names_L1=["Loss", "Press", "SLoss"] # List of folder names in first level of directory
    names_L2=["Cold", "NPT", "Ramp"] # List of folder names in second level of directory
    current=os.getcwd()
    with open("{}/volumes.txt".format(current), 'w') as vol:
        vol.write("folder\taverage volume\tinitial volume\tfinal volume\tf/i ratio\n")
        for i in names_L1:
            os.chdir(i)
            for j in names_L2:
                os.chdir(j)
                sp.call('python ~/bin/CG_Crystal/crys_rdf_gen.py "equil.lammpstrj {}/static.lammpstrj" "spec0.end.data {}/gen_static.data" "all {}" -species {}'.format(current,current,args.params_num, args.species), shell=True)
                cell_vol=read_thermo("thermo.avg", "v_my_vol")
                ratio=cell_vol[2]/cell_vol[1]
                vol.write("{}/{}\t{}\t{}\t{}\n".format(i,j,cell_vol[0],cell_vol[1],cell_vol[2]))
                os.chdir("../")
            os.chdir("../")

    
    
    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])







