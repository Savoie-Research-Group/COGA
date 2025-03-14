#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
import pickle
home=str(Path.home())
sys.path.append('{}/bin/CG_Crystal'.format(home))
from auto_genetic_martini import rdf_loss
from CG_xyz import extend_map

def main(argv):
    parser = argparse.ArgumentParser(description='''Enter description here                                                                                                                                           
    Input: 
                                                                                                                                                                                                      
    Output:                                                                                                                                                                                         
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('data_file', type=str, help='Name of pickle file to read.')

    # Optional Arguments
    parser.add_argument('-cm', dest='cm', type=int, default=1, help='moles per cell ')
        
    # Parse Arguments
    args = parser.parse_args()
    
    # Write Script Below
    # Read data and assign
    cell_moles=args.cm
    with open(args.data_file, 'rb') as f:
        data=pickle.load(f)
    args=data["args"]
    betas=data["betas"]
    seed_loss="nothing"
    p_scale_vals=args.p_scale_vals

    # Split up values for the pressure scaling function
    if args.p_scale:
        p_scale_vals=args.p_scale_vals.split()
        p_scale_vals=[float(i) for i in p_scale_vals]
        press_c=p_scale_vals[0]
        press_a=p_scale_vals[1]-press_c
        press_b=p_scale_vals[2]
        p_scale_vals=[press_a, press_b, press_c]

    if args.order:
        order=args.order.split('-')
    else:
        order=[]
    if order: # Change map file for final run, may be redundant if same dims are requested.
        dim_list=args.dimsf.split()
        cells=1
        for c in dim_list:
            cells=cells*int(c)
        all_moles=cells*cell_moles
        extend_map("CG.map", all_moles)

    lossf, all_lossf, short_lossf, static_rdf_allf, minned_rdf_allf, seed_lossf, data_dictf=rdf_loss(betas[-1], "_final2", args.user, args.alpha, args.sr, args.gaus_std, args.rdf_short,seed_loss,final=True,p_scale=args.p_scale, p_scale_vals=p_scale_vals, loss_func=args.loss_func, eps=args.epsilon, order=order, dims=args.dimsf, cell_sym=args.cell_sym, nps=64)
    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
