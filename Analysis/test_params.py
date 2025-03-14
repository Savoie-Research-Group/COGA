#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
home=str(Path.home())
from auto_genetic_martini import rdf_loss, gsmooth
from crys_rdf_gen import rdf_smoother
import subprocess as sp
import matplotlib.pyplot as plt
from matplotlib import cm

def main(argv):
    parser = argparse.ArgumentParser(description='''Uses the rdf loss function to calculate the loss for a specified set of parameters                                                                                                                                        
    Input: bead_params,gen,user,alpha,sr,gaus_std,rdf_short
                                                                                                                                                                                                      
    Output:                                                                                                                                                                                         
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('xyz_list', type=str, help='String containing the list of xyz files.')
    parser.add_argument('cif_file', type=str, help='String containing name of cif file to be edited.')
    parser.add_argument('map_list', type=str, help='String containing the list of map files. Map files are txt format, the first column contains martini bead types, the second column contains atom numbers in those beads (one indexed ,column separated.)')
    parser.add_argument('bead_params', type=str, help='String containing params to test: "1.0 2.0 3.0 ..." ')
    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='genetic_martini', help='Output name. Default: genetic martini')
    parser.add_argument('-map_mols', dest='map_mols', type=int, default=1, help='Number of molecules identified in mapping provided.')   
    parser.add_argument('--p_scale', dest='p_scale', default=False, action='store_const', const=True, help = 'When present, the script uses the pressure value as a scale for the rdf, valuing lower pressures over higher ones.')
    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below

    # Writes initial input files
    dump_var=open("dump.txt",'w')
    sp.call('python ~/bin/CG_Crystal/CG_xyz.py {} {} {} -map_mols {}'.format(args.xyz_list, args.cif_file, args.map_list, args.map_mols), shell=True, stdout=dump_var)
    # Write periodic files for general and static systems.
    sp.call('python ~/bin/CG_Crystal/write_periodic_CG.py CG.cif martini.data martini.in.settings CG.map -o gen_static -T 200.0 --NVT --static', shell=True, stdout=dump_var)
    current_dir=os.getcwd()
    os.chdir(current_dir+'/gen_static')
    # bead_params=dict(gen0=[])
    # gen0_dict={}
    # with open('gen_static.in.settings', 'r') as sett:
    #     for lines in sett:
    #         words=lines.split()
    #         if words==[]:
    #             pass
    #         elif words[0]=='pair_coeff':
    #             beads = words[-1][1:len(words[-1])].split('-')
    #             if beads[0]==beads[1]:
    #                 gen0_dict[beads[0]]= words[5]
    # bead_params['gen0']=gen0_dict
    # bead_types=list(bead_params['gen0'].keys())        

    os.chdir(current_dir+'/gen_static')

    sp.call('mpirun -np 20 /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501 -in gen_static.in.init >> gen_static.in.out & wait', shell=True)

    os.chdir('../')
    user="ddfortne"
    alpha=0.125
    sr=2.0
    gaus_std=1.0
    rdf_short=6.0
    bead_params=np.array([[x.split()] for x in args.bead_params.split(',')],dtype=float)
    print(bead_params)
    static_rdfs=[[] for x in range(len(bead_params))]
    minned_rdfs=[[] for x in range(len(bead_params))]
    static_rdffs=[[] for x in range(len(bead_params))]
    minned_rdffs=[[] for x in range(len(bead_params))]
    for x in range(len(bead_params)):
        loss, all_loss, short_loss, srdf, mrdf=rdf_loss(bead_params[x],"_test_min_{}".format(x),user,alpha,sr,gaus_std,rdf_short,final=False,p_scale=args.p_scale)
        print("minimization step: \n")
        print("loss: ", loss)
        print("all loss: ", all_loss)
        print("short loss: ", short_loss)
        print("long loss: ", all_loss-short_loss)
        print("\n\n")
        #loss, all_loss, short_loss, srdff, mrdff=rdf_loss(bead_params[x][0],"_test_full_{}".format(x),user,alpha,sr,gaus_std,rdf_short,final=True,p_scale=args.p_scale)
        # print("final step: \n")
        # print("loss: ", loss)
        # print("all loss: ", all_loss)
        # print("short loss: ", short_loss)
        # print("long loss: ", all_loss-short_loss)
        static_rdfs[x]=srdf
        minned_rdfs[x]=mrdf
        #static_rdffs[x]=srdff
        #minned_rdffs[x]=mrdff
    #print(static_rdfs)
    #print(minned_rdfs)
    #for z in range(len(bead_params)):
    print("Plotting final rdfs...")
    rdf_ref_plot(static_rdfs, minned_rdfs, "minned_test", sr, gaus_std)
    #rdf_ref_plot(static_rdffs[z], minned_rdffs[z], "final_test", sr, gaus_std)

def rdf_ref_plot(rdf_ref, rdf_comps, output, sr, std): #rdf_comps must have first col as x values, subsequent as y values
    comps_size=np.shape(rdf_comps[0])
    #print(rdf_comps[0][0])
    #print(comps_size)
    param_num=comps_size[1]-1
    Blues = cm.get_cmap('Blues', len(rdf_comps))
    colormap=Blues(np.linspace(0.5,0.95,len(rdf_comps)))
    for i in range(len(rdf_comps)): # number of parameter sets
        for j in range(param_num): # number of beads
            plt.figure("{}".format(j))
            if i==0:
                #ref_smooth=gsmooth(rdf_ref[i][:,[0,j+1]],sr,std) # Smoothing on
                #ref_smooth=rdf_smoother(rdf_ref[i][:,[0,j+1]],sr,std) # Smoothing on but different
                ref_smooth=rdf_ref[i][:,[0,j+1]]                              # Smoothing off (ACTUALLY ON BECUASE DATA IS ALREADY SMOOTHED)
                plt.plot(ref_smooth[:,0],ref_smooth[:,1],'g')
            #rdf_smooth=gsmooth(rdf_comps[i][:,[0,j+1]],sr,std)  # Smoothing on
            rdf_smooth=rdf_comps[i][:,[0,j+1]]                 # Smoothing off
            plt.plot(rdf_smooth[:,0], rdf_smooth[:,1], color=colormap[i])
            if i==len(rdf_comps)-1:
                legend_list=["ref"]
                for x in range(len(rdf_comps)):
                    legend_list.append(str(x))
                plt.legend(legend_list)
                #plt.legend(["ref", str(x) for x in range(len(rdf_comps))])
                plt.savefig("{}_{}.png".format(output, j))
   
    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
