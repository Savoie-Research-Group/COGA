#!/bin/env python
# Author: Dylan Fortney
import os, sys, argparse, io
import numpy as np
import subprocess as sp
from pathlib import Path
from scipy.special import binom
from scipy import stats
import fileinput as FI
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
# home=str(Path.home())
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-1])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
# sys.path.append('{}/bin/'.format(home))
from crys_rdf_gen import rdf_smoother
from general_plot import plot_density
from GA_plot import plot_ac
from CG_xyz import extend_map
from monitor_jobs import *
import json
import pickle
import traceback
#from logger import Logger

def main(argv):
    parser = argparse.ArgumentParser(description='''Uses a genetic algorithm to modify martini sigma (distance) parameters by comparing to a ground truth structure. The error function tries to minimize
        difference in the rdfs, orientation, and thermodynamics to find the best parameters. Prepares for additional simulations of best parameters to be run.  
        
        Recommended: print this function's output to a file!!!! ex: $Function call$ > GA.log
         
                                                                                                                                       
    Input: xyz list, cif file, map list (Match the inputs to CG_xyz.py). Optionally, various specifications for the genetic algorithm.
                                                                                                                                                                                                      
    Output: Several folders containing the specified number of generations and species ran by the genetic algorithm, in addition to a "final" full simulation that tests and analyzes the best found parameters.                                                                                                                                                                                      
                                                                                                                                                                                                            
    Assumptions: Assumes a form of the difference between rdfs.
    ''')

    # Required Arguments
    parser.add_argument('xyz_file', type=str, help='String containing the list of xyz files.')
    parser.add_argument('cif_file', type=str, help='String containing name of cif file to be edited.')
    parser.add_argument('map_list', type=str, help='String containing the list of map files. Map files are txt format, the first column contains martini bead types, the second column contains atom numbers in those beads (one indexed ,column separated.)')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='genetic_martini', help='Output name. Default: genetic martini')
    parser.add_argument('-map_mols', dest='map_mols', type=int, default=1, help='Number of molecules identified in mapping provided. Only necessary if moles in xyz and cif differ (just make sure they are the same)')
    parser.add_argument('-user', dest='user', type=str, default='ddfortne', help='User account name to check jobs running.')
    parser.add_argument('-gens', dest='gens', type=int, default=50, help='Number of generations for algorithm.')
    parser.add_argument('-specs', dest='specs', type=int, default=50, help='Starting Population in algorithm.')
    parser.add_argument('-f_keep', dest='f_keep', type=float, default=0.3, help='Fraction of population to keep.')
    parser.add_argument('-f_mut', dest='f_mut', type=float, default=0.5, help='Fraction of mutations to introduce.')
    parser.add_argument('-mut_range', dest='mut_range', type=float, default=1.0, help='Range over which parameters can mutate.')
    parser.add_argument('-alpha', dest='alpha', type=float, default=0.125, help='Value of alpha constant in rdf loss used to weight distances. Sets "standard" distance at 1/alpha.')
    parser.add_argument('-xmin', dest='xmin', type=int, default=2, help='Minimum value your parameters can assume.')
    parser.add_argument('-xmax', dest='xmax', type=int, default=10, help='Maximum value your parameters can assume.')
    parser.add_argument('-ep_mut_range', dest='ep_mut_range', type=float, default=0.25, help='Range over which parameters can mutate.')
    parser.add_argument('-epmin', dest='epmin', type=float, default=0.0, help='Minimum value your parameters can assume.')
    parser.add_argument('-epmax', dest='epmax', type=float, default=2.0, help='Maximum value your parameters can assume.')
    parser.add_argument('-sr', dest='sr', type=float, default=9999999.0, help='Smooth Radius. Radius about which rdf will be introduced Gaussian error. If not specified, rdf will not be smoothed.')
    parser.add_argument('-gaus_std', dest='gaus_std', type=float, default=0.25, help='Standard deviation used in gaussian calculation.')
    parser.add_argument('-sample_num', dest='sample_num', type=int, default=5, help='Frequency at which generations are sampled. For example, 5 samples every 5th generation.')
    parser.add_argument('-rdf_short', dest='rdf_short', type=float, default=6.0, help='Defines the length (in angstroms) at which the rdf gets split to compare the "short" and "long" loss functions.')
    parser.add_argument('-p_scale_vals', dest='p_scale_vals', type=str, default='1 3 500', help='Parameters for p scaling function. Order: 1) value at P=0, 2) value as P approaches infinity, 3) P value which splits 1 and 2. Default: 1 2 100')
    parser.add_argument('-scale_switch', dest='scale_switch', type=float, default=0.5, help='Fractional value of generation at which script switches from p_scale to v_scale if both are selected. If time_switch selected, it is the fraction of generations not in the longer time scale.')
    parser.add_argument('-time_switch', dest='time_switch', type=float, default=0.8, help='Fractional value of generation at which script begins running longer time NPT simulations.')
    parser.add_argument('-loss_func', dest='loss_func', type=str, default='add', help='Define which loss function style to use. Options: "mult": multiply scaling terms. "add": Add scaling terms. Added terms are scaled by seed guess. "frac": rdf and thermo terms are added with fractional weights, normalized by seed value.')
    parser.add_argument('-dims', dest='dims', type=str, default='3 3 3', help='Dimensions of cell used in main body of simulation. Usually smaller to save time.')
    parser.add_argument('-dimsf', dest='dimsf', type=str, default='12 12 12', help='Dimensions of cell used in final (longer) simulation. Can be larger if scalar order parameter is calculated for better statistical information.')
    parser.add_argument('-accf', dest='accf', type=str, default='bsavoie', help='The name of the queue/account name you wish to submit the final job to.')
    parser.add_argument('-npsf', dest='npsf', type=int, default=128, help='Number of processors used in the final simulation of a larger cell. This may need to be high simply to accomodate the memory required to calculate the rdf.')
    parser.add_argument('-order', dest='order', type=str, default='', help='If provided, calculates and includes scalar order parameter in loss function and final simulation. Format "i-j", where i and j are the atom numbers to use as the directional vector.')
    parser.add_argument('-python', dest='python', type=str, default='python', help='Location/name of the python version to be used.')
    parser.add_argument('-mpirun', dest='mpirun', type=str, default='/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun', help='Location/name of the mpi to be used to run with your version of lammps. Bell: /apps/cent7/intel/impi/2017.1.132/bin64/mpirun')
    parser.add_argument('-lammps', dest='lammps', type=str, default='/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322', help='Location/name of the lammps version to be run. Note, some changes to submissions may be necessary. Bell: /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501')
    parser.add_argument('--narrow', dest='narrow', default=False, action='store_const', const=True, help = 'When present, the script narrows the smooth radius and mutation range. (default: False)')
    parser.add_argument('--p_scale', dest='p_scale', default=False, action='store_const', const=True, help = 'When present, the script uses the pressure value as a scale for the rdf, valuing lower pressures over higher ones.')
    parser.add_argument('--v_scale', dest='v_scale', default=False, action='store_const', const=True, help = 'When present, the script uses the volume value as a scale for the rdf, valuing volumes closer to the baseline value.')
    parser.add_argument('--epsilon', dest='epsilon', default=False, action='store_const', const=True, help = 'When present, the script varies epsilon values in addition to sigma values in the LJ potential.')
    parser.add_argument('--cell_sym', dest='cell_sym', default=False, action='store_const', const=True, help = 'When present, cell symmmetry operations are used in the gen_periodic script.')
    parser.add_argument('--no_dila', dest='dila', default=True, action='store_const', const=False, help = 'When present, script will NOT submit dilatometry simulations.')
    parser.add_argument('--no_final', dest='no_final', default=False, action='store_const', const=True, help = 'When present, script will NOT submit final simulations, and will instead create submission files for a final run.')
    parser.add_argument('--jm', dest='jm', default=False, action='store_const', const=True, help='When present, script will run the final simulation using the unchanged martini parameters.')
    parser.add_argument('--bfw', dest='bfw', default=False, action='store_const', const=True, help='When present, script will utilize the Bead Frequency Weight (bfw) in the loss function, meaning when calculating the loss each contribution will be weighted by the beads frequency in the molecule.')
    # Parse Arguments
    args = parser.parse_args()
    # global p_scale
    # p_scale=args.p_scale

    # Write Script Below
    t_start=perf_counter() # Start timer!
    print("This version of the script was last edited on December 5th, 2024.")
    p_scale_vals=args.p_scale_vals
    # Split up values for the pressure scaling function
    if args.p_scale:
        p_scale_vals=args.p_scale_vals.split()
        p_scale_vals=[float(i) for i in p_scale_vals]
        press_c=p_scale_vals[0]
        press_a=p_scale_vals[1]-press_c
        press_b=p_scale_vals[2]
        p_scale_vals=[press_a, press_b, press_c]
        # print("Scaling: ", p_scale_vals)
    if args.time_switch==0.0:
        normal_gens=args.gens
        time_gen=''
    elif args.time_switch>1.0:
        print("Error: Must provide time_switch value between 0.0 and 1.0")
        quit()
    else:
        time_gen=int(round(args.gens*args.time_switch)) # Calculate generation where new time scale is used.
        normal_gens=time_gen
        print("Short simulations being run through generation {}\n".format(normal_gens))
        print("Long simulations begin at generation {}\n".format(time_gen+1))
    # Define scale switching generation
    if args.p_scale and args.v_scale:
        switch_gen=int(round(normal_gens*args.scale_switch))
        print("Switching Scaling at generation {}".format(switch_gen))
    else:
        switch_gen=''
    if args.order:
        order=args.order.split('-')
    else:
        order=[]
    print("order ", order)
    # Check if valid loss function type.
    loss_funcs=["mult", "add", "frac"]
    if args.loss_func in loss_funcs:
        print("Loss Function: {}\n".format(args.loss_func))
    else: 
        print("ERROR: Invalid loss function type selected. Please select a valid loss function type:\n {}".format(loss_funcs))
        print("Exiting...")
        quit()

    # Writes initial input files
    dump_var=open("dump.txt",'w')
    sp.call('{} {}/Input_Operations/CG_xyz.py {} {} {} -map_mols {}'.format(args.python, script_directory, args.xyz_file, args.cif_file, args.map_list, args.map_mols), shell=True, stdout=dump_var)
    bead_type_ind=[]
    with open("CG.xyz", 'r') as xyz:
        for num, line in enumerate(xyz):
            if num == 0:
                base_atoms=int(line)
            elif line.split()==[]:
                pass
            else:
                bead_type_ind.append(line.split()[0])
        print("base: ", base_atoms)
        print("bead types: ", bead_type_ind)
    xyz.close()
    # Write periodic files for general and static systems.
    if args.cell_sym:
        cell_sym="--cell_sym"
    else:
        cell_sym=""
    sp.call('{} {}/Input_Operations/write_periodic_CG.py CG.cif martini.data martini.in.settings CG.map -o gen_static -dims "{}" -T 200.0 -lammps {} -python "{}" -mpirun {} --NVT --static {}'.format(args.python, script_directory, args.dims, args.lammps, args.python, args.mpirun, cell_sym), shell=True, stdout=dump_var)
    print('''*---------------------*
|                     |
|     Submitting      |
|      Initial        |
|        Job!         |
|                     |
*---------------------*''')
    # Get current working directory
    current_dir=os.getcwd()
    os.chdir(current_dir+'/gen_static')
    
    # Get number of bead parameters from settings files.
    bead_params=dict(gen0=[])
    gen0_dict={}
    with open('gen_static.in.settings', 'r') as sett:
        for lines in sett:
            words=lines.split()
            if words==[]:
                pass
            elif words[0]=='pair_coeff':
                beads = words[-1][1:len(words[-1])].split('-')
                if beads[0]==beads[1]:
                    gen0_dict[beads[0]]= words[5]
    bead_params['gen0']=gen0_dict
    bead_types=list(bead_params['gen0'].keys())

    if args.bfw:    # If requested, weight the loss function using the frequency with which each bead occurs in the molecule.
        bead_freq=[]
        for b in bead_types:
            bead_freq.append(bead_type_ind.count(b))
        print("bead_freq", bead_freq)
        bead_freq=np.array(bead_freq)
        bead_weight=bead_freq*len(bead_types)/len(bead_type_ind)
        
    else:
        bead_weight=[1 for i in range(len(bead_types))]
    print("bead_weight", bead_weight, np.sum(bead_weight))

    # Generate static files, which act as the "ground truth".
    os.chdir(current_dir+'/gen_static')
    print("mpirun: ", args.mpirun)
    sp.call('{} -np 20 {} -in gen_static.in.init >> gen_static.in.out & wait'.format(args.mpirun, args.lammps), shell=True)
    
    with open('gen_static.xyz', 'r') as xyz:
        for num, line in enumerate(xyz):
            if num == 0:
                cell_atoms=int(line)
            else:
                break
        print("cell atoms", cell_atoms)
    xyz.close()

    # Determine the initial rdf peak in each bead for the static run.
    rdf_maxes=np.zeros((len(bead_types), 2))
    for b in range(1,len(bead_types)+1):
        rdf_maxes[b-1,:]=get_rdf_max("equil.lammpstrj", "gen_static.end.data", b)
    print("rdf_maxes", rdf_maxes)
    best_dist=rdf_maxes[:,0]
    print("best dist: ", best_dist)

    os.chdir('../')
    
    
    dim_list=args.dims.split()
    cells=1
    for c in dim_list:
        cells=cells*int(c)
    all_moles=int(cell_atoms/base_atoms)
    cell_moles=all_moles/cells
    if order: # Make additional map file if requested.
        extend_map("CG.map", all_moles)
        
    # Set up the minimum and maximum of the mutation range, including the arrays if the values are selected to narrow. 
    mut_min=args.mut_range*-1.0
    mut_max=args.mut_range
    ep_mut_min=args.ep_mut_range*(-1.0)
    ep_mut_max=args.ep_mut_range
    if args.narrow==True:
        narrow=args.gens
        mut_min=np.linspace(mut_min,-0.5,num=args.gens)
        mut_max=np.linspace(mut_max,0.5, num=args.gens)
        print("mutation minima: ", mut_min)
        print("mutation maxima: ", mut_max)
    else:
        narrow=-1
    
    if args.epsilon:
        param_num=2*len(bead_types)
        vars_dict={"param_num":param_num, "bead_num":len(bead_types), "LJs":[0, 1], "start_ind":[0, len(bead_types)], "end_ind":[len(bead_types), param_num], "min":[args.xmin, args.epmin],"max":[args.xmax, args.epmax] , "mut_max":[mut_max, ep_mut_max], "mut_min":[mut_min, ep_mut_min], "alpha":args.alpha}
    else:
        param_num=len(bead_types)
        vars_dict={"param_num":param_num, "bead_num":len(bead_types), "LJs":[0], "start_ind":[0], "end_ind":[len(bead_types)], "min":[args.xmin],"max":[args.xmax] , "mut_max":[mut_max], "mut_min":[mut_min], "alpha":args.alpha}
    # Start Algorithm
    vars_dict["order"]=order
    vars_dict["dims"]=args.dims
    vars_dict["cell_sym"]=args.cell_sym
    folder=current_dir.split("/")[-1]
    vars_dict["folder"]=folder
    print(("*"*50).center(120))
    print("*****        Beginning Initial Generation...        *****".center(120))
    print(("*"*50).center(120))
    #model = GA(rdf_loss, args.user, args.alpha, sr=args.sr, gaus_std=args.gaus_std, rdf_short=args.rdf_short, narrow=narrow, loss_func=args.loss_func, n=args.specs,nx=len(bead_types),f_keep=args.f_keep, f_mut=args.f_mut, mut_min=mut_min, mut_max=mut_max, seed=np.random.randint(1,99999999), p_scale=args.p_scale, p_scale_vals=p_scale_vals, v_scale=args.v_scale, switch_gen=switch_gen, time_gen=time_gen)
    model = GA(rdf_loss, args.user, vars_dict, sr=args.sr, gaus_std=args.gaus_std, rdf_short=args.rdf_short, narrow=narrow, loss_func=args.loss_func, n=args.specs,nx=param_num,f_keep=args.f_keep, f_mut=args.f_mut, mut_min=mut_min, mut_max=mut_max, seed=np.random.randint(1,99999999), p_scale=args.p_scale, p_scale_vals=p_scale_vals, v_scale=args.v_scale, switch_gen=switch_gen, time_gen=time_gen, bfw=bead_weight, python=args.python, mpirun=args.mpirun, lammps=args.lammps)
    #dist_losses, dist_all_loss, dist_short_loss=model.initialize(best_dist, args.xmin, args.xmax)                              # Initialize model, (minvalue, maxvalue)
    dist_losses, dist_all_loss, dist_short_loss=model.initialize(best_dist)                              # Initialize model, (minvalue, maxvalue)

    # We'll keep the best params discovered up to each generation
    betas = [model.best_x] 
    losses = [model.best_L] 
    all_loss=[model.best_all]
    short_loss=[model.best_short]
    sp.call('mkdir gen_progress', shell=True)
#### Run the algorithm for set number of generations ####
    sampled_gens=[]
    legend_list=["experimental"]
    for i in range(1, args.gens+1):
        #print("\t\t\t\t\t\t\t\t*************Beginning Generation {}... *************\n".format(i))
        print(("*"*50).center(120))
        print("*****        Beginning Generation {}...        *****".format(i).center(120))
        print(("*"*50).center(120))
        model.selection()
        model.crossover()
        if args.narrow==True:
            ind=model.mutation(mut_min[i-1], mut_max[i-1])
        else:
            ind=model.mutation(mut_min, mut_max)
        if (i%args.sample_num)==0:
            sp.call('mkdir gen_progress/gen{}'.format(i), shell=True)
            sp.call('cp -r gen{}/spec{}/*.rdf gen_progress/gen{}'.format(i, ind, i), shell=True)
            sampled_gens.append("gen{}".format(i))
            legend_list.append("gen{}".format(i))
        betas += [model.best_x] 
        losses += [model.best_L] 
        all_loss += [model.best_all]
        short_loss += [model.best_short]
        print("Current best Overall: ", betas[-1])
####
# Curate data to be passed on to plotting function. Exporting the data is more efficient for tweaking graphs, as it negates the need to rerun the whole algorithm to fix plots.
    data_dict=model.data_dict
    save_data={"betas":betas, "sampled_gens":sampled_gens, "bead_types":bead_types, "args":args, "dist_losses":dist_losses, "current_dir":current_dir, "legend_list":legend_list, "best_dist":best_dist, "bfw":bead_weight}

    gen_range=range(0, args.gens+1)
    
    # Prep short and long losses. Short being loss below a specified value, long being loss above that value. 
    dist_long_loss=np.array([dist_all_loss-dist_short_loss for i in gen_range])
    dist_all_loss=np.array([dist_all_loss for i in gen_range])
    dist_short_loss=np.array([dist_short_loss for i in gen_range])

    save_data["dist_long_loss"]=dist_long_loss
    save_data["dist_all_loss"]=dist_all_loss
    save_data["dist_short_loss"]=dist_short_loss

    plot_folder="gen_plots_"+folder
    all_loss=np.array(all_loss)
    short_loss=np.array(short_loss)
    long_loss=all_loss-short_loss
    save_data.update({"losses":losses, "all_loss":all_loss, "short_loss":short_loss, "gen_range":gen_range})
    print("betas: ", model.beta)
    beta_sizes=[]
    for b in model.beta:
        # print(len(b))
        beta_sizes.append(len(b))
    beta_sizes_set=set(beta_sizes)
    if len(beta_sizes_set)!=1:
        print("beta_sizes: ", beta_sizes)
    all_betas=np.array(model.beta)
    print("betas, np: ", all_betas)
    gens_for_params=np.zeros((args.gens+1)*args.specs)
    seed_loss=model.seed_loss
    save_data.update({"all_betas":all_betas, "gens_for_params":gens_for_params, "switch_gen":switch_gen, "time_gen":time_gen, "seed_loss":seed_loss})
    save_data.update({"data_dict":data_dict, "vars_dict":vars_dict, "best_params":betas[-1], "plot_folder":plot_folder})

# Export Data
    with open('{}.p'.format(args.output), 'wb') as p:
        pickle.dump(save_data, p)
    sp.call("{} {}/Plotting/GA_plot.py {}.p".format(args.python, script_directory, args.output), shell=True)
    plot_submit(current_dir, "{}.p".format(args.output), script_directory=script_directory, python=args.python)
# Starting final simulation of best parameters.
    print(("*"*50).center(120))
    print("*****     Starting Simulation for best Parameters...     *****".center(120))
    print(("*"*50).center(120))
    print("Best parameters: ", betas[-1])
    print("Losses for each step: ", losses)
    if order: # Change map file for final run, may be redundant if same dims are requested.
        dim_list=args.dimsf.split()
        cells=1
        for c in dim_list:
            cells=cells*int(c)
        all_moles=cells*cell_moles
        save_data.update({"cell_moles":cell_moles})
        extend_map("CG.map", all_moles, output="CG_final")
    
    # If we have a larger final dimension, we probably need more cpus in order to handle the simulations/memory
    if args.dims == args.dimsf and args.no_final==False: # If the dimensions are the same, who cares? Submit in this session.
        lossf, all_lossf, short_lossf, static_rdf_allf, minned_rdf_allf, seed_lossf, data_dictf=rdf_loss(betas[-1], "_final", args.user, args.alpha, args.sr, args.gaus_std, args.rdf_short,seed_loss,folder,final=True,p_scale=args.p_scale, p_scale_vals=p_scale_vals, loss_func=args.loss_func, eps=args.epsilon, order=order, dims=args.dimsf, cell_sym=args.cell_sym, dila=args.dila, jm=args.jm, cell_moles=cell_moles, bfw=bead_weight, python=args.python, mpirun=args.mpirun, lammps=args.lammps)
        # Print some relevant information, clean up plots and unneeded logs.
        print("Final rdf loss: ", np.sum(data_dictf["raw_rdf"][0]))
        print("SLoss: ", losses[-1])
        os.chdir('{}/gen_final/spec0/'.format(current_dir))
        sp.call('mv *.pdf {}/{}'.format(current_dir, plot_folder),shell=True)
    else: # If the dimensions are different, we care. Make new submit.
        bs="["+', '.join([str(i) for i in betas[-1]])+"]"
        print(bs)
        print(seed_loss)
        sls="["
        for i in seed_loss:
            sls+="np.array(["
            for j in i:
                sls+=str(j)+","
            sls+="]),"
        sls+="]"
        print(sls)
        final_command='lossf, all_lossf, short_lossf, static_rdf_allf, minned_rdf_allf, seed_lossf, data_dictf=rdf_loss({}, "_final", "{}", {}, {}, {}, {}, {}, "{}", final=True, v_scale=True, p_scale={}, p_scale_vals={}, loss_func="{}", eps={}, order={}, dims="{}", cell_sym={}, nps={}, dila={}, jm={}, cell_moles={}, bfw={}, python="{}", mpirun="{}", lammps="{}")'.format(bs, args.user, args.alpha, args.sr, args.gaus_std, args.rdf_short,sls,folder,args.p_scale,p_scale_vals,args.loss_func,args.epsilon,order,args.dimsf,args.cell_sym, str(args.npsf), args.dila, args.jm, cell_moles, list(bead_weight), args.python, args.mpirun, args.lammps)
        final_submit(current_dir, final_command, args.accf, args.npsf, plot_folder, script_directory)
        if args.no_final:
            pass
        else:
            sp.call("sbatch final_GA.submit", shell=True)

    os.chdir(current_dir)
    sp.call('rm dump.txt', shell=True)
    t_stop=perf_counter() # Stop timer!
    print("Algorithm completed in {} hours.".format(str((t_stop-t_start)/(60*60))))



# rdf error function:
def rdf_loss(bead_params,gen,user,alpha,sr,gaus_std,rdf_short,seed_loss,folder,final=False, p_scale=False, p_scale_vals=[1, 2, 100],v_scale=False, t_scale=False, loss_func="mult", eps=False, order=[], dims="3 3 3", cell_sym=False, r0=0, nps=20, dila=False, jm=False, cell_moles=1, final_time=None, bfw=None, python="python", mpirun="/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun", lammps="/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322"):
    if final==True: # "if final==True:" statements indicate separation between final run and standard algorithm run. This is required because parameters supplied during and after algorithm have different sizes. 
        sample=1
        print("bead params:", bead_params)
        print("bead params:", type(bead_params))
        params_num=np.size(bead_params)
        o_plot=True # Plot order parameters only if this is the final run. 
        extmap_file="CG_final_extend.map"
    else:
        sample=np.size(bead_params, 0)
        params_num=np.size(bead_params, 1)
        o_plot=False
        extmap_file="CG_extend.map"
    if eps:
        params_num=int(params_num/2)
    if cell_sym==True:
        cell_sym="--cell_sym"
    else:
        cell_sym=""
    # r0 represents the first peak. If these peaks are provided for each bead, they will be usedto set the max loss at the initial peak.
    # If r0 is not provided, r0=0.0 will be used, so all rdf starts will be equivalent.
    if r0:
        pass
    else:
        r0=np.zeros((params_num))

    if bfw is not None:
        pass
    else:
        bfw=[1 for i in range(len(bead_type))]
    
    print("lammps version: ", lammps)
    print("mpirun version: ", mpirun)
    print("current directory: ", os.getcwd())
    loss=np.zeros((sample))
    all_loss=np.zeros((sample,params_num))
    short_loss=np.zeros((sample,params_num))
    sl_len=int(rdf_short*100) # Length for calculating the shortened rdf. equal to len in angstroms * 100 (number of values)
    dump_var=open("dump.txt",'w')
    data_dict={"thermo_scale":np.zeros((sample))}
    data_dict["thermo_delta"]=np.zeros((sample))
    data_dict["raw_rdf"]=np.zeros((sample,params_num))
    if loss_func=="frac" or loss_func=="add":
        data_dict["rdf_frac"]=np.zeros((sample))
        rdf_vals=np.zeros((sample, params_num))
        thermo_vals=np.zeros((sample, params_num))
        if order:
            order_vals=np.zeros((sample, params_num))
    # Make new folder, copy relevant files over.
    sp.call('mkdir gen{}'.format(gen), shell=True)
    sp.Popen(['cp', 'gen_static/equil.lammpstrj', 'gen{}/static.lammpstrj'.format(gen)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #sp.Popen(['cp', 'martini.data', 'gen{}/'.format(gen)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sp.call('cp martini.data gen{}/'.format(gen), shell=True)
    sp.Popen(['cp', 'martini.in.settings', 'gen{}/'.format(gen)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sp.Popen(['cp', 'CG.map', 'gen{}/'.format(gen)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sp.Popen(['cp', 'CG.cif', 'gen{}/'.format(gen)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sp.call('cp gen_static/*.rdf gen{}/'.format(gen), shell=True)
    
    if order:
        data_dict["order"]=np.zeros((sample))
    # Move into new folder, begin writing necessary files
    os.chdir('gen{}'.format(gen))
    for i in range(sample): # Iterate over every species in this generation. 
        O=-1.0 # Set this value to -1.0 to show the order hasn't been calculated yet. 
        # Split based on if this is a final run or not.
        if final==True:
            if final_time:
                runtime=final_time
                atime=final_time
                print("Using provided run time for final run.")
            else:
                runtime=1000000
                atime=100000 # Time for annealing step
            if v_scale:
                print("Final NPT simulation...")
                sp.call('{} {}/Input_Operations/write_periodic_CG.py CG.cif martini.data martini.in.settings CG.map -o spec{} -T 200.0 -P "1atm" -t {} -t_A {} -dims "{}" -lammps {} -python "{}" -mpirun {} {}'.format(python, script_directory, i, runtime, atime, dims, lammps, python, mpirun, cell_sym), shell=True, stdout=dump_var)
            else:
                print("Final NVT simulation...")
                sp.call('{} {}/Input_Operations/write_periodic_CG.py CG.cif martini.data martini.in.settings CG.map -o spec{} -T 200.0 -t {} -t_A {} -dims "{}" -lammps {} -python "{}" -mpirun {} --NVT {}'.format(python, script_directory, i, runtime, atime, dims, lammps, python, mpirun, cell_sym), shell=True, stdout=dump_var)
            runtime=atime+10000
            data_file="spec{}.end.data".format(i)
            traj_file="equil.lammpstrj"
            o_traj=traj_file
        else:
            #print('\t\t\t\t\t\t\t\t~~~~~Starting generation {} species {}...~~~~~'.format(gen, i))
            print(("~"*50).center(120))
            print('~~~~~   Starting generation {} species {}...   ~~~~~'.format(gen,i).center(120))
            print(("~"*50).center(120))
            print("Current directory: ", os.getcwd())
            if v_scale:
                print("NPT simulation...")
                runtime=10000
                # Run NPT simulation because minimization maintains volume by default.
                sp.call('python {}/Input_Operations/write_periodic_CG.py CG.cif martini.data martini.in.settings CG.map -o spec{} -T 200.0 -P "1atm" -nodes 1 -ppn 20 -t {} -t_A 0 -t_ext 0 -dims "{}" -lammps {} -python "{}" -mpirun {} {}'.format(script_directory, i, runtime, dims, lammps, python, mpirun, cell_sym), shell=True, stdout=dump_var)
                data_file="spec{}.end.data".format(i)
                traj_file="equil.lammpstrj"
                o_traj=traj_file
            elif p_scale:
                if t_scale: # Longer time NVT simulations similar to final run.
                    print("NVT simulation...")
                    runtime=30000
                    sp.call('python {}/Input_Operations/write_periodic_CG.py CG.cif martini.data martini.in.settings CG.map -o spec{} -T 200.0 -nodes 1 -ppn 20 --NVT -t {} -t_A 0 -t_ext 0 -dims "{}" -lammps {} -python "{}" -mpirun {} {}'.format(script_directory, i, runtime, dims, lammps, python, mpirun, cell_sym), shell=True, stdout=dump_var)
                    data_file="spec{}.end.data".format(i)
                    traj_file="equil.lammpstrj"
                    
                    o_traj=traj_file
                else:
                    print("NVE simulation...")
                    runtime=10000
                    sp.call('python {}/Input_Operations/write_periodic_CG.py CG.cif martini.data martini.in.settings CG.map -o spec{} -T 200.0 -nodes 1 -ppn 20 --NVT --minrun -dims "{}" -lammps {} -python "{}" -mpirun {} {}'.format(script_directory, i, dims, lammps, python, mpirun, cell_sym), shell=True, stdout=dump_var)
                    data_file="minimize.end.data"
                    traj_file="minimize.lammpstrj"             
                    o_traj="combined.lammpstrj"

        frame_count=(runtime-10000)/1000+1    
        sp.Popen(['cp', 'static.lammpstrj', 'spec{}/'.format(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sp.call('cp *.rdf spec{}/'.format(i), shell=True)
        #sp.call('cp static.lammpstrj /spec{}'.format(i), shell=True)
        os.chdir('spec{}'.format(i))
        ### EDIT SETTINGS FILE
        # print("line 375 GA", os.getcwd())
        if jm==False: # If jm (just martini) is true, skip the file editing to use the raw martini parameters.
            with FI.FileInput('spec{}.in.settings'.format(i), inplace=True, backup='.bak') as sett:
                for line in sett:
                    if line.split()==[]:
                        print(line, end='')
                    elif line.split()[0]=='pair_coeff':
                        words=line.split()
                        if final==True:
                            this_sigma=bead_params[int(words[1])-1]
                            next_sigma=bead_params[int(words[2])-1]
                        else:
                            this_sigma=bead_params[i, int(words[1])-1]
                            next_sigma=bead_params[i, int(words[2])-1]
                        if eps:
                            if final==True:
                                this_eps=bead_params[int(words[1])-1+int(len(bead_params)/2)]
                                next_eps=bead_params[int(words[2])-1+int(len(bead_params)/2)]
                            else:
                                #print("dim",len(bead_params[i])/2)
                                this_eps=bead_params[i, int(words[1])-1+int(len(bead_params[i])/2)]
                                next_eps=bead_params[i, int(words[2])-1+int(len(bead_params[i])/2)]
                            combo_eps=(this_eps*next_eps)**0.5           # Geometric mean of epsilon values
                        else:
                            this_eps=words[4]
                            combo_eps=words[4]
                        if words[1]==words[2]:
                            print("pair_coeff   {}  {}  {}  {}  {}  {}".format(words[1], words[2], words[3], this_eps, this_sigma, words[6]))
                        else:
                            combo_sigma=0.5*(this_sigma+next_sigma)-0.01
                            print("pair_coeff   {}  {}  {}  {}  {}  {}".format(words[1], words[2], words[3], combo_eps, combo_sigma, words[6]))
                    else:
                        print(line, end='')
                sett.close()
            sp.call('rm *.bak', shell=True) # Remove backup files.
        else:
            print("Using default martini parameters for selected bead types...")
        # Create Dilatometry experiments and Submit if requested.
        if final==True:        
            print("Prepping Dilatometry and Melting Simulations...")
            submit_dilatometry("spec{}".format(i), folder, dims, submit=dila, cell_moles=cell_moles, order="0-{}-{}".format(order[0], order[1]), python=python, mpirun=mpirun, lammps=lammps, script_directory=script_directory)
            submit_melt("spec{}".format(i), folder, dims, submit=dila, cell_moles=cell_moles, order="0-{}-{}".format(order[0], order[1]), state="crys", python=python, mpirun=mpirun, lammps=lammps, script_directory=script_directory)
            submit_melt("spec{}".format(i), folder, dims, submit=dila, cell_moles=cell_moles, order="0-{}-{}".format(order[0], order[1]), state="LC", python=python, mpirun=mpirun, lammps=lammps, script_directory=script_directory)
        sp.call('{} -np {} {} -in spec{}.in.init >> spec{}.in.out & wait'.format(mpirun, nps, lammps, i,i), shell=True) # RUN MD!!!
        if glob.glob('*.end.data'): # Check if the simulation ran.
            pass
        else: 
            print("Can't find .end.data file... Simulation likely failed.")
        #loss=0.0

        # If requested, use the pressure value as a scaling factor for the loss function.
        if p_scale:
            if t_scale:
                p_ave, p_init, p_final=read_thermo("thermo.avg", "v_my_P", data_start=10000)
                p_test=p_ave # Try average pressure for this measure
            else:
                p_final, p_init=get_inout("spec{}.in.out".format(i), "Press")
                p_test=p_final
                #sp.call("python ~/bin/CG_Crystal/string_trajectories.py 'static.lammpstrj minimize.lammpstrj'", shell=True)
                if not final:
                    with open("combined.lammpstrj", 'w') as c, open('static.lammpstrj', 'r') as s, open("minimize.lammpstrj", 'r') as m:
                        c.write(s.read())
                        c.write(m.read())
                    c.close()
                    s.close()
                    m.close()
            print("P_test:", p_test)
            Press=abs(1.0-float(p_test)/1.0) # Added 1.0-... Want delta P relative to reference system. Probably doesn't matter a lot since 1.0 is so small relative to the magnitudes of pressures.
            # Assign appropriate values to pressure scaling equation.
            P=p_scale_vals[0]*Press/(Press+p_scale_vals[1])+p_scale_vals[2] # New Pressure scaling method, of form ax/(x+b)+c, where the max value of P is a+c, the value at Press=0 is c, and the value at which P splits the max and zero value (1/2*a+c)=P(b)
            print("Pressure: ", Press)
            data_dict["thermo_scale"][i]=P # assign value to appropriate position in thermo list.
            data_dict["thermo_delta"][i]=Press
            V=1.0
            PorV=2
        elif v_scale:
            #v_ave, v_init=get_inout("spec{}.in.out".format(i), "Volume")
            v_ave, v_init, v_final=read_thermo("thermo.avg", "v_my_vol")
            #V_ave=float(v_ave)
            V_final=float(v_final)
            V_init=float(v_init)
            #V=abs(1-V_ave/V_init)
            #V=abs(1-V_final/V_init)+1
            Vol=abs(1-V_final/V_init)
            V=2*Vol/(Vol+1.25)+1 # Similar to the pressure scaling, this will bound the volume between 1 and 3.
            print("Volume: ", V)
            data_dict["thermo_scale"][i]=V # assign value to appropriate position in thermo list.
            data_dict["thermo_delta"][i]=Vol
            P=1.0
            PorV=3
        else:
            P=1.0
            V=1.0
        # Try to take the rdf. if an error occurs in the process, it will trip up some commands and set a high loss value.
        # We want this because we don't want one failed simulation to interrupt the whole algorithm. Especially because this means faulty parameters will be avoided. 
        try:
            # Split between final and nonfinal simulations once more.
            if final==True:
                if nps>20: # If we are using more than 20 processors, we probably are working with a larger system. In which case, use the end of the rdf to save time. 
                    print("Using shorter trajectory file for rdfs...")
                    sp.call("source {}/Input_Operations/copybottom.sh; copybottom {} end.lammpstrj".format(script_directory, traj_file), shell=True)
                    traj_file="end.lammpstrj"
                print("Trajectory file used for rdf calculations: {}".format(traj_file))
                sp.call('{} {}/Analysis/crys_rdf_gen.py "{} static.lammpstrj" "spec{}.end.data spec{}.data" "all {}" -reuse_rdf "0 static" -species "Best parameters"'.format(python, script_directory, traj_file, i,i,params_num), shell=True, stdout=dump_var)
                if sr!=9999999.0: # Do smoothed for comparison. ## Added crys_rdf_0 to reuse for efficiency. This should prevent the algorithm from calculating the rdf twice during the final parameter set, saving a lot of time. Old value was "0 static"
                    sp.call('{} {}/Analysis/crys_rdf_gen.py "{} static.lammpstrj" "spec{}.end.data spec{}.data" "all {}" -reuse_rdf "crys_rdf_0 static" -species "Best parameters" -o smooth -sr {} -gaus_std {}'.format(python, script_directory, traj_file, i,i,params_num, sr, gaus_std), shell=True, stdout=dump_var)
                ave_P=read_thermo("thermo.avg", "v_my_P", data_start=10000)
                traj_file="equil.lammpstrj"
            else:
                sp.call('{} {}/Analysis/crys_rdf_gen.py "{} static.lammpstrj" "{} spec{}.data" "all {}" -reuse_rdf "0 static" -species "gen {} sample {}" -sr {} -gaus_std {}'.format(python, script_directory, traj_file, data_file, i,params_num,gen,i,sr,gaus_std), shell=True)#, stdout=dump_var)
            ## SECTION REMOVED FROM IF ELSE BECUASE IT IS SHARED BY BOTH PATHS.  
            # Begin rdf analysis           
            for j in range(params_num):
                minned_rdf=np.loadtxt("crys_rdf_0_{}_{}.rdf".format(j+1, j+1), skiprows=1)
                static_rdf=np.loadtxt("static_{}_{}.rdf".format(j+1, j+1), skiprows=1)
                min_len=len(minned_rdf)
                static_len=len(static_rdf)
                #print("min: ",min_len)
                #print("static: ", static_len)
                if min_len!=static_len:
                    len_min=min(min_len, static_len)
                    minned_rdf=minned_rdf[0:len_min,:]
                    static_rdf=static_rdf[0:len_min,:]
                if sr!=9999999.0:
                    # Smooth rdfs if so indicated.
                    minned_rdf=rdf_smoother(minned_rdf,sr,gaus_std)
                    static_rdf=rdf_smoother(static_rdf,sr,gaus_std)
                if j==0:
                    # For the first bead, set up the arrays that will hold rdfs
                    minned_rdf_all=np.zeros((len(minned_rdf),params_num+1)) # Create matrix of all minned rdfs, first column is the distances, remaining columns are values.
                    minned_rdf_all[:,0]=minned_rdf[:,0]
                    static_rdf_all=np.zeros((len(static_rdf),params_num+1)) # Create matrix of all minned rdfs, first column is the distances, remaining columns are values.
                    static_rdf_all[:,0]=static_rdf[:,0]
                    # AL_scale=np.zeros((params_num, 1))
                    # SL_scale=np.zeros((params_num, 1))
                    AL_scale=np.zeros(params_num)
                    SL_scale=np.zeros(params_num)

                ## Scale rdfs by maximum value
                try:
                    minned_rdf[:,1]=minned_rdf[:,1]/minned_rdf[:,1].max()
                except:
                    print("error in normalizing rdf!")
                    print(minned_rdf[:,1])
                    print(minned_rdf[:,1].max())
                static_rdf[:,1]=static_rdf[:,1]/static_rdf[:,1].max()
                minned_rdf_all[:,j+1]=minned_rdf[:,1]
                static_rdf_all[:,j+1]=static_rdf[:,1]
                ##
                if order: # Calculate order parameter (2nd legendre polynomial of autocorrelation.)
                    if O==-1.0: # Check if the order parameter has been calculated yet. This value should only occur if not calculated. Otherwise, we needlessly recalculate the same value for each bead
                        print("vautocorr call: {} {}/Analysis/vautocorr.py -map ../../{} -traj {} --modes '0-{}-{}' -legendres 2 -f_end {} -o legendre -log 20 -log_frames 100 --overwrite --no_xc >> vautocorr.txt".format(python, script_directory, extmap_file, o_traj, order[0], order[1], frame_count))
                        sp.call("{} {}/Analysis/vautocorr.py -map ../../{} -traj {} --modes '0-{}-{}' -legendres 2 -f_end {} -o legendre -log 20 -log_frames 100 --overwrite --no_xc >> vautocorr.txt".format(python, script_directory, extmap_file, o_traj, order[0], order[1], frame_count), shell=True)
                        frames, ac_vals, stdevs=plot_ac("legendre/vectors/0_{}_{}/ac-P2.txt".format(order[0], order[1]), plot=o_plot)
                        data_dict["order"][i]=ac_vals[-1]
                        O=(1-ac_vals[-1])
                    else:
                        pass
                    loss_frac=[0.6, 0.2, 0.2] # Fraction values for frac loss function. [rdf, thermo]
                else:
                    if loss_func=="add" or loss_func=="frac": # For these loss function types setting O=0 makes the order term disappear.
                        O=0
                    else: # For multiplicative loss functions setting O=1 has no effect on the rest of the loss function.
                        O=1
                    loss_frac=[0.7, 0.3, 0.0] # Fraction values for frac loss function. [rdf, thermo]
                # Evaluate loss function and pressure scaling (if requested) We want the loss for each bead (all loss), the loss for each bead before a threshold value (short loss) and the sum total of all bead loss functions (loss)
                # Loss function is evaluated on a bead by bead basis. Then summed together to into the loss variable and stored across each species.
                AL=sum(np.exp(-1.0*alpha*(minned_rdf[:,0]-r0[j]))*(minned_rdf[:,1]-static_rdf[:,1])**2.0)
                data_dict["raw_rdf"][i,j]=AL
                SL=sum(np.exp(-1.0*alpha*(minned_rdf[0:sl_len,0]-r0[j]))*(minned_rdf[0:sl_len,1]-static_rdf[0:sl_len,1])**2.0)
                print("AL, SL", AL, SL)
                # loss[i]=loss[i]+bfw[j]*all_loss[i,j]
                # print("gen {}, seed_loss: {} \n {}".format(gen, type(seed_loss), seed_loss))
                if loss_func=="mult": # Use loss function that multiplies scaling factors
                    all_loss[i,j]=AL*P*V*O
                    short_loss[i,j]=SL*P*V*O
                    loss[i]=loss[i]+bfw[j]*all_loss[i,j]
                elif loss_func=="add" and type(seed_loss) == str : # seed_loss=="find": # If using addition for the loss function, we will need to find what our scalar will be before we can fully evaluate loss function.
                    #print("Finding seed loss...\n")
                    AL_scale[j]=AL
                    SL_scale[j]=SL
                    all_loss[i,j]=AL+0.25*(P*V+O)*AL_scale[j]
                    short_loss[i,j]=SL+0.25*(P*V+O)*SL_scale[j] # Use the initial seed test as a scalar for adding the pressure and volume. 0.25 is currently an immutable HYPERPARAMETER that may need tuning!
                    loss[i]=loss[i]+bfw[j]*all_loss[i,j]
                    rdf_vals[i,j]=bfw[j]*AL
                    thermo_vals[i,j]=bfw[j]*0.25*P*V*AL_scale[j]
                    if order:
                        order_vals[i,j]=bfw[j]*0.25*O*AL_scale[j]
                elif loss_func=="add" and type(seed_loss)!= str:
                    print("Using seed loss: \n", seed_loss)
                    print("P, V, O", P, V, O)
                    print("seed_loss: ", seed_loss[0][j])
                    print("all_loss: ", all_loss)
                    all_loss[i,j]=AL+0.25*(P*V+O)*seed_loss[0][j]
                    short_loss[i,j]=SL+0.25*(P*V+O)*seed_loss[1][j] # Use the initial seed test as a scalar for adding the pressure and volume. 0.25 is currently an immutable HYPERPARAMETER that may need tuning!
                    loss[i]=loss[i]+bfw[j]*all_loss[i,j]
                    rdf_vals[i,j]=bfw[j]*AL
                    thermo_vals[i,j]=bfw[j]*0.25*P*V*seed_loss[0][j]
                    print("bfw: ", bfw)
                    print("seed_loss: ", seed_loss)
                    print("seed_loss ind: ", seed_loss[0][j])
                    if order:
                        order_vals[i,j]=bfw[j]*0.25*O*seed_loss[0][j]
                elif loss_func=="frac" and type(seed_loss) == str:
                    AL_scale[j]=AL
                    if SL==0:
                        SL_scale[j]=1
                    else:
                        SL_scale[j]=SL
                    all_loss[i,j]=loss_frac[0]*AL/AL+loss_frac[1]*P*V/(P*V)+loss_frac[2]*O
                    short_loss[i,j]=loss_frac[0]*SL_scale[j]/SL_scale[j]+loss_frac[1]*P*V/(P*V)+loss_frac[2]*O # Use the initial seed test as a scalar for adding the pressure and volume. 0.8,0.2 are currently immutable HYPERPARAMETERS that may need tuning!
                    loss[i]=loss[i]+bfw[j]*all_loss[i,j]
                    rdf_vals[i,j]=bfw[j]*loss_frac[0]*AL/AL
                    thermo_vals[i,j]=bfw[j]*loss_frac[1]*P*V/(P*V)
                    if order:
                        order_vals[i,j]=bfw[j]*loss_frac[2]*O
                elif loss_func=="frac" and type(seed_loss)!= str :
                    all_loss[i,j]=loss_frac[0]*AL/seed_loss[0][j]+loss_frac[1]*P*V/(seed_loss[PorV])+loss_frac[2]*O
                    short_loss[i,j]=loss_frac[0]*SL/seed_loss[0][j]+loss_frac[1]*P*V/(seed_loss[PorV])+loss_frac[2]*O# Use the initial seed test as a scalar for adding the pressure and volume. 0.8,0.2 are currently immutable HYPERPARAMETERS that may need tuning!
                    loss[i]=loss[i]+bfw[j]*all_loss[i,j]
                    rdf_vals[i,j]=bfw[j]*loss_frac[0]*AL/seed_loss[0][j]
                    thermo_vals[i,j]=bfw[j]*loss_frac[1]*P*V/(seed_loss[PorV])
                    if order:
                        order_vals[i,j]=bfw[j]*loss_frac[2]*O
                    
                #all_loss[i]=loss
            # else:
            #     sp.call('python ~/bin/CG_Crystal/crys_rdf_gen.py "minimize.lammpstrj static.lammpstrj" "minimize.end.data spec{}.data" "all {}" -species "gen {} sample {}" -sr {} -gaus_std {}'.format(i,params_num,gen,i,sr,gaus_std), shell=True)#, stdout=dump_var)


        except Exception as e: # This exception is usually triggered by failed simulations.
            print("Error calculating rdf... Simulation likely failed. Error message: \n")
            print(e)
            traceback.print_exc()
            print("Setting artificially high loss value.")
            for j in range(params_num): # put a value in each entry of the parameter's individual loss values.
                all_loss[i,j]=999999.9999
                short_loss[i,j]=999999.9999
            loss[i]=999999.9999
        os.chdir('../')

        if loss_func=="add" and seed_loss=="find":
            seed_loss=[AL_scale, SL_scale]
        elif loss_func=="frac" and seed_loss=="find":
            seed_loss=[AL_scale, SL_scale, P, V]
        elif loss_func=="mult":
            seed_loss=0
            
    os.chdir('../')
    
    if final==True: # Print some information if its the final run.
        print("Final loss: ", loss)
        print("Average Pressure: ", ave_P)
    elif type(gen)==str or gen==1:
        pass
    else:   # remove files that aren't the current generation, the first generation, or the final generation
        print("loss: ", loss)
        sp.call('rm -r gen{}'.format(gen-1), shell=True) # Remove previous files.
    
    if loss_func=="frac" or loss_func=="add":
        data_dict["rdf_vals"]=rdf_vals
        data_dict["rdf_frac"]=np.sum(rdf_vals, axis=1)/loss
        print("rdf_frac", data_dict["rdf_frac"])
        data_dict["thermo_vals"]=thermo_vals
        data_dict["thermo_frac"]=np.sum(thermo_vals, axis=1)/loss
        print("thermo_frac", data_dict["thermo_frac"])
        if order:
            data_dict["order_vals"]=order_vals
            data_dict["order_frac"]=np.sum(order_vals, axis=1)/loss
            print("order_frac", data_dict["order_frac"])
            print("sum: ", data_dict["thermo_frac"]+data_dict["rdf_frac"]+data_dict["order_frac"])
        else:
            print("sum: ", data_dict["thermo_frac"]+data_dict["rdf_frac"])
    # Loss: Total loss for all beads, all_loss: loss for each individual beads (2d array), short_loss: loss for each individual bead to a certain threshold distance (2d array)
    return loss, all_loss, short_loss, static_rdf_all, minned_rdf_all, seed_loss, data_dict


# Genetic Algorithm Class
class GA:
    def __init__(self, L, user, vars_dict, sr=9999999.0, gaus_std=1.0, rdf_short=6.0, narrow=-1, loss_func="mult", n=100, nx=2, f_mut=0.1, f_keep=0.5, mut_min=-1.0, mut_max=1.0,gmax=100,seed=10104041,p_scale=False, p_scale_vals=[1, 2, 100],v_scale=False, switch_gen='', time_gen='', bfw=None, python="python", mpirun="/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun", lammps="/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322"):
    #def __init__(self, L, user, alpha, sr=9999999.0, gaus_std=1.0, rdf_short=6.0, narrow=-1, loss_func="mult", n=100, nx=2, f_mut=0.1, f_keep=0.5, mut_min=-1.0, mut_max=1.0,gmax=100,seed=10104041,p_scale=False, p_scale_vals=[1, 2, 100],v_scale=False, switch_gen='', time_gen=''):
        
        # Fix the seed for reproducibility
        np.random.seed(seed)

        # Initialize attributes
        self.L = L # loss function
        self.n = n # population size
        self.nx = nx # number of variables in the model
        self.f_mut = f_mut # frequency (0,1) of introducing mutations
        self.f_keep = f_keep # fraction (0,1) of population to keep in each generation
        # 
        self.mut_min = vars_dict["mut_min"] # mutations are sampled uniformly from [mut_min,mut_max]
        self.mut_max = vars_dict["mut_max"] # mutations are sampled uniformly from [mut_min,mut_max]
        self.g = 1 # keeps track of how many generations have been run
        self.N_keep = int(self.f_keep*self.n)
        self.user=user
        self.alpha=vars_dict["alpha"]
        self.rdf_short=rdf_short
        if sr==9999999.0 or narrow==-1: # Keep sr if it is not being used or the narrow functionality is not desired.
            self.sr=sr
        else: # Narrow sr as generations continue. 
            self.sr=np.linspace(sr,0.2,num=narrow)
        self.gaus_std=gaus_std
        self.narrow=narrow
        self.p_scale=p_scale
        self.p_scale_vals=p_scale_vals
        self.v_scale=v_scale
        self.switch_gen=switch_gen
        self.time_gen=time_gen
        self.loss_func=loss_func
        self.data_dict={}
        self.xmin=vars_dict["min"]
        self.xmax=vars_dict["max"]
        self.vars_dict=vars_dict
        self.dims=vars_dict["dims"]
        self.order=vars_dict["order"]
        self.cell_sym=vars_dict["cell_sym"]
        self.folder=vars_dict["folder"]
        self.bfw=bfw
        self.python=python
        self.mpirun=mpirun
        self.lammps=lammps
        if len(vars_dict["LJs"])==1:
            self.eps=False
        else:
            self.eps=True
        # Check that a sufficiently large threshold is kept at each generation for replenishment. # Edit, make this a while loop to change f_keep until it is valid.
        keep_check=0
        while binom(self.N_keep,2) + self.N_keep < self.n:
            self.f_keep+=0.05
            self.N_keep = int(self.f_keep*self.n)
            keep_check+=1
            #raise ValueError("The supplied combination of f_keep ({}) and n ({}) parameters will lead to a shrinking population.".format(f_keep,n))
        if keep_check!=0:
            print("NOTE: The supplied combination of f_keep ({}) and n ({}) parameters will lead to a shrinking population.\n f_keep value of ({}) is being used instead...".format(f_keep,n, self.f_keep))
            
            
    def initialize(self,best_dist,xmin=None,xmax=None):

        # Initialize pool with uniformly sampled random numbers
        # self.xmin=xmin
        # self.xmax=xmax
        self.beta=[np.zeros((self.n, self.nx))]
        #print(self.beta)
        for i in self.vars_dict["LJs"]:
            if self.xmin and self.xmax:
                xmin=self.xmin[i]
                xmax=self.xmax[i]
                #print(self.beta[0][:,self.vars_dict["start_ind"][i]:self.vars_dict["end_ind"][i]])
                self.beta[0][:,self.vars_dict["start_ind"][i]:self.vars_dict["end_ind"][i]] = (np.random.random(size=(self.n,self.vars_dict["bead_num"]))-0.5) * np.array(xmax-xmin) + np.array((xmax+xmin)/2.0)
            else:
                self.beta[0][:,self.vars_dict["start_ind"][i]:self.vars_dict["end_ind"][i]] = np.random.random(size=(self.n,self.vars_dict["bead_num"]))-0.5
        #print(self.beta[0][0])
        self.beta[0][0][self.vars_dict["start_ind"][0]:self.vars_dict["end_ind"][0]]=best_dist
        self.best_dist=best_dist
        #print("betas: ", self.beta)
        # Calculate losses for initial pool
        if self.narrow==-1:
            sr_init=self.sr
        else:
            sr_init=round(self.sr[self.g-1],2)
        if self.p_scale and self.v_scale:
            p=True
            v=False
        elif self.p_scale:
            p=True
            v=False
        elif self.v_scale:
            p=False
            v=True
        else:
            p=False
            v=False
        # Loss Function Call
        self.losses, self.all_loss, self.short_loss, static_rdf, minned_rdf, seed_loss, data_dict= self.L(self.beta[-1],"_init",self.user,self.alpha,sr_init,self.gaus_std,self.rdf_short,"find", self.folder,p_scale=p,p_scale_vals=self.p_scale_vals,v_scale=v,loss_func=self.loss_func,eps=self.eps, order=self.order, dims=self.dims, cell_sym=self.cell_sym, bfw=self.bfw, python=self.python, lammps=self.lammps, mpirun=self.mpirun)
        self.data_dict["0"]=data_dict
        self.losses=[self.losses]
        self.all_loss=[self.all_loss]
        self.short_loss=[self.short_loss]
        
        dist_losses=self.losses[-1][0]
        dist_all_loss=self.all_loss[-1][0]
        dist_short_loss=self.short_loss[-1][0]

        # Keep found seed loss for later use.
        self.seed_loss=seed_loss
        
        # Sort by fitness
        inds = np.argsort(self.losses[-1])
        self.beta[-1] = self.beta[-1][inds]
        self.losses[-1] = self.losses[-1][inds]
        self.all_loss[-1]=self.all_loss[-1][inds]
        self.short_loss[-1]=self.short_loss[-1][inds]

        self.best_x = self.beta[-1][0]
        self.best_L = self.losses[-1][0]
        self.best_all=self.all_loss[-1][0]
        self.best_short=self.short_loss[-1][0]
        return dist_losses, dist_all_loss, dist_short_loss # Return losses of the seed values
                             
    def selection(self):
        if self.best_x in self.beta[-1][:self.N_keep]:
            self.beta += [self.beta[-1][:self.N_keep]]
        else:
            self.beta += [self.best_x, self.beta[-1][:self.N_keep]]
            print("Adding current best back into population")
                             
    def crossover(self):
        
        # Initialize list to hold children and shuffle the parent list
        self.children = [] 
        np.random.shuffle(self.beta[-1]) 
        
        # Cross-over is performed on j,j+i pairs. This means, j,j+1 pairs, then j,j+2 pairs, and so-on. We don't use itertools because we need to ensure diversity
        N_c = 0                
        for i in range(1,self.N_keep):
            for j in range(self.N_keep):

                # break after enough children have been generated to replenish the pool
                if N_c == self.n-self.N_keep:
                    break

                # tuple of parent indices         
                p = (j,j+i)

                # randomly choose which parent passes on each gene (i.e., parameter)         
                which = np.random.randint(2,size=self.nx)                

                # generate the offspring and update increment. We use try because the loop structure sometimes runs over the end of the list.
                try:   
                    self.children += [[ self.beta[-1][p[_],count] for count,_ in enumerate(which) ]]
                    N_c += 1
                except Exception as e:
                    print("exception!! Uh oh!")
                    print(e)
                    pass
                
        # Convert list of children to an array (expected by mutation method)
        self.children = np.array(self.children)
        #print(self.children)                     
    def mutation(self,mut_min,mut_max):
        self.mut_min[0]=mut_min
        self.mut_max[0]=mut_max
        print("min: ", self.mut_min)
        print("max: ", self.mut_max)
        # Select indices in self.children for random mutations
        inds = np.random.randint(self.children.size, size = int(self.children.size * self.f_mut)) # sample random numbers from 1 to # of elements in children elements
        inds = [ (int(_/self.nx), _ % self.nx) for _ in inds ] # convert numbers into row/col positions

        # Initialize list of random mutations uniformly sampled from mut_min to mut_max
        r_list = (np.random.random(size=int(self.children.size * self.f_mut))-0.5) 
        #print(r_list)
        # for i in self.vars_dict["LJs"]:
        #     r_list[:, self.vars_dict["start_ind"][i]:self.vars_dict["end_ind"][i]]=r_list[:,self.vars_dict["start_ind"][i]:self.vars_dict["end_ind"][i]] * 
        # print("r_list: ", r_list)
        # Apply mutations                     
        for count_i,i in enumerate(inds):
            if i[1] < self.vars_dict["end_ind"][0]: # Essentially, determine if this is a sigma or epsilon value and use the appropriate set of mutation min/maxes
                mmi=0
            else:
                mmi=1
            #print(i)
            #print(mmi)
            self.children[i] += r_list[count_i]*(self.mut_max[mmi] - self.mut_min[mmi]) + 0.5 * (self.mut_max[mmi] + self.mut_min[mmi])
        #print("children: ", self.children)
        # Children outside of range are set to nearest boundary (ie min or max)
        for k in range(len(self.children)):
            for m in range(len(self.children[k])):
                #print("before: ", self.children[k][m])
                if self.eps:
                    if m < self.vars_dict["end_ind"][0]:
                        minmaxind=0
                    else:
                        minmaxind=1
                else:
                    minmaxind=0
                if self.children[k][m] < self.xmin[minmaxind]:
                    #self.children[k][m]=self.children[k][m]+half_range
                    self.children[k][m]=self.xmin[minmaxind]
                elif self.children[k][m] > self.xmax[minmaxind]:
                    #self.children[k][m]=self.children[k][m]-half_range
                    self.children[k][m]=self.xmax[minmaxind]
                #print("after: ", self.children[k][m])
        # Update generation members
        #print(self.beta[-1])
        #print(self.children)
        self.beta[-1] = np.vstack([self.beta[-1],self.children])
        #print("new betas: ", self.beta[-1])
        # Calculate generation losses
        if self.narrow==-1:
            sr_mut=self.sr
        else:
            sr_mut=round(self.sr[self.g-1],2)
        just_switch=False # Did the scaling switch this generation?
        # Specify p or v scaling
        if self.p_scale and self.v_scale:
            if self.g > self.switch_gen:
                p=False
                v=True
                if self.g==self.switch_gen+1:
                    just_switch=True # We need to know when the scaling switches because we need to define a new best loss value. This is because the best parameter in the new regime is not guaranteed to be better than the current best.
                    if self.loss_func=="frac":
                        self.seed_loss="find" # For fractional loss function we need to create a new seed loss to reference.
            else:
                p=True
                v=False
        elif self.p_scale:
            p=True
            v=False
        elif self.v_scale:
            p=False
            v=True
        else:
            p=False
            v=False
        if self.time_gen and self.g>self.time_gen: # Override previous scaling decision if its time for longer time steps.
            p=True  # Testing Pressure in longer time steps. Functionality could be added to test volume as well/instead.
            v=False
            t=True
            if self.g==self.time_gen+1:
                just_switch=True # We need to know when the scaling switches because we need to define a new best loss value. This is because the best parameter in the new regime is not guaranteed to be better than the current best.
                if self.loss_func=="frac":
                    self.seed_loss="find" # For fractional loss function we need to create a new seed loss to reference.
        else:
            t=False
        # Loss Function Call
        new_losses, new_all_loss, new_short_loss, static_rdf, minned_rdf, seed_loss_trash, data_dict = self.L(self.beta[-1], self.g, self.user,self.alpha,sr_mut,self.gaus_std,self.rdf_short,self.seed_loss,self.folder,p_scale=p,p_scale_vals=self.p_scale_vals, v_scale=v, t_scale=t, loss_func=self.loss_func, eps=self.eps, order=self.order, dims=self.dims, cell_sym=self.cell_sym, bfw=self.bfw, python=self.python, lammps=self.lammps, mpirun=self.mpirun)
        self.data_dict[str(self.g)]=data_dict
        self.losses+=[new_losses]
        self.all_loss+=[new_all_loss]
        self.short_loss+=[new_short_loss]
        if self.loss_func=="frac" and just_switch==True:
            self.seed_loss=seed_loss_trash # Change the seed loss if required by the loss function type.
        print("sr: ", sr_mut)
        
        # Sort by fitness
        inds = np.argsort(self.losses[-1])
        self.beta[-1] = self.beta[-1][inds]
        self.losses[-1] = self.losses[-1][inds]
        self.all_loss[-1]=self.all_loss[-1][inds]
        self.short_loss[-1]=self.short_loss[-1][inds]
        # Print current best
        print("Current best parameters from gen {} found in spec {}".format(self.g, inds[0]))
        print(self.beta[-1][0])

        # Update increment
        self.g += 1

        # Update best
        if self.losses[-1][0] < self.best_L:
            self.best_x = self.beta[-1][0]
            self.best_L = self.losses[-1][0]
            self.best_all=self.all_loss[-1][0]
            self.best_short=self.short_loss[-1][0]
        elif just_switch==True:
            self.best_x = self.beta[-1][0]
            self.best_L = self.losses[-1][0]
            self.best_all=self.all_loss[-1][0]
            self.best_short=self.short_loss[-1][0]
            print("SWITCH: Scaling has changed, forcing new best loss function...\n")
        return inds[0]

# This is /a/ smoothing function, but not the best one. Please use rdf_smoother in crys_rdf_gen.py for consistency
def gsmooth(rdf, sr, std):
    grdf=np.zeros((len(rdf),2))
    grdf[:,0]=rdf[:,0]
    for q in range(len(rdf)):
        if rdf[q,1]==0.0:
            pass
        else:
            srind=int(sr*100) # Convert smoothing distance into number of indices
            xvals=grdf[q-srind:q+srind+1,0]
            gaus=stats.norm.pdf(xvals, rdf[q,0], std)
            scale=rdf[q,1]/gaus.sum()
            grdf[q-srind:q+srind+1,1]=grdf[q-srind:q+srind+1,1]+gaus*scale
            #rdf[:,1]=grdf[:,1]
    return grdf

# Retrieves pressure values from .in.out files of minimization runs.
def get_inout(file_name,data_type):
    Vals=[]
    with open(file_name, 'r') as f:
        step_flag=False
        for lines in f:
            words=lines.split()
            if words==[]:
                pass
            elif words[0]=="Step":
                step_flag=True
                for i,word in enumerate(words):
                    if word==data_type:    # Find what position value is in
                        index=i
                    else:
                        pass
            elif step_flag==True:
                try:
                    int(words[0])
                    Vals.append(float(words[-1]))
                except Exception as e:
                    print(e)
                    step_flag=False
                    break
    #Pressure = np.mean(Pressures)
    # Use last pressure instead of average. Average pressure is not representative of the final system because it hasn't been minimized yet. Additionally, average pressure could be skewed for systems that take more steps to equilibrate.
    Val = Vals[-1] 
    Val_init=Vals[0]
    return Val, Val_init

# Identifies initial peak of rdf supplied. 
def get_rdf_max(traj, data, bead_num, name="static"):
    sp.call('python {}/Analysis/rdf.py {} {} {} {} type type -o {} -r_max 30.0 -bond_sep -1'.format(script_directory, traj, data, bead_num, bead_num, name),shell=True) 
    with open('{}_{}_{}.rdf'.format(name, bead_num, bead_num), 'r') as f:
        max_counter=0
        current_max=[0.0, 0.0]
        for lines in f:
            vals=lines.split()
            if vals[0]=="r":    # Skip header
                pass
            elif max_counter>10:    # Exit if you've not found a new max in long enough.
                break
            elif float(vals[1])>=current_max[1]:    # Update if new value is larger than or equal to original
                current_max=[float(vals[0]), float(vals[1])]
                max_counter=0
            else:   # If its not greater, update the max counter. 
                max_counter+=1
    return current_max  

# Reads a specified atrribute from a thermo file and returns the average.
def read_thermo(thermo_file, data_type, data_start=0, raw_data=False): # get an average value for a thermo value of choice. Supply thermo file name and data column header as strings.
    with open(thermo_file, 'r') as thermo:
        thermo_total=[]
        time_total=[]
        thermo_count=0
        for line in thermo:
            words=line.split()
            if words[0]=="#" and words[1]=="TimeStep":
                for i,word in enumerate(words):
                    if word==data_type:    # Find what position density is in
                        index=i-1
                    else:
                        pass
            if words[0]!="#":               # After header, start summing and counting number of rows.
                if int(words[0])>data_start:
                    thermo_total.append(float(words[index]))
                    time_total.append(int(words[0]))
                    thermo_count+=1
        thermo.close()
    # Return average, initial, and final values found in thermo file.
    #print("thermo total: ", thermo_total)
    try:
        thermo_ave=sum(thermo_total)/thermo_count
        thermo_init=thermo_total[0]
        thermo_final=thermo_total[-1]
    except Exception as e: 
        print("Error finding thermodynamic information... Simulation likely failed. Error message: \n")
        print(e)
        traceback.print_exc()
        if thermo_count == 0:
            print("Thermo count = 0, empty thermo.avg file ! :(\n")
        print("Setting artificially high thermo values.")
        thermo_ave=999999.9999
        thermo_init=999999.9999
        thermo_final=999999.9999
    if raw_data:
        data_np=np.array([time_total, thermo_total]).T
        return data_np, thermo_ave
    else:
        return thermo_ave, thermo_init, thermo_final

def plot_submit(current_dir, file_name, script_directory="~/bin/COGA", python="python"):
    with open('plot_GA.submit', 'w') as submit:
        submit.write('#!/bin/sh\n') 
        submit.write('#SBATCH --job-name plot_GA\n')
        submit.write('#SBATCH -A debug\n')
        submit.write('#SBATCH --nodes=1\n')
        submit.write('#SBATCH --ntasks=20\n')
        submit.write('#SBATCH --time=00:30:00\n')
        submit.write('#SBATCH --output plot_GA.out\n')
        submit.write('#SBATCH --error plot_GA.err\n')
        submit.write('#cd into submission trajectory\n')
        submit.write('cd {} # CHANGE DIRECTORY\n'.format(current_dir))
        submit.write('echo Working directory is {} # CHANGE DIRECTORY\n'.format(current_dir))
        submit.write('echo Running on host `hostname`\n')
        submit.write('echo Time is `date`\n\n')
        submit.write('cd .\n')
        submit.write('{} -u {}/Plotting/GA_plot.py {} > plot_GA.log      &\n'.format(python, script_directory, file_name))
        submit.write('wait\n\n')
        submit.write('echo Time is `date`\n')
        submit.write('cd {} & # CHANGE DIRECTORY\n'.format(current_dir))
        submit.write('wait\n')

def final_submit(current_dir, final_command, account, nps, plot_folder, script_directory):
    with open('final_GA.py', 'w') as pyth:
        pyth.write("#!/bin/env python\n")
        pyth.write("# Author: Dylan Fortney\n")
        pyth.write("import os, sys, argparse, io\n")
        pyth.write("import numpy as np\n")
        pyth.write("import subprocess as sp\n")
        pyth.write("from pathlib import Path\n")
        #pyth.write("home=str(Path.home())\n")
        pyth.write("script_directory='{}'\n".format(script_directory))
        pyth.write("sys.path.append('{}'.format(script_directory))\n")
        pyth.write("from COGA import rdf_loss")
        pyth.write("\n")
        pyth.write("\n{}\n".format(final_command))
        pyth.write('print("Final rdf loss: ", np.sum(data_dictf["raw_rdf"][0]))\n')
        pyth.write('print("SLoss: ", short_lossf)\n')
        pyth.write("os.chdir('{}/gen_final/spec0/')\n".format(current_dir))
        pyth.write('for d in os.listdir():\n')
        pyth.write('\tif d.split(".")[-1]=="pdf":\n')
        pyth.write('\t\tnew_name = "final_"+d\n')
        pyth.write("\t\tsp.call('mv {} {}'.format(d, new_name), shell=True)\n")
        pyth.write("sp.call('mv *.pdf {}/{}', shell=True)\n".format(current_dir, plot_folder))
        pyth.close()

    with open('final_GA.submit', 'w') as submit:
        submit.write('#!/bin/sh\n') 
        submit.write('#SBATCH --job-name {}_final\n'.format(current_dir.split("/")[-1]))
        submit.write('#SBATCH -A {}\n'.format(account))
        submit.write('#SBATCH --nodes=1\n')
        submit.write('#SBATCH --ntasks={}\n'.format(str(nps)))
        submit.write('#SBATCH --time=120:00:00\n')
        submit.write('#SBATCH --output GA_final.out\n')
        submit.write('#SBATCH --error GA_final.err\n')
        submit.write('#cd into submission trajectory\n')
        submit.write('cd {} # CHANGE DIRECTORY\n'.format(current_dir))
        submit.write('echo Working directory is {} # CHANGE DIRECTORY\n'.format(current_dir))
        submit.write('echo Running on host `hostname`\n')
        submit.write('echo Time is `date`\n\n')
        submit.write('cd .\n')
        submit.write('python -u {}/final_GA.py > GA_final.log      &\n'.format(current_dir))
        submit.write('wait\n\n')
        submit.write('echo Time is `date`\n')
        submit.write('cd {} & # CHANGE DIRECTORY\n'.format(current_dir))
        submit.write('wait\n')
        submit.close()
        
def submit_dilatometry(file_name, folder, dims, submit=False, order="", t_reshape=500000, t_ramp=1000000, t_eqhigh=10000000, t_cool=30000000, t_eqlow=10000000, t_warm=30000000, t_eqhigh2=10000000, t_cool2=30000000, t_eqlow2=10000000, t_warm2=30000000, cell_moles=1, orient="align", o_axis="z", python="python", mpirun="/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun", lammps="/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322", script_directory="~/bin/COGA"):
    folder_name=file_name+"dila"
    os.mkdir(folder_name)
    sp.call("cp {}.in.settings {}".format(file_name, folder_name), shell=True)
    sp.call("cp {}.data {}/{}_org.data".format(file_name, folder_name, file_name), shell=True) # copy original data file in just for fun. Change the name because we'd otherwise overwrite it.
    sp.call("cp ../../CG_final_extend.map {}".format(folder_name), shell=True)
    sp.call("cp ../../CG.xyz {}".format(folder_name), shell=True)
    os.chdir(folder_name)
    print("in dila: ", os.getcwd())
    sp.call("ls ../../../martini_triclinic.data", shell=True)
    sp.call("ls ../../../CG.map", shell=True)
    sp.call("ls {}/Input_Operations/gen_md_inputs_martini.py".format(script_directory), shell=True)
    Ns=1 # Number of Molecules
    for i in dims.split(): Ns=Ns*int(i)
    Ns=int(Ns*cell_moles) # Add this correction so dila matches final simulation size. Need to make it an integer for gen_md_inputs_martini.py
    if orient == "random":
        sp.call('{} {}/Input_Operations/gen_md_inputs_martini.py -maps "CG.map" -xyzs "CG.xyz" -Ns {} -o {} -python {} -martiniFF "{}/Input_Operations/martini_v2.P_supp-material_original.itp"'.format(python, script_directory, Ns, file_name, python, script_directory), shell=True) # Generate Diffuse gas data file!
    elif orient == "align": # Align with a specific axis in a packed arrangment. The axis is decided by o_axis, and can be changed between x, y, and z if the molcules are not successfully aligned due to gimbal locking.
        sp.call('{} {}/Dilatometry/CG_arbitrary_box.py ../../../martini_triclinic.data -N {} -o {} -orient {} -map ../../../CG.map -vol_scale 3.0  --tight > arbitrary.log'.format(python, script_directory, Ns, file_name, o_axis), shell=True)
        sp.call('mv {}_{}.data {}.data'.format(file_name, o_axis, file_name), shell=True)
    else:
        print("Error, please provide a valid orientation type: random, align.")
        quit()
    out=folder # Name of parent folder, which typically contains the molecule name or some other identifier for checking jobs.
    print("Reading data file {}_org.data to find cell dimensions...".format(file_name))
    with open("{}_org.data".format(file_name), 'r') as data_file:
        for line in data_file:
            words = line.split()
            if len(words)==0:
                pass
            elif words[-1]=="xhi":
                xlo=float(words[0])
                xhi=float(words[1])
                xl=float(xhi-xlo)
            elif words[-1]=="yhi":
                ylo=float(words[0])
                yhi=float(words[1])
                yl=float(yhi-ylo)
            elif words[-1]=="zhi":
                zlo=float(words[0])
                zhi=float(words[1])
                zl=float(zhi-zlo)
                # break
            elif words[-1]=="yz":
                xy=float(words[0])
                xz=float(words[1])
                yz=float(words[2])
                break
            elif len(words)==2 and words[-1]=="atoms":
                bead_num = int(words[0]) # Number of atoms in the system. Dilatometry file will necessarily have the same number of atoms as the final file.
            else:
                pass
    # Assign cell size with cell dimensions.
    # cell_size="{} {} {} {} {} {}".format(xlo, xhi, ylo, yhi, zlo, zhi)
    # cell_vol=triclinic_volume(xl, yl, zl, xy, xz, yz)
    # cell_side=cell_vol**(1/3)

    # Assign cell size with number density.
    num_density=0.09         # Assign number density, using 0.09/angstrom**3 per Savoie's recommendation
    cell_vol=bead_num*4/num_density # Cell volume in angstroms**3. Multiply by 4, assuming approximately 4 atoms per bead.
    cell_side=(cell_vol ** (1/3))/2.0
    print("cell_side ", cell_side)
    cell_size = "-{} {} -{} {} -{} {}".format(cell_side, cell_side, cell_side, cell_side, cell_side, cell_side) # Make cell dimensions equal to the cube root of the crystalline cell volume.
    print("cell_size", cell_size)
    print("gen_dilatometry_inputs.py function call: \n")
    dila_call="{} {}/Dilatometry/gen_dilatometry_inputs.py {} -o dila -folder {} -t_reshape {} -t_ramp {} -t_eqhigh {} -t_cool {} -t_eqlow {} -t_warm {} -cell_size '{}' -order {} -python '{}' -mpirun '{}' -lammps '{}' ".format(python, script_directory, file_name, out, t_reshape, t_ramp, t_eqhigh, t_cool, t_eqlow, t_warm, cell_size, order, python, mpirun, lammps)
    print(dila_call)
    sp.call(dila_call, shell=True)
    if submit:
        print("Submitting Dilatometry Simulations...")
        sp.call("sbatch {}dila.submit".format(file_name), shell=True)
    os.chdir("../")

def submit_melt(file_name, folder, dims, submit=False, order="", t_reshape=1000000, t_eq=10000000, t_ramp=10000000,  cell_moles=1, orient="align", o_axis="z", state="crys", python="python", mpirun="/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun", lammps="/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322", script_directory="~/bin/COGA"):
    folder_name=file_name+state
    os.mkdir(folder_name)
    sp.call("cp {}.in.settings {}".format(file_name, folder_name), shell=True)
    sp.call("cp {}.data {}/{}_org.data".format(file_name, folder_name, file_name), shell=True) # copy original data file in just for fun. Change the name because we'd otherwise overwrite it.
    sp.call("cp ../../CG_final_extend.map {}".format(folder_name), shell=True)
    sp.call("cp ../../CG.xyz {}".format(folder_name), shell=True)
    os.chdir(folder_name)
    print("in dila: ", os.getcwd())
    Ns=1 # Number of Molecules
    for i in dims.split(): Ns=Ns*int(i)
    Ns=int(Ns*cell_moles) # Add this correction so dila matches final simulation size. Need to make it an integer for gen_md_inputs_martini.py
    if state == "crys":
        sp.call('{} {}/Input_Operations/gen_md_inputs_martini.py -maps "CG.map" -xyzs "CG.xyz" -Ns {} -o {} -python {} -martiniFF "{}/Input_Operations/martini_v2.P_supp-material_original.itp"'.format(python, script_directory, Ns, file_name, python, script_directory), shell=True) # Generate Diffuse gas data file!
    elif state == "LC":
        sp.call('{} {}/Dilatometry/CG_arbitrary_box.py ../../../martini_triclinic.data -N {} -o {} -orient {} -map ../../../CG.map -vol_scale 3.0  --tight > arbitrary.log'.format(python, script_directory, Ns, file_name, o_axis), shell=True)
        sp.call('mv {}_{}.data {}.data'.format(file_name, o_axis, file_name), shell=True)
    else:
        print("Error, please provide a valid state type: crys, LC.")
        quit()
    out=folder # Name of parent folder, which typically contains the molecule name or some other identifier for checking jobs.
    print("Reading data file {}_org.data to find cell dimensions...".format(file_name))
    with open("{}_org.data".format(file_name), 'r') as data_file:
        for line in data_file:
            words = line.split()
            if len(words)==0:
                pass
            elif words[-1]=="xhi":
                xlo=float(words[0])
                xhi=float(words[1])
                xl=float(xhi-xlo)
            elif words[-1]=="yhi":
                ylo=float(words[0])
                yhi=float(words[1])
                yl=float(yhi-ylo)
            elif words[-1]=="zhi":
                zlo=float(words[0])
                zhi=float(words[1])
                zl=float(zhi-zlo)
                # break
            elif words[-1]=="yz":
                xy=float(words[0])
                xz=float(words[1])
                yz=float(words[2])
                break
            elif len(words)==2 and words[-1]=="atoms":
                bead_num = int(words[0]) # Number of atoms in the system. Dilatometry file will necessarily have the same number of atoms as the final file.
            else:
                pass

    # Assign cell size with number density.
    num_density=0.09         # Assign number density, using 0.09/angstrom**3 per Savoie's recommendation
    cell_vol=bead_num*4/num_density # Cell volume in angstroms**3. Multiply by 4, assuming approximately 4 atoms per bead.
    cell_side=(cell_vol ** (1/3))/2.0
    print("cell_side ", cell_side)
    cell_size = "-{} {} -{} {} -{} {}".format(cell_side, cell_side, cell_side, cell_side, cell_side, cell_side) # Make cell dimensions equal to the cube root of the crystalline cell volume.
    print("cell_size", cell_size)
    print("gen_dilatometry_inputs.py function call: \n")
    T_start=10
    T_end=500
    cycles=10
    dila_call="{} {}/Dilatometry/gen_slowmelt_inputs.py {} -o {} -folder {} -t_reshape {} -t_ramp {} -t_eq {} -cell_size '{}' -order {} -T_start {} -T_end {} -cycles {} -python '{}' -mpirun '{}' -lammps '{}' ".format(python, script_directory, file_name, state, out, t_reshape, t_ramp, t_eq, cell_size, order, T_start, T_end, cycles, python, mpirun, lammps)
    print(dila_call)
    sp.call(dila_call, shell=True)
    if submit:
        print("Submitting Dilatometry Simulations...")
        sp.call("sbatch {}dila.submit".format(file_name), shell=True)
    os.chdir("../")

def triclinic_volume(lx, ly, lz, xy, xz, yz): # l* is the diference bewteen *lo and *hi in lammps.
    a = lx
    b = np.sqrt(ly**2+xy**2)
    c = np.sqrt(lz**2+xz**2+yz**2)
    alpha = np.arccos((xy*xz+ly*yz)/(b*c))
    beta = np.arccos(xz/c)
    gamma = np.arccos(xy/b)
    a_vec=np.array([0, a])
    b_vec=np.array([0, b])
    c_vec=np.array([0, c])
    V_tr=np.dot(np.dot(a_vec, b_vec), c_vec)*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
    V_tr=V_tr[1]
    print("Volume: ", V_tr)
    return V_tr


if __name__ == '__main__':
    main(sys.argv[1:])
