#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
import subprocess as sp
from pathlib import Path
home=str(Path.home())
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission", "/Dilatometry"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
from COGA import read_thermo
import matplotlib.pyplot as plt
from matplotlib import cm
from GA_plot import plot_ac
import random

def main(argv):
    parser = argparse.ArgumentParser(description='''Runs analysis on dilatometry simulations.                                                                                                                                           
    Input: Thermo file to be read
                                                                                                                                                                                                      
    Output: Plot of temperature vs density.                                                                                                                                                                   
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('thermo_file', type=str, help='Name of thermo file')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG_files/', help='Output. Default: CG_files/')

    parser.add_argument('-prefix', dest='prefix', type=str, default='spec0', help='Prefix for results from simulation. Default: spec0')

    parser.add_argument('-map', dest='map', default='z_extend.map', type=str, help='Map file for autocorrelation')

    parser.add_argument('-t_reshape', dest='t_reshape', default=1000000, type=int, help='Timesteps for reshaping the box. Default: 1,000,000 (1 ns)')

    parser.add_argument('-t_eq', dest='t_eq', default=10000000, type=int, help='Timesteps for ramping the temperature up from 10K. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_ramp', dest='t_ramp', default=10000000, type=int, help='Timesteps for high temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-restart_steps', dest='restart_steps', default='', type=str, help='The simulation segment name and number of relax steps if the simulation was restarted. Format: "warmup2 10000" ')

    parser.add_argument('-order', dest='order', default='', type=str, help='Determines if the scalar order parameter will be caclulated and if so what vector to be used. Format "0-x-y", x,y being the atom numbers that compose the vector. ')

    parser.add_argument('-T_start', dest='T_start', default=10, type=int, help='Temperature to start the simulations at, in Kelvin')

    parser.add_argument('-T_end', dest='T_end', default=500, type=int, help='Temperature to end the simulations at, in Kelvin')

    parser.add_argument('-cycles', dest='cycles', default=10, type=int, help='Number of ramp-equilibration cycles to perform.')

    parser.add_argument('-cyc_start', dest='cyc_start', default=1, type=int, help='starting cycle number. Useful if you restarted and want to do analysis on only a set of output files.')

    parser.add_argument('--submit_restart', dest='submit_restart', default=False, action='store_const', const=True, help='When present, the script will automatically write and submit the restart files for the incompleted steps and then quit prior to beginning analysis.')

    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    temp_steps = np.linspace(args.T_start, args.T_end, args.cycles+1, dtype=int)
    step_keys=["reshape"]
    step_dict={}
    step_dict.update({"reshape":{}})
    step_dict["reshape"].update({"steps":args.t_reshape,"colors":"Greys", "label":"Reshape Box", "zorder":1,"color_min":0.4})
    for c in range(args.cyc_start-1, args.cyc_start-1+args.cycles):
        step_keys.append("ramp{}".format(c+1))
        step_keys.append("eq{}".format(c+1))
        step_dict.update({"ramp{}".format(c+1):{}})
        step_dict.update({"eq{}".format(c+1):{}})
        step_dict["ramp{}".format(c+1)].update({"steps":args.t_ramp,"colors":"Blues", "label":"Ramp from {}K to {}K".format(temp_steps[c], temp_steps[c+1]), "zorder":(2*c+2),"color_min":0.4})
        step_dict["eq{}".format(c+1)].update({"steps":args.t_eq,"colors":"Greens", "label":"Eq at {}K".format(temp_steps[c+1]), "zorder":(2*c+3),"color_min":0.4})
    print("step keys: ", step_keys)
    print("dict keys: ", step_dict.keys())
    

    plot_dict={"Dilatometry":step_keys, "Dilatometry_prod":step_keys[1:], "Dilatometry_Ramps":[r for r in step_keys if "ramp" in r]}

    # Read in data from the thermo file.
    temp_np, temp_ave = read_thermo(args.thermo_file, "v_my_temp", raw_data=True)
    rho_np, rho_ave = read_thermo(args.thermo_file, "v_my_rho", raw_data=True)
    print(temp_np)
    print(rho_np[:,1])
    #data_np=np.concatenate((data_np, np.reshape(rho_np[:,1], (len(rho_np[:,1]), 1))), axis=1)
    data_np=np.zeros((len(temp_np), 3))
    data_np[:,0]=temp_np[:,0]
    data_np[:,1]=temp_np[:,1]
    data_np[:,2]=rho_np[:, 1]
    # print("all data: ", data_np)
    # print(len(data_np))

    total_steps=0 # Moving counter for the current number of steps for the following for loop. Will become the total number of steps when for loop is done.
    total_inds=0 # Moving counter for indices when organizing data, particularly for the all_colors array
    # Prep plotting information
    # color_min=0.4
    completed_steps=[]
    uncompleted_steps=[]
    completed_steps_num=0
    all_colors=np.zeros((len(data_np), 4))
    # Loop over each of the data steps to organize and analyze data:
    for j,s in enumerate(step_keys):
        step_dict[s].update({"data":[]})                    # Make empty list for data
        step_dict[s].update({"start_step":total_steps})     # Input start step (0 for the initial step)
        total_steps+=step_dict[s]["steps"]                    # Update total steps with the current segment steps 
        step_dict[s].update({"end_step":total_steps})       # Input end step (Sum of steps including current step)
        for i in range(len(data_np)):
            if data_np[i, 0]>=step_dict[s]["start_step"] and data_np[i,0]<step_dict[s]["end_step"]:
                step_dict[s]["data"].append(data_np[i,:])
            else:
                pass
        step_dict[s]["data"]=np.array(step_dict[s]["data"])
        if len(step_dict[s]["data"])>0:
            completed_steps.append(s)
            completed_steps_num+=len(step_dict[s]["data"])*1000 # Calculate the number of steps that successfully completed.
        # Check to see if each step has a restart file written. If not, it didn't finish.
        if os.path.exists(args.prefix+"."+s+".end.restart"):
            pass
        else:
            uncompleted_steps.append(s)
            if s in completed_steps:        # If only a partial simulation finished, remove it from completed steps.
                completed_steps.remove(s)
        print(s, ": ", step_dict[s]["data"])

        # Generate colormaps, darker being for later time steps. Plot temp on x axis, rho on y axis.
        # Separate into if statements to deterine if a regime has been filled. This way, if a simulation doesn't finish we still can have a graph, albeit an incomplete one.
        if step_dict[s]["data"].size > 0: # Check if we got any data.
            step_dict[s].update({"color_map":cm.get_cmap(step_dict[s]["colors"], len(step_dict[s]["data"]))})       # Generate color map from data and chosen map type
            step_dict[s].update({"color_input":(1-step_dict[s]["color_min"])*(step_dict[s]["data"][:,0]-min(step_dict[s]["data"][:,0]))/(max(step_dict[s]["data"][:,0]-min(step_dict[s]["data"][:,0])))+step_dict[s]["color_min"]}) # Generate input based on color_min
            step_dict[s].update({"color_output":step_dict[s]["color_map"](step_dict[s]["color_input"])})        # Generate colors for use in the graphs.
            step_dict[s].update({"start_ind":total_inds})   # Set intial index for the all_colors array
            total_inds+=len(step_dict[s]["data"])   # Update the total inds variable with the length of the current array of data.
            step_dict[s].update({"end_ind":total_inds})     # Set end index for the all_colors array.           
            all_colors[step_dict[s]["start_ind"]:step_dict[s]["end_ind"] ,:]=step_dict[s]["color_output"] # Update all_colors array
    # Loop over desired plot formats
    for p in plot_dict.keys():
        plt.figure(p)
        loop_step_keys=plot_dict[p]
        for s in loop_step_keys:
            #print(s)
            if step_dict[s]["data"].size > 0:
                plt.scatter(step_dict[s]["data"][:,1], step_dict[s]["data"][:,2], c=step_dict[s]["color_output"], label=step_dict[s]["label"], zorder=step_dict[s]["zorder"])
        plt.title("Dilatometry Simluation Results")
        plt.ylabel("Density g/cm^3")
        plt.xlabel("Temperature (K)")
        # plt.xlim((0,max(max(warm_data[:,1]), max(cool_data[:,1]))+100)) See if we actually need to devise axis limits...
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig("{}.pdf".format(p), bbox_inches="tight")
        plt.savefig("{}.png".format(p), bbox_inches="tight", dpi = 288)
    

    # plot temperature vs. timestep
    plt.figure("tempvtime")
    plt.scatter(data_np[:,0], data_np[:,1], c=all_colors)
    plt.xlabel("timestep")
    plt.ylabel("Temperature")
    plt.title("Temp Change over Time")
    plt.savefig("DilaTempvTime.pdf")
    plt.savefig("DilaTempvTime.png", dpi=288)
    current_dir=os.getcwd()
    folder=current_dir.split("/")[-4]
    #sp.call("mv *.pdf ../../../gen_plots_{}".format(folder), shell=True)

    # plot density vs. timestep
    plt.figure("rhovtime")
    plt.scatter(data_np[:,0], data_np[:,2], c=all_colors)
    plt.xlabel("timestep")
    plt.ylabel("Density g/cm^3")
    plt.title("Rho Change over Time")
    plt.savefig("DilaRhovTime.pdf")
    plt.savefig("DilaRhovTime.png", dpi=288)
    
    # String together trajectories and feed them to skim_traj.py to be skimmed.
    print("Total number of steps attempted: {}, ({} ns)".format(total_steps, total_steps/1000000))
    print("Total number of steps completed: {}, ({} ns)".format(completed_steps_num, completed_steps_num/1000000))
    print("Completed Steps: ", completed_steps)
    step_string=""
    for i in completed_steps: # We want to find the completed steps so we don't try to reference trajectories that aren't there.
        step_string+=args.prefix+"."+i+".lammpstrj "
    if uncompleted_steps and args.submit_restart:
        print("It looks like some steps didn't complete:")
        ucs_str=""
        for u in uncompleted_steps:
            ucs_str+=u+"\t"
            try:
                completed_steps.remove(u)   # Remove items from the complete steps if it didn't complete.
            except:
                pass
        ucs_str+"\n"
        print(ucs_str)
        print("Writing and submitting restart submission...")
        input_dict = {"T_start":args.T_start, "T_end":args.T_end, "cycles":args.cycles, "cyc_start":args.cyc_start, "t_reshape":args.t_reshape, "t_eq":args.t_eq, "t_ramp":args.t_ramp, "temp_steps":temp_steps, "map":args.map}
        submit_slowmelt_restart(uncompleted_steps, step_dict, args.thermo_file, args.prefix, completed_steps[-1], args.order, input_dict)
        print("Quitting before combining trajectories...")
        quit()

    step_string=step_string.strip()
    print("String for combined.lammpstrj: ", step_string)
    if os.path.exists("combined_dila.lammpstrj"):   # Save computational time by skipping combination or skimming steps if they are completed.
        print("Combined trajectory already exists, new one is not being created. If you wish to overwrite the old combined trajectory, delete it and rerun this script.")
    else:
        print("Combining trajectories...")
        sp.call("python {}/Input_Operations/string_trajectories.py '{}' -every 500 -o combined_dila.lammpstrj".format(script_directory, step_string), shell=True)
    # if os.path.exists("{}.skim_dila.lammpstrj".format(args.prefix)):
    #     print("Skimmed trajectory already exists, new one is not being created. If you wish to overwrite the old combined trajectory, delete it and rerun this script.")
    # else:
    #     print("Creating skimmed trajectory...")
    #     sp.call("python ~/bin/skim_traj.py combined_dila.lammpstrj -o {}.skim_dila -e 500 > skim.log".format(args.prefix), shell=True)
    if args.order:  # Calculate scalar order parameter if necessary
        so_list = plot_dict["Dilatometry_prod"]
        order_folder="_".join(args.order.split('-'))
        # so_list.append("skim_dila")
        # step_dict.update({"skim_dila":{"steps":total_steps/500}})
        print("Calculating Scalar Order: ")
        comb_name = "so-P2-comb.txt"    # Want to make sure we don't accidentally append new data to old data. Oops!
        dir_name = "dir-P2-comb.txt"
        if os.path.exists(comb_name):
            comb_count = 0
            comb_exists = True
            while comb_exists == True:
                comb_count += 1
                comb_name = "so-P2-comb{}.txt".format(str(comb_count))
                dir_name = "dir-P2-comb{}.txt".format(str(comb_count))
                comb_exists = os.path.exists(comb_name)
            print("comb file so-P2-comb.txt already exists... To avoid overwriting, writing to {}".format(comb_name))
        sp.call("touch {}".format(comb_name), shell=True) # Create the file that will house all of the scalar order data.
        sp.call("touch {}".format(dir_name), shell=True) # Create the file that will house all of the scalar order data.
        for d in so_list:   # The script by default only calculates on a logarithmic scale. This will be detrimental to the later steps as they will be sampled less. To remedy this, we sample each step individually, then combine them.
            print("\tOn {}...\n".format(d))
            step_traj=args.prefix+"."+d+".lammpstrj"
            if d == "skim_dila":
                f_every=1
            else:
                f_every=100
            print("\t\tCalculating every {} frames...".format(f_every))
            sp.call("python ~/bin/CG_Crystal/vautocorr.py -map {} -traj {} --modes {} -legendres 2 -f_end {} -f_every {} -o legendre_{} --overwrite --scalord --no_ac --no_xc > vautocorr_{}.txt".format(args.map, step_traj, args.order, step_dict[d]["steps"], f_every, d, d), shell=True)
            sp.call("cat legendre_{}/vectors/{}/so-P2.txt >> {}".format(d, order_folder, comb_name), shell=True) # Append appropriate file to the master file we created earlier.
            sp.call("cat legendre_{}/vectors/{}/dir-P2.txt >> {}".format(d, order_folder, dir_name), shell=True) # Append appropriate file to the master file we created earlier.
        plot_ac(comb_name) # Plot. This function was built for plotting ac but there is no functional difference. 

        skim_steps=total_steps/500
        sp.call("python {}/Analysis/vautocorr.py -map ../../../CG_final_extend.map -traj combined_dila.lammpstrj --modes {} -legendres 2 -f_end {} -f_every 1 -o legendre_skim --overwrite --scalord --no_ac --no_xc > vautocorr_skim.txt".format(script_directory, args.order, skim_steps), shell=True)
        plot_ac("legendre_skim/vectors/{}/so-P2.txt".format(order_folder)) # Plot. This function was built for plotting ac but there is no functional difference. 
        sp.call("mv so-P2.pdf so-P2-skim.pdf", shell=True)
        current_dir=os.getcwd()
        folder_name=current_dir.split('/')[-4]  # This gets the name of the folder assigned to this molecule, which is required for navigating some of the folders.
        print("folder name: ", folder_name)
        # sp.call("mv *.pdf ../../../gen_plots_{}".format(folder_name), shell=True)



def submit_slowmelt_restart(uncompleted_steps, step_dict, thermo, prefix, last_completed, order, input_dict): # Writes an init and submission file for the uncompleted steps. Takes the names of completed steps and the dictionary of dilatometry information.
    settings_file=prefix+".in.settings"
    init_file="restart.in.init"
    relax_steps=10000
    with open(init_file,'w') as f:
        f.write(
            "# LAMMPS input file for restarting dilatometry simulations of liquid crystaline systems from GA. \n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Variables\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "# File-wide variables\n"+\
            "variable    settings_name        index    {}\n".format(settings_file)+\
            "variable    prefix               index    {}\n".format(prefix)+\
            "variable    avg_freq             index    1000\n"+\
            "variable    coords_freq          index    1000\n"+\
            "variable    thermo_freq          index    1000\n"+\
            "variable    dump4avg             index    100\n"+\
            "variable    vseed                index    {}  \n".format(random.randint(100, 999))+\
            "\n"+\
            "# Reshape\n"+\
            "variable    nSteps_reshape        index    {}          # {} ps ({} ns) box reshaping (NVT)\n".format(step_dict["reshape"]["steps"], step_dict["reshape"]["steps"]/1000, step_dict["reshape"]["steps"]/1000000)+\
            "variable    temp_reshape          index    {}          #  {} K\n".format(input_dict["T_start"], input_dict["T_start"])+\
            "\n"
        )

        for c, cycle in enumerate(range(input_dict["cyc_start"], input_dict["cyc_start"]+input_dict["cycles"])):
            f.write(
                "# Ramp {}\n".format(cycle)+\
                "variable    nSteps_ramp{}        index    {}          # {} ps ({} ns) ramp to higher temperature (NPT) \n".format(cycle, input_dict["t_ramp"], input_dict["t_ramp"]/1000, input_dict["t_ramp"]/1000000)+\
                "\n"+\

                "# Equilibration at High Temp\n"+\
                "variable    nSteps_eq{}        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(cycle, input_dict["t_eq"], input_dict["t_eq"]/1000, input_dict["t_eq"]/1000000)+\
                "variable    press_eq{}         index    1.0         # 1.0 atm\n".format(cycle)+\
                "variable    temp_eq{}          index    {}          # {} K\n".format(cycle, input_dict["temp_steps"][c+1], input_dict["temp_steps"][c+1])+\
                "\n"
            )   

        f.write(
            # Change the name of the log output #
            "log ${prefix}dila_restart.log\n"+\

            "#===========================================================\n"+\
            "# GENERAL PROCEDURES\n"+\
            "#===========================================================\n"+\
            "units		real	# g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n"+\
            "dimension	3	# 3 dimensional simulation\n"+\
            "newton		off	# use Newton's 3rd law\n"+\
            "boundary	p p p	# periodic boundary conditions \n"+\
            "atom_style	full    # molecular + charge\n"+\

            "#===========================================================\n"+\
            "# FORCE FIELD DEFINITION\n"+\
            "#===========================================================\n"+\
            "special_bonds  lj   0.0 0.0 0.0  coul 0.0 0.0 0.0     # NO     1-4 LJ/COUL interactions\n"+\
            "pair_style   hybrid lj/gromacs/coul/gromacs 9.0 12.0 0.0 12.0 # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n"+\
            "bond_style   hybrid harmonic         # parameters needed: k_bond, r0\n"+\
            "angle_style  hybrid harmonic         # parameters needed: k_theta, theta0\n"+\
            "pair_modify    shift yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n"+\

            "#===========================================================\n"+\
            "# SETUP SIMULATIONS\n"+\
            "#===========================================================\n"+\

            "# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n"+\
            "read_restart {}.{}.end.restart\n".format(prefix, last_completed)+\
            "include ${prefix}.in.settings\n"+\

            "# SET RUN PARAMETERS\n"+\
            "timestep	1.0		# fs\n"+\
            "run_style	verlet 		# Velocity-Verlet integrator\n"+\
            "neigh_modify every 1 delay 0 check no one 10000 # More relaxed rebuild criteria can be used\n"+\

            "# SET OUTPUTS\n"+\
            "thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n"+\
            "thermo_modify   format float %14.6f\n"+\
            "thermo ${thermo_freq}\n"+\

            "# DECLARE RELEVANT OUTPUT VARIABLES\n"+\
            "variable        my_step equal   step\n"+\
            "variable        my_temp equal   temp\n"+\
            "variable        my_rho  equal   density\n"+\
            "variable        my_pe   equal   pe\n"+\
            "variable        my_ebon equal   ebond\n"+\
            "variable        my_eang equal   eangle\n"+\
            "variable        my_edih equal   edihed\n"+\
            "variable        my_evdw equal   evdwl\n"+\
            "variable        my_eel  equal   (ecoul+elong)\n"+\
            "variable        my_ent  equal   enthalpy\n"+\
            "variable        my_P    equal   press\n"+\
            "variable        my_vol  equal   vol\n"+\
            "variable        my_etot equal   etotal\n"+\

            "fix  averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file thermo_restart.avg format %20.10g\n"+\
            
            "#===========================================================\n"+\
            "# RUN CONSTRAINED RELAXATION\n"+\
            "#===========================================================\n\n"+\

            "fix relax all nve/limit 0.01\n"+\
            "dump relax all custom ${coords_freq} relax.lammpstrj id type x y z \n"+\
            "dump_modify relax sort  id\n"+\
            "run             {}\n".format(str(relax_steps))+\
            "unfix relax\n\n"

        )
        for c, cycle in enumerate(range(input_dict["cyc_start"], input_dict["cyc_start"]+input_dict["cycles"])):
            if c == 0:
                ramp_start = "temp_reshape"
            else:
                ramp_start = "temp_eq"+str(cycle-1)
            if "ramp{}".format(cycle) in uncompleted_steps:
                f.write(
                    "#===========================================================\n"+\
                    "# Ramp {} (NPT, Nose-Hoover)\n".format(cycle)+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "dump ramp{} all custom ${{coords_freq}} ${{prefix}}.ramp{}.lammpstrj id mol type x y z\n".format(cycle, cycle)+\
                    "dump_modify ramp{} sort id format float %20.10g\n".format(cycle)+\
                    "fix ramp{} all npt temp ${{{}}} ${{temp_eq{}}} 100.0 iso ${{press_eq{}}} ${{press_eq{}}} 1000.0\n".format(cycle, ramp_start, cycle, cycle, cycle)+\
                    "run ${{nSteps_ramp{}}}\n".format(cycle)+\
                    "unfix ramp{}\n".format(cycle)+\
                    "undump ramp{}\n".format(cycle)+\
                    "write_restart ${{prefix}}.ramp{}.end.restart\n".format(cycle)+\
                    "\n"
                )
            if "eq{}".format(cycle) in uncompleted_steps:
                f.write(
                    "#===========================================================\n"+\
                    "# Equilibration {} (NPT, Nose-Hoover)\n".format(cycle)+\
                    "#===========================================================\n"+\
                    "\n"+\
                    "dump eq{} all custom ${{coords_freq}} ${{prefix}}.eq{}.lammpstrj id mol type x y z\n".format(cycle, cycle)+\
                    "dump_modify eq{} sort id format float %20.10g\n".format(cycle)+\
                    "fix eq{} all npt temp ${{temp_eq{}}} ${{temp_eq{}}} 100.0 iso ${{press_eq{}}} ${{press_eq{}}} 1000.0\n".format(cycle, cycle, cycle, cycle, cycle)+\
                    "run ${{nSteps_eq{}}}\n".format(cycle)+\
                    "unfix eq{}\n".format(cycle)+\
                    "undump eq{}\n".format(cycle)+\
                    "write_restart ${{prefix}}.eq{}.end.restart\n".format(cycle)+\
                    "\n"
                )
        f.write(
            "#===========================================================\n"+\
            "# Clean and exit\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "unfix averages\n"
            )
    # Write slurm submission script
    slurm_file = prefix+"dila_restart"+'.submit'
    directory=os.getcwd()
    job_name=directory.split('/')[-4]+"dila_restart"   # Get main directory for job name
    outname=prefix+"dila_restart"
    queue = "bsavoie"
    nodes = 1
    cores = nodes * 128
    timelim = "14-00:00:00"
    # queue = "debug"
    # nodes = 1
    # cores = nodes * 20
    # timelim = "00:30:00"
    step_dict[uncompleted_steps[0]]["steps"]+=relax_steps
    with open(slurm_file,'w') as f:
        f.write(
            "#!/bin/bash\n"+\
            "#\n"+\
            "#SBATCH --job-name {}\n".format(job_name)+\
            "#SBATCH -o {}.out\n".format(outname)+\
            "#SBATCH -e {}.err\n".format(outname)+\
            "#SBATCH -A {}\n".format(str(queue))+\
            "#SBATCH -N {}\n".format(str(nodes))+\
            "#SBATCH -n {}\n".format(str(cores))+\
            "#SBATCH -t {}\n\n".format(str(timelim))+\

            "# Write out job information\n"+\
            "echo \"Running on host: $SLURM_NODELIST\"\n"+\
            "echo \"Running on node(s): $SLURM_NNODES\"\n"+\
            "echo \"Number of processors: $SLURM_NPROCS\"\n"+\
            "echo \"Current working directory: $SLURM_SUBMIT_DIR\"\n\n"+\

            "# User supplied shell commands\n"+\
            "cd $SLURM_SUBMIT_DIR\n\n"+\

            "# Run script\n"+\
            "echo \"Start time: $(date)\"\n"+\
            # "/apps/cent7/intel/impi/2017.1.132/bin64/mpirun -np {} /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501 -in  {}.in.init >> {}_lammps.out & wait \n".format(str(args.cores),outname,outname)+\
            # This change should allow us to run on multiple nodes...
            "/apps/cent7/intel/impi/2017.1.132/bin64/mpirun -np $SLURM_NPROCS /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501 -in  {} > restart.out & wait \n".format(init_file)+\
            "cat thermo.avg thermo_restart.avg > thermo2.avg\n"
            "python {}/Dilatometry/CG_slowmelt.py thermo2.avg -prefix {} -map {} -t_reshape {} -t_eq {} -t_ramp {} -T_start {} -T_end {} -cycles {} -cyc_start {} -order '{}' > dila_analysis.log \n".format(script_directory, prefix, input_dict["map"], input_dict["t_reshape"], input_dict["t_eq"], input_dict["t_ramp"], input_dict["T_start"], input_dict["T_end"], input_dict["cycles"], input_dict["cyc_start"], order)+\
            "echo \"End time: $(date)\""
            )
    f.close()
    #sp.call("sbatch {}".format(slurm_file), shell=True)
    return
    
    
    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
