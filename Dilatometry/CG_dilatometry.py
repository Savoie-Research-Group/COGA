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

    parser.add_argument('-t_reshape', dest='t_reshape', default=5000000, type=int, help='Timesteps for reshaping the box. Default: 5,000,000 (5 ns)')

    parser.add_argument('-t_ramp', dest='t_ramp', default=10000000, type=int, help='Timesteps for ramping the temperature up from 10K. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_eqhigh', dest='t_eqhigh', default=10000000, type=int, help='Timesteps for high temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_cool', dest='t_cool', default=30000000, type=int, help='Timesteps for cooling the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-t_eqlow', dest='t_eqlow', default=10000000, type=int, help='Timesteps for low temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_warm', dest='t_warm', default=30000000, type=int, help='Timesteps for warming the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-t_eqhigh2', dest='t_eqhigh2', default=10000000, type=int, help='Timesteps for the second high temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_cool2', dest='t_cool2', default=30000000, type=int, help='Timesteps for the second cooling of the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-t_eqlow2', dest='t_eqlow2', default=10000000, type=int, help='Timesteps for the second low temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_warm2', dest='t_warm2', default=30000000, type=int, help='Timesteps for the secondwarming of the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-restart_steps', dest='restart_steps', default='', type=str, help='The simulation segment name and number of relax steps if the simulation was restarted. Format: "warmup2 10000" ')

    parser.add_argument('-order', dest='order', default='', type=str, help='Determines if the scalar order parameter will be caclulated and if so what vector to be used. Format "0-x-y", x,y being the atom numbers that compose the vector. ')

    parser.add_argument('-python', dest='python', type=str, default='python', help='Location/name of the python version to be used.')

    parser.add_argument('-mpirun', dest='mpirun', type=str, default='/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun', help='Location/name of the mpi to be used to run with your version of lammps. Bell: /apps/cent7/intel/impi/2017.1.132/bin64/mpirun')

    parser.add_argument('-lammps', dest='lammps', type=str, default='/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322', help='Location/name of the lammps version to be run. Note, some changes to submissions may be necessary. Bell: /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501')

    parser.add_argument('--submit_restart', dest='submit_restart', default=False, action='store_const', const=True, help='When present, the script will automatically write and submit the restart files for the incompleted steps and then quit prior to beginning analysis.')

    parser.add_argument('--align', dest='align', default=False, action='store_const', const=True, help='When present, script will utilize the settings for an aligned dilatometry run, which usually has different simulation stages.')
    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    
    if args.align == False:
        step_keys=["reshape", "ramp", "eqhigh", "cooldown", "eqlow", "warmup", "eqhigh2", "cooldown2", "eqlow2", "warmup2"]
        step_dict={}
        for s in step_keys:
            step_dict.update({s:{}})
        # Initialize number of steps in each regime. May want to be switched to variables later in case these values change.
        # Transformed into dictionary to hopefully make changes to order or number of steps easier in the future.
        step_dict["reshape"].update({"steps":args.t_reshape,"colors":"Greys", "label":"Reshape Box", "zorder":1,"color_min":0.4})
        step_dict["ramp"].update({"steps":args.t_ramp,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["eqhigh"].update({"steps":args.t_eqhigh,"colors":"Greens", "label":"Equil at 500K", "zorder":11,"color_min":0.4})
        step_dict["cooldown"].update({"steps":args.t_cool,"colors":"Blues", "label":"Cooling", "zorder":4,"color_min":0.4})
        step_dict["eqlow"].update({"steps":args.t_eqlow,"colors":"Purples", "label":"Equil at 100K", "zorder":10,"color_min":0.4})
        step_dict["warmup"].update({"steps":args.t_warm,"colors":"Reds", "label":"Warming", "zorder":6,"color_min":0.4})
        step_dict["eqhigh2"].update({"steps":args.t_eqhigh2,"colors":"Oranges", "label":"Second Equil at 500K", "zorder":12,"color_min":0.4})
        step_dict["cooldown2"].update({"steps":args.t_cool2,"colors":"BuPu", "label":"Second Cooling", "zorder":13,"color_min":0.4})
        step_dict["eqlow2"].update({"steps":args.t_eqlow2,"colors":"PuRd", "label":"Second Equil at 100K", "zorder":14,"color_min":0.4})
        step_dict["warmup2"].update({"steps":args.t_warm2,"colors":"YlOrRd", "label":"Second Warming", "zorder":15,"color_min":0.4})
    
        plot_dict={"Dilatometry":step_keys, "Dilatometry_prod":["eqhigh", "cooldown", "eqlow", "warmup", "eqhigh2", "cooldown2", "eqlow2", "warmup2"], "Dilatometry_Ramps":["cooldown","warmup", "cooldown2", "warmup2"]}
    else:
        print("Using aligned settings...")
        step_keys=["reshape", "ramp1", "eq1", "ramp2", "eq2", "ramp3", "eq3", "ramp4", "eq4", "ramp5", "eqhigh", "cooldown", "eqlow", "warmup", "eqhigh2", "cooldown2", "eqlow2", "warmup2"]
        step_dict={}
        for s in step_keys:
            step_dict.update({s:{}})
        # Initialize number of steps in each regime. May want to be switched to variables later in case these values change.
        # Transformed into dictionary to hopefully make changes to order or number of steps easier in the future.
        step_dict["reshape"].update({"steps":args.t_reshape,"colors":"Greys", "label":"Reshape Box", "zorder":1,"color_min":0.4})
        step_dict["ramp1"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["eq1"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["ramp2"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["eq2"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["ramp3"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["eq3"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["ramp4"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["eq4"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["ramp5"].update({"steps":10000000,"colors":"Greys", "label":"Ramp up to 500K", "zorder":2,"color_min":0.4})
        step_dict["eqhigh"].update({"steps":args.t_eqhigh,"colors":"Greens", "label":"Equil at 500K", "zorder":11,"color_min":0.4})
        step_dict["cooldown"].update({"steps":args.t_cool,"colors":"Blues", "label":"Cooling", "zorder":4,"color_min":0.4})
        step_dict["eqlow"].update({"steps":args.t_eqlow,"colors":"Purples", "label":"Equil at 100K", "zorder":10,"color_min":0.4})
        step_dict["warmup"].update({"steps":args.t_warm,"colors":"Reds", "label":"Warming", "zorder":6,"color_min":0.4})
        step_dict["eqhigh2"].update({"steps":args.t_eqhigh2,"colors":"Oranges", "label":"Second Equil at 500K", "zorder":12,"color_min":0.4})
        step_dict["cooldown2"].update({"steps":args.t_cool2,"colors":"BuPu", "label":"Second Cooling", "zorder":13,"color_min":0.4})
        step_dict["eqlow2"].update({"steps":args.t_eqlow2,"colors":"PuRd", "label":"Second Equil at 100K", "zorder":14,"color_min":0.4})
        step_dict["warmup2"].update({"steps":args.t_warm2,"colors":"YlOrRd", "label":"Second Warming", "zorder":15,"color_min":0.4})
    
        plot_dict={"Dilatometry":step_keys, "Dilatometry_prod":["eqhigh", "cooldown", "eqlow", "warmup", "eqhigh2", "cooldown2", "eqlow2", "warmup2"], "Dilatometry_Ramps":["cooldown","warmup", "cooldown2", "warmup2"]}
    # reshape_steps=args.t_reshape
    # ramp_steps=args.t_ramp
    # eqhigh_steps=args.t_eqhigh
    # cool_steps=args.t_cool
    # eqlow_steps=args.t_eqlow
    # warm_steps=args.t_warm
    # eqhigh2_steps=args.t_eqhigh2
    # cool2_steps=args.t_cool2
    # eqlow2_steps=args.t_eqlow2
    # warm2_steps=args.t_warm2

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
    

    # plot temperature vs. timestep
    plt.figure("tempvtime")
    plt.scatter(data_np[:,0], data_np[:,1], c=all_colors)
    plt.xlabel("timestep")
    plt.ylabel("Temperature")
    plt.title("Temp Change over Time")
    plt.savefig("DilaTempvTime.pdf")
    current_dir=os.getcwd()
    folder=current_dir.split("/")[-4]
    sp.call("mv *.pdf ../../../gen_plots_{}".format(folder), shell=True)
    
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
        submit_dila_restart(uncompleted_steps, step_dict, args.thermo_file, args.prefix, completed_steps[-1], args.order)
        print("Quitting before combining trajectories...")
        quit()

    step_string=step_string.strip()
    print("String for combined.lammpstrj: ", step_string)
    if os.path.exists("combined_dila.lammpstrj"):   # Save computational time by skipping combination or skimming steps if they are completed.
        print("Combined trajectory already exists, new one is not being created. If you wish to overwrite the old combined trajectory, delete it and rerun this script.")
    else:
        print("Combining trajectories...")
        sp.call("{} {}/Input_Operations/string_trajectories.py '{}' -every 500 -o combined_dila.lammpstrj".format(args.python, script_directory, step_string), shell=True)
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
        sp.call("touch so-P2-comb.txt", shell=True) # Create the file that will house all of the scalar order data.
        for d in so_list:   # The script by default only calculates on a logarithmic scale. This will be detrimental to the later steps as they will be sampled less. To remedy this, we sample each step individually, then combine them.
            print("\tOn {}...\n".format(d))
            step_traj=args.prefix+"."+d+".lammpstrj"
            if d == "skim_dila":
                f_every=1
            else:
                f_every=100
            print("\t\tCalculating every {} frames...".format(f_every))
            sp.call("{} {}/Analysis/vautocorr.py -map ../../../CG_final_extend.map -traj {} --modes {} -legendres 2 -f_end {} -f_every {} -o legendre_{} --overwrite --scalord --no_ac --no_xc > vautocorr_{}.txt".format(args.python, script_directory, step_traj, args.order, step_dict[d]["steps"], f_every, d, d), shell=True)
            sp.call("cat legendre_{}/vectors/{}/so-P2.txt >> so-P2-comb.txt".format(d, order_folder), shell=True) # Append appropriate file to the master file we created earlier.
        plot_ac("so-P2-comb.txt") # Plot. This function was built for plotting ac but there is no functional difference. 

        skim_steps=total_steps/500
        sp.call("{} {}/Analysis/vautocorr.py -map ../../../CG_final_extend.map -traj combined_dila.lammpstrj --modes {} -legendres 2 -f_end {} -f_every 1 -o legendre_skim --overwrite --scalord --no_ac --no_xc > vautocorr_skim.txt".format(args.python, script_directory, args.order, skim_steps), shell=True)
        plot_ac("legendre_skim/vectors/{}/so-P2.txt".format(order_folder)) # Plot. This function was built for plotting ac but there is no functional difference. 
        sp.call("mv so-P2.pdf so-P2-skim.pdf", shell=True)
        current_dir=os.getcwd()
        folder_name=current_dir.split('/')[-4]  # This gets the name of the folder assigned to this molecule, which is required for navigating some of the folders.
        print("folder name: ", folder_name)
        sp.call("mv *.pdf ../../../gen_plots_{}".format(folder_name), shell=True)



def submit_dila_restart(uncompleted_steps, step_dict, thermo, prefix, last_completed, order): # Writes an init and submission file for the uncompleted steps. Takes the names of completed steps and the dictionary of dilatometry information.
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
            "variable    vseed                index    813  \n"+\
            "\n"+\
            "# Reshape\n"+\
            "variable    nSteps_reshape        index    {}          # {} ps ({} ns) box reshaping (NVT)\n".format(step_dict["reshape"]["steps"], step_dict["reshape"]["steps"]/1000, step_dict["reshape"]["steps"]/1000000)+\
            "variable    temp_reshape          index    10.0       # 10.0 K\n"+\
            "\n"+\
            "# Ramp\n"+\
            "variable    nSteps_ramp        index    {}          # {} ps ({} ns) ramp to higher temperature (NPT) \n".format(step_dict["ramp"]["steps"], step_dict["ramp"]["steps"]/1000, step_dict["ramp"]["steps"]/1000000)+\
            "\n"+\
            "# Equilibration at High Temp\n"+\
            "variable    nSteps_eqhigh        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(step_dict["eqhigh"]["steps"], step_dict["eqhigh"]["steps"]/1000, step_dict["eqhigh"]["steps"]/1000000)+\
            "variable    press_eqhigh         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqhigh          index    500.0       # 500.0 K\n"+\
            "\n"+\
            "# Cool Down\n"+\
            "variable    nSteps_cooldown      index    {}          # {} ps ({} ns) decreasing temp anneal (NPT)\n".format(step_dict["cooldown"]["steps"], step_dict["cooldown"]["steps"]/1000, step_dict["cooldown"]["steps"]/1000000)+\
            "variable    press_cooldown       index    1.0         # 1.0 atm\n"+\
            "variable    temp0_cooldown       index    500.0       # 500.0 K initial temp\n"+\
            "variable    tempf_cooldown       index    100.0       # 100.0 K final temp\n"+\
            "\n"+\
            "# Equilibration at Low Temp\n"+\
            "variable    nSteps_eqlow        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(step_dict["eqlow"]["steps"], step_dict["eqlow"]["steps"]/1000, step_dict["eqlow"]["steps"]/1000000)+\
            "variable    press_eqlow         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqlow          index    100.0       # 100.0 K\n"+\
            "\n"+\
            "# Warm Up\n"+\
            "variable    nSteps_warmup        index    {}          # {} ps ({} ns) increasing temp anneal (NPT)\n".format(step_dict["warmup"]["steps"], step_dict["warmup"]["steps"]/1000, step_dict["warmup"]["steps"]/1000000)+\
            "variable    press_warmup         index    1.0         # 1.0 atm\n"+\
            "variable    temp0_warmup         index    100.0       # 100.0 K initial temp\n"+\
            "variable    tempf_warmup         index    500.0       # 500.0 K final temp\n"+\
            "\n"+\
            "# Second Equilibration at High Temp\n"+\
            "variable    nSteps_eqhigh2        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(step_dict["eqhigh2"]["steps"], step_dict["eqhigh2"]["steps"]/1000, step_dict["eqhigh2"]["steps"]/1000000)+\
            "variable    press_eqhigh2         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqhigh2          index    500.0       # 500.0 K\n"+\
            "\n"+\
            "# Second Cool Down\n"+\
            "variable    nSteps_cooldown2      index    {}          # {} ps ({} ns) decreasing temp anneal (NPT)\n".format(step_dict["cooldown2"]["steps"], step_dict["cooldown2"]["steps"]/1000, step_dict["cooldown2"]["steps"]/1000000)+\
            "variable    press_cooldown2       index    1.0         # 1.0 atm\n"+\
            "variable    temp0_cooldown2       index    500.0       # 500.0 K initial temp\n"+\
            "variable    tempf_cooldown2       index    100.0       # 100.0 K final temp\n"+\
            "\n"+\
            "# Second Equilibration at Low Temp\n"+\
            "variable    nSteps_eqlow2        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(step_dict["eqlow2"]["steps"], step_dict["eqlow2"]["steps"]/1000, step_dict["eqlow2"]["steps"]/1000000)+\
            "variable    press_eqlow2         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqlow2          index    100.0       # 100.0 K\n"+\
            "\n"+\
            "# Second Warm Up\n"+\
            "variable    nSteps_warmup2        index    {}          # {} ps ({} ns) increasing temp anneal (NPT)\n".format(step_dict["warmup2"]["steps"], step_dict["warmup2"]["steps"]/1000, step_dict["warmup2"]["steps"]/1000000)+\
            "variable    press_warmup2         index    1.0         # 1.0 atm\n"+\
            "variable    temp0_warmup2         index    100.0       # 100.0 K initial temp\n"+\
            "variable    tempf_warmup2         index    500.0       # 500.0 K final temp\n"+\
            "\n"+\

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
        if "eqhigh" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# High Temperature Equilibration (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump eqhigh all custom ${coords_freq} ${prefix}.eqhigh.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify eqhigh sort id format float %20.10g\n"+\
                "fix eqhigh all npt temp ${temp_eqhigh} ${temp_eqhigh} 100.0 iso ${press_eqhigh} ${press_eqhigh} 1000.0\n"+\
                "run ${nSteps_eqhigh}\n"+\
                "unfix eqhigh\n"+\
                "undump eqhigh\n"+\
                "write_restart ${prefix}.eqhigh.end.restart\n"+\
                "\n"
            )
        if "cooldown" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# Cool Down (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump cooldown all custom ${coords_freq} ${prefix}.cooldown.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify cooldown sort id format float %20.10g\n"+\
                "fix cooldown all npt temp ${temp0_cooldown} ${tempf_cooldown} 100.0 iso ${press_cooldown} ${press_cooldown} 1000.0\n"+\
                "run ${nSteps_cooldown}\n"+\
                "unfix cooldown\n"+\
                "undump cooldown\n"+\
                "write_restart ${prefix}.cooldown.end.restart\n"+\
                "\n"
            )
        if "eqlow" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# Low Temperature Equilibration (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump eqlow all custom ${coords_freq} ${prefix}.eqlow.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify eqlow sort id format float %20.10g\n"+\
                "fix eqlow all npt temp ${temp_eqlow} ${temp_eqlow} 100.0 iso ${press_eqlow} ${press_eqlow} 1000.0\n"+\
                "run ${nSteps_eqlow}\n"+\
                "unfix eqlow\n"+\
                "undump eqlow\n"+\
                "write_restart ${prefix}.eqlow.end.restart\n"+\
                "\n"
            )
        if "warmup" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# Warm Up (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump warmup all custom ${coords_freq} ${prefix}.warmup.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify warmup sort id format float %20.10g\n"+\
                "fix warmup all npt temp ${temp0_warmup} ${tempf_warmup} 100.0 iso ${press_warmup} ${press_warmup} 1000.0\n"+\
                "run ${nSteps_warmup}\n"+\
                "unfix warmup\n"+\
                "undump warmup\n"+\
                "write_restart ${prefix}.warmup.end.restart\n"+\
                "\n"
            )
        if "eqhigh2" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# Second High Temperature Equilibration (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump eqhigh2 all custom ${coords_freq} ${prefix}.eqhigh2.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify eqhigh2 sort id format float %20.10g\n"+\
                "fix eqhigh2 all npt temp ${temp_eqhigh2} ${temp_eqhigh2} 100.0 iso ${press_eqhigh2} ${press_eqhigh2} 1000.0\n"+\
                "run ${nSteps_eqhigh2}\n"+\
                "unfix eqhigh2\n"+\
                "undump eqhigh2\n"+\
                "write_restart ${prefix}.eqhigh2.end.restart\n"+\
                "\n"
            )
        if "cooldown2" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# Second Cool Down (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump cooldown2 all custom ${coords_freq} ${prefix}.cooldown2.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify cooldown2 sort id format float %20.10g\n"+\
                "fix cooldown2 all npt temp ${temp0_cooldown2} ${tempf_cooldown2} 100.0 iso ${press_cooldown2} ${press_cooldown2} 1000.0\n"+\
                "run ${nSteps_cooldown2}\n"+\
                "unfix cooldown2\n"+\
                "undump cooldown2\n"+\
                "write_restart ${prefix}.cooldown2.end.restart\n"+\
                "\n"
            )
        if "eqlow2" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# Second Low Temperature Equilibration (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump eqlow2 all custom ${coords_freq} ${prefix}.eqlow2.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify eqlow2 sort id format float %20.10g\n"+\
                "fix eqlow2 all npt temp ${temp_eqlow2} ${temp_eqlow2} 100.0 iso ${press_eqlow2} ${press_eqlow2} 1000.0\n"+\
                "run ${nSteps_eqlow2}\n"+\
                "unfix eqlow2\n"+\
                "undump eqlow2\n"+\
                "write_restart ${prefix}.eqlow2.end.restart\n"+\
                "\n"
            )
        if "warmup2" in uncompleted_steps:
            f.write(
                "#===========================================================\n"+\
                "# Second Warm Up (NPT, Nose-Hoover)\n"+\
                "#===========================================================\n"+\
                "\n"+\
                "dump warmup2 all custom ${coords_freq} ${prefix}.warmup2.lammpstrj id mol type xu yu zu\n"+\
                "dump_modify warmup2 sort id format float %20.10g\n"+\
                "fix warmup2 all npt temp ${temp0_warmup2} ${tempf_warmup2} 100.0 iso ${press_warmup2} ${press_warmup2} 1000.0\n"+\
                "run ${nSteps_warmup2}\n"+\
                "unfix warmup2\n"+\
                "undump warmup2\n"+\
                "write_restart ${prefix}.warmup2.end.restart\n"+\
                "\n"
            )
        f.write(
            "#===========================================================\n"+\
            "# Clean and exit\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "unfix averages\n"
            )
    f.close()
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
            "python {}/Dilatometry/CG_dilatometry.py thermo2.avg -prefix {} -t_reshape {} -t_ramp {} -t_eqhigh {} -t_cool {} -t_eqlow {} -t_warm {} -t_eqhigh2 {} -t_cool2 {} -t_eqlow2 {} -t_warm2 {} -order '{}' > dila_analysis.log \n".format(script_directory, prefix, step_dict["reshape"]["steps"], step_dict["ramp"]["steps"], step_dict["eqhigh"]["steps"], step_dict["cooldown"]["steps"], step_dict["eqlow"]["steps"], step_dict["warmup"]["steps"], step_dict["eqhigh2"]["steps"], step_dict["cooldown2"]["steps"], step_dict["eqlow2"]["steps"], step_dict["warmup2"]["steps"], order)+\
            "echo \"End time: $(date)\""
            )
    f.close()
    sp.call("sbatch {}".format(slurm_file), shell=True)
    return
    
    
    # END OF SCRIPT. Below is parts of the old script before it was converted to dictionaries and for loops.

    # reshape_data=[]
    # ramp_data=[]
    # eqhigh_data=[]
    # cool_data=[]
    # eqlow_data=[]
    # warm_data=[]
    # for i in range(len(data_np)):
    #     # Divide data based on which regime it falls under.
    #     if data_np[i,0]<=reshape_steps:
    #         reshape_data.append(data_np[i,:])
    #     elif data_np[i,0]>reshape_steps and data_np[i,0]<= reshape_steps+ramp_steps:
    #         ramp_data.append(data_np[i,:])
    #     elif data_np[i,0]>reshape_steps+ramp_steps and data_np[i,0]<=reshape_steps+ramp_steps+eqhigh_steps:
    #         eqhigh_data.append(data_np[i,:])
    #     elif data_np[i,0]>reshape_steps+ramp_steps+eqhigh_steps and data_np[i,0]<=reshape_steps+ramp_steps+cool_steps+eqhigh_steps:
    #         cool_data.append(data_np[i,:])
    #     elif data_np[i,0]>reshape_steps+ramp_steps+cool_steps+eqhigh_steps and data_np[i,0]<=reshape_steps+ramp_steps+cool_steps+eqhigh_steps+eqlow_steps:
    #         eqlow_data.append(data_np[i,:])
    #     else:
    #         warm_data.append(data_np[i,:])
    # reshape_data=np.array(reshape_data)
    # ramp_data=np.array(ramp_data)
    # warm_data=np.array(warm_data)
    # eqlow_data=np.array(eqlow_data)
    # eqhigh_data=np.array(eqhigh_data)
    # cool_data=np.array(cool_data)
    # print(len(warm_data))
    # print(warm_data)
    # print(equil_data)
    # print(cool_data)
    # plt.figure()
    # color_min=0.4

    
    # # Generate colormaps, darker being for later time steps. Plot temp on x axis, rho on y axis.
    # # Separate into if statements to deterine if a regime has been filled. This way, if a simulation doesn't finish we still can have a graph, albeit an incomplete one.
    # all_colors=np.zeros((len(data_np), 4))

    # if eqhigh_data.size>0:
    #     eqhighCM = cm.get_cmap('Greens', len(eqhigh_data))
    #     eqhigh_time=(1-color_min)*(eqhigh_data[:,0]-min(eqhigh_data[:,0]))/(max(eqhigh_data[:,0]-min(eqhigh_data[:,0])))+color_min
    #     plt.scatter(eqhigh_data[:,1], eqhigh_data[:,2], c=eqhighCM(eqhigh_time), label="Equil at 500K", zorder=11)
    #     all_colors[len(reshape_data)+len(ramp_data):len(reshape_data)+len(ramp_data)+len(eqhigh_data) ,:]=eqhighCM(eqhigh_time)
    #     # print("normal color shape: ", np.array(eqhighCM(eqhigh_time)).shape)
    #     # print(all_colors.shape)
    # if cool_data.size>0:
    #     coolCM = cm.get_cmap('Blues', len(cool_data))
    #     cool_time=(1-color_min)*(cool_data[:,0]-min(cool_data[:,0]))/(max(cool_data[:,0]-min(cool_data[:,0])))+color_min
    #     plt.scatter(cool_data[:,1], cool_data[:,2], c=coolCM(cool_time), label="Cooling", zorder=4)
    #     all_colors[len(reshape_data)+len(ramp_data)+len(eqhigh_data):len(reshape_data)+len(ramp_data)+len(eqhigh_data)+len(cool_data) ,:]= coolCM(cool_time)
    #     # print(all_colors.shape)
    # if eqlow_data.size>0:
    #     eqlowCM = cm.get_cmap('Purples', len(eqlow_data))
    #     eqlow_time=(1-color_min)*(eqlow_data[:,0]-min(eqlow_data[:,0]))/(max(eqlow_data[:,0]-min(eqlow_data[:,0])))+color_min
    #     plt.scatter(eqlow_data[:,1], eqlow_data[:,2], c=eqlowCM(eqlow_time), label="Equil at 100K", zorder=10)
    #     all_colors[len(reshape_data)+len(ramp_data)+len(eqhigh_data)+len(cool_data):len(reshape_data)+len(ramp_data)+len(eqhigh_data)+len(cool_data)+len(eqlow_data),:]= eqlowCM(eqlow_time)
    #     # print(all_colors.shape)
    # if warm_data.size>0:
    #     warmCM = cm.get_cmap('Reds', len(warm_data))
    #     warm_time=(1-color_min)*warm_data[:,0]/max(warm_data[:,0])+color_min
    #     plt.scatter(warm_data[:,1], warm_data[:,2], c=warmCM(warm_time), label="Warming", zorder=6)
    #     all_colors[len(reshape_data)+len(ramp_data)+len(eqhigh_data)+len(cool_data)+len(eqlow_data):,:]= warmCM(warm_time)
    #     # print(all_colors.shape)
    #     # print(all_colors)
    # if reshape_data.size>0:
    #     reshapeCM = cm.get_cmap('Greys', len(reshape_data))
    #     reshape_time=(1-color_min)*(reshape_data[:,0]-min(reshape_data[:,0]))/(max(reshape_data[:,0]-min(reshape_data[:,0])))+color_min
    #     plt.scatter(reshape_data[:,1], reshape_data[:,2], c=reshapeCM(reshape_time), label="Reshape Box", zorder=1)
    #     all_colors[0:len(reshape_data) ,:]=reshapeCM(reshape_time)
    # if ramp_data.size>0:
    #     rampCM = cm.get_cmap('Greys', len(ramp_data))
    #     ramp_time=(1-color_min)*(ramp_data[:,0]-min(ramp_data[:,0]))/(max(ramp_data[:,0]-min(ramp_data[:,0])))+color_min
    #     plt.scatter(ramp_data[:,1], ramp_data[:,2], c=rampCM(ramp_time), label="Ramp up to 500K", zorder=2)
    #     all_colors[len(reshape_data):len(reshape_data)+len(ramp_data) ,:]=rampCM(ramp_time)

    # handles, labels = plt.gca().get_legend_handles_labels()

    # #specify order of items in legend
    # order = [4,5,0,1,2,3]

    # #add legend to plot
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
    # plt.savefig("Dilatometry_pre.pdf")
    # plt.close()
    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
