# !/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
home=str(Path.home())
import subprocess as sp
from scipy import stats
import matplotlib.pyplot as plt
#from matplotlib import cm
from matplotlib import colormaps
sys.path.append('{}/bin/CG_Crystal'.format(home))
sys.path.append('{}/bin/'.format(home))
from crys_rdf_gen import rdf_smoother
from Dylans_plot import plot_density
import json
import pickle

def main(argv):
    parser = argparse.ArgumentParser(description='''Enter description here                                                                                                                                           
    Input: 
                                                                                                                                                                                                      
    Output:                                                                                                                                                                                         
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('data_file', type=str, help='.json data file output from auto_genetic_martini.py to be used to plotting.')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG_files/', help='Output. Default: ')
        
    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    with open(args.data_file, 'rb') as f:
        data=pickle.load(f)
    #print(data)
    # Reassign dictionary components to appropriate variables
    betas=data["betas"]
    sampled_gens=data["sampled_gens"]
    bead_types=data["bead_types"]
    args=data["args"]
    dist_losses=data["dist_losses"]
    dist_long_loss=data["dist_long_loss"]
    dist_all_loss=data["dist_all_loss"]
    dist_short_loss=data["dist_short_loss"]
    losses=data["losses"]
    all_loss=data["all_loss"]
    short_loss=data["short_loss"]
    gen_range=data["gen_range"]
    all_betas=data["all_betas"]
    gens_for_params=data["gens_for_params"]
    switch_gen=data["switch_gen"]
    time_gen=data["time_gen"]
    data_dict=data["data_dict"]
    #current_dir=data["current_dir"]
    current_dir=os.getcwd()
    legend_list=data["legend_list"]
    best_dist=data["best_dist"]
    vars_dict=data["vars_dict"]
    regimes=[switch_gen, time_gen]
    best_params=data["best_params"]
    plot_folder=data["plot_folder"]

    # Plotting sampled generations
    static_maxes=np.zeros(len(bead_types))
    os.chdir(current_dir+"/gen_progress")
    #Blues = cm.get_cmap('Blues', len(sampled_gens))
    Blues = colormaps["Blues"]
    colormap=Blues(np.linspace(0.4,0.95,len(sampled_gens)))
    for c, v in enumerate(sampled_gens):
        os.chdir(v)
        for x in range(len(bead_types)):
            minned_rdf=np.loadtxt("crys_rdf_0_{}_{}.rdf".format(x+1, x+1), skiprows=1)
            plt.figure("Bead {}".format(x+1))
            if c==0:
                static_rdf=np.loadtxt("static_{}_{}.rdf".format(x+1, x+1), skiprows=1)
                if args.sr!=9999999.0:
                    static_rdf=rdf_smoother(static_rdf, args.sr, args.gaus_std)
                static_maxes[x]=max(static_rdf[:,1])
                plt.title("Sampled Generation rdfs, bead {}".format(x+1))
                plt.xlabel("Distance (Ang)")
                plt.ylabel("RDF Value")
                plt.plot(static_rdf[:,0], static_rdf[:,1]/static_maxes[x], color="green")    
            if args.sr!=9999999.0:
                minned_rdf=rdf_smoother(minned_rdf, args.sr, args.gaus_std)
            plt.plot(minned_rdf[:,0],minned_rdf[:,1]/max(minned_rdf[:,1]), color=colormap[c]) # REMOVED TO TEST EFFECT OF SCALING
            #plt.plot(minned_rdf[:,0],minned_rdf[:,1], color=colormap[c])
            if c == len(sampled_gens)-1:
                plt.legend(legend_list,bbox_to_anchor=(1,1), loc="upper left")
                plt.savefig("Progress_Bead_{}.pdf".format(x+1), bbox_inches="tight")
                plt.close("Bead {}".format(x+1))
                sp.call("mv Progress_Bead_{}.pdf ../../".format(x+1), shell=True)
        os.chdir("../")
    os.chdir("../")

# Plot losses across generations
    try:
        gen_range=range(0, args.gens+1)
        dist_losses=[dist_losses for i in gen_range]
        plt.plot(gen_range, dist_losses, 'b--') # Reference values
        plt.plot(gen_range, losses, 'b-')
        for r in regimes:
            if r:
                plt.axvline(x=r+0.5, color='grey', linestyle='--')
            else:
                pass
        plt.title('Best loss in each generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss Function')
        plt.savefig('loss.pdf')
        plt.close()
    except Exception as e:
        print("error plotting loss function: ")
        print(e)
        gen_range=range(0, args.gens+1)
    
# Prep short and long losses. Short being loss below a specified value, long being loss above that value. 
    # dist_long_loss=np.array([dist_all_loss-dist_short_loss for i in gen_range])
    # dist_all_loss=np.array([dist_all_loss for i in gen_range])
    # dist_short_loss=np.array([dist_short_loss for i in gen_range])
    
# Plotting Losses from individual beads, as well as short and long losses
    # all_loss=np.array(all_loss)
    # short_loss=np.array(short_loss)
    long_loss=all_loss-short_loss
    for x in range(len(bead_types)):
        plt.figure("Bead {} Losses".format(x+1))
        plt.plot(gen_range, all_loss[:,x], '-', color='tab:purple')
        plt.plot(gen_range, short_loss[:,x], 'b-')
        plt.plot(gen_range, long_loss[:,x], 'r-')
        plt.plot(gen_range, dist_all_loss[:,x], '--', color='tab:purple')      # Reference values
        plt.plot(gen_range, dist_short_loss[:,x], 'b--')    # Reference values
        plt.plot(gen_range, dist_long_loss[:,x], 'r--')     # Reference values
        for r in regimes:
            if r:
                plt.axvline(x=r+0.5, color='grey', linestyle='--')
            else:
                pass
        plt.title("Bead {} Losses".format(x+1))
        plt.xlabel("Generation")
        plt.ylabel("Loss Function")
        plt.legend(["Total Loss", "Loss below {} A".format(args.rdf_short), "Loss above {} A".format(args.rdf_short),"Ref Total Loss", "Ref Loss below {} A".format(args.rdf_short), "Ref Loss above {} A".format(args.rdf_short)])
        plt.savefig("Loss_Bead_{}.pdf".format(x+1))
        plt.close("Bead {} Losses".format(x+1))

# Plot all beta values through generations, showing heat map for common values with a line designating peaks of static rdfs found earlier.
    for g in range(0, args.gens+1):
        ind_start=g*args.specs
        ind_end=(g+1)*args.specs
        gens_for_params[ind_start:ind_end]=g
    best_dist_list=np.array([best_dist for i in range(0,args.gens+1)])
    x_vec=np.zeros((len(gens_for_params), vars_dict["param_num"]))
    y_vec=np.zeros((len(gens_for_params), vars_dict["param_num"]))
    z_vec=np.zeros((len(gens_for_params), vars_dict["param_num"]))
    x_vec_sort=np.zeros((len(gens_for_params), vars_dict["param_num"]))
    y_vec_sort=np.zeros((len(gens_for_params), vars_dict["param_num"]))
    z_vec_sort=np.zeros((len(gens_for_params), vars_dict["param_num"]))
    #all_betas=np.array(all_betas)
    #print("betas: ", all_betas[0])
    count=0
    for l in range(len(all_betas)):
        #print("length of {}:".format(l), len(all_betas[l]))
        if len(all_betas[l])!=50:
            count=count+1
    #print("Found nonconformers: {}".format(count))
    #print("Actual generations: {}".format(len(all_betas)-count))
    beta_idx_list=[]
    if all_betas.ndim==1:
        for l in range(len(all_betas)):
            if len(all_betas[l])!=args.specs:
                pass
            else:
                beta_idx_list.append(l)
        #print(beta_idx_list)
        all_betas=np.array(list(all_betas[beta_idx_list]))
        # counta=0
        # for m in range(len(all_betas)):
        #     for l in range(len(all_betas[m])):
        #         if len(all_betas[m][l])!=5:
        #             print("gen {}, spec {}".format(m, l))
        #             print("specs here: ", len(all_betas[m]))
        #             counta=counta+1
    # print("Found nonconformers: {}".format(counta))
    # print("Actual generations: {}".format(len(all_betas)-counta))  
    #all_betas=all_betas.reshape(args.gens+1, args.specs, len(bead_types))
    #print(all_betas)
    #print(all_betas.ndim, all_betas.shape)  
    #for x in range(len(bead_types)):
    gif_dict={"Sigma":{}, "Epsilon":{}}
    for x in range(vars_dict["param_num"]):
        plt.figure("All Bead Params {}".format(x+1))
        x_vec[:,x]=gens_for_params
        if all_betas.ndim==1:
            y_vec[:,x]=all_betas[:,:,x].flatten()
        else:
            y_vec[:,x]=all_betas[:,:,x].flatten()
        
        # This sets up a scatter colored by density of points to better illustrate point distribution
        xy=np.vstack([x_vec[:,x], y_vec[:,x]])
        z_vec[:,x]=stats.gaussian_kde(xy)(xy)
        idx=z_vec[:,x].argsort()
        x_vec_sort[:,x], y_vec_sort[:,x], z_vec_sort[:,x] = x_vec[idx,x], y_vec[idx,x], z_vec[idx,x]
        plt.scatter(x_vec_sort[:,x], y_vec_sort[:,x],c=z_vec_sort[:,x])
        for r in regimes:
            if r:
                plt.axvline(x=r+0.5, color='grey', linestyle='--')
            else:
                pass
        
        plt.xlabel("Generation")
        plt.ylabel("Parameter Value")
        if x>=len(bead_types):
            plt.title("Bead {} All Epsilons Tested".format(x-len(bead_types)+1))
            plt.savefig("All_Eps_Bead_{}.pdf".format(x-len(bead_types)+1))
            gif_dict["Epsilon"].update({"x_vec":x_vec[:, len(bead_types):], "x_vec_sort":x_vec_sort[:, len(bead_types):],"y_vec":x_vec[:, len(bead_types):], "y_vec_sort":y_vec_sort[:, len(bead_types):],"z_vec":y_vec[:, len(bead_types):], "z_vec_sort":z_vec_sort[:, len(bead_types):]})
        else:
            plt.plot(gen_range, best_dist_list[:,x], 'k--')
            plt.title("Bead {} All Sigmas Tested".format(x+1))
            plt.savefig("All_Sigmas_Bead_{}.pdf".format(x+1))
            gif_dict["Sigma"].update({"x_vec":x_vec, "x_vec_sort":x_vec_sort,"x_vec":x_vec, "y_vec_sort":y_vec_sort,"y_vec":y_vec, "z_vec_sort":z_vec_sort})
        plt.close("All Bead Params {}".format(x+1))
# Plotting loss and parameter exploration over each individual generation for creation of progress gif.
    gif_list=["Sigma", "Epsilon", "Sigma_noL", "Epsilon_noL"] #, "sigma eps"]  # Create multiple sets of plots for sigma, epsilon, and both.
    gif_dict["Sigma"].update({"lims":[args.xmin, args.xmax]})
    gif_dict["Epsilon"].update({"lims":[args.epmin, args.epmax]})
    for gif in gif_list:
        mult = len(gif.split()) # This tells us how much to multiply the len(bead_types) by (2 if we are doing both)
        noL = gif.split("_")
        L_flag = 0
        if noL[-1] == "noL":
            val_label = gif
            gif = noL[0]
            L_flag = 1
        else:
            val_label=gif
        for g in range(0, args.gens+1):
            # by lumping this prep work inside the for loop, we erase the previous plot so our new plots don't have so much overlap.
            giffig, axs=plt.subplots(len(bead_types)+(2-mult-L_flag), mult, sharex='all', figsize=(13.0,6.5)) # This means that for plots where we have both value types, we will have two columns and will not plot the loss function with them.
            giffig.suptitle("Algorithm Exploration", fontsize=16)
            plt.xlim(0,args.gens)
            plt.xlabel("Generation")

            #print("label space: ", label_space)
            if L_flag == 0:
                label_space=(1-1/(len(bead_types)+1))/2+1/(len(bead_types)+1)
                giffig.text(0.08, 1-label_space, '{} Value'.format(gif), ha='center', va='center', rotation='vertical')
                axs[0].set_ylim(max(int(min(losses)*0.9-1),0), int(max([max(losses),dist_losses[0]])*1.1+1))    
                axs[0].plot(gen_range[0:g+1],losses[0:g+1],'b-') # Plotting loss function on top.
                axs[0].plot(gen_range[0:g+1], dist_losses[0:g+1], 'b--')
                axs[0].set_ylabel("Loss Function")
            else: # Skip loss function sometimes
                label_space = 0.5 
                giffig.text(0.08, 1-label_space, '{} Value'.format(gif), ha='center', va='center', rotation='vertical')       
            # Adds grey vertical line if there is a switch in the loss function. Only plots for generations over switch gen.
            # if switch_gen: 
            #     if g>switch_gen:
            #         axs[0].axvline(x=switch_gen+0.5, color='grey', linestyle='--')
            for r in regimes:
                if r:
                    if g>r:
                        axs[0].axvline(x=r+0.5, color='grey', linestyle='--')
                    else:
                        pass
            for s in range(1, len(bead_types)+1):
                gidx=np.where(gif_dict[gif]["x_vec_sort"][:,s-1]<=g)    # Find indices in current bead type belonging to current or previous generation
                axs[s-L_flag].scatter(gif_dict[gif]["x_vec_sort"][gidx,s-1],gif_dict[gif]["y_vec_sort"][gidx,s-1],c=gif_dict[gif]["z_vec_sort"][gidx,s-1]) # Plot from sorted data to hopefully preserve color stacking...
                axs[s-L_flag].plot(gen_range[0:g+1], best_dist_list[0:g+1, s-1], 'k--')
                #axs[s].set_ylabel("Parameter Value")
                axs[s-L_flag].set_ylim(gif_dict[gif]["lims"][0], gif_dict[gif]["lims"][1]) # Limit axes to parameter bounds.
                # Adds grey vertical line if there is a switch in the loss function. Only plots for generations over switch gen.
                for r in regimes:
                    if r:
                        if g>r:
                            axs[s-L_flag].axvline(x=r+0.5, color='grey', linestyle='--')
                        else:
                            pass
            if g<10:
                plt.savefig("gif_{}_plots_0{}.png".format(val_label, g), bbox_inches="tight")
                plt.savefig("gif_{}_plots_0{}.pdf".format(val_label, g), bbox_inches="tight")
            else:
                plt.savefig("gif_{}_plots_{}.png".format(val_label, g), bbox_inches="tight")
                plt.savefig("gif_{}_plots_{}.pdf".format(val_label, g), bbox_inches="tight")
            plt.close()
    

    # Generate a plot of all best parameters with each generation
    # Rainbow = cm.get_cmap('rainbow', len(bead_types))
    Rainbow = colormaps["rainbow"]
    RainCM=Rainbow(np.linspace(0.05,0.95,len(bead_types)))
    try:
        betasnp=np.array(betas)
        gen_range=range(0, args.gens+1)
        legend_list=[]
        for p in range(len(bead_types)):
            plt.plot(gen_range, betasnp[:,p],color=RainCM[p])
            plt.plot(gen_range, best_dist_list[:,p],'--',color=RainCM[p])
            legend_list.append("Bead {}".format(p+1))
            legend_list.append("Ref Bead {}".format(p+1))
        plt.title('Best parameters in each generation')
        plt.xlabel('Generation')
        plt.ylabel('Parameter Value')
        plt.legend(legend_list, bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig('params.pdf', bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("error plotting parameters: ")
        print(e)
    #folder=current_dir.split('/')[0]
    os.chdir(current_dir)

# Plot thermodynamic data
    compare_stats={}
    if time_gen:
        normal_gens=time_gen
    else:
        normal_gens=args.gens
    #time_ind=max(max(np.where(gen_vec==normal_gens))) # Set up time indices if needed

    for t in ["thermo_delta", "thermo_scale"]:
        all_thermo=np.zeros((args.gens+1, args.specs))
        for g in range(0, args.gens+1):
            all_thermo[g,:]=data_dict[str(g)][t] # Compile data from data dictionary into single 2-D array.
        gen_vec=gens_for_params
        time_ind=max(max(np.where(gen_vec==normal_gens))) # Set up time indices if needed
        thermo_means=np.mean(all_thermo, axis=1)
        #print("means: ", thermo_means)
        thermo_vec=all_thermo.flatten()
        # This sets up a scatter colored by density of points to better illustrate point distribution
        # P_colors=cm.get_cmap('Blues', len(gen_vec))
        # V_colors=cm.get_cmap('Purples', len(gen_vec))
        P_colors = colormaps["Blues"]
        V_colors = colormaps["Purples"]
        color_min=0.2                       # Color min defines a minimum value for the color maps that linearly scales the values between color_min and 1 so colors don't get too faint. 
        plt.figure(t)
        if switch_gen:
            switch_ind=max(max(np.where(gen_vec==switch_gen)))
            #print("gen_vec: ", gen_vec)
            #print("switch_ind: ", switch_ind)
            combP=np.vstack([gen_vec[0:switch_ind], thermo_vec[0:switch_ind]])
            z_thermoP=stats.gaussian_kde(combP)(combP)
            idx_thermoP=z_thermoP.argsort()
            gen_vec_sortP, thermo_vec_sortP, z_thermo_sortP = gen_vec[idx_thermoP], thermo_vec[idx_thermoP], z_thermoP[idx_thermoP]            
            combV=np.vstack([gen_vec[switch_ind+1:time_ind], thermo_vec[switch_ind+1:time_ind]])
            #print("combV", combV)
            z_thermoV=stats.gaussian_kde(combV)(combV)
            idx_thermoV=z_thermoV.argsort()+len(gen_vec[0:switch_ind])+1
            gen_vec_sortV, thermo_vec_sortV, z_thermo_sortV = gen_vec[idx_thermoV], thermo_vec[idx_thermoV], z_thermoV[idx_thermoV-len(gen_vec[0:switch_ind])-1]
            #print("Before: ", z_thermo_sortP)
            z_thermo_sortP=(1-color_min)*z_thermo_sortP+color_min
            z_thermo_sortV=(1-color_min)*z_thermo_sortV+color_min
            #print("After: ", z_thermo_sortP)
            # P_colors=cm.get_cmap('Blues', len(z_thermo_sortP))
            # V_colors=cm.get_cmap('Purples', len(z_thermo_sortV))
            compare_stats[t]={"Pressure":[np.mean(thermo_vec[idx_thermoP]), np.std(thermo_vec[idx_thermoP])], "Volume":[np.mean(thermo_vec[idx_thermoV]), np.std(thermo_vec[idx_thermoV])]}
            dl_val=0.8
            if t=="thermo_delta":
                fig, ax1=plt.subplots(num=t)
                plot_1 = ax1.scatter(gen_vec_sortP, thermo_vec_sortP, c=P_colors(z_thermo_sortP), label="Pressure")
                ax1.plot(range(0, switch_gen+1), thermo_means[0:switch_gen+1], color=P_colors(dl_val), linestyle='--')
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('$\Delta$P')
                print("z P", z_thermo_sortP)
                print("P vals", thermo_vec_sortP)
                ax2=ax1.twinx()
                plot_2 = ax2.scatter(gen_vec_sortV, thermo_vec_sortV, c=V_colors(z_thermo_sortV), label="Volume")
                ax2.plot(range(switch_gen+1, normal_gens+1), thermo_means[switch_gen+1:normal_gens+1], color=V_colors(dl_val), linestyle='--')
                ax2.set_ylabel("$\Delta$V")
                lns = [plot_1, plot_2]
                labels=[l.get_label() for l in lns]
                plt.legend(lns,labels,loc=0)
                plt.title("Thermo Delta")
                plt.sca(ax1)
            elif t=="thermo_scale":
                plt.scatter(gen_vec_sortP, thermo_vec_sortP, c=P_colors(z_thermo_sortP), label="Pressure")
                plt.scatter(gen_vec_sortV, thermo_vec_sortV, c=V_colors(z_thermo_sortV), label="Volume")
                plt.plot(range(0, switch_gen+1), thermo_means[0:switch_gen+1], color=P_colors(dl_val), linestyle='--')
                plt.plot(range(switch_gen+1, normal_gens+1), thermo_means[switch_gen+1:normal_gens+1], color=V_colors(dl_val), linestyle='--')
                plt.title("Thermo Scale")
                plt.ylabel("$\Theta$")
                plt.xlabel("Generation")
                plt.legend()
            #plt.axvline(x=switch_gen+0.5, color='grey', linestyle='--')
        else:
            comb=np.vstack([gen_vec[0:time_ind], thermo_vec[0:time_ind]])
            z_thermo=stats.gaussian_kde(comb)(comb)
            idx_thermo=z_thermo.argsort()
            gen_vec_sort, thermo_vec_sort, z_thermo_sort = gen_vec[idx_thermo], thermo_vec[idx_thermo], z_thermo[idx_thermo]
            compare_stats[t]={"Pressure":[np.mean(thermo_vec[idx_thermo]), np.std(thermo_vec[idx_thermo])]}
            z_thermo_sort=(1-color_min)*z_thermo_sort+color_min
            if args.p_scale:
                thermo_color_list=P_colors(z_thermo_sort)
                if t=="thermo_delta":
                    plt.title("Pressure Delta")
                    plt.ylabel("$\Delta$P")
                elif t=="thermo_scale":
                    plt.title("Pressure Scale")
                    plt.ylabel("$\Theta$")
                plt.plot(thermo_means, range(0, normal_gens+1), color=P_colors(dl_val), linestyle='--')
            elif args.v_scale:
                thermo_color_list=V_colors(z_thermo_sort)
                if t=="thermo_delta":
                    plt.title("Volume Delta")
                    plt.ylabel("$\Delta$V")
                elif t=="thermo_scale":
                    plt.title("Volume Scale")
                    plt.ylabel("$\Theta$")
                plt.plot(thermo_means, range(0, normal_gens+1), color=V_colors(dl_val), linestyle='--')
            plt.scatter(gen_vec_sort, thermo_vec_sort,c=thermo_color_list)

        for r in regimes:
            if r:
                if g>r:
                    plt.axvline(x=r+0.5, color='grey', linestyle='--')
                else:
                    pass
        if time_gen:   
            plot_density(gen_vec[time_ind+1:], thermo_vec[time_ind+1:], "{}_change".format(t), ["", "", ""], color_map="Blues", color_min=0.2, save_it=False, close_it=False)
            plt.plot(range(normal_gens+1, args.gens+1), thermo_means[normal_gens+1:args.gens+1], color=P_colors(dl_val), linestyle='--')
        # plt.xlabel("Generation")
        # plt.savefig("{}_change.pdf".format(t))
        plt.savefig("{}_change.pdf".format(t), bbox_inches='tight')
        plt.close(t)

# Plot rdf fraction in loss function if appropriate loss function chosen.
    if args.loss_func=="frac" or args.loss_func=="add":
        if args.order:
            frac_keys=["rdf", "thermo", "order"]
            frac_colors=["Blues", "Greens", "autumn"]
        else:
            frac_keys=["rdf", "thermo"]
            frac_colors=["Blues", "Greens"]
        frac_dict={}
        for fc, fk in zip(frac_colors, frac_keys):
            frac_dict.update({fk:{"fracs":np.zeros((args.gens+1, args.specs)), "aves":[]}})
            for g in range(0, args.gens+1):
                frac_dict[fk]["fracs"][g,:]=data_dict[str(g)]["{}_frac".format(fk)] # Compile data from data dictionary into single 2-D array.
                frac_dict[fk]["aves"].append(np.mean(data_dict[str(g)]["{}_frac".format(fk)])) # Get average value of the fractions for each generation.
            frac_dict[fk].update({"gen_vec":gens_for_params})
            frac_dict[fk]["fracs"]=frac_dict[fk]["fracs"].flatten()
            # This sets up a scatter colored by density of points to better illustrate point distribution
            frac_dict[fk]["colors"]=colormaps[fc]
            plt.figure("fracs")
            frac_dict[fk].update({"combf":np.vstack([frac_dict[fk]["gen_vec"], frac_dict[fk]["fracs"]])})
            frac_dict[fk].update({"z":stats.gaussian_kde(frac_dict[fk]["combf"])(frac_dict[fk]["combf"])})
            frac_dict[fk].update({"idx_frac":frac_dict[fk]["z"].argsort()})
            frac_dict[fk].update({"gv_sort":frac_dict[fk]["gen_vec"][frac_dict[fk]["idx_frac"]], "fv_sort":frac_dict[fk]["fracs"][frac_dict[fk]["idx_frac"]], "z_sort":frac_dict[fk]["z"][frac_dict[fk]["idx_frac"]]})
            frac_dict[fk]["z_sort"]=(1-color_min)*frac_dict[fk]["z_sort"]+color_min
            frac_dict[fk]["color_list"]=frac_dict[fk]["colors"](frac_dict[fk]["z_sort"])
            plt.scatter(frac_dict[fk]["gv_sort"], frac_dict[fk]["fv_sort"],c=frac_dict[fk]["color_list"], label=fk)
            plt.plot(list(range(0, args.gens+1)), frac_dict[fk]["aves"], c=frac_dict[fk]["colors"](dl_val), ls='--')  # Plot average line too

        plt.title("Loss Function Fractions")
        plt.ylabel("Fractional values")
        plt.ylim(0,1)
        for r in regimes:
            if r:
                plt.axvline(x=r+0.5, color='grey', linestyle='--')
            else:
                pass
        plt.xlabel("Generation")
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig("frac.pdf", bbox_inches='tight')
        plt.close("fracs")
    
    if args.epsilon:
        best_sigmas=best_params[0:len(bead_types)]
        best_eps=best_params[len(bead_types):]
        print(best_sigmas)
        print(best_eps)
        plt.figure()
        plt.title("LJ Potential {}".format(plot_folder))
        plt.ylabel("Potential Value (kcal/mol)")
        plt.xlabel("Distance (Angstroms)")
        r_vals=np.linspace(0.01, 10, 1000)
        for i in range(len(best_sigmas)):
            LJ_vals=4*best_eps[i]*((best_sigmas[i]/r_vals)**12-(best_sigmas[i]/r_vals)**6)
            plt.plot(r_vals, LJ_vals, c=RainCM[len(best_sigmas)-1-i], label='Bead {}'.format(i+1))
        plt.legend()
        plt.xlim(args.xmin-1, args.xmax+1)
        plt.ylim((args.epmax+1)*-1.0, 5)
        plt.savefig("LJ_plots.pdf")
        plt.close()

# Plot rdf losses
    all_rdfs=np.zeros((args.gens+1, args.specs))
    for g in range(0, args.gens+1):
        all_rdfs[g,:]=np.sum(data_dict[str(g)]["raw_rdf"], axis=1) # Compile data from data dictionary into single 2-D array.
        #print(data_dict[str(g)]["raw_rdf"])
    rdf_vec=all_rdfs.flatten()
    #print("rdf_vec: ", rdf_vec)
    plot_density(gens_for_params, rdf_vec, "rdf_losses", ["RDF Losses", "gen", "rdf loss"], color_map="Reds", color_min=0.3)
# Plot Autocorrelation
    if args.order:
        all_os=np.zeros((args.gens+1, args.specs))
        for g in range(0, args.gens+1):
            print("order vals, gen {}: ".format(g), data_dict[str(g)]["order"])
            all_os[g,:]=data_dict[str(g)]["order"] # Compile data from data dictionary into single 2-D array.
        Os_vec=all_os.flatten()
        print("Os_vec: ", Os_vec)
        no_zero_Os=[]
        no_zero_gens=[]
        for v, vals in enumerate(Os_vec):
            if vals == 0.:
                pass
            else:
                no_zero_Os.append(vals)
                no_zero_gens.append(gens_for_params[v])
        no_zero_Os=np.array(no_zero_Os)
        no_zero_gens=np.array(no_zero_gens)
        print("nz Os", no_zero_Os)
        print("nz g", no_zero_gens)
        plot_density(no_zero_gens, no_zero_Os, "order_vals", ["Autocorrelation", "gen", "Value"], color_map="cividis", color_min=0.3, save_it=False, close_it=False)
        # plot_density(gens_for_params, Os_vec, "order_vals", ["Autocorrelation", "gen", "Value"], color_map="cividis", color_min=0.3, save_it=False, close_it=False)
        plt.ylim(0.0, 1.0)
        plt.savefig("order_vals.pdf", bbox_inches='tight')
        plt.close()

    sp.call("mkdir {}".format(plot_folder), shell=True)
    sp.call("mv *.pdf {}".format(plot_folder), shell=True)
    sp.call("mv *.png {}".format(plot_folder), shell=True)
    os.chdir("{}".format(plot_folder))
    for gif in gif_list:
        sp.call("mkdir gif_{}_plots".format(gif), shell=True)
        sp.call("mv gif_{}_plots_* gif_{}_plots".format(gif, gif), shell=True)
    os.chdir("../")
# Stats
    print("Statistics, format [mean, std]:\n")
    for k in compare_stats.keys():
        print("{}: \n".format(k), compare_stats[k])


def plot_ac(ac_file, plot=True):
    frames=[]
    ac_vals=[]
    stdevs=[]
    header_count = 0
    frame_mod = 0.0
    with open(ac_file, 'r') as ac:
        for lines in ac:
            fields=lines.split()
            if len(fields)==0:
                pass
            elif fields[0]=="frame":
                if header_count == 0:
                    pass
                else: # If you see a new header (ie, multiple sets of files have been appended), modify the frame number by the previous frame number
                    frame_mod = frames[-1] # its okay to have frames of the first and last frames be the same index because they are of the same snap shot of the trajectory.
                header_count += 1
            elif fields[1]=="nan":
                break
            else:
                frames.append(float(fields[0])+frame_mod)
                ac_vals.append(float(fields[1]))
                stdevs.append(float(fields[2]))
    ac.close()
    if plot:
        plt.figure()
        frames=np.array(frames)
        ac_vals=np.array(ac_vals)
        stdevs=np.array(stdevs)
        plt.errorbar(frames, ac_vals, yerr=stdevs, ecolor="#b8cdf9", c="#104dc9", )
        plt.title("Order Parameter During Simulation")
        plt.xlabel("Frames (ps)")
        plt.ylabel("Order parameter value")
        plt.ylim(0, 1)
        plt.savefig(ac_file.split('/')[-1].split('.')[0]+".pdf")
        plt.close()
    return frames, ac_vals, stdevs
    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])