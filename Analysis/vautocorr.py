
import sys,os,argparse
from scipy.linalg import norm
import scipy.stats
import scipy.spatial
import scipy.spatial.distance
import scipy.optimize
import scipy.special
import numpy as np
import matplotlib.pyplot as mpl
import time
import math


# Add TAFFY Lib to path
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
from pathlib import Path
home=str(Path.home())
from Mol_ops import *
# from file_parsers import *
# from id_types import *
import random

def main(argv):

    parser = argparse.ArgumentParser(description='Reads in a lammps datafile, trajectory file, and a taffi map file and parses a user-specified self-averaged autocorrelation function.')

    # Input arguments                                                                                                  
    parser.add_argument('-map',     dest="map_file",   default=False,     help = 'Name of the mapfile that should be used for determining atomtypes')
    parser.add_argument('-traj',    dest="traj_file",  default=False,     help = 'Name of the trajectory file that should be used for determining atomtypes')
    parser.add_argument('-o',       dest="o_folder",   default="results", help = 'Name of the results folder (default: results)')
    parser.add_argument('-f_start', dest="f_start",    default=0,         help = 'Index of the frame to start on (0-indexing, inclusive, with the first frame being 0 irrespective of timestamp) (default: 0)')
    parser.add_argument('-f_end',   dest="f_end",      default=1000000,   help = 'Index of the frame to end on (0-indexing, inclusive, with the first frame being 0 irrespective of timestamp) (default: 1E6)')
    parser.add_argument('-f_every', dest="f_every",    default=1,         help = 'frequency of frames to parse (default: 1, every frame is parsed)')
    parser.add_argument('-log',     dest="log",        default=None,      help = 'Logarithmically spaced autocorrelation periods. Give the number of periods desired. Will overwrite f_every. If nothing is given, calculation will be performed in standard linear fashion')
    parser.add_argument('-log_frames', dest="num_frames", default=200,    help = "Total number of frames to parse when looking logarithmically (note that the actual number of frames parsed will be less)")
    parser.add_argument('--list',   dest="list_flag",  default=False,     action="store_const", const=True, help = 'Lists the keys for all modes')
    parser.add_argument('--modes',  dest="parse_list", default=None,      help = "Holds the list of modes whose autocorrelation behavior is to be tracked. Input 'all' to get all vectors between atoms, bonded or not.")
    parser.add_argument('--omit',   dest="omit",       default='',        help = "Modes involving any atomtypes, atom ids, or molids in this list are omitted from the parse. atomtypes are prefixed with a 't' and atomids are " +\
                                                                                 "prefixed with an 'a'. Hyphenated terms are automatically expanded. For example 't0-5 a0 a2 a1000-2000 m0' would omit modes involving types "+\
                                                                                 "0-5, atom ids 0 2 and 1000-2000, and any atoms in molecule 0. (default: None)")
    parser.add_argument('--velocity', dest="vel",       default=False,    action="store_const", const=True, help = "If this flag is checked, velocity autocorrelation will be done in addition to rotational autocorrelation")
    parser.add_argument('--rem_zeros', dest="shave_opt", default=False,   action="store_const", const=True, help = 'When this option is supplied only the non-zero entries of the histogram are saved. (default: off)')
    parser.add_argument('--abs',       dest="abs_opt",   default=False,   action="store_const", const=True, help = 'When this option is supplied only the absolute value of the dihedral is used for the histogram. (default: off)')
    parser.add_argument('-unwrap',     dest="unwrap_opt",   default=True,   help = 'This option controls whether the coordinates are unwrapped for parsing the histograms. True and False are allowed options (default: True)')
    parser.add_argument('--no_corr',    dest="corr_opt",   default=True,   action="store_const", const=False, help = 'This flag disables the  autocorrelation calculation. By default it is performed on each mode/mol') 
    parser.add_argument('--thetaphi',   dest="thetaphi",    default=False,  action="store_const",   const=True, help="If this flag is triggered, the parser will calculate a theta-phi plot as a function of time")
    parser.add_argument('--no_ac',   dest="autocorr",   default=True,  action="store_const",        const=False, help="If this flag is triggered, the parser will not calculated autocorrelations")
    parser.add_argument('--no_xc', dest="crosscorr", default=True, action="store_const", const=False,          help="If this flag is triggered, the parser will not calculate cross-correlation.")
    parser.add_argument('--no_vel_xc', dest="vel_crosscorr", default=True, action="store_const", const=False,  help="If this flag is triggered, the parser will not calculate velocity cross-correlation.")
    parser.add_argument('--pairwise', dest="pairwise", default=False, action="store_const", const=True, help="If this flag is triggered, the script will compute the full correlator pairwise instead of with the shortcut method. WARNING: this is N-times slower! Only use if you want deeper analysis")
    parser.add_argument('-r', dest="radius", default=None, help="If a value is given, only pairs within this radius will contribute to the full correlation. Requires pairwise flag")
    parser.add_argument('--histogram', dest="histogram", default=False, action="store_const", const=True, help="If this flag is triggered, the script will compute the histogram for the contributions of the pairs to the full correlation. Requires the pairwise flag to be triggered.")
    parser.add_argument('--double_hist', dest="double_hist", default=False, action="store_const", const=True, help="If this flag is triggered, will produce a double histogram of pair contributions and distance. Requires pairwise flag")
    parser.add_argument('--verbose', dest="verbose", default=False, action="store_const", const=True, help="If this flag is triggered, the script will print out each molecule and frame it is working on.")
    parser.add_argument('--no_vel_norm', dest="no_vel_norm", default=False, action="store_const", const=True, help="If this flag is triggered, the script will not normalize velocities before computing the cross-correlation.")
    parser.add_argument('--overwrite', dest="overwrite", default=False, action="store_const", const=True, help="If this flag is triggered, the script will overwrite previously stored data if there is already a folder with this output")
    parser.add_argument('-velfcbox', dest="vfcbox", default=None, help="Will compute velocity full correlator with subsets in box form. Provide number of subsets (perfect cube is best)")
    parser.add_argument('-velfcradius', dest="vfcrad", default=None, help="Will compute velocity full correlator with subsets in radius form. Provide radius")
    parser.add_argument('-vel_atom', dest="vel_atom", default=None, help="Give 'x y', where x is number of different atom types, y is single atom type you want to be investigated")
    parser.add_argument('--P2', dest="second_leg", default=False, action="store_const", const=True, help="If this flag is triggered, script will compute 2nd legendre polynomial for autocorrelation")
    parser.add_argument('-U_dist', dest="u_dist", default=False, help="If this flag is triggered, will compute the U distribution. Provide the number of atoms in a unit cell")
    parser.add_argument('--lindemann', dest="lind", default=False,action="store_const", const=True,help="If this flag is triggered, will compute the the Lindemann index for the system.")
    parser.add_argument('--high_eng_traj', dest="high_eng_traj", default=False,action="store_const", const=True,help="If this flag is triggered, the script will compute the spatial distribution of the 10% highest energy molecules and write this to a new trajectory file")
    parser.add_argument('--ang_mom', dest="ang_mom", default=False, action="store_const", const=True, help="If this flag is triggered, the script will compute the angular momentum full correlator.")
    parser.add_argument('--inertia', dest="inertia", default=False, action="store_const", const=True, help="This flag activates the inertia parser. Requires --vel and --ang_mom to function")
    parser.add_argument('-cluster_size', dest="cluster_size", default=14*3, help="Number of atoms to include in the chunk")
    parser.add_argument('--force', dest="force", default=False, action="store_const", const=True, help="If this flag is triggered, the script will compute force distribution on the given trajectory.")
    parser.add_argument('-molecule_clust', dest="molecule_clust", default=None, help="If this flag is triggered, molecules will be considered clusters. Provide the number of atoms in a molecule.")
    parser.add_argument('--random', dest="rand", default=False, action="store_const", const=True, help="If this flag is triggered, the script will compute force distribution on the given trajectory with random clusters instead of nearest neighbor clusters.")
    parser.add_argument('-species', dest="species", default=None, help="Quick fix for handling different species")
    parser.add_argument('--no_force_ac', dest="force_ac", default=True, action="store_const", const=False, help="If this flag is triggered, the autocorrelation of the force will not be computed.")
    parser.add_argument('--individual', dest="individual", default=False, action="store_const", const=True, help="If this flag is triggered, the parser will print individual autocorrelations and non-time-averaged rotations for individual molecules. For best results, don't use log spacing.")
    parser.add_argument('--time_hist', dest="time_hist", default=False, action="store_const", const=True, help="If this flag is triggered, the parser will print out histograms of costheta as a function of time")
    parser.add_argument('--origin_only', dest="origin_only", default=False, action="store_const", const=True, help="If this flag is triggered, the parser will not time average, and will only compute acs from timepoint 0")
    parser.add_argument('--scalord', dest="scalord", default=False, action="store_const", const=True, help="If this flag is triggered, the parser will calculate the scalar order parameter using the provided vector(s)")
    parser.add_argument('-legendres', dest="legendres", default=2, help="Number of diferent legendre polynomials of the data to compute. Default=2")

    # Check the validitity of the user supplied inputs.
    args = check_validity(parser.parse_args())
    print(args)
    # Parse the modes and molecules
    objects = parse_objects(args.map_file)
    # for i in objects:
    #     print "{}:\n{}\n".format(i,objects[i])

    if args.vel_atom is not None:
        vel_atom = args.vel_atom.split()
    else:
        vel_atom = None

    # Print summary if requested by the user
    if args.list_flag is True:
        summarize(objects,args.traj_file)
        quit()
    else:
        
        # Grab the modes to be parsed
        parse_dict = process_request(args.parse_list,objects,args.omit)

        # Remove omitted modes from the parse_dict
        parse_dict = remove_omitted(parse_dict,objects,args.omit)

        # Check that data won't be overwritten
        if not args.overwrite:
            for i in parse_dict.keys():
                if i != "velocity" and i != "vel_loc" and i != "energy" and i != "force" and i != "eng_loc" and i != "force_loc":
                    for j in parse_dict[i].keys():
                        if os.path.isdir(("{}/{}/{}".format(args.o_folder,i,"_".join([ str(_) for _ in j ])))):
                            print(os.getcwd())
                            print("j", j, type(j))
                            print("i", i, type(i))
                            print("_".join(j))
                            print("ERROR in autocorr_calc: folder {} already exists. Exiting to avoid overwriting data.".format("{}/{}/{}".format(args.o_folder,i,"_".join(j))))
                            quit()

        # Determine frames to be parsed for log flag
        log_frames = None
        deltas = None
        if args.log is not None:
            spacing = np.logspace(0,np.log10(int(args.f_end)-int(args.f_start)), int(args.log), base=10)
            deltas = np.unique(np.around(spacing).astype(int)) # Converts spacing to int and removes duplicate entries
            deltas = np.append(0, deltas)

            #deltas = np.array([0,unique(around(spacing).astype(int))])

            log_frames = set([i+int(args.f_start) for i in deltas]) # shift log frames up by f_start to get proper frames
            log_frames.add(args.f_start) # Probably extraneous

            n_frames = args.num_frames
            n_shifts = int(math.ceil(float(n_frames) / float(args.log)))
            shifts = np.linspace(args.f_start, args.f_end, n_shifts) # dtype=int
            shifts = shifts.astype(int) #convert all shifts to integer number of frames


            # for i in deltas:
            #     for j in deltas:
            #         if i+j < args.f_end:
            #             log_frames.add(i+j) # Have same number of samples for each delta as we have deltas (where possible). Possibly change this to give more statistics for shorter deltas
            for i in deltas:
                for j in shifts:
                    if i+j < args.f_end:
                        log_frames.add(int(i+j))

        if args.high_eng_traj is not False:
            os.mkdir("{}/{}".format(args.o_folder, "energy"))

        box_t = []


        ########################
        # Beginning of trajectory reading
        ########################
        # Loop over configurations and calculate the user-specified properties
        count = 0
        for geo,ids,types,box,vels,engs,forces in frame_generator(args.traj_file,args.f_start,args.f_end,args.f_every,unwrap=args.unwrap_opt,adj_list=objects["adj_list"], vel=args.vel, log_frames=log_frames, u_dist=args.u_dist != False, force=args.force):
            if args.verbose:
                print("working on frame {}...".format(count))
            count += 1

            box_t.append(box)

            # # Parse CoM positions
            # mol_counter = -1
            # for count_i, i in enumerate(geo):
            #     if count_i%N_atoms == 0: #N_atoms is the number of atoms per molecule
            #         com[mol_counter] /= MM #MM is molar mass
            #         mol_counter += 1
            #     com[mol_counter] += mass[count_i]*i # mass-weight the position for each atom

            # Calculate inter-molecular modes from CoM
            # for i in com:
            #     for j in com:
            #         dist[i][j] = dist(i,j)
            #         dist[j][i] = dist(i,j)
            # for n in range(n_modes):
            #     r = box_length/2/(n+1)


            # Parse bond lengths
            for i in parse_dict["bonds"].keys():
                for j in parse_dict["bonds"][i].keys():
                    parse_dict["bonds"][i][j] += [ norm(geo[j[0]] - geo[j[1]]) ]

            # Parse angles
            for i in parse_dict["angles"].keys():
                for j in parse_dict["angles"][i].keys():
                    parse_dict["angles"][i][j] += [ calc_angle(geo[j[0]],geo[j[1]],geo[j[2]])*180.0/pi ]

            # Parse dihedrals
            for i in parse_dict["dihedrals"].keys():
                for j in parse_dict["dihedrals"][i].keys():
                    parse_dict["dihedrals"][i][j] += [ calc_dihedral(geo[j[0]],geo[j[1]],geo[j[2]],geo[j[3]])*180.0/pi ]

            # Parse impropers
            for i in parse_dict["impropers"].keys():
                for j in parse_dict["impropers"][i].keys():
                    parse_dict["impropers"][i][j] += [ calc_improper(geo[j[0]],geo[j[1]],geo[j[2]],geo[j[3]])*180.0/pi ]

            # Parse orientation vectors
            for i in parse_dict["vectors"].keys():
                if i == "all":
                    parse_dict["vectors"][i] = np.zeros(3)
                    for count_j, j in enumerate(geo):
                        for count_k, k in enumerate(geo):
                            if count_j < count_k: # only compute 1 half of each pair
                                parse_dict["vectors"][i] += (k-j) / norm(k-j) # sum all together to save memory


                elif isinstance(next(iter(parse_dict["vectors"][i].keys()))[0], int): # single vector
                    for j in parse_dict["vectors"][i].keys():
                        parse_dict["vectors"][i][j] += [ (geo[j[0]] - geo[j[1]]) / norm(geo[j[0]] - geo[j[1]]) ]
                        if args.radius is not None:
                            parse_dict["location"][i][j] += [ (geo[j[0]] + geo[j[1]]) / 2.0 ] # average position of 2 atoms is the location assigned to the vector

                else: # multiple vectors, summed and normed
                    for j in parse_dict["vectors"][i].keys():
                        temp = np.array([0.,0.,0.])
                        temp_loc = np.array([0.,0.,0.])
                        loc_count = 0
                        for k in j:
                            temp += (geo[k[0]] - geo[k[1]]) / norm(geo[k[0]] - geo[k[1]])
                            temp_loc += geo[k[0]] + geo[k[1]]
                            loc_count += 2

                        parse_dict["location"][i][j] += [ temp_loc / loc_count ]

                        if norm(temp) == 0:
                            parse_dict["vectors"][i][j] += [ temp ]
                        else:
                            parse_dict["vectors"][i][j] += [ temp / norm(temp)]


            # Parse velocity vectors
            for i in parse_dict["velocity"].keys():
                parse_dict["velocity"][i] += [vels[i-1]]
                parse_dict["vel_loc"][i] += [geo[i-1]]
            for i in parse_dict["energy"].keys():
                parse_dict["energy"][i] += [engs[i-1]]
                parse_dict["eng_loc"][i] += [geo[i-1]]
            for i in parse_dict["force"].keys():
                parse_dict["force"][i] += [forces[i-1]]
                parse_dict["force_loc"][i] += [geo[i-1]]


        # Calculate histograms
        # Bonds
        if len(parse_dict["bonds"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder,"bonds")) is False: os.mkdir("{}/{}".format(args.o_folder,"bonds"))
        for i in parse_dict["bonds"].keys():
            bins,h = calc_hist([ k for j in parse_dict["bonds"][i] for k in parse_dict["bonds"][i][j] ],h_min=0.0,h_max=5.0,h_step=0.01,shave=args.shave_opt,normalize=True)
            os.mkdir("{}/{}/{}".format(args.o_folder,"bonds","_".join(i)))
            write_cols("{}/{}/{}/{}".format(args.o_folder,"bonds","_".join(i),"histogram.txt"),[bins,h],labels=["bin_centers (Angstroms)","probability"])
                     
        # Angles
        if len(parse_dict["angles"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder,"angles")) is False: os.mkdir("{}/{}".format(args.o_folder,"angles"))
        for i in parse_dict["angles"].keys():            
            bins,h = calc_hist([ k for j in parse_dict["angles"][i] for k in parse_dict["angles"][i][j] ],h_min=0.0,h_max=180.0,h_step=5.0,shave=args.shave_opt,normalize=True)
            os.mkdir("{}/{}/{}".format(args.o_folder,"angles","_".join(i)))
            write_cols("{}/{}/{}/{}".format(args.o_folder,"angles","_".join(i),"histogram.txt"),[bins,h],labels=["bin_centers (Degrees)","probability"])

        # Dihedrals
        if len(parse_dict["dihedrals"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder,"dihedrals")) is False: os.mkdir("{}/{}".format(args.o_folder,"dihedrals"))
        for i in parse_dict["dihedrals"].keys():            
            bins,h = calc_hist([ k for j in parse_dict["dihedrals"][i] for k in parse_dict["dihedrals"][i][j] ],h_min=-180.0,h_max=180.0,h_step=5.0,shave=args.shave_opt,normalize=True,abs_opt=args.abs_opt)
            os.mkdir("{}/{}/{}".format(args.o_folder,"dihedrals","_".join(i)))
            write_cols("{}/{}/{}/{}".format(args.o_folder,"dihedrals","_".join(i),"histogram.txt"),[bins,h],labels=["bin_centers (Degrees)","probability"])

        # Impropers
        if len(parse_dict["impropers"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder,"impropers")) is False: os.mkdir("{}/{}".format(args.o_folder,"impropers"))
        for i in parse_dict["impropers"].keys():            
            bins,h = calc_hist([ k for j in parse_dict["impropers"][i] for k in parse_dict["impropers"][i][j] ],h_min=0.0,h_max=180.0,h_step=5.0,shave=args.shave_opt,normalize=True)
            os.mkdir("{}/{}/{}".format(args.o_folder,"impropers","_".join(i)))
            write_cols("{}/{}/{}/{}".format(args.o_folder,"impropers","_".join(i),"histogram.txt"),[bins,h],labels=["bin_centers (Degrees)","probability"])

        # Process correlations
        if args.corr_opt is True:

            # Bonds
            for i in parse_dict["bonds"].keys():
                ac,errs = process_autocorr([ parse_dict["bonds"][i][_] for _ in parse_dict["bonds"][i].keys() ])
                write_cols("{}/{}/{}/{}".format(args.o_folder,"bonds","_".join(i),"ac.txt"),[[j*args.f_every for j in range(len(ac))],ac,errs],labels=["frame","autocorrelation","stdev"])

            # Angles
            for i in parse_dict["angles"].keys():
                ac,errs = process_autocorr([ parse_dict["angles"][i][_] for _ in parse_dict["angles"][i].keys() ])
                write_cols("{}/{}/{}/{}".format(args.o_folder,"angles","_".join(i),"ac.txt"),[[j*args.f_every for j in range(len(ac))],ac,errs],labels=["frame","autocorrelation","stdev"])

            # Dihedrals
            for i in parse_dict["dihedrals"].keys():
                ac,errs = process_autocorr([ parse_dict["dihedrals"][i][_] for _ in parse_dict["dihedrals"][i].keys() ])
                write_cols("{}/{}/{}/{}".format(args.o_folder,"dihedrals","_".join(i),"ac.txt"),[[j*args.f_every for j in range(len(ac))],ac,errs],labels=["frame","autocorrelation","stdev"])

            # Impropers
            for i in parse_dict["impropers"].keys():
                ac,errs = process_autocorr([ parse_dict["impropers"][i][_] for _ in parse_dict["impropers"][i].keys() ])
                write_cols("{}/{}/{}/{}".format(args.o_folder,"impropers","_".join(i),"ac.txt"),[[j*args.f_every for j in range(len(ac))],ac,errs],labels=["frame","autocorrelation","stdev"])

            # Vectors
            # Autocorrelation
            if args.autocorr is True:
                if len(parse_dict["vectors"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder,"vectors")) is False: os.mkdir("{}/{}".format(args.o_folder,"vectors"))
                print("parse_dict: ", parse_dict["vectors"])
                for i in parse_dict["vectors"].keys():
                    if os.path.isdir("{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i ]))) is False: os.mkdir("{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i ])))

                    # Have not yet implemented deltas/log-spacing for theta phi plot
                    if args.thetaphi:
                        os.mkdir("{}/{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i ]), "thetaphi"))
                        for j in parse_dict["vectors"][i].keys():
                            os.mkdir("{}/{}/{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i ]), "thetaphi", j[0]))
                            theta, phi = thetaphiextract(parse_dict["vectors"][i][j])
                            write_cols("{}/{}/{}/{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i ]), "thetaphi", j[0], "thetaphi.txt"), [[j * args.f_every for j in range(len(theta))], theta, phi], labels=["frame", "theta", "phi"])

                    if deltas is not None:
                        ac,errs = process_autocorr([parse_dict["vectors"][i][_] for _ in parse_dict["vectors"][i].keys()],algorithm='P1', deltas=deltas, times=sorted(list(log_frames)),verbose=args.verbose,time_hist=args.time_hist,out="{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i])),origin_only=args.origin_only,legendres=int(args.legendres))
                        for leg_n in range(int(args.legendres)):
                            write_cols("{}/{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i]),"ac-P{}.txt".format(leg_n+1)),[deltas,ac[leg_n,:],errs[leg_n,:]],labels=["frame","autocorrelation","stdev"])
                        # if args.second_leg:
                        #     ac, errs = process_autocorr([parse_dict["vectors"][i][_] for _ in parse_dict["vectors"][i].keys()], algorithm='P2',deltas=deltas, times=sorted(list(log_frames)), verbose=args.verbose,time_hist=args.time_hist,out="{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i])),origin_only=args.origin_only)
                        #     write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]),"ac-P2.txt"), [deltas, ac, errs],labels=["frame", "autocorrelation", "stdev"])
                    else:
                        ac, errs = process_autocorr([parse_dict["vectors"][i][_] for _ in parse_dict["vectors"][i].keys()],algorithm='P1',verbose=args.verbose, indiv=args.individual, out="{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i])),f_every=args.f_every,time_hist=args.time_hist,origin_only=args.origin_only,legendres=int(args.legendres))
                        for leg_n in range(int(args.legendres)):
                            write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]), "ac-P{}.txt".format(leg_n+1)),[[j * args.f_every for j in range(len(ac))], ac[leg_n,:], errs[leg_n,:]],labels=["frame", "autocorrelation", "stdev"])
            ### Added by Dylan! Calculate scalar order parameter.
            if args.scalord:
                for i in parse_dict["vectors"].keys():
                    if os.path.isdir("{}/{}".format(args.o_folder,"vectors")) is False: 
                        os.mkdir("{}/{}".format(args.o_folder,"vectors"))
                    if os.path.isdir("{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i ]))) is False: 
                        os.mkdir("{}/{}/{}".format(args.o_folder,"vectors","_".join([ str(_) for _ in i ])))
                    so, errs, director_vec = process_scalord([parse_dict["vectors"][i][_] for _ in parse_dict["vectors"][i].keys()],algorithm='P1',verbose=args.verbose, indiv=args.individual, out="{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i])),f_every=args.f_every,time_hist=args.time_hist,origin_only=args.origin_only,legendres=int(args.legendres))
                    print("so: ", so)
                    for leg_n in range(int(args.legendres)):
                        write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]), "so-P{}.txt".format(leg_n+1)),[[j * args.f_every for j in range(len(so[leg_n,:]))], so[leg_n,:], errs[leg_n,:]],labels=["frame", "autocorrelation", "stdev"])
                        write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]), "dir-P{}.txt".format(leg_n+1)),[[j * args.f_every for j in range(len(so[leg_n,:]))], director_vec[:, 0], director_vec[:, 1],director_vec[:, 2]],labels=["frame", "dir_x", "dir_y", "dir_z"])
            ###

            # Crosscorrelation
            if args.crosscorr is True: # Calculate cross-corr with end-to-end method

                    if len(parse_dict["vectors"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder, "vectors")) is False: os.mkdir("{}/{}".format(args.o_folder, "vectors"))

                    for i in parse_dict["vectors"].keys():
                        if os.path.isdir("{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]))) is False: os.mkdir("{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i])))

                        if args.pairwise is False:
                            xc = process_crosscorr(parse_dict["vectors"][i], deltas=deltas)
                            if deltas is not None:
                                write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]), "xc-P1.txt"),
                                    [deltas, xc], labels=["frame", "crosscorrelation"])
                            else:
                                write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]), "xc-P1.txt"),
                                    [[j*args.f_every for j in range(len(xc))], xc], labels=["frame", "crosscorrelation"])
                        else:
                            parse_dict["location"][i]["box"] = box
                            if args.radius is None and args.double_hist is False: parse_dict["location"][i] = None
                            xc, hist = process_crosscorr_pairwise(parse_dict["vectors"][i], deltas=deltas, r=args.radius, location_dict_i=parse_dict["location"][i],hist=args.histogram,double_hist=args.double_hist,name="{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i])))
                            write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]),"xc-P1.txt"),
                                       [deltas, xc], labels=["frame", "crosscorrelation"])
                            if args.histogram:
                                write_cols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]),"histogram.txt"),
                                           [hist[0],hist[1]], labels=["bin", "histogram"])
                            if args.double_hist:
                                write_2dcols("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]),"2dhistogram.txt"), hist)
                                plot2dcrosssec("{}/{}/{}/{}".format(args.o_folder, "vectors", "_".join([str(_) for _ in i]),"cross_sections.pdf"), hist)



            # Velocities
            if args.vel is True:
                ####################################################################
                ###### ADJUST TO NOT OVERWRITE ORIGINAL VELOCITIES WHEN NORMING - PUT THEM IN A NEW DICT #####
                ####################################################################
                if len(parse_dict["velocity"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder, "velocity")) is False: os.mkdir("{}/{}".format(args.o_folder, "velocity"))
                if parse_dict["velocity"].keys() is not {}:
                    if deltas is not None:
                        ac, errs = process_autocorr(parse_dict["velocity"], algorithm='P1', deltas=deltas, times=sorted(list(log_frames)))
                        write_cols("{}/{}/{}".format(args.o_folder, "velocity","vac-P1.txt"), [deltas, ac, errs],labels=["frame", "autocorrelation", "stdev"])
                    else:
                        ac, errs = process_autocorr(parse_dict["velocity"], algorithm='P1')
                        write_cols("{}/{}/{}".format(args.o_folder, "velocity","vac-P1.txt"),[[j * args.f_every for j in range(len(ac))], ac, errs],labels=["frame", "autocorrelation", "stdev"])

                if args.vel_crosscorr:
                    if not args.no_vel_norm:
                        # normalize vectors for full correlation:
                        for i in parse_dict["velocity"]: # i = atom number
                            parse_dict["vel_norm"][i] = [j/norm(j) for j in parse_dict["velocity"][i]]
                            #parse_dict["velocity"][i] = [j/norm(j) for j in parse_dict["velocity"][i]]
                    xc = process_crosscorr(parse_dict["vel_norm"], deltas=deltas,vel=True)
                    if args.no_vel_norm:
                        vsq_avg = 0
                        for j in parse_dict["velocity"]: vsq_avg += norm(parse_dict["velocity"][j][0])**2
                        xc = xc/vsq_avg*len(parse_dict["velocity"])
                    write_cols("{}/{}/{}".format(args.o_folder, "velocity", "vac-xc.txt"), [deltas, xc], labels=["frame", "vel_fc"])

                if args.vfcbox is not None:
                    parse_dict["vel_loc"]["box"] = box
                    if not args.no_vel_norm:
                        # normalize vectors for full correlation:
                        for i in parse_dict["velocity"]:  # i = atom number
                            parse_dict["velocity"][i] = [j / norm(j) for j in parse_dict["velocity"][i]]
                    xc = process_crosscorr_subset(parse_dict["velocity"], location_dict_i = parse_dict["vel_loc"], deltas=deltas, vel=True, normed=not args.no_vel_norm, subboxes=args.vfcbox, vel_atom=vel_atom)
                    xc = np.array(xc)
                    # if args.no_vel_norm:
                    #     vsq_avg = 0
                    #     for j in parse_dict["velocity"]: vsq_avg += norm(parse_dict["velocity"][j][0]) ** 2
                    #     xc = xc / vsq_avg #* len(parse_dict["velocity"])
                    xcm = np.mean(xc,axis=0)
                    print("{}".format(xcm))
                    xcstd = np.std(xc,axis=0)
                    write_cols("{}/{}/{}".format(args.o_folder, "velocity", "vac-xc_subbox.txt"), [deltas, xcm, xcstd], labels=["frame", "vel_fc", "fcstd"])

                if args.vfcrad is not None:
                    parse_dict["vel_loc"]["box"] = box
                    if not args.no_vel_norm:
                        # normalize vectors for full correlation:
                        for i in parse_dict["velocity"]:  # i = atom number
                            parse_dict["velocity"][i] = [j / norm(j) for j in parse_dict["velocity"][i]]
                    xc,subsize_t = process_crosscorr_subset(parse_dict["velocity"], location_dict_i=parse_dict["vel_loc"],
                                                  deltas=deltas, vel=True, normed=not args.no_vel_norm,
                                                  radius=args.vfcrad, vel_atom=vel_atom)
                    xc = np.array(xc)
                    # if args.no_vel_norm:
                    #     vsq_avg = 0
                    #     for j in parse_dict["velocity"]: vsq_avg += norm(parse_dict["velocity"][j][0]) ** 2
                    #     xc = xc / vsq_avg #* len(parse_dict["velocity"])
                    xcm = np.mean(xc, axis=0)
                    print("{}".format(xcm))
                    xcstd = np.std(xc, axis=0)
                    write_cols("{}/{}/{}".format(args.o_folder, "velocity", "vac-xc_subrad.txt"), [deltas, xcm, xcstd, subsize_t],
                               labels=["frame", "vel_fc", "fcstd", "subset_size"])

            if args.u_dist is not False:
                if len(parse_dict["energy"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder, "energy")) is False: os.mkdir("{}/{}".format(args.o_folder, "energy"))

                pehist, ehist, ksstat, kp, skew, kurtosis = compute_energy_distr(parse_dict["energy"], deltas=deltas, cell_size=args.u_dist)
                ksstat_arr = ksstat*np.ones(len(ehist[0]))
                kp_arr = kp * np.ones(len(ehist[0]))
                skew_arr = skew * np.ones(len(ehist[0]))
                kurt_arr = kurtosis * np.ones(len(ehist[0]))
                write_cols("{}/{}/{}".format(args.o_folder, "energy", "ehist.txt"), [ehist[1][:-1], ehist[0], ksstat_arr, kp_arr, skew_arr, kurt_arr],
                               labels=["bins", "ehist", "kstat", "kpval","skew","kurtosis"])
                write_cols("{}/{}/{}".format(args.o_folder, "energy", "pehist.txt"), [pehist[1][:-1], pehist[0]],
                               labels=["bins", "pehist"])
                full_data = [parse_dict["energy"][i][j] for i in parse_dict["energy"] for j in parse_dict["energy"][i]]
                write_cols("{}/{}/{}".format(args.o_folder, "energy", "raw_e.txt"), [full_data],labels=["full_data"])

            # if args.eng_space_distr is not False:
            #     if len(parse_dict["energy"]) > 0 and os.path.isdir("{}/{}".format(args.o_folder, "energy")) is False: os.mkdir("{}/{}".format(args.o_folder, "energy"))


            if args.lind:
                if os.path.isdir("{}/{}".format(args.o_folder, "lind")) is False: os.mkdir("{}/{}".format(args.o_folder, "lind"))
                for i in parse_dict["location"]:
                    lind_ind = compute_lindemann(parse_dict["location"][i], box, deltas=deltas)

                    write_cols("{}/{}/{}".format(args.o_folder, "lind", "lind_{}.txt".format("_".join([ str(_) for _ in i ]))), [[lind_ind,0]],
                                   labels=["lind"])

            if args.ang_mom:
                if os.path.isdir("{}/{}".format(args.o_folder, "ang_mom")) is False: os.mkdir("{}/{}".format(args.o_folder, "ang_mom"))

                # Cluster
                cluster_size = int(args.cluster_size)
                clust_dict = cluster(parse_dict["vel_loc"], cluster_size, deltas, box_t, args.molecule_clust)
                tot_am_vec, ang_mom_corr, am_distribution, Lx_distr, vel_distr = compute_ang_mom_fc(parse_dict["vel_loc"],parse_dict["velocity"],clust_dict,deltas)

                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "am.txt"), [[ang_mom_corr]*len(am_distribution), am_distribution], labels=["am_corr_avg", "am_corr_values"])
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "lx.txt"),[Lx_distr],labels=["Lx_values"])
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "vel.txt"), [vel_distr], labels=["vel_values"])

            if args.inertia:
                if not args.ang_mom:
                    if os.path.isdir("{}/{}".format(args.o_folder, "ang_mom")) is False: os.mkdir("{}/{}".format(args.o_folder, "ang_mom"))
                    # Cluster
                    cluster_size = int(args.cluster_size)
                    clust_dict = cluster(parse_dict["vel_loc"], cluster_size, deltas, box_t, args.molecule_clust)
                    #tot_am_vec, ang_mom_corr, am_distribution, Lx_distr, vel_distr = compute_ang_mom_fc(parse_dict["vel_loc"], parse_dict["velocity"], clust_dict, deltas)

                c1, c2, c3, maxinvar2, maxgeom, abs_max_geom, mingeom, abs_min_geom, max_geo_molids, min_geo_molids = compute_inertia(parse_dict["vel_loc"], clust_dict, box_t, deltas=deltas)
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "inertia1.txt"), [c1], labels=["a+b+c"])
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "inertia2.txt"), [c2], labels=["ab+ac+bc"])
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "inertia3.txt"), [c3], labels=["abc"])
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "max_invar_2.lammpstrj"), [range(1,cluster_size+1), [1]*cluster_size, [i[0] for i in maxgeom], [i[1] for i in maxgeom], [i[2] for i in maxgeom]], comment="ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n{}\nITEM: BOX BOUNDS pp pp pp\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\nITEM: ATOMS id type x y z".format(cluster_size))
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "min_invar_2.lammpstrj"),[range(1, cluster_size+1), [1] * cluster_size, [i[0] for i in mingeom], [i[1] for i in mingeom],[i[2] for i in mingeom]],comment="ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n{}\nITEM: BOX BOUNDS pp pp pp\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\nITEM: ATOMS id type x y z".format(cluster_size))
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "abs_max_invar_2.lammpstrj"), [range(1, cluster_size+1), [1] * cluster_size, [i[0] for i in abs_max_geom], [i[1] for i in abs_max_geom],[i[2] for i in abs_max_geom]], comment="ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n{}\nITEM: BOX BOUNDS pp pp pp\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\nITEM: ATOMS id type x y z".format(cluster_size))
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "abs_min_invar_2.lammpstrj"),[range(1, cluster_size+1), [1] * cluster_size, [i[0] for i in abs_min_geom], [i[1] for i in abs_min_geom], [i[2] for i in abs_min_geom]],comment="ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n{}\nITEM: BOX BOUNDS pp pp pp\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\n-5.3825518349914383e+00 5.3825518349914383e+00\nITEM: ATOMS id type x y z".format(cluster_size))
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "max_molids.txt"), [range(1,cluster_size+1),max_geo_molids], labels=["num","molid"])
                write_cols("{}/{}/{}".format(args.o_folder, "ang_mom", "min_molids.txt"), [range(1, cluster_size+1), min_geo_molids], labels=["num", "molid"])
                write_psf("{}/{}/{}".format(args.o_folder, "ang_mom", "abs_max_invar_2.psf"), max_geo_molids)
                write_psf("{}/{}/{}".format(args.o_folder, "ang_mom", "abs_min_invar_2.psf"), min_geo_molids)

            if args.force:
                if os.path.isdir("{}/{}".format(args.o_folder, "force")) is False: os.mkdir("{}/{}".format(args.o_folder, "force"))
                if (not args.ang_mom and not args.inertia) or clust_dict == {}:
                    # Cluster
                    cluster_size = int(args.cluster_size)
                    clust_dict = cluster(parse_dict["force_loc"], cluster_size, deltas, box_t, args.molecule_clust, species=args.species, rand=args.rand)
                    #forces = compute_force(parse_dict["force"], clust_dict, deltas)

                forces, fx, fy, fz, max_frame, max_mol, min_frame, min_mol = compute_force(parse_dict["force"],clust_dict,deltas=deltas)

                write_cols("{}/{}/{}".format(args.o_folder, "force", "forces.txt"), [forces], labels=["forces"])
                write_cols("{}/{}/{}".format(args.o_folder, "force", "fx.txt"), [fx], labels=["fx"])
                write_cols("{}/{}/{}".format(args.o_folder, "force", "fy.txt"), [fy], labels=["fy"])
                write_cols("{}/{}/{}".format(args.o_folder, "force", "fz.txt"), [fz], labels=["fz"])
                write_cols("{}/{}/{}".format(args.o_folder, "force", "max.txt"), [[max_frame],[max_mol]], labels=["max_frame","max_mol"])
                write_cols("{}/{}/{}".format(args.o_folder, "force", "min.txt"), [[min_frame], [min_mol]],labels=["min_frame", "min_mol"])

                if args.force_ac:
                    force_ac, force_ac_errs = process_autocorr(parse_dict["force"], algorithm='P1', deltas=deltas, times=sorted(list(log_frames)),legendres=1)
                    write_cols("{}/{}/{}".format(args.o_folder, "force", "force_ac.txt"), [deltas, force_ac, force_ac_errs],labels=["time", "force_ac","force_ac_errs"])

    print("vautocorr for {} completed successfully!".format(args.o_folder))

    quit()


def compute_force(force_dict, clust_dict, deltas=None):

    forces = []
    fx = []
    fy = []
    fz = []

    max_force = 0
    max_frame = 0
    max_mol = 0
    min_force = np.inf
    min_frame = 0
    min_mol = 0

    if deltas is not None:
        for count_t, t in enumerate(deltas):
            for mol_i in clust_dict.keys():
                temp_force = np.zeros(3)

                for cl_mol in clust_dict[mol_i][count_t]:
                    temp_force += force_dict[cl_mol][count_t]

                forces.append(np.dot(temp_force,temp_force))
                fx.append(temp_force[0])
                fy.append(temp_force[1])
                fz.append(temp_force[2])

                if np.dot(temp_force,temp_force) > max_force:
                    max_force = np.dot(temp_force,temp_force)
                    max_frame = t
                    max_mol = mol_i
                if np.dot(temp_force,temp_force) < min_force:
                    min_force = np.dot(temp_force,temp_force)
                    min_frame = t
                    min_mol = mol_i
    else:
        for count_t in range(len(clust_dict[clust_dict.keys()[0]])):
            for mol_i in clust_dict.keys():
                temp_force = np.zeros(3)

                for cl_mol in clust_dict[mol_i][count_t]:
                    temp_force += force_dict[cl_mol][count_t]

                forces.append(np.dot(temp_force,temp_force))
                fx.append(temp_force[0])
                fy.append(temp_force[1])
                fz.append(temp_force[2])

                if np.dot(temp_force,temp_force) > max_force:
                    max_force = np.dot(temp_force,temp_force)
                    max_frame = count_t
                    max_mol = mol_i
                if np.dot(temp_force,temp_force) < min_force:
                    min_force = np.dot(temp_force,temp_force)
                    min_frame = count_t
                    min_mol = mol_i

    return forces, fx, fy, fz, max_frame, max_mol, min_frame, min_mol


def write_psf(filename, mol_ids):
    # This function assumes molecules of size n=3 with organization of 1 bonded to 2 and 3, 4 bonded to 5 and 6,... (1-indexed)
    n=3

    with open(filename, 'w') as f:
        f.write("PSF\n\n")

        f.write("{:>8d} !NTITLE\n".format(1))
        f.write("{:>8s} Generated by vautocorr.py\n\n".format("REMARKS"))

        f.write("{:>8d} !NATOM\n".format(len(mol_ids)))


        for count_i,i in enumerate(mol_ids):
            center = (i-1)%3==0 # If true, it is a center molecule
            if center:
                element = "B"
                atype = 1
            else:
                element = "C"
                atype = 2
            f.write("{:>8d}      {:<6d}    {:<4s} {:<5d} {:< 15.6f} {:<17.4f} {:<11d}\n".format(count_i+1,0,element,atype,0.00,78.000,0))

        # Find bonds:
        bond_count = 0
        bonds = []
        for count_i, i in enumerate(mol_ids):
            center = (i - 1) % 3 == 0
            # Only need to check center atoms to avoid double counting
            if center:
                if i+1 in mol_ids:
                    bonds.extend([i,i+1])
                    bond_count += 1
                if i+2 in mol_ids:
                    bonds.extend([i,i+2])
                    bond_count += 1
        # Write bonds:
        f.write("\n{:>8d} !NBOND: bonds\n".format(bond_count))
        for i in range(int(ceil(float(bond_count/4.)))):
            f.write(" {}\n".format(" ".join([ "{:>7s}".format(j) for j in bonds[i*8:(i+1)*8] ])))


        # Write the rest
        f.write("\n{:>8d} !NTHETA: angles\n\n".format(0))
        f.write("\n{:>8d} !NPHI: dihedrals\n\n".format(0))
        f.write("\n{:>8d} !NIMPHI: impropers\n\n".format(0))
        f.write("\n{:>8d} !NDON: donors\n\n".format(0))
        f.write("\n{:>8d} !NACC: acceptors\n\n".format(0))
        f.write("\n{:>8d} !NNB\n\n".format(0))
        f.write("\n{:>8d} {:>7d} !NGRP\n".format(1, 0))

    return


def compute_inertia(loc_dict, clust_dict, box_t, deltas=None):


    inertia_dict = {}
    mass = 78.0
    char1_t = []
    char2_t = []
    char3_t = []
    max_invar2 = 0
    max_geom = []
    min_invar2 = np.inf
    min_geom = []
    max_geo_molids = []
    min_geo_molids = []

    for count_t, t in enumerate(deltas):

        box = box_t[count_t][:,1]-box_t[count_t][:,0]
        box2 = (box_t[count_t][:,1]-box_t[count_t][:,0])/2.0

        # Compute inertia tensor for each cluster
        for mol_i in range(1,len(loc_dict)+1):
            temp_inertia = np.zeros(6) # xx, xy, xz, yy, yz, zz

            for cl_mol in clust_dict[mol_i][count_t]:
                rx = loc_dict[cl_mol][count_t][0] - loc_dict[mol_i][count_t][0]
                ry = loc_dict[cl_mol][count_t][1] - loc_dict[mol_i][count_t][1]
                rz = loc_dict[cl_mol][count_t][2] - loc_dict[mol_i][count_t][2]
                if rx > box2[0]:
                    rx-=box[0]
                if rx < -box2[0]:
                    rx+=box[0]
                if ry > box2[1]:
                    ry-=box[1]
                if ry < -box2[1]:
                    ry+=box[1]
                if rz > box2[2]:
                    rz-=box[2]
                if rz < -box2[2]:
                    rz+=box[2]
                temp_inertia[0] += mass*(ry**2+rz**2) #xx
                temp_inertia[1] += mass*(-rx*ry) #xy
                temp_inertia[2] += mass*(-rx*rz) #xz
                temp_inertia[3] += mass*(rx**2+rz**2) #yy
                temp_inertia[4] += mass*(-ry*rz) #yz
                temp_inertia[5] += mass*(rx**2+ry**2) #zz

                # temp_vec += np.cross(loc_dict[cl_mol][count_t]-loc_dict[mol_i][count_t], vel_dict[cl_mol][count_t]) # Subtract off distance to the central atom of group to make central atom the origin
                # temp_vel += vel_dict[cl_mol][count_t]
            temp_inertia_matrix = np.zeros((3,3))
            temp_inertia_matrix[0, 0] = temp_inertia[0]
            temp_inertia_matrix[0, 1] = temp_inertia[1]
            temp_inertia_matrix[0, 2] = temp_inertia[2]
            temp_inertia_matrix[1, 0] = temp_inertia[1]
            temp_inertia_matrix[1, 1] = temp_inertia[3]
            temp_inertia_matrix[1, 2] = temp_inertia[4]
            temp_inertia_matrix[2, 0] = temp_inertia[2]
            temp_inertia_matrix[2, 1] = temp_inertia[4]
            temp_inertia_matrix[2, 2] = temp_inertia[5]

            eig_vals, eig_vecs = np.linalg.eig(temp_inertia_matrix)

            char1 = eig_vals[0] + eig_vals[1] + eig_vals[2]
            char2 = eig_vals[0]*eig_vals[1] + eig_vals[0]*eig_vals[2] + eig_vals[1]*eig_vals[2]
            char3 = eig_vals[0]*eig_vals[1]*eig_vals[2]


            char1_t.append(char1)
            char2_t.append(char2)
            char3_t.append(char3)
            ###########
            # Separate invars(t) for different molecules
            ###########
            if char2 > max_invar2:
                max_invar2 = char2
                max_geom = []
                abs_max_geom = []
                max_geo_molids = []
                for cl_mol in clust_dict[mol_i][count_t]:
                    rx = loc_dict[cl_mol][count_t][0] - loc_dict[mol_i][count_t][0]
                    ry = loc_dict[cl_mol][count_t][1] - loc_dict[mol_i][count_t][1]
                    rz = loc_dict[cl_mol][count_t][2] - loc_dict[mol_i][count_t][2]
                    if rx > box2[0]:
                        rx -= box[0]
                    if rx < -box2[0]:
                        rx += box[0]
                    if ry > box2[1]:
                        ry -= box[1]
                    if ry < -box2[1]:
                        ry += box[1]
                    if rz > box2[2]:
                        rz -= box[2]
                    if rz < -box2[2]:
                        rz += box[2]
                    max_geom.append([rx,ry,rz])
                    abs_max_geom.append(loc_dict[cl_mol][count_t])
                    max_geo_molids.append(cl_mol)

                print("max: t={}, mol_id={}".format(t, mol_i))

            if char2 < min_invar2:
                min_invar2 = char2
                min_geom = []
                abs_min_geom = []
                min_geo_molids = []

                for cl_mol in clust_dict[mol_i][count_t]:
                    rx = loc_dict[cl_mol][count_t][0] - loc_dict[mol_i][count_t][0]
                    ry = loc_dict[cl_mol][count_t][1] - loc_dict[mol_i][count_t][1]
                    rz = loc_dict[cl_mol][count_t][2] - loc_dict[mol_i][count_t][2]
                    if rx > box2[0]:
                        rx -= box[0]
                    if rx < -box2[0]:
                        rx += box[0]
                    if ry > box2[1]:
                        ry -= box[1]
                    if ry < -box2[1]:
                        ry += box[1]
                    if rz > box2[2]:
                        rz -= box[2]
                    if rz < -box2[2]:
                        rz += box[2]
                    min_geom.append([rx,ry,rz])
                    abs_min_geom.append(loc_dict[cl_mol][count_t])
                    min_geo_molids.append(cl_mol)

                print("max: t={}, mol_id={}".format(t, mol_i))


    return char1_t, char2_t, char3_t, max_invar2, max_geom, abs_max_geom, min_geom, abs_min_geom, max_geo_molids, min_geo_molids


def compute_ang_mom_fc(loc_dict, vel_dict, clust_dict, deltas=None):

    amfc_t = 0
    am_vec_t = []
    ang_mom_dict = {}
    distr_arr = []
    v_distr = []
    Lx_distr = []

    for count_t, t in enumerate(deltas):

        # Compute angular momentum for each cluster
        for mol_i in range(1,len(loc_dict)+1):
            if count_t == 0:
                ang_mom_dict[mol_i] = []
            temp_vec = np.zeros(3)
            temp_vel = np.zeros(3)
            for cl_mol in clust_dict[mol_i][count_t]:
                temp_vec += np.cross(loc_dict[cl_mol][count_t]-loc_dict[mol_i][count_t], vel_dict[cl_mol][count_t]) # Subtract off distance to the central atom of group to make central atom the origin
                temp_vel += vel_dict[cl_mol][count_t]

            ang_mom_dict[mol_i].append(temp_vec)
            v_distr.append(np.dot(temp_vel,temp_vel))

        am_vec_sum = np.zeros(3)

        for mol_i in range(1,len(loc_dict)+1):
            # for count_2, amom_vec_2 in enumerate(ang_mom_dict):
            #     if count_1 > count_2:
            #         continue
            #     else:
            am_vec_sum += ang_mom_dict[mol_i][count_t]
            distr_arr.append(np.dot(ang_mom_dict[mol_i][count_t],ang_mom_dict[mol_i][count_t]))
            Lx_distr.append(ang_mom_dict[mol_i][count_t][0])

        amfc = np.dot(am_vec_sum,am_vec_sum)
        amfc_t += amfc
        am_vec_t.append(am_vec_sum)

    amfc_t = amfc_t / len(deltas)

    #rand_vals = compute_ang_mom_rand(loc_dict, vel_dict, deltas)

    return am_vec_t, amfc_t, distr_arr, Lx_distr, v_distr #, rand_vals


def compute_ang_mom_rand(loc_dict, vel_dict, deltas):

    # THIS IS ONLY FOR CGOTP

    n_vals = len(vel_dict)*len(deltas)
    random_vals = np.zeros(n_vals)
    n_in_group = 14
    n_atom_in_group = 3*14
    density = 1.0
    max_rad = (n_in_group / (density * (4. / 3. * np.pi))) ** (1. / 3.)

    amfc_t = 0
    am_vec_t = []
    ang_mom_dict = {}
    distr_arr = []

    for count_t, t in enumerate(deltas):
        for i in range(n_vals):
            ang_mom_dict[i] = []

            # location of center of mass
            xs = np.random.normal(loc=0, scale=1, size=n_in_group)
            ys = np.random.normal(loc=0, scale=1, size=n_in_group)
            zs = np.random.normal(loc=0, scale=1, size=n_in_group)
            vecs = np.zeros((n_in_group, 3))
            vecs[:, 0] = xs
            vecs[:, 1] = ys
            vecs[:, 2] = zs

            norms = np.linalg.norm(vecs, axis=1)
            fullnorms = np.zeros((n_in_group, 3))
            fullnorms[:, 0] = norms
            fullnorms[:, 1] = norms
            fullnorms[:, 2] = norms
            vecs = vecs / fullnorms

            rads = np.random.rand(n_in_group) * max_rad
            fullrads = np.zeros((n_in_group, 3))
            fullrads[:, 0] = rads
            fullrads[:, 1] = rads
            fullrads[:, 2] = rads

            final_vecs = vecs * fullrads

            rand_locs = np.zeros((n_in_group+1)*3, 3)
            rand_vels = np.zeros((n_in_group+1)*3, 3)

            # get orientations and velocities from simulation
            for count_cl, cl in enumerate(np.random.choice(len(vel_dict), size=n_in_group+1, replace=False)):
                if count_cl == 0:
                    while np.linalg.norm(loc_dict[3*cl][count_t] - loc_dict[3*cl+1][count_t]) > 6.0 or np.linalg.norm(loc_dict[3*cl][count_t] - loc_dict[3*cl+2][count_t]) > 6.0:
                        print("{} over periodic boundary".format(3*cl))
                        cl += 1
                    com_vec = 1./3.*(loc_dict[3*cl][count_t] + loc_dict[3*cl+1][count_t] + loc_dict[3*cl+2][count_t])
                    rand_locs[3*count_cl, :] = loc_dict[3*cl][count_t] - com_vec
                    rand_locs[3*count_cl+1,:] = loc_dict[3*cl+1][count_t] - com_vec
                    rand_locs[3*count_cl+2,:] = loc_dict[3*cl+2][count_t] - com_vec

                    rand_vels[3*count_cl,:] = vel_dict[3*cl][count_t]
                    rand_vels[3*count_cl+1, :] = vel_dict[3*cl+1][count_t]
                    rand_vels[3*count_cl+2, :] = vel_dict[3*cl+2][count_t]

                else:
                    while np.linalg.norm(loc_dict[3*cl][count_t] - loc_dict[3*cl+1][count_t]) > 6.0 or np.linalg.norm(loc_dict[3*cl][count_t] - loc_dict[3*cl+2][count_t]) > 6.0:
                        print("{} over periodic boundary".format(3*cl))
                        cl += 1
                    com_vec = 1./3.*(loc_dict[3*cl][count_t] + loc_dict[3*cl+1][count_t] + loc_dict[3*cl+2][count_t]) - final_vecs[count_cl-1] # get center of mass in simulation to subtract off, and subtract the final destination within the sphere which will be added back
                    rand_locs[3*count_cl, :] = loc_dict[3*cl][count_t] - com_vec
                    rand_locs[3*count_cl+1,:] = loc_dict[3*cl+1][count_t] - com_vec
                    rand_locs[3*count_cl+2,:] = loc_dict[3*cl+2][count_t] - com_vec

                    rand_vels[3*count_cl, :] = vel_dict[3 * cl][count_t]
                    rand_vels[3*count_cl + 1, :] = vel_dict[3 * cl + 1][count_t]
                    rand_vels[3*count_cl + 2, :] = vel_dict[3 * cl + 2][count_t]


            # Compute angular momentum for each cluster
            temp_vec = np.zeros(3)
            for mol_i in range(1, 3*(n_in_group + 1)):
                temp_vec += np.cross(rand_locs[mol_i], rand_vels[mol_i])  # Don't need to subtract off since the sphere is centered at 0

            ang_mom_dict[i].append(temp_vec)

            am_vec_sum = np.zeros(3)

            for mol_i in range(1, 3*(n_in_group + 1)):
                # for count_2, amom_vec_2 in enumerate(ang_mom_dict):
                #     if count_1 > count_2:
                #         continue
                #     else:
                am_vec_sum += ang_mom_dict[mol_i]
                distr_arr.append(np.dot(ang_mom_dict[mol_i][count_t], ang_mom_dict[mol_i][count_t]))

            amfc = np.dot(am_vec_sum, am_vec_sum)
            amfc_t += amfc
            am_vec_t.append(am_vec_sum)


    return random_vals


def cluster(pos_dict, n, deltas, box_t, molecule, species=None, rand=False):

    clust_dict = {}

    N_atoms = len(pos_dict)

    # DRAW RANDOM ATOMS WITHOUT REPLACEMENT for cluster of given size
    if rand:
        for count_t, t in enumerate(deltas):
            for count_i, i in enumerate(pos_dict):
                if count_t == 0:
                    clust_dict[i] = []
                selection = [j for j in range(1,N_atoms+1) if j != i]
                clust_dict[i].append(random.sample(selection,n))
        return clust_dict

    if molecule is not None:
        if species is not None: # Only works for single atom species. species should take the form "x y" where x is the frequency of the species you want to observe (i.e. 2 for a 50/50 mixture, 5 for a 80/20 mixture
            mod = int(species.split()[0])
            remainder = int(species.split()[1])
            for count_t, t in enumerate(deltas):
                for mol in range(int(N_atoms/float(molecule))):
                    if (mol-remainder)%mod != 0:
                        continue
                    if count_t == 0:
                        clust_dict[mol] = []
                    clust_dict[mol].append([mol+1])
            return clust_dict
        else:
            molecule = int(molecule) # Number of atoms in a molecule
            if deltas is not None:
                for count_t, t in enumerate(deltas):
                    for mol in range(int(N_atoms/float(molecule))):
                        if count_t == 0:
                            clust_dict[mol] = []
                        clust_dict[mol].append(range(molecule*mol+1,(molecule)*(mol+1)+1))
            else:
                for count_t in range(len(pos_dict[pos_dict.keys()[0]])):
                    for mol in range(int(N_atoms/float(molecule))):
                        if count_t == 0:
                            clust_dict[mol] = []
                        clust_dict[mol].append(range(molecule*mol+1,(molecule)*(mol+1)+1))

            return clust_dict

    for count_t, t in enumerate(deltas):
        print("{}: {}".format(t,time()))
        box = box_t[count_t][:, 1] - box_t[count_t][:,0]
        dist_arr = np.zeros((len(pos_dict),3))

        for count_i, i in enumerate(pos_dict):
            dist_arr[count_i] = pos_dict[i][count_t]

        #dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(dist_arr))
        dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(dist_arr, lambda u, v, box: np.sqrt((min(u[0] - v[0],box[0] - np.abs((u[0] - v[0])),key=abs) ** 2 + min(u[1] - v[1],box[1] - np.abs((u[1] - v[1])),key=abs) ** 2 + min(u[2] - v[2], box[2] - np.abs((u[2] - v[2])), key=abs) ** 2).sum()), box=box))

        for count_i, i in enumerate(pos_dict):
            if count_t == 0:
                clust_dict[i] = []
            clust_dict[i].append(np.argpartition(dists[count_i],n+1)[:n+1] + 1)
            #clust_dict[i] = np.where(dists==sorted(dists[count_i])[0:n+1], dists)

    return clust_dict

def compute_lindemann(loc_dict, box, deltas=None):

    global box_arr
    box_arr = np.array([np.abs(box[0][1]-box[0][0]), np.abs(box[1][1]-box[1][0]), np.abs(box[2][1]-box[2][0])])
    dists = np.zeros(len(deltas))
    linds = []
    for count_t, t in enumerate(deltas):
        loc_arr = []
        for i in range(len(loc_dict.values())):
            loc_arr.append(loc_dict.values()[i][count_t])
        loc_arr = np.array(loc_arr)
        temp_dists = scipy.spatial.distance.pdist(loc_arr, lambda u, v: (np.sqrt((min(u[0]-v[0],box_arr[0]-np.abs((u[0]-v[0])),key=abs)**2 + min(u[1]-v[1],box_arr[1]-np.abs((u[1]-v[1])),key=abs)**2 + min(u[2]-v[2],box_arr[2]-np.abs((u[2]-v[2])),key=abs)**2).sum())))

        N_pairs = len(temp_dists) # N*(N-1)
        temp_lind = 1./N_pairs*np.sqrt(np.sum(temp_dists**2) - np.sum(temp_dists)**2/len(temp_dists))/np.sum(temp_dists)
        linds.append(temp_lind)

    lind = np.mean(linds)

    return lind


def compute_energy_distr(edict, deltas, cell_size=1):


    cell_size=int(cell_size)
    N_atoms = len(edict)
    N_cells = N_atoms/int(cell_size)
    num_bins = 2001


    pe_min = np.min([j[0] for i in edict for j in edict[i]])*cell_size # Find min and max so we can have consistent bins for the histogram
    pe_max = np.max([j[0] for i in edict for j in edict[i]])*cell_size
    pe_bins = np.linspace(pe_min,pe_max,num=num_bins)
    ke_min = np.min([j[1] for i in edict for j in edict[i]])*cell_size
    ke_max = np.max([j[1] for i in edict for j in edict[i]])*cell_size
    ke_bins = np.linspace(ke_min, ke_max, num=num_bins)
    e_bins = pe_bins+ke_bins

    for count_t,t in enumerate(deltas):
        pedist = np.zeros(N_cells)
        kedist = np.zeros(N_cells)
        edist = np.zeros(N_cells)
        for mol_i in edict: # cluster atomic energies by molecule or unit cells
            if mol_i == 0:
                pedist[0] += edict[mol_i][count_t][0]
                kedist[0] += edict[mol_i][count_t][1]
                edist[0] += edict[mol_i][count_t][0] + edict[mol_i][count_t][1]
            else:
                pedist[int((mol_i-1)/cell_size)] += edict[mol_i][count_t][0]
                kedist[int((mol_i-1)/cell_size)] += edict[mol_i][count_t][1]
                edist[int((mol_i-1)/cell_size)] += edict[mol_i][count_t][0] + edict[mol_i][count_t][1]

        if count_t == 0:
            pehist, pebins = np.histogram(pedist, bins=pe_bins, density=False)
            kehist, kebins = np.histogram(kedist, bins=ke_bins, density=False)
            ehist, ebins = np.histogram(edist,bins=e_bins, density=False)
        else:
            pehist += np.histogram(pedist, bins=pe_bins, density=False)[0]
            kehist += np.histogram(kedist, bins=ke_bins, density=False)[0]
            ehist += np.histogram(edist, bins=e_bins, density=False)[0]

        ksstat, kp = scipy.stats.kstest(edist,"norm")

        skew = scipy.stats.skew(edist)
        kurtosis = scipy.stats.kurtosis(edist)

        # # Trim 0s for the purposes of fitting:
        # # Trim right
        # trim_ehist = np.trim_zeros(ehist,'b')
        # trim_ebins = ebins[:len(trim_ehist)]
        # # Trim left
        # trim_ehist = np.trim_zeros(trim_ehist,'f')
        # trim_ebins = trim_ebins[(len(trim_ebins)-len(trim_ehist)):]

    return (pehist,pebins), (ehist,ebins),ksstat,kp,skew,kurtosis


def thetaphiextract(values):
    theta = []
    phi = []
    for i in values:
        # Here theta is in [0,2pi)
        if i[0] > 0 and i[1] > 0: # This is the domain arctan is valid
            theta += [np.arctan(i[1]/i[0])] # y and x component of our vector. Here theta is the polar angle
        elif i[0] > 0 and i[1] < 0: # quadrant 4 needs to add 2pi
            theta += [np.arctan(i[1]/i[0]) + 2*np.pi]
        else: # quadrants 2 and 3 need to add pi
            theta += [np.arctan(i[1]/i[0]) + np.pi]
        phi += [np.arccos(i[2])] # z component of vector (assuming already normalized). Here phi is the azimuthal angle

    return theta, phi


def remove_omitted(parse_dict,objects,omitted):

    for i in parse_dict:
        for j in parse_dict[i]:
            del_list = []
            for k in parse_dict[i][j]:
                if True in [ objects["molids"][n_atom] in omitted["molids"] for tup in k for n_atom in tup ]:
                    del_list += [k]
                    continue
                elif True in [ _ in omitted["ids"] for _ in k ]:
                    del_list += [k]
                    continue

            for k in del_list:
                parse_dict[i][j].pop(k)
    return parse_dict

# Description: Generate autocorrelation:
def process_autocorr(values,algorithm='scalar', deltas=None, times=None, verbose=False, indiv=False, out=None, f_every=None, time_hist=False, origin_only=False, all_origins=False,legendres=2):

    # # Process the inputs
    # if isinstance(values, dict):
    #     if isinstance(values.iterkeys().next(), int): # checks if keys are int (implies velocity) # Works for python 2, might need to change for python 3.
    #         lengths = [ len(i) for i in values.values() ]
    #     else:
    #         lengths = [ len(i) for i in values ] # if not int, probably tuples from vector
    # else:
    #     lengths = [ len(i) for i in values ]

    try:
        if isinstance(values.iterkeys().next(), int): # checks if keys are int (implies velocity) # Works for python 2, might need to change for python 3.
            lengths = [len(i) for i in values.values()]
        else:
            lengths = [len(i) for i in values]
    except:
        lengths = [len(i) for i in values]

    if False in [ lengths[0] == i for i in lengths ]:
        print("ERROR in process_autocorr: all values lists must have the same number of elements.")
        quit()
    if algorithm not in ['scalar','P1','P2']:
        print("ERROR in process_autocorr: algorithm argument must be either 'scalar', 'P1', 'P2'")
        quit()

    if time_hist:
        t_h_bin_min = -1
        t_h_bin_max = 1
        t_h_bin_step = 0.1
        t_h_n_bins = 100
        t_h_bins = [i for i in range(t_h_n_bins+1)]
        time_dict = {}

    # Process individual autocorrelations
    acs = []
    if isinstance(values, dict): #for velocity or force
        for i,j in values.items():
            if verbose:
                print("Working on molecule {}".format(i))
            acs += [calc_autocorr(j,algorithm=algorithm,deltas=deltas,times=times,legendres=legendres)]
    else: #for vector
        for count_i,i in enumerate(values):
            if verbose:
                print("Working on molecule {}".format(count_i))
            if time_hist:
                temp_ac, temp_time_dict = calc_autocorr(i, algorithm=algorithm, deltas=deltas, times=times, indiv=indiv, num=count_i, out=out, f_every=f_every, origin_only=origin_only,time_hist=time_hist,legendres=legendres)
                acs += [temp_ac]
                if deltas is not None:
                    if count_i == 0:
                        for count_t, t in enumerate(deltas):
                            t_hist,t_bins = np.histogram(temp_time_dict[t],bins=101,range=[-1.01,1.01])
                            time_dict[t] = t_hist
                    else:
                        for count_t, t in enumerate(deltas):
                            t_hist,t_bins = np.histogram(temp_time_dict[t],bins=101,range=[-1.01,1.01])
                            time_dict[t] += t_hist
                else:
                    if count_i == 0:
                        for t in range(0,len(i)*f_every,f_every):
                            t_hist,t_bins = np.histogram(temp_time_dict[t][temp_time_dict[t]!=0],bins=101,range=[-1.01,1.01])
                            time_dict[t] = t_hist
                    else:
                        for t in range(0,len(i)*f_every,f_every):
                            t_hist,t_bins = np.histogram(temp_time_dict[t][temp_time_dict[t]!=0],bins=101,range=[-1.01,1.01])
                            time_dict[t] += t_hist
            else:
                acs += [calc_autocorr(i, algorithm=algorithm, deltas=deltas, times=times, indiv=indiv, num=count_i, out=out, f_every=f_every, origin_only=origin_only,time_hist=time_hist,legendres=legendres)]
            if indiv:
                write_cols("{}/{}".format(out, "{}-ac-{}.txt".format(count_i,algorithm)),[[j * f_every for j in range(len(acs[count_i]))], acs[count_i]],labels=["frame", "autocorrelation"])

    # Generate an average correlation plot from all instances
    if deltas is not None:
        ac = np.zeros((legendres,len(deltas)))
    else:
        ac = np.zeros((legendres,lengths[0]))

    if time_hist and deltas is not None:
        t_hist_vals = np.vstack(acs)
        t_bins += (t_bins[1]-t_bins[0])/2.
        for count_i, t in enumerate(deltas):
            write_cols("{}/{}".format(out,"{}-hist-{}.txt".format(t,algorithm)), [t_bins[1:],time_dict[t]],labels=["bins","hist"])
            write_cols("{}/{}".format(out,"{}-hist-ids-{}.txt".format(t,algorithm)), [t_h_bins,[list(np.where(np.floor((t_hist_vals[:,count_i]-(t_h_bin_min))/(t_h_bin_step))==bin_loc)[0]) for bin_loc in range(t_h_n_bins+1)]],labels=["bins","hist"],non_float=True)
            
    elif time_hist:
        t_hist_vals = np.vstack(acs)
        for count_i in time_dict.keys():
            #t_hist,t_bins = np.histogram(t_hist_vals[:,count_i],bins=100,range=[-1.,1.])
            t_bins += (t_bins[1]-t_bins[0])/2.
            write_cols("{}/{}".format(out,"{}-hist-{}.txt".format(count_i,algorithm)), [t_bins[1:],time_dict[count_i]],labels=["bins","hist"])
            write_cols("{}/{}".format(out,"{}-hist-ids-{}.txt".format(count_i,algorithm)), [t_h_bins,[list(np.where(np.floor((t_hist_vals[:,count_i//f_every]-(t_h_bin_min))/(t_h_bin_step))==bin_loc)[0]) for bin_loc in range(t_h_n_bins+1)]],labels=["bins","hist"],non_float=True)

    for i in acs: ac += i
    ac = ac/float(len(acs))

    # Generate a stddev error estimate across all instances
    if deltas is not None:
        errs = np.zeros((legendres,len(deltas)))
    else:
        errs = np.zeros((legendres,lengths[0]))
    for i in acs: errs += i**2.0
    errs = errs/float(len(acs))
    errs[1:] = abs(( errs[1:] - ac[1:]**(2.0) ))**(0.5) #/ (float(len(acs[1:])))**(0.5) # Added absolute value to correct negative values DF
    errs[0] = 0.0

    if isinstance(values, dict):
        return ac[0],errs[0]
    return ac,errs

#
# Rewrite to only calc once, allow for more efficient computation of both P1 and P2.
#
# Description: Calculates the autocorrelation of a time-series of values    
def calc_autocorr(values,algorithm='scalar',deltas=None, times=None, indiv=False, num=None, out=None, f_every=None, origin_only=False, time_hist=False, legendres=2):
    # print("times", times)
    # Process inputs
    if algorithm not in ['scalar','P1','P2']:
        print("ERROR in calc_autocorr: algorithm argument must be either 'scalar', 'P1', 'P2'")
        quit()

    # Scalar algorithm
    if algorithm == "scalar":

        # Initialize values
        avg    = mean(values)
        var    = std(values)**(2.0)    
        ac     = np.zeros(len(values))
        counts = np.zeros(len(values))
        N_vals = len(values)

        # Internal averaging of scalar autocorrelation 
        # Loop over unique origins
        for count_i,i in enumerate(values):

            # Calculate deviation at origin
            ac[:N_vals-count_i] += (values[count_i:]-avg) * (values[count_i] - avg)
            counts[0:N_vals-count_i] += 1

        ac = ac / var
        ac = ac / counts

    # P1 (Legendre-vac P1) algorithm
    if time_hist:
        time_vals = {}
        if deltas is not None:
            for t in deltas:
                time_vals[t] = []
        else:
            for t in range(0,len(values)*f_every,f_every):
                time_vals[t] = []
    if algorithm == "P1":
        #print("is it here?? in P1?")
        if not origin_only:
            # Internal averaging of dot products
            # Loop over unique origins
            if deltas is not None:
                ac = np.zeros((len(deltas),legendres))
                counts = np.zeros(len(deltas))
                # print("Values: ", values)
                print(type(values), np.size(values[0]))
                for count_i,i in enumerate(values): # origin
                    for count_j,j in enumerate(values): # shifted vector
                        # if time_hist:
                        # # Need to report each instance of each cos theta, not averaged over all time for each molecule
                        #     a=1
                        # else:
                        
                        if times[count_j]-times[count_i] in deltas: # Check if vectors are offset by the correct amount
                            costheta = np.dot(values[count_i],values[count_j])
                            if time_hist:
                                time_vals[times[count_j]-times[count_i]].append(costheta)
                            for leg_n in range(1,legendres+1):
                                ac[np.where(deltas == (times[count_j]-times[count_i]))[0][0],leg_n-1] += np.polyval(scipy.special.legendre(leg_n),costheta)
                            counts[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += 1
                        else:
                            continue
            else:
                ac = np.zeros((len(values),legendres))
                counts = np.zeros(len(values))
                N_vals = len(values)
                time_vals = np.zeros((len(values),len(values)))
                for count_i,i in enumerate(values):
                    costheta = np.dot(values[count_i:],values[count_i])
                    if time_hist:
                        time_vals[count_i,:len(costheta)] = costheta
                    for leg_n in range(1,legendres+1):
                        ac[:N_vals-count_i,leg_n-1] += np.polyval(scipy.special.legendre(leg_n),costheta)
                    if count_i == 0 and indiv:
                        write_cols("{}/{}".format(out, "{}-raw_thetaphi.txt".format(num)),[[j * f_every for j in range(len(ac))], np.arctan2(np.array(values)[:,1],np.array(values)[:,0]), np.arccos(np.array(values)[:,2]), ac],labels=["frame", "theta", "phi", "costheta"])
                    counts[:N_vals-count_i] += 1
        else:
            # Internal averaging of dot products
            # Loop over unique origins
            if deltas is not None:
                ac = np.zeros(len(deltas))
                counts = np.zeros(len(deltas))
                count_i, i = 0, values[0]
                for count_j,j in enumerate(values): # shifted vector
                    ac[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += np.dot(values[count_i],values[count_j])
                    counts[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += 1
            else:
                ac = np.zeros(len(values))
                counts = np.zeros(len(values))
                N_vals = len(values)
                #for count_i,i in enumerate(values):
                count_i,i = 0,values[0]
                ac[:N_vals-count_i] += np.dot(values[count_i:],values[count_i])
                if count_i == 0 and indiv:
                    write_cols("{}/{}".format(out, "{}-raw_thetaphi.txt".format(num)),[[j * f_every for j in range(len(ac))], np.arctan2(np.array(values)[:,1],np.array(values)[:,0]), np.arccos(np.array(values)[:,2]), ac],labels=["frame", "theta", "phi", "costheta"])
                counts[:N_vals-count_i] += 1

        #counts = counts / legendres
        ac = (ac.T / counts)

    # P2 (Legendre-vac P2) algorithm
    if algorithm == "P2":

        if deltas is not None:
            ac = np.zeros(len(deltas))
            counts = np.zeros(len(deltas))
            for count_i,i in enumerate(values): # origin
                for count_j,j in enumerate(values): # shifted vector
                    if times[count_j]-times[count_i] in deltas: # Check if vectors are offset by the correct amount
                        #print("is this where the script is even going?")
                        ac[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += (3.0*(np.dot(values[count_i],values[count_j])**2) - 1.0)/2.0
                        counts[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += 1
                    else:
                        continue

        else:
            # Initialize values
            ac     = np.zeros(len(values))
            counts = np.zeros(len(values))
            N_vals = len(values)

            # Internal averaging of dot products convolved with Langrange function P2
            # Loop over unique origins
            for count_i,i in enumerate(values):
                ac[:N_vals-count_i] += ( 3.0*np.dot(values[count_i:],values[count_i])**(2.0) - 1.0 ) / 2.0
                counts[:N_vals-count_i] += 1

        # Perform average
        ac = ac / counts
        ac = ac / ac[0]

    if time_hist and deltas is None:
        time_dict = {}
        for i in range(len(values)):
            time_dict[i*f_every] = time_vals[:,i]
        return ac,time_dict

    if time_hist:
        return ac,time_vals
    else:
        return ac

### Dylan

def process_scalord(values,algorithm='scalar', deltas=None, times=None, verbose=False, indiv=False, out=None, f_every=None, time_hist=False, origin_only=False, all_origins=False,legendres=2):

    # # Process the inputs
    # if isinstance(values, dict):
    #     if isinstance(values.iterkeys().next(), int): # checks if keys are int (implies velocity) # Works for python 2, might need to change for python 3.
    #         lengths = [ len(i) for i in values.values() ]
    #     else:
    #         lengths = [ len(i) for i in values ] # if not int, probably tuples from vector
    # else:
    #     lengths = [ len(i) for i in values ]

    try:
        if isinstance(values.iterkeys().next(), int): # checks if keys are int (implies velocity) # Works for python 2, might need to change for python 3.
            lengths = [len(i) for i in values.values()]
        else:
            lengths = [len(i) for i in values]
    except:
        lengths = [len(i) for i in values]

    if False in [ lengths[0] == i for i in lengths ]:
        print("ERROR in process_autocorr: all values lists must have the same number of elements.")
        quit()
    if algorithm not in ['scalar','P1','P2']:
        print("ERROR in process_autocorr: algorithm argument must be either 'scalar', 'P1', 'P2'")
        quit()

    if time_hist:
        t_h_bin_min = -1
        t_h_bin_max = 1
        t_h_bin_step = 0.1
        t_h_n_bins = 100
        t_h_bins = [i for i in range(t_h_n_bins+1)]
        time_dict = {}

    # Process individual autocorrelations
    acs = []
        # Calculate order vectors:
    n_steps=len(values[0])
    n_moles=len(values)
    director_vec=[]
    for q in range(n_steps):
        sum_vec=[0, 0, 0]
        for m in values:
            m_mag=np.linalg.norm(m[q])
            sum_vec+=m[q]/m_mag
        director_vec.append(sum_vec/np.linalg.norm(sum_vec))    # normalize. We want a unit vector, so dividing by the number of molecules doesn't actually matter.
    # print("director_vec", director_vec)

    if isinstance(values, dict): #for velocity or force
        for i,j in values.items():
            if verbose:
                print("Working on molecule {}".format(i))
                print("from is instance")
            acs += [calc_scalarorder(j, director_vec, algorithm=algorithm,deltas=deltas,times=times,legendres=legendres)]

    else: #for vector
        for count_i,i in enumerate(values):
            if verbose:
                print("Working on molecule {}".format(count_i))
            if time_hist:
                # print("from time_hist")
                temp_ac, temp_time_dict = calc_scalarorder(i, director_vec, algorithm=algorithm, deltas=deltas, times=times, indiv=indiv, num=count_i, out=out, f_every=f_every, origin_only=origin_only,time_hist=time_hist,legendres=legendres)
                acs += [temp_ac]
                if deltas is not None:
                    if count_i == 0:
                        for count_t, t in enumerate(deltas):
                            t_hist,t_bins = np.histogram(temp_time_dict[t],bins=101,range=[-1.01,1.01])
                            time_dict[t] = t_hist
                    else:
                        for count_t, t in enumerate(deltas):
                            t_hist,t_bins = np.histogram(temp_time_dict[t],bins=101,range=[-1.01,1.01])
                            time_dict[t] += t_hist
                else:
                    if count_i == 0:
                        for t in range(0,len(i)*f_every,f_every):
                            t_hist,t_bins = np.histogram(temp_time_dict[t][temp_time_dict[t]!=0],bins=101,range=[-1.01,1.01])
                            time_dict[t] = t_hist
                    else:
                        for t in range(0,len(i)*f_every,f_every):
                            t_hist,t_bins = np.histogram(temp_time_dict[t][temp_time_dict[t]!=0],bins=101,range=[-1.01,1.01])
                            time_dict[t] += t_hist
            else:
                # print("from else")
                acs += [calc_scalarorder(i, director_vec, algorithm=algorithm, deltas=deltas, times=times, indiv=indiv, num=count_i, out=out, f_every=f_every, origin_only=origin_only,time_hist=time_hist,legendres=legendres)]
            if indiv:
                write_cols("{}/{}".format(out, "{}-ac-{}.txt".format(count_i,algorithm)),[[j * f_every for j in range(len(acs[count_i]))], acs[count_i]],labels=["frame", "autocorrelation"])

    # Generate an average correlation plot from all instances
    if deltas is not None:
        ac = np.zeros((legendres,len(deltas)))
    else:
        ac = np.zeros((legendres,lengths[0]))

    if time_hist and deltas is not None:
        t_hist_vals = np.vstack(acs)
        t_bins += (t_bins[1]-t_bins[0])/2.
        for count_i, t in enumerate(deltas):
            write_cols("{}/{}".format(out,"{}-hist-{}.txt".format(t,algorithm)), [t_bins[1:],time_dict[t]],labels=["bins","hist"])
            write_cols("{}/{}".format(out,"{}-hist-ids-{}.txt".format(t,algorithm)), [t_h_bins,[list(np.where(np.floor((t_hist_vals[:,count_i]-(t_h_bin_min))/(t_h_bin_step))==bin_loc)[0]) for bin_loc in range(t_h_n_bins+1)]],labels=["bins","hist"],non_float=True)
            
    elif time_hist:
        t_hist_vals = np.vstack(acs)
        for count_i in time_dict.keys():
            #t_hist,t_bins = np.histogram(t_hist_vals[:,count_i],bins=100,range=[-1.,1.])
            t_bins += (t_bins[1]-t_bins[0])/2.
            write_cols("{}/{}".format(out,"{}-hist-{}.txt".format(count_i,algorithm)), [t_bins[1:],time_dict[count_i]],labels=["bins","hist"])
            write_cols("{}/{}".format(out,"{}-hist-ids-{}.txt".format(count_i,algorithm)), [t_h_bins,[list(np.where(np.floor((t_hist_vals[:,count_i//f_every]-(t_h_bin_min))/(t_h_bin_step))==bin_loc)[0]) for bin_loc in range(t_h_n_bins+1)]],labels=["bins","hist"],non_float=True)

    for i in acs: ac += i
    ac = ac/float(len(acs))

    # Generate a stddev error estimate across all instances
    if deltas is not None:
        errs = np.zeros((legendres,len(deltas)))
    else:
        errs = np.zeros((legendres,lengths[0]))
    for i in acs: errs += i**2.0
    errs = errs/float(len(acs))
    errs[1:] = abs(( errs[1:] - ac[1:]**(2.0) ))**(0.5) #/ (float(len(acs[1:])))**(0.5) # Added absolute value to correct negative values DF
    errs[0] = 0.0

    if isinstance(values, dict):
        return ac[0],errs[0]
    director_vec = np.array(director_vec)
    return ac,errs,director_vec

### Dylan

#############
#
# Rewrite to only calc once, allow for more efficient computation of both P1 and P2.
#
# Description: Calculates the scalar order parameter of the system.   
def calc_scalarorder(values, director_vec, algorithm='scalar',deltas=None, times=None, indiv=False, num=None, out=None, f_every=None, origin_only=False, time_hist=False, legendres=2):
    # print("Calculating Scalar Order Parameter")
    # print("Director: ", director_vec)
    # Process inputs
    if algorithm not in ['scalar','P1','P2']:
        print("ERROR in calc_autocorr: algorithm argument must be either 'scalar', 'P1', 'P2'")
        quit()
    print("director: ", director_vec)
    # Scalar algorithm
    if algorithm == "scalar":

        # Initialize values
        avg    = mean(values)
        var    = std(values)**(2.0)    
        ac     = np.zeros(len(values))
        counts = np.zeros(len(values))
        N_vals = len(values)

        # Internal averaging of scalar autocorrelation 
        # Loop over unique origins
        for count_i,i in enumerate(values):

            # Calculate deviation at origin
            ac[:N_vals-count_i] += (values[count_i:]-avg) * (values[count_i] - avg)
            counts[0:N_vals-count_i] += 1

        ac = ac / var
        ac = ac / counts

    # P1 (Legendre-vac P1) algorithm
    if time_hist:
        time_vals = {}
        if deltas is not None:
            for t in deltas:
                time_vals[t] = []
        else:
            for t in range(0,len(values)*f_every,f_every):
                time_vals[t] = []
    if algorithm == "P1":
        print("in P1")
        if not origin_only:
            # Internal averaging of dot products
            # Loop over unique origins
            if deltas is not None:
                ac = np.zeros((len(deltas),legendres))
                counts = np.zeros(len(deltas))
                for count_i,i in enumerate(values): # origin
                    print("Line 1524")
                    # if time_hist:
                    # # Need to report each instance of each cos theta, not averaged over all time for each molecule
                    #     a=1
                    # else:
                    if times[count_j]-times[count_i] in deltas: # Check if vectors are offset by the correct amount
                        costheta = np.dot(values[count_i],director_vec[count_i])
                        if time_hist:
                            time_vals[times[count_j]-times[count_i]].append(costheta)
                        for leg_n in range(1,legendres+1):
                            ac[np.where(deltas == (times[count_j]-times[count_i]))[0][0],leg_n-1] += np.polyval(scipy.special.legendre(leg_n),costheta)
                            counts[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += 1
                        else:
                            continue
            else:
                print("Line 1539")
                ac = np.zeros((len(values),legendres))
                counts = np.zeros(len(values))
                N_vals = len(values)
                time_vals = np.zeros((len(values),len(values)))
                for count_i,i in enumerate(values):
                    costheta = np.dot(values[count_i],director_vec[count_i])
                    # if time_hist:
                    #     time_vals[count_i,:len(costheta)] = costheta
                    for leg_n in range(1,legendres+1):
                        ac[count_i,leg_n-1] += np.polyval(scipy.special.legendre(leg_n),costheta)
                    # if count_i == 0 and indiv:
                    #     write_cols("{}/{}".format(out, "{}-raw_thetaphi.txt".format(num)),[[j * f_every for j in range(len(ac))], np.arctan2(np.array(values)[:,1],np.array(values)[:,0]), np.arccos(np.array(values)[:,2]), ac],labels=["frame", "theta", "phi", "costheta"])
                    counts[count_i] += 1
        else:
            # Internal averaging of dot products
            # Loop over unique origins
            if deltas is not None:
                print("Line 1557")
                ac = np.zeros(len(deltas))
                counts = np.zeros(len(deltas))
                count_i, i = 0, values[0]
                ac[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += np.dot(values[count_i],director_vec[count_i])
                counts[np.where(deltas == (times[count_i]-times[count_i]))[0][0]] += 1
            else:
                ac = np.zeros(len(values))
                print("Line 1565")
                counts = np.zeros(len(values))
                N_vals = len(values)
                #for count_i,i in enumerate(values):
                count_i,i = 0,values[0]
                ac[:N_vals-count_i] += np.dot(values[count_i:],director_vec[count_i])
                if count_i == 0 and indiv:
                    write_cols("{}/{}".format(out, "{}-raw_thetaphi.txt".format(num)),[[j * f_every for j in range(len(ac))], np.arctan2(np.array(values)[:,1],np.array(values)[:,0]), np.arccos(np.array(values)[:,2]), ac],labels=["frame", "theta", "phi", "costheta"])
                counts[:N_vals-count_i] += 1

        #counts = counts / legendres
        ac = (ac.T / counts)

    # P2 (Legendre-vac P2) algorithm
    if algorithm == "P2":
        print("In P2")
        # if deltas is not None:
        #     ac = zeros(len(deltas))
        #     counts = zeros(len(deltas))
        #     print("values: ", values)
        #     for count_i,i in enumerate(values): # origin
        #         for count_j,j in enumerate(values): # shifted vector
        #             if times[count_j]-times[count_i] in deltas: # Check if vectors are offset by the correct amount
        #                 ac[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += (3.0*(dot(values[count_i],director_vec[count_i])**2) - 1.0)/2.0
        #                 counts[np.where(deltas == (times[count_j]-times[count_i]))[0][0]] += 1
        #             else:
        #                 continue

        # else:
        # Initialize values
        ac     = np.zeros(len(values))
        counts = np.zeros(len(values))
        N_vals = len(values)
        print("Attempting calculation of scalar order...")
        # Internal averaging of dot products convolved with Langrange function P2
        # Loop over unique origins
        for count_i,i in enumerate(values):
            ac[:N_vals-count_i] += ( 3.0*np.dot(values[count_i],director_vec[count_i])**(2.0) - 1.0 ) / 2.0
            counts[:N_vals-count_i] += 1

        # Perform average
        ac = ac / counts
        ac = ac / ac[0]

    if time_hist and deltas is None:
        time_dict = {}
        for i in range(len(values)):
            time_dict[i*f_every] = time_vals[:,i]
        return ac,time_dict

    if time_hist:
        return ac,time_vals
    else:
        return ac

##########

def process_crosscorr(parse_dict_i, deltas=None, vel=False, normed=True):

    end2end_t = []
    #vsq_t = []

    if deltas is not None:
        for count_t, t in enumerate(deltas):
            e2e = np.array([0., 0., 0.])
            # if vel is True and normed is False:
            #     vsq = 0
            #     for mol_i in parse_dict_i:
            #         vsq += np.linalg.norm(np.array(parse_dict_i[mol_i][count_t]))
            #     vsq_t.append(vsq)
            for mol_i in parse_dict_i:  # np.sum(parse_dict_i[""]...[],axis=0)
                # take t-th time point for each molecule and sum
                e2e += np.array(parse_dict_i[mol_i][count_t])  # Not as straightforward as an axis sum, need a little off-axis?

            end2end_t.append(e2e)

        end2end_t = np.array(end2end_t)

        xc = np.zeros(len(deltas))
        counts = np.zeros(len(deltas))



        # for t_origin_index, t_origin in enumerate(deltas):
        #     for t_shifted_index, t_shifted in enumerate(deltas):
        #         if t_shifted-t_origin in deltas:
        #             xc[np.where(deltas == (t_shifted-t_origin))[0][0]] += dot(end2end_t[t_origin_index],end2end_t[t_shifted_index])
        #             counts[np.where(deltas == (t_shifted-t_origin))[0][0]] += 1

        xc = np.sum(end2end_t * end2end_t[0], axis=1)
        counts = 1

        # xc = []
        # for t in range(len(deltas)):
        #     xc.append(np.dot(end2end_t[t],end2end_t[0]))

        xc = xc / len(parse_dict_i)
        xc = xc / counts
        # if vel is True and normed is False:
        #     xc = xc / vsq

    else:
        for t in range(len(parse_dict_i[next(iter(parse_dict_i))])):
            e2e = np.array([0., 0., 0.])
            for mol_i in parse_dict_i:
                e2e += np.array(parse_dict_i[mol_i][t])

            end2end_t.append(e2e)

        end2end_t = np.array(end2end_t)
        N_vals = len(end2end_t)
        xc = np.zeros(N_vals)
        counts = np.zeros(N_vals)

        for count_i in range(len(end2end_t)):
            xc[:N_vals - count_i] += np.dot(end2end_t[count_i:], end2end_t[count_i])
            counts[:N_vals - count_i] += 1
        #xc = np.sum(end2end_t * end2end_t[0], axis=1)

        # xc = []
        # for t in range(len(end2end_t)):
        #     xc.append(np.dot(end2end_t[t], end2end_t[0]))

        xc = np.array(xc) / len(parse_dict_i)  # divide by number of molecules
        xc = xc / counts

    print(end2end_t)
    return xc


def process_crosscorr_subset(parse_dict_i, location_dict_i=None, deltas=None, vel=False, normed=True, subboxes=None, radius=None, vel_atom=None):

    xc_s = []
    xc_s_corr = []

    #vsq_t = []

    box = location_dict_i["box"]
    bx_2 = (box[0, 1] - box[0, 0]) / 2.0
    by_2 = (box[1, 1] - box[1, 0]) / 2.0
    bz_2 = (box[2, 1] - box[2, 0]) / 2.0

    box_2 = [bx_2,by_2,bz_2]

    box_lower = np.array([box[0,0],box[1,0],box[2,0]])

    box_full = np.array([bx_2*2,by_2*2,bz_2*2])
    if subboxes is not None:
        divs = int(round(int(subboxes)**(1./3.)))

    # sort subsets by location
    subsets = {}
    subsets_t = {}


    if subboxes is not None: # sort by cubic subboxes

        for count_t, t in enumerate(deltas):
            for i in range(int(subboxes)):
                subsets[i] = []
            for i in location_dict_i:
                if i != "box":
                    if vel_atom is None:
                        for coord in range(3): # periodic wrapping, 3 = dimensionality
                            if location_dict_i[i][count_t][coord] > box[coord,1]:
                                location_dict_i[i][count_t][coord] -= box[coord,1]-box[coord,0]
                            elif location_dict_i[i][count_t][coord] < box[coord,0]:
                                location_dict_i[i][count_t][coord] += box[coord,1]-box[coord,0]
                        temp_loc = (location_dict_i[i][count_t] - box_lower) / (box_full / divs ) # normalize to make it easier to divide up the box
                        subsets[int(temp_loc[0])+int(temp_loc[1])*divs+int(temp_loc[2])*divs**2].append(i) # assign to the unique subset using a number base divs
                    elif i%int(vel_atom[0])==int(vel_atom[1]):
                        for coord in range(3): # periodic wrapping
                            if location_dict_i[i][count_t][coord] > box[coord,1]:
                                location_dict_i[i][count_t][coord] -= box[coord,1]-box[coord,0]
                            elif location_dict_i[i][count_t][coord] < box[coord,0]:
                                location_dict_i[i][count_t][coord] += box[coord,1]-box[coord,0]
                        temp_loc = (location_dict_i[i][count_t] - box_lower) / (box_full / divs ) # normalize to make it easier to divide up the box
                        subsets[int(temp_loc[0])+int(temp_loc[1])*divs+int(temp_loc[2])*divs**2].append(i) # assign to the unique subset using a number base divs

            subsets_t[t] = subsets

    if radius is not None: # sort subsets by atoms within a given radius

        # cdist solution which incorporates periodic boundary (but only in 3D)
        # d = scipy.spatial.distance.cdist(dist, dist, lambda u, v: np.sqrt((min(u[0] - v[0],box[0] - np.abs((u[0] - v[0])),key=abs) ** 2 + min(u[1] - v[1],box[1] - np.abs((u[1] - v[1])),key=abs) ** 2 + min(u[2] - v[2], box[2] - np.abs((u[2] - v[2])), key=abs) ** 2).sum()))
        # rad = 2.1
        # mask = d <= rad

        N_subset_avg_t = []

        for count_t, t in enumerate(deltas):
            N_subset_avg = 0
            for i in range(1, len(parse_dict_i) + 1):  # every particle gets its own subset
                subsets[i] = []
            for i in location_dict_i:
                for j in location_dict_i:

                    if i=="box" or j=="box":
                        continue
                    if j > i:
                        continue

                    if vel_atom is None:
                        disp = np.zeros(3)
                        for coord in range(3):
                            temp_calc = location_dict_i[i][count_t][coord] - location_dict_i[j][count_t][coord]

                            if temp_calc > box_2[coord]: #unwrapping
                                temp_calc -= 2*box_2[coord]
                            elif temp_calc < -box_2[coord]:
                                temp_calc += 2*box_2[coord]

                            disp[coord] = temp_calc
                        if np.linalg.norm(disp) < float(radius):
                            subsets[i].append(j)
                            subsets[j].append(i) # symmetry

                    elif i%int(vel_atom[0])==int(vel_atom[1]):
                        disp = np.zeros(3)
                        for coord in range(3):
                            temp_calc = location_dict_i[i][count_t][coord] - location_dict_i[j][count_t][coord]

                            if temp_calc > box_2[coord]:  # unwrapping
                                temp_calc -= 2 * box_2[coord]
                            elif temp_calc < box_2[coord]:
                                temp_calc += 2 * box_2[coord]

                            disp[coord] = temp_calc
                        if np.linalg.norm(disp) < float(radius):
                            subsets[i].append(j)

            N_subsets = [len(subsets[i]) for i in subsets]
            N_subset_avg_t.append(float(sum(N_subsets)) / len(N_subsets))

            subsets_t[t] = subsets
        #N_subset_avg = N_subset_avg / len(deltas)


# Compute full correlator:

    if deltas is not None:
            vsqs = []
            for subs in subsets: #range(int(subboxes)):
                end2end_t = []
                xc = np.zeros(len(deltas))
                xc_corr = np.zeros(len(deltas))
                for count_t, t in enumerate(deltas):
                    e2e = np.array([0., 0., 0.])
                    vsq = 0
                    # if vel is True and normed is False:
                    #     vsq = 0
                    #     for mol_i in subsets_t[t][subs]:
                    #         vsq += np.linalg.norm(np.array(parse_dict_i[mol_i][count_t]))
                    #     vsq_t.append(vsq)
                    for mol_i in subsets_t[t][subs]:  # np.sum(parse_dict_i[""]...[],axis=0)
                        # take t-th time point for each molecule and sum
                        e2e += np.array(parse_dict_i[mol_i][count_t])  # Not as straightforward as an axis sum, need a little off-axis?
                        vsq += np.linalg.norm(parse_dict_i[mol_i][count_t])**2

                    #print(vsq)

                    if count_t == 0:
                        xc[count_t] = np.sum(e2e*e2e)
                        sub_corr = 1. #/ len(parse_dict_i) * (len(parse_dict_i) - len(subsets_t[t][subs]))  # 1/N*(N-n)
                        if np.isnan(xc[count_t]):
                            print("nan in box {} at time {}".format(subs, t))
                            xc[count_t] = 0
                            xc_corr[count_t] = 0
                        else:
                            xc[count_t] = xc[count_t] / vsq
                            xc_corr[count_t] = xc[count_t] / sub_corr
                            vsqs.append(vsq)
                        # xc[count_t] = xc[count_t]
                    else:
                        temp_xc = np.sum(e2e*end2end_t[0])
                        sub_corr = 1. #/ len(parse_dict_i) * (len(parse_dict_i) - len(subsets_t[t][subs]))  # 1/N*(N-n)
                        # temp_xc = temp_xc
                        if not np.isnan(temp_xc):
                            xc[count_t] += temp_xc / vsq
                            xc_corr[count_t] += xc[count_t] / sub_corr
                        else:
                            print("nan in box {} at time {}".format(subs,t))


                    end2end_t.append(e2e)


                #end2end_t = np.array(end2end_t)

                xc_s.append(xc)
                xc_s_corr.append(xc_corr)


                #xc = np.zeros(len(deltas))
                #counts = np.zeros(len(deltas))
            #print(np.mean(np.array(vsqs)))



                # for t_origin_index, t_origin in enumerate(deltas):
                #     for t_shifted_index, t_shifted in enumerate(deltas):
                #         if t_shifted-t_origin in deltas:
                #             xc[np.where(deltas == (t_shifted-t_origin))[0][0]] += dot(end2end_t[t_origin_index],end2end_t[t_shifted_index])
                #             counts[np.where(deltas == (t_shifted-t_origin))[0][0]] += 1

                # xc = np.sum(end2end_t * end2end_t[0], axis=1)
                # counts = 1
                #
                # sub_corr = 1./len(parse_dict_i)*(len(parse_dict_i) - len(subsets_t[t][subs])) # 1/N*(N-n)

                # xc = []
                # for t in range(len(deltas)):
                #     xc.append(np.dot(end2end_t[t],end2end_t[0]))

                # xc = xc / len(subsets_t[t][subs])
                # xc = xc / counts
                # xc = xc / sub_corr

                # if vel is True and normed is False:
                #     xc = xc / vsq

                #xc_s.append(xc)


    # else:
    #     for t in range(len(parse_dict_i[next(iter(parse_dict_i))])):
    #         e2e = np.array([0., 0., 0.])
    #         for mol_i in parse_dict_i:
    #             e2e += np.array(parse_dict_i[mol_i][t])
    #
    #         end2end_t.append(e2e)
    #
    #     end2end_t = np.array(end2end_t)
    #     N_vals = len(end2end_t)
    #     xc = np.zeros(N_vals)
    #     counts = np.zeros(N_vals)
    #
    #     for count_i in range(len(end2end_t)):
    #         xc[:N_vals - count_i] += dot(end2end_t[count_i:], end2end_t[count_i])
    #         counts[:N_vals - count_i] += 1
    #     #xc = np.sum(end2end_t * end2end_t[0], axis=1)
    #
    #     # xc = []
    #     # for t in range(len(end2end_t)):
    #     #     xc.append(np.dot(end2end_t[t], end2end_t[0]))
    #
    #     xc = np.array(xc) / len(parse_dict_i)  # divide by number of molecules
    #     xc = xc / counts

    print(end2end_t)

    if radius is not None:
        return xc_s, N_subset_avg_t
    return xc_s


def process_crosscorr_pairwise(parse_dict_i, deltas=None, r=None, location_dict_i=None, hist=False, double_hist=False, name=None):

    histogram = None

    if deltas is not None:

        xc = np.zeros(len(deltas))
        counts = np.zeros(len(deltas))
        if hist:
            pair_list = {} # holds all dot products calculated so that we can histogram them later. Keys are times, values are lists of dot products
        if double_hist:
            double_pair_list = {}
            double_loc_list = {}

        # if r is None:
        #     for t_origin_index, t_origin in enumerate(deltas):
        #         for t_shifted_index, t_shifted in enumerate(deltas):
        #             for mol_i in parse_dict_i:
        #                 for mol_j in parse_dict_i:
        #                     if t_shifted-t_origin in deltas:
        #                         temp_dot = dot(parse_dict_i[mol_i][t_origin_index], parse_dict_i[mol_j][t_shifted_index])
        #                         xc[np.where(deltas == (t_shifted-t_origin))[0][0]] += temp_dot
        #                         counts[np.where(deltas == (t_shifted-t_origin))[0][0]] += 1
        if r is None:
            for t_index,t in enumerate(deltas):
                if hist:
                    pair_list[t] = []
                for mol_i in parse_dict_i:
                    for mol_j in parse_dict_i:
                        temp_dot = np.dot(parse_dict_i[mol_i][0], parse_dict_i[mol_j][t_index])
                        xc[t_index] += temp_dot
                        counts[t_index] += 1
                        if hist:
                            pair_list[t].append(temp_dot)



        else:
            box = location_dict_i["box"]
            bx_2 = (box[0, 1] - box[0, 0]) / 2.0
            by_2 = (box[1, 1] - box[1, 0]) / 2.0
            bz_2 = (box[2, 1] - box[2, 0]) / 2.0
            # Time-origin shifting
            # for t_origin_index, t_origin in enumerate(deltas):
            #     for t_shifted_index, t_shifted in enumerate(deltas):
            #         for mol_i in parse_dict_i:
            #             for mol_j in parse_dict_i:
            #                 temp_loc_i = location_dict_i[mol_i][t_origin_index]
            #                 temp_loc_j = location_dict_i[mol_j][t_shifted_index]
            #
            #                 if (temp_loc_i[0] - temp_loc_j[0]) > bx_2:
            #                     temp_loc_i[0] -= (bx_2 * 2.0)
            #                 elif (temp_loc_i[0] - temp_loc_j[0]) < -bx_2:
            #                     temp_loc_i[0] += (bx_2 * 2.0)
            #                 if (temp_loc_i[1] - temp_loc_j[1]) > by_2:
            #                     temp_loc_i[1] -= (by_2 * 2.0)
            #                 elif (temp_loc_i[1] - temp_loc_j[1]) < -by_2:
            #                     temp_loc_i[1] += (by_2 * 2.0)
            #                 if (temp_loc_i[2] - temp_loc_j[2]) > bz_2:
            #                     temp_loc_i[2] -= (bz_2 * 2.0)
            #                 elif (temp_loc_i[2] - temp_loc_j[2]) < -bz_2:
            #                     temp_loc_i[2] += (bz_2 * 2.0)
            #
            #                 if norm(temp_loc_i - temp_loc_j) < r:
            #                     if t_shifted - t_origin in deltas:
            #                         temp_dot = dot(parse_dict_i[mol_i][t_origin_index], parse_dict_i[mol_j][t_shifted_index])
            #                         xc[np.where(deltas == (t_shifted - t_origin))[0][0]] += temp_dot
            #                         counts[np.where(deltas == (t_shifted - t_origin))[0][0]] += 1

            for t_index, t in enumerate(deltas):
                if hist:
                    pair_list[t] = []
                if double_hist:
                    double_pair_list[t] = []
                    double_loc_list[t] = []
                for mol_i in parse_dict_i:
                    for mol_j in parse_dict_i:
                        temp_loc_i = location_dict_i[mol_i][0]
                        temp_loc_j = location_dict_i[mol_j][t_index]

                        if (temp_loc_i[0] - temp_loc_j[0]) > bx_2:
                            temp_loc_i[0] -= (bx_2 * 2.0)
                        elif (temp_loc_i[0] - temp_loc_j[0]) < -bx_2:
                            temp_loc_i[0] += (bx_2 * 2.0)
                        if (temp_loc_i[1] - temp_loc_j[1]) > by_2:
                            temp_loc_i[1] -= (by_2 * 2.0)
                        elif (temp_loc_i[1] - temp_loc_j[1]) < -by_2:
                            temp_loc_i[1] += (by_2 * 2.0)
                        if (temp_loc_i[2] - temp_loc_j[2]) > bz_2:
                            temp_loc_i[2] -= (bz_2 * 2.0)
                        elif (temp_loc_i[2] - temp_loc_j[2]) < -bz_2:
                            temp_loc_i[2] += (bz_2 * 2.0)

                        if norm(temp_loc_i - temp_loc_j) < r:
                            temp_dot = np.dot(parse_dict_i[mol_i][0], parse_dict_i[mol_j][t_index])
                            xc[t_index] += temp_dot
                            counts[t_index] += 1
                            if hist and t==0:
                                pair_list[t].append(temp_dot)
                            if double_hist and t==0:
                                temp_double = (temp_dot, norm(temp_loc_i - temp_loc_j)) # (pair value, pair distance)
                                double_pair_list[t].append(temp_double)
                                #double_loc_list[t].append(norm(temp_loc_i - temp_loc_j))


        #xc = np.sum(end2end_t * end2end_t[0], axis=1)

        # xc = []
        # for t in range(len(deltas)):
        #     xc.append(np.dot(end2end_t[t],end2end_t[0]))

        xc = xc / len(parse_dict_i)
        #xc = xc / counts

        if hist:
            bins, hs = calc_hist(pair_list[0], h_min= -1, h_max=1, normalize=False, h_step=0.05)
            histogram = (bins,hs)
        if double_hist:
            max_length = sqrt(bx_2**2+by_2**2+bz_2**2)
            bins, hs = calc_2dhist(double_pair_list[0], x_min=-1, x_max=1, x_step=0.05, y_min=0, y_max=max_length, y_step=2)
            histogram = (bins,hs)
            plot2dhist(name,double_pair_list[0], max_length)


    else: # NOT YET IMPLEMENTED

        N_vals = len(parse_dict_i[next(iter(parse_dict_i))])
        xc = np.zeros(N_vals)
        histogram = (xc,xc)
        # counts = np.zeros(N_vals)
        #
        # for t in range(len(parse_dict_i[next(iter(parse_dict_i))])):
        #     xc[:N_vals - count_i] += dot(end2end_t[count_i:], end2end_t[count_i])
        #     counts[:N_vals - count_i] += 1
        # #xc = np.sum(end2end_t * end2end_t[0], axis=1)
        #
        # # xc = []
        # # for t in range(len(end2end_t)):
        # #     xc.append(np.dot(end2end_t[t], end2end_t[0]))
        #
        # xc = np.array(xc) / len(parse_dict_i)  # divide by number of molecules
        # xc = xc / counts

    return xc, histogram


# Description: Function for writing a histogram
def write_cols(name, values, labels=None, comment=None, non_float=False):
    # Check inputs
    print("name : ", name)
    print("values : ", values)
    if labels is not None:
        if len(values) != len(labels):
            print("ERROR in write_cols: lengths of 'values' and 'labels' lists must be equal.")
            quit()
    # Write file
    if non_float:
        with open(name, 'w') as f:
            if comment is not None:
                f.write("{}\n".format(comment))
            if labels is not None:
                f.write(" " + " ".join(["{:<40s}".format(i) for i in labels]) + "\n")
            for i in range(len(values[0])):
                f.write(" ".join(["{}".format(values[j][i]) for j in range(len(values))]) + "\n")
    else:
        with open(name, 'w') as f:
            if comment is not None:
                f.write("{}\n".format(comment))
            if labels is not None:
                f.write(" " + " ".join(["{:<40s}".format(i) for i in labels]) + "\n")
            for i in range(len(values[0])):
                f.write(" ".join(["{:< 40.12f}".format(values[j][i]) for j in range(len(values))]) + "\n")

# Description: Function for writing a histogram
def write_2dcols(name,values):

    bins = values[0]
    hist = values[1]

    # Write file
    with open(name,'w') as f:
        #f.write(" "+" ".join([ "{:<40s}".format(i) for i in labels ])+"\n")
        f.write("r\\dot ")
        f.write(" ".join([ "{:< 40.12f}".format(bins[0][i]) for i in range(len(bins[0])) ])+"\n")
        for i in range(len(bins[1])):
            f.write("{:< 40.12f} ".format(bins[1][i]))
            f.write(" ".join([ "{:< 40.12f}".format(hist[j][i]) for j in range(len(bins[0])) ])+"\n")


def plot2dhist(name, pc, max_length):

    # x = pair_contributions
    # y = distance

    pc_list,dist_list=zip(*pc)

    x_min = -1
    x_max = 1
    x_step = 0.05
    y_min = 0
    y_max = max_length
    y_step = 2

    pc_bins = np.array([x_min + x_step * i for i in range(int(round((x_max - x_min) / x_step)) + 1)])
    dist_bins = np.array([y_min + y_step * i for i in range(int(round((y_max - y_min) / y_step)) + 1)])
    bins = [pc_bins, dist_bins]

    mpl.hist2d(pc_list, dist_list, bins=bins, cmap=mpl.cm.jet)
    mpl.colorbar()

    mpl.savefig(name+"/2dhist.pdf")

    mpl.close()


def plot2dcrosssec(name, data):

    bins = data[0]
    hist = data[1]

    for count_r, r in enumerate(bins[1]):
        mpl.plot(bins[0], hist[:,count_r], label=r)
    mpl.xlabel("Dot product")
    mpl.ylabel("Counts")
    mpl.legend()

    mpl.savefig(name)
    mpl.close()

# Description: Function for calculating a histogram from an array or list of values
def calc_hist(values,h_min=None,h_max=None,h_step=1.0,shave=False,normalize=False,abs_opt=False):

    if abs_opt is True:
        values = [ abs(_) for _ in values ]

    # Check input arguments
    if h_step is None: h_step = float(max(values) - min(values))/100.0
    if h_min is None: h_min = min(values)-0.5*h_step
    if h_max is None: h_max = max(values)+0.5*h_step

    # Generate histogram
    h = np.zeros(int(round((h_max - h_min)/h_step)) + 1)

    for i in values:

        if i >= h_min and i <= h_max:
            h[int(float(i-h_min)/h_step)] += 1

    bin_centers = np.array([ h_min + h_step/2.0 + h_step*i for i in range(int(round((h_max - h_min)/h_step)) + 1) ])

    # Only keep from the first non-zero to last non-zero indices
    if shave:
        min_ind = next( count_i for count_i,i in enumerate(h) if i != 0 )    
        max_ind = (len(h) - 1) - next( count_i for count_i,i in enumerate(h[::-1]) if i != 0 )         
    else:
        min_ind = 0
        max_ind = len(h)-1

    # Normalize the probability
    if normalize:
        h = h / float(sum(h))

    # Apply mirror convention
    if abs_opt is True:
        for count_i,i in enumerate(bin_centers):

            # Set negative hist bin equal to the corresponding positive bin value
            if i < 0:
                pos_ind = where(bin_centers == abs(i))[0]
                if len(pos_ind) > 0:
                    h[count_i] = h[pos_ind]
                    
                # If the positive bin doesn't exist, then drop this index from the return call
                else:
                    min_ind = count_i+1

    return bin_centers[min_ind:max_ind+1],h[min_ind:max_ind+1]


# Description: Function for calculating a histogram from an array or list of values. List of tuples (x,y)
def calc_2dhist(values, x_min=None, x_max=None, x_step=1.0, y_min=None, y_max=None, y_step=1.0, shave=False, normalize=False, abs_opt=False):
    if abs_opt is True:
        values = [abs(_) for _ in values]

    # Check input arguments
    if x_step is None: x_step = float(max(values) - min(values)) / 100.0
    if x_min is None: x_min = min(values) - 0.5 * x_step
    if x_max is None: x_max = max(values) + 0.5 * x_step
    if y_step is None: y_step = float(max(values) - min(values)) / 100.0
    if y_min is None: y_min = min(values) - 0.5 * y_step
    if y_max is None: y_max = max(values) + 0.5 * y_step

    # Generate histogram
    h = np.zeros((int(round((x_max - x_min) / x_step)) + 1, int(round((y_max - y_min) / y_step)) + 1))

    for i in values:

        if i[0] >= x_min and i[0] <= x_max and i[1] >= y_min and i[1] <= y_max:
            h[int(float(i[0] - x_min) / x_step)][int(float(i[1] - y_min) / y_step)] += 1

    x_centers = np.array([x_min + x_step / 2.0 + x_step * i for i in range(int(round((x_max - x_min) / x_step)) + 1)])
    y_centers = np.array([y_min + y_step / 2.0 + y_step * i for i in range(int(round((y_max - y_min) / y_step)) + 1)])
    bin_centers = (x_centers, y_centers)

    # # Only keep from the first non-zero to last non-zero indices
    # if shave:
    #     min_ind = next(count_i for count_i, i in enumerate(h) if i != 0)
    #     max_ind = (len(h) - 1) - next(count_i for count_i, i in enumerate(h[::-1]) if i != 0)
    # else:
    #     min_ind = 0
    #     max_ind = len(h) - 1
    #
    # # Normalize the probability
    # if normalize:
    #     h = h / float(sum(h))
    #
    # # Apply mirror convention
    # if abs_opt is True:
    #     for count_i, i in enumerate(bin_centers):
    #
    #         # Set negative hist bin equal to the corresponding positive bin value
    #         if i < 0:
    #             pos_ind = where(bin_centers == abs(i))[0]
    #             if len(pos_ind) > 0:
    #                 h[count_i] = h[pos_ind]
    #
    #             # If the positive bin doesn't exist, then drop this index from the return call
    #             else:
    #                 min_ind = count_i + 1

    return bin_centers, h
        
# Calculate the angle
def calc_angle(atom_1,atom_2,atom_3):
    return arccos(np.dot(atom_1-atom_2,atom_3-atom_2)/(norm(atom_1-atom_2)*norm(atom_3-atom_2)))

# Calculate the dihedral
def calc_dihedral(atom_1,atom_2,atom_3,atom_4):

    # Calculate all dihedral angles
    v1 = atom_2-atom_1
    v2 = atom_3-atom_2
    v3 = atom_4-atom_3
    return arctan2( np.dot(v1,cross(v2,v3))*(np.dot(v2,v2))**(0.5) , np.dot(cross(v1,v2),cross(v2,v3)) )

# Calcaulte the improper angle
# calculates the angle between the two planes 1,2,3 and 2,3,4 (i.e., atom_1 is the "out-of-plane" atom)
def calc_improper(atom_1,atom_2,atom_3,atom_4):

    # calculate plane 1-2-3 normal
    v1 = atom_1 - atom_2
    v2 = atom_1 - atom_3
    norm_1 = cross(v1,v2)

    # calculate plane 2-3-4 normal
    v3 = atom_2 - atom_3
    v4 = atom_2 - atom_4
    norm_2 = cross(v3,v4)
    
    return arccos(np.dot(norm_1,norm_2)/(norm(norm_1)*norm(norm_2)))


# Write a geometry
def write_xyz(name,geo,elements,open_condition="w"):
    with open(name,open_condition) as f:
        f.write("{}\n\n".format(len(geo)))
        for count_i,i in enumerate(geo):
            f.write("{:<40s} {:<20.6f} {:<20.6f} {:<20.6f}\n".format(elements[count_i],i[0],i[1],i[2]))

# Generator function that yields the geometry, atomids, and atomtypes of each frame
# with a user specified frequency
def frame_generator(name,start,end,every,unwrap=True,adj_list=None, vel=False, log_frames=None, u_dist=False, force=False):

    if unwrap is True and adj_list is None:
        print("ERROR in frame_generator: unwrap option is True but no adjacency_list is supplied. Exiting...")
        quit()

    # Parse data for the monitored molecules from the trajectories
    # NOTE: the structure of the molecule based parse is almost identical to the type based parse
    #       save that the molecule centroids and charges are used for the parse
    # Initialize subdictionary and "boxes" sub-sub dictionary (holds the box dimensions for each parsed frame)

    # Parse Trajectories
    frame       = -1                                                  # Frame counter (total number of frames in the trajectory)
    frame_count = -1                                                  # Frame counter (number of parsed frames in the trajectory)
    frame_flag  =  0                                                  # Flag for marking the start of a parsed frame
    atom_flag   =  0                                                  # Flag for marking the start of a parsed Atom data block
    N_atom_flag =  0                                                  # Flag for marking the place the number of atoms should be updated
    atom_count  =  0                                                  # Atom counter for each frame
    box_flag    =  0                                                  # Flag for marking the start of the box parse
    box_count   = -1                                                  # Line counter for keeping track of the box dimensions.
    pe_ind      = None

    # Open the trajectory file for reading
    with open(name,'r') as f:

        # Iterate over the lines of the original trajectory file
        for lines in f:

            fields = lines.split()

            # Find the start of each frame and check if it is included in the user-requested range
            if len(fields) == 2 and fields[1] == "TIMESTEP":
                frame += 1
                #print("{} seen".format(frame))
                if log_frames is not None:
                    if frame in log_frames:
                        frame_flag = 1
                        frame_count += 1
                        #print("{} in log_frames".format(frame))
                elif frame >= start and frame <= end and (frame-start) % every == 0:
                    frame_flag = 1
                    frame_count += 1

                if frame > end:
                    blah=1
                    break
            # Parse commands for when a user-requested frame is being parsed
            if frame_flag == 1:

                # Header parse commands
                if atom_flag == 0 and N_atom_flag == 0 and box_flag == 0:
                    if len(fields) > 2 and fields[1] == "ATOMS":
                        atom_flag = 1
                        id_ind   = fields.index('id')   - 2
                        type_ind = fields.index('type') - 2
                        if 'xu' in fields:
                            field_add='u' # Add a u to the index call if we find unwrapped coordinates.
                        else:
                            field_add=''
                        x_ind    = fields.index('x'+field_add) - 2
                        y_ind    = fields.index('y'+field_add) - 2
                        z_ind    = fields.index('z'+field_add) - 2
                        if vel is True:
                            vx_ind   = fields.index('vx') - 2
                            vy_ind   = fields.index('vy') - 2
                            vz_ind   = fields.index('vz') - 2
                        if u_dist is True:
                            vx_ind = fields.index('vx') - 2
                            vy_ind = fields.index('vy') - 2
                            vz_ind = fields.index('vz') - 2
                            pe_ind = fields.index('c_pot') - 2
                            ke_ind = fields.index('c_kin') - 2
                        if force is True:
                            if vel is True:
                                vx_ind = fields.index('vx') - 2
                                vy_ind = fields.index('vy') - 2
                                vz_ind = fields.index('vz') - 2
                            if 'c_pot' in fields:
                                pe_ind = fields.index('c_pot') - 2
                                ke_ind = fields.index('c_kin') - 2
                            fx_ind = fields.index('fx') - 2
                            fy_ind = fields.index('fy') - 2
                            fz_ind = fields.index('fz') - 2
                        continue
                    if len(fields) > 2 and fields[1] == "NUMBER":                        
                        N_atom_flag = 1
                        continue

                    if len(fields) > 2 and fields[1] == "BOX":
                        box      = np.zeros([3,2])
                        box_flag = 1
                        continue

                # Update the number of atoms in each frame
                if N_atom_flag == 1:

                    # Intialize total geometry of the molecules being parsed in this frame
                    # Note: from here forward the N_current acts as a counter of the number of atoms that have been parsed from the trajectory.
                    N_atoms     = int(fields[0])
                    geo         = np.zeros([N_atoms,3])
                    vels        = np.zeros([N_atoms,3])
                    engs        = np.zeros([N_atoms,2])
                    forces      = np.zeros([N_atoms,3])
                    ids         = [ -1 for _ in range(N_atoms) ]
                    types       = [ -1 for _ in range(N_atoms) ]
                    N_current   = 0                    
                    N_atom_flag = 0
                    continue

                # Read in box dimensions
                if box_flag == 1:
                    box_count += 1
                    box[box_count] = [float(fields[0]),float(fields[1])]

                    # After all box data has been parsed, save the box_lengths/2 to temporary variables for unwrapping coordinates and reset flags/counters
                    if box_count == 2:
                        box_count = -1
                        box_flag = 0
                    continue

                # Parse relevant atoms
                if atom_flag == 1:
                    geo[atom_count]   = np.array([ float(fields[x_ind]),float(fields[y_ind]),float(fields[z_ind]) ])
                    if force is True:
                        if vel is True:
                            vels[atom_count]  = np.array([ float(fields[vx_ind]),float(fields[vy_ind]),float(fields[vz_ind]) ])
                        if pe_ind is not None:
                            engs[atom_count] = np.array([float(fields[pe_ind]), float(fields[ke_ind])])
                        forces[atom_count] = np.array([ float(fields[fx_ind]),float(fields[fy_ind]),float(fields[fz_ind])  ])
                    elif u_dist is True:
                        vels[atom_count] = np.array([float(fields[vx_ind]), float(fields[vy_ind]), float(fields[vz_ind])])
                        engs[atom_count] = np.array([ float(fields[pe_ind]), float(fields[ke_ind]) ])
                    elif vel is True:
                        vels[atom_count] = np.array([float(fields[vx_ind]), float(fields[vy_ind]), float(fields[vz_ind])])
                    ids[atom_count]   = int(fields[id_ind])
                    types[atom_count] = int(fields[type_ind])                    
                    atom_count += 1

                    # Reset flags once all atoms have been parsed
                    if atom_count == N_atoms:

                        frame_flag = 0
                        atom_flag  = 0
                        atom_count = 0       

                        # Sort based on ids
                        ids,sort_ind =  zip(*sorted([ (k,count_k) for count_k,k in enumerate(ids) ]))
                        geo = geo[list(sort_ind)]
                        vels = vels[list(sort_ind)]
                        forces = forces[list(sort_ind)]
                        types = [ types[_] for _ in sort_ind ]

                        # Velocity and energy has no need for box unwrapping
                        #yield geo, ids, types, box, vels, engs, forces

                        # Upwrap the geometry
                        if unwrap is True:
                            geo = unwrap_geo(geo,adj_list,box)

                        yield geo,ids,types,box,vels,engs,forces

# Description: Performed the periodic boundary unwrap of the geometry
def unwrap_geo(geo,adj_list,box):

    bx_2 = ( box[0,1] - box[0,0] ) / 2.0
    by_2 = ( box[1,1] - box[1,0] ) / 2.0
    bz_2 = ( box[2,1] - box[2,0] ) / 2.0

    # Unwrap the molecules using the adjacency matrix
    # Loops over the individual atoms and if they haven't been unwrapped yet, performs a walk
    # of the molecular graphs unwrapping based on the bonds. 
    unwrapped = []
    for count_i,i in enumerate(geo):

        # Skip if this atom has already been unwrapped
        if count_i in unwrapped:
            continue

        # Proceed with a walk of the molecular graph
        # The molecular graph is cumulatively built up in the "unwrap" list and is initially seeded with the current atom
        else:
            unwrap     = [count_i]    # list of indices to unwrap (next loop)
            unwrapped += [count_i]    # list of indices that have already been unwrapped (first index is left in place)
            for j in unwrap:

                # new holds the index in geo of bonded atoms to j that need to be unwrapped
                new = [ k for k in adj_list[j] if k not in unwrapped ] 

                # unwrap the new atoms
                for k in new:
                    unwrapped += [k]
                    if (geo[k][0] - geo[j][0])   >  bx_2: geo[k,0] -= (bx_2*2.0) 
                    elif (geo[k][0] - geo[j][0]) < -bx_2: geo[k,0] += (bx_2*2.0) 
                    if (geo[k][1] - geo[j][1])   >  by_2: geo[k,1] -= (by_2*2.0) 
                    elif (geo[k][1] - geo[j][1]) < -by_2: geo[k,1] += (by_2*2.0) 
                    if (geo[k][2] - geo[j][2])   >  bz_2: geo[k,2] -= (bz_2*2.0) 
                    elif (geo[k][2] - geo[j][2]) < -bz_2: geo[k,2] += (bz_2*2.0) 

                # append the just unwrapped atoms to the molecular graph so that their connections can be looped over and unwrapped. 
                unwrap += new

    return geo

# Creates a dictionary with the information for the various objects whose autocorrelation behavior is parsed.
def process_request(parse_list,objects,omit):

    # parse dictionary
    parse_dict = { 'vectors': {},
                   'bonds': {},
                   'angles': {},
                   'dihedrals': {},
                   'impropers': {},
                   'velocity': {},
                   'location': {},
                   'vel_loc': {},
                   'energy': {},
                   'force': {}}

    # Loop over modes
    for i in parse_list:

        # Create dictionary elements for the internal degrees of freedom
        # Since these all have the same form they are processed using this key_1 key_2 device 
        # with a flag for avoiding the vectors-based autocorrelations.
        vec_flag = False
        if i == "all":
            key_1 = "vectors"
            key_2 = "all"
            parse_dict["vectors"]["all"] = [] # 1-off special case
        elif "a" == i[0] and "l" == i[3]:
            key_1 = "angles"
            key_2 = int(i[1:])  
        elif "b" == i[0]:
            key_1 = "bonds"
            key_2 = int(i[1:])
        elif "d" == i[0]:
            key_1 = "dihedrals"
            key_2 = int(i[1:])
        elif "i" == i[0]:
            key_1 = "impropers"
            key_2 = int(i[1:])
        elif "v" == i[0] or i=="ang_mom":
            key_1 = "velocity"
        elif "e" == i[0]:
            key_1 = "energy"
        elif "f" == i[0]:
            key_1 = "force"
        else:
            vec_flag = True

        # Create the actual dictionary
        if vec_flag is False and key_1 != "velocity" and key_1 != "vectors" and key_1 != "energy" and key_1 != "force":

            specific_mode = sorted(objects["mode_dict"][key_1].keys())[key_2]
            parse_dict[key_1][specific_mode] = {}
            for j in objects["mode_dict"][key_1][specific_mode]:
                parse_dict[key_1][specific_mode][j] = []

        # If the velocity flag is triggered, include it in the parse_dict - perhaps expand to account for velocities of specific molecules/atoms
        elif vec_flag is False and key_1 == "velocity":
            parse_dict["velocity"] = {}
            N_atoms = len(objects["molinds"])
            for atomid in range(1, N_atoms+1):
                parse_dict["velocity"][atomid] = []
                parse_dict["vel_loc"][atomid] = []

        elif vec_flag is False and key_1 =="energy":
            parse_dict["energy"] = {}
            parse_dict["eng_loc"] =  {}
            N_atoms = len(objects["molinds"])
            for atomid in range(1,N_atoms+1):
                parse_dict["energy"][atomid] = []
                parse_dict["eng_loc"][atomid] = []

        elif vec_flag is False and key_1 == "force":
            parse_dict["force"] = {}
            parse_dict["force_loc"] = {}
            N_atoms = len(objects["molinds"])
            for atomid in range(1, N_atoms+1):
                parse_dict["force"][atomid] = []
                parse_dict["force_loc"][atomid] = []

        # Create dictionary elements for the orientational vectors
        # These need to be parsed differently from the internal degrees of freedom and so are handled separately        
        else:

            # Check for proper formatting
            fields = i.split('-')
            if len(fields) < 3:
                print("ERROR in process_request: the -parse argument '{}' could not be processed.".format(i))
                quit()  
            elif fields[1] == fields[2]:
                print("ERROR in process_request: the two atom indices are the same in the user-supplied")
                print("                          vector parse argument '{}'.".format(i))
                quit()  
            elif int(fields[0]) not in objects["molecules"].keys():
                print("ERROR in process_request: molecule {} is not in the trajectory. The relevant user-supplied".format(i))
                print("                          argument is '{}'.".format(i))
                quit()
            elif int(fields[1]) >= len(objects["molecules"][int(fields[0])]["atomtypes"]) or int(fields[2]) >= len(objects["molecules"][int(fields[0])]["atomtypes"]):
                print("ERROR in process_request: atom indices ({}) exceed the the size of molecule {} ({}).".format(i,int(fields[0]),len(objects["molecules"][int(fields[0])]["atomtypes"])))
                quit()
            elif len(fields) % 2 != 1:
                print("ERROR in process_request: Need an odd number of arguments for vector modes. Ex: molecule-atom1a-atom1b-atom2a-atom2b, etc.")
                quit()

            # Create the dictionary using the (moleculeID,atom_1_ind,atom_2_ind) as a key
            mol = int(fields[0])
            atoms = []
            for j in range(len(fields[1:])//2):
                atoms.append([int(fields[2*j+1]), int(fields[2*j+2])])
            #mol,atom_1,atom_2 = (int(fields[0]),int(fields[1]),int(fields[2]))
            specific_mode = tuple([int(i) for i in fields])
            parse_dict["vectors"][specific_mode] = {}
            parse_dict["location"][specific_mode] = {}


            # Loop over the molids that are of "mol" type
            for j in objects["molecules"][mol]["molids"]:
                sub_inds = [ count_k for count_k,k in enumerate(objects["molids"]) if k == j ]
                vec_inds = []
                for vec_ind in range(len(fields[1:])//2):
                    vecs = (sub_inds[next( (count_k for count_k,k in enumerate(sub_inds) if objects["molinds"][k] ==  int(fields[2*vec_ind+1])), 0)],
                                sub_inds[next( (count_k for count_k,k in enumerate(sub_inds) if objects["molinds"][k] == int(fields[2*vec_ind+2])), 0)])
                    vec_inds.append(vecs)

                parse_dict["vectors"][specific_mode][tuple(vec_inds)] = []
                parse_dict["location"][specific_mode][tuple(vec_inds)] = []

    return parse_dict

def summarize(o,traj_file):

    
    # Iterate over the unique molecules found in the trajectories
    # NOTE: Since the program has already ensured that all trajectories have the same molecules, only the
    #       data from the first file is used for printing the diagnostic
    for i in range(len(o["molecules"])):
        print("\nMolecule: {}".format(i))

        # Get the geometry from the first frame
        geo,ids,types = next(frame_generator(traj_file,0,0,1,unwrap=True,adj_list=o["adj_list"]))           # grab the geometry from the first frame
        inds = [ count_j for count_j,j in enumerate(o["molids"]) if j == o["molecules"][i]["molids"][0] ]   # get the indices of the first representative molecule
        geo = geo[inds]                                 # Grab the subset of atoms that are in the geometry
        geo = geo - mean(geo,axis=0)                    # Center the geometry
        elements = [ o["elements"][j] for j in inds ]   # Grab the elements
        write_xyz("molecule_{}.xyz".format(i),geo,elements) # Write the sample molecule to file

        # Print atomtypes and charges for the current molecule
        print("\n\t{:<10s} {:<10s} {:<40s}  {:<20s} {:<20s}".format("Index","Elements","Atom_types","Charges","Sample_Geometry"))
        for count_j,j in enumerate(o["molecules"][i]["atomtypes"]):
            print("\t{:<10d} {:<10s} {:<40s} {:< 20.6f} {:< 20.6f} {:< 20.6f} {:< 20.6f}".format(count_j,elements[count_j],j,o["molecules"][i]["charges"][count_j],geo[count_j][0],geo[count_j][1],geo[count_j][2]))

        # Print the molids that correspond to this molecule 
        corr_list = o["molecules"][i]["molids"]
        for j in range(int(ceil(len(corr_list)/5.0))):
            if j == 0:
                print("\n\tCorrespondence with trajectory mol_id: {}".format(', '.join([ str(k) for k in corr_list[j*5:j*5+5]])))
            else:
                print("\t                                       {}".format(', '.join([ str(k) for k in corr_list[j*5:j*5+5]])))

        # Print the total charge for this molecule type
        print("\tTotal charge: {:< f}".format(sum(o["molecules"][i]["charges"])))

    # Print the mode summary
    print("\nMode Summary:")
    print("\n\tBonds (instances):\n")
    for count_i,i in enumerate(sorted(o["mode_dict"]["bonds"].keys())):
        print("\t\t{:<4s}: {:<40s} {:<40s} ({})".format("b{}".format(count_i),i[0],i[1],len(o["mode_dict"]["bonds"][i])))

    print("\n\tAngles (instances):\n")
    for count_i,i in enumerate(sorted(o["mode_dict"]["angles"].keys())):
        print("\t\t{:<4s}: {:<40s} {:<40s} {:<40s} ({})".format("a{}".format(count_i),i[0],i[1],i[2],len(o["mode_dict"]["angles"][i])))

    print("\n\tDihedrals (instances):\n")
    for count_i,i in enumerate(sorted(o["mode_dict"]["dihedrals"].keys())):
        print("\t\t{:<4s}: {:<40s} {:<40s} {:<40s} {:<40s} ({})".format("d{}".format(count_i),i[0],i[1],i[2],i[3],len(o["mode_dict"]["dihedrals"][i])))

    print("\n\tImpropers (instances):\n")
    for count_i,i in enumerate(sorted(o["mode_dict"]["impropers"].keys())):
        print("\t\t{:<4s}: {:<40s} {:<40s} {:<40s} {:<40s} ({})".format("i{}".format(count_i),i[0],i[1],i[2],i[3],len(o["mode_dict"]["impropers"][i])))
    return

# Loop for parsing the mapfile information
def parse_map(map_file):

    atomtypes = []
    elements = []
    masses = []
    charges = []
    adj_list = []
    with open(map_file,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc==1:
                # Find field headers
                atom_ind=fields.index("Atom_type")
                element_ind=fields.index("Element")
                mass_ind=fields.index("Mass")
                charge_ind=fields.index("Charge")
                adj_ind=fields.index("Adj_mat")
            elif lc > 1 and len(fields) > 4:
                atomtypes += [fields[atom_ind]]
                elements  += [fields[element_ind]]
                masses    += [float(fields[mass_ind])]
                charges   += [float(fields[charge_ind])]
                adj_list  += [ [int(_) for _ in fields[adj_ind:] ] ]

    return atomtypes,elements,masses,charges,adj_list

# algorithm for finding the connected components in a graph defined by the adjacency matrix/list A
def find_subgraphs(A,algorithm="matrix"):

    # Check for the algorithm
    if algorithm not in ["matrix","list"]:
        print("ERROR in find_subgraphs: only 'matrix' and 'list' are valid algorithms")
        quit()

    # Find subgraphs
    subgraph_list = []
    placed_list   = []

    # Might be more efficient just to convert the adj_mat into an adj_list and use the list algorithm
    if algorithm == "matrix":
        for count_i,i in enumerate(A):

            # skip if the current node has already been placed in a subgraph
            if count_i in placed_list: continue

            # current holds the subgraph seedlings
            subgraph = [count_i] + [ count_j for count_j,j in enumerate(i) if j == 1 ]

            # new holds the updated list of nodes in the subgraph at the end of each bond search
            new = [ count_j for count_j,j in enumerate(i) if j == 1 ]

            # recursively add nodes to new until no new nodes are found
            while new != []:

                # Initialize list to hold new connnected nodes
                connections = []

                # Iterate over the new nodes and find new connections
                for count_j,j in enumerate(new):
                    connections += [ count_k for count_k,k in enumerate(A[j]) if k == 1 and count_k not in subgraph ] # add new connections not already in the subgraph

                # Update lists
                connections = list(set(connections))
                subgraph += connections
                new = connections

            # Sort nodes in the subgraph (makes them unique)
            subgraph = sorted(subgraph)

            # Add new subgraphs to the master list and add its nodes to the placed_list to avoid redundant searches
            if subgraph not in subgraph_list:
                subgraph_list += [subgraph]
                placed_list += subgraph

    # adjacency list based algorithm
    if algorithm == "list":
        for count_i,i in enumerate(A):

            # skip if the current node has already been placed in a subgraph
            if count_i in placed_list: continue

            # current holds the subgraph seedlings
            subgraph = [count_i] + i

            # new holds the updated list of nodes in the subgraph at the end of each bond search
            new = i

            # recursively add nodes to new until no new nodes are found
            while new != []:

                # Initialize list to hold new connnected nodes
                connections = []

                # Iterate over the new nodes and find new connections
                for count_j,j in enumerate(new):
                    connections += [ k for k in A[j] if k not in subgraph ] # add new connections not already in the subgraph

                # Update lists
                connections = list(set(connections))
                subgraph += connections
                new = connections

            # Sort nodes in the subgraph (makes them unique)
            subgraph = sorted(subgraph)

            # Add new subgraphs to the master list and add its nodes to the placed_list to avoid redundant searches
            if subgraph not in subgraph_list:
                subgraph_list += [subgraph]
                placed_list += subgraph
    return subgraph_list
    
# Function for finding the unique molecules in a map file
def find_molecules(atomtypes,elements,masses,charges,adj_list,adj_mat=None):

    subgraphs = find_subgraphs(adj_list,algorithm="list")    
    molids = [ next(count_j for count_j,j in enumerate(subgraphs) if i in j ) for i in range(len(atomtypes)) ]
    molinds = [ -1 for i in range(len(molids)) ]
    if adj_mat is None:
        adj_mat = np.zeros([len(adj_list),len(adj_list) ])
        for count_i,i in enumerate(adj_list): 
            adj_mat[i,count_i] = 1
            adj_mat[count_i,i] = 1

    # Iterate over the subgraphs to id unique molecules based on topology and atom type(s)
    mol_id2type = {}            
    adj_mat_list  = []
    atomtype_list = []
    atomid_list   = []
    charges_list  = []    
    for count_i,i in enumerate(subgraphs):
        mol_adj_mat = np.zeros([len(i),len(i)])
        for count_j,j in enumerate(i):
            mol_adj_mat[count_j,:] = adj_mat[j,i] # NOTE: i is a list whereas j is an int
            mol_adj_mat[:,count_j] = adj_mat[i,j] # NOTE: i is a list whereas j is an int
        mol_atomtypes = [ atomtypes[j] for j in i ]
        mol_charges   = [ charges[j] for j in i ]        

        # Check if this is a new molecule based on comparisons against adj_mats, atom types, and charges. 
        match = 0
        for count_j,j in enumerate(adj_mat_list):
            if np.array_equal(mol_adj_mat,j) and atomtype_list[count_j] == mol_atomtypes and charges_list[count_j] == mol_charges:
                mol_id2type[count_i] = count_j 
                match = 1
                break

        # If no match was found then add the new molecule info to the appropriate lists
        if match == 0:
            adj_mat_list  += [mol_adj_mat]
            atomtype_list += [mol_atomtypes]
            charges_list  += [mol_charges]            
            mol_id2type[count_i] = len(adj_mat_list)-1 

        # Assign the molinds values (i.e., the indices within the molecule of each atom, 
        # this internal reference indexing is useful for some autocorrelations
        for count_j,j in enumerate(i): molinds[j] = count_j

    # Reassign moltypes based on size (smallest first) then charge 
    # NOTE: scaling the size by 1000 opens up space to rank by charge as a secondary measure
    # NOTE: -sum(charges) leads to cations being ranked first.
    scores = [ len(i)*1000.0-sum(charges_list[count_i]) for count_i,i in enumerate(atomtype_list) ]
    ind_map = sorted(range(len(scores)), key=lambda k: scores[k])
    for i in mol_id2type: mol_id2type[i] = ind_map.index(mol_id2type[i])
    if len(set(scores)) > 1:
        scores,adj_mat_list,atomtype_list,charges_list = zip(*sorted(zip(scores,adj_mat_list,atomtype_list,charges_list)))

    # Create the molecules dictionary that holds the general information (key: adj_mat, charges, adn atomtypes) for each unique molecule and the molids that correspond to that type (key: molids)
    molecules = {}
    for i in range(len(adj_mat_list)):        
        molecules[i] = { "adj_mat": adj_mat_list[i], "charges": charges_list[i], "atomtypes": atomtype_list[i], "molids": sorted([ j for j in mol_id2type.keys() if mol_id2type[j] == i ]) }

    # Create a list of mol types indexed to each atom 
    moltypes = [ mol_id2type[i] for i in molids ]

    return molids,moltypes,molinds,molecules

# Parses the information for unique molecules in the trajectories. Automatic typing of molecules is based on their 
# adjacency matrix as determined from the bonds in the data file.
def parse_objects(map_file):
    
    atomtypes,elements,masses,charges,adj_list = parse_map(map_file)    

    # Calculate the adjacency matrix
    adj_mat = np.zeros([len(adj_list),len(adj_list) ])
    for count_i,i in enumerate(adj_list): 
        adj_mat[i,count_i] = 1
        adj_mat[count_i,i] = 1

    # Parse the molecules
    molids,moltypes,molinds,molecules = find_molecules(atomtypes,elements,masses,charges,adj_list,adj_mat=adj_mat)

    # Gather the modes
    mode_dict = find_modes_list(adj_list,atomtypes)
        
    return { "atomtypes": atomtypes,
             "elements":  elements,
             "masses":    masses,
             "charges":   charges,
             "molids":    molids,
             "moltypes":  moltypes,
             "molinds":   molinds,
             "molecules": molecules,
             "mode_dict": mode_dict,
             "adj_list":  adj_list }

# returns a dictionary with the modes
def find_modes(Adj_mat,Atom_types):

    # List comprehension to determine bonds from a loop over the adjacency matrix. Iterates over rows (i) and individual elements
    # ( elements A[count_i,count_j] = j ) and stores the bond if the element is "1". The count_i < count_j condition avoids
    # redudant bonds (e.g., (i,j) vs (j,i) ). By convention only the i < j definition is stored.
    print("Parsing bonds...")
    Bonds          = [ (count_i,count_j) for count_i,i in enumerate(Adj_mat) for count_j,j in enumerate(i) if j == 1 ]
    Bond_types     = [ (Atom_types[i[0]],Atom_types[i[1]]) for i in Bonds ]

    # List comprehension to determine angles from a loop over the bonds. Note, since there are two bonds in every angle, there will be
    # redundant angles stored (e.g., (i,j,k) vs (k,j,i) ). By convention only the i < k definition is stored.
    print("Parsing angles...")
    Angles          = [ (count_j,i[0],i[1]) for i in Bonds for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Angle_types     = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]]) for i in Angles ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("Parsing dihedrals...")
    Dihedrals      = [ (count_j,i[0],i[1],i[2]) for i in Angles for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Dihedral_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Dihedrals ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("Parsing impropers...")
    Impropers      = [ (i[1],i[0],i[2],count_j) for i in Angles for count_j,j in enumerate(Adj_mat[i[1]]) if j == 1 and count_j not in i ]
    Improper_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Impropers ]

    # Canonicalize the modes
    for i in range(len(Bonds)):
        Bond_types[i],Bonds[i] = canon_bond(Bond_types[i],ind=Bonds[i])
    for i in range(len(Angles)):
        Angle_types[i],Angles[i] = canon_angle(Angle_types[i],ind=Angles[i])
    for i in range(len(Dihedrals)):
        Dihedral_types[i],Dihedrals[i] = canon_dihedral(Dihedral_types[i],ind=Dihedrals[i])
    for i in range(len(Impropers)):
        Improper_types[i],Impropers[i] = canon_improper(Improper_types[i],ind=Impropers[i])        

    # Remove redundancies    
    if len(Bonds) > 0: Bonds,Bond_types = map(list, zip(*[ (i,Bond_types[count_i]) for count_i,i in enumerate(Bonds) if count_i == [ count_j for count_j,j in enumerate(Bonds) if j == i or j[::-1] == i ][0]  ]))
    if len(Angles) > 0: Angles,Angle_types = map(list, zip(*[ (i,Angle_types[count_i]) for count_i,i in enumerate(Angles) if count_i == [ count_j for count_j,j in enumerate(Angles) if j == i or j[::-1] == i ][0]  ]))
    if len(Dihedrals) > 0: Dihedrals,Dihedral_types = map(list, zip(*[ (i,Dihedral_types[count_i]) for count_i,i in enumerate(Dihedrals) if count_i == [ count_j for count_j,j in enumerate(Dihedrals) if j == i or j[::-1] == i ][0]  ]))
    if len(Impropers) > 0: Impropers,Improper_types = map(list, zip(*[ (i,Improper_types[count_i]) for count_i,i in enumerate(Impropers) if count_i == [ count_j for count_j,j in enumerate(Impropers) if j[0] == i[0] and len(set(i[1:]).intersection(set(j[1:]))) ][0] ]))

    return { "bonds"     : { i:[ Bonds[count_j] for count_j,j in enumerate(Bond_types) if i == j ] for i in set(Bond_types) },\
             "angles"    : { i:[ Angles[count_j] for count_j,j in enumerate(Angle_types) if i == j ] for i in set(Angle_types) },\
             "dihedrals" : { i:[ Dihedrals[count_j] for count_j,j in enumerate(Dihedral_types) if i == j ] for i in set(Dihedral_types) },\
             "impropers" : { i:[ Impropers[count_j] for count_j,j in enumerate(Improper_types) if i == j ] for i in set(Improper_types) } }


# returns a dictionary with the modes
def find_modes_list(A,Atom_types,algorithm="list"):

    # List comprehension to determine bonds from a loop over the adjacency matrix. Iterates over rows (i) and individual elements
    # ( elements A[count_i,count_j] = j ) and stores the bond if the element is "1". The count_i < count_j condition avoids
    # redudant bonds (e.g., (i,j) vs (j,i) ). By convention only the i < j definition is stored.
    print("Parsing bonds...")
    Bonds          = [ (count_i,j) for count_i,i in enumerate(A) for j in i ]
    Bond_types     = [ (Atom_types[i[0]],Atom_types[i[1]]) for i in Bonds ]

    # List comprehension to determine angles from a loop over the bonds. Note, since there are two bonds in every angle, there will be
    # redundant angles stored (e.g., (i,j,k) vs (k,j,i) ). By convention only the i < k definition is stored.
    print("Parsing angles...")
    Angles          = [ (j,i[0],i[1]) for i in Bonds for j in A[i[0]] if j not in i ]
    Angle_types     = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]]) for i in Angles ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("Parsing dihedrals...")
    Dihedrals      = [ (j,i[0],i[1],i[2]) for i in Angles for j in A[i[0]] if j not in i ]
    Dihedral_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Dihedrals ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("Parsing impropers...")
    Impropers      = [ (i[1],i[0],i[2],j) for i in Angles for j in A[i[1]] if j not in i ]
    Improper_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Impropers ]

    # Canonicalize the modes
    for i in range(len(Bonds)):
        Bond_types[i],Bonds[i] = canon_bond(Bond_types[i],ind=Bonds[i])
    for i in range(len(Angles)):
        Angle_types[i],Angles[i] = canon_angle(Angle_types[i],ind=Angles[i])
    for i in range(len(Dihedrals)):
        Dihedral_types[i],Dihedrals[i] = canon_dihedral(Dihedral_types[i],ind=Dihedrals[i])
    for i in range(len(Impropers)):
        Improper_types[i],Impropers[i] = canon_improper(Improper_types[i],ind=Impropers[i])        

    # Remove the redundancies
    if len(Bonds) > 0: 
        assigned_list = set([])
        keep_ind = []
        for count_i,i in enumerate(Bonds):
            if i not in assigned_list:
                keep_ind += [count_i]
                assigned_list.update([i,i[::-1]])
        Bonds = [ Bonds[i] for i in keep_ind ]
        Bond_types = [ Bond_types[i] for i in keep_ind ]

    if len(Angles) > 0: 
        assigned_list = set([])
        keep_ind = []
        for count_i,i in enumerate(Angles):
            if i not in assigned_list:
                keep_ind += [count_i]
                assigned_list.update([i,i[::-1]])
        Angles = [ Angles[i] for i in keep_ind ]
        Angle_types = [ Angle_types[i] for i in keep_ind ]

    if len(Dihedrals) > 0: 
        assigned_list = set([])
        keep_ind = []
        for count_i,i in enumerate(Dihedrals):
            if i not in assigned_list:
                keep_ind += [count_i]
                assigned_list.update([i,i[::-1]])
        Dihedrals = [ Dihedrals[i] for i in keep_ind ]
        Dihedral_types = [ Dihedral_types[i] for i in keep_ind ]

    if len(Impropers) > 0: 
        assigned_list = set([])
        keep_ind = []
        for count_i,i in enumerate(Impropers):
            if (i[0],i[1]) not in assigned_list and (i[0],i[2]) not in assigned_list and (i[0],i[3]) not in assigned_list:
                keep_ind += [count_i]
                assigned_list.update([(i[0],i[1]),(i[0],i[2]),(i[0],i[3])])
        Impropers = [ Impropers[i] for i in keep_ind ]
        Improper_types = [ Improper_types[i] for i in keep_ind ]

    return { "bonds"     : { i:[ Bonds[count_j] for count_j,j in enumerate(Bond_types) if i == j ] for i in set(Bond_types) },\
             "angles"    : { i:[ Angles[count_j] for count_j,j in enumerate(Angle_types) if i == j ] for i in set(Angle_types) },\
             "dihedrals" : { i:[ Dihedrals[count_j] for count_j,j in enumerate(Dihedral_types) if i == j ] for i in set(Dihedral_types) },\
             "impropers" : { i:[ Impropers[count_j] for count_j,j in enumerate(Improper_types) if i == j ] for i in set(Improper_types) } }

# Function for keeping tabs on the validity of the user supplied inputs
def check_validity(args):

    # Convert inputs to the proper data type
    args.f_start = int(round(float(args.f_start)))
    args.f_end = int(round(float(args.f_end)))
    args.f_every = int(round(float(args.f_every)))
    args.omit = args.omit.split()
    if args.unwrap_opt == "True" or args.unwrap_opt == "true":
        args.unwrap_opt = True
    elif args.unwrap_opt == "False" or args.unwrap_opt == "false":
        args.unwrap_opt = False

    # Consistency checks
    if args.map_file is False:
        print("ERROR in autocorr_calc: you must supply a taffi map file via the -map flag")
        quit()
    elif os.path.isfile(args.map_file) is False:
        print("ERROR in autocorr_calc: could not find the map file ({}). Exiting...".format(args.map_file))
        quit()
    else:
        args.map_file = os.path.abspath(args.map_file)
    if args.traj_file is False:
        print("ERROR in autocorr_calc: you must supply a trajectory file via the -traj flag")
        quit()
    elif os.path.isfile(args.traj_file) is False:
        print("ERROR in autocorr_calc: could not find the trajectory file ({}). Exiting...".format(args.traj_file))
        quit()
    else:
        args.traj_file = os.path.abspath(args.traj_file)
    if args.unwrap_opt not in (True,False):
        print("ERROR in autocorr_calc: -unwrap only allows True and False as an argument. Exiting...")
        quit()
    if args.list_flag is False and args.parse_list is None:
        print("ERROR in autocorr_calc: if you are not running with the --list option, at least one mode needs to be supplied to the --modes argument.")
        quit()
    if args.parse_list is not None:
        args.parse_list = args.parse_list.split()        
    if args.f_start > args.f_end: 
        print("ERROR in autocorr_calc: the starting frame (f_start: {}) is greater than the ending fram (f_end: {}).".format(args.f_start,args.f_end))
        quit()
    if os.path.isdir(args.o_folder) is False:
        os.mkdir(args.o_folder)
        args.o_folder = os.path.abspath(args.o_folder) 
    
    # Turn omit into a dictionary
    tmp = { "ids":[], "molids":[] }
    for i in args.omit:
        if i[0] == "a":
            i = i.strip('a').strip('t').strip('m')
            if "-" in i:
                i = i.split('-')
                i = range(int(i[0]),int(i[1])+1)
            else:
                i = [int(i)]
            tmp["ids"] += i
        elif i[0] == "m":
            i = i.strip('a').strip('t').strip('m')
            if "-" in i:
                i = i.split('-')
                i = range(int(i[0]),int(i[1])+1)
            else:
                i = [int(i)]
            tmp["molids"] += i
        else:
            print("ERROR in check_validity: term {} in --omit could not be processed. Exiting...".format(i))
            quit()
    args.omit = { i:set(tmp[i]) for i in tmp }
    
    return args

def dist(i,j):
    return np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2 + (i[2]-j[2])**2)

# main sentinel
if __name__ == "__main__":
    main(sys.argv[1:])
