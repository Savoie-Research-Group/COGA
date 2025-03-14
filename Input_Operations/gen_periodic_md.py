#!/bin/env python                                                                                                                                                             
# Author: Brett Savoie (brettsavoie@gmail.com)
# Edits by Dylan Fortney to add symmetry to coarse grain section.

import sys,argparse,os,datetime,fnmatch,os,re,math

# Add TAFFY Lib to path
#sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/Lib')
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
from pathlib import Path
home=str(Path.home())
#sys.path.append('{}/bin/taffi_beta/Lib'.format(home))
#sys.path.append('{}/bin/taffi/CG'.format(home))
from gen_md_inputs_martini import adj_list2mat
#sys.path.append('{}/bin/taffi_beta/Lib/'.format(home))
from Mol_ops import *
# from transify import *
# from adjacency import *
# from file_parsers import *
# from id_types import *
# from kekule import *
from vautocorr import parse_map

# NOTE: the order of the imports matters until I resolve the namespace issues (XXX DON'T USE import *)
import random
from numpy import *
import numpy as np
import time
import ast
from scipy import *
from scipy.spatial.distance import *
from numpy.linalg import *
from shutil import move,copyfile
import scipy.linalg
import operator
import subprocess as sp

def main(argv):

    parser = argparse.ArgumentParser(description='Reads in a geometry and FF.db file (usually outputted from an intramolecular mode parse) '+\
                                                 'and writes inputs for a LAMMPS job to generate configurations for the parsing VDW parameters.')


    #required (positional) arguments                                                                                                  
    parser.add_argument('coord_files', help = 'A quoted list of the *.cif files to be included in the sim, or a folder holding a previous md cycles. If a folder is supplied then the FF parameters and last configuration '+\
                                              'are used to start the new job along with the most recently fit VDW parameters.')


    #optional arguments    
    parser.add_argument('-FF',dest='FF_files', default="",
                         help = 'A quoted list of the force-field files. Formatting of the force-field files is as produced by the FF_gen.py/FF_extract.py programs)')

    parser.add_argument('-dims', dest='dims', default="0 0 0",
                        help = 'Controls the number of times the unit cell is replicated in each dimension, with 1 implying a single unit cell. '+\
                               'This is supplied as a space-delimited string of integers. (default: "0 0 0")')

    parser.add_argument('-T', dest='T_equil', default="none",
                        help = 'Controls the temperature of the MD simulation. (in Kelvin; default: ambient temperature from cif file)')

    parser.add_argument('-t', dest='t_equil', default=1E6,
                        help = 'Controls the length of the MD equilibration segment. (default: 1E6)')

    parser.add_argument('-T_A', dest='T_anneal', default="none",
                        help = 'Controls the temperature of the MD simulation. (in Kelvin; default: ambient temperature from cif file)')
    
    parser.add_argument('-T_R', dest='T_relax', default="none",
                        help = 'Controls the temperature of the constrained relaxation step. (in Kelvin; default: one half ambient temperature from cif file)')

    parser.add_argument('-t_A', dest='t_anneal', default=1E6,
                        help = 'Controls the length of the MD equilibration segment. (default: 1E6)')

    parser.add_argument('-t_ext', dest='t_ext', default=1E6,
                        help = 'Controls the length of the MD extension job script. (default: 1E6)')

    parser.add_argument('-T_D', dest='T_dilatometry', default=None,
                        help = 'Setting this temperature affects the extend.in.init file that is generated. When this variable is set it corresponds to the final temperature that the user would '+\
                               'like to ramp the simulation to over -N_dil md runs. Using this option, the user can easily generate a dilatometry run from an equilibrated md trajectory by sequential submission of the '+\
                               'extend.in.init file. (default: None)')

    parser.add_argument('-P_D', dest='P_D', default=None,
                        help='Setting this pressure affects the extend file which is generated. P_D corresponds to the final pressure which is ramped to from the P tag. (Default=None)')

    parser.add_argument('-N_dil', dest='N_dilatometry', default=20,
                        help = 'This variable is only used when the user wants to run a dilatometry simulation, and the -T_D variable is explicitly set. This variable controls the number of md cycles that the temperature '+\
                               'is ramped over until reaching the final -T_D temperature. In total, the dilatometry will be carried out over a time of t_ext * N_dil. (default: 20 cycles)')

    parser.add_argument('-P', dest='pressure', default="1atm",
                        help = 'Controls the pressure during the MD simulation. (in ATM; default: 1)')

    parser.add_argument('-fix', dest='fixes', default="",
                        help = 'Determines if bonds and/or angles are contstrained in the simulation. "bonds" and "angles" are values options that can be supplied as a space-delimited string. Or use "linear" to apply rigid/small for linear triatomics. By default the rattle '+\
                               'algorithm is used for constraining the modes. (default: no constrained modes)')

    parser.add_argument('-f', dest='frequency', default=1000,
                        help = 'Controls the sampling frequency during the MD simulation. (default: 1000)')

    parser.add_argument('-q', dest='q_list', default="none",
                        help = 'Controls the total charge on the molecule. By default, the rounded integer charge is used for each molecule (round). The program expects a list of integers (or none, or round for the default). '+\
                               'If less integers are supplied than the number of coord files, then the list is automatically expanded using the last supplied (or default) element. (default: None)') 

    parser.add_argument('-o', dest='outputname', default='',
                        help = 'Sets the output filename prefix (default: VDW_selfterms)')

    parser.add_argument('-units', dest='units', default='real',
                        help='LAMMPS units to run the simulation in. Chosen from real, lj, metal, si, cgs, electron, micro, nano. Default = real')

    parser.add_argument('-ext', dest='extname', default='extend',
                        help='Sets the extension filename prefix (default: extend)')

    parser.add_argument('-ts', dest='timestep', default=1.0,
                        help = 'Sets the timestep for the md simulation (default: 1.0, units of fs)')

    parser.add_argument('-gens', dest='gens', default=2,
                        help = 'When atomtypes are being automatically determined (i.e. when the supplied *.xyz files do not have atomtype information) this variable controls the bond depth used to define unique atomtypes. (default: 2)')

    parser.add_argument('-onefourscale_coul', dest='onefourscale_coul', default=0.0,
                        help = 'Sets scaling for 1-4 Electrostatic interactions (default: 0.0)')

    parser.add_argument('-onefourscale_lj', dest='onefourscale_lj', default=0.0,
                        help = 'Sets scaling for 1-4 VDW interactions (default: 0.0)')

    parser.add_argument('-eps_scale', dest='eps_scale', default=1.0,
                        help = 'Sets scaling for the default UFF eps parameters. To increase configurational sampling, it can be advantageous '+\
                               'to reduce the stiffness of the VDW potentials. (default: 1.0; full value)')

    parser.add_argument('-sigma_scale', dest='sigma_scale', default=1.0,
                        help = 'Sets scaling for the default UFF sigma parameters. To increase configurational sampling, it can be advantageous '+\
                               'to reduce the LJ minima of the VDW potentials. (default: 1.0; full value)')

    parser.add_argument('-charge_scale', dest='charge_scale', default=1.0,
                        help = 'Sets scaling for the simulation charges. To increase configurational sampling, it can be advantageous '+\
                               'to reduce the charges (especially for ions). (default: 1.0; full value)')

    parser.add_argument('--UFF', dest='force_UFF', default=0, action='store_const', const=1,
                        help = 'Forces the use of UFF parameters (regardless of the presence of corresponding parameters in the read FF files). (default: off)')

    parser.add_argument('-mixing_rule', dest='mixing_rule', default="none",
                        help = 'Defines the mixing rule to be used for missing LJ parameters. Waldman-Hagler (wh) and Lorentz-Berthelot (lb) and "none", are valid options. When set to "none", the program will print a message and '+\
                               'exit if there are missing parameters. (default: "none")')

    parser.add_argument('-pair_styles', dest='pair_styles', default='lj/cut/coul/long 14.0 14.0',
                        help = 'Supplies the information for the pair style setting(s). Supplied as a string and should be formatted for comply '+\
                               'with the LAMMPS "pair_style hybrid" command (default: "lj/cut/coul/long 14.0 14.0")')

    parser.add_argument('--tail', dest='tail_opt', default=False, action='store_const', const=True,
                        help = 'Use tail correction to compensate for LJ cutoff. (default: off, i.e., the shifted potential is used)')

    parser.add_argument('--impropers', dest='improper_flag', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will use improper dihedral terms. (default: False)')

    parser.add_argument('--velocities', dest='velocity_flag', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will print the velocities to the dump file (default: False)')

    parser.add_argument('--NVT', dest='NVT_opt', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will run in the NVT ensemble. (default: False)')

    parser.add_argument('--molecule', dest='molecule_flag', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will print the molids to the dump file (default: False)')

    parser.add_argument('--cg', dest='cg', default=False, action='store_const', const=True,
                        help='When this flag is triggered, the script will generate necessary files for a coarse-grained simulation with tabulated potentials. Must provide single molecule map file, as generated from cg process.')
    
    parser.add_argument('--cell_sym', dest='cell_sym', default=False, action='store_const', const=True,
                        help='When this flag is triggered, the script will use symmetry calculations for cg simulation cell generation.')

    parser.add_argument('-single_map', dest='single_map', default=None,
                        help="Provide address of single molecule map file, since adjacency can't be parsed for cg molecules.")

    parser.add_argument('-data_file', dest='data_file', default=None, help="Address of data file, for use with cg simulations")

    parser.add_argument('-settings_file', dest='settings_file', default=None, help="Settings file, for use with cg simulations. Must be provided in addition to data file.")

    parser.add_argument('-cg_uc', dest="unitcell", default=None, help="Give unit cell parameters as space-delinated string in order: 'a b c alpha beta gamma'.")

    parser.add_argument('-cg_nuc', dest="n_uc", default=None, help="Number of atoms in a unit cell")

    parser.add_argument('--fix_constrain', dest="constrain", default=False, action='store_const', const=True, help="When this flag is triggered, harmonic constraints about the initial position of the atoms will be implemented.")

    parser.add_argument('-tether', dest="tether", default=None, help="This flag activates tethering to the positions in the data file. Give spring constant for tethering.")

    parser.add_argument('--tabulated', dest="tabulated", default=False, action='store_const', const=True)

    parser.add_argument('--overwrite', dest="ow", default=False, action='store_const', const=True, help="If true, overwrite the files at the current directory")

    #parser.add_argument('--cg_rand_vel', dest='cg_rand_vel', default=False, action='store_const', const=True, help="If triggered, will randomize velocities of the given ")


    # Make parse inputs
    args=parser.parse_args()

    if args.tether is not None:
        args.NVT_opt = True

    # Converting each list input argument. 
    args.coord_files = args.coord_files.split()
    args.FF_files = args.FF_files.split()
    args.q_list = [ str(i) for i in args.q_list.split() ]
    args.dims = [ int(_) for _ in args.dims.split() ]
    if args.single_map is not None:
        args.single_map = args.single_map.split()
    while len(args.dims) < 3: args.dims += [args.dims[-1]]

    # Check if a directory was supplied to the coord_files list.
    bools = [ os.path.isdir(i) for i in args.coord_files ]
    if True in bools: 
        if len([ True for i in bools if i == True ]) > 1:
            print("ERROR: multiple directories were supplied to the script. Only one directory can\nbe supplied if the user intends to start the job from a previous run. Exiting...")
            quit()        
        args.coord_files = [args.coord_files[bools.index(True)]]
        read_flag = 1
    else:
        read_flag = 0

    # Extend N_mol and q_list to match the length of coord_file
    while len(args.q_list) < len(args.coord_files): args.q_list = args.q_list + [args.q_list[-1]] 

    args.T_equil      = str(args.T_equil) if args.T_equil == "none" else float(args.T_equil)
    args.T_anneal     = str(args.T_anneal) if args.T_anneal == "none" else float(args.T_anneal)
    args.T_relax      = str(args.T_relax) if args.T_relax == "none" else float(args.T_relax)
    args.t_equil      = int(float(args.t_equil))
    args.t_anneal     = int(float(args.t_anneal))
    args.t_ext        = int(float(args.t_ext))
    if "gpa" in args.pressure.lower():
        args.pressure = float(args.pressure.lower().strip("gpa"))*9869.23169314269
    elif "atm" in args.pressure.lower():
        args.pressure = float(args.pressure.lower().strip("atm"))
    else:
        args.pressure     = float(args.pressure)
    if args.P_D is not None:
        if "gpa" in args.P_D.lower():
            args.P_D = float(args.P_D.lower().strip("gpa"))*9869.23169314269
        elif "atm" in args.P_D.lower():
            args.P_D = float(args.P_D.lower().strip("atm"))
        else:
            args.P_D     = float(args.P_D)
    args.frequency    = int(float(args.frequency))
    args.eps_scale    = float(args.eps_scale)
    args.sigma_scale  = float(args.sigma_scale)
    args.charge_scale = float(args.charge_scale)
    args.gens         = int(args.gens)
    args.timestep     = float(args.timestep)
    args.mixing_rule = str(args.mixing_rule).lower()
    if args.T_dilatometry is not None:
        args.T_dilatometry = float(args.T_dilatometry)
        if args.T_dilatometry < 0: print("ERROR: -T_D must be set to a positive value. Exiting..."); quit()
    if args.mixing_rule not in ["wh","lb","none"]:
        print("ERROR: '{}' is not a valid argument for -mixing_rule. Valid options are 'wh' 'lb' and 'none'. Exiting...".format(args.mixing_rule))
        quit()
    if args.timestep < 0: print("ERROR: -ts must be set to a positive value. Exiting..."); quit()

    # Parse output filename 
    if args.outputname != '':

        # Handle the special case where the script is run from without the output directory
        if args.outputname == ".":
            print("ERROR: the output folder must differ from the current folder.")
            quit()

        # Else, directly assign the output directory based on the -o argument
        else:
            Filename = args.outputname        

    # If the output folder is not specified then the program defaults to the root of the first supplied *xyz file
    else:
        Filename = args.coord_files[0].split('.')[0]
    
    # Check that the input is an .cif file. 
    if read_flag == 0:
        for i in args.coord_files:
            if i.split('.')[-1] != 'cif':
                print("ERROR: Check to ensure that the input coordinate file(s) are in .cif format.")
                quit()
            elif os.path.isfile(i) != True:
                print("ERROR: Specified *.cif file ({}) does not exit. Please check the path. Exiting...".format(i))
                quit()

    # Check that the supplied FF files exist
    for i in args.FF_files:
        if os.path.isfile(i) != True:
            print("ERROR: Specified FF_file ({}) does not exist. Please check the path. Exiting...".format(i))
            quit()

    # Make the output directory if it doesn't already exist.
    if os.path.exists(Filename) and not args.ow:
        print('ERROR: The desired output folder ({}) already exists. Exiting to avoid overwriting data...'.format(Filename))
        return
    # Else, create a bare directory
    else:
        if os.path.exists(Filename) and args.ow:
            sp.call('rm -r {}'.format(Filename), shell=True)
        os.makedirs(Filename)
        sys.stdout = Logger(Filename)
        #print("PROGRAM CALL: python gen_periodic_md.py {}\n".format(' '.join([ i for i in argv])))

    # Catenate the FF files and copy them to the output folder
    FF_all = Filename+"/"+Filename.split('/')[-1]+'.db'
    with open(FF_all,'w') as f:
        for i in args.FF_files:
            with open(i,'r') as ff_file:
                for j in ff_file:
                    f.write(j)
            f.write("\n")


    # If a cg sim, get the lengths of different potential files
    if args.cg and args.tabulated:
        table_counts = get_table_counts(FF_all)
    else:
        table_counts = None

    if args.data_file is None:

        # Grab data on each molecule being added to the MD simulation.
        Data,unit_cell,cif_data = get_data(FF_all,args.coord_files,args.q_list,args.gens)#,single_map=args.single_map)
        # Pack the simulation cell
        Elements,Atom_types,Geometry,Bonds,Bond_types,Angles,Angle_types,Dihedrals,Dihedral_types,Impropers,Improper_types,Charges,Molecule,Molecule_names,Adj_mat,Sim_Box = \
        Generate_Box(Data,unit_cell,args.dims)
        # Generate VDW parameters
        VDW_params = initialize_VDW(sorted(set(Atom_types)),sigma_scale=args.sigma_scale,eps_scale=args.eps_scale,VDW_FF=Data[0]["VDW_params"],
                                    Force_UFF=args.force_UFF,mixing_rule=args.mixing_rule)

        # Generate Simulation Dictionaries
        # The bond, angle, and diehdral parameters for each molecule are combined into one dictionary
        Bond_params = {}; Angle_params = {}; Dihedral_params = {}; Improper_params = {}; Masses = {}
        for i in Data.keys():
            for j in Data[i]["Bond_params"].keys(): Bond_params[j] = Data[i]["Bond_params"][j]
            for j in Data[i]["Angle_params"].keys(): Angle_params[j] = Data[i]["Angle_params"][j]
            for j in Data[i]["Dihedral_params"].keys(): Dihedral_params[j] = Data[i]["Dihedral_params"][j]
            for j in Data[i]["Improper_params"].keys(): Improper_params[j] = Data[i]["Improper_params"][j]
            for j in Data[i]["Masses"].keys(): Masses[j] = Data[i]["Masses"][j]

        # Write the lammps datafile (the function returns a dictionary, Atom_type_dict, holding the mapping between atom_types and the lammps id numbers; this mapping is needed for setting fixes)
        print("Writing LAMMPS datafile ({})...".format(Filename.split('/')[-1]+'.data'))
        print("Writing LAMMPS settings file ({})...".format(Filename.split('/')[-1]+'.in.settings'))
        Atom_type_dict,fixed_modes = Write_data(Filename,Atom_types,Sim_Box,Elements,Geometry,Bonds,Bond_types,Bond_params,Angles,Angle_types,Angle_params,
                                                Dihedrals,Dihedral_types,Dihedral_params,Impropers,Improper_types,Improper_params,Charges,VDW_params,Masses,Molecule,Improper_flag = args.improper_flag,fixes=args.fixes)

        # Generate fix statement for scaled charges
        fixes = scale_charges(args.charge_scale, Atom_type_dict, Atom_types, Charges)

        # Gather the different dihedral_styles
        Dihedral_styles = set([Dihedral_params[i][0] for i in Dihedral_types])

    else:
        if args.unitcell is None or args.n_uc is None:
            print("Error: give unit cell parameters with flag -cg_uc or number of atoms in unit cell with -cg_nuc")
            quit()
        unit_cell = args.unitcell.split()
        unit_cell = [float(i) for i in unit_cell]
        # Read data file for unitcell data:
        Data = read_data_file(args.data_file, unit_cell, int(args.n_uc))
        cg_moles=Data.pop("moles", None)
        if args.cell_sym:
            cell_sym={"cif_filename":args.coord_files, "cg_moles":cg_moles, "map":args.single_map[0]}
        else:
            cell_sym={}
        cg_write_data(Data, unit_cell, Filename, args.dims, args.tether,args.coord_files, cell_sym=cell_sym)


        #copyfile(args.data_file,"{}/{}.data".format(Filename,Filename.split('/')[-1]))
        if not os.path.isfile(args.settings_file):
            print("No settings file provided with data file. Quitting...")
            quit()
        else:
            copyfile(args.settings_file,"{}/{}.in.settings".format(Filename,Filename.split('/')[-1]))

        # Initialize stuff we don't need
        # WILL NOT WORK FOR FIXED BONDS/ANGLES
        Bond_types = []
        Angle_types = []
        fixed_modes = {"bonds":[],"angles":[]}
        fixes = []
        Dihedral_styles = []
        cif_data = {"ambient_temp": 400.0}


    # Write the lammps input files
    print("Writing LAMMPS input file ({})...".format(Filename.split('/')[-1]+'.in.init'))
    Write_input(Filename,args.extname,cif_data["ambient_temp"] if args.T_equil == "none" else args.T_equil, args.t_equil, cif_data["ambient_temp"] if args.T_anneal == "none" else args.T_anneal,args.t_anneal,args.t_ext,
                args.pressure,args.frequency,args.onefourscale_coul,args.onefourscale_lj,args.pair_styles,Dihedral_styles,fixes,args.fixes,Bond_types,Angle_types,
               (cif_data["ambient_temp"]/2) if args.T_relax == "none" else args.T_relax,tail_opt=args.tail_opt,timestep=args.timestep,fixed_modes=fixed_modes,improper_flag=args.improper_flag,
               velocity_opt=args.velocity_flag,NVT_opt=args.NVT_opt,T_D=args.T_dilatometry,P_D=args.P_D,N_dilatometry=args.N_dilatometry,mol_opt=args.molecule_flag,units=args.units,table_counts=table_counts,tether=args.tether)

    if args.cg is None:
        # Write map file to more easily post-process the trajectory
        print("Writing mapfile ({})...".format(Filename.split('/')[-1]+'.map'))
        write_map(Filename,Elements,Atom_types,Charges,Masses,Adj_mat,zeros([len(Elements)]),Molecule[-1]+1)

        # Writing molecule map file
        print("Writing molfile ({})...".format(Filename.split('/')[-1]+'.mol.txt'))
        write_molecule(Filename,Molecule_names)

        # Write *.pairs file used by gen_jobs_for_vdw.py for determining the initial guess
        print("Writing pair file ({})...".format(Filename.split('/')[-1]+'.pairs'))
        Write_pairs(Filename,Atom_types,VDW_params,args.eps_scale,args.sigma_scale)

        # Remove concatenated FF file from run folder
        os.remove(FF_all)

    # Print banner
    #print("\n{}\n* {:^173s} *\n{}\n".format("*"*177,"Success! Have a Nice Day!","*"*177))

# Description: A wrapper for the write commands for generating the lammps input file
def Write_input(Filename,extname,equil_temp,equil_time,anneal_temp,anneal_time,extend_time,pressure,frequency,Onefourscale_coul,Onefourscale_lj,Pair_styles,
                Dihedral_styles,fixes,ba_fixes,Bond_types,Angle_types,relax_temp,tail_opt=False,timestep=1.0,skip_resize=False,fixed_modes={},
                improper_flag=False,velocity_opt=False,NVT_opt=False,T_D=None,P_D=None,N_dilatometry=20,mol_opt=False,units="real",table_counts=None,tether=None):

    # Initialize number of atom types for tether
    n_atom_types = 3

    # Initialize rigid bond/angle fix commands
    # NOTE: the ba_fixes arguments override the specific bond and angle fixes.
    ba_fix_cmd = ''

    # Case: linear triatomic:
    if "linear" in ba_fixes:
        ba_fix_cmd="fix rigid all rigid/small molecule\n"

    # Case: all bonds and angles are treated as fixed
    elif 'bonds' in ba_fixes  and 'angles' in ba_fixes:
        ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(" ".join([ str(count_i+1) for count_i,i in enumerate(set(Bond_types))])," ".join([ str(count_i+1) for count_i,i in enumerate(set(Angle_types))]))

    # Case: all bonds and only a subset of angles are treated as fixed
    elif 'bonds' in ba_fixes:
        bond_fixes  = " ".join([ str(count_i+1) for count_i,i in enumerate(set(Bond_types))])
        angle_fixes = " ".join([ str(i) for i in fixed_modes["angles"] ])
        if len(angle_fixes) == 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {}\n".format(bond_fixes)
        else:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(bond_fixes,angle_fixes)

    # Case: all angles and only a subset of bonds are treated as fixed
    elif 'angles' in ba_fixes:
        bond_fixes  = " ".join([ str(i) for i in fixed_modes["bonds"] ])
        angle_fixes = " ".join([ str(count_i+1) for count_i,i in enumerate(set(Angle_types))])
        if len(bond_fixes) == 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"a {}\n".format(angle_fixes)
        else:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(bond_fixes,angle_fixes)

    # Case: only a subset of bonds/angles are treated as fixed
    else:
        bond_fixes  = " ".join([ str(i) for i in fixed_modes["bonds"] ])
        angle_fixes = " ".join([ str(i) for i in fixed_modes["angles"] ])
        if len(bond_fixes) != 0 and len(angle_fixes) != 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {} a {}\n".format(bond_fixes,angle_fixes)            
        elif len(bond_fixes) != 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"b {}\n".format(bond_fixes)
        elif len(angle_fixes) != 0:
            ba_fix_cmd="fix rigid all rattle 0.0001 20 ${coords_freq} "+"a {}\n".format(angle_fixes)

    # Initialize velocity_string and molecule_string
    if velocity_opt is True:
        velocity_string = " vx vy vz"
    else:
        velocity_string = ""
    if mol_opt is True:
        molecule_string = " mol"
    else:
        molecule_string = ""

    with open(Filename+'/'+Filename.split('/')[-1]+'.in.init','w') as f:
        f.write("# lammps input file for polymer simulation with dilute ions\n\n"+\
                "# VARIABLES\n"+\
                "variable        data_name       index   {}\n".format(Filename.split('/')[-1]+'.data')+\
                "variable        settings_name   index   {}\n".format(Filename.split('/')[-1]+'.in.settings')+\
                "variable        log_name        index   {}\n".format(Filename.split('/')[-1]+'.log')+\
                "variable        nSteps_ramp     index   {} # number of data steps for the ramped anneal\n".format(int(round((anneal_time/2)/timestep)))+\
                "variable        nSteps_equil    index   {} # number of data steps for the ramped anneal\n".format(int(round(equil_time/timestep)))+\
                "variable        avg_freq        index   1000\n".format(int(round(frequency/timestep)))+\
                "variable        coords_freq     index   1000\n".format(int(round(frequency/timestep)))+\
                "variable        thermo_freq     index   1000\n".format(int(round(frequency/timestep)))+\
                "variable        dump4avg        index   100\n"+\
                "variable        vseed           index   {: <6d}\n".format(int(random.random()*100000))+\
                "variable        RELAX_TEMP      index   {} # Temperature during the constrained relaxation\n".format(relax_temp)+\
                "variable        ANNEAL_TEMP     index   {} # Temperature during the initial anneal\n".format(anneal_temp)+\
                "variable        FINAL_TEMP      index   {} # Temperature ramped to during the final anneal\n".format(equil_temp)+\
                "variable        pressure        index   {} # Pressure during the simulations\n\n".format(pressure)+\

                "# Change the name of the log output #\n"+\
                "log ${log_name}\n\n"+\
                
                "#===========================================================\n"+\
                "# GENERAL PROCEDURES\n"+\
                "#===========================================================\n"+\
                "units		{}	# g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n".format(units)+\
                "dimension	3	# 3 dimensional simulation\n"+\
                "newton		off	# use Newton's 3rd law\n"+\
                "boundary	p p p	# periodic boundary conditions \n"+\
                "atom_style	full    # molecular + charge\n\n"+\

                "#===========================================================\n"+\
                "# FORCE FIELD DEFINITION\n"+\
                "#===========================================================\n"+\
                'special_bonds  lj   0.0 0.0 {}  coul 0.0 0.0 {}     # NO     1-4 LJ/COUL interactions\n'.format(Onefourscale_lj,Onefourscale_coul))
        if table_counts is None:
            if "linear" in ba_fixes:
                f.write('pair_style     hybrid {} # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n'.format(Pair_styles))
            else:
                f.write('pair_style     hybrid {} # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n'.format(Pair_styles)+\
                        'bond_style     harmonic             # parameters needed: k_bond, r0\n'+\
                        'angle_style    harmonic             # parameters needed: k_theta, theta0\n')
            if len(Dihedral_styles) >= 1:
                f.write('dihedral_style hybrid {}            # parameters needed: V1, V2, V3, V4\n'.format(' '.join(Dihedral_styles)))
        else:
            f.write('pair_style     hybrid table linear {}\n'.format(table_counts['pair'])+\
                    'bond_style     hybrid table linear {}\n'.format(table_counts['bond']))
            if table_counts['angle'] != '':
                f.write('angle_style    hybrid table linear {}\n'.format(table_counts['angle']))
            if table_counts['dihedral'] != '':
                f.write('dihedral_style hybrid table linear {}\n'.format(table_counts['dihedral']))
        if 'coul/long' in Pair_styles:
            f.write('kspace_style   pppm 0.0001          # long-range electrostatics sum method\n')
        else:
            f.write('kspace_style   none          # long-range electrostatics sum method\n')

        if improper_flag:
            f.write('improper_style harmonic             # parameters needed: k_psi, psi0\n')
        if tail_opt is True:
            f.write('pair_modify    tail yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n\n')
        else:
            f.write('pair_modify    shift yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n\n')

        f.write("#===========================================================\n"+\
                "# SETUP SIMULATIONS\n"+\
                "#===========================================================\n\n"+\
                "# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n"+\
                'read_data ${data_name}\n'+\
                'include ${settings_name}\n\n'+\

                "# SET RUN PARAMETERS\n"+\
                "timestep	{}		# fs\n".format(timestep)+\
                "run_style	verlet 		# Velocity-Verlet integrator\n"+\
                "neigh_modify every 1 delay 0 check no one 10000 # More relaxed rebuild criteria can be used\n")
        # if tether:
        #     a_type_str = ""
        #     for i in range(1, n_atom_types):
        #         a_type_str+="{} ".format(i)
        #     f.write("group ghost type {}\n".format(n_atom_types)+\
        #             "group nonghost type {}\n".format(a_type_str)+\
        #             "neigh_modify exclude group ghost nonghost\n\n")
        if 'coul/long' in Pair_styles:
            f.write('kspace_style   pppm 0.0001          # long-range electrostatics sum method, redefine is necessary for triclinic box initialization.\n\n')
        else:
            f.write('kspace_style   none          # long-range electrostatics sum method, redefine is necessary for triclinic box initialization.\n\n')

        f.write("# SET OUTPUTS\n"+\
                "thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n"+\
                "thermo_modify   format float %14.6f\n"+\
                "thermo ${thermo_freq}\n\n"+\

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
                "variable        my_etot equal   etotal\n\n")

        # if tether:
        #     f.write("fix  nonghostaverages nonghost ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file thermo.avg format %20.10g\n\n")
        #     f.write("fix  ghostaverages ghost ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file ghostthermo.avg format %20.10g\n\n")
        # else:
        f.write("fix  averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file thermo.avg format %20.10g\n\n")



        if fixes != []:
            f.write("{}".format(fixes))

        f.write("#===========================================================\n"+\
                "# RUN CONSTRAINED RELAXATION\n"+\
                "#===========================================================\n\n"+

                "velocity        all create ${RELAX_TEMP} ${vseed} mom yes rot yes     # DRAW VELOCITIES\n")
        if "linear" in ba_fixes:
            f.write("fix relax all rigid/small molecule\n")
        else:
            f.write("fix relax all nve/limit 0.01\n")

        # Write rattle constraints
        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write(ba_fix_cmd)
        # if tether:
        #     f.write("fix still ghost move linear 0.0 0.0 0.0\n"+\
        #             "fix immob ghost setforce 0.0 0.0 0.0\n")
        if tether is not None:
            f.write("fix tether all spring/self {}\n".format(tether))

        f.write("run             10000\n")

        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write("unfix rigid\n")
        f.write("unfix relax\n\n")
        
        f.write("#===========================================================\n"+\
                "# RUN RAMPED ANNEAL\n"+\
                "#===========================================================\n\n")

        f.write("fix mom all momentum 1000 linear 1 1 1 angular # Zero out system linear and angular momentum every ps \n")
        if NVT_opt:
            # if tether:
            #     f.write("fix anneal nonghost nvt temp ${ANNEAL_TEMP} ${FINAL_TEMP} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n")
            # else:
            f.write("fix anneal all nvt temp ${ANNEAL_TEMP} ${FINAL_TEMP} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n")
        else:
            # if tether:
            #     f.write("fix anneal nonghost npt temp ${ANNEAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n")
            # else:
            if "linear" in ba_fixes:
                f.write("fix anneal all rigid/npt/small molecule temp ${ANNEAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n")
            else:
                f.write("fix anneal all npt temp ${ANNEAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n")

        f.write("# CREATE COORDINATE DUMPS FOR ANNEAL\n"+\
                "dump anneal all custom ${coords_freq} anneal.lammpstrj id type x y z " + velocity_string + "\n"+\
                "dump_modify anneal sort  id\n\n"+\

                "# RUN RAMPED ANNEAL\n")

        # Write rattle constraints
        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write(ba_fix_cmd)
        f.write("run ${nSteps_ramp}\n")
        # if tether:
        #     f.write("unfix still\n")
        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write("unfix rigid\n")
        f.write("unfix anneal\n\n"+\

                "# RUN EQUILIBRATION PHASE AT FINAL TEMP\n")
        if NVT_opt:
            f.write("fix anneal all nvt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 # NVT, nose-hoover 100 fs T relaxation\n")
        else:
            if "linear" in ba_fixes:
                f.write("fix anneal all rigid/npt/small molecule temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n")
            else:
                f.write("fix anneal all npt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n")

        # Write rattle constraints
        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write(ba_fix_cmd)
        f.write("run ${nSteps_ramp}\n")
        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write("unfix rigid\n")
        f.write("unfix anneal\n"+\
                "undump anneal\n\n")

        if equil_time > 0:
            f.write("#===========================================================\n"+\
                    "# RUN EQUILIBRIUM SIM\n"+\
                    "#===========================================================\n\n"+

                    "# UPDATE RUN PARAMETERS AND CREATE FIX\n")
            if NVT_opt:
                f.write("fix equil all nvt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n")
            else:
                if "linear" in ba_fixes:
                    f.write("fix equil all rigid/npt/small molecule temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n")
                else:
                    f.write("fix equil all npt temp ${FINAL_TEMP} ${FINAL_TEMP} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n")

            f.write("# CREATE COORDINATE DUMPS FOR EQUILIBRIUM\n"+\
                    "dump equil all custom ${coords_freq} equil.lammpstrj id type x y z " + velocity_string + "\n"+\
                    "dump_modify equil sort  id\n\n"+\

                    "# RUN EQUIL\n")

            # Write rattle constraints
            if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
                f.write(ba_fix_cmd)
            f.write("run		${nSteps_equil}\n")
            # if tether:
            #     f.write("unfix immob\n")
            if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
                f.write("unfix rigid\n")
            f.write("unfix equil\n"+\
                    "undump equil\n\n")
                        
        f.write("# WRITE RESTART FILES, CLEANUP, AND EXIT\n"+\
                "write_restart   {}\n".format(Filename.split('/')[-1]+'.end.restart')+\
                'write_restart   extend.end.restart\n'+\
                "write_data      {} pair ii\n".format(Filename.split('/')[-1]+'.end.data')+\
                "unfix		averages\n")




    # Write header of trajectory extension file
    with open(Filename+'/{}.in.init'.format(extname),'w') as f:
        f.write("# lammps input file for insertion trajectory\n\n"+\
                "# VARIABLES\n"+\
                "log {}\n".format('{}.log'.format(extname)))

        # No dilatometry option
        if T_D is None and P_D is None:
            f.write("variable    data_name       index   {}\n".format(Filename.split('/')[-1]+'.end.data')+\
                    "variable    restart_name    index   {}.end.restart\n".format(extname)+\
                    "variable    settings_name   index   {}\n".format(Filename.split('/')[-1]+'.in.settings')+\
                    "variable    n_equil         index   {}	  # number of data steps equilibration segment\n".format(int(round(extend_time/timestep)))+\
                    "variable    coords_freq     index   {} # the frequency that coordinates are prints\n".format(int(round(frequency/timestep)))+\
                    "variable    thermo_freq     index   {} # the frequency that instantaneous thermodynamic variables are printed to output.\n".format(int(round(frequency/timestep)))+\
                    "variable    avg_freq        index   {} # the frequency that the thermodynamics averages are printed (to file)\n".format(int(round(frequency/timestep)))+\
                    "variable    avg_spacing     index   1    # the spacing of snapshots used for average values calculations (printed to file)\n"+\
                    "variable    Temp            index   {}\n".format(equil_temp)+\
                    "variable    pressure        index   {} # Pressure during the simulations\n".format(pressure)+\
                    "variable    run             index   0\n\n")

        # Dilatometry related modification
        elif T_D is not None and P_D is None:
            f.write("variable    data_name       index   {}\n".format(Filename.split('/')[-1]+'.end.data')+\
                    "variable    restart_name    index   {}.end.restart\n".format(extname)+\
                    "variable    settings_name   index   {}\n".format(Filename.split('/')[-1]+'.in.settings')+\
                    "variable    n_equil         index   {}	  # number of data steps equilibration segment\n".format(int(round(extend_time/timestep)))+\
                    "variable    coords_freq     index   {} # the frequency that coordinates are prints\n".format(int(round(frequency/timestep)))+\
                    "variable    thermo_freq     index   {} # the frequency that instantaneous thermodynamic variables are printed to output.\n".format(int(round(frequency/timestep)))+\
                    "variable    avg_freq        index   {} # the frequency that the thermodynamics averages are printed (to file)\n".format(int(round(frequency/timestep)))+\
                    "variable    avg_spacing     index   1    # the spacing of snapshots used for average values calculations (printed to file)\n"+\
                    "variable    pressure        index   {} # Pressure during the simulations\n".format(pressure)+\
                    "variable    T0              index   {}\n".format(equil_temp)+\
                    "variable    T1              index   {}\n".format(T_D)+\
                    "variable    run             index   0\n"+\
                    "variable    intervals       index   {}\n".format(N_dilatometry)+\
                    'variable    T_init          equal   "v_T0 - (v_T0 - v_T1) / v_intervals * v_run"\n'+\
                    'variable    T_final         equal   "v_T0 - (v_T0 - v_T1) / v_intervals * ( v_run + 1 )"\n\n')
        elif P_D is not None and T_D is None:
            f.write("variable    data_name       index   {}\n".format(Filename.split('/')[-1] + '.end.data') + \
                    "variable    restart_name    index   {}.end.restart\n".format(extname) + \
                    "variable    settings_name   index   {}\n".format(Filename.split('/')[-1] + '.in.settings') + \
                    "variable    n_equil         index   {}	  # number of data steps equilibration segment\n".format(int(round(extend_time / timestep))) + \
                    "variable    coords_freq     index   {} # the frequency that coordinates are prints\n".format(int(round(frequency / timestep))) + \
                    "variable    thermo_freq     index   {} # the frequency that instantaneous thermodynamic variables are printed to output.\n".format(int(round(frequency / timestep))) + \
                    "variable    avg_freq        index   {} # the frequency that the thermodynamics averages are printed (to file)\n".format(int(round(frequency / timestep))) + \
                    "variable    avg_spacing     index   1    # the spacing of snapshots used for average values calculations (printed to file)\n" + \
                    "variable    Temp            index   {} # Temperature during the simulations\n".format(equil_temp) + \
                    "variable    P0              index   {}\n".format(pressure) + \
                    "variable    P1              index   {}\n".format(P_D) + \
                    "variable    run             index   0\n" + \
                    "variable    intervals       index   {}\n".format(N_dilatometry) + \
                    'variable    P_init          equal   "v_P0 - (v_P0 - v_P1) / v_intervals * v_run"\n' + \
                    'variable    P_final         equal   "v_P0 - (v_P0 - v_P1) / v_intervals * ( v_run + 1 )"\n\n')

        f.write("#===========================================================\n"+\
                "# GENERAL PROCEDURES\n"+\
                "#===========================================================\n"+\
                "units		{}	# g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n".format(units)+\
                "dimension	3	# 3 dimensional simulation\n"+\
                "newton		off	# use Newton's 3rd law\n"+\
                "boundary	p p p	# periodic boundary conditions \n"+\
                "atom_style	full    # molecular + charge\n\n"+\

                "#===========================================================\n"+\
                "# FORCE FIELD DEFINITION\n"+\
                "#===========================================================\n"+\
                'special_bonds  lj   0.0 0.0 {}  coul 0.0 0.0 {}     # NO     1-4 LJ/COUL interactions\n'.format(Onefourscale_lj,Onefourscale_coul))
        if table_counts is None:
            if "linear" in ba_fixes:
                f.write('pair_style     hybrid {} # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n'.format(Pair_styles))
            else:
                f.write('pair_style     hybrid {} # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n'.format(Pair_styles)+\
                        'bond_style     harmonic             # parameters needed: k_bond, r0\n'+\
                        'angle_style    harmonic             # parameters needed: k_theta, theta0\n')
            if len(Dihedral_styles) >= 1:
                f.write('dihedral_style hybrid {}            # parameters needed: V1, V2, V3, V4\n'.format(' '.join(Dihedral_styles)))
        else:
            f.write('pair_style     hybrid table linear {}\n'.format(table_counts['pair'])+\
                    'bond_style     hybrid table linear {}\n'.format(table_counts['bond']))
            if table_counts['angle'] != '':
                f.write('angle_style    hybrid table linear {}\n'.format(table_counts['angle']))
            if table_counts['dihedral'] != '':
                f.write('dihedral_style hybrid table linear {}\n'.format(table_counts['dihedral']))
        if 'coul/long' in Pair_styles:
            f.write('kspace_style   pppm 0.0001          # long-range electrostatics sum method\n')
        else:
            f.write('kspace_style   none          # long-range electrostatics sum method\n')

        if improper_flag:
            f.write('improper_style harmonic             # parameters needed: k_psi, psi0\n')
        if tail_opt is True:
            f.write('pair_modify    tail yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n\n')
        else:
            f.write('pair_modify    shift yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n\n')

        f.write("#===========================================================\n"+\
                "# SETUP SIMULATIONS\n"+\
                "#===========================================================\n\n"+\
                "# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n")
        f.write('read_restart ${restart_name}\n'+\
                'include ${settings_name}\n\n')

        f.write("# SET RUN PARAMETERS\n"+\
                "timestep	{}		# fs\n".format(timestep)+\
                "run_style	verlet 		# Velocity-Verlet integrator\n"+\
                "neigh_modify every 1 delay 10 check yes one 10000\n")
        # if tether:
        #     a_type_str = ""
        #     for i in range(1, n_atom_types):
        #         a_type_str+="{} ".format(i)
        #     f.write("group ghost type {}\n".format(n_atom_types)+\
        #             "group nonghost type {}\n".format(a_type_str)+\
        #             "neigh_modify exclude group ghost nonghost\n\n")
        if 'coul/long' in Pair_styles:
            f.write('kspace_style   pppm 0.0001          # long-range electrostatics sum method, redefine is necessary for triclinic box initialization.\n\n')
        else:
            f.write('kspace_style   none          # long-range electrostatics sum method, redefine is necessary for triclinic box initialization.\n\n')


        f.write("# SET OUTPUTS\n"+\
                "thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press etotal\n"+\
                "thermo_modify   format float %14.6f\n"+\
                "thermo ${thermo_freq}\n\n"+\

                "# Declare relevant output variables and create averages fix\n"+\
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
                "variable        my_etot equal   etotal\n\n")

        # if tether:
        #     f.write("fix nonghostaverages nonghost ave/time ${avg_spacing} $(v_thermo_freq/v_avg_spacing) ${thermo_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file ${run}.thermo.avg format %20.10g\n")
        #     f.write("fix ghostaverages ghost ave/time ${avg_spacing} $(v_thermo_freq/v_avg_spacing) ${thermo_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file ${run}.ghostthermo.avg format %20.10g\n\n")
        # else:
        f.write("fix averages all ave/time ${avg_spacing} $(v_thermo_freq/v_avg_spacing) ${thermo_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file ${run}.thermo.avg format %20.10g\n\n")

        f.write("#===========================================================\n"+\
                "# RUN EXTENSION\n"+\
                "#===========================================================\n\n"+\

                "# Set npt/nvt fix for the runs\n")
        # if tether:
        #     f.write("fix mom nonghost momentum 1000 linear 1 1 1 angular # Zero out system linear and angular momentum every ps \n")
        # else:
        f.write("fix mom all momentum 1000 linear 1 1 1 angular # Zero out system linear and angular momentum every ps \n")

        # No dilatometry option
        if T_D is None and P_D is None:
            if NVT_opt:
                # if tether:
                #     f.write('fix tmp nonghost nvt temp ${Temp} ${Temp} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')
                # else:
                f.write('fix tmp all nvt temp ${Temp} ${Temp} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')
            else:
                # if tether:
                #     f.write('fix tmp nonghost npt temp ${Temp} ${Temp} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')
                # else:
                if "linear" in ba_fixes:
                    f.write('fix tmp all rigid/npt/small molecule temp ${Temp} ${Temp} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')
                else:
                    f.write('fix tmp all npt temp ${Temp} ${Temp} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')

        # Dilatometry related modifications
        elif T_D is not None and P_D is None:
            if NVT_opt:
                f.write('fix tmp all nvt temp ${T_init} ${T_final} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')
            else:
                if "linear" in ba_fixes:
                    f.write('fix tmp all rigid/npt/small molecule temp ${T_init} ${T_final} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')
                else:
                    f.write('fix tmp all npt temp ${T_init} ${T_final} 100.0 iso ${pressure} ${pressure} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')
        elif P_D is not None and T_D is None:
            if NVT_opt:
                f.write('fix tmp all nvt temp ${Temp} ${Temp} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')
                print("WARNING: NVT option chosen with pressure sweep.")
            else:
                if "linear" in ba_fixes:
                    f.write('fix tmp all rigid/npt/small molecule temp ${Temp} ${Temp} 100.0 iso ${P_init} ${P_final} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')
                else:
                    f.write('fix tmp all npt temp ${Temp} ${Temp} 100.0 iso ${P_init} ${P_final} 1000.0 # NPT, nose-hoover 100 fs T relaxation\n\n')


        f.write('# Create coordinate dump\n')
        f.write('dump equil all custom ${coords_freq} ${run}.sys.lammpstrj id type x y z' + velocity_string + molecule_string + "\n"+\
                'dump_modify equil sort  id\n')

        # Write rattle constraints
        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write(ba_fix_cmd)

        # if tether:
        #     f.write("fix immob ghost setforce 0.0 0.0 0.0\n")

        # Write run command
        f.write('\n# Run equilibration segment\n'+\
                'run		${n_equil}\n\n')

        # if tether:
        #     f.write("unfix immob\n")
        
        # Write unfix command
        if len(ba_fix_cmd) != 0 and "linear" not in ba_fixes:
            f.write("unfix rigid\n\n")

        f.write('# Write restart files, cleanup, and exit\n'+\
                'write_restart   {}.end.restart\n'.format(extname)+\
                'write_data      {}.end.data pair ii\n\n'.format(extname)+\

                '# Reset dump and increment\n'+\
                'undump equil\n\n'+\

                '# Update run number\n'+\
                'variable sub equal (v_run+1)\n'+\
                'shell sed -i /variable.*run.*index/s/${run}/${sub}/g '+'{}.in.init\n\n'.format(extname))

        f.close()

    return


# Description: A wrapper for the commands to generate a cubic box and array of molecules for the lammps run
def Generate_Box(Data,unit_cell,dims):

    N_atoms = sum([ len(Data[_]["Elements"]) for _ in Data ])
    N_atoms = ( dims[0] ) * ( dims[1] ) * ( dims[2] ) * N_atoms
    Geometry_sim = zeros([N_atoms,3])
    Adj_mat_sim  = zeros([N_atoms,N_atoms])
    Molecule_sim       = []
    Molecule_files     = []
    Elements_sim       = [] 
    Atom_types_sim     = []
    Bonds_sim          = []
    Bond_types_sim     = []
    Angles_sim         = []
    Angle_types_sim    = []
    Dihedrals_sim      = []
    Dihedral_types_sim = []
    Impropers_sim      = []
    Improper_types_sim = []
    Charges_sim        = []
    Masses_sim         = []
    Elements_sim       = []
    centroids_list     = []

    # Initialize t_mat
    t_mat = fract_to_cartesian(unit_cell["a"],unit_cell["b"],unit_cell["c"],unit_cell["alpha"],unit_cell["beta"],unit_cell["gamma"])

    # Place the translations
    atoms_placed = 0
    atoms_not_placed = 0
    mols_placed = 0
    print("\nGenerating simulation cell...")
    print("\nTotal number of atoms in simulation: {}".format(N_atoms))
    for a in range(dims[0]):
        for b in range(dims[1]):
            for c in range(dims[2]):
            
                # Append the translated geometry only if its centroid does not match any other centroid currently placed
                for i in Data:
                    
                    # Check its centroid against a list of other centroids that have been placed
                    place = True
                    geom = Data[i]["Frac_geo"]+array([a,b,c])
                    geom_centroid = geom.mean(axis=0)
                    
                    for j in range(len(centroids_list)):
                        distance = around(linalg.norm(geom_centroid-centroids_list[j]), decimals=5)                        
                        if distance == 0:
                            atoms_not_placed += len(Data[i]["Elements"])
                            place = False
                            break
                    
                    if place:

                        # Add the appropriately transformed unit cell to the geometry
                        centroids_list.append(geom_centroid)
                        Geometry_sim[atoms_placed:atoms_placed + len(Data[i]["Elements"])] = dot(geom,t_mat)

                        # Add duplicates
                        for d in [ 1*(dims[0]), 0, -1*(dims[0]) ]:
                            for e in [ 1*(dims[1]), 0, -1*(dims[1]) ]:
                                for f in [ 1*(dims[2]), 0,  -1*(dims[2]) ]:
                                    if d == 0 and e == 0 and f == 0:
                                        continue
                                    else:
                                        centroids_list.append(geom_centroid + array([d,e,f]))

                        # Extend various lists (total elements lists, atomtypes lists, etc)
                        # Note: the lammps input expects bonds,angles,dihedrals, etc to be defined in terms of atom
                        #       id so, the atom_index is employed to keep track of how many atoms have been placed.
                        Adj_mat_sim[atoms_placed:(atoms_placed+len(Data[i]["Geometry"])),atoms_placed:(atoms_placed+len(Data[i]["Geometry"]))] = Data[i]["Adj_mat"]
                        Molecule_sim       = Molecule_sim + [mols_placed]*len(Data[i]["Elements"])
                        Molecule_files     = Molecule_files + [i]
                        Elements_sim       = Elements_sim + Data[i]["Elements"]
                        Atom_types_sim     = Atom_types_sim + Data[i]["Atom_types"]
                        Bonds_sim          = Bonds_sim + [ (j[0]+atoms_placed,j[1]+atoms_placed) for j in Data[i]["Bonds"] ]
                        Bond_types_sim     = Bond_types_sim + Data[i]["Bond_types"]
                        Angles_sim         = Angles_sim + [ (j[0]+atoms_placed,j[1]+atoms_placed,j[2]+atoms_placed) for j in Data[i]["Angles"] ]
                        Angle_types_sim    = Angle_types_sim + Data[i]["Angle_types"]
                        Dihedrals_sim      = Dihedrals_sim + [ (j[0]+atoms_placed,j[1]+atoms_placed,j[2]+atoms_placed,j[3]+atoms_placed) for j in Data[i]["Dihedrals"] ]
                        Dihedral_types_sim = Dihedral_types_sim + Data[i]["Dihedral_types"]
                        Impropers_sim      = Impropers_sim + [ (j[0]+atoms_placed,j[1]+atoms_placed,j[2]+atoms_placed,j[3]+atoms_placed) for j in Data[i]["Impropers"] ]
                        Improper_types_sim = Improper_types_sim + Data[i]["Improper_types"]
                        Charges_sim        = Charges_sim + Data[i]["Charges"]
                        Masses_sim         = Masses_sim + [ Data[i]["Masses"][j] for j in Data[i]["Atom_types"] ] 

                        # Increment atom_index based on the number of atoms in the current geometry
                        atoms_placed += len(Data[i]["Elements"])
                        mols_placed  += 1

    # Remove any empty rows if molecules were not placed
    if atoms_not_placed > 0:
        Geometry_sim = Geometry_sim[:-atoms_not_placed]
        Adj_mat_sim = Adj_mat_sim[:-atoms_not_placed]
    
    # The relevant relationships are taken from the lammps doc page for "How To: Triclinic"
    lx = unit_cell["a"] * (dims[0])
    xy = ( unit_cell["b"] * (dims[1]))*np.cos(unit_cell["gamma"]*pi/180.)
    xz = ( unit_cell["c"] * (dims[2]))*np.cos(unit_cell["beta"]*pi/180.)
    ly = ( (unit_cell["b"] * (dims[1]) )**(2.0) - xy**(2.0) )**(0.5)
    yz = ( (unit_cell["b"] * (dims[1]) )*(unit_cell["c"] * (dims[2])) * np.cos(unit_cell["alpha"]*pi/180.) - xy*xz ) / ly
    lz = ( (unit_cell["c"] * (dims[2]) )**(2.0) - xz**(2.0) - yz**(2.0) )**(0.5)

    # OLD
    xlo = ylo = zlo =  0.0
    xhi, yhi, zhi = Geometry_sim.max(axis=0)        # get max value in each column
    xlo_bound = xlo + min([0.0, xy, xz, xy+xz])
    xhi_bound = xhi + max([0.0, xy, xz, xy+xz])
    ylo_bound = ylo + min([0.0, yz])
    yhi_bound = yhi + max([0.0, yz])
    zlo_bound = zlo
    zhi_bound = zhi

    with open("box_dims.txt",'w') as f:
        f.write('xlo ylo zlo\n')
        f.write('{} {} {}\n\n'.format(xlo, ylo, zlo))
        f.write('xhi yhi zhi\n')
        f.write('{} {} {}\n\n'.format(xhi, yhi, zhi))
        
        f.write('lx: {}\n'.format(lx))
        f.write('xy: {}\n'.format(xy))
        f.write('xz: {}\n'.format(xz))
        f.write('ly: {}\n'.format(ly))
        f.write('yz: {}\n'.format(yz))
        f.write('lz: {}\n\n'.format(lz))
        
        f.write('xlo_bound: {}\n'.format(xlo_bound))
        f.write('xhi_bound: {}\n'.format(xhi_bound))
        f.write('ylo_bound: {}\n'.format(ylo_bound))
        f.write('yhi_bound: {}\n'.format(yhi_bound))
        f.write('zlo_bound: {}\n'.format(zlo_bound))
        f.write('zhi_bound: {}\n'.format(zhi_bound))
    
    # Find the smallest (currently set to 0,0,0) and largest values in each dimension, then add some padding depending on the above bounding cals.
# OLD:    Sim_Box = array([xlo_bound, xhi_bound, ylo_bound, yhi_bound, zlo_bound, zhi_bound, xy, xz, yz])
    Sim_Box = array([0, lx, 0.0, ly, 0.0, lz, xy, xz, yz])
    
    return Elements_sim,Atom_types_sim,Geometry_sim,Bonds_sim,Bond_types_sim,Angles_sim,Angle_types_sim,Dihedrals_sim,Dihedral_types_sim,Impropers_sim,Improper_types_sim,Charges_sim,Molecule_sim,Molecule_files,Adj_mat_sim,Sim_Box
                            
# Wrapper function for writing the lammps data file which is read by the .in.init file during run initialization
def Write_data(Filename,Atom_types,Sim_Box,Elements,Geometry,Bonds,Bond_types,Bond_params,Angles,Angle_types,Angle_params,Dihedrals,Dihedral_types,Dihedral_params,
               Impropers,Improper_types,Improper_params,Charges,VDW_params,Masses,Molecule,Improper_flag=False,fixes=""):

    # Write an xyz for easy viewing
    with open(Filename+'/'+Filename.split('/')[-1]+'.xyz','w') as f:
        f.write('{}\n\n'.format(len(Geometry)))
        for count_i,i in enumerate(Geometry):
            f.write('{:20s} {:< 20.6f} {:< 20.6f} {:< 20.6f}\n'.format(Elements[count_i],i[0],i[1],i[2]))

    # Create type dictionaries (needed to convert each atom,bond,angle, and dihedral type to consecutive numbers as per LAMMPS convention)
    # Note: LAMMPS orders atomtypes, bonds, angles, dihedrals, etc as integer types. Each of the following dictionaries holds the mapping between
    #       the true type (held in the various types lists) and the lammps type_id, obtained by enumerated iteration over the respective set(types).
    Atom_type_dict = {}
    for count_i,i in enumerate(sorted(set(Atom_types))):
        for j in Atom_types:
            if i == j:
                Atom_type_dict[i]=count_i+1
            if i in Atom_type_dict.keys():
                break
    Bond_type_dict = {}
    for count_i,i in enumerate(sorted(set(Bond_types))):
        for j in Bond_types:
            if i == j:
                Bond_type_dict[i]=count_i+1
            if i in Bond_type_dict.keys():
                break
    Angle_type_dict = {}
    for count_i,i in enumerate(sorted(set(Angle_types))):
        for j in Angle_types:
            if i == j:
                Angle_type_dict[i]=count_i+1
            if i in Angle_type_dict.keys():
                break
    Dihedral_type_dict = {}
    for count_i,i in enumerate(sorted(set(Dihedral_types))):
        for j in Dihedral_types:
            if i == j:
                Dihedral_type_dict[i]=count_i+1
            if i in Dihedral_type_dict.keys():
                break
    Improper_type_dict = {}
    for count_i,i in enumerate(sorted(set(Improper_types))):
        for j in Improper_types:
            if i == j:
                Improper_type_dict[i]=count_i+1
            if i in Improper_type_dict.keys():
                break

    # Write the data file
    with open(Filename+'/'+Filename.split('/')[-1]+'.data','w') as f:
        
        # Write system properties
        f.write("LAMMPS data file via gen_periodic_md.py, on {}\n\n".format(datetime.datetime.now()))

        f.write("{} atoms\n".format(len(Elements)))
        f.write("{} atom types\n".format(len(set(Atom_types))))
        if "linear" not in fixes:
            if len(Bonds) > 0:
                f.write("{} bonds\n".format(len(Bonds)))
                f.write("{} bond types\n".format(len(set(Bond_types))))
            if len(Angles) > 0:
                f.write("{} angles\n".format(len(Angles)))
                f.write("{} angle types\n".format(len(set(Angle_types))))
            if len(Dihedrals) > 0:
                f.write("{} dihedrals\n".format(len(Dihedrals)))
                f.write("{} dihedral types\n".format(len(set(Dihedral_types))))
            if Improper_flag and len(Impropers) > 0:
                f.write("{} impropers\n".format(len(Impropers)))
                f.write("{} improper types\n".format(len(set(Improper_types))))
        f.write("\n")

        # Write box dimensions
        f.write("{:< 20.16f} {:< 20.16f} xlo xhi\n".format(Sim_Box[0],Sim_Box[1]))
        f.write("{:< 20.16f} {:< 20.16f} ylo yhi\n".format(Sim_Box[2],Sim_Box[3]))
        f.write("{:< 20.16f} {:< 20.16f} zlo zhi\n".format(Sim_Box[4],Sim_Box[5]))
        if Sim_Box[6] > 1e-4 or Sim_Box[7] > 1e-4 or Sim_Box[8] > 1e-4:
            f.write("{:< 20.16f} {:< 20.16f} {:< 20.16f} xy xz yz\n\n".format(Sim_Box[6],Sim_Box[7],Sim_Box[8]))
        else:
            f.write("\n")

        # Write Masses
        f.write("Masses\n\n")
        for count_i,i in enumerate(sorted(set(Atom_types))):
            for j in set(Atom_types):
                if Atom_type_dict[j] == count_i+1:
                    f.write("{} {:< 8.6f}\n".format(count_i+1,Masses[str(j)])) # count_i+1 bc of LAMMPS 1-indexing
        f.write("\n")

        # Write Atoms
        f.write("Atoms\n\n")
        for count_i,i in enumerate(Atom_types):
            f.write("{:<8d} {:< 4d} {:< 4d} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:d} {:d} {:d}\n"\
            .format(count_i+1,Molecule[count_i],Atom_type_dict[i],Charges[count_i],Geometry[count_i,0],Geometry[count_i,1],Geometry[count_i,2],0,0,0))


        if "linear" not in fixes:
            # Write Bonds
            if len(Bonds) > 0:
                f.write("\nBonds\n\n")
                for count_i,i in enumerate(Bonds):
                    f.write("{:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Bond_type_dict[Bond_types[count_i]],i[0]+1,i[1]+1))

            # Write Angles
            if len(Angles) > 0:
                f.write("\nAngles\n\n")
                for count_i,i in enumerate(Angles):
                    f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Angle_type_dict[Angle_types[count_i]],i[0]+1,i[1]+1,i[2]+1))

            # Write Dihedrals
            if len(Dihedrals) > 0:
                f.write("\nDihedrals\n\n")
                for count_i,i in enumerate(Dihedrals):
                    f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Dihedral_type_dict[Dihedral_types[count_i]],i[0]+1,i[1]+1,i[2]+1,i[3]+1))

            # Write Impropers
            if Improper_flag and len(Impropers) > 0:
                f.write("\nImpropers\n\n")
                for count_i,i in enumerate(Impropers):
                    f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Improper_type_dict[Improper_types[count_i]],i[0]+1,i[1]+1,i[2]+1,i[3]+1))

    # Write the settings file
    fixed_modes = {'bonds':[], 'angles':[]}
    with open(Filename+'/'+Filename.split('/')[-1]+'.in.settings','w') as f:

        # Write non-bonded interactions (the complicated form of this loop is owed
        # to desire to form a nicely sorted list in terms of the lammps atom types
        # Note: Atom_type_dict was initialize according to sorted(set(Atom_types) 
        #       so iterating over this list (twice) orders the pairs, too.
        f.write("     {}\n".format("# Non-bonded interactions (pair-wise)"))
        for count_i,i in enumerate(sorted(set(Atom_types))):     
            for count_j,j in enumerate(sorted(set(Atom_types))): 

                # Skip duplicates
                if count_j < count_i:
                    continue

                # Conform to LAMMPS i <= j formatting
                if Atom_type_dict[i] <= Atom_type_dict[j]:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[i],Atom_type_dict[j]))
                else:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[j],Atom_type_dict[i]))

                # Determine key (ordered by initialize_VDW function such that i > j)
                if i > j:
                    key = (i,j)
                else:
                    key = (j,i)

                # Write the parameters
                for k in VDW_params[key]:
                    if type(k) is str:
                        f.write("{:20s} ".format(k))
                    if type(k) is float:
                        f.write("{:< 20.6f} ".format(k))
                f.write("\n")                        

        # Write stretching interactions
        # Note: Bond_type_dict was initialized by looping over sorted(set(Bond_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Stretching interactions"))
        for i in sorted(set(Bond_types)):
            if "fixed" in Bond_params[i]:
                continue
            f.write("     {:20s} {:<10d} ".format("bond_coeff",Bond_type_dict[i]))
            for j in Bond_params[i]:
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
            f.write("\n")

            # populate fixed_modes
            if Bond_params[i][0] == "fixed":
                fixed_modes["bonds"] += [Bond_type_dict[i]]

        # Write bending interactions
        # Note: Angle_type_dict was initialized by looping over sorted(set(Angle_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Bending interactions"))
        for i in sorted(set(Angle_types)):
            if "fixed" in Angle_params[i]:
                continue
            f.write("     {:20s} {:<10d} ".format("angle_coeff",Angle_type_dict[i]))
            for j in Angle_params[i]:
                if j == "fixed":
                    continue
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
            f.write("\n")

            # populate fixed_modes
            if Angle_params[i][0] == "fixed":
                fixed_modes["angles"] += [Angle_type_dict[i]]

        # Write dihedral interactions
        # Note: Dihedral_type_dict was initialized by looping over sorted(set(Dihedral_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Dihedral interactions"))
        for i in sorted(set(Dihedral_types)):
            f.write("     {:20s} {:<10d} ".format("dihedral_coeff",Dihedral_type_dict[i]))
            for j in Dihedral_params[i]:
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
                if type(j) is int:
                    f.write("{:< 20d} ".format(j))
            f.write("\n")

        # Write improper interactions
        # Note: Improper_type_dict was initialized by looping over sorted(set(Improper_types)), so 
        #       iterating over this list resulted in ordered parameters in the in.settings file
        if Improper_flag:
            f.write("\n     {}\n".format("# Improper interactions"))
            for i in sorted(set(Improper_types)):
                f.write("     {:20s} {:<10d} ".format("improper_coeff",Improper_type_dict[i]))
                for j in Improper_params[i][1:]:
                    if type(j) is str:
                        f.write("{:20s} ".format(j))
                    if type(j) is float:
                        f.write("{:< 20.6f} ".format(j))
                    if type(j) is int:
                        f.write("{:< 20d} ".format(j))
                f.write("\n")

    return Atom_type_dict,fixed_modes

# A wrapper for the commands to parse the bonds, angles, and dihedrals from the adjacency matrix.
# Returns:   list of atomtypes, bond_types, bond instances, angle_types, angle instances, dihedral_types,
#            diehdral instances, charges, and VDW parameters.
def Find_parameters(Adj_mat,Geometry,Atom_types,FF_db="FF_file",Improper_flag = False):

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
    if Improper_flag:    print("Parsing impropers...")
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
    if len(Bonds) > 0: Bonds,Bond_types = zip(*[ (i,Bond_types[count_i]) for count_i,i in enumerate(Bonds) if count_i == [ count_j for count_j,j in enumerate(Bonds) if j == i or j[::-1] == i ][0]  ])
    if len(Angles) > 0: Angles,Angle_types = zip(*[ (i,Angle_types[count_i]) for count_i,i in enumerate(Angles) if count_i == [ count_j for count_j,j in enumerate(Angles) if j == i or j[::-1] == i ][0]  ])
    if len(Dihedrals) > 0: Dihedrals,Dihedral_types = zip(*[ (i,Dihedral_types[count_i]) for count_i,i in enumerate(Dihedrals) if count_i == [ count_j for count_j,j in enumerate(Dihedrals) if j == i or j[::-1] == i ][0]  ])
    if len(Impropers) > 0: Impropers,Improper_types = zip(*[ (i,Improper_types[count_i]) for count_i,i in enumerate(Impropers) if count_i == [ count_j for count_j,j in enumerate(Impropers) if j[0] == i[0] and len(set(i[1:]).intersection(set(j[1:]))) ][0] ])

    ##############################################################
    # Read in parameters: Here the stretching, bending, dihedral #
    # and non-bonding interaction parameters are read in from    #
    # parameters file. Mass and charge data is also included.    #
    # The program looks for a simple match for the first entry   #
    # in each line with one of the bond or angle types.          #
    # INPUT: param_file, BOND_TYPES_LIST, ANGLE_TYPES_LIST       #
    #        DIHEDRAL_TYPES_LIST, ELEMENTS                       #
    # OUTPUT: CHARGES, MASSES, BOND_PARAMS, ANGLE_PARAMS,        #
    #         DIHERAL_PARAMS, PW_PARAMS                          #
    ##############################################################

    # Initialize dictionaries

    # Read in masses and charges
    Masses = {}
    with open(FF_db,'r') as f:
        content=f.readlines()
        
    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'atom':
            Masses[fields[1]] = float(fields[3]) 

    # Read in bond parameters
    Bond_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0] == 'bond':
            if fields[3] == "harmonic":
                Bond_params[(fields[1],fields[2])] = [float(fields[4]),float(fields[5])]
            elif fields[3] == "fixed" and fields[4] == "linear":
                Bond_params[(fields[1], fields[2])] = ["fixed"]
            elif fields[3] == "fixed":
                Bond_params[(fields[1],fields[2])] = ["fixed",0.0,float(fields[4])]
            elif fields[3] == "table":
                Bond_params[(fields[1], fields[2])] = ["table", fields[4], "POTENTIAL_1"]
            else:
                print("ERROR: only harmonic bond definitions are currently supported by gen_md_for_sampling.py. Exiting...")
                quit()

    # Read in angle parameters
    Angle_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'angle':
            if fields[4] == "harmonic":
                Angle_params[(fields[1],fields[2],fields[3])] = [float(fields[5]),float(fields[6])]
            elif fields[4] == "fixed" and fields[5] == "linear":
                Angle_params[(fields[1], fields[2], fields[3])] = ["fixed"]
            elif fields[4] == "fixed":
                Angle_params[(fields[1],fields[2],fields[3])] = ["fixed",0.0,float(fields[5])]
            elif fields[4] == "table":
                Angle_params[(fields[1], fields[2], fields[3])] = ["table", fields[5], "POTENTIAL_1"]
            else:
                print("ERROR: only harmonic angle definitions are currently supported by gen_md_for_sampling.py. Exiting...")
                quit()

    # Read in dihedral parameters
    Dihedral_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['dihedral','torsion']:            
            if fields[5] == "opls":
                Dihedral_params[(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(i) for i in fields[6:10] ]
            elif fields[5] == "harmonic":
                Dihedral_params[(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(fields[6]),int(float(fields[7])),int(float(fields[8])) ]
            elif fields[5] == "table":
                Dihedral_params[(fields[1], fields[2], fields[3], fields[4], fields[5])] = ["table", fields[6],
                                                                                            "POTENTIAL_1"]
            else:
                print("ERROR: Only opls and harmonic dihedral types are currently supported by gen_md_for_sampling.py. Exiting...")
                quit()

    # Read in improper parameters
    Improper_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['improper']:
            if fields[5] == "harmonic":
                Improper_params[(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(fields[6]),float(fields[7])]
            elif fields[5] == "table":
                Improper_params[(fields[1], fields[2], fields[3], fields[4])] = ["table", fields[6], "POTENTIAL_1"]
            else:
                print("ERROR: Only opls type dihedral definitions are currently supported by gen_md_for_vdw.py. Exiting...")
                quit()
                
    # Search for charges based on atom type
    with open(FF_db,'r') as f:
        content=f.readlines()

    Charges = zeros(len(Atom_types))
    for i in range(len(Atom_types)):
        for lines in content:
            fields=lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue
                    
            if fields[0].lower() in ['charge'] and Atom_types[i] == fields[1]:
                Charges[i] = float(fields[2])

    # Search for VDW parameters
    VDW_params = {}
    with open(FF_db,'r') as f:
        for lines in f:
            fields = lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue
                    
            if fields[0].lower() in ['vdw']:

                # Only two parameters are required for lj types
                if fields[3] == "lj":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5])]
                    else:
                        VDW_params[(fields[2],fields[1])] = [fields[3],float(fields[4]),float(fields[5])]
                elif fields[3] == "buck":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5]),float(fields[6])]
                    else:
                        VDW_params[(fields[2],fields[1])] = [fields[3],float(fields[4]),float(fields[5]),float(fields[6])]
                elif fields[3] == "table":
                    VDW_params[(fields[1], fields[2])] = ["table", fields[4], "POTENTIAL_1"]

    # Check for missing parameters
    Missing_masses = [ i for i in Atom_types if str(i) not in Masses.keys() ] 
    Missing_charges = [ count_i for count_i,i in enumerate(Charges) if i == -100.0 ]; Missing_charges = [ Atom_types[i] for i in Missing_charges ]
    Missing_bonds = [ i for i in Bond_types if (i[0],i[1]) not in Bond_params.keys() ]
    Missing_angles = [ i for i in Angle_types if (i[0],i[1],i[2]) not in Angle_params.keys() ]
    Missing_dihedrals = [ i for i in Dihedral_types if (i[0],i[1],i[2],i[3]) not in Dihedral_params.keys() ]
    Missing_impropers = []
    if Improper_flag is True: Missing_impropers = [ i for i in Improper_types if (i[0],i[1],i[2],i[3]) not in Improper_params.keys() ]

    # Print diagnostics on missing parameters and quit if the prerequisites are missing.
    if ( len(Missing_masses) + len(Missing_charges) + len(Missing_bonds) + len(Missing_angles) + len(Missing_dihedrals) + len(Missing_impropers) ) > 0:
        print("\nUh Oh! There are missing FF parameters...\n")

        if Missing_masses:
            print("Missing masses for the following atom types: {}".format([ i for i in set(Missing_masses) ]))
        if Missing_charges:
            print("Missing charges for the following atom types: {}".format([ i for i in set(Missing_charges) ]))
        if Missing_bonds:
            print("Missing bond parameters for the following bond types: {}".format([ i for i in set(Missing_bonds) ]))
        if Missing_angles:
            print("Missing angle parameters for the following angle types: {}".format([ i for i in set(Missing_angles) ]))
        if Missing_dihedrals:
            print("Missing dihedral parameters for the following dihedral types: {}".format([ i for i in set(Missing_dihedrals) ]))
        if Improper_flag and Missing_impropers:
            print("Missing improper parameters for the following improper types: {}".format([ i for i in set(Missing_impropers) ]))
        
        print("\nEnsure the specification of the missing parameters. Exiting...")
        quit()

    return list(Bonds),list(Bond_types),Bond_params,list(Angles),list(Angle_types),Angle_params,list(Dihedrals),list(Dihedral_types),Dihedral_params,list(Impropers),list(Improper_types),Improper_params,Charges.tolist(),Masses,VDW_params

# Description: finds the number of disconnected subnetworks in the 
#              adjacency matrix, which corresponds to the number of 
#              separate molecules.
#
# Inputs:      adj_mat: numpy array holding a 1 in the indices of bonded
#                        atom types. 
#
# Returns:     mol_count: scalar, the number of molecules in the adj_mat
def mol_count(adj_mat):
    
    # Initialize list of atoms assigned to molecules and a counter for molecules
    placed_idx = []    
    mol_count = 0

    # Continue the search until all the atoms have been assigned to molecules
    while len(placed_idx)<len(adj_mat):

        # Use sequential elements of the adj_mat as seeds for the spanning network search
        for count_i,i in enumerate(adj_mat):

            # Only proceed with search if the current atom hasn't been placed in a molecule
            if count_i not in placed_idx:

                # Increment mol_count for every new seed and add the seed to the list of placed atoms
                mol_count += 1               
                placed_idx += [count_i]
                
                # Find connections
                idx = [ count_j for count_j,j in enumerate(i) if j==1 and count_j not in placed_idx ]
                
                # Continue until no new atoms are found
                while len(idx) > 0:
                    current = idx.pop(0)
                    if current not in placed_idx:
                        placed_idx += [current]
                        idx += [ count_k for count_k,k in enumerate(adj_mat[current]) if k == 1 and count_k not in placed_idx ]
    return mol_count

# Wrapper function for the write commands for creating the *.map file
def write_map(Filename,Elements,Atom_types,Charges,Masses,Adj_mat,Structure,N_mol):

    # Open file for writing and write header (first two lines of the map file are header)    
    with open(Filename+'/'+Filename.split('/')[-1]+'.map','w') as f:
        f.write('{} {}\n {:<50} {:<10} {:<10} {:<14}  {:<13} {}\n'.format(len(Atom_types),sum(N_mol),'Atom_type','Element','Structure','Mass','Charge','Adj_mat'))
        for count_i,i in enumerate(Atom_types):
            adj_mat_entry = (' ').join([ str(count_j) for count_j,j in enumerate(Adj_mat[count_i,:]) if j == 1 ])
            f.write(' {:<50} {:<10} {:< 9d} {:<14.6f} {:< 14.8f} {}\n'.format(i,Elements[count_i],int(Structure[count_i]),Masses[str(i)],Charges[count_i],adj_mat_entry))
        f.close()

# Wrapper function for the write commands for creating the *.map file
def write_molecule(Filename,Molecules):

    # Open file for writing and write header (first two lines of the map file are header)    
    with open(Filename+'/'+Filename.split('/')[-1]+'.mol.txt','w') as f:
        f.write('{:<30s} {:<30s}\n'.format("Molecule_id","Molecule_Name"))
        for count_i,i in enumerate(Molecules):
            f.write("{:<30} {:<30}\n".format(count_i,i))

        # Write mol clusters
        f.write('\nLists for each molecule:\n')
        for i in sorted(set(Molecules)):
            f.write('{}: '.format(i))
            for count_j,j in enumerate(Molecules):
                if i == j:
                    f.write(' {}'.format(count_j))
            f.write('\n')

# Description: Initialize VDW_dict based on UFF parameters for the initial guess of the fit.
def initialize_VDW(atomtypes,sigma_scale=1.0,eps_scale=1.0,VDW_FF={},Force_UFF=0,mixing_rule='lb'):

    # Initialize UFF parameters (tuple corresponds to eps,sigma pairs for each element)
    # Taken from UFF (Rappe et al. JACS 1992)
    # Note: LJ parameters in the table are specificed in the eps,r_min form rather than eps,sigma
    #       the conversion between r_min and sigma is sigma = r_min/2^(1/6)
    # Note: Units for sigma = angstroms and eps = kcal/mol 
    UFF_dict = { 1:(0.044,2.5711337005530193),  2:(0.056,2.1043027722474816),  3:(0.025,2.183592758161972),  4:(0.085,2.4455169812952313),\
                 5:(0.180,3.6375394661670053),  6:(0.105,3.4308509635584463),  7:(0.069,3.260689308393642),  8:(0.060,3.1181455134911875),\
                 9:(0.050,2.996983287824101),  10:(0.042,2.88918454292912),   11:(0.030,2.657550876212632), 12:(0.111,2.6914050275019648),\
                13:(0.505,4.008153332913386),  14:(0.402,3.82640999441276),   15:(0.305,3.694556984127987), 16:(0.274,3.594776327696269),\
                17:(0.227,3.5163772404999194), 18:(0.185,3.4459962417668324), 19:(0.035,3.396105913550973), 20:(0.238,3.0281647429590133),\
                21:(0.019,2.935511276272418),  22:(0.017,2.828603430095577),  23:(0.016,2.800985569833227), 24:(0.015,2.6931868249382456),\
                25:(0.013,2.6379511044135446), 26:(0.013,2.5942970672246677), 27:(0.014,2.558661118499054), 28:(0.015,2.5248069672097215),\
                29:(0.005,3.113691019900486),  30:(0.124,2.4615531582217574), 31:(0.415,3.904809081609107), 32:(0.379,3.813046513640652),\
                33:(0.309,3.7685015777336357), 34:(0.291,3.746229109780127),  35:(0.251,3.731974730289881), 36:(0.220,3.689211591819145),\
                37:(0.040,3.6651573264293558), 38:(0.235,3.2437622327489755), 39:(0.072,2.980056212179435), 40:(0.069,2.78316759547042),\
                41:(0.059,2.819694442914174),  42:(0.056,2.7190228877643157), 43:(0.048,2.670914356984738), 44:(0.056,2.6397329018498255),\
                45:(0.053,2.6094423454330538), 46:(0.048,2.5827153838888437), 47:(0.036,2.804549164705788), 48:(0.228,2.537279549263686),\
                49:(0.599,3.976080979060334),  50:(0.567,3.9128271700723705), 51:(0.449,3.937772334180300), 52:(0.398,3.982317270087316),\
                53:(0.339,4.009044231631527),  54:(0.332,3.923517954690054),  55:(0.045,4.024189509839913), 56:(0.364,3.2989979532736764),\
                72:(0.072,2.798312873678806),  73:(0.081,2.8241489365048755), 74:(0.067,2.734168165972701), 75:(0.066,2.631714813386562),\
                76:(0.037,2.7796040005978586), 77:(0.073,2.5301523595185635), 78:(0.080,2.453535069758495), 79:(0.039,2.9337294788361374),\
                80:(0.385,2.4098810325696176), 81:(0.680,3.872736727756055),  82:(0.663,3.828191791849038), 83:(0.518,3.893227398273283),\
                84:(0.325,4.195242063722858),  85:(0.284,4.231768911166611),  86:(0.248,4.245132391938716) }

    # NEW: Initialize VDW_dict first guess based on element types and Lorentz-Berthelot mixing rules
    # Order of operations: (1) If the parameters are in the supplied FF database then they are used as is
    # (2) If the self-terms are in the supplied FF datebase then they are used to generate the mixed
    # interactions (3) UFF parameters are used. 
    VDW_dict = {}
    origin = {}
    VDW_styles = []
    for count_i,i in enumerate(atomtypes):
        for count_j,j in enumerate(atomtypes):
            if count_i < count_j:
                continue

            # Check for parameters in the database
            if (i,j) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(i,j)][0] == "lj":
                    if "coul" not in VDW_FF[(i,j)]:
                        VDW_type = "lj/cut"
                    else:
                        VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(i,j)][0] == "buck":
                    VDW_type = "buck/coul/long"
                elif VDW_FF[(i,j)][0] == "table":
                    VDW_type = "table"
                else:
                    print("ERROR in initialize_VDW: only lj, buck, and table pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type] + VDW_FF[(i,j)][1:]
                else:
                    VDW_dict[(j,i)] = [VDW_type] + VDW_FF[(i,j)][1:]
                origin[(i,j)] = origin[(j,i)] = "read"

            # Check for reverse combination
            elif (j,i) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(j,i)][0] == "lj":
                    if "coul" not in VDW_FF[(j,i)]:
                        VDW_type = "lj/cut"
                    else:
                        VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(j,i)][0] == "buck":
                    VDW_type = "buck/coul/long"
                elif VDW_FF[(j,i)][0] == "table":
                    VDW_type = "table"
                else:
                    print("ERROR in initialize_VDW: only lj, buck, and table pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type] + VDW_FF[(j,i)][1:]
                else:
                    VDW_dict[(j,i)] = [VDW_type] + VDW_FF[(j,i)][1:]
                origin[(i,j)] = origin[(j,i)] = "read"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i,i) in VDW_FF and (j,j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'lb':

                # Check compatibility with mixing rules
                if VDW_FF[(i,i)][0] != "lj" or VDW_FF[(j,j)][0] != "lj":
                    print("ERROR in initialize_VDW: only lj styles support mixing rules. Exiting...")
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                eps    = (VDW_FF[(i,i)][1]*VDW_FF[(j,j)][1])**(0.5) * eps_scale
                sigma  = (VDW_FF[(i,i)][2]+VDW_FF[(j,j)][2])/2.0 * sigma_scale
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "lb"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i,i) in VDW_FF and (j,j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'wh':

                # Check compatibility with mixing rules
                if VDW_FF[(i,i)][0] != "lj" or VDW_FF[(j,j)][0] != "lj":
                    print("ERROR in initialize_VDW: only lj styles support mixing rules. Exiting...")
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                sigma  = ((VDW_FF[(i,i)][2]**(6.0)+VDW_FF[(j,j)][2]**(6.0))/2.0)**(1.0/6.0)
                eps    = (VDW_FF[(i,i)][1]*VDW_FF[(i,i)][2]**(6.0) * VDW_FF[(j,j)][1]*VDW_FF[(j,j)][2]**(6.0) )**(0.5) / sigma**(6.0)
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "wh"

            # Last resort: Use UFF parameters.
            else:
                VDW_type = "lj/cut/coul/long"
                type_1 = int(i.split('[')[1].split(']')[0])
                type_2 = int(j.split('[')[1].split(']')[0])
                eps    = (UFF_dict[type_1][0]*UFF_dict[type_2][0])**(0.5) * eps_scale
                sigma  = (UFF_dict[type_1][1]+UFF_dict[type_2][1])/2.0 * sigma_scale
                if i > j:
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "UFF"
                
            # Collect a list of the LAMMPS styles used in the simulation
            VDW_styles += [VDW_type]

    # Print summary
    print("\n{}".format("*"*177))
    print("* {:^173s} *".format("Initializing VDW parameters for the simulation (those with * were read from the FF file(s))"))
    print("*{}*".format("-"*175))
    print("* {:<50s} {:<50s} {:<20s}  {:<18s} {:<18s} {:<8s}    *".format("Type","Type","VDW_type","eps (kcal/mol)","sigma (angstroms)","origin"))
    print("{}".format("*"*177))
    for j in VDW_dict.keys():
        print("  {:<50s} {:<50s} {:<20s} {:< 18.4f} {:< 18.4f}  {:<18s}".format(j[0],j[1],VDW_dict[j][0],VDW_dict[j][1],VDW_dict[j][2],origin[j]))
    print("")

    return VDW_dict

# Description:   A wrapper for the commands to create a data dictionary holding the FF, mode, and geometry information for each molecule
# Inputs:  Filename:
# Returns: Data: a dictionary keyed to the list of coord_files
#                a keyed entry is itself a dictionary holding the geometry, 
#                elements, charges, bonds, bond_types, angles, angle_types,
#                dihedrals, dihedral_types and masses.
#                subdictionary keys (e.g., Data[*]["Geometry"])
#                Geometry: array of the molecule's geometry
#                Elements: list of the molecule's elements (indexed to the rows of "Geometry")
#                Atom_types: list of the molecule's atom_types (indexed to the rows of "Geometry")
#                Charges: list of the molecule's charges (indexed to the rows of "Geometry")
#                Bonds: a list of the molecule's bonds, each element being a tuple of the atom indices that are bonded
#                Angles: a list of the molecule's angles, each element being a tuple of the atom indices that form an angle mode
#                Dihedrals: a list of the molecule's dihedrals, each element being a tuple of the atom indices that form a dihedral
#                Bond_types: a list of the molecule's bond_types, indexed to Bonds
#                Angle_types: a list of the molecule's angle_types, indexed to Angles
#                Dihedral_types: a list of the molecule's dihedral_types, indexed to Dihedrals
#
def get_data(FF_all,coord_files,q_list,gens,Improper_flag=False):

    # Initialize dictionary to hold the FF, mode, and geometry information about each molecule
    Data,unit_cell,cif_data = parse_cif(coord_files[0],gens=gens)
    while len(q_list) < len(Data):
        q_list += [q_list[-1]]

    # Iterate over all molecules being simulated and collect their FF, mode, and geometry information.
    placed_list = []
    for count_i,i in enumerate(Data):

        # Check the number of molecules
        mol_in_in = mol_count(Data[i]["Adj_mat"])
        if mol_in_in > 1:
            print("ERROR: {} molecules were discovered in geometry {}. Check the geometry of the input file. Exiting...".format(mol_in_in,i))
            quit()

        # Generate list of bonds angles and dihedrals    
        print("\n{}\n* {:^163s} *\n{}\n".format("*"*167,"Parsing Modes and FF Information for Molecule {}".format(i),"*"*167))
        Data[i]["Bonds"],Data[i]["Bond_types"],Data[i]["Bond_params"],Data[i]["Angles"],Data[i]["Angle_types"],Data[i]["Angle_params"],\
        Data[i]["Dihedrals"],Data[i]["Dihedral_types"],Data[i]["Dihedral_params"],Data[i]["Impropers"],Data[i]["Improper_types"],Data[i]["Improper_params"],Data[i]["Charges"],Data[i]["Masses"],Data[i]["VDW_params"] =\
            Find_parameters(Data[i]["Adj_mat"],Data[i]["Geometry"],Data[i]["Atom_types"],FF_db=FF_all,Improper_flag = Improper_flag)

        # Subtract off residual
        if q_list[count_i] == "round":
            q_tot = int(round(sum(Data[i]["Charges"])))
        elif q_list[count_i] != "none":
            q_tot = int(q_list[count_i])

        # Avoid subtracting off the residual if q_list is none
        if q_list[count_i] != "none":
            correction = (float(q_tot)-sum(Data[i]["Charges"]))/float(len(Data[i]["Atom_types"]))
            for j in range(len(Data[i]["Atom_types"])):
                Data[i]["Charges"][j] += correction

        # Print diagnostics if this is the first instance of this molecule type
        if True not in [ array_equal(Data[i]["Adj_mat"],_) for _ in placed_list ]:

            print("\n{}\n* {:^163s} *\n{}\n".format("*"*167,"Parsing Modes and FF Information for Molecule {}".format(i),"*"*167))

            # Print System characteristics
            print("\n{}\n* {:^163s} *\n{}\n".format("*"*167,"Mode Summary for Molecule {}".format(i),"*"*167))
            print("\nAtom_types ({}):\n".format(len(set(Data[i]["Atom_types"]))))
            for j in sorted(set(Data[i]["Atom_types"])):
                print("\t{}".format(j))
            print("\nBond types ({}):\n".format(len(set(Data[i]["Bond_types"]))))
            for j in sorted(set(Data[i]["Bond_types"])):
                print("\t{}".format(j))
            print("\nAngle types ({}):\n".format(len(set(Data[i]["Angle_types"]))))
            for j in sorted(set(Data[i]["Angle_types"])):
                print("\t{}".format(j))
            print("\nDihedral types ({}):\n".format(len(set(Data[i]["Dihedral_types"]))))
            for j in sorted(set(Data[i]["Dihedral_types"])):
                print("\t{}".format(j))
            if Improper_flag:
                print("\nImproper types ({}):\n".format(len(set(Data[i]["Improper_types"]))))
                for j in sorted(set(Data[i]["Improper_types"])):
                    print("\t{}".format(j))


            print("\n{:40s} {}".format("Residual Charge:",sum(Data[i]["Charges"])))

            if q_list[count_i] != "none":
                print("{:40s} {}".format("Charge added to each atom:",correction))

            print("{:40s} {}".format("Final Total Charge:",sum(Data[i]["Charges"])))
            print("\n{}".format("*"*167))
            print("* {:^163s} *".format("System Characteristics"))
            print("*{}*".format("-"*165))
            print("* {:<87s} {:<24s} {:<24s} {:<25s} *".format("Type","Element","Mass","Charge"))
            print("{}".format("*"*167))
            for j in range(len(Data[i]["Atom_types"])):
                print(" {:<88s} {:<23s} {:< 24.6f} {:< 24.6f}".format(Data[i]["Atom_types"][j],Data[i]["Elements"][j],Data[i]["Masses"][str(Data[i]["Atom_types"][j])],Data[i]["Charges"][j]))

        placed_list += [ Data[i]["Adj_mat"] ]

    return Data,unit_cell,cif_data

# A simple wrapper function for generating a fix statement for scaled charges
def scale_charges(charge_scale,Atom_type_dict,Atom_types,Charges):

    fixes = []
    if charge_scale != 1.0:
        fixes = '# Setting up fixes to scale the charges by {}\n'.format(charge_scale)
        for i in sorted([ Atom_type_dict[j] for j in Atom_type_dict.keys() ]):
            fixes += 'group charge_{} type {}\n'.format(i,i)        
        for i in sorted([ Atom_type_dict[j] for j in Atom_type_dict.keys() ]):
            index = Atom_types.index([ k for k in Atom_type_dict.keys() if Atom_type_dict[k] == i ][0])
            fixes += 'variable charge_{} equal {}\n'.format(i,Charges[index]*charge_scale)
        for i in sorted([ Atom_type_dict[j] for j in Atom_type_dict.keys() ]):        
            index = Atom_types.index([ k for k in Atom_type_dict.keys() if Atom_type_dict[k] == i ][0])
            fixes += 'fix {} charge_{} adapt 0 atom charge v_charge_{} reset yes\n'.format(i,i,i)
        fixes += '\n'
    return fixes

# A Simple wrapper function for writing the unscaled pair parameters
def Write_pairs(Filename,Atom_types,VDW_params,eps_scale,sigma_scale):

    # EXACT DUPLICATE OF THE LOOP IN THE WRITE_DATA FUNCTION
    # Create type dictionaries (needed to convert each atom,bond,angle, and dihedral type to consecutive numbers as per LAMMPS convention)
    # Note: LAMMPS orders atomtypes, bonds, angles, dihedrals, etc as integer types. Each of the following dictionaries holds the mapping between
    #       the true type (held in the various types lists) and the lammps type_id, obtained by enumerated iteration over the respective set(types).
    Atom_type_dict = {}
    for count_i,i in enumerate(sorted(set(Atom_types))):
        for j in Atom_types:
            if i == j:
                Atom_type_dict[i]=count_i+1
            if i in Atom_type_dict.keys():
                break

    # Write the settings file
    with open(Filename+'/'+Filename.split('/')[-1]+'.pairs','w') as f:

        # Write non-bonded interactions (the complicated form of this loop is owed
        # to desire to form a nicely sorted list in terms of the lammps atom types
        # Note: Atom_type_dict was initialize according to sorted(set(Atom_types) 
        #       so iterating over this list (twice) orders the pairs, too.
        f.write("     {}\n".format("# Unscaled non-bonded interactions used for fitting vdw parameters"))
        for count_i,i in enumerate(sorted(set(Atom_types))):     
            for count_j,j in enumerate(sorted(set(Atom_types))): 

                # Skip duplicates
                if count_j < count_i:
                    continue

                # Conform to LAMMPS i <= j formatting
                if Atom_type_dict[i] <= Atom_type_dict[j]:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[i],Atom_type_dict[j]))
                else:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[j],Atom_type_dict[i]))

                # Determine key (ordered by initialize_VDW function such that i > j)
                if i > j:
                    key = (i,j)
                else:
                    key = (j,i)

                # Write the parameters
                f.write("{:20s} {:< 20.6f} {:< 20.6f}\n".format(VDW_params[key][0],VDW_params[key][1]/eps_scale,VDW_params[key][2]/sigma_scale))
                    
def fract_to_cartesian(a, b, c, alpha, beta, gamma, degrees=True):
    # Return the transformation matrix to convert fractional coords to cartesian coords
    # a, b, c = lengths
    # alpha, beta, gamma = angles
    # degrees = True if angles are in degrees, False if in radians
    # Returns a 3x3 transformation matrix
    
    # Convert angles to radians
    if degrees:
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
    
    # Calculate required 
    alpha_cos = np.cos(alpha)
    beta_cos = np.cos(beta)
    gamma_cos = np.cos(gamma)
    gamma_sin = np.sin(gamma)
    
    volume = np.sqrt(1.0 - alpha_cos**2 - beta_cos**2 - gamma_cos**2 + 2.0 * alpha_cos * beta_cos * gamma_cos)
    
    trans_mat = np.matrix( [
            [a, b * gamma_cos,  c * beta_cos],
            [0, b * gamma_sin, (c * (alpha_cos - beta_cos * gamma_cos) / gamma_sin)],
            [0, 0,              c * volume / gamma_sin] ])
    
    return trans_mat.T

def cartesian_to_fract(a, b, c, alpha, beta, gamma, degrees=True):
    # Return the transformation matrix to convert cartesian coords to fractional coords
    # a, b, c = lengths
    # alpha, beta, gamma = angles
    # degrees = True if angles are in degrees, False if in radians
    # Returns a 3x3 transformation matrix
    
    # Convert angles to radians
    if degrees:
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
    
    # Calculate required 
    alpha_cos = np.cos(alpha)
    beta_cos = np.cos(beta)
    gamma_cos = np.cos(gamma)
    gamma_sin = np.sin(gamma)
    
    volume = np.sqrt(1.0 - alpha_cos**2 - beta_cos**2 - gamma_cos**2 + 2.0 * alpha_cos * beta_cos * gamma_cos)
    
    trans_mat = np.matrix( [
            [1/a, -gamma_cos / (a * gamma_sin), (alpha_cos * gamma_cos - beta_cos)/(a * volume * gamma_sin)],
            [0, 1 / (b * gamma_sin), (beta_cos * gamma_cos - alpha_cos) / (b * volume * gamma_sin)],
            [0, 0, gamma_sin / (c * volume)]
            ])
    
    return trans_mat

def find_index(array, value):
    # Finds the index of a given value in a list of a list array ([[index1, value1], [index2, value2]])
    # Returns a list of any matches
    # e.g.: [(2,1), (0,0)], where the first number corresponds to the outter list index and the second array corresponds to the inner list
    index = [(i, row.index(value)) for i, row in enumerate(array) if value in row]
    return index

def find_molecules(adj_mat):
    # Searches through the supplied adjacency matrix and finds subgraphs, which correspond to
    # individual molecules. Returns a list containing lists of the atom indices.
    subgraph_list = []
    placed_list   = []

    for count_i,i in enumerate(adj_mat):
        # skip if the current node has already been placed in a subgraph                                                                                                                                                               
        if count_i in placed_list: continue

        # subgraph is seeded with count_i and its connections
        subgraph = [count_i] + [ count_j for count_j,j in enumerate(i) if j == 1 ]

        # new holds the updated list of nodes in the subgraph at the end of each bond search                                                                                                                                           
        new = [ count_j for count_j,j in enumerate(i) if j == 1 ]

        # recursively add nodes to new until no new nodes are found                                                                                                                                                                    
        while new != []:
            # Initialize list to hold new connnected nodes                                                                                                                                                                             
            connections = []

            # Iterate over the new nodes and find new connections                                                                                                                                                                      
            for count_j,j in enumerate(new):
                connections += [ count_k for count_k,k in enumerate(adj_mat[j]) if k == 1 and count_k not in subgraph ] # add new connections not already in the subgraph                                                              

            # Update lists                                                                                                                                                                                                             
            connections = list(set(connections))
            subgraph += connections
            new = connections

        # Sort nodes in the subgraph (makes them unique)                                                                                                                                                                               
        subgraph = sorted(subgraph)
        subgraph_list += [subgraph]
        placed_list += subgraph
    return subgraph_list

def read_cif(cif_file):
    # Reads a .cif file and returns its data. The main body of the cif is returned as a dictionary (with unique key: value), while the information contained
    # inside block statments are returned as two list of list arrays (each blocks' header/key information and each blocks' values) sharing the same index
    # Returns cif_data (a dictionary), loop_header, loop_data
    cif_data = {}                   # Directory holding the main body key/value pairs.
    loop = False                    # Flag if we are inside a loop block
    loop_header = []                # Holds the loop header information (the corresponding column names). List of a list. [[col1, col2, col3],[col1, col2]]
    loop_header_holder = []         # Temporary holder to collect all of a given loop block's headers
    loop_data = []                  # Holds thh loop data. List of a list. Order should match the order in loop_header (i.e. index n of header corresponds to data in n)
    loop_data_holder = []           # Temporary holder to collect all of a given loop block's data
    loop_header_flag = False        # Flag if we are in the header section of a loop block
    semicolon_flag = False          # Flag to avoid those semicolon areas. Probably should redo, since this corresponds to a string with spaces....
    multi_line_key = ''             # Key value for when a multi-line comment is encounterd. The value will be empty (only have 1 component on a line), while the actual key
                                    # value will be on the following lines bounded by ';'. This is done so for large sections of text.
    multi_line_value = ''           # Holds the value for the multi-lined key
    multi_line_flag = False

    # Read .cif file and store all relevant data
    with open(cif_file) as f:
        for line in f:
            
            if line[0:1] == '#':
                # Skip comments
                continue
        
            elif line[0:1] == ';':
                multi_line_flag == False
                # Indicates the start/end of a text block. Include everything between these in the key that is missing a value.
                # If there is text on the ; line, include that too (but omit the leading '; ')
                
                # Start of block
                if semicolon_flag == False:
                    semicolon_flag = True
                    
                    line2 = line.split()
                    if len(line2) > 1:
                        # If there is text on this line, include everything after the ';'.
                        multi_line_value = multi_line_value + line[1:]
            
                elif semicolon_flag == True:
                     # End of block
                     semicolon_flag = False
                     
                     line2 = line.split()
                     if len(line2) > 1:
                         # If there is text on this line, include everything after the ';' and following space.
                        multi_line_value = multi_line_value + line[2:]
                     
                     cif_data.update({multi_line_key:multi_line_value})
                     multi_line_key = ''
                     multi_line_value = ''
            
            elif len(line.split()) == 0:
                # skip blank lines
                loop = False
            
                # also interpret this blank line as an end to a loop block
                if len(loop_data_holder) > 0:
                    loop_data.append(loop_data_holder)
                    loop_data_holder = []
                
            elif line[0:1] == '_':
                # Are either in a matching key/value data section, or in the header section for a loop block
                if loop == True and loop_header_flag == True:
                    # We are in the header information of the loop section. Add to the loop_header_holder array
                    line = line.strip()
                    loop_header_holder.append(line)
                
                else:
                    # We have cleared loop header and data sections and have encountered the next new data key/value pair outside of the loop block
                    # (i.e. the current loop block has ended). Set the loop flag to False. Append the line.
                    loop == False
                
                    if len(loop_data_holder) > 0:
                        loop_data.append(loop_data_holder)
                        loop_data_holder = []
                    line2 = line.split()
                
                    if len(line2) > 2:
                        # Since spaces can only appear inside quotations, this means that the value is encapsulated in quotations
                        # If a line has >2 entries, value is a string with spaces
                        # If a line has 2 entries, key/value pair, since the key and value will be separated by spaces
                        # If the line has 1 entry, that means the value appears on the following line between ';'

                        line = line[line.find("'"):].strip()
                        line = line.strip("'")
                        cif_data.update({line2[0]:line})

                    elif len(line2) == 2:
                        cif_data.update({line2[0]:line2[1]})

                    elif len(line2) == 1:
                        # We have a blank value, assume that the value is on the following lines, bounded by ';'
                        multi_line_key = line2[0]
                        multi_line_flag = True
                
            elif 'loop_' in line:
                # We are entering a loop block. Set appropriate flags. The combination of loop and loop_header_flag
                # control if we append to the loop header (header and loop_header_flag both equal true) or to loop data
                # (only loop equals true)
                loop = True
                loop_header_flag = True
                
                # Have we just ended a previous loop block? If so, append that previous loop data and clear everything.
                if len(loop_data_holder) > 0:
                    loop_data.append(loop_data_holder)
                    loop_data_holder = []
        
            elif line[0:1] != '_' and loop == True and semicolon_flag == False:
                # this is the data section of the loop file. we have cleared the header, so set that flag to False.
                # add the header holder list for this section to the loop_header list.
                # add these values to their own holder array 
                
                if len(loop_header_holder) > 0:
                    loop_header.append(loop_header_holder)
               
                loop_header_holder = []
                loop_header_flag = False
                line = line.strip()
                loop_data_holder.append(line)
            
            elif line[0:1] != '_' and semicolon_flag == True:
                # Inside the text block delimited by ';'. Add this line to the value to be included with the key.
                multi_line_value = multi_line_value + line
            
            elif line[0:1] != '_' and multi_line_flag == True and semicolon_flag == False:
                # This is the matching value for when a key is blank but the value isn't delimited by ';'.
                cif_data.update({multi_line_key:line})
            
            elif '#END' in line:
                if len(loop_data_holder) > 0:
                    loop_data.append(loop_data_holder)
                    loop_data_holder = []
    
    # Append any loop data we may have missed, especially if the last section of data was a loop
    # block and this loop block was terminated by a space and then the end of the file without
    # a '#END' tag
    if len(loop_data_holder) > 0:
        loop_data.append(loop_data_holder)
        loop_data_holder = []

    loop_header_holder = []
    loop_data_holder = []
    
    return cif_data, loop_header, loop_data

# Function for parsing cif file
def parse_cif(cif_filename,gens=2):
        
    # Append the file extension to the .cif input filename if it's missing.
    if not cif_filename.lower().endswith('.cif'):
        cif_filename = cif_filename + '.cif'
    
    # Check that the .cif file exists
    if not os.path.isfile(cif_filename):
        print('ERROR: .cif file not found. Aborting....')
        exit()
        
    # Read cif file
    cif_data, loop_header, loop_data = read_cif(cif_filename)

    # Check integrety of loop headers and data (for each loop block, the the number of header columns
    # should match the number of columns in the data array
    if len(loop_header) != len(loop_data):
        print('ERROR: Number of loop header blocks does not match the number of loop data blocks. Aborting...')
        exit()
        
    # Check for error in parsing cif file
    for i in range(len(loop_header)):
        for j in range(len(loop_data[i])):
            if len(loop_header[i]) != len(loop_data[i][j].split()):
                if "'" not in loop_data[i][j]:
                    print('ERROR: Number of header and data columns does not match. Aborting....')
                    print('Header: ' + str(loop_header[i]))
                    print('Data: ' + str(loop_data[i][j]))
                    exit()

    # Get the unit cell dimensions (angles alpha, beta, gamma and lengths a, b, c)
    # Check to make sure they aren't zero. Remove standard deviations, if present.
    if '_cell_angle_alpha' not in cif_data:
        print('ERROR: Alpha value is empty or not found in cif. Aborting...')
        exit()
    else:
        alpha = cif_data['_cell_angle_alpha']
        
    if '_cell_angle_beta' not in cif_data:
        print('ERROR: Beta value is empty or not found in cif. Aborting...')
        exit()
    else:
        beta = cif_data['_cell_angle_beta']
    
    if '_cell_angle_gamma' not in cif_data:
        print('ERROR: Gamma value is empty or not found in cif. Aborting...')
        exit()
    else:
        gamma = cif_data['_cell_angle_gamma']

    if '_cell_length_a' not in cif_data:
        print('ERROR: Cell length a value is empty or not found in cif. Aborting...')
        exit()
    else:
        a = cif_data['_cell_length_a']
        
    if '_cell_length_b' not in cif_data:
        print('ERROR: Cell length b value is empty or not found in cif. Aborting...')
        exit()
    else:
        b = cif_data['_cell_length_b']
        
    if '_cell_length_c' not in cif_data:
        print('ERROR: Cell length c value is empty or not found in cif. Aborting...')
        exit()
    else:
        c = cif_data['_cell_length_c']
    
    ambient_flag = False
    if '_diffrn_ambient_temperature' not in cif_data:
        print('Ambient temperature not specified in .cif file. Defaulting to room temperature...')
        ambient_temp = 300
        ambient_flag = True
    else:
        ambient_temp = cif_data['_diffrn_ambient_temperature']
    
    # Remove standard deviation, if present
    if '(' in alpha:
       alpha = alpha[:alpha.find('(')]
    alpha = float(alpha)
    
    if '(' in beta:
       beta = beta[:beta.find('(')]
    beta = float(beta)

    if '(' in gamma:
       gamma = gamma[:gamma.find('(')]
    gamma = float(gamma)
    
    if '(' in a:
        a = a[:a.find('(')]
    a = float(a)
    
    if '(' in b:
        b = b[:b.find('(')]
    b = float(b)
    
    if '(' in c:
       c = c[:c.find('(')]
    c = float(c)
    
    if not ambient_flag:
        if '(' in ambient_temp:
            ambient_temp = ambient_temp[:ambient_temp.find('(')]
    ambient_temp = float(ambient_temp)
    
    # Get the fractional-to-cartesian transformation matrix
    t_mat = fract_to_cartesian(a, b, c, alpha, beta, gamma, True)

    # Check to make sure that all the coordinates are actually present. Use these indices to get
    # the corresponding values.
    fract_x_i = find_index(loop_header, '_atom_site_fract_x')
    fract_y_i = find_index(loop_header, '_atom_site_fract_y')
    fract_z_i = find_index(loop_header, '_atom_site_fract_z')

    # Check that the fractional coordinates were properly parsed
    if len(fract_x_i) == 0:
        print('ERROR: Unable to find fractional x coordinate. Aborting....')
        exit()

    if len(fract_y_i) == 0:
        print('ERROR: Unable to find fractional y coordinate. Aborting....')
        exit()

    if len(fract_z_i) == 0:
        print('ERROR: Unable to find fractional z coordinate. Aborting....')
        exit()
        
    # Get the atom label and symbol
    atom_label = find_index(loop_header, '_atom_site_label') 
    atom_symbol = find_index(loop_header, '_atom_site_type_symbol') 

    if len(atom_label) == 0:
        print('ALERT: Atom label is empty.')
    
    if len(atom_symbol) == 0:
        print('ALERT: Missing atomic symbol. Will use atom label instead.')
    
    if len(atom_label) and len(atom_symbol) == 0:
        print('ERROR: Missing atomic information. Both _atom_site_label and _atom_site_type_symbol are empty. Aborting....')
        exit()

    # Get the atomic coordinates.
    atoms = [] # stores the atom label and atom symbol for each atom [[atom_label, atom_symbol]]
    fract_coords = np.zeros([len(loop_data[fract_x_i[0][0]]),3])    # Initialize numpy array to hold coordinates
    
    # Loop over all the rows holding the coordinates
    for i in range(len(loop_data[fract_x_i[0][0]])):

        # Split the row into its constituent parts
        row = loop_data[fract_x_i[0][0]][i].split()
        
        # If either the atom label or symbol are missing, only append the other. Otherwise, append both
        # No guarantee that the atom symbol is required/included in the .cif file, but the atom label is required
        # (where the atom label is the element letter and its ID, e.g. C1, C2, ..., Cn and H1, H2, ..., Hm)
        if len(atom_label) == 0:
            atoms.append([row[atom_symbol[0][1]]])
        elif len(atom_symbol) == 0:
            atoms.append([row[atom_label[0][1]]])
        else:
            atoms.append([row[atom_label[0][1]], row[atom_symbol[0][1]]])
        
        # For each coordinate, remove any standard deviation value and store as a float
        if '(' in row[fract_x_i[0][1]]:
            row[fract_x_i[0][1]] = row[fract_x_i[0][1]][:row[fract_x_i[0][1]].find('(')]
        frac_x = float(row[fract_x_i[0][1]])
                    
        if '(' in row[fract_y_i[0][1]]:
            row[fract_y_i[0][1]] = row[fract_y_i[0][1]][:row[fract_y_i[0][1]].find('(')]
        frac_y = float(row[fract_y_i[0][1]])
                    
        if '(' in row[fract_z_i[0][1]]:
            row[fract_z_i[0][1]] = row[fract_z_i[0][1]][:row[fract_z_i[0][1]].find('(')]
        frac_z = float(row[fract_z_i[0][1]])
        
        # Insert into numpy array
        fract_coords[i] = frac_x,frac_y,frac_z

    #fract_coords = np.array([1 + fract_coords[:, j] if sum(fract_coords[:, j]) < 0 else fract_coords[:, j] for j in range(3)]).T  #this seems to replicate mercury's output
    minimal_unit_cell_cartesian = fract_coords.dot(t_mat)

    # Parse element labels
    if len(atom_symbol) != 0:
        elements_list = []
        for item in atoms:
            elements_list.append(item[1])

    # Generate the adjacency matrix
    adj_mat = Table_generator(elements_list, minimal_unit_cell_cartesian)

    # Find all individual molecules in the geometry
    minimum_unit_cell_molecules_list = find_molecules(adj_mat)    
    
    # Collect all of an individual molecule's atoms and append them as a whole to a list
    # min_unit_cell_molecules_fract = [[molecule1], [molecule2]]
    # where [moleculen] = [numpy array holding fractional a, b, c coordinates]
    min_unit_cell_molecules_fract = []
    for subset in minimum_unit_cell_molecules_list:
        min_fract_coords = np.zeros([len(subset),3])    # Initialize numpy array to hold coordinates
        
        for i in range(len(subset)):
            min_fract_coords[i] = fract_coords[subset[i]]
        
        min_unit_cell_molecules_fract.append(min_fract_coords)
    
    # Now separate the list of elements into the individual molecules
    elements_list_blocked = []          
    for subset in minimum_unit_cell_molecules_list:
        elements_list_blocked_holder = []
        
        for i in range(len(subset)):
            elements_list_blocked_holder.append(elements_list[subset[i]])
            
        elements_list_blocked.append(elements_list_blocked_holder)
    
    # For reference:
    # entire row: minimal_unit_cell_cartesian[minimum_unit_cell_molecules_list[0][0]]
    # x: minimal_unit_cell_cartesian[minimum_unit_cell_molecules_list[0][0],0]
    # y: minimal_unit_cell_cartesian[minimum_unit_cell_molecules_list[0][0],1]
    # z: minimal_unit_cell_cartesian[minimum_unit_cell_molecules_list[0][0],2]
    

    # Evaluate the symmetry operations, make them into something useable
    symm_operations = [] # stores the symmetry operations to construct the unit cell
    symm_i = find_index(loop_header, '_symmetry_equiv_pos_as_xyz')  # Find index

    # Loop over the rows containing the symmetry operations and separate them into their components.
    for line in loop_data[symm_i[0][0]]:
        # There are several different formats we have to handle:
        # 1:  #, x,y,z
        # 2: 'x, y, z'
        # 3: x,y,z
        # 4: # 'x, y, z'
        # First remove any single quotation marks:
        line = line.replace("'", "")
        line = line.split()                     # split the line by spaces to separate molecule # and translation
        if len(line) == 2:
            line2 = line[1].split(',')              # split the second half by ',' to get the individual axis components
        elif len(line) == 3:
            for i in range(len(line)):
                line[i] = line[i].replace(',', '')
            line2 = line                    
        else:
            if line[0].isdigit(): # Have line now like ['1', '-x,', '-y+1/2,', 'z+1/2']
                line = line[1:]
                for i in range(3):
                    line[i] = line[i].replace(',', '')
                line2 = line
            else:
                line2 = line[0].split(',')              # split the line by ',' to get the individual axis components
        symm_operations.append([line2[0], line2[1], line2[2]])  # now make all of this row to the master array
    
    '''    
    # Convert the molecular formula into something useable
    molecular_formula = []      # holds the atoms and their expected number from the molecular formula. [str(element), int(count)] (e.g., ['C', 8])
    formula = cif_data['_chemical_formula_moiety'].split()
    s = ''
    n = ''
    for element in formula:
        if "'" in element:
            element = element.strip("'")
        
        for l in element:
            if l.isdigit():
                n += l
            elif l.isalpha():
                s += l
        
        molecular_formula.append([str(s), int(n)])
        s = ''
        n = ''
    '''

            
    unit_cell_coords_fract = []     # each list inside of unit_cell_coords_fract stores a set of coordinates for a molecule inside the unit cell
    unit_cell_elements = []         # stores the corresponding list of elements for each molecule placed inside the unit cell
    exclusion_list = []             # indices of molecules to exclude from final printing of unit cell. These may be duplicates or lie outside the unit cell dimensions.
    
    # Perform the symmetry operations and start placing molecules in the unit cell
    # Each count in the pos_as_xyz is a molecule inside the unit cell. We have already added the 
    # original molecule above, so now loop over the remaining molecules and add them to  
    # unit_cell_coords according to their symmetry operation
    for l in range(len(min_unit_cell_molecules_fract)):
        
        for i in range(len(symm_operations)):

            # Make a copy of the original molecule's fractional coordinates to modify
            current_coords = deepcopy(min_unit_cell_molecules_fract[l])
            unit_cell_elements.append(elements_list_blocked[l])
        
            # Now operate on its coordinates. Loop over the pos_as_xyz to get the operations
            # we need to perform on each axis. If + or - does appear in that axis's operations,
            # then it's the identity case, so we will not operate upon it.
            for j in range(0, 3): # x, y, z
                symm_operations[i][j] = symm_operations[i][j].replace('-1/2','+1/2') # standardizes input, necessary based on how code is structured
                if '-' in symm_operations[i][j]:
                    # Subtractive case. Check the constant offset and put it in a variable to operate on mathematically
                    #coeff = symm_operations[i][j][:symm_operations[i][j].find('-')]

                    # if len(coeff) == 0:
                    #     offset = 1
                    # else:
                    #     if coeff in ['1/2', '-x', '-y', '-z']:
                    #         offset = 0.5
                    if '1/2' in symm_operations[i][j]:
                        offset = 0.5
                    elif '1/4' in symm_operations[i][j]:
                        offset = 0.25
                    elif '3/4' in symm_operations[i][j]:
                        offset = 0.75
                    else:
                        offset = 1

                    # Loop over all the selected coordinate in each atom
                    for k in range(len(current_coords)):
                        current_coords[k, j] = offset - current_coords[k, j]

                elif '+' in symm_operations[i][j]:
                    # Additive case. Check the constant offset and put it in a variable to operate on mathematically
                
                    #coeff = symm_operations[i][j][:symm_operations[i][j].find('+')]
                
                    if '1/2' in symm_operations[i][j]:
                        offset = 0.5
                    elif '1/4' in symm_operations[i][j]:
                        offset = 0.25
                    elif '3/4' in symm_operations[i][j]:
                        offset = 0.75
                    else:
                        offset = 1

                    # Loop over all the selected coordinate in each atom
                    for k in range(len(current_coords)):
                        current_coords[k, j] = offset + current_coords[k, j]
                
            # if newly generated molecule's centroid is outside the unit cell, translate it in
            centroid = current_coords.mean(axis=0)
            #centroid = np.array([centroid])  # put it into a 1x3 array to properly insert (contenante) with the existing array holding the coordinates
            #print str(i) + '|' + str(centroid)
            for j in range(0, 3):
                # if centroid[j] > 1.05:
                #     current_coords[:,j] -= 1
                # elif centroid[j] < -0.05:
                #     current_coords[:,j] += 1
                if centroid[j] >= 1.00:
                    current_coords[:,j] -= 1
                elif centroid[j] < 0.00:
                    current_coords[:,j] += 1

            unit_cell_coords_fract.append(deepcopy(current_coords))
    
    # The ultimate goal is to reproduce the output generated by Mercury
    # The target is the unit cell for the option that the molecular centroids must fit inside the unit cell (unit cell is the packing option with a = b = c = 1)
    # For internal consistency, a,b,c refers to the fractional unit cell coordinates, while x,y,z refers to the cartesian coordinates
    
    # To match Mercury, we have to do the following steps (based on comparing outputs at various steps to see the operations performed)
    # 1. Check if any of the symmetry operations produced molecules with identical centroid coordinates. For any duplicate, translate by (1-b) and (0.5+c)
    # 2. Check for duplicate molecular coordinates and add any such molecules to an exclusion list (we won't print out any molecules on this list)
    # 3. Add to the exclusion list any molecule whose centroid coordinates lie outisde the unit cell (currently just x,y,z > 1.2, with 1.2 for a little fudge room)
    # 4. If an entire column is negative OR greater than 1, then add 1 to that each element in that column or subtraact 1 from each element in that column.
    #  --> NOTE: this is not currently implemented, due to issues of sorting out the individual molecules inside the unit cell geometries. As the geometries may contain multiple molecules,
    #           currently this script does not treat them individually, but rather as an entire group. Thus, there is no way to determine whether if an entire axis of a single molecule
    #           lies outside the unit cell. There is implemented a parser that will read the chemical formula and store that, in the event that eventually this functionality is needed.
    #           Since all this does is transform the molecular coordinates to be inside the unit cell, we choose to ignore this because it's not changing the quantity of molecules
    #           inside the unit cell, rather is just using a mirror image of the molecule that is inside the unit cell.
    # 4. Check to make sure there aren't any duplicate fractional coordinates, and add the molecules to the exclusion list if there are
    # 5. Transform from fractional to cartesian coordinates.
    # 6. Check to make sure that there aren't any duplicate coordinates.
    # 6a. Also check to make sure that there aren't any duplicate cartesian centroids, because the transform from fractional to cartesian may wrap some molecules back onto themselves in the unit cell
    
    # If any centroids are identical (have a distance of 0), then translate the duplicate by 1-b and 0.5+c
    for i in range(len(unit_cell_coords_fract)):
        i_centroid = unit_cell_coords_fract[i].mean(axis=0)

        for j in range(len(unit_cell_coords_fract)):
            if j > i :
                j_centroid = unit_cell_coords_fract[j].mean(axis=0)
                dist = np.around(np.linalg.norm(i_centroid-j_centroid), decimals=5)
                #dist = np.linalg.norm(i_centroid-j_centroid)
                if dist == 0:
                    for m in range(len(unit_cell_coords_fract[j])):
                        for k in range(1,3):
                            if k == 1:
                                # b coordinate. 1 -b
                                unit_cell_coords_fract[j][m, k] = 1 - unit_cell_coords_fract[j][m, k]
                            elif k == 2:
                                # c coordinate. 0.5+c
                                unit_cell_coords_fract[j][m, k] = 0.5 + unit_cell_coords_fract[j][m, k]

    # Move a molecule's centroids back inside the unit cell if they lie outside of the unit cell dimensions (0 < a, b, c, < 1)
    for i in range(len(unit_cell_coords_fract)):
        centroid = unit_cell_coords_fract[i].mean(axis=0)
        for j in range(len(centroid)):
            if centroid[j] < -0.4:
                # lies behind a unit cell dimension. Add 1 to the dimension.
                for k in range(len(unit_cell_coords_fract[i])):
                    unit_cell_coords_fract[i][k,j] = 1 + unit_cell_coords_fract[i][k,j]
            elif centroid[j] > 1.4:
                # lies ahead of a unit cell dimension. Subtract 1 from the dimension.
                for k in range(len(unit_cell_coords_fract[i])):
                    unit_cell_coords_fract[i][k,j] = unit_cell_coords_fract[i][k,j] - 1 
    
    # Check for any duplicate fractional coordinates and add these to the exclusion list
    exclusion_list = []     # Store any duplicates so we can subsequently ignore them
    
    '''
    # Add to the exclusion list any molecule that is a duplicate of another molecule (by centroid)
    print '\n:: checking for duplicate fractional centroids ::'
    for i in range(len(unit_cell_coords_fract)):
        if i not in exclusion_list:
            i_centroid = unit_cell_coords_fract[i].mean(axis=0)
            
            for j in range(len(unit_cell_coords_fract)):
                if j > i and j not in exclusion_list:
                    j_centroid = unit_cell_coords_fract[j].mean(axis=0)
                    dist = np.around(np.linalg.norm(i_centroid-j_centroid), decimals=5)
                    
                    if dist == 0:
                        exclusion_list.append(j)
    '''             
    
    '''

    # Add to the exclusion list any molecule whose centroid lies outside the unit cell (0.2 < a,b,c < 1.2)
    #print '\n\n:: removing any molecule whose centroids lie outside the unit cell ::'
    # Make sure that the centroid does not lie outside the unit cell
    for i in range(len(unit_cell_coords_fract)):
        centroid = unit_cell_coords_fract[i].mean(axis=0)
        centroid = np.array([ centroid ])  # put it into a 1x3 array to properly insert (contenante) with the existing array holding the coordinates
        print str(i) + '|' + str(centroid)
        for j in range(0,3):
            if centroid[0,j] > 1.2:
                #print '\nmolecule ' + str(i) + ' with axis ' + str(j) + ' is greater than 1.2 with ' + str(centroid[0,j])
                if i not in exclusion_list:
                    #print '    added molecule ' + str(i) + ' to exclusion list'
                    exclusion_list.append(i)
            
            elif centroid[0,j] < -0.2:
                if i not in exclusion_list:
                    exclusion_list.append(i)
    '''

    '''
    # Code to check if an entire column in a molecule is negative or greater than 1. Requires sorting each molecule individually first. 
    for i in range(len(unit_cell_coords_fract)):
        if i not in exclusion_list:
            total_rows = len(unit_cell_coords_fract[i]) # count the number of rows
            for j in range(0,3):
                negative_count = 0
                greater_than_one_count = 0
                
                for k in range(len(unit_cell_coords_fract[i])):
                    if unit_cell_coords_fract[i][k,j] < 0:
                        negative_count += 1
                    elif unit_cell_coords_fract[i][k,j] > 1:
                        greater_than_one_count += 1
                
                if negative_count == total_rows:
                    for k in range(len(unit_cell_coords_fract[i])):
                        unit_cell_coords_fract[i][k,j] = unit_cell_coords_fract[i][k,j] + 1
                        
                elif greater_than_one_count == total_rows:
                    for k in range(len(unit_cell_coords_fract[i])):
                        unit_cell_coords_fract[i][k,j] = unit_cell_coords_fract[i][k,j] - 1
                
    '''                
        
    # Transform from frational to cartesian coordinates and store these.
    cartesian_coords = []
    cartesian_elements = []
    for i in range(len(unit_cell_coords_fract)):
        if i not in exclusion_list:
            molecule = unit_cell_coords_fract[i].dot(t_mat)
            cartesian_coords.append(molecule)
            cartesian_elements.append(unit_cell_elements[i])
               
    cartesian_exclusion_list = []
    # Check for duplicate coordinates by comparing centroids
    for i in range(len(cartesian_coords)):
        if i not in cartesian_exclusion_list:
            i_centroid = cartesian_coords[i].mean(axis=0)
            
            for j in range(len(cartesian_coords)):
                if j > i and j not in cartesian_exclusion_list:
                    j_centroid = cartesian_coords[j].mean(axis=0)
                    dist = np.around(np.linalg.norm(i_centroid-j_centroid), decimals=5)
                    
                    if dist == 0:
                        cartesian_exclusion_list.append(j)
    
    initialize_array = 1
    master_element_list = []            # this holds all the elements that have passed all the conditions above to be placed inside the unit cell
    for i in range(len(cartesian_coords)):
        if i not in cartesian_exclusion_list:
            if initialize_array == 1:
                master_cartesian_coords = np.array(cartesian_coords[i])
                initialize_array = 0
            else:
                master_cartesian_coords = np.concatenate([master_cartesian_coords, cartesian_coords[i]])
            
            master_element_list += cartesian_elements[i]
    
    #adj_mat = Table_generator(master_element_list, master_cartesian_coords)
    # Adjusted full unit cell adj list because sometimes failed. This still relies on the first molecules being correct.
    #temp_adj_mat = scipy.linalg.block_diag(adj_mat, adj_mat)
    temp_adj_mat = adj_mat
    num_mols = int(len(master_cartesian_coords)/float(len(adj_mat))) # possibly len(master_element_list) ?
    if num_mols > 2:
        adj_mat_holder = [adj_mat]*(num_mols)
        temp_adj_mat = scipy.linalg.block_diag(*adj_mat_holder)
    big_adj_mat = temp_adj_mat

    # Find all individual molecules inside the unit cell                                                                                                                                                                                                                   
    unit_cell_molecules_list = find_molecules(big_adj_mat)
    mol_data = {}
    for count_i,i in enumerate(unit_cell_molecules_list):
        
        mol_data[count_i] = { "Elements":[ master_element_list[j] for j in i ],
                              "Geometry":master_cartesian_coords[i,:],
                              "Frac_geo":unit_cell_coords_fract[count_i],
                              "Adj_mat":big_adj_mat[:,i][i,:] }

        mol_data[count_i]["Atom_types"] = id_types(mol_data[count_i]["Elements"],mol_data[count_i]["Adj_mat"],gens=gens)
                 
    return mol_data,{"a" : a, "b" : b, "c" : c, "alpha" : alpha, "beta" : beta, "gamma": gamma },{"ambient_temp" : ambient_temp}

def get_table_counts(FF_file):
    table_counts = {'pair': '', 'bond': '', 'angle': '', 'dihedral': ''}
    table_locations = {'pair': '', 'bond': '', 'angle': '', 'dihedral': ''}

    with open(FF_file) as f:
        for count, line in enumerate(f):
            fields = line.split()

            # Skip empty lines
            if len(fields) == 0:
                continue

            if fields[0].lower() == "bond" and table_locations['bond'] == '':
                table_locations['bond'] = fields[4]
                continue
            if fields[0].lower() == "angle" and table_locations['angle'] == '':
                table_locations['angle'] = fields[5]
                continue
            if fields[0].lower() == "dihedral" and table_locations['dihedral'] == '':
                table_locations['dihedral'] = fields[6]
                continue
            if fields[0].lower() == "vdw" and table_locations['pair'] == '':
                table_locations['pair'] = fields[4]
                continue
    f.close()

    for interaction in table_locations.keys():
        if table_locations[interaction] == '':
            continue
        with open(table_locations[interaction]) as file:
            for count, line in enumerate(file):
                if count == 1:
                    table_counts[interaction] = line.split()[1]
                    break
        file.close()


    return table_counts

# Reads data file for cg
def read_data_file(data_file, unit_cell, N_atoms_uc):
    mass_flag = False
    atom_flag = False
    bond_flag = False
    angle_flag = False
    dihedral_flag = False
    # charge_flag = False

    masses = []
    atoms = []
    bonds = []
    angles = []
    dihedrals = []

    with open(data_file, 'r') as f:
        for lines in f:
            fields = lines.split()

            if len(fields) == 0: # Blank line
                continue
            elif len(fields) == 1:
                if fields[0] == "Masses":
                    mass_flag = True
                    continue
                elif fields[0] == "Atoms":
                    mass_flag = False
                    atom_flag = True
                    continue
                elif fields[0] == "Bonds":
                    mass_flag = False
                    atom_flag = False
                    bond_flag = True
                    continue
                elif fields[0] == "Angles":
                    mass_flag = False
                    atom_flag = False
                    bond_flag = False
                    angle_flag = True
                    continue
                elif fields[0] == "Dihedrals":
                    mass_flag = False
                    atom_flag = False
                    bond_flag = False
                    angle_flag = False
                    dihedral_flag = True
                    continue
                elif fields[0] == "Velocities":
                    continue
            # Header stuff:
            elif fields[1] == "atoms":
                N_atoms = int(fields[0])
                continue
            elif fields[1] == "atom" and fields[2] == "types":
                N_atom_types = int(fields[0])
                continue
            elif fields[1] == "bonds":
                N_bonds = int(fields[0])
                continue
            elif fields[1] == "bond" and fields[2] == "types":
                N_bond_types = int(fields[0])
                continue
            elif fields[1] == "angles":
                N_angles = int(fields[0])
                continue
            elif fields[1] == "angle" and fields[2] == "types":
                N_angle_types = int(fields[0])
                continue
            elif fields[1] == "dihedrals":
                N_dihedrals = int(fields[0])
                continue
            elif fields[1] == "dihedral" and fields[2] == "types":
                N_dihedral_types = int(fields[0])
                continue
            elif len(fields)==4 and fields[2] == "xlo" and fields[3] == "xhi":
                x_low = float(fields[0])
                x_hi = float(fields[1])
                continue
            elif len(fields)==4 and fields[2] == "ylo" and fields[3] == "yhi":
                y_low = float(fields[0])
                y_hi = float(fields[1])
                continue
            elif len(fields)==4 and fields[2] == "zlo" and fields[3] == "zhi":
                z_low = float(fields[0])
                z_hi = float(fields[1])
                continue
            elif len(fields)==6 and fields[3] == "xy" and fields[4] == "xz" and fields[5] == "yz":
                tilt_xy = float(fields[0])
                tilt_xz = float(fields[1])
                tilt_yz = float(fields[2])
            elif mass_flag:
                if fields[0] != "":  # skip blank lines
                    masses.append(float(fields[1]))
            elif atom_flag:
                if fields[0] != "":  # skip blank lines
                    if len(fields) == 10:
                        #       atomid          moleculeid      atomtype            charge          x_pos           y_pos               z_pos               x_vel           y_vel           z_vel
                        atoms.append(
                            [int(fields[0]), int(fields[1]), int(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7]), float(fields[8]), float(fields[9])])
                    elif len(fields) == 7:  # No velocity
                        atoms.append([int(fields[0]), int(fields[1]), int(fields[2]), float(fields[3]), float(fields[4]),
                             float(fields[5]), float(fields[6])])
            elif bond_flag:
                if fields[0] != "":
                    #               bondid          bond_type            atomid1         atomid2
                    bonds.append([int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])])
            elif angle_flag:
                if fields[0] != "":
                    #               angleid             angle_tpye       atom1           atom2           atom3
                    angles.append([int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4])])
            elif dihedral_flag:
                if fields[0] != "":
                    #                   dihed_id        dihed_type      atom1           atom2           atom3           atom4
                    dihedrals.append([int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5])])
            else:
                # Unrecognized line
                continue
    f.close()
    # Check if data file is single unit cell or more, and reduce down if necessary
    x_len = x_hi - x_low
    y_len = y_hi - y_low
    z_len = z_hi - z_low

    a_read = x_len
    b_read = (y_len**2 + tilt_xy**2) ** (0.5)
    c_read = (z_len**2 + tilt_xz**2 + tilt_yz**2) ** (0.5)

    x_ratio = a_read / unit_cell[0]
    if abs(x_ratio - int(x_ratio)) > 0.1:
        print("WARNING: x_ratio far from integer")
    y_ratio = b_read / unit_cell[1]
    if abs(y_ratio - int(y_ratio)) > 0.1:
        print("WARNING: y_ratio far from integer")
    z_ratio = c_read / unit_cell[2]
    if abs(z_ratio - int(z_ratio)) > 0.1:
        print("WARNING: x_ratio far from integer")

    x_ratio = int(x_ratio)
    y_ratio = int(y_ratio)
    z_ratio = int(z_ratio)

    tot_ratio = x_ratio * y_ratio * z_ratio

    tot_ratio = N_atoms/N_atoms_uc
    #N_atoms_uc = int(N_atoms / tot_ratio)
    atoms = sorted(atoms, key=operator.itemgetter(0))[:N_atoms_uc]
    if bonds != []:
        N_bonds_uc = int(N_bonds / tot_ratio)
        bonds = sorted(bonds, key=operator.itemgetter(2))[:N_bonds_uc]
    if angles != []:
        N_angles_uc = int(N_angles / tot_ratio)
        angles = sorted(angles, key=operator.itemgetter(3))[:N_angles_uc]
    if dihedrals != []:
        N_dihedrals_uc = int(N_dihedrals / tot_ratio)
        dihedrals = sorted(dihedrals, key=operator.itemgetter(4))[:N_dihedrals_uc]

    # Put together data
    Data = {}
    Data["atoms"] = atoms
    Data["bonds"] = bonds
    Data["angles"] = angles
    Data["dihedrals"] = dihedrals
    Data["masses"] = masses
    Data["moles"]=tot_ratio
    return Data


def cg_write_data(Data, unit_cell, Filename, dims, tether,cif, cell_sym={}):
    # Wrapper function for writing the lammps data file which is read by the .in.init file during run initialization

    Atom_types, Geometry, Bonds, Bond_types, Angles, Angle_types, Dihedrals, Dihedral_types, Charges, Molecule, Sim_Box, Data= cg_Generate_Box(Data, unit_cell, dims, cell_sym=cell_sym)


    # Write an xyz for easy viewing
    with open(Filename + '/' + Filename.split('/')[-1] + '.xyz', 'w') as f:
        f.write('{}\n\n'.format(len(Geometry)))
        for count_i, i in enumerate(Geometry):
            f.write('{:20s} {:< 20.6f} {:< 20.6f} {:< 20.6f}\n'.format("C", i[0], i[1], i[2]))

    # Create type dictionaries (needed to convert each atom,bond,angle, and dihedral type to consecutive numbers as per LAMMPS convention)
    # Note: LAMMPS orders atomtypes, bonds, angles, dihedrals, etc as integer types. Each of the following dictionaries holds the mapping between
    #       the true type (held in the various types lists) and the lammps type_id, obtained by enumerated iteration over the respective set(types).
    Atom_type_dict = {}
    for count_i, i in enumerate(sorted(set(Atom_types))):
        for j in Atom_types:
            if i == j:
                Atom_type_dict[i] = count_i + 1
            if i in Atom_type_dict.keys():
                break
    Bond_type_dict = {}
    for count_i, i in enumerate(sorted(set(Bond_types))):
        for j in Bond_types:
            if i == j:
                Bond_type_dict[i] = count_i + 1
            if i in Bond_type_dict.keys():
                break
    Angle_type_dict = {}
    for count_i, i in enumerate(sorted(set(Angle_types))):
        for j in Angle_types:
            if i == j:
                Angle_type_dict[i] = count_i + 1
            if i in Angle_type_dict.keys():
                break
    Dihedral_type_dict = {}
    for count_i, i in enumerate(sorted(set(Dihedral_types))):
        for j in Dihedral_types:
            if i == j:
                Dihedral_type_dict[i] = count_i + 1
            if i in Dihedral_type_dict.keys():
                break
    # Improper_type_dict = {}
    # for count_i, i in enumerate(sorted(set(Improper_types))):
    #     for j in Improper_types:
    #         if i == j:
    #             Improper_type_dict[i] = count_i + 1
    #         if i in Improper_type_dict.keys():
    #             break


    # Multiply out the unit cell to the dims requested

    # if tether:
    #     max_a_type = max(Atom_type_dict.values())


    # Write the data file
    with open(Filename + '/' + Filename.split('/')[-1] + '.data', 'w') as f:

        # Write system properties
        f.write("LAMMPS data file via gen_periodic_md.py, on {}\n\n".format(datetime.datetime.now()))

        # if tether:
        #     f.write("{} atoms\n".format(len(Data["atoms"]) * int(dims[0]) * int(dims[1]) * int(dims[2]) * 2))
        #     f.write("{} atom types\n".format(len(set(Atom_types))+1))
        # else:
        f.write("{} atoms\n".format(len(Data["atoms"])*int(dims[0])*int(dims[1])*int(dims[2])))
        f.write("{} atom types\n".format(len(set(Atom_types))))
        if len(Bonds) > 0:
            # if tether:
            #     f.write("{} bonds\n".format(len(Bonds) + len(Data["atoms"])*int(dims[0])*int(dims[1])*int(dims[2])))
            #     f.write("{} bond types\n".format(len(set(Bond_types))+1))
            # else:
            f.write("{} bonds\n".format(len(Bonds)))
            f.write("{} bond types\n".format(len(set(Bond_types))))
        if len(Angles) > 0:
            f.write("{} angles\n".format(len(Angles)))
            f.write("{} angle types\n".format(len(set(Angle_types))))
        if len(Dihedrals) > 0:
            f.write("{} dihedrals\n".format(len(Dihedrals)))
            f.write("{} dihedral types\n".format(len(set(Dihedral_types))))
        # if Improper_flag and len(Impropers) > 0:
        #     f.write("{} impropers\n".format(len(Impropers)))
        #     f.write("{} improper types\n".format(len(set(Improper_types))))
        f.write("\n")

        # Write box dimensions
        f.write("{:< 20.16f} {:< 20.16f} xlo xhi\n".format(Sim_Box[0], Sim_Box[1]))
        f.write("{:< 20.16f} {:< 20.16f} ylo yhi\n".format(Sim_Box[2], Sim_Box[3]))
        f.write("{:< 20.16f} {:< 20.16f} zlo zhi\n".format(Sim_Box[4], Sim_Box[5]))
        f.write("{:< 20.16f} {:< 20.16f} {:< 20.16f} xy xz yz\n\n".format(Sim_Box[6], Sim_Box[7], Sim_Box[8]))

        # Write Masses
        f.write("Masses\n\n")
        #for count_i, i in enumerate(sorted(set(Atom_types))):
        for count_j, j in enumerate(Data["masses"]):
                #if Atom_type_dict[j] == count_i + 1:
            f.write("{} {:< 8.6f}\n".format(count_j + 1, j))  # count_i+1 bc of LAMMPS 1-indexing
        # if tether:
        #     f.write("{} {:< 8.6f}\n".format(len(Data["masses"]) + 1, 1.00000))
        f.write("\n")

        # Write Atoms
        f.write("Atoms\n\n")
        #print(Molecule, len(Atom_types))
        print(len(Atom_types))
        print(len(Molecule))
        for count_i, i in enumerate(Atom_types):
            f.write("{:<8f} {:< 4f} {:< 4f} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:f} {:f} {:f}\n".format(count_i+1, Molecule[count_i], Atom_type_dict[i], Charges[count_i],
                            Geometry[count_i, 0], Geometry[count_i, 1], Geometry[count_i, 2], 0, 0, 0))
        # if tether:
        #     for count_i, i in enumerate(Atom_types):
        #         f.write("{:<8d} {:< 4d} {:< 4d} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:d} {:d} {:d}\n".format(len(Atom_types) + count_i + 1, max(Molecule) + count_i + 1, max_a_type+1, 0,
        #                     Geometry[count_i, 0] + 0.0001, Geometry[count_i, 1], Geometry[count_i, 2], 0, 0, 0))

        # Write Bonds
        if len(Bonds) > 0:
            f.write("\nBonds\n\n")
            for count_i, i in enumerate(Bonds):
                f.write(
                    "{:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i + 1, Bond_type_dict[Bond_types[count_i]],
                                                           i[0], i[1]))
            # if tether:
            #     for count_i, i in enumerate(Atom_types):
            #         f.write("{:<8d} {:<8d} {:<8d} {:<8d}\n".format(len(Bonds) + count_i + 1, len(set(Bond_types))+1,
            #                                                len(Atom_types)+count_i+1, count_i + 1))

        # Write Angles
        if len(Angles) > 0:
            f.write("\nAngles\n\n")
            for count_i, i in enumerate(Angles):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i + 1,
                                                                      Angle_type_dict[Angle_types[count_i]],
                                                                      i[0], i[1], i[2]))

        # Write Dihedrals
        if len(Dihedrals) > 0:
            f.write("\nDihedrals\n\n")
            for count_i, i in enumerate(Dihedrals):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i + 1,
                                                                             Dihedral_type_dict[
                                                                                 Dihedral_types[count_i]],
                                                                             i[0], i[1], i[2],
                                                                             i[3]))

    return


# Description: A wrapper for the commands to generate a cubic box and array of molecules for the lammps run
def cg_Generate_Box(Data, unit_cell, dims, cell_sym={}):
    N_atoms = len(Data["atoms"])
    N_atoms = (dims[0]) * (dims[1]) * (dims[2]) * N_atoms
    Geometry_sim = np.zeros([N_atoms, 3]) # Geometry_sim = zeros([N_atoms, 3])
    Molecule_sim = []
    Atom_types_sim = []
    Bonds_sim = []
    Bond_types_sim = []
    Angles_sim = []
    Angle_types_sim = []
    Dihedrals_sim = []
    Dihedral_types_sim = []
    Charges_sim = []
    centroids_list = []

    # # unpack data:
    # atom_types = [i[2] for i in Data["atoms"]]
    # charges = [i[3] for i in Data["atoms"]]
    N_mol = Data["atoms"][-1][0] # operating as number of atoms in molecule
    print(N_mol)

    # Initialize t_mat
    t_mat = fract_to_cartesian(unit_cell[0], unit_cell[1], unit_cell[2], unit_cell[3], unit_cell[4], unit_cell[5])
    inv_t_mat = cartesian_to_fract(unit_cell[0], unit_cell[1], unit_cell[2], unit_cell[3], unit_cell[4], unit_cell[5])
    uc_geo = [[i[4],i[5],i[6]] for i in Data["atoms"]]
    uc_frac_geo = np.matmul(uc_geo,inv_t_mat) # uc_frac_geo = matmul(uc_geo,inv_t_mat)
    print("uc_geo", uc_geo)
    print("uc_frac_geo", uc_frac_geo, type(uc_frac_geo))
    ### Addition for cell symmetry! v
    if cell_sym:
        cif_filename=cell_sym["cif_filename"][0]
        # Find all individual molecules in the geometry
        minimum_unit_cell_molecules_list = cell_sym["cg_moles"]
        map_file=cell_sym["map"]
        atomtypes,elements,masses,charges,adj_list=parse_map(map_file)    
        adj_mat = adj_list2mat(adj_list) # Find adjacency matrix with supplied file
        # Find all individual molecules in the geometry
        minimum_unit_cell_molecules_list = find_molecules(adj_mat)
        # Find fractional coords in cif file
        fract_coords=cif_coords(cif_filename)
        # Collect all of an individual molecule's atoms and append them as a whole to a list
        # min_unit_cell_molecules_fract = [[molecule1], [molecule2]]
        # where [moleculen] = [numpy array holding fractional a, b, c coordinates]
        min_unit_cell_molecules_fract = []
        for subset in minimum_unit_cell_molecules_list:
            min_fract_coords = np.zeros([len(subset),3])    # Initialize numpy array to hold coordinates
            
            for i in range(len(subset)):
                min_fract_coords[i] = fract_coords[subset[i]]
            
            min_unit_cell_molecules_fract.append(min_fract_coords)
        cif_data, loop_header, loop_data = read_cif(cif_filename)

            # Now separate the list of elements into the individual molecules
        elements_list_blocked = []          
        for subset in minimum_unit_cell_molecules_list:
            elements_list_blocked_holder = []
            
            for i in range(len(subset)):
                elements_list_blocked_holder.append(elements[subset[i]])
                
            elements_list_blocked.append(elements_list_blocked_holder)

            # Evaluate the symmetry operations, make them into something useable
        symm_operations = [] # stores the symmetry operations to construct the unit cell
        symm_i = find_index(loop_header, '_symmetry_equiv_pos_as_xyz')  # Find index

        # Loop over the rows containing the symmetry operations and separate them into their components.
        for line in loop_data[symm_i[0][0]]:
            # There are several different formats we have to handle:
            # 1:  #, x,y,z
            # 2: 'x, y, z'
            # 3: x,y,z
            # 4: # 'x, y, z'
            # First remove any single quotation marks:
            line = line.replace("'", "")
            line = line.split()                     # split the line by spaces to separate molecule # and translation
            if len(line) == 2:
                line2 = line[1].split(',')              # split the second half by ',' to get the individual axis components
            elif len(line) == 3:
                for i in range(len(line)):
                    line[i] = line[i].replace(',', '')
                line2 = line                    
            else:
                if line[0].isdigit(): # Have line now like ['1', '-x,', '-y+1/2,', 'z+1/2']
                    line = line[1:]
                    for i in range(3):
                        line[i] = line[i].replace(',', '')
                    line2 = line
                else:
                    line2 = line[0].split(',')              # split the line by ',' to get the individual axis components
            symm_operations.append([line2[0], line2[1], line2[2]])  # now make all of this row to the master array

                
        unit_cell_coords_fract = []     # each list inside of unit_cell_coords_fract stores a set of coordinates for a molecule inside the unit cell
        unit_cell_elements = []         # stores the corresponding list of elements for each molecule placed inside the unit cell
        exclusion_list = []             # indices of molecules to exclude from final printing of unit cell. These may be duplicates or lie outside the unit cell dimensions.
        
        # Perform the symmetry operations and start placing molecules in the unit cell
        # Each count in the pos_as_xyz is a molecule inside the unit cell. We have already added the 
        # original molecule above, so now loop over the remaining molecules and add them to  
        # unit_cell_coords according to their symmetry operation
        for l in range(len(min_unit_cell_molecules_fract)):
            
            for i in range(len(symm_operations)):

                # Make a copy of the original molecule's fractional coordinates to modify
                current_coords = deepcopy(min_unit_cell_molecules_fract[l])
                unit_cell_elements.append(elements_list_blocked[l])
            
                # Now operate on its coordinates. Loop over the pos_as_xyz to get the operations
                # we need to perform on each axis. If + or - does appear in that axis's operations,
                # then it's the identity case, so we will not operate upon it.
                for j in range(0, 3): # x, y, z
                    print("symm_ops 1", symm_operations[i][j])
                    symm_operations[i][j] = symm_operations[i][j].replace('-1/2','+1/2') # standardizes input, necessary based on how code is structured
                    print("symm_ops 2", symm_operations[i][j])
                    if '-' in symm_operations[i][j]:
                        # Subtractive case. Check the constant offset and put it in a variable to operate on mathematically
                        #coeff = symm_operations[i][j][:symm_operations[i][j].find('-')]

                        # if len(coeff) == 0:
                        #     offset = 1
                        # else:
                        #     if coeff in ['1/2', '-x', '-y', '-z']:
                        #         offset = 0.5
                        if '1/2' in symm_operations[i][j]:
                            offset = 0.5
                        elif '1/4' in symm_operations[i][j]:
                            offset = 0.25
                        elif '3/4' in symm_operations[i][j]:
                            offset = 0.75
                        else:
                            offset = 1
                        print("offset: ", offset)
                        # Loop over all the selected coordinate in each atom
                        for k in range(len(current_coords)):
                            current_coords[k, j] = offset - current_coords[k, j]

                    elif '+' in symm_operations[i][j]:
                        # Additive case. Check the constant offset and put it in a variable to operate on mathematically
                    
                        #coeff = symm_operations[i][j][:symm_operations[i][j].find('+')]
                    
                        if '1/2' in symm_operations[i][j]:
                            offset = 0.5
                        elif '1/4' in symm_operations[i][j]:
                            offset = 0.25
                        elif '3/4' in symm_operations[i][j]:
                            offset = 0.75
                        else:
                            offset = 1
                        print("offset: ", offset)
                        # Loop over all the selected coordinate in each atom
                        for k in range(len(current_coords)):
                            current_coords[k, j] = offset + current_coords[k, j]
                    
                # if newly generated molecule's centroid is outside the unit cell, translate it in
                centroid = current_coords.mean(axis=0)
                #centroid = np.array([centroid])  # put it into a 1x3 array to properly insert (contenante) with the existing array holding the coordinates
                #print str(i) + '|' + str(centroid)
                for j in range(0, 3):
                    # if centroid[j] > 1.05:
                    #     current_coords[:,j] -= 1
                    # elif centroid[j] < -0.05:
                    #     current_coords[:,j] += 1
                    if centroid[j] >= 1.00:
                        current_coords[:,j] -= 1
                    elif centroid[j] < 0.00:
                        current_coords[:,j] += 1

                unit_cell_coords_fract.append(deepcopy(current_coords))
        
        # The ultimate goal is to reproduce the output generated by Mercury
        # The target is the unit cell for the option that the molecular centroids must fit inside the unit cell (unit cell is the packing option with a = b = c = 1)
        # For internal consistency, a,b,c refers to the fractional unit cell coordinates, while x,y,z refers to the cartesian coordinates
        
        # To match Mercury, we have to do the following steps (based on comparing outputs at various steps to see the operations performed)
        # 1. Check if any of the symmetry operations produced molecules with identical centroid coordinates. For any duplicate, translate by (1-b) and (0.5+c)
        # 2. Check for duplicate molecular coordinates and add any such molecules to an exclusion list (we won't print out any molecules on this list)
        # 3. Add to the exclusion list any molecule whose centroid coordinates lie outisde the unit cell (currently just x,y,z > 1.2, with 1.2 for a little fudge room)
        # 4. If an entire column is negative OR greater than 1, then add 1 to that each element in that column or subtraact 1 from each element in that column.
        #  --> NOTE: this is not currently implemented, due to issues of sorting out the individual molecules inside the unit cell geometries. As the geometries may contain multiple molecules,
        #           currently this script does not treat them individually, but rather as an entire group. Thus, there is no way to determine whether if an entire axis of a single molecule
        #           lies outside the unit cell. There is implemented a parser that will read the chemical formula and store that, in the event that eventually this functionality is needed.
        #           Since all this does is transform the molecular coordinates to be inside the unit cell, we choose to ignore this because it's not changing the quantity of molecules
        #           inside the unit cell, rather is just using a mirror image of the molecule that is inside the unit cell.
        # 4. Check to make sure there aren't any duplicate fractional coordinates, and add the molecules to the exclusion list if there are
        # 5. Transform from fractional to cartesian coordinates.
        # 6. Check to make sure that there aren't any duplicate coordinates.
        # 6a. Also check to make sure that there aren't any duplicate cartesian centroids, because the transform from fractional to cartesian may wrap some molecules back onto themselves in the unit cell
        
        # If any centroids are identical (have a distance of 0), then translate the duplicate by 1-b and 0.5+c
        dup_list=[]
        for i in range(len(unit_cell_coords_fract)):
            i_centroid = unit_cell_coords_fract[i].mean(axis=0)

            for j in range(len(unit_cell_coords_fract)):
                if j > i :
                    j_centroid = unit_cell_coords_fract[j].mean(axis=0)
                    dist = np.around(np.linalg.norm(i_centroid-j_centroid), decimals=5)
                    #dist = np.linalg.norm(i_centroid-j_centroid)
                    #print("dist: ", dist)
                    if dist == 0:
                        print("duplicate found!")
                        dup_list.append(j)
                        # for m in range(len(unit_cell_coords_fract[j])):
                        #     for k in range(1,3):
                        #         if k == 1:
                        #             # b coordinate. 1 -b
                        #             unit_cell_coords_fract[j][m, k] = 1 - unit_cell_coords_fract[j][m, k]
                        #         elif k == 2:
                        #             # c coordinate. 0.5+c
                        #             unit_cell_coords_fract[j][m, k] = 0.5 + unit_cell_coords_fract[j][m, k]
        #print("uc pre dup:", unit_cell_coords_fract)
        if dup_list:
            #print("removing moles...")
            unit_cell_coords_fract = [x for b,x in enumerate(unit_cell_coords_fract) if b not in dup_list] 
        #print("uc post dup:", unit_cell_coords_fract)
        # Move a molecule's centroids back inside the unit cell if they lie outside of the unit cell dimensions (0 < a, b, c, < 1)
        for i in range(len(unit_cell_coords_fract)):
            centroid = unit_cell_coords_fract[i].mean(axis=0)
            for j in range(len(centroid)):
                if centroid[j] < -0.4:
                    # lies behind a unit cell dimension. Add 1 to the dimension.
                    for k in range(len(unit_cell_coords_fract[i])):
                        unit_cell_coords_fract[i][k,j] = 1 + unit_cell_coords_fract[i][k,j]
                elif centroid[j] > 1.4:
                    # lies ahead of a unit cell dimension. Subtract 1 from the dimension.
                    for k in range(len(unit_cell_coords_fract[i])):
                        unit_cell_coords_fract[i][k,j] = unit_cell_coords_fract[i][k,j] - 1 
        
        # Check for any duplicate fractional coordinates and add these to the exclusion list
        exclusion_list = []     # Store any duplicates so we can subsequently ignore them      
            
        # Transform from frational to cartesian coordinates and store these.
        cartesian_coords = []
        cartesian_elements = []
        for i in range(len(unit_cell_coords_fract)):
            if i not in exclusion_list:
                molecule = unit_cell_coords_fract[i].dot(t_mat)
                cartesian_coords.append(molecule)
                cartesian_elements.append(unit_cell_elements[i])
                
        cartesian_exclusion_list = []
        # Check for duplicate coordinates by comparing centroids
        for i in range(len(cartesian_coords)):
            if i not in cartesian_exclusion_list:
                i_centroid = cartesian_coords[i].mean(axis=0)
                
                for j in range(len(cartesian_coords)):
                    if j > i and j not in cartesian_exclusion_list:
                        j_centroid = cartesian_coords[j].mean(axis=0)
                        dist = np.around(np.linalg.norm(i_centroid-j_centroid), decimals=5)
                        
                        if dist == 0:
                            cartesian_exclusion_list.append(j)
        initialize_array = 1
        master_element_list = []            # this holds all the elements that have passed all the conditions above to be placed inside the unit cell
        for i in range(len(cartesian_coords)):
            if i not in cartesian_exclusion_list:
                if initialize_array == 1:
                    master_cartesian_coords = np.array(cartesian_coords[i])
                    initialize_array = 0
                else:
                    master_cartesian_coords = np.concatenate([master_cartesian_coords, cartesian_coords[i]])
                
                master_element_list += cartesian_elements[i]
        print("unit_cell_coords_fract: ", unit_cell_coords_fract)
        print("symm_operations: ", symm_operations)
        uc_frac_geo=np.matrix(np.concatenate((unit_cell_coords_fract)))
        uc_geo=master_cartesian_coords
        sym_moles=len(unit_cell_coords_fract)
        data_holder={}
        # Need to edit data to include information for expanded unit cell.
        for k in Data.keys():
            data_holder[k]=[]
            for n in range(0,sym_moles):
                if k=="atoms":
                    for i in (Data[k]):
                        atom_id=i[0]+N_mol*n
                        data_holder[k].append([atom_id, i[1]+n, i[2], i[3], uc_geo[atom_id-1,0], uc_geo[atom_id-1,1], uc_geo[atom_id-1,2]])
                elif k=="bonds":
                    for i in (Data[k]):
                        bond_id=i[0]+len(Data[k])*n
                        data_holder[k].append([bond_id, i[1], i[2]+N_mol*n, i[3]+N_mol*n])
                elif k=="angles":
                    for i in (Data[k]):
                        ang_id=i[0]+len(Data[k])*n
                        data_holder[k].append([ang_id, i[1], i[2]+N_mol*n, i[3]+N_mol*n, i[4]+N_mol*n])
                elif k=="dihedrals":
                    for i in (Data[k]):
                        dih_id=i[0]+len(Data[k])*n
                        data_holder[k].append([dih_id, i[1]+N_mol*n, i[2]+N_mol*n, i[3]+N_mol*n, i[4]+N_mol*n, i[5]+N_mol*n])
                elif k=="masses":
                    data_holder[k]=Data[k]
        N_atoms=N_atoms*sym_moles
        N_mol=N_mol*sym_moles
        Geometry_sim = np.zeros([N_atoms, 3]) # Geometry_sim = zeros([N_atoms, 3])
        Data=data_holder
    ### Addition for cell symmetry ^

    # Place the translations
    atoms_placed = 0
    atoms_not_placed = 0
    mols_placed = 0

    # for i in uc_frac_geo: # ensure all are inside same box for initial unit cell:
    #     for j in range(3):
    #         while i[0,j] > 1.2:
    #             i[0,j] -= 1.0
    #         while i[0,j] < -0.2:
    #             i[0,j] += 1.0

    # Check bonds don't extend over the entire box
    for i in Data["bonds"]:
        for j in range(3):
            if uc_frac_geo[i[2]-1, j] - uc_frac_geo[i[3]-1, j] > 0.5:
                uc_frac_geo[i[3] - 1 ,j] += 1.0
            elif uc_frac_geo[i[2]-1, j] - uc_frac_geo[i[3]-1, j] < -0.5:
                uc_frac_geo[i[3] - 1, j] -= 1.0


    print("\nGenerating simulation cell...")
    print("\nTotal number of atoms in simulation: {}".format(N_atoms))
    for a in range(dims[0]):
        for b in range(dims[1]):
            for c in range(dims[2]):

                # Do whole unit cell at once

                # Check its centroid against a list of other centroids that have been placed
                place = True
                geom = uc_frac_geo + np.array([a, b, c]) #geom = uc_frac_geo + array([a, b, c])
                geom_centroid = geom.mean(axis=0)

                for j in range(len(centroids_list)):
                    distance = np.around(linalg.norm(geom_centroid - centroids_list[j]), decimals=5) #distance = around(linalg.norm(geom_centroid - centroids_list[j]), decimals=5)
                    # if distance <= 1.0:
                    #     print("distance: ", distance)
                    if distance == 0:
                        atoms_not_placed += len(Data["atoms"])
                        place = False
                        break

                if place:

                    # Add the appropriately transformed unit cell to the geometry
                    centroids_list.append(geom_centroid)
                    Geometry_sim[atoms_placed:atoms_placed+len(Data["atoms"])] = np.dot(geom, t_mat) #dot(geom, t_mat)

                    # Add duplicates
                    for d in [1 * (dims[0]), 0, -1 * (dims[0])]:
                        for e in [1 * (dims[1]), 0, -1 * (dims[1])]:
                            for f in [1 * (dims[2]), 0, -1 * (dims[2])]:
                                if d == 0 and e == 0 and f == 0:
                                    continue
                                else:
                                    centroids_list.append(geom_centroid + np.array([d, e, f])) #centroids_list.append(geom_centroid + array([d, e, f]))

                    # Extend various lists (total elements lists, atomtypes lists, etc)
                    # Note: the lammps input expects bonds,angles,dihedrals, etc to be defined in terms of atom
                    #       id so, the atom_index is employed to keep track of how many atoms have been placed.
                    # Adj_mat_sim[atoms_placed:(atoms_placed + len(Data[i]["Geometry"])),atoms_placed:(atoms_placed + len(Data[i]["Geometry"]))] = Data[i]["Adj_mat"]
                    for z in range(mols_placed, mols_placed+N_mol):
                        Molecule_sim.append(np.floor(z/N_mol)) #[mols_placed] * len(Data["atoms"])
                    # Molecule_files = Molecule_files + [i]
                    # Elements_sim = Elements_sim + Data[i]["Elements"]
                    Atom_types_sim.extend([i[2] for i in Data["atoms"]])
                    Bonds_sim.extend([(j[2] + atoms_placed, j[3] + atoms_placed) for j in Data["bonds"]])
                    Bond_types_sim = Bond_types_sim + [i[1] for i in Data["bonds"]]
                    Angles_sim = Angles_sim + [(j[2] + atoms_placed, j[3] + atoms_placed, j[4] + atoms_placed) for j in Data["angles"]]
                    Angle_types_sim = Angle_types_sim + [i[1] for i in Data["angles"]]
                    Dihedrals_sim = Dihedrals_sim + [(j[2] + atoms_placed, j[3] + atoms_placed, j[4] + atoms_placed, j[5] + atoms_placed) for j in Data["dihedrals"]]
                    Dihedral_types_sim = Dihedral_types_sim + [i[1] for i in Data["dihedrals"]]
                    # Impropers_sim = Impropers_sim + [
                    #     (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                    #     in Data[i]["Impropers"]]
                    # Improper_types_sim = Improper_types_sim + Data[i]["Improper_types"]
                    Charges_sim = Charges_sim + [i[3] for i in Data["atoms"]]
                    #Masses_sim = Masses_sim + [Data[i]["Masses"][j] for j in Data[i]["Atom_types"]]

                    # Increment atom_index based on the number of atoms in the current geometry
                    atoms_placed += len(Data["atoms"])
                    mols_placed += N_mol

    # Remove any empty rows if molecules were not placed
    if atoms_not_placed > 0:
        Geometry_sim = Geometry_sim[:-atoms_not_placed]

    uc = {"a": unit_cell[0], "b": unit_cell[1], "c": unit_cell[2], "alpha": unit_cell[3], "beta": unit_cell[4], "gamma":unit_cell[5]}

    # The relevant relationships are taken from the lammps doc page for "How To: Triclinic"
    lx = uc["a"] * (dims[0])
    xy = (uc["b"] * (dims[1])) * np.cos(uc["gamma"] * pi / 180.)
    xz = (uc["c"] * (dims[2])) * np.cos(uc["beta"] * pi / 180.)
    ly = ((uc["b"] * (dims[1])) ** (2.0) - xy ** (2.0)) ** (0.5)
    yz = ((uc["b"] * (dims[1])) * (uc["c"] * (dims[2])) * np.cos(uc["alpha"] * pi / 180.) - xy * xz) / ly
    lz = ((uc["c"] * (dims[2])) ** (2.0) - xz ** (2.0) - yz ** (2.0)) ** (0.5)

    # OLD
    xlo = ylo = zlo = 0.0
    xhi, yhi, zhi = Geometry_sim.max(axis=0)  # get max value in each column
    xlo_bound = xlo + min([0.0, xy, xz, xy + xz])
    xhi_bound = xhi + max([0.0, xy, xz, xy + xz])
    #xhi_bound = xlo_bound + lx
    ylo_bound = ylo + min([0.0, yz])
    yhi_bound = yhi + max([0.0, yz])
    #yhi_bound = ylo_bound + ly
    zlo_bound = zlo
    zhi_bound = zhi
    #zhi_bound = zlo_bound + lz

    with open("box_dims.txt", 'w') as f:
        f.write('xlo ylo zlo\n')
        f.write('{} {} {}\n\n'.format(xlo, ylo, zlo))
        f.write('xhi yhi zhi\n')
        f.write('{} {} {}\n\n'.format(xhi, yhi, zhi))

        f.write('lx: {}\n'.format(lx))
        f.write('xy: {}\n'.format(xy))
        f.write('xz: {}\n'.format(xz))
        f.write('ly: {}\n'.format(ly))
        f.write('yz: {}\n'.format(yz))
        f.write('lz: {}\n\n'.format(lz))

        f.write('xlo_bound: {}\n'.format(xlo_bound))
        f.write('xhi_bound: {}\n'.format(xhi_bound))
        f.write('ylo_bound: {}\n'.format(ylo_bound))
        f.write('yhi_bound: {}\n'.format(yhi_bound))
        f.write('zlo_bound: {}\n'.format(zlo_bound))
        f.write('zhi_bound: {}\n'.format(zhi_bound))

    # Find the smallest (currently set to 0,0,0) and largest values in each dimension, then add some padding depending on the above bounding cals.
    #Sim_Box = array([xlo_bound, xhi_bound, ylo_bound, yhi_bound, zlo_bound, zhi_bound, xy, xz, yz])
    Sim_Box = np.array([0, lx, 0.0, ly, 0.0, lz, xy, xz, yz]) #Sim_Box = array([0, lx, 0.0, ly, 0.0, lz, xy, xz, yz])

    return Atom_types_sim, Geometry_sim, Bonds_sim, Bond_types_sim, Angles_sim, Angle_types_sim, Dihedrals_sim, Dihedral_types_sim, Charges_sim, Molecule_sim, Sim_Box, Data


def cif_coords(cif_filename):
        # Read cif file
    cif_data, loop_header, loop_data = read_cif(cif_filename)

    # Check integrety of loop headers and data (for each loop block, the the number of header columns
    # should match the number of columns in the data array
    if len(loop_header) != len(loop_data):
        print('ERROR: Number of loop header blocks does not match the number of loop data blocks. Aborting...')
        exit()
        
    # Check for error in parsing cif file
    for i in range(len(loop_header)):
        for j in range(len(loop_data[i])):
            if len(loop_header[i]) != len(loop_data[i][j].split()):
                if "'" not in loop_data[i][j]:
                    print('ERROR: Number of header and data columns does not match. Aborting....')
                    print('Header: ' + str(loop_header[i]))
                    print('Data: ' + str(loop_data[i][j]))
                    exit()

    # Get the unit cell dimensions (angles alpha, beta, gamma and lengths a, b, c)
    # Check to make sure they aren't zero. Remove standard deviations, if present.
    if '_cell_angle_alpha' not in cif_data:
        print('ERROR: Alpha value is empty or not found in cif. Aborting...')
        exit()
    else:
        alpha = cif_data['_cell_angle_alpha']
        
    if '_cell_angle_beta' not in cif_data:
        print('ERROR: Beta value is empty or not found in cif. Aborting...')
        exit()
    else:
        beta = cif_data['_cell_angle_beta']
    
    if '_cell_angle_gamma' not in cif_data:
        print('ERROR: Gamma value is empty or not found in cif. Aborting...')
        exit()
    else:
        gamma = cif_data['_cell_angle_gamma']

    if '_cell_length_a' not in cif_data:
        print('ERROR: Cell length a value is empty or not found in cif. Aborting...')
        exit()
    else:
        a = cif_data['_cell_length_a']
        
    if '_cell_length_b' not in cif_data:
        print('ERROR: Cell length b value is empty or not found in cif. Aborting...')
        exit()
    else:
        b = cif_data['_cell_length_b']
        
    if '_cell_length_c' not in cif_data:
        print('ERROR: Cell length c value is empty or not found in cif. Aborting...')
        exit()
    else:
        c = cif_data['_cell_length_c']
    
    ambient_flag = False
    if '_diffrn_ambient_temperature' not in cif_data:
        print('Ambient temperature not specified in .cif file. Defaulting to room temperature...')
        ambient_temp = 300
        ambient_flag = True
    else:
        ambient_temp = cif_data['_diffrn_ambient_temperature']
    
    # Remove standard deviation, if present
    if '(' in alpha:
       alpha = alpha[:alpha.find('(')]
    alpha = float(alpha)
    
    if '(' in beta:
       beta = beta[:beta.find('(')]
    beta = float(beta)

    if '(' in gamma:
       gamma = gamma[:gamma.find('(')]
    gamma = float(gamma)
    
    if '(' in a:
        a = a[:a.find('(')]
    a = float(a)
    
    if '(' in b:
        b = b[:b.find('(')]
    b = float(b)
    
    if '(' in c:
       c = c[:c.find('(')]
    c = float(c)
    
    if not ambient_flag:
        if '(' in ambient_temp:
            ambient_temp = ambient_temp[:ambient_temp.find('(')]
    ambient_temp = float(ambient_temp)
    
    # Get the fractional-to-cartesian transformation matrix
    t_mat = fract_to_cartesian(a, b, c, alpha, beta, gamma, True)

    # Check to make sure that all the coordinates are actually present. Use these indices to get
    # the corresponding values.
    fract_x_i = find_index(loop_header, '_atom_site_fract_x')
    fract_y_i = find_index(loop_header, '_atom_site_fract_y')
    fract_z_i = find_index(loop_header, '_atom_site_fract_z')

    # Check that the fractional coordinates were properly parsed
    if len(fract_x_i) == 0:
        print('ERROR: Unable to find fractional x coordinate. Aborting....')
        exit()

    if len(fract_y_i) == 0:
        print('ERROR: Unable to find fractional y coordinate. Aborting....')
        exit()

    if len(fract_z_i) == 0:
        print('ERROR: Unable to find fractional z coordinate. Aborting....')
        exit()
        
    # Get the atom label and symbol
    atom_label = find_index(loop_header, '_atom_site_label') 
    atom_symbol = find_index(loop_header, '_atom_site_type_symbol') 

    if len(atom_label) == 0:
        print('ALERT: Atom label is empty.')
    
    if len(atom_symbol) == 0:
        print('ALERT: Missing atomic symbol. Will use atom label instead.')
    
    if len(atom_label) and len(atom_symbol) == 0:
        print('ERROR: Missing atomic information. Both _atom_site_label and _atom_site_type_symbol are empty. Aborting....')
        exit()

    # Get the atomic coordinates.
    atoms = [] # stores the atom label and atom symbol for each atom [[atom_label, atom_symbol]]
    fract_coords = np.zeros([len(loop_data[fract_x_i[0][0]]),3])    # Initialize numpy array to hold coordinates
    
    # Loop over all the rows holding the coordinates
    for i in range(len(loop_data[fract_x_i[0][0]])):

        # Split the row into its constituent parts
        row = loop_data[fract_x_i[0][0]][i].split()
        
        # If either the atom label or symbol are missing, only append the other. Otherwise, append both
        # No guarantee that the atom symbol is required/included in the .cif file, but the atom label is required
        # (where the atom label is the element letter and its ID, e.g. C1, C2, ..., Cn and H1, H2, ..., Hm)
        if len(atom_label) == 0:
            atoms.append([row[atom_symbol[0][1]]])
        elif len(atom_symbol) == 0:
            atoms.append([row[atom_label[0][1]]])
        else:
            atoms.append([row[atom_label[0][1]], row[atom_symbol[0][1]]])
        
        # For each coordinate, remove any standard deviation value and store as a float
        if '(' in row[fract_x_i[0][1]]:
            row[fract_x_i[0][1]] = row[fract_x_i[0][1]][:row[fract_x_i[0][1]].find('(')]
        frac_x = float(row[fract_x_i[0][1]])
                    
        if '(' in row[fract_y_i[0][1]]:
            row[fract_y_i[0][1]] = row[fract_y_i[0][1]][:row[fract_y_i[0][1]].find('(')]
        frac_y = float(row[fract_y_i[0][1]])
                    
        if '(' in row[fract_z_i[0][1]]:
            row[fract_z_i[0][1]] = row[fract_z_i[0][1]][:row[fract_z_i[0][1]].find('(')]
        frac_z = float(row[fract_z_i[0][1]])
        
        # Insert into numpy array
        fract_coords[i] = frac_x,frac_y,frac_z
    return fract_coords

class Logger(object):
    def __init__(self,folder):
        self.terminal = sys.stdout
        self.log = open(folder+"/gen_periodic_md.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        pass
        
if __name__ == "__main__":
   main(sys.argv[1:])
