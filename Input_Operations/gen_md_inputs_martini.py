#!/bin/env python                                    
# Author: Aditi Khot (akhot@purdue.edu)

import argparse, os, sys, ast
from copy import deepcopy
from numpy import *
from itertools import permutations, combinations_with_replacement
import shutil
from collections import OrderedDict
import random
import subprocess as sp
# Add TAFFY Lib to path
from pathlib import Path
home=str(Path.home())
#sys.path.append('{}/bin/taffi/CG/'.format(home))
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
import numpy as np
from Mol_ops import find_bonds,find_angles,find_dihedrals,read_xyz, write_xyz




def main(argv):
    parser =argparse.ArgumentParser(description='''This script can be used. To partially generate data file and settings files for a mixed conducting system. The purpose of this script is basically to parse all modes and write data and settings file appropriately. It requires the map file and xyz file for the molecules!

    The input map files should contain additional columns compared to two TAFFI-style map file: "Martin_type","Shapex","Shapey","Shapez" for Martini atom types and shapes of all beads. If not using ellipsoids, the Shape columns are moot and set them to 1 and set ellipsoid_flag argument to 0. The element column is for user reference and can have names useful to the user. These names are important for parsing the bonded interactions  (see below.)
    
    Additionally the user has to supply gromacs itp file for Martini from which appropriate non-bonded interactions are parsed.

    The bonded interactions parsing is not sophisticated and can be modified in future. But essentially, the file format is 
    # Bonds
    E1 E2 type p1 p2 ...
    # Angles
    E1 E2 E3 type p1 p2 ...
    # Dihedrals
    E1 E2 E3 E4 type p1 p2 p3 p4...

    where Ei is the user defined element type and pi is the i^th parameter is force-field in the same format as lammps. If it doesn't find the force-field it will leave the entry empty for that mode, and the user can enter the values manually.''')

    # For polymer generation
    parser.add_argument('-maps', dest="maps", default='',  type=str, help='Space delimited string of input (modified) taffi-style map files for different molecule types')

    parser.add_argument('-xyzs', dest="xyzs", default='',  type=str, help='Space delimited string of input (modified) taffi-style xyz files for the respective molecule types')


    parser.add_argument('-Ns', dest="Ns", default='',  type=str, help='Space delimited string of corresponding number of each molecule type')

    parser.add_argument('-bFF', dest="bFF", default='', type=str, help='''
     1. If you want to use some other forcefields supply them in a text file with a similar format as described below:
    # Bonds
    E1 E2 type p1 p2 ...
    # Angles
    E1 E2 E3 type p1 p2 ...
    # Dihedrals
    E1 E2 E3 E4 type p1 p2 p3 p4...

    where Ei is the user defined element type and pi is the i^th parameter is force-field in the same format as lammps.
    2. If you dont know what you are doing just supply an empty string and the script will not fill any force-fields, but generate a settings file with all the modes and you can hard-code the FF in this file later''')

    parser.add_argument('-gb', dest="gb", default='', type=str, help='Supports only homogeneous GB interactions, i.e. self interaction between likewise particles can be GB but other interactions remain isotropic. Supply string containing: "e p1 p2...", where e is the ellipsoid element and p1 p2.. are the parameter of GB forcefield in the same order as expected by LAMMPS')


    parser.add_argument('-martiniFF', dest="martiniFF", default='/home/ddfortne/bin/mixed_cond/martini_v2.P_supp-material.itp',type=str, help='The martini paramaters gromacs style itp file')

    parser.add_argument('-ellipsoid_flag', dest="ellipsoid_flag", default=0,type=int, help='Set to 1 if the system contains ellipsoids')

    parser.add_argument('-o', dest="o", default='output', type=str, help='Output file name. Default: output')

    parser.add_argument('-python', dest="python", default='python', type=str, help='Directory of the python folder. If empty, will simply call "python", using the default python of the user.')
    
    parser.add_argument('-martini_v', dest="martini_v", default=2, type=int, help='Martini version to be used. Default: 2')
    
    args = parser.parse_args()

    print(("PROGRAM CALL: python gen_md_inputs.py {}\n".format(' '.join([ i for i in argv]))))            


    # Output folder
    o=args.o
    nmols=len(args.maps.split())
    if nmols==0:
        print("Supply at least one map file. Exiting...")
        quit()
    maps=args.maps.split()
    xyzs=args.xyzs.split()
    Ns=[int(_) for _ in args.Ns.split()]
    if len(xyzs)!=nmols:
        print('The number of xyz and map files do not match. Exiting...')
        quit()
    if len(Ns)!=nmols:
        print('The number of molecules and map files do not match. Exiting...')
        quit()

    # Read the map files
    AllPs=[]
    for i in range(nmols):
        if os.path.exists(maps[i]):
            print(('Reading map file {}...'.format(maps[i])))
            P= read_map(maps[i])
            P['A']=adj_list2mat(P['Adj_mat'])
            P['Density']=[P['Mass'][j]/((1/6.0)*pi*prod([P['Shapex'][j],P['Shapey'][j],P['Shapez'][j]])) for j in range(len(P['A']))]
        elif not os.path.exists(maps[i]):
            print(('Map file {} not found. Exiting...'.format(maps[i]))) 
            quit()
        # Read xyz file
        if os.path.exists(xyzs[i]):
            print(('Reading xyz file {}...'.format(xyzs[i])))
            _,P['Geometry'],_= read_xyz(xyzs[i])
        elif not os.path.exists(xyzs[i]):
            print(('xyz file {} not found. Exiting...'.format(xyzs[i])))
            quit()
        # No of molecules
        if Ns[i]:
            P['N_mol']=Ns[i]
        else:
            print('No of molecules should be non-zero. Exiting...')
            quit()
        AllPs.append(P)

    AllBox={}
    # Transform individual dictionararies for Pack_Box and settings
    for n, P in enumerate(AllPs):
        PBox={'Elements':P['Element'],\
          'Atom_types':P['Atom_type'],\
          'Martini_types':P['Martini_type'],\
          'Geometry':array(P['Geometry']),\
          'N_mol':P['N_mol'],\
          'Masses':{ i: P['Mass'][P['Atom_type'].index(i)] for i in set(P['Atom_type']) },\
          'Charges':{ i: P['Charge'][P['Atom_type'].index(i)] for i in set(P['Atom_type']) },\
          'Density':{ i: P['Density'][P['Atom_type'].index(i)] for i in set(P['Atom_type']) },\
          'Ellipsoid':{ i: [ P['Shapex'][P['Atom_type'].index(i)], P['Shapey'][P['Atom_type'].index(i)], P['Shapez'][P['Atom_type'].index(i)] ]+[1.0,0.0,0.0,0.0] for i in set(P['Atom_type'])  },\
          'Ellipsoid_flag': [1]*len(P['A'])}
        PBox.update(parse_modes(P['A'],P['Atom_type'],P['Element']))
        AllBox.update({n:PBox})
        #molno+=1

    Elements,Atom_types,Geometry,Bonds,Bond_types,Angles,Angle_types,Dihedrals,Dihedral_types,Impropers,Improper_types,Masses,Charges,Molecule,Molecule_names,Density,Ellipsoid_flag,Ellipsoids,Martini_types,Sim_Box,Atom_types2Ele,Atom_types2Martini,Bond_types2Ele,Angle_types2Ele,Dihedral_types2Ele =Pack_Box(AllBox)


    # Write data file
    print('\nWriting data file...')
    if args.ellipsoid_flag:
        write_data(Geometry, Atom_types, Sim_Box, Molecule, Masses, Charges, Bonds, Bond_types, Angles, Angle_types, Dihedrals, Dihedral_types, Density, Ellipsoid_flag, Ellipsoids, '{}.data'.format(o))
        print('\nWriting data file for ovito...')
        write_data(Geometry, Atom_types, Sim_Box, Molecule, Masses, Charges, Bonds, Bond_types, Angles, Angle_types, Dihedrals, Dihedral_types, Density, [], [], '{}_ovito.data'.format(o))
    else:
        write_data(Geometry, Atom_types, Sim_Box, Molecule, Masses, Charges, Bonds, Bond_types, Angles, Angle_types, Dihedrals, Dihedral_types, [],[],[], '{}.data'.format(o))


    

    # Write settings file
    print('\nWriting settings file...')
    write_settings(Atom_types2Ele,Atom_types2Martini,Bond_types2Ele,Angle_types2Ele,Dihedral_types2Ele,max(abs(array(Charges))), '{}.in.settings'.format(o),args.python,bFF=args.bFF,gb=args.gb,martiniFF=args.martiniFF,martini_v=args.martini_v)

    return


def parse_modes(A, Atom_types, Elements):
    O={}
    # Bonds is a list of sublists for each bond-type, each sublist is a list of all atom ids forming that bond-type
    # Bond_atypes is the correspoding bond-types described by the atom types forming that bond
    # For instance, Bond_atype= [(1,2), (1,3), (2,3)] and Bonds=[[ Atom ids of atoms of type 1 and 2 which are bonded],[... for type 1 and 3 ],[...]]
    # Bonds
    O['Bonds'],O['Bond_atypes'] =find_bonds(A,Atom_types)
    
    # Angles
    O['Angles'],O['Angle_atypes'] =find_angles(A,Atom_types)

    # Dihedrals
    O['Dihedrals'],O['Dihedral_atypes'] =find_dihedrals(A,Atom_types)
    
    # Impropers (for now, not used)
    O.update({'Impropers':[],'Improper_types':[]})
    
    # Transform them, so that they can be used in Pack_Box function
    O['Atom_types2Ele']={}
    for i,v in enumerate(Elements):
        if Atom_types[i] not in list(O['Atom_types2Ele'].keys()): O['Atom_types2Ele'][Atom_types[i]]=v
    

    for i in ['Bonds','Angles','Dihedrals']:
        O[i[:-1]+'_types']=[j+1 for j,k in enumerate(O[i]) for l in range(len(k))] # Assign a number to each bond type from 1 and make a list which matches the total number of bonds in the system
        O[i] = [tuple(k) for j in O[i] for k in j] # The corresponding list of tuples of atom-ids forming all the bonds in the system
        O[i[:-1]+'_types2Ele']={_+1: '-'.join([O['Atom_types2Ele'][k] for k in j]) for _,j in enumerate(O[i[:-1]+'_atypes'])} # Dictionary with keys as numeric for a bond-type and its objects as the associated atom-types involved and the element-types which make that bond


    return O

# Description: A wrapper for the commands to generate a cubic box and array of molecules for the lammps run
def Pack_Box(Data,Box_offset=0,Mol_offset=3.0,Step_size=0.0,Center_pi=0):


    # Center each molecule at the origin
    for i in list(Data.keys()):
        Data[i]["Geometry"] -= (mean(Data[i]["Geometry"][:,0]),mean(Data[i]["Geometry"][:,1]),mean(Data[i]["Geometry"][:,2]))

    # Define box circumscribing the molecule
    for i in list(Data.keys()):
        Data[i]["Mol_Box"] = \
        ( min(Data[i]["Geometry"][:,0]),max(Data[i]["Geometry"][:,0]),min(Data[i]["Geometry"][:,1]),max(Data[i]["Geometry"][:,1]),min(Data[i]["Geometry"][:,2]),max(Data[i]["Geometry"][:,2]) )


    # If a given step size is specified use that, or else determine the largest step_size 
    # Use the geometric norm of the circumscribing cube as the step size for tiling
    if not Step_size:
        for i in list(Data.keys()):
            Current_step = ( (Data[i]["Mol_Box"][1]-Data[i]["Mol_Box"][0])**2 +\
                             (Data[i]["Mol_Box"][3]-Data[i]["Mol_Box"][2])**2 +\
                            (Data[i]["Mol_Box"][5]-Data[i]["Mol_Box"][4])**2 )**(0.5)+Mol_offset  # +3 is just to be safe in the case of perfect alignment
            if Step_size < Current_step:
                Step_size = Current_step
            low,high= 0.0, 0.0
    else:
        Current_step = ( (Data[i]["Mol_Box"][1]-Data[i]["Mol_Box"][0])**2 +\
                         (Data[i]["Mol_Box"][3]-Data[i]["Mol_Box"][2])**2 +\
                         (Data[i]["Mol_Box"][5]-Data[i]["Mol_Box"][4])**2 )**(0.5)+Mol_offset  # +3 is just to be safe in the case of perfect alignment
        
        low = -Box_offset-Current_step
        high = Step_size + Box_offset + Current_step

            


    # Find the smallest N^3 cubic lattice that is greater than N_tot
    N_tot = sum([ Data[i]["N_mol"] for i in list(Data.keys())]) # Define the total number of molecules to be placed
    N_lat = 1
    while(N_lat**3 < N_tot):
        N_lat = N_lat + 1

    # Find molecular centers
    Centers = zeros([N_lat**3,3])
    count = 0
    for i in range(N_lat):
        for j in range(N_lat):
            for k in range(N_lat):
                Centers[i*N_lat**2 + j*N_lat + k] = array([Step_size*i,Step_size*j,Step_size*k])
                count = count + 1    

        
    # Select centers in the box where molecules are placed
    if Center_pi:     # If asked to arrange in already pi-stack position (along z)
        idx=[0,2]
    else: #Randomize the order of the centers so that molecules are randomly placed
        idx =list(range(Centers.shape[0]))
        random.shuffle(idx)

    Centers=Centers[idx[0:N_tot],:]

    if not high and not low:
         low = min([min(Centers[:,0])-Box_offset-Step_size,min(Centers[:,1])-Box_offset-Step_size,min(Centers[:,2])-Box_offset-Step_size])
         high = max([max(Centers[:,0])+Box_offset+Step_size,max(Centers[:,1])+Box_offset+Step_size,max(Centers[:,2])+Box_offset+Step_size])


    # Center the box at the origin
    disp = -1*mean([low,high])
    low += disp
    high += disp
    for count_i,i in enumerate(Centers):
        Centers[count_i] = i + disp

    # Define sim box
    Sim_Box = array([low,high,low,high,low,high])

    # Intialize lists for iterating over molecules and keeping track of how many have been placed
    keys = sorted(Data.keys())     # list of unique molecule keys
    placed_num = [0]*len(keys)     # list for keeping track of how many of each molecule have been placed
    atom_index = 0                 # an index for keeping track of how many atoms have been placed
    mol_index = 0                  # an index for keeping track of how many molecules have been placed

    # Initialize various lists to hold the elements, atomtype, molid labels etc.
    # Create a list of molecule ids for each atom        
    Geometry_sim       = zeros([sum([ Data[i]["N_mol"]*len(Data[i]["Geometry"]) for i in list(Data.keys()) ]),3])
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
    Density_sim        = []
    Ellipsoid_flag     = []
    Ellipsoid_sim      = []   
    Martini_types_sim  = []

    # Determine the new atom type, bond type, angle type, dihedral type corresponding the order in which the molecules are placed
    Atom_types_new={}
    for count_k,k in enumerate(keys):
        if count_k:
            Atom_types_new[k]={_: _ + max(Atom_types_new[k-1].values()) for _ in range(1,max(Data[k]["Atom_types"])+1)}
        else:
            Atom_types_new[k]={_: _ for _ in range(1,max(Data[k]["Atom_types"])+1)}

    Bond_types_new ={}
    maxtype=0
    for count_k,k in enumerate(keys):
        if len(Data[k]["Bond_types"]):
            Bond_types_new[k]={_: _ + maxtype for _ in range(1,max(Data[k]["Bond_types"])+1)}
            maxtype=max(Bond_types_new[k].values())  
        #elif len(Data[k]["Bond_types"]) and count_k==0:
        #    Bond_types_new[k]={_: _ for _ in range(1,max(Data[k]["Bond_types"])+1)}
         
    Angle_types_new={}
    maxtype=0 
    for count_k,k in enumerate(keys):
        if len(Data[k]["Angle_types"]):
            #and count_k:
            Angle_types_new[k]={_: _ + maxtype for _ in range(1,max(Data[k]["Angle_types"])+1)}
            maxtype=max(Angle_types_new[k].values())
        #elif len(Data[k]["Angle_types"]) and count_k==0:        
        #    Angle_types_new[k]={_: _ for _ in range(1,max(Data[k]["Angle_types"])+1)}

    Dihedral_types_new={}
    maxtype=0
    for count_k,k in enumerate(keys):
        if len(Data[k]["Dihedral_types"]):
            #and count_k:
            Dihedral_types_new[k]={_: _ + maxtype for _ in range(1,max(Data[k]["Dihedral_types"])+1)}
            maxtype=max(Dihedral_types_new[k].values())
        #elif len(Data[k]["Dihedral_types"]) and count_k==0:
        #    Dihedral_types_new[k]={_: _ for _ in range(1,max(Data[k]["Dihedral_types"])+1)}
        #    maxtype=max(Dihedral_types_new[k].values())
    # Obtain a new dictionary for the new atom types and their correspong elements and martini types
    Atom_types2Ele= {Atom_types_new[k][_]: Data[k]["Atom_types2Ele"][_] for k in keys for _ in list(Atom_types_new[k].keys()) }
    Atom_types2Martini= {Atom_types_new[k][_]: Data[k]["Martini_types"][Data[k]['Atom_types'].index(_)] for k in keys for _ in list(Atom_types_new[k].keys())}


    # Obtain a new dictionary for the new mode types and the corresponding element types which form that mode
    Bond_types2Ele  = {Bond_types_new[k][_]: Data[k]["Bond_types2Ele"][_] for k in keys for _ in set(Data[k]["Bond_types"])}
    Angle_types2Ele  = {Angle_types_new[k][_]: Data[k]["Angle_types2Ele"][_]  for k in keys for _ in set(Data[k]["Angle_types"])}
    Dihedral_types2Ele  = {Dihedral_types_new[k][_]: Data[k]["Dihedral_types2Ele"][_]  for k in keys for _ in set(Data[k]["Dihedral_types"])}


    # Place molecules in the simulation box and extend simulation lists 
    while (sum(placed_num) < N_tot):

        # Place the molecules in round-robin fashion to promote mixing
        # Each molecule key is iterated over, if all molecules of this type have been
        # placed then the molecule type is skipped
        for count_k,k in enumerate(keys):
            
            # If all the current molecules types have been placed continue
            # else: increment counter
            if placed_num[count_k] >= Data[k]["N_mol"]:
                continue
            else:
                placed_num[count_k]+=1
            #print(("placing atoms {}:{} with molecule {}".format(atom_index,atom_index+len(Data[k]["Geometry"]),k)))

            
            # Move the current molecule to its box, append to Geometry_sim, return to the molecule to the origin
            Data[k]["Geometry"] += Centers[mol_index]
            Geometry_sim[atom_index:(atom_index+len(Data[k]["Geometry"])),:] = Data[k]["Geometry"]
            Data[k]["Geometry"] -= Centers[mol_index]

            # Extend various lists (total elements lists, atomtypes lists, etc)
            # Note: the lammps input expects bonds,angles,dihedrals, etc to be defined in terms of atom
            #       id so, the atom_index is employed to keep track of how many atoms have been placed.
            Molecule_sim       = Molecule_sim + [mol_index]*len(Data[k]["Elements"])
            Molecule_files     = Molecule_files + [k]
            Elements_sim       = Elements_sim + Data[k]["Elements"]
            Atom_types_sim     = Atom_types_sim + [Atom_types_new[k][j] for j in  Data[k]["Atom_types"]]

            Bonds_sim          = Bonds_sim + [ (j[0]+atom_index,j[1]+atom_index) for j in Data[k]["Bonds"] ]
            
            Bond_types_sim     = Bond_types_sim + [Bond_types_new[k][j] for j in  Data[k]["Bond_types"]]

            Angles_sim         = Angles_sim + [ (j[0]+atom_index,j[1]+atom_index,j[2]+atom_index) for j in Data[k]["Angles"] ]

            Angle_types_sim    = Angle_types_sim + [Angle_types_new[k][j] for j in  Data[k]["Angle_types"]]

            Dihedrals_sim      = Dihedrals_sim + [ (j[0]+atom_index,j[1]+atom_index,j[2]+atom_index,j[3]+atom_index) for j in Data[k]["Dihedrals"] ]
            
            Dihedral_types_sim = Dihedral_types_sim + [Dihedral_types_new[k][j] for j in  Data[k]["Dihedral_types"]]

            Charges_sim        = Charges_sim + [ Data[k]["Charges"][j] for j in Data[k]["Atom_types"] ]
            Masses_sim         = Masses_sim + [ Data[k]["Masses"][j] for j in Data[k]["Atom_types"] ]
            Density_sim        = Density_sim + [ Data[k]["Density"][j] for j in Data[k]["Atom_types"] ]
            Ellipsoid_sim      = Ellipsoid_sim + [ Data[k]["Ellipsoid"][j] for j in Data[k]["Atom_types"] ]
            Ellipsoid_flag     = Ellipsoid_flag + Data[k]["Ellipsoid_flag"]

            # Additional features like Martini type
            Martini_types_sim   = Martini_types_sim + Data[k]["Martini_types"]

            # Increment atom_index based on the number of atoms in the current geometry
            atom_index += len(Data[k]["Geometry"])
            mol_index += 1

    return Elements_sim,Atom_types_sim,Geometry_sim,Bonds_sim,Bond_types_sim,Angles_sim,Angle_types_sim,Dihedrals_sim,Dihedral_types_sim,Impropers_sim,Improper_types_sim,Masses_sim,Charges_sim,Molecule_sim,Molecule_files,Density_sim,Ellipsoid_flag,Ellipsoid_sim,Martini_types_sim,Sim_Box,Atom_types2Ele,Atom_types2Martini,Bond_types2Ele,Angle_types2Ele,Dihedral_types2Ele

# Writes data file
def write_data(traj, atom_types, box,  molecule, mass, charge,  bonds, bond_types, angles, angle_types, dihedrals, dihedral_types, density, ellipsoid_flag, ellipsoids, output_name):

    nbond_types = max(bond_types) if len(bond_types) else 0
    nbond = len(bonds)
    nangle_types = max(angle_types) if len(angle_types) else 0
    nangle = len(angles)
    ndihedral_types = max(dihedral_types) if len(dihedral_types) else 0
    ndihedral = len(dihedrals) 
    natoms = traj.shape[0]
    natom_types = max(atom_types)
    nellipsoids= len([_ for _ in ellipsoid_flag if _])

    #Writing file
    data_f = open(output_name,'w')
    data_f.write('LAMMPS data file via polygen.toy.py, on [timestamp]\n\n')
    data_f.write(str(natoms) + ' atoms\n')
    data_f.write(str(natom_types)+ ' atom types\n')
    if nellipsoids: data_f.write(str(nellipsoids)+' ellipsoids\n')
    data_f.write(str(nbond) + ' bonds\n'+ str(nbond_types) +' bond types\n'+ str(nangle)+' angles\n'+ str(nangle_types)+' angle types\n'+ str(ndihedral)+' dihedrals\n'+str(ndihedral_types)+' dihedral types\n\n')
    if len(box)==2:
        data_f.write(str(box[0]) + ' ' + str(box[1]) + ' xlo xhi\n' + str(box[0]) + ' ' + str(box[1]) + ' ylo yhi\n' + str(box[0]) + ' ' + str(box[1]) + ' zlo zhi\n\n')
    elif len(box)==6:
        data_f.write(str(box[0]) + ' ' + str(box[1]) + ' xlo xhi\n' + str(box[2]) + ' ' + str(box[3]) + ' ylo yhi\n' + str(box[4]) + ' ' + str(box[5]) + ' zlo zhi\n\n')
    
    a_list = list(range(1,natom_types+1))
    #print(a_list)
    #print(atom_types)
    #print(mass)
    Mass_ave = mass_average(atom_types, a_list, mass)
    #print("mass ave", Mass_ave)
    data_f.write('Masses\n\n')
    for i in range(natom_types):
        # a = atom_types.index(i+1)
        # print("a", a)
        #print("gen_md mass", mass)
        #print("gen_md atom_types", atom_types)
        data_f.write(str(i+1) + ' ' +str(Mass_ave[i]) + '\n')


    data_f.write('\nAtoms\n\n')
    if len(ellipsoid_flag):
        line = '{:<5d} {:<5d} {:<20f} {:<20f} {:<20f} {:<5d} {:<5f} {:<5d} {:<5f}\n'
        #    atom number, atom type, geometry, molecule number, charge , ellipsoid_flag, density
        for i in range(natoms):
            data_f.write(line.format(i+1, atom_types[i] , traj[i,0], traj[i,1], traj[i,2], molecule[i], charge[i], ellipsoid_flag[i], density[i] ))
    else:
        line = '{:<5d} {:<5d} {:<5d} {:<10f} {:<20f} {:<20f} {:<20f}\n'
        #    atom number, molecule number, atom type, charge, geometry
        for i in range(natoms):
            data_f.write(line.format(i+1,  molecule[i], atom_types[i], charge[i], traj[i,0], traj[i,1], traj[i,2]))

    
    if len(ellipsoids): data_f.write('\nEllipsoids\n\n')
    for i,e in enumerate(ellipsoids):
        data_f.write('{:<5d} {:<5f} {:<5f} {:<5f} {:<5f} {:<5f} {:<5f} {:<5f}\n'.format(i+1,e[0],e[1],e[2],e[3],e[4],e[5],e[6]))
            
    if bonds:
        data_f.write('\nBonds\n\n')
        line = '{:<5d} {:<5d} {:<5d} {:<5d} \n'
        # bond number, bond type, atom1, atom2
        for i in range(len(bonds)):
            data_f.write(line.format(i+1,bond_types[i],bonds[i][0]+1,bonds[i][1]+1))

    if angles:
        data_f.write('\nAngles\n\n')
        line = '{:<5d} {:<5d} {:<5d} {:<5d} {:<5d}\n'
        # angle number, angle type, atom1, atom2, atom3
        for i in range(len(angles)):
            data_f.write(line.format(i+1,angle_types[i],angles[i][0]+1,angles[i][1]+1,angles[i][2]+1))

    if dihedrals:
        data_f.write('\nDihedrals\n\n')
        line = '{:<5d} {:<5d} {:<5d} {:<5d} {:<5d} {:<5d}\n'
        # dihedral number, dihedral type, atom1, atom2
        for i in range(len(dihedrals)):
            data_f.write(line.format(i+1,dihedral_types[i],dihedrals[i][0]+1,dihedrals[i][1]+1,dihedrals[i][2]+1,dihedrals[i][3]+1))

    data_f.close()
    return

def write_settings(atoms2ele,atoms2martini,b2ele,a2ele,d2ele,charge,filename,python,bFF='', gb='' ,martiniFF='/home/ddfortne/bin/mixed_cond/martini_v2.P_supp-material.itp', martini_v=2):
                                

    # Get all pair-types which have martini type defined
    FF={}
    pair_types=' '.join(['{},{}'.format(atoms2martini[i1],atoms2martini[i2]) for i1, i2 in combinations_with_replacement(list(range(1,len(list(atoms2ele.keys()))+1)),2) if atoms2martini[i1] not in ['N/A','WM','WP'] and atoms2martini[i2] not in ['N/A','WM','WP']])
    # if python: # If a specific python library is requested, use it. Otherwise just call python and use the default version.
    #     python+='/'
    # else:
    #     pass
    # Parse force-fields for these pair-types in kcal/mol
    if len(pair_types):
        if martini_v==2:
            sp.call('{} {}/gromacs2lammps.py {} -pair_types "{}"; wait'.format(python,os.path.dirname(sys.argv[0]),martiniFF,pair_types), shell=True)
        elif martini_v==3:
            sp.call('{} {}/gromacs2lammps_m3.py {} -pair_types "{}"; wait'.format(python,os.path.dirname(sys.argv[0]),martiniFF,pair_types), shell=True)
        if not os.path.exists("lammps_para.txt"): print('Error! The Martini force-fields were not generated.Exiting...'); quit()
        f=open("lammps_para.txt",'r')
        for line in f:
            if 'Error' in line:
                print(line)
                quit()
            line=line.split()
            FF[(line[0],line[1])]=[line[2],line[3]]
        f.close()
        
        
    # Read bonded force-fields from mixed conducting system if bonded FF file is specified
    b2ff, a2ff, d2ff ={} , {}, {}
    if len(bFF):
        f=open(bFF,'r')
        for line in f:
            if '# Bonds' in line:
                flag='b'
                continue
            if '# Angles' in line:
                flag='a'
                continue
            if '# Dihedrals' in line:
                flag='d'
                continue
            if flag =='b' and line!='\n':
                line=line.split()
                for i in list(b2ele.keys()):
                    eles=b2ele[i].split('-')
                    if tuple(eles)==tuple(line[0:2]) or tuple(eles[::-1])==tuple(line[0:2]):  # Check if all elements match
                        b2ff[i]="\t".join(line[2:])

            if flag =='a' and line!='\n':
                line=line.split()
                for i in list(a2ele.keys()):
                    eles=a2ele[i].split('-')
                    if tuple(eles)==tuple(line[0:3]) or tuple(eles[::-1])==tuple(line[0:3]):  # Check if all elements match
                        a2ff[i]="\t".join(line[3:])

            if flag =='d' and line!='\n':
                line=line.split()
                for i in list(d2ele.keys()):
                    eles=d2ele[i].split('-')
                    if tuple(eles)==tuple(line[0:4]) or tuple(eles[::-1])==tuple(line[0:4]):  # Check if all elements match
                        d2ff[i]="\t".join(line[4:])

        f.close()
        
        for i in list(b2ele.keys()):
            if i not in list(b2ff.keys()):
                print(("Error! Force-fields not found for {} mode in file {}. Exiting...".format(b2ele[i],bFF))); quit()
        for i in list(a2ele.keys()):
            if i not in list(a2ff.keys()):
                print(("Error! Force-fields not found for {} mode in file {}. Exiting...".format(a2ele[i],bFF))); quit()
        for i in list(d2ele.keys()):
            if i not in list(d2ff.keys()):
                print(("Error! Force-fields not found for {} mode in file {}. Exiting...".format(d2ele[i],bFF))); quit()

    else:
        b2ff={_:'' for _ in list(b2ele.keys())}
        a2ff={_:'' for _ in list(a2ele.keys())}
        d2ff={_:'' for _ in list(d2ele.keys())}

    # Element with gay berne potential
    if len(gb):
        gb= gb.split() 
        gb_atype=[_ for  _ in list(atoms2ele.keys()) if atoms2ele[_]==gb[0]][0]
        gb=" ".join(gb[1:])



    # Write the pair_coeff section
    f=open(filename,'w')
    f.write('# Non-bonded interactions (pair-wise)\n') 
    for i1, i2 in combinations_with_replacement(list(atoms2ele.keys()),2):
        if (atoms2martini[i1],atoms2martini[i2]) in list(FF.keys()):
            if len(gb) and gb_atype==i1 and gb_atype==i2: #gayberne potential
                f.write('pair_coeff\t{:<5d}\t{:<5d}\tgayberne\t{}\t\t#{}-{}\n'.format(i1,i2,gb,atoms2ele[i1],atoms2ele[i2]))
            else:
                f.write('pair_coeff\t{:<5d}\t{:<5d}\t{:<5s}\t\t{:<5s}\t{:<5s}\t\t#{}-{}\n'.format(i1,i2,'lj/gromacs/coul/gromacs',FF[(atoms2martini[i1],atoms2martini[i2])][0],FF[(atoms2martini[i1],atoms2martini[i2])][1],atoms2ele[i1],atoms2ele[i2]))
        elif atoms2martini[i1] in ['WM','WP'] or atoms2martini[i2] in ['WM','WP']:
            f.write('pair_coeff\t{:<5d}\t{:<5d}\t{:<5s}\t\t0.00\t0.00\t\t#{}-{}\n'.format(i1,i2,'lj/gromacs/coul/gromacs',atoms2ele[i1],atoms2ele[i2]))            
        else:
            f.write('pair_coeff\t{:<5d}\t{:<5d}\t\t\t\t\t\t#{}-{}\n'.format(i1,i2,atoms2ele[i1],atoms2ele[i2]))
        

    # Write the bonded potentials section
    if len(b2ele): 
        f.write('\n# Stretching interactions\n')
        for i in sorted(b2ele.keys()):
            f.write('bond_coeff\t{:<5d}\t{}\t#{}\n'.format(i,b2ff[i],b2ele[i]))
        
    # Write the angle potentials section
    if len(a2ele): 
        f.write('\n# Bending interactions\n')
        for i in sorted(a2ele.keys()):
            f.write('angle_coeff\t{:<5d}\t{}\t#{}\n'.format(i,a2ff[i],a2ele[i]))

    # Write the angle potentials section
    if len(d2ele): 
        f.write('\n# Dihedral interactions\n')
        for i in sorted(d2ele.keys()):
            f.write('dihedral_coeff\t{:<5d}\t{}\t#{}\n'.format(i,d2ff[i],d2ele[i]))

    f.close()
    return 

# TAFFI map file
def read_map(file_map):
    # Reads map file info 
    # Input:
    # file_map: name of map file
    # Output:
    # dictionary of all columns in the map file
    objects={'Nmol':0,'Natom':0}

    f = open(file_map, 'r')
    for i, line in enumerate(f):
        if i==0:
            objects={'Natom': int(line.split()[0]), 'Nmol': int(line.split()[1])} # Reads first line, assigns to numer of atoms and molecules
        elif i==1:
            keys=line.split()   
            idx={k:i for i,k in enumerate(keys)} # Read second line, assign headers to keys of dictionary
        elif line.split()==[]: # Skip blank lines!
            pass
        else:
            line=line.split()
            for k in list(idx.keys()):
                #print(k)
                if k not in list(objects.keys()):
                    objects[k]=[]
                if k in ['Atom_type']:
                    objects[k].append(int(line[idx[k]]))
                elif k in ['Charge','Mass','Shapex','Shapey','Shapez']:
                    objects[k].append(float(line[idx[k]]))   
                elif k=='Adj_mat':
                    objects[k].append([int(_) for _ in line[idx[k]:]]) 
                else:
                    #print((idx[k]))
                    #print(line)
                    objects[k].append(line[idx[k]])
    f.close()
    return objects

# Convert adjacency list to matrix 
def adj_list2mat(A_list):
    A=zeros((len(A_list),len(A_list)))
    for i in range(len(A_list)):
        for j in A_list[i]:
            A[i,j]=1

    return A

def mass_average(bead_list, bead_types, masses):
    Mass=np.zeros(len(bead_types))
    for j,t in enumerate(bead_types):
        count = 0
        running_mass=0
        for i,l in enumerate(bead_list):
            if t==l:
                count+=1
                running_mass+=masses[i]
            else:
                pass
        Mass[j]=running_mass/count
    print("mass", Mass)
    return Mass

if __name__ == "__main__":
   main(sys.argv[1:])
