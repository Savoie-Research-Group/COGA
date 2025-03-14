#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import *
from itertools import combinations,permutations
from copy import deepcopy
home=str(Path.home())           # Path to home directory
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-1])        # Path to this script.

######### Contains a number of functions that are useful for analyzing and placing molecules, writing files, etc. ######### 

def read_xyz(file_xyz):
    # Reads the geometry and elements of a molecule from xyz file
    elements, geometry, charge = [[],[],[]]
    with open(file_xyz,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc>1:
                elements += [fields[0]]
                geometry.append([float(fields[1]),float(fields[2]),float(fields[3])])
                if len(fields)>4:
                    try:
                        charge  += [float(fields[4])]
                    except:
                        charge += [0]
                else:
                   charge += [0]

    return elements, geometry, charge

def write_xyz(traj, output_name, ele=[]):
    natom=traj.shape[0]
    xyz = open(output_name,'w')
    xyz.write(str(natom)+'\n')
    xyz.write('\n')
    line = '{:<15s} {:<15f} {:<15f} {:<15f}\n'
    for a in range(natom):
        # atom type (given as C for all), x, y, z 
        if not len(ele):
            xyz.write(line.format( 'C', traj[a,0], traj[a,1], traj[a,2]))
        else:
            xyz.write(line.format( ele[a], traj[a,0], traj[a,1], traj[a,2]))
    xyz.close()
    return

def find_bonds(A, atom_types):
    # Generates list of pairs of bonded atoms for a given Adjacency matrix
    # Input: 
    # A: adjacency matrix
    # Output:
    # bonds: list of pairs of bonded atoms 
    bonds_atom_types =[]
    bonds_atom_id = []
    for i in range(len(A)):
        for j in range(i,len(A)):
            if np.triu(A)[i,j]==1:
                bond = sorted([atom_types[i],atom_types[j]])
                if bond not in bonds_atom_types:
                    bonds_atom_types.append(bond)
                    bonds_atom_id.append([[i,j]])
                else:
                    idx = bonds_atom_types.index(bond)
                    bonds_atom_id[idx].append([i,j])

    return bonds_atom_id, bonds_atom_types

def find_angles(A, atom_types):
    # Generates list of atoms forming an angle for a given Adjacency matrix
    # Input: 
    # A: adjacency matrix
    # Output:
    # angles: list of atoms forming a angles
    angles_atom_types = []
    angles_atom_id = []
    for a in range(len(A)):
        ind = [j for j,v in enumerate(A[a]) if A[a][j]==1]
        if len(ind)>1:
            for a1, a2 in combinations(ind,2):
                angle = sorted([atom_types[a1], atom_types[a2]])
                angle.insert(1,atom_types[a])
                if angle not in angles_atom_types: 
                    angles_atom_types.append(angle)
                    angles_atom_id.append([[a1,a,a2]])
                else:
                    idx = angles_atom_types.index(angle)
                    # print([a1+1,a+1,a2+1])
                    angles_atom_id[idx].append([a1,a,a2])

    return angles_atom_id, angles_atom_types

def find_dihedrals(A, atom_types):
    # Generates list of atoms forming an angle for a given Adjacency matrix                                                                                                                                                                      
    # Input:                                                                                                                                                                                                                                     
    # A: adjacency matrix                                                                                                                                                                                          
    dihedrals_atom_types = []
    dihedrals_atom_id = []
    for a2 in range(len(A)):
        ind2= [j for j,v in enumerate(A[a2]) if A[a2][j]==1]
        if len(ind2)>1:
            for a1, a3 in permutations(ind2,2):
                ind3= [j for j,v in enumerate(A[a3]) if A[a3][j]==1]
                if len(ind3)>1:
                    for a4 in ind3:
                        if a4 not in [a1,a2]:
                            if atom_types[a1]<atom_types[a4]:
                                dihed=[atom_types[a1], atom_types[a2], atom_types[a3], atom_types[a4]]
                                dihed_id=[a1,a2,a3,a4]                                    
                            elif atom_types[a1]>atom_types[a4]:
                                dihed=[atom_types[a4], atom_types[a3], atom_types[a2], atom_types[a1]]
                                dihed_id=[a4,a3,a2,a1]
                            else:
                                if atom_types[a2]<atom_types[a3]:
                                    dihed=[atom_types[a1], atom_types[a2], atom_types[a3], atom_types[a4]]
                                    dihed_id=[a1,a2,a3,a4]
                                else:
                                    dihed=[atom_types[a4], atom_types[a3], atom_types[a2], atom_types[a1]]
                                    dihed_id=[a4,a3,a2,a1]

                            if dihed not in dihedrals_atom_types:
                                dihedrals_atom_types.append(dihed)
                                dihedrals_atom_id.append([dihed_id])                                
                            else:
                                idx = dihedrals_atom_types.index(dihed)
                                if dihed_id not in dihedrals_atom_id[idx] and list(reversed(dihed_id)) not in dihedrals_atom_id[idx]:
                                    dihedrals_atom_id[idx].append(dihed_id)


    return dihedrals_atom_id, dihedrals_atom_types

def adj_list2mat(A_list):
    # Convert adjacency list to matrix 
    A=np.zeros((len(A_list),len(A_list)))
    for i in range(len(A_list)):
        for j in A_list[i]:
            A[i,j]=1

    return A

def Table_generator(Elements,Geometry,File=None,Radii_dict=False):
    module_path =  '/'.join(os.path.abspath(__file__).split('/')[:-1])
    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # NOTE: Units of angstroms 
    # NOTE: These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    Radii = {  'H':0.354, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.244, 'Si':1.117,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # SAME AS ABOVE BUT WITH A SMALLER VALUE FOR THE Al RADIUS ( I think that it tends to predict a bond where none are expected
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # Use Radii json file in Lib folder if sepcified
    if Radii_dict == True:
      if os.path.isfile(module_path+'/Radii.json') is False:
         print("ERROR: {}/Radii.json doesn't exist, check for Radii.json file in the Library".format(module_path))
         quit()
      Radii = read_alljson(module_path+'/Radii.json')


    Max_Bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                   'K':None, 'Ca':None, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':3,    'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
                     
    # Scale factor is used for determining the bonding threshold. 1.2 is a heuristic that give some lattitude in defining bonds since the UFF radii correspond to equilibrium lengths. 
    scale_factor = 1.2

    # Print warning for uncoded elements.
    for i in Elements:
        if i not in Radii.keys():
            print( "ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i)+\
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(Geometry,Geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Initialize Adjacency Matrix
    Adj_mat = np.zeros([len(Geometry),len(Geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*scale_factor:            
            Adj_mat[i,y_ind[count]]=1
    
        if Elements[i] == 'H' and Elements[y_ind[count]] == 'H':
            if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*1.5:
                Adj_mat[i,y_ind[count]]=1

    # Hermitize Adj_mat
    Adj_mat=Adj_mat + Adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in Radii.keys() }
    conditions = { "H":1, "C":4, "F":1, "Cl":1, "Br":1, "I":1, "O":2, "N":4, "B":4 }
    for count_i,i in enumerate(Adj_mat):

        if Max_Bonds[Elements[count_i]] is not None and sum(i) > Max_Bonds[Elements[count_i]]:
            problem_dict[Elements[count_i]] += 1
            cons = sorted([ (Dist_Mat[count_i,count_j],count_j) if count_j > count_i else (Dist_Mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(Adj_mat[count_i]) > Max_Bonds[Elements[count_i]]:
                sep,idx = cons.pop(0)
                Adj_mat[count_i,idx] = 0
                Adj_mat[idx,count_i] = 0



    # Print warning messages for obviously suspicious bonding motifs.
    if sum( [ problem_dict[i] for i in problem_dict.keys() ] ) > 0:
        print( "Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if File is None:
                    if i == "H": print( "WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(problem_dict[i]))
                    if i == "C": print( "WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "Si": print( "WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "F": print( "WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Cl": print( "WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Br": print( "WARNING in Table_generator: {} bromine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "I": print( "WARNING in Table_generator: {} iodine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "O": print( "WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(problem_dict[i]))
                    if i == "N": print( "WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "B": print( "WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(problem_dict[i]))
                else:
                    if i == "H": print( "WARNING in Table_generator: parsing {}, {} hydrogen(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "C": print( "WARNING in Table_generator: parsing {}, {} carbon(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "Si": print( "WARNING in Table_generator: parsing {}, {} silicons(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "F": print( "WARNING in Table_generator: parsing {}, {} fluorine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "Cl": print( "WARNING in Table_generator: parsing {}, {} chlorine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "Br": print( "WARNING in Table_generator: parsing {}, {} bromine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "I": print( "WARNING in Table_generator: parsing {}, {} iodine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "O": print( "WARNING in Table_generator: parsing {}, {} oxygen(s) have more than two bonds.".format(File,problem_dict[i]))
                    if i == "N": print( "WARNING in Table_generator: parsing {}, {} nitrogen(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "B": print( "WARNING in Table_generator: parsing {}, {} bromine(s) have more than four bonds.".format(File,problem_dict[i]))
        print( "")

    return Adj_mat

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

def id_types(elements,A,gens=2,which_ind=None,avoid=[],geo=None,hybridizations=[],algorithm="matrix"):

    # On first call initialize dictionaries
    if not hasattr(id_types, "mass_dict"):

        # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
        # NOTE: It's inefficient to reinitialize this dictionary every time this function is called
        id_types.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                             'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                              'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                             'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                             'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                             'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                             'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                             'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    if algorithm == "matrix":

        # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
        masses = [ id_types.mass_dict[i] for i in elements ]
        atom_types = [ "["+taffi_type(i,elements,A,masses,gens)+"]" for i in range(len(elements)) ]

        # Add ring atom designation for atom types that belong are intrinsic to rings 
        # (depdends on the value of gens)
        for count_i,i in enumerate(atom_types):
            if ring_atom_new(A,count_i,ring_size=(gens+2)) == True:
                atom_types[count_i] = "R" + atom_types[count_i]            

    elif algorithm == "list":

        # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
        masses = [ id_types.mass_dict[i] for i in elements ]
        atom_types = [ "["+taffi_type_list(i,elements,A,masses,gens)+"]" for i in range(len(elements)) ]

        # Add ring atom designation for atom types that belong are intrinsic to rings 
        # (depdends on the value of gens)
        for count_i,i in enumerate(atom_types):
            if ring_atom_list(A,count_i,ring_size=(gens+2)) == True:
                atom_types[count_i] = "R" + atom_types[count_i]            

    return atom_types

# adjacency matrix based algorithm for identifying the taffi atom type
def taffi_type(ind,elements,adj_mat,masses,gens=2,avoid=[]):

    # On first call initialize dictionaries
    if not hasattr(taffi_type, "periodic"):

        # Initialize periodic table
        taffi_type.periodic = { "h": 1,  "he": 2,\
                               "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                               "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                               "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                               "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

    # Find connections, avoid is used to avoid backtracking
    cons = [ count_i for count_i,i in enumerate(adj_mat[ind]) if i == 1 and count_i not in avoid ]

    # Sort the connections based on the hash function 
    if len(cons) > 0:
        cons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in cons ])[::-1]))[1]

    # Calculate the subbranches
    # NOTE: recursive call with the avoid list results 
    if gens == 0:
        subs = []
    else:
        subs = [ taffi_type(i,elements,adj_mat,masses,gens=gens-1,avoid=[ind]) for i in cons ]

    return "{}".format(taffi_type.periodic[elements[ind].lower()]) + "".join([ "["+i+"]" for i in subs ])

def ring_atom_new(adj_mat,idx,start=None,ring_size=10,counter=0,avoid=[]):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    
    # Loop over connections and recursively search for idx
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid ]
    if len(cons) == 0:
        return False
    elif start in cons:
        return True
    else:
        for i in cons:
            if ring_atom_new(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid=[idx]) == True:
                return True
        return False

# hashing function for canonicalizing geometries on the basis of their adjacency matrices and elements
# ind  : index of the atom being hashed
# A    : adjacency matrix
# M    : masses of the atoms in the molecule
# gens : depth of the search used for the hash   
def atom_hash(ind,A,M,alpha=100.0,beta=0.1,gens=10):    
    if gens <= 0:
        return rec_sum(ind,A,M,beta,gens=0)
    else:
        return alpha * sum(A[ind]) + rec_sum(ind,A,M,beta,gens)

# recursive function for summing up the masses at each generation of connections. 
def rec_sum(ind,A,M,beta,gens,avoid_list=[]):
    if gens != 0:
        tmp = M[ind]*beta
        new = [ count_j for count_j,j in enumerate(A[ind]) if j == 1 and count_j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i,A,M,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return M[ind]*beta

# hashing function for canonicalizing geometries on the basis of their adjacency lists and elements
# ind  : index of the atom being hashed
# A    : adjacency list
# M    : masses of the atoms in the molecule
# gens : depth of the search used for the hash   
def atom_hash_list(ind,A,M,alpha=100.0,beta=0.1,gens=10):    
    if gens <= 0:
        return rec_sum_list(ind,A,M,beta,gens=0)        
    else:
        return alpha * len(A[ind]) + rec_sum_list(ind,A,M,beta,gens)

# recursive function for summing up the masses at each generation of connections. 
def rec_sum_list(ind,A,M,beta,gens,avoid_list=[]):
    if gens != 0:
        tmp = M[ind]*beta
        new = [ j for j in A[ind] if j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum_list(i,A,M,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return M[ind]*beta

# Description: returns a canonicallized TAFFI bond. TAFFI bonds are written so that the lesser *atom_type* between 1 and 2 is first. 
# 
# inputs:      types: a list of taffi atom types defining the bond
#              ind:   a list of indices corresponding to the bond
#
# returns:     a canonically ordered bond (and list of indices if ind was supplied)
def canon_bond(types,ind=None):

    # consistency checks
    if len(types) != 2: 
        print("ERROR in canon_bond: the supplied dihedral doesn't have two elements. Exiting...")
        quit()
    if ind != None and len(ind) != 2: 
        print("ERROR in canon_bond: the iterable supplied to ind doesn't have two elements. Exiting...")
        quit()
        
    # bond types are written so that the lesser *atom_type* between 1 and 2 is first.
    if types[0] <= types[1]:
        if ind == None:
            return types
        else:
            return types,ind
    else:
        if ind == None:
            return types[::-1]
        else:
            return types[::-1],ind[::-1]

# Description: returns a canonicallized TAFFI angle. TAFFI angles are written so that the lesser *atom_type* between 1 and 3 is first. 
# 
# inputs:      types: a list of taffi atom types defining the angle
#              ind:   a list of indices corresponding to the angle
#
# returns:     a canonically ordered angle (and list of indices if ind was supplied)
def canon_angle(types,ind=None):

    # consistency checks
    if len(types) != 3: 
        print("ERROR in canon_angle: the supplied dihedral doesn't have three elements. Exiting...")
        quit()
    if ind != None and len(ind) != 3: 
        print("ERROR in canon_angle: the iterable supplied to ind doesn't have three elements. Exiting...")
        quit()
        
    # angle types are written so that the lesser *atom_type* between 1 and 3 is first.
    if types[0] <= types[2]:
        if ind == None:
            return types
        else:
            return types,ind
    else:
        if ind == None:
            return types[::-1]
        else:
            return types[::-1],ind[::-1]

# # Description: returns a canonicallized TAFFI dihedral. TAFFI dihedrals are written so that the lesser *atom_type* between 1 and 4 is first. 
# #              In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first. 
# #
# # inputs:      types: a list of taffi atom types defining the dihedral
# #              ind:   a list of indices corresponding to the dihedral
# #
# # returns:     a canonically ordered dihedral (and list of indices if ind was supplied)
# def canon_dihedral_old(types,ind=None):

#     # consistency checks
#     if len(types) != 4: 
#         print "ERROR in canon_dihedral: the supplied dihedral doesn't have four elements. Exiting..."
#         quit()
#     if ind != None and len(ind) != 4: 
#         print "ERROR in canon_dihedral: the iterable supplied to ind doesn't have four elements. Exiting..."
#         quit()
        
#     # dihedral types are written so that the lesser *atom_type* between 1 and 4 is first.
#     # In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first
#     if types[0] == types[3]:
#         if types[1] <= types[2]:
#             if ind == None:
#                 return types
#             else:
#                 return types,ind
#         else:
#             if ind == None:
#                 return types[::-1]
#             else:
#                 return types[::-1],ind[::-1]
#     elif types[0] < types[3]:
#         if ind == None:
#             return types
#         else:
#             return types,ind
#     else:
#         if ind == None:
#             return types[::-1]
#         else:
#             return types[::-1],ind[::-1]

# Description: returns a canonicallized TAFFI dihedral. TAFFI dihedrals are written so that the lesser *atom_type* between 1 and 4 is first. 
#              In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first. 
#
# inputs:      types: a list of taffi atom types defining the dihedral
#              ind:   a list of indices corresponding to the dihedral
#
# returns:     a canonically ordered dihedral (and list of indices if ind was supplied)
def canon_dihedral(types_0,ind=None):
    
    # consistency checks
    if len(types_0) < 4: 
        print("ERROR in canon_dihedral: the supplied dihedral has less than four elements. Exiting...")
        quit()
    if ind != None and len(ind) != 4: 
        print("ERROR in canon_dihedral: the iterable supplied to ind doesn't have four elements. Exiting...")
        quit()

    # Grab the types and style component (the fifth element if available)
    types = list(types_0[:4])
    if len(types_0) > 4:
        style = [types_0[4]]
    else:
        style = []

    # dihedral types are written so that the lesser *atom_type* between 1 and 4 is first.
    # In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first
    if types[0] == types[3]:
        if types[1] <= types[2]:
            if ind == None:
                return tuple(types+style)
            else:
                return tuple(types+style),ind
        else:
            if ind == None:
                return tuple(types[::-1]+style)
            else:
                return tuple(types[::-1]+style),ind[::-1]
    elif types[0] < types[3]:
        if ind == None:
            return tuple(types+style)
        else:
            return tuple(types+style),ind
    else:
        if ind == None:
            return tuple(types[::-1]+style)
        else:
            return tuple(types[::-1]+style),ind[::-1]

# Description: returns a canonicallized TAFFI improper. TAFFI impropers are written so that 
#              the three peripheral *atom_types* are written in increasing order.
#
# inputs:      types: a list of taffi atom types defining the improper
#              ind:   a list of indices corresponding to the improper
#
# returns:     a canonically ordered improper (and list of indices if ind was supplied)
def canon_improper(types,ind=None):

    # consistency checks
    if len(types) != 4: 
        print("ERROR in canon_improper: the supplied improper doesn't have four elements. Exiting...")
        quit()
    if ind != None and len(ind) != 4: 
        print("ERROR in canon_improper: the iterable supplied to ind doesn't have four elements. Exiting...")
        quit()
        
    # improper types are written so that the lesser *atom_type* between 1 and 4 is first.
    # In the event that 1 and 4 are of the same type, then the lesser of 2 and 3 goes first
    if ind == None:
        return tuple([types[0]]+sorted(types[1:]))
    else:
        tmp_types,tmp_ind = list(zip(*sorted(zip(types[1:],ind[1:]))))
        return tuple([types[0]]+list(tmp_types[:])),tuple([ind[0]]+list(tmp_ind[:]))