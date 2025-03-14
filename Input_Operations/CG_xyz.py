#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
import math
from pathlib import Path
from itertools import permutations, combinations_with_replacement
import subprocess as sp
# home=str(Path.home())
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
from Mol_ops import *
from gen_periodic_md import cartesian_to_fract, parse_cif

# sys.path.append('{}/bin/taffi/CG'.format(home))
# from file_operations import read_xyz, write_xyz
# from post_process import find_bonds, find_angles, find_dihedrals
# from gen_md_inputs_martini import adj_list2mat
# sys.path.append('{}/bin/taffi_beta_old/Lib/'.format(home))
# from adjacency import Table_generator
# sys.path.append('{}/bin/taffi_beta_old/Parsers'.format(home))
# from percolation_calc import find_subgraphs

# sys.path.append('{}/bin/CG_Crystal'.format(home))
# from gen_periodic_md import cartesian_to_fract, parse_cif
# sys.path.append('{}/bin/taffi_beta_old/FF_functions/'.format(home))

def main(argv):
    parser = argparse.ArgumentParser(description='''Rewrites .xyz files based on CG mapping applied. Writes a FF, data, settings, map, and cif files for the CG model.                                                                                                                                     
    Input: list of xyz files, original cif file, list of map files, optional output file name,  optional number of molecules specified in map file.
                                                                                                                                                                                                      
    Output: CGed .xyz files, FF template (with several values missing/assumed), data file, settings file, map file, and CG cif files.                                                                                                                                                                               
                                                                                                                                                                                                            
    Assumptions: For the purposes of creating the FF file, molecules are assumed to be identical in structure. Bond, Angle, and Dihedral values are assumed or required to be replaced.
                If a list of xyz files and maps is provided, it is assumed they designated individual atoms. If one file is supplied it will be searched for molecules. 
    ''')

    # Required Arguments
    parser.add_argument('xyz_list', type=str, help='String containing the list of xyz files.')
    parser.add_argument('cif_file', type=str, help='String containing name of cif file to be edited.')
    parser.add_argument('map_list', type=str, help='String containing the list of map files. Map files are txt format, the first column contains martini bead types, the second column contains atom numbers in those beads (one indexed ,column separated.)')
    

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG', help='Output file name. Default: CG.xyz')
    parser.add_argument('-map_mols', dest='map_mols', type=int, default=1, help='Number of molecules identified in mapping provided.')

    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    xyzs = str(args.xyz_list).split()
    maps = str(args.map_list).split()
    suffix_name=args.output
    if len(xyzs)!=len(maps):
        print("ERROR: Please supply the same number of xyz and map files :) ")
        return
    # Split function based on how molecules are defined.
    all_beads=list()
    # If molecules are supplied as separate xyz files as in split_cif_moles, they can be read in this block.
    if len(xyzs)>1:
        N_Mols=len(xyzs)
        for i in range(N_Mols):
            #args.output=str(xyzs[i]).split('.')[0]+"_"+suffix_name
            m = open(maps[i],'r')
            map_list=list()
            geo_CG=list()
            bead_CG=list()
            adj_list=list()
            mass_list=list()
            ele_np=np.array(elements)
            geo_np=np.array(geometry)
            for lines in m:
                words = lines.split()
                if words[0]=="Bead":
                    pass
                else:
                    map_list.append(words)
            for b in range(len(map_list)):
                bead_CG=np.append(bead_CG, map_list[b][0])
                atom_nums=np.array(map_list[b][1].split(',')).astype(int)-1
                COM, Mass = Element_COM(ele_np[atom_nums],geo_np[atom_nums])
                geo_CG=np.append(geo_CG, list(COM))
                mass_list.append(Mass)
                adj_list.append([int(i) for i in (map_list[b][2].split(','))])
            geo_CG_mat=np.reshape(np.array(geo_CG),(len(bead_CG),3))
            write_xyz(geo_CG_mat, args.output+"{}.xyz".format(i+1), np.array(bead_CG))
            all_beads=np.append(all_beads, bead_CG)
    # If molecules need to be parsed from one large xyz file and a new map file made, it is done here.
    else:
        #args.output=str(xyzs[0]).split('.')[0]+"_"+suffix_name
        elements, geometry, charge=read_xyz(xyzs[0])
        Adj_mat = Table_generator(elements, geometry)
        SG_list=find_subgraphs(Adj_mat)
        N_Mols=len(SG_list[:])
        print("Number of molecules found: "+str(N_Mols))
        m = open(maps[0],'r')
        map_list=list()
        for lines in m:
                words = lines.split()
                if words==[]:
                    pass
                elif words[0]=="Bead":
                    pass
                else:
                    map_list.append(words)
        N_beads=len(map_list)
        print("Number of beads found: "+str(N_beads))
        geo_CG=list()
        bead_CG=list()
        adj_list=list()
        atom_set=[]
        atom_list=[]
        mass_list=[]
        ele_np=np.array(elements)
        geo_np=np.array(geometry)
        N_Atoms_PM=len(ele_np)/N_Mols
        print("Number of atoms found per molecule: "+str(N_Atoms_PM))
        if N_Mols%args.map_mols!=0:
            print("Number of molecules in xyz is not evenly divisible by requested number of molecules. Exiting...")
            quit()
        for moles in range(int(N_Mols/args.map_mols)):
            for b in range(len(map_list)):
                bead_CG=np.append(bead_CG, map_list[b][0])
                atom_nums=np.array(map_list[b][1].split(',')).astype(int)-1+int(moles*N_Atoms_PM*args.map_mols)
                atom_set=np.append(atom_set,atom_nums)
                atom_list.append(atom_nums)
                COM,Mass = Element_COM(ele_np[atom_nums],geo_np[atom_nums])
                mass_list.append(Mass)
                geo_CG=np.append(geo_CG, list(COM))
                adj_list.append(np.array([int(i) for i in (map_list[b][2].split(','))])+moles*N_beads)
                all_beads=np.append(all_beads, bead_CG)
        geo_CG_mat=np.reshape(np.array(geo_CG),(len(bead_CG),3))
        write_xyz(geo_CG_mat, args.output+".xyz", np.array(bead_CG))
        atom_set=set(atom_set)
        if len(atom_set)!=N_Atoms_PM*N_Mols: # Try to catch errors in submitted mapping file.
            print("WARNING! {} of {} atoms are represented in bead assignment!".format(len(atom_set), N_Atoms_PM*N_Mols))
            quit() 
   

# Try to write FF files
    print("bead CG: ", bead_CG)
    all_beads=list(dict.fromkeys(all_beads))
    print("all beads: ", all_beads)
    Mass, Mass_by_atom=mass_average(bead_CG, all_beads, mass_list)
    print("Mass outside: ", Mass)
    pair_types=' '.join(['{},{}'.format(all_beads[i1],all_beads[i2]) for i1, i2 in combinations_with_replacement(range(0,len(all_beads)),2)])
    martiniFF='~/bin/martini_v300/martini_v3.0.0.itp'
    sp.call('python ~/bin/CG_Crystal/gromacs2lammps_m3.py {} -pair_types "{}"; wait'.format(martiniFF,pair_types), shell=True) #~/bin/taffi/CG/gromacs2lammps.py
    write_FF_from_martini(args.output,all_beads,bead_CG,adj_list,geo_CG_mat, Mass) # Write FF file if needed
    rewrite_cif(args.cif_file,bead_CG,geo_CG_mat,args.output, atom_list)
    write_map(bead_CG, all_beads, adj_list, args.output, N_Mols, Mass_by_atom)
    sp.call('python ~/bin/taffi/CG/gen_md_inputs_martini.py -maps {} -xyzs {} -Ns "1" -martiniFF {} -o "martini" -bFF FF.txt > martini_info.txt; wait'.format(args.output+".map",args.output+".xyz",martiniFF), shell=True)
    
    print('Success!! Good job! Have a good day!')

def Element_COM(elements, geometry):
    masses=[] #np.empty(len(elements))
    for e in elements:
        if e == "C":
            masses=np.append(masses,12)
        elif e == "N":
            masses=np.append(masses,14)
        elif e == "O":
            masses=np.append(masses,16)
        elif e == "H":
            masses=np.append(masses,1)
        else:
            print("Found unsupported atom type: {}, atomic mass of 12 will be assumed.".format(e))
            masses.append(12)
    geo_np=np.array(geometry)
    Mass=sum(masses)
    COM=np.matmul(masses, geo_np)/Mass
    return COM,Mass

def write_FF_from_martini(file_name,all_beads,each_bead,adj_list,geo_mat,Masses,gro_name='lammps_para.txt'):
    param_array=np.loadtxt(gro_name,dtype=str)
    Adj_mat=adj_list2mat(adj_list)
    bond_id, bond_type=find_bonds(Adj_mat, each_bead)
    angle_id, angle_type=find_angles(Adj_mat, each_bead)
    dihedral_id, dihedral_type=find_dihedrals(Adj_mat, each_bead)
    #print(bond_id,angle_id)
    #print(bond_type,angle_type)
    # Writes FF files
    with open('{}_FF.db'.format(file_name),'w') as f:
        f.write('# Atom type definitions\n#\n#  Atom_type   Label   Mass    Mol_ID\n')
        for i in range(len(all_beads)):
            f.write('atom   {}  {}  {}\n'.format(all_beads[i],all_beads[i], Masses[i]))
        f.write('\n# VDW definitions\n#\n#  Atom_type   Atom_type   Potential   params\n')
        for j in range(len(param_array)):
            f.write('{} {} {}  {}  {}  {}\n'.format('vdw',param_array[j][0],param_array[j][1],'lj',param_array[j][2],param_array[j][3]))
        f.write('\n# Bond type definitions\n#\n# Atom_type   Atom_type   style   params (k, r0)  Mol_ID\n')
        for k in range(len(bond_type)):
            f.write('bond   {}  {}  harmonic    10000.0 5.0\n'.format(bond_type[k][0],bond_type[k][1]))
        f.write('\n# Angle type definitions\n#\n# Atom_type   Atom_type   Atom_type  style   params (k, theta0)  Mol_ID\n')
        for l in range(len(angle_type)):
            f.write('angle  {}  {}  {}  harmonic    3.0 180.0\n'.format(angle_type[l][0],angle_type[l][1],angle_type[l][2]))
        f.write('\n# Dihedral type definitions\n# Atom_type   Atom_type   Atom_type   Atom_type   style   params (k, r0)  Mol_ID\n')
        for m in range(len(dihedral_type)):
            f.write('torsion    {}  {}  {}  {}  opls    0.0 0.0 0.0 0.0\n'.format(dihedral_type[m][0],dihedral_type[m][1],dihedral_type[m][2],dihedral_type[m][3]))
    # Calculates bond and angle values
    bond_vals=np.zeros((len(bond_type),1))
    angle_vals=np.zeros((len(angle_type),1))
    for r in range(len(bond_id)):
        ind_bonds=[]
        for s in range(len(bond_id[r])):
            ind_bonds.append(np.linalg.norm(geo_mat[int(bond_id[r][s][0]),:]-geo_mat[int(bond_id[r][s][1]),:]))
        bond_vals[r]=np.mean(ind_bonds)
        #print("individual bond lengths step "+str(r))
        #print(ind_bonds)
    #print("average bond lengths")
    #print(bond_vals)
    for t in range(len(angle_id)):
        ind_angles=[]
        for u in range(len(angle_id[t])):
            p1=geo_mat[int(angle_id[t][u][0]),:]
            c0=geo_mat[int(angle_id[t][u][1]),:]
            p2=geo_mat[int(angle_id[t][u][2]),:]
            v1=p1-c0
            v2=p2-c0
            cos_val=np.inner(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            #print(cos_val)
            ind_angles.append(np.rad2deg(np.arccos(np.clip(cos_val,-1.0,1.0))))
        angle_vals[t]=np.mean(ind_angles)
        #print("individual angle lengths step "+str(t))
        #print(ind_angles)
        #print(angle_id[t])
    with open('FF.txt','w') as f:
        f.write('# Bonds\n')
        for v in range(len(bond_type)):
            f.write('{} {} harmonic     10000.0 {}\n'.format(bond_type[v][0],bond_type[v][1],bond_vals[v][0]))
        f.write('# Angles\n')
        for w in range(len(angle_type)):
            f.write('{} {} {} harmonic   3.0 {}\n'.format(angle_type[w][0],angle_type[w][1],angle_type[w][2],angle_vals[w][0]))
        f.write('# Dihedrals\n')
        for m in range(len(dihedral_type)):
            f.write('{} {} {} {} opls    0.0 0.0 0.0 0.0\n'.format(dihedral_type[m][0],dihedral_type[m][1],dihedral_type[m][2],dihedral_type[m][3]))


def rewrite_cif(cif_file,each_bead,geo_CG,file_name, atom_list):
    with open(cif_file,'r') as ref_file, open('{}.cif'.format(file_name),'w') as new_file:
        # Copy first lines of cif file over to new file
        cell_dims=[]
        for line in ref_file:
            if line.split()==[]:
                new_file.write(line)
            elif line.split()[0]=='_atom_site_fract_z':
                new_file.write(line)
                break
            else:
                if line.split()[0]=="_cell_length_a":
                    a=float(line.split()[1].split('(')[0])
                elif line.split()[0]=="_cell_length_b":
                    b=float(line.split()[1].split('(')[0])
                elif line.split()[0]=="_cell_length_c":
                    c=float(line.split()[1].split('(')[0])
                elif line.split()[0]=='_cell_angle_alpha':
                    alpha = float(line.split()[1].split('(')[0])
                elif line.split()[0]=='_cell_angle_beta':
                    beta = float(line.split()[1].split('(')[0])
                elif line.split()[0]=='_cell_angle_gamma':
                    gamma = float(line.split()[1].split('(')[0])
                new_file.write(line)
        # Add new geometry to new cif file
        # alpha=float(alpha)*math.pi/180.0 # Get angles and convert to radians
        # beta=float(beta)*math.pi/180.0 # Get angles and convert to radians
        # gamma=float(gamma)*math.pi/180.0 # Get angles and convert to radians
        alpha=np.radians(float(alpha))
        beta=np.radians(float(beta))
        gamma=np.radians(float(gamma))
        lx=a
        xy=b*math.cos(gamma)
        xz=c*math.cos(beta)
        ly=(b**(2.0)-xy**(2.0))**(0.5)
        yz=(b*c*math.cos(alpha)-xy*xz)/ly
        lz=(c**(2.0)-xz**(2.0)-yz**(2.0))**(0.5)
        trans_mat=cartesian_to_fract(a,b,c,alpha,beta,gamma,degrees=False)
        mol_data, crystal_data, temp_data=parse_cif(cif_file)
        # print("mol_data: ", mol_data)
        # print("atom_list: ", atom_list)
        # print("one entry: ", atom_list[0])
        all_frac_geos=[]
        all_elements=[]
        for l in mol_data.keys():
            all_frac_geos=np.append(all_frac_geos, mol_data[l]['Frac_geo'])
            all_elements=np.append(all_elements, mol_data[l]['Elements'])
        all_frac_geos=np.reshape(all_frac_geos, (int(len(all_frac_geos)/3), 3))
        # print("all_frac_geos", all_frac_geos)
        # print("all_elements", all_elements)
        cif_COM=np.zeros((len(atom_list),3))
        for ind, i in enumerate(atom_list):
            # print(all_elements[i])
            # print(all_frac_geos[i])
            # print(ind)
            cif_COM[ind,:], Mass = Element_COM(all_elements[i],all_frac_geos[i])
        # print("cif_COM: ",cif_COM) # This should be an array which lists the fractional coordinates of the beads!
        for n in range(len(each_bead)):
            # new_file.write('{}  {}  {}  {}  {}\n'.format(each_bead[n],each_bead[n],geo_CG[n][0]/lx,geo_CG[n][1]/ly,geo_CG[n][2]/lz))
            # frac_geo=np.matmul(geo_CG[n], trans_mat)
            # print(frac_geo)
            # new_file.write('{}  {}  {}  {}  {}\n'.format(each_bead[n],each_bead[n],frac_geo[0,0],frac_geo[0,1],frac_geo[0,2]))
            new_file.write('{}  {}  {}  {}  {}\n'.format(each_bead[n],each_bead[n],cif_COM[n,0],cif_COM[n,1],cif_COM[n,2]))

# Accepts beads and adjacency and writes a TAFFI style map file.
def write_map(each_bead,all_beads,adj_list,output,N_mols, Masses):
    if len(each_bead)!=len(adj_list):
        print("Uh oh! List of beads and adjacency list to not match lengths!")
        print(each_bead)
        print(adj_list)
        quit()
    else:
        bead_num=[]
        for q in range(len(each_bead)):
            bead_num.append(all_beads.index(each_bead[q])+1)
        #print(bead_num)
        with open("{}.map".format(output),'w') as map:
            map.write("{} {}\nAtom_type 	 Element  	  Shapex  	  Shapey  	  Shapez  	Martini_type	   Mass   	  Charge  	 Adj_mat".format(len(each_bead), N_mols))
            for o in range(len(each_bead)):
                map.write('\n{}   {}  5.000000    5.000000    5.000000    {}  {}   0.000000    '.format(bead_num[o],each_bead[o],each_bead[o], Masses[o]))
                for p in range(len(adj_list[o])):
                    map.write('{} '.format(adj_list[o][p]))

# Reads a map file and creates an extended version with more molecules (extends is number of molecules)
def extend_map(map_file, extends, output=''):
    extends=int(extends)
    # Read in map file
    with open(map_file, 'r') as o:
        map_data=[]
        adj_data=[]
        for inds, lines in enumerate(o):
            if len(lines.split())==0:
                pass
            else:
                fields=lines.split()
                if inds==0:
                    atoms=int(fields[0])
                    moles=int(fields[1])
                    #map_data=np.zeros(atoms)
                elif inds==1:
                    header=fields
                    adj_ind=header.index("Adj_mat")
                else:
                    map_data+=[fields[0:adj_ind]]
                    adj_data+=[fields[adj_ind:]]
    o.close()
    if output:
        pass
    else:
        output=map_file.split('.')[0]
    with open("{}_extend.map".format(output), 'w') as new:
        new.write("{} {}\n".format(atoms*extends, moles*extends))
        for _ in header:
            new.write("{}\t".format(_))
        new.write("\n")
        for e in range(extends):
            for d in range(len(map_data)):
                for m in map_data[d]:
                    new.write("{}\t".format(m))
                for a in adj_data[d]:
                    new.write("{} ".format(int(a)+e*atoms))
                new.write("\n")
    new.close()

def mass_average(bead_list, bead_types, masses):
    Mass=np.zeros(len(bead_types))
    Mass_list=np.zeros(len(bead_list))
    for j,t in enumerate(bead_types):
        count = 0
        running_mass=0
        indexes=[]
        for i,l in enumerate(bead_list):
            if t==l:
                count+=1
                running_mass+=masses[i]
                indexes.append(i)
            else:
                pass
        Mass[j]=running_mass/count
        Mass_list[indexes]=Mass[j]
    print("mass", Mass)
    print('mass list', Mass_list)
    return Mass, Mass_list

    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])