#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
import math
import random
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission", "/Dilatometry"]
for subf in subfolders:
    sys.path.append(script_directory+subf)
from pathlib import Path
home=str(Path.home())
from CG_xyz import extend_map

def main(argv):
    parser = argparse.ArgumentParser(description='''This script reads a data file for a Coarse-Grained Molecule and creates a data file for a box based on the orientations requested.                                                                                                                                         
    Input: data file of a CG molecule
                                                                                                                                                                                                      
    Output: .init file and .data file.                                                                                                                                                                                         
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('data', type=str, help='data file for the desired CG molecule.')

    # Optional Arguments
    parser.add_argument('-N', dest='N', type=int, default='125', help='Number of molecules to be added to the simulation box. ')
    parser.add_argument('-o', dest='output', type=str, default='CG_box', help='Name of .lmp file to be produced. ')
    parser.add_argument('-orient', dest='orient', type=str, default='random', help='How the orientation of the molecules should be generated. "random" randomly rotates the molecules. "align" performs no rotations and leaves molecules aligned.' +\
                        'The letters "x", "y", or "z", when provided, rotate the molecule to align along the provided axis. Due to gimbal lock some axis alignments may not work properly, always double check!')
    parser.add_argument('-map', dest='map', type=str, default='', help='If a map file is provided, it will be extended for use with analysis scripts such as vautocorr.py to reflect the number of molecules in the box.')
    parser.add_argument('-vol_scale', dest='vol_scale', type=float, default=3.0, help='Determines how many times larger the box space will be relative to the size of the molecule in each axis. i.e. for a molecule 5 wide in x, with this value 3, the cell will be 15 units in x.')
    parser.add_argument('-side_dims', dest='side_dims', type=str, default="", help='If provided, will generate a custom set of side dimensions for the box in terms of numbers of molecules.' + \
                        'Providing "find" will generate a box that is approximately cubic in length, but will likely require slightly adjusting the number of molecules simulated.')
    parser.add_argument('-rho', dest='rho', type=float, default=0.0, help='If provided, will scale the geometry of the simulation to accomodate the provided density (in g/cm^3). Note: may cause simulation issues if scaling is dramatic.')
    # Store Constants
    parser.add_argument('--dihed', dest='dihed', default=False, action='store_const', const=True, help='When present, script will use dihedrals provided, otherwise they will be ignored.')
    parser.add_argument('--tight', dest='tight', default=False, action='store_const', const=True, help='When present, script will create tight boxes around the molecules for more crystaline arrangements, rather than a large box that would be suited for diffuse systems.')

    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below

    # Read in .lmp data
    data_dict={}
    section_flag=''
    header=[]
    suffix = args.data.split(".")[-1]
    if suffix == "lmp":
        with open(args.data, 'r') as data:
            for line in data:
                words = line.split()
                if words==[]:
                    pass
                elif len(words)==1:
                    section_flag=words[0]
                    data_dict.update({section_flag:[]})
                elif words[-1] == "Coeffs":
                    section_flag=words[0]+" "+words[1]
                    data_dict.update({section_flag:[]})
                elif section_flag:
                    data_dict[section_flag].append(words)
                elif not section_flag:
                    header.append(words)
                else:
                    pass
        data.close()
    elif suffix == "data":
        with open(args.data, 'r') as data:
            for line in data:
                words = line.split()
                if words==[]:
                    pass
                elif len(words)==1:
                    section_flag=words[0]
                    data_dict.update({section_flag:[]})
                elif section_flag:
                    data_dict[section_flag].append(words)
                elif not section_flag:
                    header.append(words)
                else:
                    pass
        data.close()
    else:
        print("Error... Unrecognized data file suffix. This script is currently suited for .lmp or .data files.\n Please supply a correct data file. Exiting...")
        quit()
    print("header: ", header)
    print(int(header[1][0]))
    print("keys: ", data_dict.keys())
    print("data dictionary: ", data_dict)
    len_dict={}
    final_dict={}
    for k in data_dict.keys():
        len_dict.update({k:len(data_dict[k])})
        if k in ['Masses']: # These enumerate the types of interactions, so they don't need to be modified.
            final_dict.update({k:data_dict[k]})
        else:
            final_dict.update({k:np.zeros((args.N*np.shape(data_dict[k])[0], np.shape(data_dict[k])[1]))})
    print("final dict", final_dict)
           
    # Shift molecule coordinates to be positive in the example box

    ind_geom=np.array(data_dict["Atoms"])
    ind_geom=ind_geom.astype(float)
    ind_geom[:, -3:]=ind_geom[:, -3:]-np.min(ind_geom, axis=0)[-3:]
    if args.orient in ["x", "y", "z"]: # Rotate the template once instead of doing it every single time.
        # print("old", xyz)
        ind_geom[:, -3:]=axis_align(ind_geom[:, -3:], axis=args.orient)
        ind_geom[:, -3:]= ind_geom[:, -3:] - np.min(ind_geom[:, -3:], axis=0)
        # if i == 0:
        #     # print("new", xyz)
        
    
    print("ind", ind_geom)
    
    #print("cell_vol_ind", cell_vol_ind)
    if args.tight:
        cell_side_ind=(np.max(ind_geom[:, -3:], axis=0)-np.min(ind_geom[:, -3:], axis=0))+np.array([1.0, 1.0, 1.0])*args.vol_scale
        # cell_side_ind=cell_vol_ind
    else:
        cell_vol_ind=np.max(ind_geom, axis=0)[-3:]*args.vol_scale
        csi=np.max(cell_vol_ind)
        cell_side_ind=np.array([csi, csi, csi])
    print("csi", cell_side_ind)

    # Determine size of the box required
    if args.side_dims == "find":
        print("Finding dimensions for geometrically cubic box...\n")
        csi_sort_inds = np.argsort(cell_side_ind)
        print("csi_s_i", csi_sort_inds)
        r1 = float(cell_side_ind[csi_sort_inds[[2]]]/cell_side_ind[csi_sort_inds[[0]]])
        r2 = float(cell_side_ind[csi_sort_inds[[2]]]/cell_side_ind[csi_sort_inds[[1]]])
        n0 = float((args.N/(r1*r2))**(1/3))
        print(r1, r2, n0)
        n1 = round(r1*n0)
        n2 = round(r2*n0)
        n0 = round(n0)
        side = [0, 0, 0]
        ns = [n1, n2, n0]
        for ind, i in enumerate(csi_sort_inds):
            side[i]=ns[ind]
        cells = side[0]*side[1]*side[2]
        args.N = cells
        print("Number of atoms now {} to fit cubic structure... \n".format(args.N))
        print("Simulation box is {}x{}x{} molecules to fit cubic structure... \n".format(side[0], side[1], side[2]))
        for k in data_dict.keys():
            if k in ['Masses']: # These enumerate the types of interactions, so they don't need to be modified.
                pass
            else:
                final_dict[k] = np.zeros((args.N*np.shape(data_dict[k])[0], np.shape(data_dict[k])[1]))
    elif args.side_dims:
        side = [int(s) for s in args.side_dims.split()]
        cells = side[0]*side[1]*side[2]
    else:
        ex_side=args.N**(1/3)
        side=math.ceil(ex_side) # Number of cells on each side of the cube
        cells=side**3           # number of cells total in the cube
        side=[side, side, side]
    print(cells)

    
            # print("new csi", cell_side_ind)
    # Create several boxes, one for each molecule that will cover the space
    # Shift a molecule into each box, randomly rotating it within the box. 
    # Loop over all molecules until box is full
    total_geom=np.zeros((args.N*np.shape(ind_geom)[0], np.shape(ind_geom)[1]))
    for i in range(args.N):
        z_ind=math.floor(i/(side[0]*side[1]))
        y_ind=math.floor((i%(side[0]*side[1])/side[0]))
        x_ind=(i%(side[0]*side[1]))%side[0]
        box_inds=[x_ind, y_ind, z_ind] # Index of this box. Tells what to add to the geometries.
        #print("box_inds", box_inds)

        # Calculate geometry by translating into box and then randomly rotating
        # print("99 ", ind_geom[0,0])
        geom_i=ind_geom.copy()
        # print("101 ", ind_geom[0,0])
        xyz=geom_i[:, -3:]
        # print("ind_geom pre old", ind_geom)
        # print("103 ", ind_geom[0,0])
        # print("pre spin", xyz)
        # Give it a good ole spin
        if args.orient == "random":
            xyz=euler_rotate(xyz, axis="z")
            xyz=euler_rotate(xyz, axis="y")
            xyz=euler_rotate(xyz, axis="x")
        elif args.orient == "align":
            pass
        # elif args.orient in ["x", "y", "z"]:
        #     # print("old", xyz)
        #     xyz=axis_align(xyz, axis=args.orient)
        #     xyz= xyz - np.min(xyz, axis=0)
        #     if i == 0:
        #         # print("new", xyz)
        #         cell_side_ind=(np.max(xyz, axis=0)-np.min(xyz, axis=0))+np.array([1.0, 1.0, 1.0])*args.vol_scale
        #         # print("new csi", cell_side_ind)
        
        # If the min is positive we want to move it back, but /less than/ the distance to the cell wall so it stays positive and inside its box.
        # If the min is negative we want to move it foward, but /more than/ the distance to the cell wall so it stays positive and inside the box.
        xyz_min=np.min(xyz, axis=0)
        # print("xyz_min", xyz_min)
        # print('sign', np.sign(xyz_min))
        # xyz_scale=np.where(np.sign(xyz_min)<0, 1.1, 0.9)
        # print("scale", xyz_scale)
        if args.tight:
            pass
        else:   
            xyz=xyz-xyz_min+0.2*cell_side_ind
        # print("xyz", xyz)
        # print("why", np.shape(np.where(xyz>cell_side_ind)))
        # if np.shape(np.where(xyz>cell_side_ind))[1]!=0:
            # print("where", np.where(xyz>cell_side_ind))
        # if np.shape(np.where(xyz<0))[1]!=0:
            # print("where negative", np.where(xyz<0))

        geom_i[:, -3:]=xyz+np.array(box_inds)*cell_side_ind
        #print("11 ", ind_geom[0,0])
        #print("post spin", xyz)
        
        # Use i to scale up appropriate bond, angle, dihedral, etc. types.
        atom_inc=i*len_dict["Atoms"]
        bond_inc=i*len_dict["Bonds"]
        angle_inc=i*len_dict["Angles"]
        dihed_inc=i*len_dict["Dihedrals"]
        for k in data_dict.keys():
            if k in ['Masses']: # These enumerate the types of interactions, so they don't need to be modified.
                pass
            else:
                if k == "Atoms":
                    temp=np.array(geom_i)
                    #print("before: ", temp[0, 0], geom_i[0,0], ind_geom[0,0])
                    temp[:,0]+=atom_inc         # Increment atom numbers
                    temp[:,1]+=i                # Increment mole numbers
                    #print("ai: ", atom_inc)
                    #print("after: ", temp[0, 0], geom_i[0,0], ind_geom[0,0])
                    #print(np.all(temp==geom_i))
                    #print(temp)
                elif k == "Bonds":
                    temp=np.array(data_dict[k]).astype(int)
                    temp[:,0]+=bond_inc
                    temp[:,2]+=atom_inc
                    temp[:,3]+=atom_inc
                elif k == "Angles":
                    temp=np.array(data_dict[k]).astype(int)
                    temp[:,0]+=angle_inc
                    temp[:,2]+=atom_inc
                    temp[:,3]+=atom_inc
                    temp[:,4]+=atom_inc

                elif k == "Dihedrals":
                    temp=np.array(data_dict[k]).astype(int)
                    temp[:,0]+=dihed_inc
                    temp[:,2]+=atom_inc
                    temp[:,3]+=atom_inc
                    temp[:,4]+=atom_inc
                    temp[:,5]+=atom_inc

                #print("temp {}".format(k), temp)
                final_dict[k][i*len_dict[k]:(i+1)*len_dict[k], :]=temp    
    #print("final_dict", final_dict)
    if args.rho:
        print(args.rho)
        print("final_dict", final_dict)
        mass_counts = {}
        for m in data_dict["Masses"]:
            mass_counts.update({m[0]:{"Mass":m[1], "Count":0}})
        print("mass counts before", mass_counts)
        for a in data_dict["Atoms"]:
            a_type = a[2]
            mass_counts[a_type]["Count"] = mass_counts[a_type]["Count"] + 1
        print("mass counts after", mass_counts)
        mol_mass = 0.0
        for m in mass_counts.keys():
            mol_mass += float(mass_counts[m]["Mass"])*mass_counts[m]["Count"]
        print(mol_mass)
        sys_mass = mol_mass * args.N
        sys_vol = cell_side_ind[0]*side[0]*cell_side_ind[1]*side[1]*cell_side_ind[2]*side[2]
        sys_rho = sys_mass/sys_vol *1.66
        print("default rho value: ", sys_rho)
        rho_ratio = sys_rho/args.rho # multiply this by dims to fix density!
        print("rho ratio", rho_ratio)
        print("Old cell dims: ", cell_side_ind)
        cell_side_ind = cell_side_ind * rho_ratio**(1/3)
        print("New cell dims: ", cell_side_ind)
        final_dict["Atoms"][:, -3:] = final_dict["Atoms"][:, -3:]*rho_ratio**(1/3)
        new_vol = cell_side_ind[0]*side[0]*cell_side_ind[1]*side[1]*cell_side_ind[2]*side[2]
        new_rho = sys_mass/new_vol *1.66
        print("new rho value: ", new_rho)
        



    # write into new data file
    with open(args.output+"_{}.data".format(args.orient), 'w') as out:
        out.write(" ".join(header[0]))
        out.write("\n\n{} atoms\n".format(len_dict["Atoms"]*args.N))
        out.write(" ".join(header[2])+"\n")
        out.write("{} bonds\n".format(len_dict["Bonds"]*args.N))
        out.write(" ".join(header[4])+"\n")
        out.write("{} angles\n".format(len_dict["Angles"]*args.N))
        out.write(" ".join(header[6])+"\n")
        if args.dihed:  # if dihedrals are requested, add them to the file.
            out.write("{} dihedrals\n".format(len_dict["Dihedrals"]*args.N))
            out.write(" ".join(header[8])+"\n\n")
        else:
            out.write("0 dihedrals\n")
            out.write("0 dihedral types\n\n")
        out.write("0.0 {} xlo xhi\n".format(cell_side_ind[0]*side[0]))
        out.write("0.0 {} ylo yhi\n".format(cell_side_ind[1]*side[1]))
        out.write("0.0 {} zlo zhi\n\n".format(cell_side_ind[2]*side[2]))
        for k in final_dict.keys():
            if args.dihed == False and k == "Dihedrals": # If we don't want dihedrals we can just skip the whole for loop.
                pass
            else:
                out.write("{}\n\n".format(k))
                for j in final_dict[k]:
                    if k in ['Masses']: 
                        out.write("{}\n".format("\t\t".join([str(i) for i in j])))
                    elif k == "Atoms":
                        out.write("{}".format("\t\t".join([str(int(i)) for i in j[0:3]])))
                        out.write("\t\t")
                        out.write("{}\n".format("\t\t".join([str(i) for i in j[3:]])))
                    elif args.dihed == False and k == "Dihedrals":  # don't write dihedrals if they are not requested.
                        pass
                    else:
                        out.write("{}\n".format("\t\t".join([str(int(i)) for i in j])))
                out.write("\n")
    out.close()

    # Write extended map file if requested.
    if args.map:
        extend_map(args.map, args.N, output=args.orient)
    

    # write init file with appropriate settings.

# rotate about a given axis at a random angle
def euler_rotate(xyz, axis="z"):   
    x=xyz[:, 0]
    y=xyz[:, 1]
    z=xyz[:, 2]
    theta=random.randint(0, 360)*(2*math.pi/360)    # generate random angle (in radians)
    if axis=="x":
        x1=x
        y1=y*np.cos(theta)-z*np.sin(theta)
        z1=y*np.sin(theta)+z*np.cos(theta)
    elif axis=="y":
        x1=x*np.cos(theta)+z*np.sin(theta)
        y1=y
        z1=-x*np.sin(theta)+z*np.cos(theta)
    elif axis=="z":
        x1=x*np.cos(theta)-y*np.sin(theta)
        y1=x*np.sin(theta)+y*np.cos(theta)
        z1=z
    xyz1=np.zeros(np.shape(xyz))
    xyz1[:,0]=x1
    xyz1[:,1]=y1
    xyz1[:,2]=z1
    return xyz1

# Rotate vector to face in line with a given axis. The "facing" direction will be based on the longest dimension of the molecule.
def axis_align(xyz, axis="z"):
    # Define vector that describes the longest dimension of the molecule...
    print("\n~~~~~~~~~~~~~~~~NEW MOLECULE~~~~~~~~~~~~~~~~\n")
    xyz_mins = np.min(xyz, axis=0)
    xyz_maxs = np.max(xyz, axis=0)
    extreme_dist = xyz_maxs-xyz_mins
    major_axis_ind = np.where(extreme_dist==np.max(extreme_dist))[0][0]
    # print("mai", major_axis_ind)
    # print("xmax", xyz_maxs)
    max_atom_ind=np.where(xyz==xyz_maxs[major_axis_ind])[0][0]
    min_atom_ind=np.where(xyz[:, major_axis_ind]==xyz_mins[major_axis_ind])[0][0]
    # min_atom_ind=np.where(np.where(xyz==xyz_mins[major_axis_ind])[1]==major_axis_ind)
    max_atom=xyz[max_atom_ind, :]
    min_atom=xyz[min_atom_ind, :]
    align_vec = max_atom-min_atom
    pre_mag=np.linalg.norm(align_vec)
    # print("max",max_atom)
    # print("min",min_atom)
    # print("av", align_vec)
    align_vec_norm = align_vec/np.linalg.norm(align_vec)
    print("org direction", align_vec_norm)
    x=xyz[:, 0]
    y=xyz[:, 1]
    z=xyz[:, 2]
    if axis=="x":
        target=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) # Zero out z to align with y, zero out y to align with x
        plane=["yz", "xy"]
        theta_signs=[-np.sign(align_vec_norm[2]), -1.0, -np.sign(align_vec_norm[1])]
    elif axis=="y":
        target=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) # Zero out x to align with z, zero out z to align with y
        plane=["xz", "yz"]
        theta_signs=[-np.sign(align_vec_norm[2]), -np.sign(align_vec_norm[0]), -1.0]
    elif axis=="z":
        target=np.array([[1.0,0.0,0.0], [0.0, 0.0, 1.0]])   # Zero out y to align with x, zero out x to align with z
        plane=["xy", "xz"]
        theta_signs=[-1.0, -np.sign(align_vec_norm[0]), -np.sign(align_vec_norm[1])]
    for r in [0, 1]:
        if plane[r]=="yz":   # Rotate around yz plane, not to be done for aligning with x axis
            proj_vec=np.array([0.0, align_vec[1], align_vec[2]])
            proj_vec=proj_vec/np.linalg.norm(proj_vec)
            theta = theta_signs[0]*np.arccos(np.dot(target[r,:], proj_vec))
            # if align_vec[1] < 0.0 and align_vec[2] < 0.0:
            #     theta += theta_signs[0]*np.pi/2
            # if align_vec[1]*align_vec[2] < 0.0:
            #     theta = theta_signs[2]*np.pi - theta
            x1=x.copy()
            y1=y*np.cos(theta)-z*np.sin(theta)
            z1=y*np.sin(theta)+z*np.cos(theta)
            # x,y,z=x1,y1,z1
            x=x1.copy()
            y=y1.copy()
            z=z1.copy()
            xyz_align=np.zeros(np.shape(xyz))
            xyz_align[:,0]=x
            xyz_align[:,1]=y
            xyz_align[:,2]=z
            post_dir=xyz_align[max_atom_ind, :]-xyz_align[min_atom_ind, :]
            print("x direction", post_dir/np.linalg.norm(post_dir))
        if plane[r]=="xz":   # Rotate around xz plane, not to be done for aligning with y axis
            proj_vec=np.array([align_vec[0], 0.0, align_vec[2]])
            proj_vec=proj_vec/np.linalg.norm(proj_vec)
            print(np.linalg.norm(proj_vec))
            theta = theta_signs[1]*np.arccos(np.dot(target[r,:], proj_vec))
            # if align_vec[0] < 0.0 and align_vec[2] < 0.0:
            #     theta += theta_signs[1]*np.pi/2
            # if align_vec[0]*align_vec[2] < 0.0:
            #     theta = theta_signs[2]*np.pi - theta
            x1=x*np.cos(theta)+z*np.sin(theta)
            y1=y.copy()
            z1=-x*np.sin(theta)+z*np.cos(theta)
            # x,y,z=x1,y1,z1
            x=x1.copy()
            y=y1.copy()
            z=z1.copy()
            xyz_align=np.zeros(np.shape(xyz))
            xyz_align[:,0]=x
            xyz_align[:,1]=y
            xyz_align[:,2]=z
            post_dir=xyz_align[max_atom_ind, :]-xyz_align[min_atom_ind, :]
            print("y direction", post_dir/np.linalg.norm(post_dir))
        if plane[r]=="xy":   # Rotate around xy plane, not to be done for aligning with z axis
            proj_vec=np.array([align_vec[0], align_vec[1], 0.0])
            proj_vec=proj_vec/np.linalg.norm(proj_vec)
            print(theta_signs[0])
            print(np.arccos(np.dot(target[r,:], proj_vec)))
            theta = theta_signs[0]*np.arccos(np.dot(target[r,:], proj_vec))
            print(theta)
            print(theta_signs)
            # if align_vec[1] < 0.0 and align_vec[0] < 0.0:
            #     theta += theta_signs[2]*np.pi/2
            # if align_vec[1]*align_vec[0] < 0.0:
            #     theta = theta_signs[2]*np.pi - theta
            x1=x*np.cos(theta)-y*np.sin(theta)
            y1=x*np.sin(theta)+y*np.cos(theta)
            z1=z.copy()
            #x,y,z=x1,y1,z1
            x=x1.copy()
            y=y1.copy()
            z=z1.copy()
            xyz_align=np.zeros(np.shape(xyz))
            xyz_align[:,0]=x
            xyz_align[:,1]=y
            xyz_align[:,2]=z
            post_dir=xyz_align[max_atom_ind, :]-xyz_align[min_atom_ind, :]
            print("z direction", post_dir/np.linalg.norm(post_dir))
        print("Theta: ", theta)
        print("Plane: ", plane[r])
        align_vec=post_dir.copy()
        post_dir_norm = post_dir/np.linalg.norm(post_dir)
        post_dir_norm = np.where(post_dir_norm==0.0, 1.0, post_dir_norm)
        post_dir_norm = np.where(post_dir_norm==-0.0, 1.0, post_dir_norm)
        if axis=="x":
            theta_signs=[-np.sign(post_dir_norm[2]), -1.0, -np.sign(post_dir_norm[1])]
        elif axis=="y":
            theta_signs=[-np.sign(post_dir_norm[2]), -np.sign(post_dir_norm[0]), -1.0]
        elif axis=="z":
            theta_signs=[-1.0, -np.sign(post_dir_norm[0]), -np.sign(post_dir_norm[1])]
    xyz_align=np.zeros(np.shape(xyz))
    xyz_align[:,0]=x
    xyz_align[:,1]=y
    xyz_align[:,2]=z
    post_dir=xyz_align[max_atom_ind, :]-xyz_align[min_atom_ind, :]
    print("final direction", post_dir/np.linalg.norm(post_dir))
    # post_mag=np.linalg.norm(xyz_align[max_atom_ind, :]-xyz_align[min_atom_ind, :])
    # print("pre magnitude", pre_mag)
    # print("post magnitude", post_mag)
    return xyz_align

    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
