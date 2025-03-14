#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
import math
#sys.path.append('/home/ddfortne/bin/taffi/CG')
#sys.path.append('/home/ddfortne/bin/taffi_beta/Lib/')
import subprocess as sp
#sys.path.append('/home/ddfortne/bin/taffi_beta/FF_functions/')
import fileinput as FI

script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)
subfolders = ["/Analysis", "/Input_Operations", "/Plotting", "/Job_Submission"]
for subf in subfolders:
    sys.path.append(script_directory+subf)

def main(argv):
    parser = argparse.ArgumentParser(description='''Uses gen_periodic_md.py, among other functions, to generate simulation files for a CG crystal system.                                                                                                                                   
    Input: CGed cif file, CGed data file, CG settings file, and CG map file, as output by the CG_xyz.py function
                                                                                                                                                                                                      
    Output: A folder containing all necessary files to submit a CG crystal simulation. Simulations will print relaxation steps as well. Simulations using the NVT ensemble, 
        simulations without movement from the original provided positioning (static), and minimzation simulations can also be requested through this script.                                                                                                                                                                             
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('cif_file', type=str, help='String containing name of cif file.')
    parser.add_argument('data_file', type=str, help='String containing name of data file from gen_md_inputs_martini.py.')
    parser.add_argument('settings_file', type=str, help='String containing name of settings file from gen_md_inputs_martini.py.')
    parser.add_argument('map_file', type=str, help='String containing name of map file.')


    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='CG_periodic', help='Output file name. Default: CG')
    parser.add_argument('-dims',dest='dims', type=str, default='"3 3 3"', help='Crystal simulation dimensions. Default: 3 3 3')
    parser.add_argument('-T',dest="T", type=float, default=300.0, help='Temperature simulation will be run at.')
    parser.add_argument('-P', dest="P", type=str,default="5atm", help='Pressure simulation will be run at.')
    parser.add_argument('-t', dest='t_equil', default=1E6,
                        help = 'Controls the length of the MD equilibration segment. (default: 1E6)')
    parser.add_argument('-t_A', dest='t_anneal', default=1E6,
                        help = 'Controls the length of the MD equilibration segment. (default: 1E6)')
    parser.add_argument('-t_ext', dest='t_ext', default=1E6,
                        help = 'Controls the length of the MD extension job script. (default: 1E6)')
    parser.add_argument('-nodes',dest="nodes", type=int, default=2, help='Number of nodes to use in simulation.')
    parser.add_argument('-ppn',dest="ppn", type=int, default=40, help='Processors per node in simulation.')
    parser.add_argument('-walltime',dest="walltime", type=int, default=1440, help='Walltime in minutes.')
    parser.add_argument('-queue',dest="queue", type=str, default="bsavoie", help='queue to be used.')
    parser.add_argument('-lammps',dest="lammps", type=str, default='/depot/bsavoie/apps/lammps/exe//lmp_mpi_180501', help='lammps executable to be used.')
    parser.add_argument('-python', dest='python', type=str, default='python', help='Location/name of the python version to be used.')
    parser.add_argument('-mpirun', dest='mpirun', type=str, default='/apps/cent7/intel/impi/2017.1.132/bin64/mpirun', help='Location/name of the mpi to be used to run with your version of lammps')
    parser.add_argument('--NVT', dest='NVT_opt', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will run in the NVT ensemble. (default: False)')
    parser.add_argument('--static', dest='static_opt', default=False, action='store_const', const=True,
                        help = 'When present, the simulation will be static, ie no time steps will occur. (default: False)')
    parser.add_argument('--minrun', dest='minrun', default=False, action='store_const', const=True,
                        help = 'When present, energy minimization step will occur. (default: False)')
    parser.add_argument('--cell_sym', dest='cell_sym', default=False, action='store_const', const=True,
                        help = 'When present, cell symmetry calculations will be used for box generation. (default: False)')

    # Parse Arguments
    args = parser.parse_args()
    print("~~~~~~~~Beginning {}~~~~~~~~".format(args.output))
    # Get cif info:
    current_dir=os.getcwd()
    print("cif file used: ", current_dir+'/'+args.cif_file)
    a,b,c,alpha,beta,gamma=get_dims_from_cif(args.cif_file)
    # Rewrite data file for gen periodic:
    new_data=rewrite_data(args.data_file, [a,b,c,alpha,beta,gamma])
    # Get number of atoms per box:
    atom_num=get_atoms_from_data(args.data_file)
    atom_num=int(atom_num)
    gen_p = "{}/Input_Operations/gen_periodic_md.py".format(script_directory)
    if args.cell_sym:
        cell_sym="--cell_sym"
        #gen_p="~/bin/CG_Crystal/gen_periodic_mdv2.py"
    else:
        cell_sym=""
        #gen_p="~/bin/CG_Crystal/gen_periodic_md.py"
    print("cell dims in write_periodic_CG: '{} {} {} {} {} {}'".format(a,b,c,alpha,beta,gamma))
    if args.NVT_opt==False:
        sp.call('{} {} {} -single_map {} -data_file {} -settings_file {} -dims "{}" -cg_uc {} -cg_nuc {} -T {} -T_A {} -T_R {} -o {} -P {} -t {} -t_A {} -t_ext {} --cg {} --overwrite'\
        .format(args.python, gen_p, args.cif_file, args.map_file, new_data, args.settings_file, args.dims, "'{} {} {} {} {} {}'".format(a,b,c,alpha,beta,gamma),atom_num,args.T, args.T, args.T/2, args.output,args.P,args.t_equil, args.t_anneal, args.t_ext, cell_sym),shell=True)
    else:
        sp.call('{} {} {} -single_map {} -data_file {} -settings_file {} -dims "{}" -cg_uc {} -cg_nuc {} -T {} -T_A {} -T_R {} -o {} -P {} -t {} -t_A {} -t_ext {} --cg --NVT {} --overwrite'\
        .format(args.python, gen_p, args.cif_file, args.map_file, new_data, args.settings_file, args.dims, "'{} {} {} {} {} {}'".format(a,b,c,alpha,beta,gamma),atom_num,args.T, args.T, args.T/2, args.output,args.P,args.t_equil, args.t_anneal, args.t_ext, cell_sym),shell=True)
    current_dir=os.getcwd()
    os.chdir(current_dir+'/'+args.output)
    print('fixing files and writing submission script...')
    correct_files(args.output, args.static_opt, args.minrun)
    sp.call('rm *.bak', shell=True) # Remove backup files.
    sp.call('{} {}/Job_Submission/write_job_header.py $PWD {}.submit -job_name {} -nodes {} -ppn {} -walltime {} -queue {} -lammps {} -lammps_init {}.in.init -mpirun {} ; wait'.format(args.python, script_directory, args.output, args.output,args.nodes,args.ppn,args.walltime,args.queue,args.lammps,args.output, args.mpirun),shell=True)
    print('it worked!! Have a good day!')

def get_dims_from_cif(cif_file):
    # Read in cell information from cif
    with open(cif_file, 'r') as cif:
        for lines in cif:
            words=lines.split()
            if words==[]:
                pass
            elif words[0]=='_cell_length_a':
                a = words[1].split('(')[0]
            elif words[0]=='_cell_length_b':
                b = words[1].split('(')[0]
            elif words[0]=='_cell_length_c':
                c = words[1].split('(')[0]
            elif words[0]=='_cell_angle_alpha':
                alpha = words[1].split('(')[0]
            elif words[0]=='_cell_angle_beta':
                beta = words[1].split('(')[0]
            elif words[0]=='_cell_angle_gamma':
                gamma = words[1].split('(')[0]
                break
    return a,b,c,alpha,beta,gamma

def get_atoms_from_data(data_file):
    with open(data_file,'r') as data:
        for lines in data:
            words=lines.split()
            if words==[]:
                pass
            elif words[1]=='atoms':
                atom_num=words[0]
                return atom_num
def rewrite_data(data_file, cell_dims):
    new_data=data_file.split('.')[0]+"_triclinic.data"
    a=float(cell_dims[0]) # Get angles and convert to radians
    b=float(cell_dims[1])
    c=float(cell_dims[2])
    alpha=float(cell_dims[3])*math.pi/180.0 # Get angles and convert to radians
    beta=float(cell_dims[4])*math.pi/180.0 # Get angles and convert to radians
    gamma=float(cell_dims[5])*math.pi/180.0 # Get angles and convert to radians
    lx=a
    xy=b*math.cos(gamma)
    xz=c*math.cos(beta)
    ly=(b**(2.0)-xy**(2.0))**(0.5)
    yz=(b*c*math.cos(alpha)-xy*xz)/ly
    lz=(c**(2.0)-xz**(2.0)-yz**(2.0))**(0.5)
    # xlo=(-lx-xy-xz)/2
    # xhi=(lx-xy-xz)/2
    # ylo=(-ly-yz)/2
    # yhi=(ly-yz)/2
    # zlo=-lz/2
    # zhi=lz/2
    xlo=min(0.0, xy,xz,xy+xz)
    xhi=lx+max(0.0, xy,xz,xy+xz)
    ylo=min(0.0, yz)
    yhi=ly+min(0.0, yz)
    zlo=0.0
    zhi=lz
    with open(data_file,'r') as ref_file, open(new_data,'w') as new_file:
        for lines in ref_file:
            if lines.split()==[]:
                new_file.write(lines)
            elif lines.split()[-1]=="dihedrals\n":  #For no dihedrals!
                new_file.write('0 dihedrals')
            elif lines.split()[-1]=="types" and lines.split()[-2]=="dihedral":  # For no dihedrals!
                new_file.write('0 dihedral types\n')
            elif lines.split()[-1]=="xhi":
                new_file.write('{} {} xlo xhi\n'.format(xlo,xhi))#min(lx,0.0),max(lx,0.0)))
            elif lines.split()[-1]=="yhi":
                new_file.write('{} {} ylo yhi\n'.format(ylo,yhi))#min(ly,0.0),max(ly,0.0)))
            elif lines.split()[-1]=="zhi":
                new_file.write('{} {} zlo zhi\n'.format(zlo,zhi ))#min(lz,0.0),max(lz,0.0)))
                new_file.write('{} {} {} xy xz yz\n'.format(xy,xz,yz))
            else:
                new_file.write(lines)
    return new_data

# This function corrects the written data files by removing dihedrals, and editing lines that lead to inappropriate settings
def correct_files(output, static, minrun):
    settings=output+".in.settings"
    data=output+".data"
    init=output+".in.init"
    # Correct Settings file by removing all dihedral terms
    with FI.FileInput(settings, inplace=True, backup='.bak') as sett:
        DSFlag=False
        for line in sett:
            if line.split()==[]:
                print(line, end='')
            elif line.split()[1]=='Dihedral':
                DSFlag=True
                pass
            elif DSFlag==True:
                pass
            else:
                print(line, end='')
        sett.close()
    # Correct data file by removing dihedral terms and setting number of dihedrals to 0.
    with FI.FileInput(data, inplace=True, backup='.bak') as dat:
        for line in dat:
            if line.split()==[]:
                print(line, end='')
            elif len(line.split())==1:
                if line.split()[0]=='Dihedrals':
                    break
                else:
                    print(line, end='')
            elif line.split()[1]=='dihedral':
                print('0 dihedral types')
            elif line.split()[1]=='dihedrals':
                print('0 dihedrals')
            else:
                print(line, end='')
        dat.close()
    # Correct init file by changing style definitions, or by adding or removing simulation steps as desired.
    with FI.FileInput(init, inplace=True, backup='.bak') as init:
        for line in init:
            if line.split()==[]:
                print(line, end='')
            elif line.split()[0]=='pair_style':
                print('pair_style   hybrid lj/gromacs/coul/gromacs 9.0 12.0 0.0 12.0 # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)')
            elif line.split()[0]=='bond_style':
                print('bond_style   hybrid harmonic         # parameters needed: k_bond, r0')
            elif line.split()[0]=='angle_style':
                print('angle_style  hybrid harmonic         # parameters needed: k_theta, theta0')
            elif line.split()[0]=='kspace_style':
                pass
            elif static and line.split()[0]=='run': # Does static run if so desired.
                print('run 0')
            elif minrun and line.split()[0]=='fix': # Does minimization run if so desired.
                print(line, end='')
                print('''\
#===========================================================
#   MINIMIZATION
#===========================================================
minimize 0.0 1.0e-8 10000 1000000

velocity        all create ${RELAX_TEMP} ${vseed} mom yes rot yes     # DRAW VELOCITIES
fix minrun all nve/limit 0.01
dump minrun all custom 1 minimize.lammpstrj id type x y z
dump_modify minrun sort  id
run             0
unfix minrun
undump minrun

# WRITE RESTART FILES, CLEANUP, AND EXIT
write_data      minimize.end.data pair ii
unfix		averages''')
                break
            elif line.split()[0]=='fix' and line.split()[1]=='relax': # Output trajectory from relax run.
                print(line, end='')
                print('''dump relax all custom ${coords_freq} relax.lammpstrj id type x y z 
dump_modify relax sort  id''')
            else:
                print(line, end='')

    # Correct init file




if __name__ == '__main__':
    main(sys.argv[1:])