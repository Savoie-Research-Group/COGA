#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
sys.path.append('/home/ddfortne/bin/taffi/CG')
from file_operations import read_xyz
from post_process import find_bonds, find_angles, find_dihedrals
sys.path.append('/home/ddfortne/bin/taffi_beta/Lib/')
from adjacency import Table_generator
from id_types import id_types

def main(argv):
    parser = argparse.ArgumentParser(description='''Enter description here                                                                                                                                           
    Input: xyz file to make dummy db file
                                                                                                                                                                                                      
    Output: FF file with fake terms in it as something to supply to some other process.                                                                                                                                                           
                                                                                                                                                                                                            
    Assumptions: 
    ''')

    # Required Arguments
    parser.add_argument('xyz', type=str, help='xyz file to make dummy db file for.')

    # Optional Arguments
    parser.add_argument('-o', dest='output', type=str, default='dummyFF', help='Output. Default:dummyFF ')
        
    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    print("Finding geometry...")
    elements, geometry, charge=read_xyz(args.xyz)
    Adj_mat=Table_generator(elements, geometry)
    all_ele=id_types(elements, Adj_mat)
    uni_ele=list(dict.fromkeys(all_ele))
    bond_id, bond_type=find_bonds(Adj_mat, all_ele)
    angle_id, angle_type=find_angles(Adj_mat, all_ele)
    dihedral_id, dihedral_type=find_dihedrals(Adj_mat, all_ele)
    #print(bond_id,angle_id)
    #print(bond_type,angle_type)
    # Writes FF files
    print("Writing FF files...")
    with open('{}.db'.format(args.output),'w') as f:
        f.write('# Atom type definitions\n#\n#  Atom_type   Label   Mass    Mol_ID\n')
        for i in range(len(uni_ele)):
            f.write('atom   {}  {}  72.000000\n'.format(uni_ele[i],uni_ele[i]))
        #f.write('\n# VDW definitions\n#\n#  Atom_type   Atom_type   Potential   params\n')
        #for j in range(len(param_array)):
        #    f.write('{} {} {}  {}  {}  {}\n'.format('vdw',param_array[j][0],param_array[j][1],'lj',param_array[j][2],param_array[j][3]))
        f.write('\n# Bond type definitions\n#\n# Atom_type   Atom_type   style   params (k, r0)  Mol_ID\n')
        for k in range(len(bond_type)):
            f.write('bond   {}  {}  harmonic    3.0 5.0\n'.format(bond_type[k][0],bond_type[k][1]))
        f.write('\n# Angle type definitions\n#\n# Atom_type   Atom_type   Atom_type  style   params (k, theta0)  Mol_ID\n')
        for l in range(len(angle_type)):
            f.write('angle  {}  {}  {}  harmonic    3.0 180.0\n'.format(angle_type[l][0],angle_type[l][1],angle_type[l][2]))
        f.write('\n# Dihedral type definitions\n# Atom_type   Atom_type   Atom_type   Atom_type   style   params (k, r0)  Mol_ID\n')
        for m in range(len(dihedral_type)):
            f.write('torsion    {}  {}  {}  {}  opls    0.0 0.0 0.0 0.0\n'.format(dihedral_type[m][0],dihedral_type[m][1],dihedral_type[m][2],dihedral_type[m][3]))
    print("Successfully wrote a bad FF File! Enjoy!")
    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])