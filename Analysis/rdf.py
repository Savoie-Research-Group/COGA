# Written by Jack Yungbluth (jyungblu@purdue.edu). March 7, 2022
from mol_classes import *
from data_file_parser import parse_data_file
from frame_generator import frame_generator

import sys, argparse, scipy
import scipy.spatial
import time

def main(argv):
    parser = argparse.ArgumentParser(description='Calculates rdf based on given groups from a LAMMPS trajectory or data file for 3D trajectories.')

    parser.add_argument('traj_file', type=str,
                        help='Name of LAMMPS trajectory file to compute RDF over.')

    parser.add_argument('data_file', type=str,
                        help='Name of LAMMPS data file.')

    parser.add_argument('group1',
                        help='Specify the identity of the first group: atom type, element. Also accepts a space-separated string given type(s) ("1 4" for atom types 1 and 4.')

    parser.add_argument('group2',
                        help='Specify the identity of the second group: atom type, element. Also accepts a space-separated string given type(s) ("1 4" for atom types 1 and 4.')

    parser.add_argument('group1_type',
                        help='Specify the type for group 1. Options: type (atom type, #), element (letter), com (center of mass). Also accepts a space-separated string given type(s) ("1 4" for atom types 1 and 4.')

    parser.add_argument('group2_type',
                        help='Specify the type for group 2. Options: type (atom type, #), element (letter), com (center of mass). Also accepts a space-separated string given type(s) ("1 4" for atom types 1 and 4.')

    parser.add_argument('-o', dest='output', type=str, default='out',
                        help='Filename of output. Extension will be ".rdf" Default=out')

    parser.add_argument('-r_max', dest='r_max', type=float, default=15.0,
                        help='Maximum distance to consider when computing rdf. Default=15.0')

    parser.add_argument('-f_start', dest='f_start', type=int, default=0,
                        help='Frame to begin calculation of rdf. Default=0')

    parser.add_argument('-f_end', dest='f_end', type=int, default=-1,
                        help='Frame to finish calculation of rdf. Default=-1 (Final frame of the trajectory)')

    parser.add_argument('-f_every', dest='f_every', type=int, default=1,
                        help='Frequency of frames to parse for rdf. Default=1.')

    parser.add_argument('-width', dest='bin_width', type=float, default=0.01,
                        help='Width of bin for calculation of rdf. Default=0.01')

    parser.add_argument('-atom_style', dest='atom_style', type=str, default='full',
                        help='LAMMPS atom_style in data file')

    parser.add_argument('-bond_sep', dest='bond_sep', type=int, default=4,
                        help='Do not include 1-2, 1-3,..., 1-bond_sep neighbors while calculating rdf. bond_sep of -1 corresponds to infinite bond separation, and will exclude all atoms in the same molecule. Default=4')

    args = parser.parse_args()

    traj_file = args.traj_file
    data_file = args.data_file

    group1 = args.group1.split()
    group2 = args.group2.split()
    group1_types = args.group1_type.split()
    group2_types = args.group2_type.split()
    acceptable_types = ['type', 'element']
    for count_i, i in enumerate(group1_types):
        if i not in acceptable_types:
            print('Group1_type must be in {}. Quitting...'.format(acceptable_types))
            quit()
        if i == 'type':
            group1[count_i] = int(group1[count_i])
    for count_i, i in enumerate(group2_types):
        if i not in acceptable_types:
            print('Group2_type must be in {}. Quitting...'.format(acceptable_types))
            quit()
        if i == 'type':
            group2[count_i] = int(group2[count_i])
    if len(group1) != len(group2):
        print('Group1 and group2 must be same size. Quitting...')
        quit()
    if len(group1_types) != len(group2_types):
        print('Group1_type and group2_type must be same size. Quitting...')
        quit()
    if len(group1) != len(group1_types):
        print('Groups must be same length as group_types. Quitting')
        quit()

    # Parse ids of groups of interest from data file
    atoms,bonds,angles,dihedrals,impropers,box,adj_mat,extra_prop = parse_data_file(data_file, args.atom_style) # uncomment when impropers class is added
    #atoms, bonds, angles, dihedrals, box, extra_prop = parse_data_file(data_file, args.atom_style)

    # Build adjacency matrix and list
    adj_list = [[] for _ in range(len(atoms.ids))]
    adj_mat = np.zeros((len(atoms.ids),len(atoms.ids)))
    for b1, b2 in bonds.atom_ids:
        adj_list[b1 - 1].append(b2 - 1)
        adj_list[b2 - 1].append(b1 - 1)
        adj_mat[b1 - 1, b2 - 1] = 1
        adj_mat[b2 - 1, b1 - 1] = 1

    # if max_box > r_max, set r_max to max_box/2, print notification of this change
    r_max = args.r_max
    # Find shortest dimension
    if len(box) == 4: # Non cubic - setup volume for shortest dimension and transformation matrices for computing distances later
        # For info about lammps tilt factors and the transformation matrix: https://docs.lammps.org/Howto_triclinic.html
        lx,ly,lz = [box[i][1] - box[i][0] for i in range(3)]
        xy,xz,yz = box[3]
        a = lx
        b = np.sqrt(ly**2+xy**2)
        c = np.sqrt(lz**2+xz**2+yz**2)
        cosalpha = (xy*xz + ly*yz) / (b*c)
        cosbeta = xz/c
        cosgamma = xy/b
        singamma = np.sin(np.arccos(cosgamma))

        volume = np.sqrt(1.0 - cosalpha ** 2 - cosbeta ** 2 - cosgamma ** 2 + 2.0 * cosalpha * cosbeta * cosgamma)

        frac2cart = np.matrix([
            [a, b * cosgamma, c * cosbeta],
            [0, b * singamma, (c * (cosalpha - cosbeta * cosgamma) / singamma)],
            [0, 0, c * volume / singamma]]).T

        cart2frac = np.matrix([
            [1. / a, -cosgamma / (a * singamma), (cosalpha * cosgamma - cosbeta) / (a * volume * singamma)],
            [0, 1 / (b * singamma), (cosbeta * cosgamma - cosalpha) / (b * volume * singamma)],
            [0, 0, singamma / (c * volume)]]).T

        # Minimum image convention for non-cubic cells: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696, page 6
        smallest_max_box_length = np.min(np.abs([a*np.sin(np.arccos(cosgamma)), b*np.sin(np.arccos(cosalpha)), c*np.sin(np.arccos(cosbeta))])) / 2
        if r_max == 0 or r_max > smallest_max_box_length:
            r_max = smallest_max_box_length
            print('rdf_r_max specified at 0 or a value greater than half the possible maximum box distance. rdf_r_max now set at: {}'.format(r_max))

    else:
        smallest_max_box_length = sorted([box[i][1] - box[i][0] for i in range(len(box))])[0] / 2
        if r_max == 0 or r_max > smallest_max_box_length:
            r_max = smallest_max_box_length
            print('rdf_r_max specified at 0 or a value greater than half the possible maximum box distance. rdf_r_max now set at: {}'.format(r_max))

    bins = np.arange(0.0,r_max,step=args.bin_width)
    rdf = np.zeros((len(group1),len(bins)-1)) # bin includes rightmost edge, which we remove here
    # Initialize shell volumes for the normalizations
    volumes = np.zeros(len(bins)-1)
    for i in range(len(volumes)):
        volumes[i] = 4.0/3.0*np.pi * ((float(i + 1) * args.bin_width)**3.0 - (float(i) * args.bin_width)**3.0)
    pairs = np.zeros(len(group1))

    # Discard distances on the same molecule (Modify with bond_sep)
    excl_mat_list = []

    if args.bond_sep == -1:
        for g in range(len(group1)):
            if group1_types[g] == 'type':
                if group2_types[g] == 'type':
                    excl_mat = np.tile(atoms.mol_id[atoms.lammps_type==group1[g]][:,np.newaxis],(1,len(atoms.mol_id[atoms.lammps_type==group2[g]]))) == np.tile(atoms.mol_id[atoms.lammps_type==group2[g]],(len(atoms.mol_id[atoms.lammps_type==group1[g]]),1))
                elif group2_types[g] == 'element':
                    excl_mat = np.tile(atoms.mol_id[atoms.lammps_type==group1[g]][:,np.newaxis],(1,len(atoms.mol_id[atoms.element==group2[g]]))) == np.tile(atoms.mol_id[atoms.element==group2[g]],(len(atoms.mol_id[atoms.lammps_type==group1[g]]),1))
            elif group1_types[g] == 'element':
                if group2_types[g] == 'type':
                    excl_mat = np.tile(atoms.mol_id[atoms.element==group1[g]][:,np.newaxis],(1,len(atoms.mol_id[atoms.lammps_type==group2[g]]))) == np.tile(atoms.mol_id[atoms.lammps_type==group2[g]],(len(atoms.mol_id[atoms.element==group1[g]]),1))
                elif group2_types[g] == 'element':
                    excl_mat = np.tile(atoms.mol_id[atoms.element==group1[g]][:,np.newaxis],(1,len(atoms.mol_id[atoms.element==group2[g]]))) == np.tile(atoms.mol_id[atoms.element==group2[g]],(len(atoms.mol_id[atoms.element==group1[g]]),1))

            excl_mat_list.append(excl_mat)

    elif args.bond_sep > 0:
        pathways_mat = adj_mat + np.eye(*adj_mat.shape)
        # Compute pathway matrix
        for sep in range(args.bond_sep-1):
            pathways_mat = np.dot(pathways_mat,adj_mat) + pathways_mat

        # Exclusion matrix finds atom types or elements within bond_sep to exclude these distances from the rdf
        # The & operator flattens the matrix, so must reshape into 2D
        for g in range(len(group1)):
            if group1_types[g] == 'type':
                if group2_types[g] == 'type':
                    excl_mat = np.reshape(pathways_mat[np.tile(atoms.lammps_type[:, np.newaxis] == group1[g], (1, adj_mat.shape[1])) & np.tile(atoms.lammps_type == group2[g], (adj_mat.shape[0], 1))], (np.sum(atoms.lammps_type==group1[g]), np.sum(atoms.lammps_type==group2[g])))
                elif group2_types[g] == 'element':
                    excl_mat = np.reshape(pathways_mat[np.tile(atoms.lammps_type[:, np.newaxis] == group1[g], (1, adj_mat.shape[1])) & np.tile(atoms.element == group2[g], (adj_mat.shape[0], 1))], (np.sum(atoms.lammps_type==group1[g]), np.sum(atoms.element==group2[g])))
            elif group1_types[g] == 'element':
                if group2_types[g] == 'type':
                    excl_mat = np.reshape(pathways_mat[np.tile(atoms.element[:, np.newaxis] == group1[g], (1, adj_mat.shape[1])) & np.tile(atoms.lammps_type == group2[g], (adj_mat.shape[0], 1))], (np.sum(atoms.element==group1[g]), np.sum(atoms.lammps_type==group2[g])))
                elif group2_types[g] == 'element':
                    excl_mat = np.reshape(pathways_mat[np.tile(atoms.element[:, np.newaxis] == group1[g], (1, adj_mat.shape[1])) & np.tile(atoms.element == group2[g], (adj_mat.shape[0], 1))], (np.sum(atoms.element==group1[g]), np.sum(atoms.element==group2[g])))

            excl_mat_list.append(excl_mat)

    for atomlist, timestep, frame_box in frame_generator(traj_file, start=args.f_start, end=args.f_end, every=args.f_every, unwrap=False, adj_list=None, return_prop=False):
        # Find relevant groups
        for g in range(len(group1)):
            # Group 1
            if group1_types[g] == 'type':
                x1 = atomlist.x[atoms.lammps_type == group1[g]]
                y1 = atomlist.y[atoms.lammps_type == group1[g]]
                z1 = atomlist.z[atoms.lammps_type == group1[g]]
            elif group1_types[g] == 'element':
                x1 = atomlist.x[atoms.element == group1[g]]
                y1 = atomlist.y[atoms.element == group1[g]]
                z1 = atomlist.z[atoms.element == group1[g]]

            # Group 2
            if group2_types[g] == 'type':
                group2[g] = int(group2[g])
                x2 = atomlist.x[atoms.lammps_type == group2[g]]
                y2 = atomlist.y[atoms.lammps_type == group2[g]]
                z2 = atomlist.z[atoms.lammps_type == group2[g]]
            elif group2_types[g] == 'element':
                x2 = atomlist.x[atoms.element == group2[g]]
                y2 = atomlist.y[atoms.element == group2[g]]
                z2 = atomlist.z[atoms.element == group2[g]]

            # Compute distances between relevant groups
            if len(box) == 4:
                dists = find_dist_triclinic(np.vstack((x1, y1, z1)).T, np.vstack((x2, y2, z2)).T, frac2cart, cart2frac)
                if group1_types[g] == group2_types[g] and group1[g] == group2[g]:  # Same list - can use pdist
                    pair_ratio = 2      # Setting up pair ratio to avoid discrepancies in double counting between like and non-like rdfs.
                else:
                    pair_ratio = 1
            elif group1_types[g] == group2_types[g] and group1[g] == group2[g]:  # Same list - can use pdist
                dists = find_dist_same(np.vstack((x1, y1, z1)).T, box)
                pair_ratio = 2
            else:
                dists = find_dist_diff(np.vstack((x1, y1, z1)).T, np.vstack((x2, y2, z2)).T, box)
                pair_ratio = 1

            if args.bond_sep != 0:
                dists[excl_mat_list[g] > 0] = 0

            # Bin distances to rdf
            rdf_hist, rdf_bins = np.histogram(dists[(dists>0) & (dists<=r_max)], bins=bins)
            pairs[g] += (dists.shape[0]*dists.shape[1]-np.count_nonzero(dists==0.0))/pair_ratio

            frame_volume = np.prod(frame_box[:,1]-frame_box[:,0])

            rdf[g,:] += rdf_hist / pair_ratio * frame_volume # Dividing by 2 because non-self pairs are double-counted. DF: This is only true for rdfs of one type with itself! Necessarily can't double count for different types.

    # Norm bins by their volume and number of pairs
    bins = (bins - args.bin_width/2)[1:] # Move from right-edged bins to center of bins
    for g in range(len(group1)):
        rdf[g,:] = rdf[g,:] / (volumes * pairs[g])

    # Write rdf to output
    for g in range(len(group1)):
        write_cols(args.output+"_{}_{}.rdf".format(group1[g],group2[g]), [bins,rdf[g,:]], ["r", "rdf"])

    print("rdf.py completed successfully!")
    del(dists)
    del(x1)
    del(x2)
    del(y1)
    del(y2)
    del(z1)
    del(z2)
    quit()


# Computes the pairwise distance between particles in a list
# Used if computing an A-A rdf (twice as fast as A-B case)
def find_dist_same(geo,box):

    rs = np.zeros((geo.shape[0], geo.shape[0]))

    for i in range(3):
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(geo[:,i:i+1], 'minkowski', p=1.0))
        l, l2 = box[i][1] - box[i][0], (box[i][1] - box[i][0]) / 2.0
        while not (dist <= l2).all():
            dist -= l * (dist > l2)
            dist = np.abs(dist)

        rs += dist**2

    rs = np.sqrt(rs)
    return rs


# Computes distances between particles of 2 different lists
# Used if computing an A-B rdf
def find_dist_diff(a, b, box):

    rs = np.zeros((a.shape[0],b.shape[0]))

    for i in range(3):
        dist = scipy.spatial.distance.cdist(a[:,i:i+1], b[:,i:i+1], 'minkowski', p=1.0)
        l, l2 = box[i][1] - box[i][0], (box[i][1] - box[i][0]) / 2.0
        while not (dist <= l2).all():
            dist -= l * (dist > l2)
            dist = np.abs(dist)

        rs += dist**2

    rs = np.sqrt(rs)
    return rs


# Computes distances between particles in a list when box is non-cubic
# Determines distance vector between each particle pair in fractional space, then transform all distance vectors to real space and find the lengths of these vectors
# Might not work for all possible unit cells - I'm not sure if it is always true that the shortest route in fractional space is the shortest route in cartesian space
def find_dist_triclinic(a, b, f2c, c2f):

    # Transform data to fractional space
    frac_geo_a = np.dot(a,c2f)
    frac_geo_b = np.dot(b,c2f)

    # Sign for each component matters in this case. It is not in general true that norm([0.1,0.1,0.1]*T) = norm([-0.1,0.1,0.1]*T) where T is the transformation matrix
    # Fractional box is [(0,1), (0,1), (0,1)] - abs value of all distances should be less than 0.5
    x_coord = np.tile(np.array(frac_geo_a[:, 0]), (1, frac_geo_b.shape[0])) - np.tile(np.array(frac_geo_b[:, 0]).T, (frac_geo_a.shape[0], 1))
    x_coord[x_coord < -0.5] += 1.0
    x_coord[x_coord > 0.5] -= 1.0
    y_coord = np.tile(np.array(frac_geo_a[:, 1]), (1, frac_geo_b.shape[0])) - np.tile(np.array(frac_geo_b[:, 1]).T, (frac_geo_a.shape[0], 1))
    y_coord[y_coord < -0.5] += 1.0
    y_coord[y_coord > 0.5] -= 1.0
    z_coord = np.tile(np.array(frac_geo_a[:, 2]), (1, frac_geo_b.shape[0])) - np.tile(np.array(frac_geo_b[:, 2]).T, (frac_geo_a.shape[0], 1))
    z_coord[z_coord < -0.5] += 1.0
    z_coord[z_coord > 0.5] -= 1.0

    # reshape to flattened
    x_coord = x_coord.ravel()
    y_coord = y_coord.ravel()
    z_coord = z_coord.ravel()

    disp_vecs_frac = np.stack([x_coord,y_coord,z_coord], axis=1)
    disp_vecs_cart = np.dot(disp_vecs_frac, f2c)

    rs = np.reshape(scipy.linalg.norm(disp_vecs_cart, axis=1), (a.shape[0],b.shape[0]))

    return rs


def write_cols(name, values, labels, comment=None):
    # Check inputs
    if len(values) != len(labels):
        print("ERROR in write_cols: lengths of 'values' and 'labels' lists must be equal.")
        quit()
    # Write file
    with open(name, 'w') as f:
        if comment is not None:
            f.write("{}\n".format(comment))
        f.write(" " + " ".join(["{:<40s}".format(i) for i in labels]) + "\n")
        for i in range(len(values[0])):
            f.write(" ".join(["{:< 40.12f}".format(values[j][i]) for j in range(len(values))]) + "\n")
    f.close()
    return


if __name__ == '__main__':
    main(sys.argv[1:])
