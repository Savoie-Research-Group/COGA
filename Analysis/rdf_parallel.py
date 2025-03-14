#!/bin/env python
# Author: Aditi Khot (akhot@purdue.edu)

import numpy as np
import os, argparse, sys
import subprocess as sp
from rdf import write_cols
from math import *

def main(argv):

    """Main driver for parallel RDF calculation.
    

    Required Arguments
    ------------------
    LAMMPS trajectory file: str
    LAMMPS data file: str
    group1: str
    group2: str
    group1 type: str
    group2 type: str
    
    Required Modules:
    -----------------
    1. mol_classes.py: Defines the classes.
    2. data_parser.py: Reads data file.
    3. frame_generator.py: Reads trajectory file.
    4. rdf.py: Parses the RDF

    Usage Example:
    --------------------------------------------
    python rdf_parallel.py cg/cg.lammpstrj cg/cg.data 1 1 type type


    """ 

    parser = argparse.ArgumentParser(description='Parallelizes the rdf.py by running into bundles of timesteps as defined by the user. It then combines all the output distributions from the parallel runs, writes them')   

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


    parser.add_argument('-f_start', dest="f_start", default=0, type=int, help='First frame of lammps trajectory which should be parsed')

    parser.add_argument('-f_every', dest='f_every', type=int, default=1, help='Frequency of frames of lammps trajectory to be parsed. Default=1.')

    parser.add_argument('-f_end', dest="f_end", default=100000, type=int, help='Last frame of lammps trajectory which should be parsed')

    parser.add_argument('-extra_arg', dest='extra_arg', default='', type=str, help='Extra arguments for rdf.py as if you were calling the script from command-line except the arguments: traj_file, data_file, group1, group2, group1_type, group2_type -f_start, -f_every -f_end, and -o. For example, " -r_max 15 -width 1.0"')

    parser.add_argument('-N', dest="N", default=20, type=int, help='Break it down into N parallel jobs')

    parser.add_argument('-n', dest="n", default=5, type=int, help='Run n of these bundles parallely on one node')

    parser.add_argument('-submit', dest="submit", default=1, type=int, help='If 1: Submit N/n jobs on cluster, 0: Run N/n jobs sequentially on the login node')

    parser.add_argument('-o', dest="output", default='rdf_combined/', type=str, help='Name of the output folder')

    # PBS

    parser.add_argument('-nodes', dest="nodes", type=str, default="1", help='Number of nodes. Default: 1')

    parser.add_argument('-ntasks', dest="ntasks", type=str, default="20", help='Number of tasks. Default: 20')

    parser.add_argument('-walltime', dest="walltime", type=str, default="240", help='Job wall-time in hours. Default: 4')

    parser.add_argument("-queue", dest="queue", type=str, default="standby", help='Name of the queue to submit to. Default: standby')
    parser.add_argument("-job", dest="job", type=str, default="job", help='Prefix for job names. Default:"" ')

    parser.add_argument('-keep_files', dest="keep_files", default=0, type=int, help='If the script should keep files from parallel runs, 1: yes, 0: no')


    # Paths
    parser.add_argument('-python', dest="python", default='/home/akhot/anaconda/bin/', type=str, help='The name main output directory of the entire coarse-graining procedure')



    args = parser.parse_args()

    print("PROGRAM CALL: python rdf_parallel.py {}\n".format(' '.join([ i for i in argv])))            

    group1 = args.group1.split()
    group2 = args.group2.split()
    if len(group1) != len(group2):
        print('Error! Group1 and group2 must be same size. Quitting...')
        quit()

    if args.output[-1]!='/': args.output+='/'
    if os.path.exists(args.output): 
        print('Warning! Output folder {} exists...\n'.format(args.output))
    else:
        os.makedirs(args.output)

    # Frames to be parsed in each bundle
    T=[int(_) for _ in np.linspace(args.f_start,args.f_end,args.N+1)]
    T=[[T[i],T[i+1]-args.f_every] for i in range(len(T)-1)]
    T[-1][1]=T[-1][1]+args.f_every
    #print(T)


    nframes=np.array([(T[j][1]-T[j][0])/args.f_every+1 for i in range(int(ceil(float(args.N)/float(args.n)))) for j in range(i*args.n,(i+1)*args.n)])


    
    # Write files which call the cluster.py parallely
    submitcommand='cd {}/;\n'.format(args.output)
    for i in range(int(ceil(float(args.N)/float(args.n)))):
        if not sum([ not os.path.exists("{}/{}_{}_{}.rdf".format(args.output, j, group1[g], group2[g]) ) for j in range(i*args.n,(i+1)*args.n) for g in range(len(group1)) ]):
            continue
        write_job_header('{}{}.submit'.format(args.output,i), '{}_rdf{}'.format(args.job,i), args.queue, args.nodes, args.ntasks, args.walltime ) 
        f=open('{}{}.submit'.format(args.output,i), 'a')
        f.write('\n\ncd {}/{}\ncd ../\n'.format(os.getcwd(), args.output))
        for j in range(i*args.n,(i+1)*args.n):
            if j==args.N: break
            f.write('{}python {}/rdf.py "{}" "{}" "{}" "{}" "{}" "{}" -f_start {} -f_end {} -f_every {} -o {}{} {} > {}{}.log &\n'.format(args.python,  "/".join(os.path.realpath('rdf_parallel.py').split('/')[:-1]), args.traj_file, args.data_file, args.group1, args.group2, args.group1_type, args.group2_type, T[j][0], T[j][1], args.f_every, args.output, j, args.extra_arg, args.output, j))
        f.write('\n\ncd {}/{}\n'.format(os.getcwd(), args.output)) 
        f.write('\nwait\n\n')
        f.close()

        # Run these files
        if args.submit:
            for i in range(int(ceil(float(args.N)/float(args.n)))):  
                submitcommand+='ids_sub+=( $( sbatch {}.submit ) )\n'.format(i)
        else:
            for i in range(int(ceil(float(args.N)/float(args.n)))):  
                submitcommand+='sh {}.submit >  {}_cluster{}.out; wait\n'.format(i,args.job,i)
        
    sp.call(submitcommand+'. ~/check_queue.sh "ids_sub[@]"; wait\n\n',shell=True)        
        


    # Check if the files ran and parse the outputs
    for i in range(int(ceil(float(args.N)/float(args.n)))):
        for j in range(i*args.n,(i+1)*args.n):
            if j==args.N: break
            for g in range(len(group1)):
                if not os.path.exists("{}/{}_{}_{}.rdf".format(args.output, j, group1[g], group2[g])):
                    print("Error! The RDF file {}/{}_{}_{}.rdf not found. Exiting...".format(args.output, j, group1[g], group2[g]))
                    quit()



    # Parse parallel rdf jobs.
    for g in range(len(group1)):
        r, rdf=[], []
        for t in range(args.N):
            r+=[[]]
            rdf+=[[]]
            f=open('{}{}_{}_{}.rdf'.format(args.output,t,group1[g],group2[g]),'r')
            for i,line in enumerate(f):
                if i:
                    r[-1].append(float(line.split()[0]))
                    rdf[-1].append(float(line.split()[1]))
            f.close()
        r, rdf = np.array(r), np.array(rdf)
        # Check r is same across all the files
        if len([ _ for _ in r[1:] if np.max(np.abs(_-r[0]))>10**-8]):
            print("Error! The array for radial distance is not same across all parallel chunks for RDF type {}. Exiting...".format(base.replace('rdf','')))
            quit()
        r=r[0] # Just set to the first array
        # Add all the RDF arrays and divide by nframes
        #print(np.tile(nframes,(rdf.shape[1],1)), sum(nframes))
                      
        rdf= np.sum(rdf*np.tile(nframes,(rdf.shape[1],1)).transpose()/sum(nframes), axis=0)
        
        # Write down
        write_cols(args.output+"{}_{}.rdf".format(group1[g],group2[g]), [r,rdf], ["r", "rdf"])
            
        
            
    # If asked, clean up files from parallel runs and combine the trajectory
    if not args.keep_files:
        submitcommand+='rm  {}; wait\n'.format(' '.join([i for base in list(files.keys()) for i in files[base]]))
        sp.call(submitcommand, shell=True)

# Write header for the slurm job submission.
def write_job_header(filename, job_name, queue, nodes, ntasks, walltime ): 
    walltime= int(walltime)
    walltime="{0:02}".format(walltime//60)+":{0:02}".format(walltime%60)+":00"
    f=open(filename, 'w')
    f.write('#!/bin/sh \n')
    f.write("#SBATCH --job-name "+ job_name+'\n')
    f.write("#SBATCH -A "+ queue+'\n')
    f.write("#SBATCH --nodes="+nodes+'\n')      
    f.write("#SBATCH --ntasks="+ntasks+'\n')
    f.write("#SBATCH --time="+walltime+'\n')
    f.write("#SBATCH --output "+job_name+".out"+'\n')
    f.write("#SBATCH --error "+job_name+".err"+'\n\n')
    return        

if __name__ == "__main__":
   main(sys.argv[1:])

    
