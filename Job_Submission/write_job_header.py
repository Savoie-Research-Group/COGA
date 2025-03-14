import sys, argparse
def main(argv):
    parser = argparse.ArgumentParser(description='Writes PBS job header to a output submission file and additionally writes a lammps job submission')
    
    parser.add_argument('dir_name', type=str, help='Name of the working directory to be written in the submission file')
    parser.add_argument('file_name', type=str, help='Name of the submission file')
    parser.add_argument('-sched', dest='sched', default='slurm-halstead',
                        help = 'Specifies the scheduler protocol to use (torque-halstead and torque-titan are implemented)')    
    parser.add_argument('-job_name', dest='job_name', type=str, default="job0" , help='Name of the PBS job. Default: job0')
    parser.add_argument('-account', dest="account", type=str, default="chm114", help='PBS account. Default: chm114')
    parser.add_argument('-nodes', dest="nodes", type=str, default="1", help='Number of nodes. Default: 1')
    parser.add_argument('-ppn', dest="ppn", type=str, default="20", help='Number of processors. Default: 20')
    parser.add_argument('-walltime', dest="walltime", type=int, default="240", help='Job wall-time in minutes. Default: 4 hours')
    parser.add_argument("-queue", dest="queue", type=str, default="standby", help='Name of the queue to submit to. Default: standby')
    parser.add_argument("-lammps", dest="lammps", type=str, default="", help='Lammps executable to be used')
    parser.add_argument("-lammps_init", dest="init", type=str, default="", help='Space delimited name of the lammps init files if a lammps job has to be submitted')
    parser.add_argument("-lammps_module", dest="lammps_module", type=str, default="", help='Commands loading other modules needed to run the given lammps version. Default: ""')
    parser.add_argument('-mpirun', dest='mpirun', type=str, default='/apps/cent7/intel/impi/2017.1.132/bin64/mpirun', help='Location/name of the mpi to be used to run with your version of lammps')
    parser.add_argument("-ntasks", dest="ntasks", type=str, default="20", help='The number of mpi tasks for lammps job')
    parser.add_argument("-mem_all", dest="mem_all", type=int, default=0, help='Allot the entire memory of the node to the job')
    
    args = parser.parse_args()
    args.walltime="{0:02}".format(args.walltime//60)+":{0:02}".format(args.walltime%60)+":00"

    f=open(args.file_name,'w')
    if args.sched=='torque-halstead':
        f.write("#PBS -N "+ args.job_name+'\n')
        f.write("#PBS -A "+ args.account+'\n')
        f.write("#PBS -l nodes="+args.nodes+":ppn="+args.ppn+'\n')
        f.write("#PBS -l walltime="+args.walltime+'\n')
        f.write("#PBS -q "+args.queue+'\n')
        f.write("#PBS -S /bin/sh"+'\n')
        f.write("#PBS -o "+args.job_name+".out"+'\n')
        f.write("#PBS -e "+args.job_name+".err"+'\n\n')
    elif args.sched=='torque-gilbreth':
        f.write("#PBS -N "+ args.job_name+'\n')
        f.write("#PBS -A "+ args.account+'\n')
        f.write("#PBS -l nodes="+args.nodes+":ppn="+args.ppn+':gpus=1\n')
        f.write("#PBS -l walltime="+args.walltime+'\n')
        f.write("#PBS -q "+args.queue+'\n')
        f.write("#PBS -S /bin/sh"+'\n')
        f.write("#PBS -o "+args.job_name+".out"+'\n')
        f.write("#PBS -e "+args.job_name+".err"+'\n\n')
    elif args.sched=='slurm-xsede':
        f.write('#!/bin/sh \n')
        f.write("#SBATCH -J "+ args.job_name+'\n')
        f.write("#SBATCH -A "+ args.account+'\n')
        f.write("#SBATCH -N "+args.nodes+'\n')
        f.write("#SBATCH -c "+args.ppn+'\n')
        f.write("#SBATCH -t "+args.walltime+'\n')
        f.write("#SBATCH -p "+args.queue+'\n')
        f.write("#SBATCH -o "+args.job_name+".out"+'\n')
        f.write("#SBATCH -e "+args.job_name+".err"+'\n\n')
    elif args.sched=='slurm-halstead':
        f.write('#!/bin/sh \n')
        f.write("#SBATCH --job-name "+ args.job_name+'\n')
        f.write("#SBATCH -A "+ args.queue+'\n')
        f.write("#SBATCH --nodes="+args.nodes+'\n')      
        if len(args.ntasks):
            f.write("#SBATCH --ntasks="+str(args.ntasks)+'\n')
        if args.mem_all:
            f.write("#SBATCH --exclusive\n")  
        f.write("#SBATCH --time="+args.walltime+'\n')
        f.write("#SBATCH --output "+args.job_name+".out"+'\n')
        f.write("#SBATCH --error "+args.job_name+".err"+'\n\n')
    elif args.sched=='slurm-gilbreth':
        f.write('#!/bin/sh \n')
        f.write("#SBATCH --job-name "+ args.job_name+'\n')
        f.write("#SBATCH -A "+ args.queue+'\n')
        f.write("#SBATCH --nodes="+args.nodes+'\n')
        f.write("#SBATCH --gpus-per-node=2\n")
        if len(args.ntasks):
            f.write("#SBATCH --ntasks="+str(args.ntasks)+'\n')
            f.write("#SBATCH --mem=0\n")  
        else:
            f.write("#SBATCH --mem=0\n")  
        f.write("#SBATCH --time="+args.walltime+'\n')
        f.write("#SBATCH --output "+args.job_name+".out"+'\n')
        f.write("#SBATCH --error "+args.job_name+".err"+'\n\n')
        
        

    f.write("#cd into submission trajectory"+'\n')
    f.write("cd "+args.dir_name+'\n')
    f.write("echo Working directory is "+args.dir_name+'\n')
    f.write("echo Running on host `hostname`"+'\n')
    f.write("echo Time is `date`"+'\n\n')
    if len(args.lammps):
        f.write(args.lammps_module+'\n\n')
        f.write('cd .\n')
        for i in args.init.split():
            f.write('{} -np {} {} -in {} >> {} &\nwait\n'.format(args.mpirun, args.ntasks, args.lammps, i, i[:i.index('.init')]+'.out'))
        f.write('\ncd {} &\n'.format(args.dir_name))
        f.write('wait\n\n')
    f.close()


if __name__ == '__main__':
    main(sys.argv[1:])


    



