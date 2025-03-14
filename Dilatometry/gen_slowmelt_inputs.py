#!/bin/env python
# Author: Dylan Fortney

import os, sys, argparse
import numpy as np
from pathlib import Path
home=str(Path.home())
import random

def main(argv):
    parser = argparse.ArgumentParser(description='This script writes the input and submission files for slow melting simulations.')

    # Required arguments    
    parser.add_argument(dest='prefix', help='Prefix name')

    # Optional arguments
    parser.add_argument('-queue', dest='queue', default="bsavoie", help='Queue')

    parser.add_argument('-o', dest='output', default="dila", help='name appended to prefix to get output files.')

    parser.add_argument('-folder', dest='folder', default="", help='name of folder submitted from for naming jobs.')

    parser.add_argument('-nodes', dest='nodes', default=1, type=int, help='Number of nodes')

    parser.add_argument('-cores', dest='cores', default=128, type=int, help='Number of cores')

    parser.add_argument('-timelim', dest='timelim', default="14-00:00:00", help='Time limit')

    parser.add_argument('-t_reshape', dest='t_reshape', default=1000000, type=int, help='Timesteps for reshaping the box. Default: 1,000,000 (1 ns)')

    parser.add_argument('-t_eq', dest='t_eq', default=10000000, type=int, help='Timesteps for ramping the temperature up from 10K. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_ramp', dest='t_ramp', default=10000000, type=int, help='Timesteps for high temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-cell_size', dest='cell_size', default="0.0 100.0 0.0 100.0 0.0 100.0", help='Cell dimensions to be shrunk to. format: xlo xhi ylo yhi zlo zhi')

    parser.add_argument('-order', dest='order', default='', type=str, help='Determines if the scalar order parameter will be caclulated and if so what vector to be used. Format "0-x-y", x,y being the atom numbers that compose the vector. ')

    parser.add_argument('-T_start', dest='T_start', default=10, type=int, help='Temperature to start the simulations at, in Kelvin')

    parser.add_argument('-T_end', dest='T_end', default=250, type=int, help='Temperature to end the simulations at, in Kelvin')

    parser.add_argument('-cycles', dest='cycles', default=5, type=int, help='Number of ramp-equilibration cycles to perform.')

    parser.add_argument('-cyc_start', dest='cyc_start', default=1, type=int, help='Number you wish to start the cycle indexing on. Useful if you restart with a new ramp.')

    parser.add_argument('-python', dest='python', type=str, default='python', help='Location/name of the python version to be used.')

    parser.add_argument('-mpirun', dest='mpirun', type=str, default='/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun', help='Location/name of the mpi to be used to run with your version of lammps. Bell: /apps/cent7/intel/impi/2017.1.132/bin64/mpirun')

    parser.add_argument('-lammps', dest='lammps', type=str, default='/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322', help='Location/name of the lammps version to be run. Note, some changes to submissions may be necessary. Bell: /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501')

    # Parse Arguments
    args = parser.parse_args()

    # Write Script Below
    # Write the data file
    outname=args.prefix+str(args.output)
    if args.folder:
        job_name=str(args.folder)+str(args.output)
    else:
        job_name=outname
    init_file = outname+'.in.init'
    settings_file=args.prefix+'.in.settings'
    outname=args.prefix+str(args.output)
    cell_size=args.cell_size.split()
    print("inputs cell_size", cell_size)

    # Calculate temperature ranges for each stage
    temp_steps = np.linspace(args.T_start, args.T_end, args.cycles+1, dtype=int)

    with open(init_file,'w') as f:
        f.write(
            "# LAMMPS input file for equilibration and annealing of liquid crystaline systems from GA. \n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Variables\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "# File-wide variables\n"+\
            "variable    settings_name        index    {}\n".format(settings_file)+\
            "variable    prefix               index    {}\n".format(args.prefix)+\
            "variable    avg_freq             index    1000\n"+\
            "variable    coords_freq          index    1000\n"+\
            "variable    thermo_freq          index    1000\n"+\
            "variable    dump4avg             index    100\n"+\
            "variable    vseed                index    {}  \n".format(random.randint(100, 999))+\
            "\n"+\
            "# Reshape\n"+\
            "variable    nSteps_reshape        index    {}          # {} ps ({} ns) box reshaping (NVT)\n".format(args.t_reshape, args.t_reshape/1000, args.t_reshape/1000000)+\
            "variable    temp_reshape          index    {}          #  {} K\n".format(args.T_start, args.T_start)+\
            "\n"
            )
        for c, cycle in enumerate(range(args.cyc_start, args.cyc_start+args.cycles)):
            f.write(
                "# Ramp {}\n".format(cycle)+\
                "variable    nSteps_ramp{}        index    {}          # {} ps ({} ns) ramp to higher temperature (NPT) \n".format(cycle, args.t_ramp, args.t_ramp/1000, args.t_ramp/1000000)+\
                "\n"+\

                "# Equilibration at High Temp\n"+\
                "variable    nSteps_eq{}        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(cycle, args.t_eq, args.t_eq/1000, args.t_eq/1000000)+\
                "variable    press_eq{}         index    1.0         # 1.0 atm\n".format(cycle)+\
                "variable    temp_eq{}          index    {}          # {} K\n".format(cycle, temp_steps[c+1], temp_steps[c+1])+\
                "\n"
            )
        f.write(
            "log ${prefix}dila.log\n"+\

            "#===========================================================\n"+\
            "# GENERAL PROCEDURES\n"+\
            "#===========================================================\n"+\
            "units		real	# g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n"+\
            "dimension	3	# 3 dimensional simulation\n"+\
            "newton		off	# use Newton's 3rd law\n"+\
            "boundary	p p p	# periodic boundary conditions \n"+\
            "atom_style	full    # molecular + charge\n"+\

            "#===========================================================\n"+\
            "# FORCE FIELD DEFINITION\n"+\
            "#===========================================================\n"+\
            "special_bonds  lj   0.0 0.0 0.0  coul 0.0 0.0 0.0     # NO     1-4 LJ/COUL interactions\n"+\
            "pair_style   hybrid lj/gromacs/coul/gromacs 9.0 12.0 0.0 12.0 # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n"+\
            "bond_style   hybrid harmonic         # parameters needed: k_bond, r0\n"+\
            "angle_style  hybrid harmonic         # parameters needed: k_theta, theta0\n"+\
            "pair_modify    shift yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n"+\

            "#===========================================================\n"+\
            "# SETUP SIMULATIONS\n"+\
            "#===========================================================\n"+\

            "# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n"+\
            "read_data ${prefix}.data\n"+\
            "include ${prefix}.in.settings\n"+\

            "# SET RUN PARAMETERS\n"+\
            "timestep	1.0		# fs\n"+\
            "run_style	verlet 		# Velocity-Verlet integrator\n"+\
            "neigh_modify every 1 delay 0 check no one 10000 # More relaxed rebuild criteria can be used\n"+\

            "# SET OUTPUTS\n"+\
            "thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n"+\
            "thermo_modify   format float %14.6f\n"+\
            "thermo ${thermo_freq}\n"+\

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
            "variable        my_etot equal   etotal\n"+\

            "fix  averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_my_etot file thermo.avg format %20.10g\n"+\

            "#===========================================================\n"+\
            "# Reshape Box (NVT) \n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump reshape all custom ${coords_freq} ${prefix}.reshape.lammpstrj id mol type x y z\n"+\
            "dump_modify reshape sort id format float %20.10g\n"+\
            "fix reshape  all deform 1 x final {} {} y final {} {} z final {} {} units box\n".format(cell_size[0], cell_size[1], cell_size[2], cell_size[3], cell_size[4], cell_size[5])+\
            "fix dynamics all nvt temp ${temp_reshape} ${temp_reshape} 100.0\n"
            "velocity all create ${temp_reshape} ${vseed}\n"
            "run ${nSteps_reshape}\n"+\
            "unfix reshape\n"+\
            "unfix dynamics\n"+\
            "undump reshape\n"+\
            "write_restart ${prefix}.reshape.end.restart\n"+\
            "\n"
            )
        # cycle through ramp and equil steps for each cycle.
        for c, cycle in enumerate(range(args.cyc_start, args.cyc_start+args.cycles)):
            if c == 0:
                ramp_start = "temp_reshape"
            else:
                ramp_start = "temp_eq"+str(cycle-1)
            f.write(
                "#===========================================================\n"+\
                "# Ramp {} (NPT, Nose-Hoover)\n".format(cycle)+\
                "#===========================================================\n"+\
                "\n"+\
                "dump ramp{} all custom ${{coords_freq}} ${{prefix}}.ramp{}.lammpstrj id mol type x y z\n".format(cycle, cycle)+\
                "dump_modify ramp{} sort id format float %20.10g\n".format(cycle)+\
                "fix ramp{} all npt temp ${{{}}} ${{temp_eq{}}} 100.0 iso ${{press_eq{}}} ${{press_eq{}}} 1000.0\n".format(cycle, ramp_start, cycle, cycle, cycle)+\
                "run ${{nSteps_ramp{}}}\n".format(cycle)+\
                "unfix ramp{}\n".format(cycle)+\
                "undump ramp{}\n".format(cycle)+\
                "write_restart ${{prefix}}.ramp{}.end.restart\n".format(cycle)+\
                "\n"+\

                "#===========================================================\n"+\
                "# Equilibration {} (NPT, Nose-Hoover)\n".format(cycle)+\
                "#===========================================================\n"+\
                "\n"+\
                "dump eq{} all custom ${{coords_freq}} ${{prefix}}.eq{}.lammpstrj id mol type x y z\n".format(cycle, cycle)+\
                "dump_modify eq{} sort id format float %20.10g\n".format(cycle)+\
                "fix eq{} all npt temp ${{temp_eq{}}} ${{temp_eq{}}} 100.0 iso ${{press_eq{}}} ${{press_eq{}}} 1000.0\n".format(cycle, cycle, cycle, cycle, cycle)+\
                "run ${{nSteps_eq{}}}\n".format(cycle)+\
                "unfix eq{}\n".format(cycle)+\
                "undump eq{}\n".format(cycle)+\
                "write_restart ${{prefix}}.eq{}.end.restart\n".format(cycle)+\
                "\n"
                )
        f.write(
            "#===========================================================\n"+\
            "# Clean and exit\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "unfix averages\n"
            )

    # Write slurm submission script
    slurm_file = outname+'.submit'
    with open(slurm_file,'w') as f:
        f.write(
            "#!/bin/bash\n"+\
            "#\n"+\
            "#SBATCH --job-name {}\n".format(job_name)+\
            "#SBATCH -o {}.out\n".format(outname)+\
            "#SBATCH -e {}.err\n".format(outname)+\
            "#SBATCH -A {}\n".format(str(args.queue))+\
            "#SBATCH -N {}\n".format(str(args.nodes))+\
            "#SBATCH -n {}\n".format(str(args.cores))+\
            "#SBATCH -t {}\n\n".format(str(args.timelim))+\

            "# Write out job information\n"+\
            "echo \"Running on host: $SLURM_NODELIST\"\n"+\
            "echo \"Running on node(s): $SLURM_NNODES\"\n"+\
            "echo \"Number of processors: $SLURM_NPROCS\"\n"+\
            "echo \"Current working directory: $SLURM_SUBMIT_DIR\"\n\n"+\

            "# User supplied shell commands\n"+\
            "cd $SLURM_SUBMIT_DIR\n\n"+\

            "# Run script\n"+\
            "echo \"Start time: $(date)\"\n"+\
            # "/apps/cent7/intel/impi/2017.1.132/bin64/mpirun -np {} /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501 -in  {}.in.init >> {}_lammps.out & wait \n".format(str(args.cores),outname,outname)+\
            # This change should allow us to run on multiple nodes...
            # "/apps/cent7/intel/impi/2017.1.132/bin64/mpirun -np $SLURM_NPROCS /depot/bsavoie/apps/lammps/exe/lmp_mpi_180501 -in  {}.in.init > {}_lammps.out & wait \n".format(outname,outname)+\
            # Actually THIS change should allow us to run on multiple nodes. For the love of God people, note which mpi you use with LAMMPS because other people don't know exactly what you did 5 years ago.
            "{} -np $SLURM_NPROCS {} -in  {}.in.init > {}_lammps.out & wait \n".format(args.mpirun, args.lammps, outname, outname)+\
            #"python ~/bin/CG_Crystal/CG_dilatometry.py thermo.avg -prefix {} -t_reshape {} -t_ramp {} -t_eqhigh {} -t_cool {} -t_eqlow {} -t_warm {} -t_eqhigh2 {} -t_cool2 {} -t_eqlow2 {} -t_warm2 {} -order '{}' > dila_analysis.log \n".format(args.prefix, args.t_reshape, args.t_ramp, args.t_eqhigh, args.t_cool, args.t_eqlow, args.t_warm, args.t_eqhigh2, args.t_cool2, args.t_eqlow2, args.t_warm2, args.order)+\
            "echo \"End time: $(date)\""
            )

    return


    # Write Script Above
if __name__ == '__main__':
    main(sys.argv[1:])
