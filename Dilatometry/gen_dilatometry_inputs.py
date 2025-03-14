# Written by: Dylan Gilley, Major edits by: Dylan Fortney

import argparse,sys,os
script_directory = "/".join(str(os.path.realpath(__file__)).split("/")[:-2])        # Path to this script.
sys.path.append(script_directory)

def main(argv):

    parser = argparse.ArgumentParser(description='This script writes the input and submission files for dilatometry simulations.')

    # Required arguments    
    parser.add_argument(dest='prefix', help='Prefix name')

    # Optional arguments
    parser.add_argument('-queue', dest='queue', default="bsavoie", help='Queue')

    parser.add_argument('-o', dest='output', default="dila", help='name appended to prefix to get output files.')

    parser.add_argument('-folder', dest='folder', default="", help='name of folder submitted from for naming jobs.')

    parser.add_argument('-nodes', dest='nodes', default=1, type=int, help='Number of nodes')

    parser.add_argument('-cores', dest='cores', default=128, type=int, help='Number of cores')

    parser.add_argument('-timelim', dest='timelim', default="14-00:00:00", help='Time limit')

    parser.add_argument('-t_reshape', dest='t_reshape', default=5000000, type=int, help='Timesteps for reshaping the box. Default: 5,000,000 (5 ns)')

    parser.add_argument('-t_ramp', dest='t_ramp', default=10000000, type=int, help='Timesteps for ramping the temperature up from 10K. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_eqhigh', dest='t_eqhigh', default=10000000, type=int, help='Timesteps for high temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_cool', dest='t_cool', default=30000000, type=int, help='Timesteps for cooling the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-t_eqlow', dest='t_eqlow', default=10000000, type=int, help='Timesteps for low temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_warm', dest='t_warm', default=30000000, type=int, help='Timesteps for warming the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-t_eqhigh2', dest='t_eqhigh2', default=10000000, type=int, help='Timesteps for the second high temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_cool2', dest='t_cool2', default=30000000, type=int, help='Timesteps for the second cooling of the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-t_eqlow2', dest='t_eqlow2', default=10000000, type=int, help='Timesteps for the second low temp equilibration. Default: 10,000,000 (10 ns)')

    parser.add_argument('-t_warm2', dest='t_warm2', default=30000000, type=int, help='Timesteps for the secondwarming of the system. Default: 30,000,000 (30 ns)')

    parser.add_argument('-cell_size', dest='cell_size', default="0.0 100.0 0.0 100.0 0.0 100.0", help='Cell dimensions to be shrunk to. format: xlo xhi ylo yhi zlo zhi')

    parser.add_argument('-order', dest='order', default='', type=str, help='Determines if the scalar order parameter will be caclulated and if so what vector to be used. Format "0-x-y", x,y being the atom numbers that compose the vector. ')

    parser.add_argument('-python', dest='python', type=str, default='python', help='Location/name of the python version to be used.')

    parser.add_argument('-mpirun', dest='mpirun', type=str, default='/apps/spack/negishi/apps/intel-mpi/2019.10.317-intel-2021.8.0-wnukven/impi/2019.10.317/intel64/bin/mpirun', help='Location/name of the mpi to be used to run with your version of lammps. Bell: /apps/cent7/intel/impi/2017.1.132/bin64/mpirun')

    parser.add_argument('-lammps', dest='lammps', type=str, default='/depot/bsavoie/apps/lammps/exe/lmp_mpi_190322', help='Location/name of the lammps version to be run. Note, some changes to submissions may be necessary. Bell: /depot/bsavoie/apps/lammps/exe//lmp_mpi_180501')

    args = parser.parse_args(argv)

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
            "variable    vseed                index    813  \n"+\
            "\n"+\
            "# Reshape\n"+\
            "variable    nSteps_reshape        index    {}          # {} ps ({} ns) box reshaping (NVT)\n".format(args.t_reshape, args.t_reshape/1000, args.t_reshape/1000000)+\
            "variable    temp_reshape          index    10.0       # 10.0 K\n"+\
            "\n"+\
            "# Ramp\n"+\
            "variable    nSteps_ramp        index    {}          # {} ps ({} ns) ramp to higher temperature (NPT) \n".format(args.t_ramp, args.t_ramp/1000, args.t_ramp/1000000)+\
            "\n"+\
            "# Equilibration at High Temp\n"+\
            "variable    nSteps_eqhigh        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(args.t_eqhigh, args.t_eqhigh/1000, args.t_eqhigh/1000000)+\
            "variable    press_eqhigh         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqhigh          index    500.0       # 500.0 K\n"+\
            "\n"+\
            "# Cool Down\n"+\
            "variable    nSteps_cooldown      index    {}          # {} ps ({} ns) decreasing temp anneal (NPT)\n".format(args.t_cool, args.t_cool/1000, args.t_cool/1000000)+\
            "variable    press_cooldown       index    1.0         # 1.0 atm\n"+\
            "variable    temp0_cooldown       index    500.0       # 500.0 K initial temp\n"+\
            "variable    tempf_cooldown       index    100.0       # 100.0 K final temp\n"+\
            "\n"+\
            "# Equilibration at Low Temp\n"+\
            "variable    nSteps_eqlow        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(args.t_eqlow, args.t_eqlow/1000, args.t_eqlow/1000000)+\
            "variable    press_eqlow         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqlow          index    100.0       # 100.0 K\n"+\
            "\n"+\
            "# Warm Up\n"+\
            "variable    nSteps_warmup        index    {}          # {} ps ({} ns) increasing temp anneal (NPT)\n".format(args.t_warm, args.t_warm/1000, args.t_warm/1000000)+\
            "variable    press_warmup         index    1.0         # 1.0 atm\n"+\
            "variable    temp0_warmup         index    100.0       # 100.0 K initial temp\n"+\
            "variable    tempf_warmup         index    500.0       # 500.0 K final temp\n"+\
            "\n"+\
            "# Second Equilibration at High Temp\n"+\
            "variable    nSteps_eqhigh2        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(args.t_eqhigh2, args.t_eqhigh2/1000, args.t_eqhigh2/1000000)+\
            "variable    press_eqhigh2         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqhigh2          index    500.0       # 500.0 K\n"+\
            "\n"+\
            "# Second Cool Down\n"+\
            "variable    nSteps_cooldown2      index    {}          # {} ps ({} ns) decreasing temp anneal (NPT)\n".format(args.t_cool2, args.t_cool2/1000, args.t_cool2/1000000)+\
            "variable    press_cooldown2       index    1.0         # 1.0 atm\n"+\
            "variable    temp0_cooldown2       index    500.0       # 500.0 K initial temp\n"+\
            "variable    tempf_cooldown2       index    100.0       # 100.0 K final temp\n"+\
            "\n"+\
            "# Second Equilibration at Low Temp\n"+\
            "variable    nSteps_eqlow2        index    {}          # {} ps ({} ns) high temp equilibration (NPT)\n".format(args.t_eqlow2, args.t_eqlow2/1000, args.t_eqlow2/1000000)+\
            "variable    press_eqlow2         index    1.0         # 1.0 atm\n"+\
            "variable    temp_eqlow2          index    100.0       # 100.0 K\n"+\
            "\n"+\
            "# Second Warm Up\n"+\
            "variable    nSteps_warmup2        index    {}          # {} ps ({} ns) increasing temp anneal (NPT)\n".format(args.t_warm2, args.t_warm2/1000, args.t_warm2/1000000)+\
            "variable    press_warmup2         index    1.0         # 1.0 atm\n"+\
            "variable    temp0_warmup2         index    100.0       # 100.0 K initial temp\n"+\
            "variable    tempf_warmup2         index    500.0       # 500.0 K final temp\n"+\
            "\n"+\

            # "#===========================================================\n"+\
            # "# General Settings\n"+\
            # "#===========================================================\n"+\
            # "\n"+\
            # "read_restart ${prefix}.in.restart\n"+\
            # "log ${prefix}.lammps.log\n"+\
            # "newton on          # use Newton's 3rd law\n"+\
            # "\n"+\

            # "#===========================================================\n"+\
            # "# Force Field Definitions\n"+\
            # "#===========================================================\n"+\
            # "\n"+\
            # "special_bonds   lj 0.0 0.0 0.0 coul 0.0 0.0 0.0    # NO 1-4 LJ/COUL interactions\n"+\
            # "pair_style      lj/cut/coul/long 14.0 14.0         # outer_LJ outer_Coul (cutoff values, see LAMMPS Doc)\n"+\
            # "kspace_style    pppm 0.0001                        # long-range electrostatics sum method\n"+\
            # "bond_style      harmonic                           # parameters needed: k_bond, r0\n"+\
            # "angle_style     harmonic                           # parameters needed: k_theta, theta0\n"+\
            # "dihedral_style  opls                               # parameters needed: V1, V2, V3, V4\n"+\
            # "improper_style  cvff                               # parameters needed: K, d, n\n"+\
            # "pair_modify     shift yes mix sixthpower           # using Waldman-Hagler mixing rules\n"+\
            # "\n"+\

            # "#===========================================================\n"+\
            # "# Setup System\n"+\
            # "#===========================================================\n"+\
            # "\n"+\
            # "include ${settings_name}\n"+\
            # "timestep 1.0                                         # 1.0 fs timestep\n"+\
            # "run_style verlet                                     # Velocity-Verlet integrator\n"+\
            # "neigh_modify every 1 delay 10 check yes one 10000    # More relaxed rebuild criteria can be used\n"+\
            # "\n"+\
            # "thermo_style custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n"+\
            # "thermo_modify format float %20.10f\n"+\
            # "thermo ${thermo_freq}\n"+\
            # "\n"+\
            # "variable        my_step equal   step\n"+\
            # "variable        my_temp equal   temp\n"+\
            # "variable        my_rho  equal   density\n"+\
            # "variable        my_pe   equal   pe\n"+\
            # "variable        my_ebon equal   ebond\n"+\
            # "variable        my_eang equal   eangle\n"+\
            # "variable        my_edih equal   edihed\n"+\
            # "variable        my_evdw equal   evdwl\n"+\
            # "variable        my_eel  equal   (ecoul+elong)\n"+\
            # "variable        my_ent  equal   enthalpy\n"+\
            # "variable        my_P    equal   press\n"+\
            # "variable        my_vol  equal   vol\n"+\
            # "\n"+\
            # "fix averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P file ${prefix}.thermo.avg\n"+\
            # "# Set momentum fix to zero out momentum (linear and angular) every ps\n"+\
            # "#     Note: this fix should remain active for the rest of the file\n"+\
            # "fix mom all momentum 1000 linear 1 1 1 angular\n"+\
            # "\n"+\

            # Change the name of the log output #
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
            "\n"+\

            "#===========================================================\n"+\
            "# Ramp (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump ramp all custom ${coords_freq} ${prefix}.ramp.lammpstrj id mol type x y z\n"+\
            "dump_modify ramp sort id format float %20.10g\n"+\
            "fix ramp all npt temp ${temp_reshape} ${temp_eqhigh} 100.0 iso ${press_cooldown} ${press_cooldown} 1000.0\n"+\
            "run ${nSteps_ramp}\n"+\
            "unfix ramp\n"+\
            "undump ramp\n"+\
            "write_restart ${prefix}.ramp.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# High Temperature Equilibration (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump eqhigh all custom ${coords_freq} ${prefix}.eqhigh.lammpstrj id mol type x y z\n"+\
            "dump_modify eqhigh sort id format float %20.10g\n"+\
            "fix eqhigh all npt temp ${temp_eqhigh} ${temp_eqhigh} 100.0 iso ${press_eqhigh} ${press_eqhigh} 1000.0\n"+\
            "run ${nSteps_eqhigh}\n"+\
            "unfix eqhigh\n"+\
            "undump eqhigh\n"+\
            "write_restart ${prefix}.eqhigh.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Cool Down (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump cooldown all custom ${coords_freq} ${prefix}.cooldown.lammpstrj id mol type x y z\n"+\
            "dump_modify cooldown sort id format float %20.10g\n"+\
            "fix cooldown all npt temp ${temp0_cooldown} ${tempf_cooldown} 100.0 iso ${press_cooldown} ${press_cooldown} 1000.0\n"+\
            "run ${nSteps_cooldown}\n"+\
            "unfix cooldown\n"+\
            "undump cooldown\n"+\
            "write_restart ${prefix}.cooldown.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Low Temperature Equilibration (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump eqlow all custom ${coords_freq} ${prefix}.eqlow.lammpstrj id mol type x y z\n"+\
            "dump_modify eqlow sort id format float %20.10g\n"+\
            "fix eqlow all npt temp ${temp_eqlow} ${temp_eqlow} 100.0 iso ${press_eqlow} ${press_eqlow} 1000.0\n"+\
            "run ${nSteps_eqlow}\n"+\
            "unfix eqlow\n"+\
            "undump eqlow\n"+\
            "write_restart ${prefix}.eqlow.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Warm Up (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump warmup all custom ${coords_freq} ${prefix}.warmup.lammpstrj id mol type x y z\n"+\
            "dump_modify warmup sort id format float %20.10g\n"+\
            "fix warmup all npt temp ${temp0_warmup} ${tempf_warmup} 100.0 iso ${press_warmup} ${press_warmup} 1000.0\n"+\
            "run ${nSteps_warmup}\n"+\
            "unfix warmup\n"+\
            "undump warmup\n"+\
            "write_restart ${prefix}.warmup.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Second High Temperature Equilibration (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump eqhigh2 all custom ${coords_freq} ${prefix}.eqhigh2.lammpstrj id mol type x y z\n"+\
            "dump_modify eqhigh2 sort id format float %20.10g\n"+\
            "fix eqhigh2 all npt temp ${temp_eqhigh2} ${temp_eqhigh2} 100.0 iso ${press_eqhigh2} ${press_eqhigh2} 1000.0\n"+\
            "run ${nSteps_eqhigh2}\n"+\
            "unfix eqhigh2\n"+\
            "undump eqhigh2\n"+\
            "write_restart ${prefix}.eqhigh2.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Second Cool Down (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump cooldown2 all custom ${coords_freq} ${prefix}.cooldown2.lammpstrj id mol type x y z\n"+\
            "dump_modify cooldown2 sort id format float %20.10g\n"+\
            "fix cooldown2 all npt temp ${temp0_cooldown2} ${tempf_cooldown2} 100.0 iso ${press_cooldown2} ${press_cooldown2} 1000.0\n"+\
            "run ${nSteps_cooldown2}\n"+\
            "unfix cooldown2\n"+\
            "undump cooldown2\n"+\
            "write_restart ${prefix}.cooldown2.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Second Low Temperature Equilibration (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump eqlow2 all custom ${coords_freq} ${prefix}.eqlow2.lammpstrj id mol type x y z\n"+\
            "dump_modify eqlow2 sort id format float %20.10g\n"+\
            "fix eqlow2 all npt temp ${temp_eqlow2} ${temp_eqlow2} 100.0 iso ${press_eqlow2} ${press_eqlow2} 1000.0\n"+\
            "run ${nSteps_eqlow2}\n"+\
            "unfix eqlow2\n"+\
            "undump eqlow2\n"+\
            "write_restart ${prefix}.eqlow2.end.restart\n"+\
            "\n"+\

            "#===========================================================\n"+\
            "# Second Warm Up (NPT, Nose-Hoover)\n"+\
            "#===========================================================\n"+\
            "\n"+\
            "dump warmup2 all custom ${coords_freq} ${prefix}.warmup2.lammpstrj id mol type x y z\n"+\
            "dump_modify warmup2 sort id format float %20.10g\n"+\
            "fix warmup2 all npt temp ${temp0_warmup2} ${tempf_warmup2} 100.0 iso ${press_warmup2} ${press_warmup2} 1000.0\n"+\
            "run ${nSteps_warmup2}\n"+\
            "unfix warmup2\n"+\
            "undump warmup2\n"+\
            "write_restart ${prefix}.warmup2.end.restart\n"+\
            "\n"+\
           

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
            "{} -np $SLURM_NPROCS {} -in  {}.in.init > {}_lammps.out & wait \n".format(args.mpirun, args.lammps, outname,outname)+\
            "{} {}/Dilatometry/CG_dilatometry.py thermo.avg -prefix {} -t_reshape {} -t_ramp {} -t_eqhigh {} -t_cool {} -t_eqlow {} -t_warm {} -t_eqhigh2 {} -t_cool2 {} -t_eqlow2 {} -t_warm2 {} -order '{}' -python '{}' -mpirun '{}' -lammps '{}' > dila_analysis.log \n".format(args.python, script_directory, args.prefix, args.t_reshape, args.t_ramp, args.t_eqhigh, args.t_cool, args.t_eqlow, args.t_warm, args.t_eqhigh2, args.t_cool2, args.t_eqlow2, args.t_warm2, args.order, args.python, args.mpirun, args.lammps)+\
            "echo \"End time: $(date)\""
            )

    return

if __name__ == "__main__":
   main(sys.argv[1:])
