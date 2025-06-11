COGA (Coarse-grain Optimizing Genetic Algorithm) is a set of scripts that use a Genetic Algorithm to develop a coarse-grain potential for a system with a crystalline phase.

Files:
    Contained in this folder are several scripts, organized by usage. These folders include:
    1. Analysis: Scripts that process results of simulations, including calculating RDF's and autocorrelation/scalar order parameter
    2. Dilatometry: Scripts that set up and examine Dilatometry and Melting simulations.
    3. Example_Inputs: Folders that contain example inputs of various systems, as well as example set ups for liquid crystalline and crystalline simulations.
    4. Input_Operations: Scripts that process inputs, formatting them for use with the GA and beyond.
    5. Job_Submission: Scripts that help organize submission to a computational cluster.
    6. Plotting: Scripts that allow outputs of other scripts to be nicely visualized.
    7. COGA.py: The core script of the GA
    8. GA.submit: An example submission file that runs the GA. 

Usage:
    1. Run the GA. An example submission file is found in GA.submit. Fill in arguments in the COGA.py function call based on use case and arguments provided in COGA.
       Be sure to correct arguments such as the python and lammps versions you will use, as these should be passed throughout the script. Following are some major inputs required by the algorithm.
       a. xyz_file: an xyz geometry file that contains the basic atoms and geometry of the molecule you are parametrizing
       b. cif_file: a cif file corresponding to the crystal structure of the molecule being parametrized. If your crystal structure requires multiple molecule to be present in the cif file, please ensure all input files
          represent the same number of molecules. This can be relatively easily done with a program like mercury, which allows you to output a given selection of molecules as both a cif and xyz file.
       c. map_file: translation from all atom to CG mappings. See Example_Inputs for examples. Has three columns: the names of each bead, the atoms in each bead, and the beads that bead is adjacent to.
          The lists of atoms in each bead are 1-indexed whereas the adjacency of each bead are 0-indexed (sorry!). Beads with the same name will use the same potential in the algorithm. 
       d. other arguments: should assume the reported values with their defaults, with the exception of certain arguments that are molecule or situation specific, such as -order, or any of the scaling switches.
          Refer to the Example_Inputs or the paper for what does done originally, and the "help" messages in the script itself for details. Warning: some combinations of settings may not work as desired at present.
    2. Depending on arguments chosen when running, the script will either create, or create and submit the final simulation of the best parameters. 
    3. When the final simulation is submitted, folders for running crystalline, dilatometric, and liquid crystalline simulations will be created, and may also be submitted.
    4. If these simulations exceed any computational time limits, edited versions of the provided restart files in the Example_Inputs files can be used to restart the simulation
    5. Analysis scripts in Dilatometry can be used to analyze the results of these simulations.
