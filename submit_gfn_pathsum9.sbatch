#!/bin/bash
#SBATCH --account=def-simine
#SBATCH --mail-user=jeremy.lackman-mincoff@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --time=0-168:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gfn_extend9

# cpu: --cpus-per-task=1
# gpu: --gres=gpu:1
#module load cuda/12.2
source ../gfn_env/bin/activate
python main_gfn_extend.py pathsum_input_file=PathSumSys9.inp dynamics_executable=run_dynamics9.out starting_paths_input_file1=specific-paths-len3-inds1-solution184paths-reward149.42207.inp #starting_paths_input_file2=Newpathways3-len3-inds1-reward148.0837647633478-trainstep1400.txt 
deactivate
