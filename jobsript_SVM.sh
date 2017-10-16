#!/bin/bash

#SBATCH --mail-user=michael.rebsamen@students.unibe.ch
#SBATCH --mail-type=begin,end,fail
#SBATCH --time=28:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --workdir=.
#SBATCH --job-name="main_SVM.py"

#### Your shell commands below this line ####
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# folders
workdir=${PWD}

echo "Starting on host:"
hostname

# activate environment
source activate MIALab
python -V

python /home/ubelix/istb/mr02b028/MIALab2017/MIALab/bin/main_SVM.py

