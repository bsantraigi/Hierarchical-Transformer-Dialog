#!/bin/bash

#SBATCH -J hier_search_acts            #Job name(--job-name)
#SBATCH -o logs/slurm_%j.log          #Name of stdout output file(--output)
#SBATCH -e logs/slurm_%j.log               #Name of stderr error file(--error)
#SBATCH -p gpu-low              #Queue (--partition) name
#SBATCH --gres=gpu:1               # request gpu card: it should be either 1 or 2
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH -c 1                    #(--cpus-per-task) Number of Threads
#SBATCH --mem=23000        # Memory per node specification is in MB. It is optional.
pwd; hostname; date
module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cuda/10.2
source /home/$USER/.bashrc
export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
mpirun -bootstrap slurm which python
mpirun -bootstrap slurm nvcc --version

mpirun -bootstrap slurm python search_params_acts.py -e 5 -bs 64 -model HIER++
#mpirun -bootstrap slurm python search_params_acts.py -e 5 -model SET++
