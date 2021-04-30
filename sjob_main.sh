#!/bin/bash
#SBATCH -J HIER_marcoAct            #Job name(--job-name)
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

# mpirun -bootstrap slurm python main.py -embed 104 -heads 4 -hid 64 -l_e1 4 -l_e2 2 -l_d 2 -d 0.2024 -bs 8 -e 30 -model HIER

mpirun -bootstrap slurm python main.py -embed 512 -hid 512 -l_e1 4 -l_e2 4 -l_d 3 -bs 64 -e 20
mpirun -bootstrap slurm python main.py -embed 512 -hid 512 -l_e1 4 -l_e2 4 -l_d 3 -bs 64 -e 20
mpirun -bootstrap slurm python main.py -embed 512 -hid 512 -l_e1 4 -l_e2 4 -l_d 3 -bs 64 -e 20
mpirun -bootstrap slurm python main.py -embed 512 -hid 512 -l_e1 4 -l_e2 4 -l_d 3 -bs 64 -e 20
mpirun -bootstrap slurm python main.py -embed 512 -hid 512 -l_e1 4 -l_e2 4 -l_d 3 -bs 64 -e 20

# {'nhead': 4, 'embedding_perhead': 26, 'nhid_perhead': 16, 'nlayers_e1': 4, 'nlayers_e2': 2, 'nlayers_d': 2, 'dropout': 0.20244212555189078, 'batch_size': 8, 'epochs': 5, 'model_type': 'HIER', 'embedding_size': 104, 'nhid': 64}

# [I 2020-09-08 23:24:51,165] Trial 2 finished with value: 55.70961030741921 and parameters: {'nhead': 4, 'embedding_perhead': 26, 'nhid_perhead': 16, 'nlayers_e1': 4, 'nlayers_e2': 2, 'nlayers_d': 2, 'dropout': 0.20244212555189078}. Best is trial 2 with value: 55.70961030741921.
