#!/bin/bash
#SBATCH -J hier_act            #Job name(--job-name)
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

# mpirun -bootstrap slurm python main_acts.py -embed 104 -heads 4 -hid 64 -l_e1 4 -l_e2 2 -l_d 2 -d 0.2024 -bs 8 -e 30 -model HIER

mpirun -bootstrap slurm python main_acts.py -embed 512 -hid 512 -l_e1 2 -l_e2 2 -l_d 3 -bs 64 -e 20 -ctmask cls
mpirun -bootstrap slurm python main_acts.py -embed 512 -hid 512 -l_e1 2 -l_e2 2 -l_d 3 -bs 64 -e 20 -ctmask cls
mpirun -bootstrap slurm python main_acts.py -embed 512 -hid 512 -l_e1 2 -l_e2 2 -l_d 3 -bs 64 -e 20 -ctmask cls
mpirun -bootstrap slurm python main_acts.py -embed 512 -hid 512 -l_e1 2 -l_e2 2 -l_d 3 -bs 64 -e 20 -ctmask cls
mpirun -bootstrap slurm python main_acts.py -embed 512 -hid 512 -l_e1 2 -l_e2 2 -l_d 3 -bs 64 -e 20 -ctmask cls

# END OF STORY

# mpirun -bootstrap slurm python main_acts.py -embed 175 -heads 7 -hid 91 -l_e1 4 -l_e2 6 -l_d 3 -d 0.071 -bs 16 -e 60 -model HIER++
# mpirun -bootstrap slurm python main_acts.py -embed 196 -heads 7 -hid 98 -l_e1 2 -l_e2 4 -l_d 6 -d 0.001 -bs 8 -e 30 -model HIER++
# mpirun -bootstrap slurm python main_acts.py -embed 175 -heads 7 -hid 91 -l_e1 4 -l_e2 6 -l_d 3 -d 0.071 -bs 8 -e 30 -model SET++

# HIER++
# {'nhead': 7, 'embedding_perhead': 28, 'nhid_perhead': 14, 'nlayers_e1': 2, 'nlayers_e2': 4, 'nlayers_d': 6, 'dropout': 0.001088208050525544, 'batch_size': 8, 'epochs': 5, 'model_type': 'HIER++', 'embedding_size': 196, 'nhid': 98, 'log_path': 'running/transformer_cls++/'}

# [I 2020-09-23 08:04:10,493] Trial 22 finished with value: 108.47518434207636 and parameters: {'nhead': 7, 'embedding_perhead': 28, 'nhid_perhead': 14, 'nlayers_e1': 2, 'nlayers_e2': 4, 'nlayers_d': 6, 'dropout': 0.001088208050525544}. Best is trial 22 with value: 108.47518434207636.

# SET++
# {'nhead': 7, 'embedding_perhead': 25, 'nhid_perhead': 13, 'nlayers_e1': 4, 'nlayers_e2': 6, 'nlayers_d': 3, 'dropout': 0.07106661193491001, 'batch_size': 8, 'epochs': 5, 'model_type': 'SET++', 'embedding_size': 175, 'nhid': 91}

# [I 2020-09-15 23:23:07,025] Trial 10 finished with value: 90.55630105889372 and parameters: {'nhead': 7, 'embedding_perhead': 25, 'nhid_perhead': 13, 'nlayers_e1': 4, 'nlayers_e2': 6, 'nlayers_d': 3, 'dropout': 0.07106661193491001}. Best is trial 10 with value: 90.55630105889372.
