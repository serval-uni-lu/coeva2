#!/bin/bash -l

#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p batch
#SBATCH -C broadwell
#SBATCH --time=0-0:10:00
#SBATCH --qos=qos-besteffort
#SBATCH -J RUN_ALL
#SBATCH --mail-type=all
#SBATCH --mail-user=thibault.simonetto@uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
conda activate moeva

"$@"