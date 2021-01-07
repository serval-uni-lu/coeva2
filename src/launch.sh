#!/bin/bash -l

#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p batch
#SBATCH --time=0-2:00:00
#SBATCH -J COEVA2-REATTACK 
#SBATCH --mail-type=all
#SBATCH --mail-user=thibault.simonetto@uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
module purge
module restore python3
source ../../coeva2/adv/bin/activate

python "$@"
# Your more useful application can be started below!
#  dos2unix job.sh sbatch job.sh
