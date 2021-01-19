#!/bin/bash -l

#SBATCH -o %x_%j.out
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p batch
#SBATCH --time=0-1:00:00
#SBATCH -J WEIGHT-SEARCH
#SBATCH --mail-type=all
#SBATCH --mail-user=thibault.simonetto@uni.lu

echo "Hello from the batch queue on node ${SLURM_NODELIST} for neural architecture generation"
module purge
module load swenv/default-env/v1.1-20180716-production lang/Python/3.6.4-foss-2018a math/Gurobi/8.1.1-intel-2018a-Python-3.6.4
source ../../coeva2/adv/bin/activate

python "$@"
# Your more useful application can be started below!
#  dos2unix job.sh sbatch job.sh
