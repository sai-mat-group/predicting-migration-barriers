#!/bin/csh
# ---------------------------
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:59:59
#SBATCH --job-name=cfe_from_6
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=gpusinglenode 
#---------------------------

#---------------------------
#**READ-THE-FOLLOWING-ON-SBATCH-COMMANDS-BEFORE-RUNNING-JOBS**#
#gres=gpu:n asks n gpu cards for a job. One gpu card can handle reasonably sized systems (~50 atoms)
#nodes=N asks N GPU nodes. One node has 2 gpu cards. Current executable doesn't work on multiple nodes as far as I know.
#ntasks-per-node=M asks for M "mpi" tasks per GPU node. Always keep this number the same as gres=gpu:n, i.e., have same mpi tasks per GPU card asked
#The partition and wall-time aren't fully optimized for any single system yet
#---------------------------

#---------------------------
#Possible partitions to be assigned in #SBATCH --partition
#small: Max 4 nodes, wall time 3 days, max running jobs per user=5, priority=0
#medium: Max 16 nodes, wall time 2 days, max running jobs per user=3, priority=50(>small)
#large: Max 32 nodes, wall time 1 day, max running jobs per user=1, priority=100 (>medium)
#debug: <Unclear>
#highmemory: <Unclear>
#---------------------------

#---------------------------
#Useful commands
#sbatch submit_script.csh -- Submit job
#scancel <job_name> or scancel <job_id> - Delete a job
#squeue - List out current jobs that are running or queued
#sinfo - Lists status of resources (how many small, medium, large, etc. jobs running)
#---------------------------


#---------------------------
#Don't change the following lines
source /home/phd/20/metdevi/anaconda3/etc/profile.d/conda.csh
conda activate alignn-nodgl

setenv GIT_DISCOVERY_ACROSS_FILESYSTEM 1
setenv I_MPI_FABRICS shm:ofi
setenv OMP_NUM_THREADS 1
#---------------------------

#Change into job submission directory
cd $SLURM_SUBMIT_DIR

python3 read-pmg-matb.py

#Finished
