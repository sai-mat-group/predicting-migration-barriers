#!/bin/bash
#SBATCH --job-name=gvrh_100k                     # Job name
#SBATCH --export=ALL				# Export environment variables
#SBATCH --qos=normal				# See qos documentation below
#SBATCH --nodes=1                   		# Run on a single node (only 1 available on this machine)
#SBATCH --ntasks=24                   		# Max number of tasks. Ask for either 24 or 48.
#SBATCH --no-requeue				# Don't requeue job unneccessarily
#SBATCH --time=23:59:00               		# Time limit hrs:min:sec
#SBATCH --output=job_%j.log   			# Standard output from SLURM and error log

####################################
#Available QOS options are
#debug: < 1hr jobs
#short: < 6hr jobs
#normal: < 1day jobs
#medium: < 2day jobs
#long: < 1week jobs
#Jobs whose walltimes don't match with appropriate QoS will be rejected by the queuing system
#Default qos is normal (if left unspecified in the header SBATCH command above)

####################################

source /opt/apps/anaconda3/etc/profile.d/conda.sh
conda activate alignn

cd 10/
python3 load-freeze-weights.py 
cd ..

cd 100/
python3 load-freeze-weights.py
cd ..

cd 200/
python3 load-freeze-weights.py
cd ..

cd 500/
python3 load-freeze-weights.py
cd ..

cd 800/
python3 load-freeze-weights.py
cd ..

