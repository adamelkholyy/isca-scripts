#!/bin/bash
#SBATCH --export=ALL      				 		# export all environment variables to the batch job.
#SBATCH --partition gpu      					# submit to the gpu queue
#SBATCH -D /lustre/projects/Research_Project-T116269	# set working directory to .
#SBATCH --mail-type=ALL						# send email at job completion
#SBATCH --mail-user=a.el-kholy@exeter.ac.uk 			# email address
#SBATCH --time=19-23:00:00    					# maximum wall time for the job.
#SBATCH --account Research_Project-T116269    		# research project to submit under. 
#SBATCH --priority=5000

#SBATCH --nodes=1                                  		# specify number of nodes.
#SBATCH --ntasks-per-node=16        				# specify number of processors per node
#SBATCH --gres=gpu:1							# num gpus	
#SBATCH --mem=16G								# requested memory	

#SBATCH --output=logs/ctsr.out   						# submit script's standard-out
#SBATCH --error=logs/ctsr.err    						# submit script's standard-error
#SBATCH --job-name=ctsr


cd /lustre/projects/Research_Project-T116269

echo Loading modules...
module load CMake/3.26.3-GCCcore-12.3.0
module load GCCcore/12.3.0
module load CUDA/12.2.2

echo Running script...

python ctsr_comparison.py

echo Script complete.
