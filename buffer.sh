#!/bin/bash
#SBATCH --export=ALL      				 		# export all environment variables to the batch job.
#SBATCH --partition gpu      					# submit to the gpu queue
#SBATCH -D /lustre/projects/Research_Project-T116269	# set working directory to .
#SBATCH --mail-type=ALL						# send email at job completion
#SBATCH --mail-user=a.el-kholy@exeter.ac.uk 			# email address
#SBATCH --time=19-23:00:00    					# maximum wall time for the job.
#SBATCH --account Research_Project-T116269    		# research project to submit under. 
#SBATCH --priority=50000

#SBATCH --ntasks-per-node=16        				# specify number of processors per node
#SBATCH --gres=gpu:1							# num gpus	
#SBATCH --mem=16G								# requested memory	

#SBATCH --output=logs/log.out   						# submit script's standard-out
#SBATCH --error=logs/log.err    						# submit script's standard-error
#SBATCH --job-name=ai-cbt2


cd /lustre/projects/Research_Project-T116269

echo Loading modules...

module load Python/3.11.3-GCCcore-12.3.0                                                   
module load CMake/3.26.3-GCCcore-12.3.0
module load GCCcore/12.3.0
module load CUDA/12.2.2

echo Running script...



python buffer2.py

echo Script complete.

