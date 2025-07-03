#!/bin/bash
#SBATCH --partition=sq
#SBATCH --account=Research_Project-T116269
#SBATCH --time=48:00:00
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1

#SBATCH --export=ALL      				 	# export all environment variables to the batch job.
#SBATCH -D /lustre/projects/Research_Project-T116269/		# set working directory to .
#SBATCH --mail-type=ALL						# send email at job completion
#SBATCH --mail-user=a.el-kholy@exeter.ac.uk 			# email address
#SBATCH --account Research_Project-T116269    			# research project to submit under. 
#SBATCH --priority=5000

#SBATCH --output=wma_to_mp3.out	   				# submit script's standard-out
#SBATCH --error=wma_to_mp3.err    				# submit script's standard-error
#SBATCH --job-name=wma_to_mp3



echo Starting wma to mp3 conversion...

python wma_to_mp3.py

echo Files converted succesfully.