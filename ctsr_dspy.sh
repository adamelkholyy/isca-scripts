



cd /lustre/projects/Research_Project-T116269 

echo Loading slurm modules 

module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0 
module load GCCcore/12.3.0 
module load CUDA/12.2.2



echo Running ctsr dspy



python ctsr_dspy.py


echo ctsr dspy complete