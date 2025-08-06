



cd /lustre/projects/Research_Project-T116269 

echo Loading modules 

module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0 
module load GCCcore/12.3.0 
module load CUDA/12.2.2



echo Running JLA code


python jla_dspy.py

echo JLA script complete