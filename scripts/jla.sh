cd /lustre/projects/Research_Project-T116269/scripts 

echo Loading slurm modules 

module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0 
module load GCCcore/12.3.0 
module load CUDA/12.2.2

echo Running ollama server 
ollama serve > ollama_server.log 2>&1 &

echo Running JLA script
python jla_dspy.py

echo JLA script complete