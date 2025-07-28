



cd /lustre/projects/Research_Project-T116269 

echo Loading slurm modules 

module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0 
module load GCCcore/12.3.0 
module load CUDA/12.2.2



echo Running ctsr experiments


# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.8,1.0,1.2 --outdir 70b --num-gpu 81 --num-batch 1024
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0,0.2,0.6,0.8,1.0,1.2 --outdir 32b-stricter --instruction prompts/ctsr-individual-stricter.txt --num-gpu 65
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0,0.2,0.6,0.8,1.0,1.2 --outdir 70b-stricter --instruction prompts/ctsr-individual-stricter.txt --num-gpu 81 
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.6,1.0,1.2 --outdir 32b --num-gpu 65

python feedback_run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.8,1.0 --outdir feedback-32b --num-gpu 65


echo ctsr experiments complete