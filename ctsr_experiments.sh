



cd /lustre/projects/Research_Project-T116269 

echo bash: Loading slurm modules 

module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0 
module load GCCcore/12.3.0 
module load CUDA/12.2.2



echo bash: Running ctsr experiments


# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.8,1.0,1.2 --outdir 70b --num-gpu 81 --num-batch 1024
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0,0.2,0.6,0.8,1.0,1.2 --outdir 32b-stricter --instruction prompts/ctsr-individual-stricter.txt --num-gpu 65
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0,0.2,0.6,0.8,1.0,1.2 --outdir 70b-stricter --instruction prompts/ctsr-individual-stricter.txt --num-gpu 81 
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.6,1.0,1.2 --outdir 32b --num-gpu 65

# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.6,0.8,1.0 --outdir 70b-agenda --num-gpu 81 --cat 1 --sys prompts/dspy_agenda.txt

# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 70b-decimal-collab --num-gpu 81 --cat 3 --instruction prompts/ctsr-individual-decimals.txt
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 70b-decimal-pacing --num-gpu 81 --cat 4 --instruction prompts/ctsr-individual-decimals.txt
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 70b-decimal-effect --num-gpu 81 --cat 5 --instruction prompts/ctsr-individual-decimals.txt
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 70b-decimal-guided --num-gpu 81 --cat 8 --instruction prompts/ctsr-individual-decimals.txt

# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 32b-decimal-collab --num-gpu 65 --cat 3 --instruction prompts/ctsr-individual-decimals.txt
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 32b-decimal-pacing --num-gpu 65 --cat 4 --instruction prompts/ctsr-individual-decimals.txt
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 32b-decimal-effect --num-gpu 65 --cat 5 --instruction prompts/ctsr-individual-decimals.txt
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.0,0.6,0.8,1.0,1.2,2.0 --outdir 32b-decimal-guided --num-gpu 65 --cat 8 --instruction prompts/ctsr-individual-decimals.txt



# test
# python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf  --outdir assessments/test --cat 8 


python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.0 0.6 0.8 1.0 1.2 2.0 --outdir 32b-revised-collab --cat 3 --instruction prompts/revised-ctsr-individual.txt
python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.0 0.6 0.8 1.0 1.2 2.0 --outdir 70b-revised-collab --cat 3 --instruction prompts/revised-ctsr-individual.txt

echo bash: ctsr experiments complete