import subprocess
import time
import datetime
import os

os.chdir("/lustre/projects/Research_Project-T116269")

now = datetime.datetime.now()
print(f"Script runner started at {now.strftime('%Y-%m-%d %H:%M:%S')}")

mins = 1
time.sleep(60 * mins)

# run a python script if flag is set
with open("flag.txt", "r", encoding="utf-8") as f:
    flag = f.read()

if "stop" in flag:
    exit()


# command = "python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0,0.6,0.8,1.0,1.2 --outdir 70b"

print("Running commands")


command = "module load Python/3.11.3-GCCcore-12.3.0 & module load CMake/3.26.3-GCCcore-12.3.0 & module load GCCcore/12.3.0 & module load CUDA/12.2.2"
subprocess.run(command, shell=True)

commands = [
    "python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0.8,1.0,1.2 --outdir 70b --num-gpu 81 --num-batch 1024",
    "python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0.6,1.0,1.2 --outdir 32b",
    "python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --temps 0,0.2,0.6,0.8,1.0,1.2 --outdir 70b-stricter --instruction prompts/ctsr-individual-stricter.txt --num-gpu 81 --num-batch 1024",
    "python run_ctsr.py --model /lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --temps 0,0.2,0.6,0.8,1.0,1.2 --outdir 32b-stricter --instruction prompts/ctsr-individual-stricter.txt",
]

for c in command:
    subprocess.run(command, shell=True)


with open("flag.txt", "w", encoding="utf-8") as f:
    f.write("stop")
