import os
import subprocess
import time
import argparse

# format prompt for llama.cpp
def read_prompt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("\n", "\\")
    return content


# TODO
# - remove os.chdir
# - update arg names and descriptions (add more aliases, sys -> sys-prompt etc.)

# initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", "-f", 
    dest="file",
    help="path to therpay transcript for assessment",
)
parser.add_argument(
    "--dir", "-d",
    dest="dir",
    help="output directory"
)
parser.add_argument(
    "--ctsr", "-c",
    dest="ctsr_prompt",
    default="prompts/ctsr-prompt.txt",
    help="path to ctsr prompt file",
)
parser.add_argument(
    "--instruction", "-i",
    dest="instruction_prompt",
    default="prompts/assessment-prompt.txt",
    help="path to instruction prompt file",
)
parser.add_argument(
    "--sys", "-s",
    dest="system_prompt",
    default="prompts/system-prompt.txt",
    help="path to system prompt file",
)
parser.add_argument(
    "--temp", "-t",
    dest="temp",
    default="0.8",
    help="model temperature",
)
parser.add_argument(
    "--model", "-m",
    dest="model",
    help="path to model",
)
parser.add_argument(
    "--num-gpu", "-g",
    dest="num_gpu",
    default="65",
    help="number of layers to offload to the GPU (29 for deepseek-7b, 65 for 32b, 81 for 70b)",
)
parser.add_argument(
    "--num-batch", "-b",
    dest="num_batch",
    default="2048",
    help="number of tokens per batch (default=2048, adjust lower if running out of CUDA memory)",
)


args = parser.parse_args()

# load prompt files
os.chdir("/lustre/projects/Research_Project-T116269/")
transcript_content = read_prompt(args.file)
ctsr_prompt = read_prompt(args.ctsr_prompt)
system_prompt = read_prompt(args.system_prompt)
base_instruction_prompt = read_prompt(args.instruction_prompt)
instruction_prompt = base_instruction_prompt.replace("[TRANSCRIPT HERE]", transcript_content)


# llama.cpp interactive mode command
command = [
    "./llama-cli",
    "--simple-io",
    "-no-cnv",
    "-sys", args.system_prompt,
    "--model", args.model, # "/lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", # DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    "--seed", "42",
    "-c", "50000",
    "--cache-type-k", "q8_0",
    "--top-k", "40",
    "--top-p", "0.950",
    "--min-p", "0.050",
    "--temp", args.temp,
    "-ngl", args.num_gpu,
    "-b", args.num_batch,
]


print(f"Running llama.cpp GPU build on {args.file}")
os.chdir("/lustre/projects/Research_Project-T116269/llama.cpp-gpu/build/bin/")
start = time.time()

# run llama.cpp 
full_prompt = f"< | User | > {ctsr_prompt} {instruction_prompt} < | Assistant | >"
command += ["--prompt", full_prompt]
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
output = proc.stdout.read()

raw_time = time.time() - start
minutes, seconds = divmod(raw_time, 60)
print(output)


# format model parameters for output file
start_index = command.index("--model") + 2
params = ""
for i in range(start_index, len(command) - 2, 2):
    params += f"{command[i]:<20} = {command[i+1]}" + "\n"


# write full output to file
os.chdir("/lustre/projects/Research_Project-T116269/assessments")
filename = f"{os.path.basename(args.file)[:-4]}_{os.path.basename(args.ctsr_prompt)}"
os.makedirs(args.dir, exist_ok=True)
outpath = os.path.join(args.dir, filename)

with open(outpath, "w", encoding="utf-8") as f:
    f.write(output + "\n\nParams:\n" + params + "Model: Deepseek-r1 32B \n" + f"Time taken: {minutes}m {seconds:.2f}s")

with open(os.path.join(args.dir, "instruction-prompt.txt"), "w", encoding="utf-8") as f:
    f.write(base_instruction_prompt)

with open(os.path.join(args.dir, "system-prompt.txt"), "w", encoding="utf-8") as f:
    f.write(system_prompt)

print(f"\nllama.cpp GPU build successfully ran cts-r on {filename} in {minutes}m {seconds:.2f}s")
