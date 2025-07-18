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
    default=0.8,
    help="model temperature",
)

args = parser.parse_args()

# load prompt files
os.chdir("/lustre/projects/Research_Project-T116269/")
transcript_content = read_prompt(args.file)
ctsr_prompt = read_prompt(args.ctsr_prompt)
instruction_prompt_raw = read_prompt(args.instruction_prompt)
instruction_prompt = instruction_prompt_raw.replace("[TRANSCRIPT HERE]", transcript_content)


# llama.cpp interactive mode command
command = [
    "./llama-cli",
    "--simple-io",
    "-no-cnv",
    "--model", "/lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", # DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    "--seed", "42",
    "-c", "50000",
    "--cache-type-k", "q8_0",
    "--top-k", "40",
    "--top-p", "0.950",
    "--min-p", "0.050",
    "--temp", f"{args.temp}",
    "-ngl", "65",
]
command += ["-sys", read_prompt(args.system_prompt)] if args.system_prompt else [] 


# start process
os.chdir("/lustre/projects/Research_Project-T116269/llama.cpp-gpu/build/bin/")
print(f"Running llama.cpp GPU build on {args.file}")
start = time.time()
full_prompt = f"< | User | > {ctsr_prompt} {instruction_prompt} < | Assistant | >"

# run command
command += ["--prompt", full_prompt]
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
output = proc.stdout.read()
print(output)

raw_time = time.time() - start
minutes, seconds = divmod(raw_time, 60)



# write output to file
os.chdir("/lustre/projects/Research_Project-T116269/assessments")
filename = f"{os.path.basename(args.file)[:-4]}_{os.path.basename(args.ctsr_prompt)}"
os.makedirs(args.dir, exist_ok=True)
outpath = os.path.join(args.dir, filename)

command = command[3:][:-2]
params = ""
for i in range(len(command) - 2):
    if i % 2 != 0:
        continue
    params += f"{command[i]:<20} = {command[i+1]}" + "\n"

with open(outpath, "w", encoding="utf-8") as f:
    f.write(output + "\n\nParams:\n" + params + "Model: Deepseek-r1 32B \n" + f"Time taken: {minutes}m {seconds:.2f}s")

with open(os.path.join(args.dir, "instruction-prompt.txt"), "w", encoding="utf-8") as f:
    f.write(instruction_prompt_raw)


print(f"\nllama.cpp GPU build successfully ran cts-r on {filename} in {minutes}m {seconds:.2f}s")
