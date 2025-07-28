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
args = parser.parse_args()

# python llama_cpp_speed_test.py -f cobalt-text-txt/B127079_101_s02.txt -c prompts/cats/cat1.txt -i prompts/ctsr-individual.txt 

# load prompt files
os.chdir("/lustre/projects/Research_Project-T116269/")
transcript_content = read_prompt(args.file)
ctsr_prompt = read_prompt(args.ctsr_prompt)
instruction_prompt = read_prompt(args.instruction_prompt)
instruction_prompt = instruction_prompt.replace("[TRANSCRIPT HERE]", transcript_content)


# llama.cpp interactive mode command
command = [
    "./llama-cli",
    "--simple-io",
    "-no-cnv",
    "--model", "/lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", # DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    "--seed", "42",
    "-c", "50000",
    "--cache-type-k", "q8_0",
    "--n-gpu-layers", "29",
    "--top-k", "40",
    "--top-p", "0.950",
    "--min-p", "0.050",
    "--temp", "0.800",
    "-ngl", "65",
]
command += ["-sys", read_prompt(args.system_prompt)] if args.system_prompt else [] 


# start process
os.chdir("/lustre/projects/Research_Project-T116269/llama.cpp-gpu/build/bin/")
full_prompt = f"< | User | > {ctsr_prompt} {instruction_prompt} < | Assistant | >"
print(f"Running llama.cpp GPU build on {args.file}")
start = time.time()


"""
proc = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# prompt model single-turn
proc.stdin.write(full_prompt + "\n")
proc.stdin.flush()

for line in proc.stdout:
    print(line, end="")  # real-time output to console
"""


command += ["--prompt", full_prompt]
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
output = proc.stdout.read()
print(output)


raw_time = time.time() - start
minutes, seconds = divmod(raw_time, 60)


command = command[3:][:-2]
params = ""
for i in range(len(command) - 2):
    if i % 2 != 0:
        continue
    params += f"{command[i]:<20} = {command[i+1]}" + "\n"

print(params)
print(f"\nllama.cpp GPU build successfully ran {args.ctsr_prompt} on {args.file} in {minutes}m {seconds:.2f}s")
