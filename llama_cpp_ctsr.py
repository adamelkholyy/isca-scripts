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

args = parser.parse_args()

# load prompt files
os.chdir("/lustre/projects/Research_Project-T116269/")
transcript_content = read_prompt(args.file)
ctsr_prompt = read_prompt("prompts/ctsr-prompt.txt")
instruction_prompt = read_prompt("prompts/assessment-prompt.txt")
instruction_prompt = instruction_prompt.replace("[TRANSCRIPT HERE]", transcript_content)

# print(ctsr_prompt)
# print(instruction_prompt)

# llama.cpp interactive mode command
os.chdir("/lustre/projects/Research_Project-T116269/llama.cpp-gpu/build/bin/")
command = [
    "./llama-cli",
    "--model", "/lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "--seed", "42",
    "--simple-io",
    "-co",
    "-c", "25000",
    "--cache-type-k", "q8_0",
    "--n-gpu-layers", "29",
    "--top-k", "40",
    "--top-p", "0.950",
    "--min-p, "0.050",
    "--temp", "0.200",
]

# start process
print(f"Running llama.cpp GPU build on {args.file}")
start = time.time()
proc = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# prompt model
proc.stdin.write(ctsr_prompt + "\n")
proc.stdin.flush()
output, _ = proc.communicate(input=instruction_prompt)
print(output)
raw_time = time.time() - start
minutes, seconds = divmod(raw_time, 60)


# write output to file
os.chdir("/lustre/projects/Research_Project-T116269/")
filename = os.path.basename(args.file)
outpath = os.path.join("assessments", filename)
with open(outpath, "w", encoding="utf-8") as f:
    f.write(f"Time taken: {minutes}m {seconds:.2f}s \n" + output)

# finish process
proc.stdin.close()
proc.wait()

# timing
print(f"\nllama.cpp GPU build completed successfully in {minutes}m {seconds:.2f}s")
