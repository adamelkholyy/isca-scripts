import os
import subprocess
import time

os.chdir("/lustre/projects/Research_Project-T116269/llama.cpp")
print("Running llama.cpp interactively with DeepSeek...")

# Prompts
first_prompt = "Hi! What's your name?"
second_prompt = "Tell me a joke."

# llama.cpp interactive mode command
command = [
    "./build/bin/llama-cli",
    "--model", "./models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "--seed", "42",
    "--simple-io",
    "-c", "15000",
    "-co"
]

# Start process
start = time.time()
proc = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)


proc.stdin.write(first_prompt + "\n")
proc.stdin.flush()

output, _ = proc.communicate(input="Tell me a joke")
print(output)

# Finish process
proc.stdin.close()
proc.wait()

print(output)

# Timing
raw_time = time.time() - start
minutes, seconds = divmod(raw_time, 60)
print(f"\nllama.cpp completed in {minutes}m {seconds:.2f}s")
