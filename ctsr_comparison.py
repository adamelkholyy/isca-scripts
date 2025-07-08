import os
import subprocess

os.chdir("/lustre/projects/Research_Project-T116269")

with open("rated_transcripts.txt", "r", encoding="utf-8") as f:
    files = f.read().split("\n")

for f in files:
    if os.path.exists(f"cobalt-text-txt/{f}"):
        # subprocess.run(f"python llama_cpp_python_ctsr.py -f cobalt-text-txt/{f} -v True", shell=True)
        subprocess.run(f"python llama_cpp_ctsr.py -f cobalt-text-txt/{f}", shell=True)
    else:
        print(f"{f} not found")

