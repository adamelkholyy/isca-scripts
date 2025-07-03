import subprocess
import time
import os

# Auto generate prompt file from csv (once transcriptions are complete; use pandas)
# Tune the justfications prompt to give enough detail

print("Starting ollama")
subprocess.run("ollama serve > ollama_server.log 2>&1 &", shell=True)
print("Ollama started succesfully")

prompt_file = "/lustre/projects/Research_Project-T116269/nemo/ctsr_justification_prompt.txt"
outpath = "/lustre/projects/Research_Project-T116269/nemo/ctsr_justification.txt"

start = time.time()
subprocess.run(f"ollama run deepseek-32b < '{prompt_file}' >> '{outpath}'", shell=True)
complete_time = time.time() - start

with open(outpath, "a", encoding="utf-8") as f:
    f.write(
        "\n"
        + "Model: deepseek-32b \n"
        + f"Time taken: {complete_time:.2f} seconds"
        + "\n"
    )

print(f"Completed in {complete_time:.2f} seconds")
