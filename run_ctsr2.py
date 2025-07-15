import os
import subprocess
import time


os.chdir("/lustre/projects/Research_Project-T116269")

with open("rated_transcripts.txt", "r", encoding="utf-8") as f:
    files = f.read().split("\n")

duplicated_transcripts = [
    "B127079_101_s02.txt",
    "B117020_102_s17.txt",
    "B125013_103_s11.txt",
    "B110023_104_s05.txt",
    "R203155_201_s06.txt",
    "B220037_202_s08.txt",
    "B229030_203_s09.txt",
    "B325006_301_s02.txt",
    "B309019_302_s02.txt",
]
files = duplicated_transcripts


temps = [0.8, 0.2, 0]
for t in temps:
    start = time.time()
    for f in files:
        if os.path.exists(f"cobalt-text-txt/{f}"):
            for i in range(1, 13):
                subprocess.run(f"python llama_cpp_ctsr.py -f cobalt-text-txt/{f} --dir 32b-temp-{t}-decimals/{f[:-4]} --ctsr prompts/cats/cat{i}.txt --instruction prompts/ctsr-individual-decimals.txt --sys prompts/system-prompt.txt --temp {t}", shell=True,)
        else:
            print(f"{f} not found")

    raw_time = time.time() - start
    hours, remainder = divmod(raw_time, 3600)
    mins, secs = divmod(remainder, 60)
    print(f"Succesfully ran cts-r on {len(files)} files  with temp {t} in {int(hours)}h {int(mins)}m {int(secs)}s")

exit()

for f in files:
    if os.path.exists(f"cobalt-text-txt/{f}"):
        # subprocess.run(f"python llama_cpp_python_ctsr.py -f cobalt-text-txt/{f} -v True", shell=True)
        subprocess.run(f"python llama_cpp_ctsr.py -f cobalt-text-txt/{f} --dir llama-cpp-gpu-deepseek-32b-ctsr-individual", shell=True,)
    else:
        print(f"{f} not found")
