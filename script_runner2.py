import datetime
import os
import subprocess
import time

os.chdir("/lustre/projects/Research_Project-T116269")

now = datetime.datetime.now()
# print(f"Script runner started at {now.strftime('%Y-%m-%d %H:%M:%S')}")

mins = 0.1
time.sleep(60 * mins)

# run script if flag is set
if os.path.isfile("flag.txt"):
    exit()

print("Running script")


# subprocess.run("./ctsr_experiments.sh", shell=True)

os.chdir("/lustre/projects/Research_Project-T116269/nemo")

subprocess.run("module load Python/3.11.3-GCCcore-12.3.0 & source ./.venv/bin/activate & python anonymise.py -f /lustre/projects/Research_Project-T116269/cobalt-text-txt", shell=True)
# subprocess.run("./whisper-diarization/diarize_test.sh", shell=True)


with open("flag.txt", "w", encoding="utf-8") as f:
    f.write(".")


print("Script completed.")
