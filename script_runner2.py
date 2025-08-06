import subprocess
import time
import datetime
import os

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


subprocess.run("./jla.sh", shell=True)
# subprocess.run("./ctsr_dspy.sh", shell=True)


with open("flag.txt", "w", encoding="utf-8") as f:
    f.write(".")


print("Script completed.")