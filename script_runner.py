import subprocess
import time
import datetime
import os

os.chdir("/lustre/projects/Research_Project-T116269")

now = datetime.datetime.now()
print(f"Script runner started at {now.strftime("%Y-%m-%d %H:%M:%S")}")

time.sleep(60)

# run a python script if flag is set
with open("flag.txt", "r", encoding="utf-8") as f:
    flag = f.read()

if "stop" in flag:
    exit()




with open("flag.txt", "w", encoding="utf-8") as f:
    f.write("stop")

