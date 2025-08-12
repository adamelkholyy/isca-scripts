import subprocess
import time

while True:
    time.sleep(1)
    subprocess.run("python script_runner2.py", shell=True)
