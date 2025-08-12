import subprocess
import time

while True:
    time.sleep(1)
    subprocess.run("python script_runner.py", shell=True)
