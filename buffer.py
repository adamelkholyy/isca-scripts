import subprocesses
import time

while True:
    time.sleep(1)
    subprocesses.run("python script_runner.py", shell=True)