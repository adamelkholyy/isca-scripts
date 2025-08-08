# isca-scripts
Author: Adam El Kholy        
adamelkholy40@yahoo.co.uk        
Feel free to contact me with any queries on how to use any of the scripts here      


Selection of scripts for running AI CBT competency assessment experiments on Exeter's ISCA server. Requires a llama.cpp install with GPU build for use on ISCA's GPU partition. This README serves as a handover guide to whoever wishes to run further ctsr LLM experiments on ISCA. The AI transcription tool, (LINK to hpc-nemo), has its own documentation. 


*These scripts can be found at /lustre/projects/Research_Project-T116269/ on ISCA.* 


## Installing llama.cpp
(Link to llama.cpp gpu build instructions on their github, include also link to the long instructions)
(We want to enable CUDA!)


## Running ctsr assessment experiments on ISCA
To run ctsr experiments you will need to use 2 scripts:
1. ctsr.sh, which runs the ctsr experiments script on ISCA and     
2. run_ctsr.py, which handles the actual ctsr LLM calling and outputs 


ctsr.sh is a standard slurm job running script. Feel free to change the parameters to your liking, however at least one GPU is always necessary. run_ctsr.py takes a number of CLI arguments to alter the llama.cpp parameters, as well as the ctsr prompts and models:

(CLI ARGS HELP)


Explanation of CLI args


## Useful linux commands for these scripts 

Run ollama server and output logs to file
```
ollama serve > logs/ollama_server.log 2>&1 &
```

See all the processes runing on the ollama port
```
netstat -tulpn | grep 11434 
```

Core slurm modules for all scripts
```
module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load GCCcore/12.3.0
module load CUDA/12.2.2
```

See all python processes running on current node (useful if you would like to kill a stuck process/script)
```
ps aux | grep python
```










## ISCA directory structure 
(make this into a table)     


/lustre/projects/Research_Project-T116269/      
- nemo: Folder containing the AI transcription and diarization tool 
- cobalt-audio-mp3: Folder containing the original cobalt audio files
- cobalt-text-txt: Folder containing all of the transcribed cobalt tapes in .txt format
- prompts: 
- assessments: 
- 


## Other scripts 
JLA 

dspy 
(didn't work well for ctsr, worked for JLA)

ollama 


misc extras: (one liners explaining remaining scripts)


## Why llama.cpp? Why not ollama, lmstudio, llama-cpp-python etc. ... 
Ollaam: no control, slow 
lmstudio: great, proprietary 
llama-cpp-python: Is this even building llama.cpp properly?? Would rather have our own control 


## How to use buffer.sh, buffer.py and script_runner.py 
Due to the long queue times on the GPU node, we utilise a buffered script running system, such that we can change out the current script running on the static GPU node as and when we would like. To use this simply edit `script_runner.py` to run the script you would like. Once the script is complete, a file called `flag.txt` will be created. To run `script_runner.py` again, simply delete `file.txt`. If you need to kill a process, you can log onto the GPU node running your script and use `ps ax | grep python` to view all currently running python proceses. You may then kill the process of your stuck script (node this is not the process of `buffer.py` or of `script_runner.py`, rather this is the process running whatever script you've called from `script_runner.py`).