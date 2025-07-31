import os
import subprocess
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", "-m",
    dest="model",
    help="path to model in .gguf format",
)
parser.add_argument(
    "--temps", "-t",
    dest="temps",
    default="0,0.6,0.8",
    help="comma separated string of temperatures to test. (default = '0,0.6,0.8')",
)
parser.add_argument(
    "--outdir", "-d",
    dest="outdir",
    default="",
    help="name of output directory",
)
parser.add_argument(
    "--instruction", "-i",
    dest="instruction_prompt",
    default="prompts/ctsr-individual.txt",
    help="path to instruction prompt file",
)
parser.add_argument(
    "--num-gpu", "-g",
    dest="num_gpu",
    default="65",
    help="number of layers to offload to the GPU (29 for deepseek-7b, 65 for 32b, 81 for 70b)",
)
parser.add_argument(
    "--num-batch", "-b",
    dest="num_batch",
    default="2048",
    help="number of tokens per batch (default=2048, adjust lower if running out of CUDA memory)",
)

args = parser.parse_args()

if not args.outdir:
    args.outdir = os.path.basename(args.model)

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


for temp in args.temps.split(","):
    start = time.time()
    for f in files:
        for i in range(1, 13):
            filename = f"{args.outdir}-temp-{temp}/{f[:-4]}"
            # subprocess.run(f"python llama_cpp_ctsr.py -f cobalt-text-txt/{f} --dir {filename} --ctsr prompts/cats/cat{i}.txt --instruction {args.instruction_prompt} --sys prompts/system-prompt.txt --temp {temp} --model {args.model} --num-gpu {args.num_gpu} --num-batch {args.num_batch}", shell=True,)
            command = [
                "python", "llama_cpp_ctsr.py",
                "-f", f"cobalt-text-txt/{f}",
                "--dir", filename,
                "--ctsr", f"prompts/cats/cat{i}.txt",
                "--instruction", args.instruction_prompt,
                "--temp",  temp,
                "--model", args.model,
                "--num-gpu", args.num_gpu, 
                "--num-batch", args.num_batch, 
            ]
            subprocess.run(command)


    raw_time = time.time() - start
    hours, remainder = divmod(raw_time, 3600)
    mins, secs = divmod(remainder, 60)
    print(f"Succesfully ran cts-r on {len(files)} files  with temp {temp} in {int(hours)}h {int(mins)}m {int(secs)}s")



