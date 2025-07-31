import os
import subprocess
import time
import argparse
from evaluate import get_scores
from evaluate import calculate_error

os.chdir("/lustre/projects/Research_Project-T116269")

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
    default="./",
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
parser.add_argument(
    "--cat", "-c",
    dest="cat",
    default="-1",
    help="ctsr category to run",
)
parser.add_argument(
    "--sys", "-s",
    dest="sys_prompt",
    default=None,
    help="path to system prompt",
)




args = parser.parse_args()

if not args.outdir:
    args.outdir = os.path.basename(args.model)

if (c := int(args.cat)) == -1:
    cats = range(1, 13)
else:
    cats = [c]


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
    total_cat_errors = {c: 0 for c in cats}
    for file in files:
        for cat in cats:
            dirname = f"{args.outdir}-temp-{temp}/{file[:-4]}"
            command = [
                "python", "llama_cpp_ctsr.py",
                "-f", f"cobalt-text-txt/{file}",
                "--dir", dirname,
                "--ctsr", f"prompts/cats/cat{cat}.txt",
                "--instruction", args.instruction_prompt,
                "--temp",  temp,
                "--model", args.model,
                "--num-gpu", args.num_gpu, 
                "--num-batch", args.num_batch, 
            ]
            command += ["--sys", args.sys_prompt] if args.sys_prompt else []
            subprocess.run(command)

            ai_score, human_score = get_scores(dirname, cat=cat)
            error = calculate_error(ai_score, human_score)
            total_cat_errors[cat] += error

    raw_time = time.time() - start
    hours, remainder = divmod(raw_time, 3600)
    mins, secs = divmod(remainder, 60)
    print(f"Succesfully ran cts-r on {len(files)} files in {args.outdir} with temp {temp} on cats {cats} in {int(hours)}h {int(mins)}m {int(secs)}s")

    print("Average errors:")
    for cat, total_err in total_cat_errors.items():
        print(f"Cat {cat}:     {total_err/len(files)}")





