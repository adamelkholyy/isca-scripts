import os
import subprocess
import time
import argparse
import json
import logging
import sys
from pathlib import Path
from evaluate import get_scores
from evaluate import calculate_error

# TODO:
# remove the human-3 exclusive transcripts from the rated_transcripts.txt file

logging.basicConfig( 
    format= "{asctime} {name}: {message}",
    style="{",
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
    stream=sys.stdout,
) 
logging.info(f'Running {__name__}')



parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", "-m",
    dest="model",
    default=None,
    help="model path",
)
parser.add_argument(
    "--temps", "-t",
    dest="temps",
    nargs='*',
    default="0",
    help="temperatures to test",
)
parser.add_argument(
    "--outdir", "-d",
    dest="outdir",
    default=None,
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
    default=None,
    help='number of layers to offload to the GPU\n max layers are automatically configured by default=None (7b: 29, 32b: 65, 70b: 81)',
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
    help="ctsr category to run \n all categories are run configured by default=-1",
)
parser.add_argument(
    "--sys", "-s",
    dest="sys_prompt",
    default=None,
    help="path to system prompt",
)
parser.add_argument(
    "--all", "-a",
    dest="all_transcripts",
    action='store_true',
    help="toggle on/off scoring across all transcripts or exclusively the double-rated ones, default=False",
)
parser.add_argument(
    "--test",
    dest="test",
    action='store_true',
    help="toggle on/off testing mode (uses only 1 file)",
)

args = parser.parse_args()

# set output directory if none specified
if not args.outdir:
    args.outdir = os.path.basename(args.model)

# set ctsr categories (all if none specified)
if (c := int(args.cat)) == -1:
    cats = range(1, 13)
else:
    cats = [c]

# select transcripts for assessment
if args.all_transcripts:
    with open("rated_transcripts.txt", "r", encoding="utf-8") as f:
        files = f.read().split("\n")
else:
    files = [ # double-rated transcripts
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
if args.test: 
    files = ["B127079_101_s02.txt"]


for temp in args.temps:
    start = time.time()
    cat_scores = {f'Category {c}': [] for c in cats}
    
    for file in files:
        for cat in cats:
            dirname = f"{args.outdir}-temp-{temp}/{Path(file).stem}"
            command = [
                "python", "llama_cpp_ctsr.py",
                "-f", f"cobalt-text-txt/{file}",
                "--dir", dirname,
                "--ctsr", f"prompts/cats/cat{cat}.txt",
                "--instruction", args.instruction_prompt,
                "--temp",  temp,
                "--model", args.model,
                "--num-batch", args.num_batch, 
            ]
            command += ["--sys", args.sys_prompt] if args.sys_prompt else []
            command += ["--num-gpu", args.num_gpu] if args.num_gpu else []

            subprocess.run(command)

            # calculate ai vs human error
            logging.info(f'{file} cat {cat} scores:')
            ai_score, human_score = get_scores(dirname, cat=cat)
            error = calculate_error(ai_score, human_score)
            cat_scores[f'Category {cat}'].append({"file": file, "human": human_score, "ai": ai_score, "error": error})
            

    # timing and logging
    raw_time = time.time() - start
    hours, remainder = divmod(raw_time, 3600)
    mins, secs = divmod(remainder, 60)
    experiment_name = os.path.basename(args.outdir)
    logging.info(f"Succesfully ran cts-r on {len(files)} files for {experiment_name} with temp {temp} on cats {cats} in {int(hours)}h {int(mins)}m {int(secs)}s")
    

    # calculate average ai error
    for cat, scores in cat_scores.items():
        total_error = sum([abs(row["error"]) for row in scores])      
        avg_error = round(total_error / len(scores), 2)
        cat_scores[cat].append({"info": {"experiment": experiment_name, "temp": temp, "average error": avg_error}})


    # log average error and output to averages file
    score_data = json.dumps(cat_scores, indent=3)
    logging.info(score_data)
    with open(os.path.join("averages", f'{experiment_name}-{temp}.json'), "w", encoding="utf-8") as f:
        f.write(score_data)

logging.info(f'{__name__} completed succesfully')


