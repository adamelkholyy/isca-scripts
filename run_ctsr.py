import os
import time
import argparse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    format="{asctime} {name}: {message}",
    style="{",
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logging.info(f'Running {__name__}')

# These scripts inherit logging from config above
from evaluate import get_scores         # noqa: E402
from evaluate import calculate_error    # noqa: E402
from llama_cpp import run_llama_cpp     # noqa: E402


# TODO
# pprint cli args and include in README

# Initialise parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--outdir",
    dest="outdir",
    default="model-outputs",
    help="name of output directory",
)
parser.add_argument(
    "--instruction",
    dest="instruction_prompt",
    default="prompts/ctsr-individual.txt",
    help="path to instruction prompt file",
)
parser.add_argument(
    "--sys",
    dest="sys_prompt",
    default="prompts/system-prompt.txt",
    help="path to system prompt",
)
parser.add_argument(
    "--cat",
    dest="cat",
    default=1,
    help="ctsr category to run (default=1.Agenda setting and adherence)",
)
parser.add_argument(
    "--all",
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
# Group llama.cpp parameters
llama_parser = parser.add_argument_group("llama")
llama_parser.add_argument(
    "--model", "-m",
    required=True,
    dest="model",
    help="model path",
)
llama_parser.add_argument(
    "--temp", "-t",
    dest="temp",
    nargs='*',
    default=[0.7],
    help="temperatures to test, accepts any number of args e.g. --temp 0.8 1.0",
)
llama_parser.add_argument(
    "-ngl", "--gpu-layers", "--n-gpu-layers",
    dest="n_gpu_layers",
    default=None,
    help='number of layers to offload to the GPU\n max layers are automatically configured by default=None (7b: 29, 32b: 65, 70b: 81)',
)
llama_parser.add_argument(
    "--batch-size", "-b",
    dest="batch_size",
    help="number of tokens per batch (default=2048, adjust lower if running out of CUDA memory)",
)
llama_parser.add_argument(
    "--ctx-size", "-c",
    dest="ctx_size",
    help='model context, default=0 for full context',
)

args = parser.parse_args()


# Set max gpu layers automatically (0 if model not found in lookup)
gpu_layers_lookup = {
    'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf': 29,
    'DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf': 65,
    'DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf': 81,
}
if not args.n_gpu_layers:
    args.n_gpu_layers = next(
        (v for k, v in gpu_layers_lookup.items() if k == args.model), 0)

# Select transcripts for assessment
if args.all_transcripts:
    with open("rated_transcripts.txt", "r", encoding="utf-8") as f:
        files = f.read().split("\n")
else:
    # 9 transcripts used to establish inter-rater reliability
    files = [
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

# Use only single file for testing
if args.test:
    logging.debug('Testing mode enabled, using single file only')
    files = ["B127079_101_s02.txt"]


# Get llama.cpp params from args
llama_params = {arg.dest: value for arg in llama_parser._group_actions if (
    value := args.__dict__[arg.dest])}


# Run ctsr on multiple transcripts
def assess_transcripts(files, temp):
    cat_scores = {}
    llama_params['temp'] = temp

    for file in files:
        dirname = os.path.join(f"{args.outdir}-temp-{temp}", Path(file).stem)

        # Run model via llama.cpp
        run_llama_cpp(
            transcript_path=os.path.join("cobalt-text-txt", file),
            ctsr_prompt_path=os.path.join(
                "prompts/cats/", f"cat{args.cat}.txt"),
            instruction_prompt_path=args.instruction_prompt,
            system_prompt_path=args.sys_prompt,
            outdir=dirname,
            **llama_params,
        )

        # Calculate ai vs human error
        logging.info(f'{file} cat {args.cat} scores:')
        ai_score, human_score = get_scores(dirname, cat=int(args.cat))
        error = calculate_error(ai_score, human_score)
        cat_scores[file] = {"human": human_score,
                            "ai": ai_score, "error": error}

    return cat_scores


# Iterate over varying model temperatures per experiment
for temp in args.temp:
    start = time.time()

    # Run ctsr assessment across transcripts at given temp
    cat_scores = assess_transcripts(files, temp)

    # Timing and logging
    raw_time = time.time() - start
    hours, remainder = divmod(raw_time, 3600)
    mins, secs = divmod(remainder, 60)
    experiment_name = os.path.basename(args.outdir)
    logging.info(
        f"Succesfully ran cts-r on {len(files)} files for {experiment_name} with temp {temp} on cat {args.cat} in {int(hours)}h {int(mins)}m {int(secs)}s")

    # Calculate average ai error, log and output to file
    average_error = round(
        sum([abs(v["error"]) for v in cat_scores.values()]) / len(cat_scores), 2)
    cat_scores["info"] = {"cat": args.cat, "experiment": experiment_name,
                          "temp": temp, "average error": average_error}

    score_data = json.dumps(cat_scores, indent=3, default=list)
    logging.info(score_data)

    with open(os.path.join("averages", f'{experiment_name}-{temp}.json'), "w", encoding="utf-8") as f:
        f.write(score_data)


logging.info(f'{__name__} completed succesfully')
