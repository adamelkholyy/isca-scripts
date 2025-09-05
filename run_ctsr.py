import argparse
import json
import logging
import os
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    format="{asctime} {name}: {message}",
    style="{",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logging.info(f"Running {__name__}")

# These scripts inherit logging from config above
from evaluate import calculate_error  # noqa: E402
from evaluate import calculate_ks_alpha  # noqa: E402
from evaluate import get_scores  # noqa: E402
from llama_cpp import run_llama_cpp  # noqa: E402


# Initialise parser
parser = argparse.ArgumentParser(description="Run llama.cpp for CTS-R assessments")

parser.add_argument(
    "--outdir",
    dest="outdir",
    type=str,
    default="ctsr-assessment",
    help="name of output directory",
)

parser.add_argument(
    "--instruction",
    dest="instruction_prompt",
    type=str,
    default="prompts/ctsr-individual.txt",
    help="path to instruction prompt file",
)

parser.add_argument(
    "--sys",
    dest="sys_prompt",
    type=str,
    default="prompts/system-prompt.txt",
    help="path to system prompt",
)

parser.add_argument(
    "--cat",
    dest="cat",
    type=int,
    default=1,
    help="ctsr category to run (default=1.Agenda setting and adherence)",
)

parser.add_argument(
    "--inter-rater-only",
    dest="inter_rater_only",
    action="store_true",
    default=False,
    help="toggle on/off scoring across all transcripts or exclusively the double-rated ones, default is off to run across all 54 transcripts",
)

parser.add_argument(
    "--test",
    dest="test",
    action="store_true",
    help="toggles test mode on to run ctsr assessment on only 1 file",
)

parser.add_argument(
    "--temp",
    "-t",
    dest="temp",
    nargs="*",
    type=float,
    default=[0.7],
    help="temperatures to test, accepts any number of args e.g. --temp 0.8 1.0, default is 0.7",
)

parser.set_defaults(llama_params={})


# Custom args action class for setting llama.cpp params
class AddLLamaParam(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.llama_params[self.dest] = values


parser.add_argument(
    "--model",
    "-m",
    required=True,
    type=str,
    dest="model",
    action=AddLLamaParam,
    help="path to model in .gguf format (if the model is split across multiple .gguf files, simply specify the first one and llama.cpp will find the others)",
)
parser.add_argument(
    "-ngl",
    "--gpu-layers",
    "--n-gpu-layers",
    dest="n_gpu_layers",
    type=int,
    action=AddLLamaParam,
    help="number of layers to offload to the GPU\n max layers are automatically configured by default (deepseek 7b: 29, 32b: 65, 70b: 81, gpt-oss 120b: 37)",
)
parser.add_argument(
    "--batch-size",
    "-b",
    dest="batch_size",
    type=int,
    action=AddLLamaParam,
    help="number of tokens per batch (default=2048, adjust lower if running out of CUDA memory)",
)
parser.add_argument(
    "--ctx-size",
    "-c",
    dest="ctx_size",
    type=int,
    action=AddLLamaParam,
    help="model context, default is 0 for full context",
)

args = parser.parse_args()
llama_params = args.llama_params


gpu_layers_lookup = {
    "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf": 29,
    "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf": 65,
    "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf": 81,
    "gpt-oss-120b-mxfp4-00001-of-00003.gguf": 37,
}

# Set max gpu layers from lookup dict if n_gpu_layers not set
if "n_gpu_layers" not in llama_params:
    model_name = os.path.basename(llama_params["model"])

    if model_name in gpu_layers_lookup:
        llama_params["n_gpu_layers"] = gpu_layers_lookup[model_name]
    else:
        logging.info(
            f"Could not find {model_name} in {gpu_layers_lookup.keys()}, setting n_gpu_layers to 0"
        )
        llama_params["n_gpu_layers"] = 0


# Set transcripts for assessment
if args.inter_rater_only:
    # Use only the 9 transcripts used to establish inter-rater reliability
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
else:
    # Use all 54 available transcripts
    with open("rated_transcripts.txt", "r", encoding="utf-8") as f:
        files = f.read().split("\n")


# Override and use only single file for testing
if args.test:
    logging.debug("Testing mode enabled, using single file only")
    files = ["B127079_101_s02.txt"]


# Run ctsr on multiple transcripts
def assess_transcripts(files, temp):
    cat_scores = {}

    for file in files:
        dirname = os.path.join(f"{args.outdir}-temp-{temp}", Path(file).stem)

        # Run model via llama.cpp
        run_llama_cpp(
            transcript_path=os.path.join("cobalt-text-txt", file),
            ctsr_prompt_path=os.path.join("prompts/cats", f"cat{args.cat}.txt"),
            instruction_prompt_path=args.instruction_prompt,
            system_prompt_path=args.sys_prompt,
            outdir=dirname,
            temp=temp,
            **llama_params,
        )

        # Calculate AI vs human error
        logging.info(f"{file} cat {args.cat} scores:")
        ai_score, human_score = get_scores(dirname, cat=int(args.cat))
        error = calculate_error(ai_score, human_score)
        cat_scores[file] = {"human": human_score, "ai": ai_score, "error": error}

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
        f"Successfully ran cts-r on {len(files)} files for {experiment_name} with temp {temp} on cat {args.cat} in {int(hours)}h {int(mins)}m {int(secs)}s"
    )

    # Calculate average ai error and k's alpha, then log and output to file
    average_error = round(
        sum([abs(transcript_data["error"]) for transcript_data in cat_scores.values()])
        / len(cat_scores),
        2,
    )
    ks_alpha = calculate_ks_alpha(cat_scores)

    cat_scores["info"] = {
        "cat": args.cat,
        "experiment": experiment_name,
        "temperature": temp,
        "average_error": average_error,
        "krippendorff_alpha": ks_alpha,
    }

    score_data = json.dumps(cat_scores, indent=3, default=list)
    logging.info(score_data)

    with open(
        os.path.join("averages", f"{experiment_name}-{temp}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(score_data)


logging.info(f"{__name__} completed succesfully")
