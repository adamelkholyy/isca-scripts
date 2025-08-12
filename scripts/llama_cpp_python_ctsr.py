import argparse
import os
import time

from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import (CallbackManager,
                                      StreamingStdOutCallbackHandler)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from llama_cpp import Llama


# Format prompt for llama.cpp
def read_prompt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("{", "")
    content = content.replace("}", "")
    return content


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    "-f",
    dest="file",
    help="path to therpay transcript for assessment",
)
parser.add_argument(
    "--verbose",
    "-v",
    dest="verbose",
    default=False,
    help="toggle verbose output",
)
args = parser.parse_args()


# Load prompts
transcript_content = read_prompt(args.file)
ctsr_prompt = read_prompt("prompts/ctsr-prompt.txt")
assistant_prompt = read_prompt("prompts/assistant-prompt.txt")
instruction_prompt = read_prompt("prompts/assessment-prompt.txt")
instruction_prompt = instruction_prompt.replace("[TRANSCRIPT HERE]", transcript_content)

prompt_template = ChatPromptTemplate(
    [
        ("system", ctsr_prompt),
        ("user", "{question}"),
        ("assistant", assistant_prompt),
    ]
)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Llama.cpp parameters
llm = LlamaCpp(
    model_path="/lustre/projects/Research_Project-T116269/llama.cpp/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    seed=42,
    n_gpu_layers=-1,
    n_batch=4096,
    n_ctx=50000,
    max_tokens=100000,
    top_k=40,
    repeat_penalty=1.1,
    top_p=0.95,
    model_kwargs={"n": -1, "temp": 0.8, "min_p": 0.05},
    verbose=args.verbose,
    callback_manager=callback_manager if args.verbose else None,
)


# Run llama.cpp with langchain
llm_chain = prompt_template | llm
print(f"Running CTS-R assessment on {args.file} using llama-cpp-python with GPU build")
start = time.time()
model_output = llm_chain.invoke({"question": instruction_prompt})
end = time.time() - start
mins, secs = divmod(end, 60)


# Write outputs to file
filename = os.path.basename(args.file)
outpath = os.path.join("assessments", filename)
with open(outpath, "w", encoding="utf-8") as f:
    f.write(f"Time taken: {int(mins)}m {secs:.2f}s" + "\n\n" + model_output)

print(f"Completed in {int(mins)}m {secs:.2f}s")
