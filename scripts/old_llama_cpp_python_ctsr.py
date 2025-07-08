import os
import subprocess
import time
import argparse

from llama_cpp import Llama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate


# format prompt for llama.cpp
def read_prompt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # content = content.replace("\n", "")
    content = content.replace("{", "")
    content = content.replace("}", "")
    return content


# initialize parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "--file", "-f", 
    dest="file",
    help="path to therpay transcript for assessment",
)

args = parser.parse_args()

# load prompt files
os.chdir("/lustre/projects/Research_Project-T116269/")
transcript_content = read_prompt(args.file)
ctsr_prompt = read_prompt("prompts/ctsr-prompt.txt")
instruction_prompt = read_prompt("prompts/assessment-prompt.txt")
instruction_prompt = instruction_prompt.replace("[TRANSCRIPT HERE]", transcript_content)

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""


template = """Question: Below is the CTS-R 

Answer: I have read and understood the CTS-R. Now I am ready to assess a therapy transcript using the CTS-R

Question: {question}"""

prompt = PromptTemplate.from_template(template)


# callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# LlamaCpp
llm = LlamaCpp(
    model_path="/lustre/projects/Research_Project-T116269/llama.cpp/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_batch=2048,
    n_ctx=50000,
    max_tokens=10000,
    model_kwargs={'n': -1, 'temp':0.8},
    callback_manager=callback_manager,
    verbose=True,  # verbose is required to pass to the callback manager
)

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ("system", ctsr_prompt),
    ("user", "{question}"),

])

prompt = prompt_template

llm_chain = prompt | llm
question = instruction_prompt
llm_chain.invoke({"question": question})

