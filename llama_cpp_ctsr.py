import os
import subprocess
import time
import argparse
import logging
import sys
from pathlib import Path


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
    '--file', '-f', 
    dest='file',
    help='path to therapy transcript for ctsr assessment',
)
parser.add_argument(
    '--dir', '-d',
    dest='dir',
    default='./',
    help='results output directory'
)
parser.add_argument(
    '--ctsr', '-c',
    dest='ctsr_prompt',
    default='prompts/ctsr-prompt.txt',
    help='path to ctsr prompt file',
)
parser.add_argument(
    '--instruction', '-i',
    dest='instruction_prompt',
    default='prompts/assessment-prompt.txt',
    help='path to instruction prompt file',
)
parser.add_argument(
    '--sys', '-s',
    dest='system_prompt',
    default='prompts/system-prompt.txt',
    help='path to system prompt file',
)
parser.add_argument(
    '--temp', '-t',
    dest='temp',
    default='0.8',
    help='model temperature (default=0.8)',
)
parser.add_argument(
    '--model', '-m',
    dest='model',
    help='path to model',
)
parser.add_argument(
    '--num-gpu', '-g',
    dest='num_gpu',
    default=None,
    help='number of layers to offload to the GPU\n max layers are automatically configured by default (7b: 29, 32b: 65, 70b: 81)',
)
parser.add_argument(
    '--num-batch', '-b',
    dest='num_batch',
    default='2048',
    help='number of tokens per batch (default=2048, adjust lower if running out of CUDA memory)',
)
args = parser.parse_args()

# set max gpu layers automatically
gpu_layers_lookup = {
    '7B': '29',
    '32B': '65',
    '70B': '81',
}
if not args.num_gpu:
    args.num_gpu = next((v for k, v in gpu_layers_lookup.items() if k in args.model), '0')


# format prompt for llama.cpp
def read_prompt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('\n', '\\')
    return content


# load prompt files
transcript_content = read_prompt(args.file)
ctsr_prompt = read_prompt(args.ctsr_prompt)
system_prompt = read_prompt(args.system_prompt)
base_instruction_prompt = read_prompt(args.instruction_prompt)
instruction_prompt = base_instruction_prompt.replace('[TRANSCRIPT HERE]', transcript_content)


# llama.cpp command
command = [
    './llama-cli',
    '--simple-io',
    '-no-cnv',
    '--no-display-prompt',
    '-sys', args.system_prompt,
    '--model', args.model, 
    '--seed', '42',
    '-c', '0', # '50000',
    '--cache-type-k', 'q8_0',
    '--top-k', '40',
    '--top-p', '0.950',
    '--min-p', '0.050',
    '--temp', args.temp,
    '-ngl', args.num_gpu,
    '-b', args.num_batch,
]


logging.info(f'Running {command[0]} build on {args.file}')
start = time.time()

# run llama.cpp 
full_prompt = f'< | User | > {ctsr_prompt} {instruction_prompt} < | Assistant | >'
command += ['--prompt', full_prompt]
out_bytes = subprocess.check_output(command, cwd='./llama.cpp-gpu/build/bin') # , stderr=subprocess.STDOUT)
output = out_bytes.decode('utf-8')

raw_time = time.time() - start
minutes, seconds = divmod(raw_time, 60)
logging.info(output)


# write outputs to file
def write_to_file(filepath, text):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

# make outdir and format filename
filename = f'{Path(args.file).stem}_{os.path.basename(args.ctsr_prompt)}'
os.makedirs(args.dir, exist_ok=True)

# pprint model parameters for output file
params = '\n'.join([f'{command[i]:<20} = {command[i+1]}' for i in range(command.index('--model') + 2, len(command) - 2, 2)])

nl = "\n"
write_to_file(os.path.join(args.dir, filename), f'{output} {nl}{nl} Model parameters: {nl} {params} {nl} Model: {os.path.basename(args.model)} {nl} Time taken: {minutes}m {seconds:.2f}s')
write_to_file(os.path.join(args.dir, 'instruction-prompt.txt'), base_instruction_prompt)
write_to_file(os.path.join(args.dir, 'system-prompt.txt'), system_prompt)

logging.info(f'{nl} llama.cpp GPU build successfully ran cts-r on {filename} in {minutes}m {seconds:.2f}s, temp: {args.temp}, model: {os.path.basename(args.model)}')
