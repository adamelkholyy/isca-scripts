import os
import subprocess
import time
import logging
from pathlib import Path


logger = logging.getLogger(__name__)
logger.info(f'Running {__name__}')


# Format prompt for llama.cpp (does not accept newlines, \ instead)
def read_prompt(filepath: str | os.PathLike):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('\n', '\\')
    return content


# Write outputs to file
def write_to_file(filepath: str | os.PathLike, text: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


# Call llama.cpp for ctsr assessment
def run_llama_cpp(
    transcript_path: str | os.PathLike,
    ctsr_prompt_path: str | os.PathLike,
    instruction_prompt_path: str | os.PathLike,
    system_prompt_path: str | os.PathLike,
    outdir: str | os.PathLike,
    model: str | os.PathLike,
    temp: float = 0.7,
    n_gpu_layers: int = 0,
    batch_size: int = 2048,
    ctx_size: int = 0,
    seed: int = 42,
    cache_type_k: str = "q8_0",
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
) -> str:
    """

    Run CTS-R evaluation on a transcript using llama.cpp.

    Args:
        transcript_path (str | os.PathLike): Path to the transcript file.
        ctsr_prompt_path (str | os.PathLike): Path to the CTS-R category prompt file.
        instruction_prompt_path (str | os.PathLike): Path to the instruction prompt file (includes placeholder for transcript).
        system_prompt_path (str | os.PathLike): Path to the system prompt file.
        outdir (str | os.PathLike): Directory to save the output files.
        model (str | os.PathLike): Path to the llama.cpp model file.
        temp (float, optional): Temperature for sampling. Default is 0.7.
        n_gpu_layers (int, optional): Number of layers to offload to GPU. Default is 0.
        batch_size (int, optional): Token batch size. Default is 2048.
        ctx_size (int, optional): Model context size. Default is 0 (use full model context).
        seed (int, optional): Random seed for reproducibility. Default is 42.
        cache_type_k (str, optional): Cache type used for key layers. Default is "q8_0".
        top_k (int, optional): Top-K sampling parameter. Default is 40.
        top_p (float, optional): Top-P sampling parameter. Default is 0.95.
        min_p (float, optional): Minimum probability for token sampling. Default is 0.05.

    Returns:
        output (str): Model output
    
    """

    # Load prompt files
    transcript_content = read_prompt(transcript_path)
    ctsr_prompt = read_prompt(ctsr_prompt_path)
    system_prompt = read_prompt(system_prompt_path)
    base_instruction_prompt = read_prompt(instruction_prompt_path)
    instruction_prompt = base_instruction_prompt.replace(
        '[TRANSCRIPT HERE]', transcript_content)
    full_prompt = f'< | User | > {ctsr_prompt} {instruction_prompt} < | Assistant | >'

    # Llama.cpp command
    command = [
        './llama-cli',
        '--simple-io',
        '-no-cnv',
        '--no-display-prompt',
        '--model', model,
        '--temp', temp,
        '--n-gpu-layers', n_gpu_layers,
        '--batch-size', batch_size,
        '--ctx-size', ctx_size,
        '--seed', seed,
        '--cache-type-k', cache_type_k,
        '--top-k', top_k,
        '--top-p', top_p,
        '--min-p', min_p,
        '-sys', system_prompt,
        '--prompt', full_prompt,
    ]
    command = list(map(str, command))

    logger.info(f'Running {command[0]} build on {transcript_path}')
    start = time.time()

    # Run llama.cpp
    out_bytes = subprocess.check_output(
        command, cwd='./llama.cpp-gpu/build/bin')
    output = out_bytes.decode('utf-8')

    raw_time = time.time() - start
    minutes, seconds = divmod(raw_time, 60)
    logger.info(output)

    # Make outdir and format filenames
    filename = f'{Path(transcript_path).stem}_{os.path.basename(ctsr_prompt_path)}'
    model_name = os.path.basename(model)
    os.makedirs(outdir, exist_ok=True)

    # Pprint model parameters for output file
    params = '\n'.join([f'{command[i]:<20} = {command[i+1]}' for i in range(
        command.index('--model') + 2, len(command) - 2, 2)])

    # Write outputs to file
    nl = "\n"
    write_to_file(os.path.join(outdir, filename),
                  f'{output} {nl}{nl} Model parameters: {nl} {params} {nl} Model: {model_name} {nl} Time taken: {int(minutes)}m {seconds:.2f}s')
    write_to_file(os.path.join(outdir, 'instruction-prompt.txt'),
                  base_instruction_prompt)
    write_to_file(os.path.join(outdir, 'system-prompt.txt'), system_prompt)

    logger.info(f'{nl} llama.cpp GPU build successfully ran cts-r on {filename} in {int(minutes)}m {seconds:.2f}s, temp: {temp}, model: {model_name}')
    return output