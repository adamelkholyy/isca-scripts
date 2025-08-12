import csv
import os
import re
import subprocess
import time

# Naive prompting for JLA categorisation.
# Replaced with jla_dspy which uses the DSpy module to limit the model's category choices

# Load data = [idx, question_id, question_num, question]
with open("/lustre/projects/Research_Project-T116269/jla.csv") as f:
    reader = csv.reader(f, delimiter=",", dialect="excel")
    data = [row for row in reader if row][1:]
    print(data)

output_path = "/lustre/projects/Research_Project-T116269/jla/output.txt"
prompt_path = "/lustre/projects/Research_Project-T116269/jla/prompt.txt"
model_path = "/lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"

# Generate JLA categorisation prompts
system_prompt = "You are a perinatal mental health researcher. You have a list of research suggestions and are categorising them by topic."


def generate_prompt(user_input):
    prompt = f"""
    Categorise the following perinatal mental health research suggestion: 
    
    {user_input}

    using up to 6 of the following categories: 

    Attitudes/beliefs
    Assessment (describing, labelling or diagnosing a problem)      
    Baby loss (termination, death, abortion, miscarriage, still birth)					
    Baby removal (loss of custody, safeguarding, social services)	    
    Birth support
    Causes/mechanism/risk
    Context
    Evolution
    Family (includes dad)
    Health and preganancy outcomes (premature, hyperemesis)					
    Infant feeding
    Infertility
    Maternity (antenatal, postnatal, obstetric, labour, childbirth)
    Neonatal care 										
    Neurodivergence (autism, adhd etc.) 
    Onset 
    Parenting
    Peer support
    Prevalence (how many of x...)
    Prevention
    Severity
    Social support
    Sigma (disgrace, shame, humiliation)    
    Substance use
    Suicide
    Trauma
    Treatment (PNMHT, MMHT, perinatal mental health service, maternal mental health service )					
    Work

    Any research suggestions with no relevance/mention of mental health are to be excluded. In this case mark their category as 'Excluded'.
    You may assign up to 6 categories. Your categories must be ranked in order of importance. You must only assign a category if it is strictly relevant to the research suggestion. 
    """
    # Write prompt to file for llama.cpp
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)


# TODO: remove hot dir switching
os.chdir("/lustre/projects/Research_Project-T116269/llama.cpp-gpu/build/bin/")

# Call llama.cpp


def run_llama():
    command = f"./llama-cli --model {model_path} --n-gpu-layers 29 -st --seed 42 --simple-io --predict -2 --temp 0 --file {prompt_path} > {output_path}"

    start = time.time()
    subprocess.run(command, shell=True)
    raw_time = time.time() - start
    minutes, seconds = divmod(raw_time, 60)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(content)
    print(f"llama.cpp completed in {minutes}m {seconds:.2f}s")
    return content


# Filter model categories via regex for csv output
# TODO: Wrap this in try, except (because regexes...)
def filter_categories(model_output):
    final_answer = model_output.split("</think>")[1]
    categories = re.findall(r"\*\*(.*)\*\*", final_answer)
    return categories if categories else [final_answer]


# Call lama.cpp `n` times
n = 10
category_data = []
for i, row in enumerate(data[:n]):
    idx, id, question_num, suggestion = row
    generate_prompt(suggestion)
    model_output = run_llama()
    categories = filter_categories(model_output)
    print(categories)
    category_data.append([id, suggestion] + categories)
    print(f"{i+1}/{n} | {(i+1)/n * 100:.2f}%")


# Write outputs to file
print(category_data)
file_str = "\n".join([",".join([col for col in row]) for row in category_data])
with open(
    "/lustre/projects/Research_Project-T116269/jla/categories.csv",
    "w",
    encoding="utf-8",
) as f:
    f.write(file_str)
