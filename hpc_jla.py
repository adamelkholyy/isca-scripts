
import subprocess
import time
import csv
import re
import os 




# data = [idx, id, question_num, question]
with open('/lustre/projects/Research_Project-T116269/jla/jla.csv') as f:
    reader = csv.reader(f, delimiter=',', dialect="excel")
    data = [row for row in reader if row][1:]
    print(data)


output_path = "/lustre/projects/Research_Project-T116269/jla/output.txt"
prompt_path = "/lustre/projects/Research_Project-T116269/jla/prompt.txt"
model_path = "/lustre/projects/Research_Project-T116269/llama.cpp-gpu/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"

# get prompts
system_prompt = "You are a perinatal mental health researcher. You have a list of research suggestions and are categorising them by topic."
def generate_prompt(user_input):
    prompt = f"""
    Categorise the following perinatal mental health research suggestion: 
    
    {user_input}

    using the following categories: 

    Assessment
    Baby loss/removal
    Birth support
    Causes/mechanism/risk
    Context
    Evolution
    Family (includes dad)
    Health and pregnancy outcomes
    Infant feeding
    Infertility
    Maternity
    Neurodivergence (autism, adhd etc.) 
    Onset 
    Parenting
    Peer support
    Prevalence
    Prevention
    Severity
    Social support
    Stigma/attitudes
    Substance use
    Suicide
    Trauma
    Treatment
    Work

    Any research suggestions with no relevance/mention of mental health are to be excluded. In this case mark their category as 'Excluded'.
    You may assign multiple categories. Your categories must be ranked in order of importance. You must only assign a category if it is strictly relevant to the research suggestion. 
    """
    # removed "Mental health" category.
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)


os.chdir("/lustre/projects/Research_Project-T116269/llama.cpp-gpu/build/bin/")
print(os.listdir("."))


def run_llama(): # DeepSeek-R1-Distill-Qwen-7B-Q4_K_M
    command = f"./llama-cli --model {model_path} --n-gpu-layers 29 -st --seed 42 --simple-io --predict -2 --temp 0 --file {prompt_path} > {output_path}"

    start = time.time()
    subprocess.run(command, shell=True)
    raw_time  = time.time() - start

    minutes, seconds = divmod(raw_time, 60)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    print(content)
    print(f"llama.cpp completed in {minutes}m {seconds:.2f}s")
    return content


def filter_categories(model_output):
    final_answer = model_output.split("</think>")[1]
    categories = re.findall(r"\*\*(.*)\*\*", final_answer)        
    return categories if categories else [final_answer]
    # return ", ".join(categories) if categories else final_answer


category_data = []

limit = 10
for i, row in enumerate(data):
    idx, id, question_num, suggestion = row
    generate_prompt(suggestion)
    model_output = run_llama()
    categories = filter_categories(model_output)
    category_data.append([id, suggestion] + categories)

    print(f"{i+1}/{limit} | {(i+1)/limit *100:.2f}%")
    if i == limit:
        break
    


print(category_data)
file_str = "\n".join([",".join([col for col in row]) for row in category_data])
with open("/lustre/projects/Research_Project-T116269/jla/categories.csv", "w", encoding="utf-8") as f:
    f.write(file_str)