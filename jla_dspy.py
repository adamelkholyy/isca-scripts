import time                         
import os                           
import dspy                         
import csv                         
import json                       
import logging 
import sys
from typing import List, Literal    

# TODO: Change Sigma -> Stigma

# settings
os.chdir("/lustre/projects/Research_Project-T116269/")
MODEL = 'deepseek-r1:70b' # "32b-ctsr"

with open('/lustre/projects/Research_Project-T116269/jla.csv') as f:
    reader = csv.reader(f, delimiter=',', dialect="excel")
    data = [row for row in reader if row][1:]


data = data[:100]

trainset = []
for row in data:
    example = dspy.Example(suggestion=row[3]).with_inputs("suggestion")
    trainset.append(example)
print(f"Generated full trainset with {len(trainset)} examples", flush=True)

AllowedCategories = Literal[
    'Attitudes/beliefs',
    'Assessment (describing, labelling or diagnosing a problem)',      
    'Baby loss (termination, death, abortion, miscarriage, still birth)',					
    'Baby removal (loss of custody, safeguarding, social services)',	    
    'Birth support',
    'Causes/mechanism/risk',
    'Context',
    'Evolution',
    'Family (includes dad)',
    'Health and preganancy outcomes (premature, hyperemesis)',					
    'Infant feeding',
    'Infertility',
    'Maternity (antenatal, postnatal, obstetric, labour, childbirth)',
    'Neonatal care', 										
    'Neurodivergence (autism, adhd etc.)', 
    'Onset', 
    'Parenting',
    'Peer support',
    'Prevalence (how many of x...)',
    'Prevention',
    'Severity',
    'Social support',
    'Sigma (disgrace, shame, humiliation)',    
    'Substance use',
    'Suicide',
    'Trauma',
    'Treatment (PNMHT, MMHT, perinatal mental health service, maternal mental health service)',					
    'Work',
    'EXCLUDED',
]
   
description = """
Categorise the perinatal mental health research suggestion using up to 6 of the available categories. 
Any research suggestions with no relevance/mention of mental health are to be excluded. In this case mark their category as 'EXCLUDED'. 
You may assign up to 6 categories. Your categories must be ranked in order of importance. 
You must only assign a category if it is strictly relevant to the research suggestion.
"""

# chain of thought class for prompt training
class JLACategoriser(dspy.Signature):
    suggestion: str = dspy.InputField()
    categories: List[AllowedCategories] = dspy.OutputField(desc=description)


# setup and test llm
print("Loading LLM", flush=True)
lm = dspy.LM(f'ollama/{MODEL}', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


# run jla categorisation
start = time.time()
category_data = {}
chain_of_thought_model = dspy.ChainOfThought(JLACategoriser)
for i, example in enumerate(trainset):
    print(f"Categorising suggestion {i+1}/{len(trainset)}:", flush=True)
    print(example.suggestion, flush=True)
    pred = chain_of_thought_model.predict(suggestion=example.suggestion)
    print(pred.categories, flush=True)
    category_data[i] = {"Research suggestion": example.suggestion, "Categories": pred.categories}


# save results
json_data = json.dumps(category_data, indent=3, default=list)
with open("/lustre/projects/Research_Project-T116269/categories.json", "w", encoding="utf-8") as f:
    f.write(json_data)

hrs, rem = divmod(time.time() - start, 3600)
mins, secs = divmod(rem, 60)
print(f"Completed in {hrs}h {mins}m {secs:.2f}s", flush=True)

