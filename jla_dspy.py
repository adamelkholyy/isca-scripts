import csv
import json
import time
from typing import List, Literal

import dspy

# TODO: Change Sigma -> Stigma

# Settings
MODEL = "deepseek-r1:70b"

with open("jla.csv") as f:
    reader = csv.reader(f, delimiter=",", dialect="excel")
    data = [row for row in reader if row][1:]


data = data[:100]

# Generate dataset
dataset = []
for row in data:
    example = dspy.Example(suggestion=row[3]).with_inputs("suggestion")
    dataset.append(example)
print(f"Generated full dataset with {len(dataset)} examples", flush=True)

# Allowed JLA categories
AllowedCategories = Literal[
    "Attitudes/beliefs",
    "Assessment (describing, labelling or diagnosing a problem)",
    "Baby loss (termination, death, abortion, miscarriage, still birth)",
    "Baby removal (loss of custody, safeguarding, social services)",
    "Birth support",
    "Causes/mechanism/risk",
    "Context",
    "Evolution",
    "Family (includes dad)",
    "Health and preganancy outcomes (premature, hyperemesis)",
    "Infant feeding",
    "Infertility",
    "Maternity (antenatal, postnatal, obstetric, labour, childbirth)",
    "Neonatal care",
    "Neurodivergence (autism, adhd etc.)",
    "Onset",
    "Parenting",
    "Peer support",
    "Prevalence (how many of x...)",
    "Prevention",
    "Severity",
    "Social support",
    "Sigma (disgrace, shame, humiliation)",
    "Substance use",
    "Suicide",
    "Trauma",
    "Treatment (PNMHT, MMHT, perinatal mental health service, maternal mental health service)",
    "Work",
    "EXCLUDED",
]

# Description of the JLA categorisation task
description = """
Categorise the perinatal mental health research suggestion using up to 6 of the available categories. 
Any research suggestions with no relevance/mention of mental health are to be excluded. In this case mark their category as 'EXCLUDED'. 
You may assign up to 6 categories. Your categories must be ranked in order of importance. 
You must only assign a category if it is strictly relevant to the research suggestion.
"""

# Chain of thought class JLA categorisation
class JLACategoriser(dspy.Signature):
    suggestion: str = dspy.InputField()
    
    # Limit model outputs to available categories only
    categories: List[AllowedCategories] = dspy.OutputField(desc=description)


# Setup llm
print("Loading LLM", flush=True)
lm = dspy.LM(f"ollama/{MODEL}", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
chain_of_thought_model = dspy.ChainOfThought(JLACategoriser)


start = time.time()
category_data = {}

# Run JLA categorisation
for i, example in enumerate(dataset):
    print(f"Categorising suggestion {i+1}/{len(dataset)}:", flush=True)
    print(example.suggestion, flush=True)
    pred = chain_of_thought_model.predict(suggestion=example.suggestion)
    print(pred.categories, flush=True)
    category_data[i] = {
        "Research suggestion": example.suggestion,
        "Categories": pred.categories,
    }


# Write results to file
json_data = json.dumps(category_data, indent=3, default=list)
with open("categories.json", "w", encoding="utf-8") as f:
    f.write(json_data)

hrs, rem = divmod(time.time() - start, 3600)
mins, secs = divmod(rem, 60)
print(f"Completed in {hrs}h {mins}m {secs:.2f}s", flush=True)
