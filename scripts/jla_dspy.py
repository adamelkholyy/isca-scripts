import csv
import json
import time
import dspy

from typing import List, Literal

# Settings
MODEL = "deepseek-r1:70b"

with open("jla.csv") as f:
    reader = csv.reader(f, delimiter=",", dialect="excel")
    data = [row for row in reader if row][1:]


# Generate dataset
dataset = []
for row in data:
    example = dspy.Example(suggestion=row[3]).with_inputs("suggestion")
    dataset.append(example)
print(f"Generated dataset with {len(dataset)} examples", flush=True)


AllowedCategories = Literal[
    "Anxiety",
    "Attitudes/beliefs",
    "Assessment (describing, screening, labelling or diagnosing a problem)",
    "Baby loss (termination, death, abortion, miscarriage, ectopic pregnancy, still birth)",
    "Baby removal (loss of custody, safeguarding, social services)",
    "Birth support",
    "Causes/mechanism (vulnerability, aetiology)",
    "Context (gender, cultural, debt, poverty, migrant/migration, isolation/loneliness, inequalities)",
    "Depression",
    "Ethnic minority (ethnicity, minoritized ethnic group, race, racism, BAME)",
    "Family (includes dad, parent, sibling)",
    "Health and pregnancy outcomes (hyperemesis/vomiting/HG, hypertension, perineal trauma, incontinence)",
    "Help-seeking",
    "Hormones (menstrual disorders etc.)",
    "Infant feeding (breastfeeding, weaning, bottlefeeding)",
    "Infertility (IVF)",
    "Interface with other systems (law, military/armed forces)",
    "Maternity (pre-conception, antenatal, postnatal/postpartum, labour, childbirth)",
    "Maternity care (midwife/midwifery, obstric/obstetrician, health visitor)",
    "Medication (side effects, teratogenic, lithium, change)",
    "Mother and baby unit (MBU, in-patient care)",
    "Neonatal care (premature birth, small for gestational age, low birthweight, high risk baby)",
    "Neurodivergence (neurodivergent, autism, adhd etc.)",
    "Online (social media, online therapy etc.)",
    "Parenting (parent child connection, bonding, attachment)",
    "Infant (baby, unborn, foetus, child, child development)",
    "Voluntary sector (peer support, charity, informal)",
    "Poverty (debt, poverty, deprivation)?",
    "Prevalence (how many of x...)",
    "Prevention",
    "Psychological therapies (CBT, art therapy, group therapy, psychotherapy",
    "Psychosis",
    "Risk factor",
    "Severity",
    "Social support",
    "Stigma (disgrace, shame, humiliation)",
    "Substance use (alcohol, smoking, vaping, drugs)",
    "Suicide",
    "Trauma",
    "Treatment (PNMHT, MMHT, perinatal mental health service, maternal mental health service, intervention)",
    "Violence (sexual, intimate partner violence, IPV, domestic abuse)",
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
