print("Importing modules")
import time
import os
import dspy
import re
import csv

# TODO:
# Tweak scoring metric: -100 for no score?, exponential errors - ((err * 10)^2) = -360, 0
# -500, 0 ->  + 500 * 0.02
# 

"""
    if pred_score == -1:
        score = -5000
    else:
        # error bounds = [0, 6] # 360, 0 
        error = abs(example.score - pred_score) * 10
        error = error * error
        score = error + 5000
        score = score * 0.0002
"""

# settings
os.chdir("/lustre/projects/Research_Project-T116269/")
MODEL = "32b-ctsr"
CAT = 1


# human-rated cts-r scores
with open('ctsr_scores.csv') as f:
    reader = csv.reader(f, delimiter=',')
    ctsr_scores = [row for row in reader if row]


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return content


# read prompts from file
ctsr_prompt = read_file(f"prompts/cats/cat{CAT}.txt")
base_transcript_prompt = read_file("prompts/ctsr-dspy-individual.txt")


# create transcript prompt for given file
def get_transcript_prompt(transcript_path):
    transcript = read_file(transcript_path)
    transcript_prompt = base_transcript_prompt.replace("[TRANSCRIPT HERE]", transcript)
    return transcript_prompt


# retrieve AI score from model output
def get_ai_score(ctsr_assessment):

    if isinstance(ctsr_assessment, int):
        return ctsr_assessment        

    try:
        # try to get ai score with regex
        ai_score = re.search(r"boxed\{(\d)\}", ctsr_assessment)
        if not ai_score:
            ai_score = re.search(r"(\d)/6", ctsr_assessment)
        if not ai_score:
            ai_score = re.search(r"(\d) out of 6", ctsr_assessment)
        if not ai_score:
            print("Could not find score")
            return -1
        ai_score = float(ai_score.group(1))
        return ai_score

    except AttributeError as e:
        print(f"Could not find score: {e}")
        return -1


def scoring_metric(example, pred, predictor_obj=None, trace=None):
    print("\nPrediction")
    print(pred)
    print("\nPredictor obj")
    print(predictor_obj)

    # scoring metric: between [0, 1], 0 for no score, else 1 - abs(error)/10
    pred_score = get_ai_score(pred.ctsr_assessment)

    
    if pred_score == -1:
        score = 0
    else:
        # error bounds = [0, 6] # 360, 0 
        error = abs(example.score - pred_score)
        score = 1 - (error*0.1)
    
    print("\n" + f"Score: {score}")
    return score
    


# generate training set of examples 
def generate_trainset(cat):
    trainset = []
    for row in ctsr_scores:
        transcript_prompt = get_transcript_prompt(os.path.join("cobalt-text-txt", row[0]))
        example = dspy.Example(transcript_prompt=transcript_prompt, score=float(row[cat])).with_inputs("transcript_prompt")
        trainset.append(example)
    print(f"Generated trainset with {len(trainset)} examples")
    return trainset


# predictor class for prompt training
class CTSRAssessor(dspy.Signature):
    transcript_prompt: str = dspy.InputField()
    ctsr_assessment: str = dspy.OutputField(desc=ctsr_prompt)


# chain of thought class for prompt training
class CTSRAssessorChain(dspy.Signature):
    transcript_prompt: str = dspy.InputField()
    ctsr_assessment: int = dspy.OutputField(desc=ctsr_prompt)




# setup and test llm
print("Loading and testing LLM")
lm = dspy.LM(f'ollama/{MODEL}', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# output = lm("Hi!")  # => ['This is a test!'] (, temperature=0.7 optional arg)
# print(output)


# run SIMBA for prompt fine-tuning
start = time.time()
predictor_model = dspy.Predict(CTSRAssessor)
chain_of_thought_model = dspy.ChainOfThought(CTSRAssessorChain)

tp = dspy.SIMBA(metric=scoring_metric)
trainset = generate_trainset(CAT)

optimized_model = tp.compile(chain_of_thought_model, trainset=trainset)
optimized_model.save("optimised_model.json")

hrs, rem = divmod(start - time.time(), 3600)
mins, secs = divmod(rem, 60)
print(f"Completed in {hrs}h {mins}m {secs:.2f}s")
