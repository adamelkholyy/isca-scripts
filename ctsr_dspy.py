print("Importing modules")
import time
import os
import dspy
import re
import csv



# settings
os.chdir("/lustre/projects/Research_Project-T116269/")
MODEL = "32b-ctsr"
CAT = 9


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

    # try to retrieve ai scores via regex
    try:
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


# scoring metrics for dspy
def scoring_metric(metric):
    def wrapped_metric(example, pred, predictor_obj=None, trace=None):
        pred_score = get_ai_score(pred.ctsr_assessment)
        score = metric(pred_score, example.score)

        print(pred)
        print(f"Example score: {example.score}")
        print(f"Predicted score: {pred_score}")
        print(f"Score: {score}")
        return score
    return wrapped_metric


@scoring_metric
def negative_absolute_error(pred_score, example_score):
    if pred_score == -1:
        score = -10
    else:
        error = abs(example_score - pred_score)
        score = -error
    return score

@scoring_metric
def scaled_absolute_error(pred_score, example_score):
    if pred_score == -1:
        score = 0
    else:
        error = abs(example_score - pred_score)
        score = 1 - (error*0.1)
    return score

@scoring_metric
def negative_squared_error(pred_score, example_score):
    if pred_score == -1:
        score = -100
    else:
        error = abs(example_score - pred_score)
        score = -(error**2)
    return score



"""
def scoring_metric(example, pred, predictor_obj=None, trace=None):

    pred_score = get_ai_score(pred.ctsr_assessment)
    score = negative_absolute_error(pred_score, example.score)

    print(pred)
    print(f"Example score: {example.score}")
    print(f"Predicted score: {pred_score}")
    print(f"Score: {score}")
    return score
"""
    

    


def generate_inter_rater_trainset(cat):
    transcripts = {}
    trainset = []
    for row in ctsr_scores:
        filename = row[0]
        if filename in transcripts:
            average_cat = (float(transcripts[filename][cat]) + float(row[cat]))/2
            transcript_prompt = get_transcript_prompt(os.path.join("cobalt-text-txt", filename))
            example = dspy.Example(transcript_prompt=transcript_prompt, score=average_cat).with_inputs("transcript_prompt")
            trainset.append(example)
        else:
            transcripts[filename] = row
    print(f"Generated inter-rater trainset with {len(trainset)} examples")
    return trainset


# generate full training set of examples 
def generate_full_trainset(cat):
    trainset = []
    for row in ctsr_scores:
        transcript_prompt = get_transcript_prompt(os.path.join("cobalt-text-txt", row[0]))
        example = dspy.Example(transcript_prompt=transcript_prompt, score=float(row[cat])).with_inputs("transcript_prompt")
        trainset.append(example)
    print(f"Generated full trainset with {len(trainset)} examples")
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
print("Loading LLM")
lm = dspy.LM(f'ollama/{MODEL}', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


# run prompt fine-tuning
start = time.time()

predictor_model = dspy.Predict(CTSRAssessor)
chain_of_thought_model = dspy.ChainOfThought(CTSRAssessorChain)
metric = negative_absolute_error

simba = dspy.SIMBA(metric=metric)
mipro = dspy.MIPROv2(metric=metric, auto="light")
copro = dspy.COPRO(metric=metric)
bootstrap_fs_random_search = dspy.BootstrapFewShotWithRandomSearch(metric=metric)
bootstrap_finetune = dspy.BootstrapFinetune(metric=metric)


model = chain_of_thought_model
tp = mipro 
trainset = generate_inter_rater_trainset(CAT)


optimized_model = tp.compile(model, trainset=trainset, requires_permission_to_run=False) # <- turn on for MIPRO!
optimized_model.save("mipro_heavy.json")

hrs, rem = divmod(time.time() - start, 3600)
mins, secs = divmod(rem, 60)
print(f"Completed in {hrs}h {mins}m {secs:.2f}s")
