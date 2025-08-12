print("Importing modules")
import csv
import os
import re
import time

import dspy

# Settings
MODEL = "32b-ctsr"
CAT = 3


# Get human-rated cts-r scores
with open("ctsr_scores.csv") as f:
    reader = csv.reader(f, delimiter=",")
    ctsr_scores = [row for row in reader if row]


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return content


# Read prompts from file
ctsr_prompt = read_file(f"prompts/cats/cat{CAT}.txt")
base_transcript_prompt = read_file("prompts/ctsr-dspy-individual.txt")


# Create transcript prompt for given file
def get_transcript_prompt(transcript_path):
    transcript = read_file(transcript_path)
    transcript_prompt = base_transcript_prompt.replace("[TRANSCRIPT HERE]", transcript)
    return transcript_prompt


# Try to retrieve AI score from model output via regex
def get_ai_score(ctsr_assessment):

    # Return raw score if given
    if isinstance(ctsr_assessment, int):
        return ctsr_assessment

    # Try to retrieve ai scores via regex
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


# Scoring metrics wrapper for dspy
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


# Scoring metrics


@scoring_metric
def negative_absolute_error(pred_score, example_score):
    if pred_score == -1:
        return -10
    error = abs(example_score - pred_score)
    return -error


@scoring_metric
def scaled_absolute_error(pred_score, example_score):
    if pred_score == -1:
        return 0
    error = abs(example_score - pred_score)
    return 1 - (error * 0.1)


@scoring_metric
def negative_squared_error(pred_score, example_score):
    if pred_score == -1:
        return -100
    error = abs(example_score - pred_score)
    return -(error**2)


# Generate trainset of first 9 transcripts used to establish inter-rater reliability
def generate_inter_rater_trainset(cat):
    transcripts = {}
    trainset = []
    for row in ctsr_scores:
        filename = row[0]

        # Calculate average scores for duplicated transcripts
        if filename in transcripts:
            average_cat = (float(transcripts[filename][cat]) + float(row[cat])) / 2
            transcript_prompt = get_transcript_prompt(
                os.path.join("cobalt-text-txt", filename)
            )
            example = dspy.Example(
                transcript_prompt=transcript_prompt, score=average_cat
            ).with_inputs("transcript_prompt")
            trainset.append(example)
        else:
            transcripts[filename] = row
    print(f"Generated inter-rater trainset with {len(trainset)} examples")
    return trainset


# Generate full training set of examples
def generate_full_trainset(cat):
    trainset = []
    for row in ctsr_scores:
        transcript_prompt = get_transcript_prompt(
            os.path.join("cobalt-text-txt", row[0])
        )
        example = dspy.Example(
            transcript_prompt=transcript_prompt, score=float(row[cat])
        ).with_inputs("transcript_prompt")
        trainset.append(example)
    print(f"Generated full trainset with {len(trainset)} examples")
    return trainset


# Predictor class for prompt training
class CTSRAssessor(dspy.Signature):
    transcript_prompt: str = dspy.InputField()
    ctsr_assessment: str = dspy.OutputField(desc=ctsr_prompt)


# Chain of thought class for prompt training
class CTSRAssessorChain(dspy.Signature):
    transcript_prompt: str = dspy.InputField()
    ctsr_assessment: int = dspy.OutputField()  # desc=ctsr_prompt)


# -------------------------------------------------------------------
# Advanced prompt tuning with dspy


# Prompt evaluator, applies prompt to transcript and outputs ctsr score
class CTSRPromptEvaluator(dspy.Signature):
    base_ctsr_prompt: str = dspy.InputField()
    transcript_prompt: str = dspy.InputField()
    ctsr_assessment: int = dspy.OutputField(desc=base_ctsr_prompt)


# Prompt generator, tunes ctsr prompt to maximise accuracy
class CTSRPromptGenerator(dspy.Signature):
    transcript_prompt: str = dspy.InputField()
    base_ctsr_prompt: str = dspy.InputField()
    score: float = dspy.InputField()
    ctsr_prompt: int = dspy.OutputField(
        desc="Modify base_ctsr_prompt for clarity and maximum effectiveness. A reviewer should be able to use your new prompt to assess the transcript_prompt and produce the correct score. Note that the correct score has been given to you for reference, in order to create a good ctsr_prompt; the score is different each time in practice, and as such you are not to hard-code the score into your new prompt."
    )


# Generate trainset for prompt tuning
def generate_ctsr_prompt_trainset(cat):
    trainset = []
    base_ctsr_prompt = ctsr_prompt
    for row in ctsr_scores:
        transcript_prompt = get_transcript_prompt(
            os.path.join("cobalt-text-txt", row[0])
        )
        example = dspy.Example(
            transcript_prompt=transcript_prompt,
            base_ctsr_prompt=base_ctsr_prompt,
            score=float(row[cat]),
        ).with_inputs("transcript_prompt", "base_ctsr_prompt", "score")
        trainset.append(example)
    print(f"Generated full trainset with {len(trainset)} examples")
    return trainset


# Prompt tuning metric, passes prompt to evaluator class and compares with ground truth score
def ctsr_prompt_metric(example, pred, predictor_obj=None, trace=None):
    chain_of_thought_model = dspy.ChainOfThought(CTSRPromptEvaluator)
    score_pred = chain_of_thought_model.predict(
        transcript_prompt=example.transcript_prompt, base_ctsr_prompt=pred.ctsr_prompt
    )
    pred_score = get_ai_score(score_pred.ctsr_assessment)

    if pred_score == -1:
        score = -10
    else:
        error = abs(example.score - pred_score)
        score = -error

    print(score_pred)
    print(f"Example score: {example.score}")
    print(f"Predicted score: {pred_score}")
    print(f"Score: {score}")
    return score


# -------------------------------------------------------------------

# Setup and test llm
print("Loading LLM")
lm = dspy.LM(f"ollama/{MODEL}", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


# MIPRO prompt tuning
start = time.time()
model = dspy.ChainOfThought(CTSRPromptGenerator)
trainset = generate_ctsr_prompt_trainset(CAT)
mipro = dspy.MIPROv2(
    metric=ctsr_prompt_metric,
    auto=None,
    verbose=True,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    num_candidates=10,
)
optimized_model = mipro.compile(
    model, trainset=trainset, num_trials=15, requires_permission_to_run=False
)
optimized_model.save("mipro_optimised_prompt.json")


hrs, rem = divmod(time.time() - start, 3600)
mins, secs = divmod(rem, 60)
print(f"Completed in {hrs}h {mins}m {secs:.2f}s")
