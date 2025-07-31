import os
import csv
import re

os.chdir("/lustre/projects/Research_Project-T116269")
# os.chdir("C://Users//ae553//Downloads")


# human-rated ctsr scores
with open('ctsr_scores.csv') as f:
    reader = csv.reader(f, delimiter=',')
    ctsr_scores = {}
    inter_rater_scores = {}
    for row in reader:
        if not row:
            continue
        filename = row[0]
        scores = list(map(float, row[1:]))
        if filename in ctsr_scores:
            averages = [(ctsr_scores[filename][i] + scores[i])/2 for i in range(len(scores))] 
            ctsr_scores[filename] = averages
            inter_rater_scores[filename] = averages
        else:
            ctsr_scores[filename] = scores



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




def get_scores(dirname, cat):
    base = os.path.basename(dirname)
    with open(os.path.join("assessments", dirname, f"{base}_cat{cat}.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    ai_score = get_ai_score(content)
    human_score = ctsr_scores[f"{base}.txt"][cat]

    return ai_score, human_score


def calculate_error(ai_score, human_score):
    error = ai_score - human_score
    print(f"Human score: {human_score}")        
    print(f"AI score   : {ai_score}")
    print(f"Error      : {error}")
    return abs(error)


if __name__ == "__main__":
    name = "B110023_104_s05"
    ai, human = get_scores(name, cat=1)
    calculate_error(ai, human)
