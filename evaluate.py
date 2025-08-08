import os
import csv
import re
import logging

logger = logging.getLogger(__name__)


# Get human-rated ctsr scores from csv
with open('ctsr_scores.csv') as f:
    reader = csv.reader(f, delimiter=',')
    ctsr_scores = {}
    for row in reader:
        # Skip empty rows
        if not row:
            continue

        filename = row[0]
        scores = list(map(float, row[1:]))

        # For duplicated scores take the average
        if filename in ctsr_scores:
            averages = [(ctsr_scores[filename][i] + scores[i]) /
                        2 for i in range(len(scores))]
            ctsr_scores[filename] = averages
        else:
            ctsr_scores[filename] = scores


# Try to retrieve ai scores via regex
def get_ai_score(ctsr_assessment):
    try:
        ai_score = re.search(r'boxed\{(\d)\}', ctsr_assessment)
        if not ai_score:
            ai_score = re.search(r'(\d)/6', ctsr_assessment)
        if not ai_score:
            ai_score = re.search(r'(\d) out of 6', ctsr_assessment)
        if not ai_score:
            logger.error('Could not find score via regex')
            return -1
        ai_score = float(ai_score.group(1))
        return ai_score

    except AttributeError as e:
        logger.error(f'Regex AttributeError due to missing group: {e}')
        return -1


# Get human and ai scores
def get_scores(dirname, cat):
    base = os.path.basename(dirname)
    with open(os.path.join(dirname, f'{base}_cat{cat}.txt'), 'r', encoding='utf-8') as f:
        content = f.read()
    ai_score = get_ai_score(content)
    human_score = ctsr_scores[f'{base}.txt'][cat]
    return ai_score, human_score


# Calculate and log human vs ai eror
def calculate_error(ai_score, human_score):
    error = ai_score - human_score
    logger.info(f'Human: {human_score}')
    logger.info(f'AI   : {ai_score}')
    logger.info(f'Error: {error}')
    return error
