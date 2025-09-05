import csv
import logging
import os
import re
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import scripts.krippendorff as krippendorff

logger = logging.getLogger(__name__)


# Get human-rated ctsr scores from csv
def read_ctsr_scores():
    with open("ctsr_scores.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        ctsr_scores = {}

        for row in reader:
            filename = row[0]
            scores = list(map(float, row[1:]))

            # For double-rated transcripts take the average score
            if filename in ctsr_scores:
                averages = [
                    (ctsr_scores[filename][i] + scores[i]) / 2
                    for i in range(len(scores))
                ]
                ctsr_scores[filename] = averages
            else:
                ctsr_scores[filename] = scores

    return ctsr_scores


# Try to retrieve ai scores via regex
def get_ai_score(ctsr_assessment):
    try:
        ai_score = re.search(r"boxed\{(\d)\}", ctsr_assessment)

        if not ai_score:
            ai_score = re.search(r"(\d)/6", ctsr_assessment)
        if not ai_score:
            ai_score = re.search(r"(\d) out of 6", ctsr_assessment)
        if not ai_score:
            logger.error("Could not find score via regex")
            return -1

        ai_score = float(ai_score.group(1))
        return ai_score

    except AttributeError as e:
        logger.error(
            f"Could not find score - regex AttributeError due to missing group: {e}"
        )
        return -1


# Get human and ai scores
def get_scores(dirname, cat):
    ctsr_scores = read_ctsr_scores()
    transcript_name = os.path.basename(dirname)

    with open(
        os.path.join(dirname, f"{transcript_name}_cat{cat}.txt"), "r", encoding="utf-8"
    ) as f:
        content = f.read()

    ai_score = get_ai_score(content)
    human_score = ctsr_scores[f"{transcript_name}.txt"][cat]

    return ai_score, human_score


# Calculate and log human vs ai eror
def calculate_error(ai_score, human_score):
    error = ai_score - human_score
    logger.info(f"Human: {human_score}")
    logger.info(f"AI   : {ai_score}")
    logger.info(f"Error: {error}")
    return error


# K's alpha typing
DEFAULT_DTYPE = np.float64
ValueScalarType = TypeVar("ValueScalarType", bound=np.generic)
MetricResultScalarType = TypeVar("MetricResultScalarType", bound=np.inexact)

# Metric must have specific signature else krippendorff's module won't accept it, hence the unused args
def custom_interval_metric(
    v1: npt.NDArray[ValueScalarType],
    v2: npt.NDArray[ValueScalarType],
    i1: npt.NDArray[np.int_],
    i2: npt.NDArray[np.int_],
    n_v: npt.NDArray[np.number],  # noqa
    dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE,  # type: ignore
) -> npt.NDArray[MetricResultScalarType]:  
    """
    Custom K's alpha metric for our data since the AI cannot score to a 0.5 resolution, yet humans can
    """

    interval_arr = (v1 - v2).astype(dtype) ** 2  # type: ignore
    interval_arr[interval_arr <= 0.5**2] = 0.0
    return interval_arr


def calculate_ks_alpha(cat_scores):
    human_scores = [transcript_data["human"] for transcript_data in cat_scores.values()]
    ai_scores = [transcript_data["ai"] for transcript_data in cat_scores.values()]

    data = [human_scores, ai_scores]

    ks_alpha = krippendorff.alpha(
        reliability_data=data, level_of_measurement=custom_interval_metric
    )
    return ks_alpha
