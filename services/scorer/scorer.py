import logging
import numpy as np
from numpy import dot
from numpy.linalg import norm
from dataclasses import dataclass
from text import pre_process
from mind.response import MindResponse

logger = logging.getLogger()


def cluster_score(sent_vec, mind_vec, mind_nsp, nsp_dampening_factor=0.7):
    cosine_sim = cosine(sent_vec, mind_vec)
    nsp_score = mind_nsp * nsp_dampening_factor
    score = np.mean([cosine_sim, nsp_score])
    return score


def cosine(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


@dataclass
class TextSegment:
    id: str
    text: str
    speaker: str


@dataclass
class Score(TextSegment):
    score: float

    def distance(self) -> float:
        return 1 / self.score


class SentenceScorer:
    def __init__(self, client):
        self.mind_client = client

    def calculate(self, mind_id: str, request: TextSegment) -> Score:
        text = pre_process(request.text)
        score = Score(id=request.id,
                      text=text,
                      speaker=request.speaker,
                      score=0.00001)

        ## TODO: penalize score

        try:
            resp: MindResponse = self.mind_client.calculate(mind_id, text)
        except Exception as err:
            logger.error("error from mind service for input: {} as {}".format(
                mind_id, err))
            return score

        if not resp.feature_vector:
            logger.warn(
                'transcript too small to process. Returning default score')
            return score

        transcript_score_list = []
        for sent_vec, sent_nsp_list in zip(resp.feature_vector, resp.nsp_list):
            sent_score_list = []
            for mind_vec, mind_nsp in zip(resp.mind_vector, sent_nsp_list):
                sent_score = cluster_score(sent_vec, mind_vec, mind_nsp)
                sent_score_list.append(sent_score)
            transcript_score_list.append(np.max(sent_score_list))
        score.score = np.mean(transcript_score_list)
        return score
