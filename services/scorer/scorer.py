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


class Scorer(object):
    def score(self, mind_id: str, request: TextSegment) -> Score:
        raise NotImplementedError("scorer protocol not implemented")


class SentenceScorer(Scorer):
    def __init__(self, client):
        self.mind_client = client

    def score(self, mind_id: str, request: TextSegment) -> Score:
        text = pre_process(request.text)
        sent_score = self.calculate_sentence_score(mind_id, text)
        sent_score = self.penalize(text, sent_score)
        distance = 1 / sent_score
        return Score(id=request.id,
                     text=text,
                     speaker=request.speaker,
                     score=distance)

    def penalize(self, text: str, score: float, min_word_count: int = 40):
        # Penalize sentences with smaller word count
        with text.split(" ") as words:
            if len(words) < min_word_count:
                return 0.1 * score
        return score

    def calculate_sentence_score(self,
                                 mind_id: str,
                                 text: str,
                                 default_score: float = 0.00001) -> float:
        if not text:
            return default_score

        try:
            resp: MindResponse = self.mind_client.calculate(mind_id, text)
        except Exception as err:
            logger.error("error from mind service for input: {} as {}".format(
                mind_id, err))

        if len(resp.feature_vector) == 0:
            logger.warn(
                'transcript too small to process. Returning default_score score'
            )
            return default_score

        transcript_score_list = []
        for sent_vec, sent_nsp_list in zip(resp.feature_vector, resp.nsp_list):
            sent_score_list = []
            for mind_vec, mind_nsp in zip(resp.mind_vector, sent_nsp_list):
                sent_score = cluster_score(sent_vec, mind_vec, mind_nsp)
                sent_score_list.append(sent_score)
            transcript_score_list.append(np.max(sent_score_list))
        return np.mean(transcript_score_list)
