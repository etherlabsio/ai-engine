import numpy as np
from numpy import dot
from numpy.linalg import norm
from dataclasses import dataclass
from text import pre_process
from mind import MindResponse


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
