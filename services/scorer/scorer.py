from dataclasses import dataclass


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
