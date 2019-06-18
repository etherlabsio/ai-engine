import numpy as np
from dataclasses import dataclass


@dataclass
class MindResponse:
    feature_vector: np.array
    mind_vector: np.array
    nsp_scores: np.array
