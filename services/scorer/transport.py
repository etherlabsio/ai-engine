import json
from dataclasses import dataclass, asdict, field
from typing import List
import logging
from copy import deepcopy
from scorer.scorer import TextSegment, Score
from group_segments.extra_preprocess import preprocess_text as pt

@dataclass
class Request:
    mind_id: str
    context_id: str
    instance_id: str
    segments: List[TextSegment] = field(default_factory=list)


@dataclass
class Response:
    scores: List[Score] = field(default_factory=list)


def decode_json_request(body) -> Request:
    req = body

    def decode_segments(seg):
        pre_processed_text = pt(seg["originalText"], scorer=True)
        if pre_processed_text == "":
            return False
        seg_id = seg["id"]
        text = seg["originalText"]
        speaker = seg["spokenBy"]
        return TextSegment(seg_id, text, speaker)

    mind_id = str(req["mindId"]).lower()
    context_id = req["contextId"]
    instance_id = req["instanceId"]
    segments = [ res for res in list(map(lambda x: decode_segments(x), req["segments"])) if res!=False]
    if segments == []:
        return False
    return Request(mind_id, context_id, instance_id, list(segments))


class AWSLambdaTransport:
    @staticmethod
    def encode_response(body: Response):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body),
        }
