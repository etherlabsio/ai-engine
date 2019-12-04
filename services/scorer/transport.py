import json
from dataclasses import dataclass, asdict, field
from typing import List
import logging
from copy import deepcopy
from scorer.scorer import TextSegment, Score


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
        seg_id = seg["id"]
        text = seg["originalText"]
        speaker = seg["spokenBy"]
        return TextSegment(seg_id, text, speaker)

    mind_id = str(req["mindId"]).lower()
    context_id = req["contextId"]
    instance_id = req["instanceId"]
    segments = list(map(lambda x: decode_segments(x), req["segments"]))
    return Request(mind_id, context_id, instance_id, list(segments))


class AWSLambdaTransport:
    @staticmethod
    def encode_response(body: Response):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body),
        }
