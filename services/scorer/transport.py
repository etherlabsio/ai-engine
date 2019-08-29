import json

from dataclasses import dataclass, asdict
from typing import List
from scorer import TextSegment, Score


@dataclass
class Request:
    mind_id: str
    segments: List[TextSegment] = []


@dataclass
class Response:
    scores: List[Score]


def decode_json_request(body) -> Request:
    req = body
    if isinstance(body, str):
        req = json.loads(body)

    def decode_segments(seg):
        seg_id = seg['id']
        text = seg["originalText"]
        speaker = seg["spokenBy"]
        return TextSegment(seg_id, text, speaker)

    mind_id = str(req['mindId']).lower()
    segments = map(req['segments'], decode_segments)
    return Request(mind_id, list(segments))


class AWSLambdaTransport:
    @staticmethod
    def encode_response(body: Response):
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(asdict(body))
        }
