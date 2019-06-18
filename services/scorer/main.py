import logging
import json
from scorer import SentenceScorer, Score
from text import pre_process
from mind.lambda_client import AWSLambdaClient
from dataclasses import dataclass
from typing import List
from scorer import TextSegment

logger = logging.getLogger()


@dataclass
class Request:
    mind_id: str
    segments: List[TextSegment] = []


def decode_request(body) -> Request:
    if isinstance(body, str):
        req = json.loads(body)
    else:
        req = body

    def decode_segments(seg):
        seg_id = seg['id']
        text = seg["originalText"]
        speaker = seg["spokenBy"]
        return TextSegment(seg_id, text, speaker)

    mind_id = str(req['mindId']).lower()
    segments = map(req['segments'], decode_segments)
    return Request(mind_id, list(segments))


def lambda_handler(event, context):
    print("event['body']: ", event['body'])

    client = AWSLambdaClient()
    scorer = SentenceScorer(client)
    request = decode_request(event['body'])
    mind_id = request.mind_id

    scores = map(lambda segment: scorer.score(mind_id, segment),
                 request.segments)

    return {
        "statusCode":
        200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body":
        json.dumps({
            'd2vResult': [{
                'text': transcript_text,
                'distance': 1 / transcript_score,
                'id': request.segments[0].id,
                'conversationLength': 1000,
                'speaker': request.segments[0].speaker,
            }]
        })
    }
