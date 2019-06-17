import logging
import json
from scorer import getScore
from text import pre_process

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
    return Request(mind_id, segments)


def lambda_handler(event, context):
    print("event['body']: ", event['body'])

    request = decode_request(event['body'])
    lambda_function_name = "mind-" + request.mind_id

    transcript_text = request.segments[0].text
    pre_processed_input = pre_process(transcript_text)

    if len(pre_processed_input) != 0:
        mind_input = json.dumps({"text": pre_processed_input})
        mind_input = json.dumps({"body": mind_input})
        transcript_score = getScore(mind_input, lambda_function_name)
    else:
        transcript_score = 0.00001
        logger.warn('processing transcript: {}'.format(transcript_text))
        logger.warn('transcript too small to process. Returning default score')

    # hack to penalize out of domain small transcripts coming as PIMs - word level
    if len(pre_processed_input.split(' ')) < 40:
        transcript_score = 0.1 * transcript_score

    out_response = json.dumps({
        'text': transcript_text,
        'distance': 1 / transcript_score,
        'id': request.segments[0].id,
        'conversationLength': 1000,
        'speaker': request.segments[0].speaker,
    })
    print("out_response", out_response)
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
