from scorer import SentenceScorer, Score
from mind import AWSLambdaClient
from dataclasses import dataclass, asdict
from typing import List
from scorer import TextSegment
from transport import decode_json_request, Response, Request, encode_aws_lambda_response

SCORER = SentenceScorer(client=AWSLambdaClient())


def lambda_handler(event, context):
    request = decode_json_request(event['body'])
    mind_id = request.mind_id
    scores = map(lambda s: SCORER.score(mind_id, s), request.segments)
    resp = Response(scores)
    return encode_aws_lambda_response(resp)
