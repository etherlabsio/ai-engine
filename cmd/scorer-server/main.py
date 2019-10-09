try:
    import unzip_requirements
except ImportError:
    pass

import os
import logging
from log.logger import setup_server_logger
import json
from mind_utils import loadMindFeatures
from transport import decode_json_request, Response, Request, AWSLambdaTransport
logger = logging.getLogger()


def lambda_handler(event, context):
    Request = decode_json_request(event)
    mind_dict = loadMindFeatures(Request.mind_id)
    scores = list(map(lambda s: Scorer.score(Request.mind_id, mind_dict, s), Request.segments))
    out_response = Request.segments
    assert(len(Request.segments) == len(scores))
    for index in range(len(scores)):
        out_response[index]['distance'] = scores[index]
    res = {'d2vResult': out_response}
    return AWSLambdaTransport.encode_response(res)
