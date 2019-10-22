# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config
from dataclasses import dataclass
from scorer.pre_process import preprocess_text

logger = logging.getLogger()

config = Config(connect_timeout=60, read_timeout=240, retries={'max_attempts': 0}, )
lambda_client = boto3_client('lambda', config=config)


@dataclass
class TextSegment:
    id: str
    text: str
    speaker: str


@dataclass
class Score(TextSegment):
    score: float


def get_score(mind_id: str, mind_dict, Request: TextSegment) -> Score:
    score = []
    pre_processed_input = preprocess_text(Request.text)
    lambda_function = "mind-" + mind_id
    transcript_text = Request.text
    if len(pre_processed_input) != 0:
        mind_input = json.dumps({"text": pre_processed_input})
        mind_input = json.dumps({"body": mind_input})
        logger.info('sending request to mind service')
        transcript_score = get_feature_vector(mind_input, lambda_function, mind_dict)
    else:
        transcript_score = 0.00001
        logger.warn('processing transcript: {}'.format(transcript_text))
        logger.warn('transcript too small to process. Returning default score')
    # hack to penalize out of domain small transcripts coming as PIMs - word level
    if len(transcript_text.split(' ')) < 40:
        transcript_score = 0.1 * transcript_score
    score = 1 / transcript_score
    return score

def getClusterScore(mind_vec, sent_vec):
    n1 = norm(mind_vec,axis=1).reshape(1,-1)
    n2 = norm(sent_vec,axis=1).reshape(-1,1)
    dotp = dot(sent_vec, mind_vec).squeeze(2)
    segment_scores = dotp/(n2*n1)
    return segment_scores

def get_feature_vector(mind_input, lambda_function, mind_dict):
    invoke_response = lambda_client.invoke(FunctionName=lambda_function, InvocationType='RequestResponse', Payload=mind_input)
    out_json = invoke_response['Payload'].read().decode('utf8').replace("'", '"')
    data = json.loads(json.loads(out_json)['body'])
    response = json.loads(out_json)['statusCode']
    feats = list(mind_dict['feature_vector'].values())
    mind_vector = np.array(feats).reshape(len(feats), -1)
    transcript_score = 0.00001
    transcript_mind_list=[]
    transcript_score_list = []
    if response == 200:
        logger.info('got {} from mind server'.format(response))
        feature_vector = np.array(data['sent_feats'][0])
        if len(feature_vector) > 0:
            # For paragraphs, uncomment below LOC
            # feature_vector = np.mean(np.array(feature_vector),0).reshape(1,-1)
            batch_size = min(10,feature_vector.shape[0])
            for i in range(0,feature_vector.shape[0],batch_size):
                mind_vec = np.expand_dims(np.array(mind_vector),2)
                sent_vec = feature_vector[i:i+batch_size]

                cluster_scores = getClusterScore(mind_vec,sent_vec)

                batch_scores = cluster_scores.max(1)
                transcript_score_list.extend(batch_scores)

                minds_selected = cluster_scores.argmax(1)
                transcript_mind_list.extend(minds_selected)
            transcript_score = np.mean(transcript_score_list)
            logger.info("Mind Selected is {}".format({ele:transcript_mind_list.count(ele) for ele in set(transcript_mind_list)}))
    else:
        logger.debug('Invalid response from mind service for input: {}'.format(mind_input))
        logger.debug('Returning default score')

    return transcript_score
