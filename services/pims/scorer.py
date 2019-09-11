import numpy as np
from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config

logger = logging.getLogger()

config = Config(connect_timeout=60, read_timeout=240, retries={'max_attempts': 0}, )
lambda_client = boto3_client('lambda', config=config)


def get_cluster_score(sent_vec, mind_vec, mind_nsp, nsp_dampening_factor=0.7):
    cosine_sim = cosine(sent_vec, mind_vec)
    nsp_score = mind_nsp * nsp_dampening_factor
    score = np.mean([cosine_sim, nsp_score])
    return score


def cosine(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def get_score(mind_input, lambda_function):
    invoke_response = lambda_client.invoke(FunctionName=lambda_function,
                                           InvocationType='RequestResponse',
                                           Payload=mind_input)
    out_json = invoke_response['Payload'].read().decode(
        'utf8').replace("'", '"')
    data = json.loads(json.loads(out_json)['body'])
    response = json.loads(out_json)['statusCode']

    transcript_score = 0.00001
    transcript_score_list = []
    if response == 200:
        logger.info('got {} from mind server'.format(response))
        feature_vector, mind_vector, nsp_list = np.array(data['sent_feats'][0]), np.array(
            data['mind_feats'][0]), data['sent_nsp_scores'][0]

        if lambda_function.lower() == "mind-gpt" and len(feature_vector) > 0:
            for sent_vec in feature_vector:
                sent_score_list = []
                for mind_vec in mind_vector:
                    sent_score_list.append(cosine(sent_vec, mind_vec))
                transcript_score_list.append(np.max(sent_score_list))
            transcript_score = np.mean(transcript_score_list)
            return transcript_score

        # Get distance metric
        if len(feature_vector) > 0:
            for sent_vec, sent_nsp_list in zip(feature_vector, nsp_list):
                sent_score_list = []
                for mind_vec, mind_nsp in zip(mind_vector, sent_nsp_list):
                    sent_score_list.append(get_cluster_score(
                        sent_vec, mind_vec, mind_nsp))
                transcript_score_list.append(np.max(sent_score_list))
            transcript_score = np.mean(transcript_score_list)
    else:
        logger.debug(
            'Invalid response from mind service for input: {}'.format(mind_input))
        logger.debug('Returning default score')
    return transcript_score
