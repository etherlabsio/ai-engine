import numpy as np
from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config
from dataclasses import dataclass
from pre_process import preprocess_text

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


class Scorer:
    @staticmethod
    def score(mind_id: str, mind_dict, Request: TextSegment) -> Score:
        score = []
        pre_processed_input = preprocess_text(Request.text)
        lambda_function = "mind-" + mind_id
        transcript_text = Request.text
        if len(pre_processed_input) != 0:
            mind_input = json.dumps({"text": pre_processed_input})
            mind_input = json.dumps({"body": mind_input})
            logger.info('sending request to mind service')
            transcript_score = getFeatureVector(mind_input, lambda_function, mind_dict)
        else:
            transcript_score = 0.00001
            logger.warn('processing transcript: {}'.format(transcript_text))
            logger.warn('transcript too small to process. Returning default score')
        # hack to penalize out of domain small transcripts coming as PIMs - word level
        if len(transcript_text.split(' ')) < 40:
            transcript_score = 0.1 * transcript_score
        score = 1 / transcript_score
        return score

    @staticmethod
    def getClusterScore(sent_vec, mind_vec):
        score = cosine(sent_vec, mind_vec)
        return score

    @staticmethod
    def cosine(vec1, vec2):
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    @staticmethod
    def getFeatureVector(mind_input, lambda_function, mind_dict):
        invoke_response = lambda_client.invoke(FunctionName=lambda_function, InvocationType='RequestResponse', Payload=mind_input)
        out_json = invoke_response['Payload'].read().decode('utf8').replace("'", '"')
        data = json.loads(json.loads(out_json)['body'])
        response = json.loads(out_json)['statusCode']
        feats = list(mind_dict['feature_vector'].values())
        mind_vector = np.array(feats).reshape(len(feats), -1)
        transcript_score = 0.00001
        transcript_score_list = []
        if response == 200:
            logger.info('got {} from mind server'.format(response))
            feature_vector = np.array(data['sent_feats'][0])
            if len(feature_vector) > 0:
                # For paragraphs, uncomment below LOC
                # feature_vector = np.mean(np.array(feature_vector),0).reshape(1,-1)
                for sent_vec in feature_vector:
                    sent_score_list = []
                    for mind_vec in mind_vector:
                        sent_score_list.append(getClusterScore(sent_vec, mind_vec))
                    transcript_score_list.append(np.max(sent_score_list))
                    # mind_selected_list.append(list(mind_dict['sentence'].values())[np.argmax(sent_score_list)])
                    # selected_mind = max(mind_selected_list,key=mind_selected_list.count)
                transcript_score = np.mean(transcript_score_list)
        else:
            logger.debug('Invalid response from mind service for input: {}'.format(mind_input))
            logger.debug('Returning default score')

        return transcript_score
