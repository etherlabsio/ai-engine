from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config
from copy import deepcopy
from numpy import dot
import math
from numpy.linalg import norm
import numpy as np
logger = logging.getLogger(__name__)

config = Config(connect_timeout=240, read_timeout=240, retries={"max_attempts": 0},)
lambda_client = boto3_client("lambda", config=config)

def cosine(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def get_ner(paragraph_list ,lambda_function):
    ner_result = []
    count = len(paragraph_list)
    for itr in range(count):
        mind_input = json.dumps({"originalText": paragraph_list[itr]})
        mind_input = json.dumps({"body": mind_input})
        invoke_response = lambda_client.invoke(
            FunctionName=lambda_function,
            InvocationType="RequestResponse",
            Payload=mind_input,
        )

        out_json = invoke_response["Payload"].read().decode("utf8").replace("'", '"')
        data = json.loads(json.loads(out_json)["body"])
        response = json.loads(out_json)["statusCode"]

        if response == 200:
            ner_result.append(data)
        else:
            print (out_json)
            print("Invalid response from  mind service")
    return ner_result

def get_feature_vector(input_list, lambda_function):
    batches_count = 300
    feature_vector = []
    mind_score = []
    count = math.ceil(len(input_list) / batches_count)
    for itr in range(count):
        extra_input = deepcopy(
            input_list[itr * batches_count : (itr + 1) * batches_count]
        )
        mind_input = json.dumps({"text": extra_input})
        mind_input = json.dumps({"body": mind_input})
        logger.info(
            "getting feature vector from mind service", extra={"iteration count:": itr},
        )
        invoke_response = lambda_client.invoke(
            FunctionName=lambda_function,
            InvocationType="RequestResponse",
            Payload=mind_input,
        )
        logger.info("Request Sent", extra={"iteration count": itr})
        out_json = invoke_response["Payload"].read().decode("utf8").replace("'", '"')
        data = json.loads(json.loads(out_json)["body"])
        response = json.loads(out_json)["statusCode"]

        if response == 200:
            feature_vector.extend(np.array(data["sent_feats"][0]))
        else:
            logger.error("Invalid response from  mind service")
    return feature_vector
