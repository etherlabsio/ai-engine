from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
import math
from botocore.client import Config
from copy import deepcopy
logger = logging.getLogger(__name__)

config = Config(connect_timeout=600, read_timeout=600, retries={'max_attempts': 0}, )
lambda_client = boto3_client('lambda', config=config)


def cosine(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def get_feature_vector(input_list, lambda_function):
    # logger.info("computing feature vector", extra={"msg": "getting feature vector from mind service"})
    batches_count = 100
    feature_vector = []
    count = math.ceil(len(input_list)/batches_count)
    logger.info("computing in batches", extra={"batches count": count, "number of sentences": len(input_list)})
    for itr in range(count):
        extra_input = deepcopy(input_list[itr*100:(itr+1)*100])
        mind_input = json.dumps({"text": extra_input})
        mind_input = json.dumps({"body": mind_input})
        logger.info("getting feature vector from mind service", extra={"iteration count:": itr})
        invoke_response = lambda_client.invoke(FunctionName=lambda_function, InvocationType='RequestResponse', Payload=mind_input)
        logger.info("Request Sent", extra={"iteration count": itr})
        # logger.info("computing feature vector", extra={"msg": "Request Sent"})
        out_json = invoke_response['Payload'].read().decode('utf8').replace("'", '"')
        print ("Recieved Response. Checking if it's valid.")
        data = json.loads(json.loads(out_json)['body'])
        print ("Valid Response from mind service.")
        response = json.loads(out_json)['statusCode']

        if response == 200:
            feature_vector.extend(data['sent_feats'][0])
            logger.info("Response Recieved")

            # logger.info("computing feature vector", extra={"msg": "Response Recieved"})
        else:
            logger.error("Invalid response from  mind service")
            # logger.error("computing feature vector", extra={"msg": "Invalid response from  mind service"})
    return feature_vector
