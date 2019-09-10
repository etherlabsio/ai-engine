from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config

#logger = logging.getLogger()

config = Config(connect_timeout=240, read_timeout = 240, retries={'max_attempts': 0}, )
lambda_client = boto3_client('lambda', config=config)

def cosine(vec1, vec2):
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))


def get_feature_vector(mind_input, lambda_function):
    print ("Getting feature vectors from Mind service.")
    invoke_response = lambda_client.invoke(FunctionName=lambda_function, InvocationType='RequestResponse', Payload=mind_input)
    print("Request sent")
    out_json = invoke_response['Payload'].read().decode('utf8').replace("'", '"')
    data = json.loads(json.loads(out_json)['body'])
    response = json.loads(out_json)['statusCode']

    if response == 200:
        feature_vector = data['sent_feats'][0]
        print ("recieved feature vector from Mind service.")
    else:
        print("Invalid response from mind service")
    return feature_vector
