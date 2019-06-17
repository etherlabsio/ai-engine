import json
import logging
from boto3 import client as boto3_client
import numpy as np
from mind.response import MindResponse

from boto3 import client as boto3_client
from botocore.client import Config

config = Config(
    connect_timeout=60,
    read_timeout=240,
    retries={'max_attempts': 0},
)
lambda_client = boto3_client('lambda', config=config)

logger = logging.getLogger()


class AWSLambdaClient:
    def __init__(self, aws=boto3_client('pim-scorer')):
        self.client = aws

    def calculate(self, mind_id: str, text: str) -> MindResponse:
        req = json.dumps({"body": {"text": text}})
        invoke_response = self.client.invoke(FunctionName="mind-" +
                                             mind_id.lower(),
                                             InvocationType='RequestResponse',
                                             Payload=req)
        return self.__decodeJSON(invoke_response)

    def __decodeJSON(self, invoke_response):
        out_json = invoke_response['Payload'].read().decode('utf8').replace(
            "'", '"')
        data = json.loads(json.loads(out_json)['body'])
        status_code = json.loads(out_json)['statusCode']
        if status_code != 200:
            raise "aws mind lambda was unsuccessful"

        feature_vector, mind_vector, nsp_list = np.array(
            data['sent_feats'][0]), np.array(
                data['mind_feats'][0]), data['sent_nsp_scores'][0]
        return MindResponse(feature_vector, mind_vector, nsp_list)
