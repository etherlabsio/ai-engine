try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import logging

import boto3
import requests

import torch
from bert_utils import BertConfig
from action_detector import get_ai_sentences, BertForActionItemDetection
import numpy as np
import pickle
from log.logger import setup_server_logger

s3 = boto3.resource('s3')

logger = logging.getLogger(__name__)
setup_server_logger(debug=True)  # default False for disabling debug mode


def load_model():
    bucket = os.getenv('BUCKET_NAME')
    model_path = os.getenv('MODEL')

    modelObj = s3.Object(
        bucket_name=bucket,
        key=model_path
    )
    state_dict = torch.load(io.BytesIO(modelObj.get()["Body"].read()), map_location='cpu')
    return state_dict


# load the model when lambda execution context is created
state_dict = load_model()
config = BertConfig()
model = BertForActionItemDetection(config)
model.load_state_dict(state_dict)
model.eval()
logger.info(f'Model loaded for evaluation')


def handler(event, context):
    logger.info("POST request recieved", extra={"event['body']:": event['body']})
    if isinstance(event['body'], str):
        json_request = json.loads(event['body'])
    else:
        json_request = event['body']

    try:
        transcript_text = json_request['segments'][0]['originalText']
        # get the AI probabilities for each sentence in the transcript
        ai_sent_list = get_ai_sentences(model, transcript_text)
        
        if len(ai_sent_list) > 0:
            has_action_item = 1
        else:
            has_action_item = 0
        
        ai_sent_list = '| '.join(ai_sent_list)
        response = json.dumps({"has_action_item": has_action_item,
            "action_item_text": ai_sent_list})
        return {
            "statusCode": 200,
            "body" : response
        }
    except Exception as e:
        logger.error(
            "Error processing request", extra={"err": e, "request": json_request}
        )
        response = json.dumps({"action_items": []})
        return {
            "statusCode": 404,
            "body" : response
        }
