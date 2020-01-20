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
import action_detector as ad
import numpy as np
import pickle
from log.logger import setup_server_logger

s3 = boto3.resource("s3")

logger = logging.getLogger(__name__)
setup_server_logger(debug=False)  # default False for disabling debug mode

whitelisted_channel = ['01DJSFMQ5MP8AX83Y89QC6T39E','01DBB3SN99AVJ8ZWJDQ57X9TGX','01DBB3SN874B4V18DCP4ATMRXA']

def load_model():
    bucket = os.getenv("BUCKET_NAME")
    model_path = os.getenv("MODEL")

    modelObj = s3.Object(bucket_name=bucket, key=model_path)
    state_dict = torch.load(
        io.BytesIO(modelObj.get()["Body"].read()), map_location="cpu"
    )
    return state_dict


# load the model when lambda execution context is created
state_dict = load_model()
config = BertConfig()
model = ad.BertForActionItemDetection(config)
model.load_state_dict(state_dict)
model.eval()
logger.info(f"Model loaded for evaluation")


def handler(event, context):

    logger.info("POST request recieved", extra={"event['body']:": event["body"]})

    if isinstance(event["body"], str):
        json_request = json.loads(event["body"])
    else:
        json_request = event["body"]

    try:
        ai_detector = ad.ActionItemDetector(json_request["segments"], model)
        (action_items, decisions,questions,) = ai_detector.get_action_decision_subjects_list()

        response = json.dumps({"actions": action_items, "decisions": decisions,"questions": questions})
        #Posting result on question-detection channel
        try:
            if len(questions)>0 and (json_request["contextId"].upper() in whitelisted_channel):
                for quest in questions:
                    slack_msg = "*Question*: {}".format(quest['subject'])
                    slack_web_hook_url = "https://hooks.slack.com/services/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"
                    slack_payload = {"text": slack_msg}
                    slack_response = requests.post(
                        url=slack_web_hook_url, data=json.dumps(slack_payload).encode()
                    )
        except KeyError:
            pass
        logger.info("Action and decision extraction success")
        return {"statusCode": 200, "body": response}
        

    except Exception as e:
        logger.error(
            "Error processing request", extra={"err": e, "request": json_request},
        )
        response = json.dumps({"actions": []})
        return {"statusCode": 404, "body": response}
