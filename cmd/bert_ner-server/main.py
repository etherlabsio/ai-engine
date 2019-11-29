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
import bert_ner.bert_ner_utils as ner
import numpy as np
import pickle

s3 = boto3.resource("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
model = ner.BertForTokenClassification_custom(config)
model.load_state_dict(state_dict)
model.eval()
logger.info(f"Model loaded for evaluation")


def handler(event, context):

    logger.info(
        "POST request recieved", extra={"event['body']:": event["body"]}
    )

    if isinstance(event["body"], str):
        json_request = json.loads(event["body"])
    else:
        json_request = event["body"]

    try:
        segment = json_request["originalText"]
        ner_model = ner.BERT_NER(model)
        entities = ner_model.get_entities(segment)
        response = json.dumps({"entities": entities})
        logger.info(
            "Entity Extraction successful with {} entities detected.".format(
                entities
            )
        )
        return {"statusCode": 200, "body": response}

    except Exception as e:
        logger.error(
            "Error processing request",
            extra={"err": e, "request": json_request},
        )
        response = json.dumps({"entities": {}})
        return {"statusCode": 404, "body": response}
