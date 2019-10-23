try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import logging
import uuid

import boto3
import requests

import torch
from bert_utils import BertConfig
import action_detector as ad #import get_ai_sentences, get_ai_users, BertForActionItemDetection
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
model = ad.BertForActionItemDetection(config)
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
        #loop through json request segments and generate action item 
        ai_sent_list = []
        ai_user_list = []
        segment_id_list = []
        assignees_list = []
        isAssigneePrevious_list = []
        isAssigneeBoth_list = []

        for seg_object in json_request['segments']:

            curr_assignees_list = []
            curr_isAssigneePrevious_list = []
            curr_isAssigneeBoth_list = []

            transcript_text = seg_object['originalText']
            # get the AI probabilities for each sentence in the transcript
            curr_ai_list = ad.get_ai_sentences(model, transcript_text)
            curr_ai_user_list = ad.get_ai_users(curr_ai_list)
            curr_segment_id_list=[seg_object['id']]*len(curr_ai_list)

            for ai_user in curr_ai_user_list:
                if ai_user==0:
                    curr_assignees_list+=[seg_object['spokenBy']]
                else:
                    curr_assignees_list+=['NA']
                if ai_user==1:
                    curr_isAssigneePrevious_list.append(True)
                else:
                    curr_isAssigneePrevious_list.append(False)
                if ai_user==2:
                    curr_isAssigneeBoth_list.append(True)
                else:
                    curr_isAssigneeBoth_list.append(False)

            ai_sent_list+=curr_ai_list
            ai_user_list+=curr_ai_user_list
            segment_id_list+=curr_segment_id_list
            assignees_list+=curr_assignees_list
            isAssigneePrevious_list+=curr_isAssigneePrevious_list
            isAssigneeBoth_list+=curr_isAssigneeBoth_list

        uuid_list = []
        for i in range(len(ai_sent_list)):
            uuid_list.append(str(uuid.uuid1()))


        ai_response_list = []
        for uuid_,segment,action_item,assignee,is_prev_user,is_both in zip(uuid_list,segment_id_list,ai_sent_list,assignees_list,isAssigneePrevious_list,isAssigneeBoth_list):
            ai_response_list.append({"id": uuid_,
                "subject": action_item,
                "segment_ids": [segment],
                "assignees": assignee,
                "is_assignee_previous": is_prev_user,
                "is_assignee_both": is_both})

        #placeholder decision list
        decision_response_list = [{'id': str(str(uuid.uuid1())),
                                    'segment_ids': ['seg1'],
                                        'subject': 'decision_text1'},
                                {'id': str(str(uuid.uuid1())),
                                    'segment_ids': ['seg2'],
                                        'subject': 'decision_text2'}]

        response = json.dumps({"action_items": ai_response_list,
            "decisions": decision_response_list})
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
