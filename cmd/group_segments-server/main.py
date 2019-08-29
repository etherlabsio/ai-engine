import json 
from group_segments.transport import decode_json_request 
from group_segments import grouper
from group_segments.extra_preprocess import formatPimsOutput
import sys


def handler(event, context):
    
    if isinstance(event['body'], str):
        json_request = json.loads(event['body'])
    else:
        json_request = event['body']

    Request_obj = decode_json_request(json_request)
    mindId = str(json_request['mindId']).lower()
    lambda_function = "mind-"+mindId

    if not Request_obj.segments:
        return json({"msg": "No segments to process"})
    
    topics = {}
    pim = {}
    topics, pim = grouper.getgroups(Request_obj, lambda_function)
    topics['contextId']=(json_request)['contextId']
    topics['instanceId']=(json_request)['instanceId']
    topics['mindId']=mindId
    output_pims = formatPimsOutput(pim, json_request, Request_obj.segments_map, mindId)
    #pim['extracted_topics'] = topics
    return output_pims


