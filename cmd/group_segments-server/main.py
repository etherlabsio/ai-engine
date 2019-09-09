import json
from group_segments.transport import decode_json_request
from group_segments import grouper
from group_segments.extra_preprocess import format_pims_output
import sys


def handler(event, context):
    try:
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
        topics, pim = grouper.get_groups(Request_obj, lambda_function)
        topics['contextId'] = (json_request)['contextId']
        topics['instanceId'] = (json_request)['instanceId']
        topics['mindId'] = mindId
        output_pims = format_pims_output(pim, json_request, Request_obj.segments_map, mindId)
    except Exception as e:
        print("Unable to extract topics:", e)
        output_pims = { "statusCode": 500, 
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps({"error": "Unable to extract topics"} )
        }
    #pim['extracted_topics'] = topics
    return output_pims
