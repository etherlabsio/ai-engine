import sys
import logging
import json
from copy import deepcopy
from artifacts_updater import update_artifacts
from log.logger import setup_server_logger

logger = logging.getLogger()
setup_server_logger(debug=False)


def handler(event, context):
    if True:
        if isinstance(event["body"], str):
            json_request = json.loads(event["body"])
        else:
            json_request = event["body"]
        response = update_artifacts.update_artifacts(json_request)
    else: #except Exception as e:
        output_pims = {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"err": "Unable to extract topics "}),
        }
    # pim['extracted_topics'] = topics
    return True
