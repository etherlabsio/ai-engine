import json
from extract_topic_pims.transport import decode_json_request
from extract_topic_pims.compute_pims import get_pims
import sys
import logging
from log.logger import setup_server_logger

logger = logging.getLogger()
setup_server_logger(debug=False)


def handler(event, context):
    try:
        if isinstance(event['body'], str):
            json_request = json.loads(event['body'])
        else:
            json_request = event['body']
        #logger.info("POST request recieved", extra={"request": json_request})
        Request_obj = decode_json_request(json_request)
        # lambda_function = "mind-" + mindId
        output_pims = json.dumps({"statusCode":200, "headers": {"Content-Type": "application/json"}, "body": get_pims(Request_obj)})
    except Exception as e:
        logger.warning("Unable to extract topic", extra={"exception": e})
        output_pims = {"statusCode": 200,
                       "headers": {"Content-Type": "application/json"},
                       "body": json.dumps({"err": "Unable to extract topics " + str(e)})}
    return output_pims
