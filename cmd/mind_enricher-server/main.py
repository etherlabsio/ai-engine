import sys
import logging
import json
from copy import deepcopy
from mind_enricher import update_artifacts, transport
from log.logger import setup_server_logger

logger = logging.getLogger()
setup_server_logger(debug=False)


def handler(event, context):
    try:
        Request, Artifacts = transport.decode_json_request(event)
        response = update_artifacts.update_artifacts(Request, Artifacts)
        output_pims = {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"enriched":True}),
        }
    except Exception as e:
        logger.error("Unable to enrich Artifacts: {}".format(e))
        output_pims = {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"err": "Unable to enrich Artifacts {}".format(e)}),
        }
    return output_pims
