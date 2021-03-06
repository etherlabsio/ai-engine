try:
    import unzip_requirements
except ImportError:
    pass

import os
import logging
from log.logger import setup_server_logger
import json
from mind_utils import load_mind_features
from scorer import transport as tp_scorer
from group_segments import transport as tp_gs
from group_segments import grouper
from scorer.scorer import get_score
from group_segments import extra_preprocess
from copy import deepcopy

logger = logging.getLogger()
setup_server_logger(debug=False)


def handler(event, context):
    logger.info("POST request Recieved: ", extra={"Request": event})
    if isinstance(event['body'], str):
        json_request = json.loads(event['body'])
    else:
        json_request = event["body"]
    try:
        mind_dict = load_mind_features(json_request["detail"]["mindId"].lower())
        if json_request["type"] == "segment_analyzer.extract_features":
            Request = tp_scorer.decode_json_request(json_request["detail"])
            scores = list(
                map(
                    lambda s: get_score(
                        Request.mind_id,
                        mind_dict,
                        s,
                        Request.context_id,
                        Request.instance_id,
                        for_pims=True,
                    ),
                    Request.segments,
                )
            )

            output_pims = {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"analyzedSegment": True}),
            }

        else:
            Request_obj = tp_gs.decode_json_request(json_request['detail'])
            mindId = str(Request_obj.mind_id).lower()
            lambda_function = "mind-" + mindId
            if not Request_obj.segments:
                return json({"msg": "No segments to process"})
            topics = {}
            pim = {}
            topics, pim = grouper.get_groups(
                Request_obj, lambda_function, mind_dict, for_pims=True
            )
            topics["contextId"] = Request_obj.context_id
            topics["instanceId"] = Request_obj.instance_id
            topics["mindId"] = Request_obj.mind_id
            output_pims = extra_preprocess.format_pims_output(
                pim,
                json_request["detail"],
                Request_obj.segments_map,
                Request_obj.mind_id,
            )
    except Exception as e:
        logger.warning("Unable to compute PIMs", extra={"exception": e})
        output_pims = {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"err": "Unable to extract topics " + str(e)}),
        }
    return output_pims
