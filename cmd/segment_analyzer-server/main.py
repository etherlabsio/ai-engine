try:
    import unzip_requirements
except ImportError:
    pass

import os
import logging
from log.logger import setup_server_logger
import json
from scorer import transport as tp_scorer
from group_segments import transport as tp_gs
from group_segments import grouper
from scorer.scorer import get_score, get_similar_entities, get_segment_rank, upload_segment_rank, get_segment_rank_pc
from group_segments import extra_preprocess
from copy import deepcopy
import numpy as np

logger = logging.getLogger()
setup_server_logger(debug=False)


def handler(event, context):
    if isinstance(event['body'], str):
        json_request = json.loads(event['body'])
    else:
        json_request = event["body"]
    try:
        # To compute embeddings for the segment recieved
        if json_request["type"] == "segment_analyzer.extract_features":
            # preprocess the segment text.
            Request = tp_scorer.decode_json_request(json_request["detail"])
            # check if segment is empty after preprocessing it.
            if isinstance(Request, bool) and not Request:
                logger.info("Warning: No fv upload. removed segments because of preprocessing conditions.")
                output_pims = {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"analyzedSegment": True}),
                }
                return output_pims

            vector_list = list(
                map(
                    lambda s: get_score(
                        Request.mind_id,
                        s,
                        Request.context_id,
                        Request.instance_id,
                    ),
                    Request.segments,
                )
            )
            assert(len(vector_list) == len(Request.segments))

            output_pims = {
               "statusCode": 200,
               "headers": {"Content-Type": "application/json"},
               "body": json.dumps({"analyzedSegment": True}),
            }

        else:
            logger.info("POST request Recieved: ", extra={"Request": json_request['detail']})
            Request_obj = tp_gs.decode_json_request(json_request['detail'])
            lambda_function = "mind-" + Request_obj.mind_id
            if not Request_obj.segments:
                # To Do: It's better to raise an error than directly return an error.
                return json({"msg": "No segments to process"})
            topics = {}
            pim = {}
            topics_extracted, pim = grouper.get_groups(
                Request_obj, lambda_function, for_pims=True
            )
            topics["contextId"] = Request_obj.context_id
            topics["instanceId"] = Request_obj.instance_id
            topics["mindId"] = Request_obj.mind_id
            output_pims = extra_preprocess.format_pims_output(
                pim,
                json_request["detail"],
                Request_obj.segments_map,
                Request_obj.mind_id,
                topics_extracted
            )
    except Exception as e:
       logger.warning("Unable to compute PIMs", extra={"exception": e})
       output_pims = {
          "statusCode": 200,
          "headers": {"Content-Type": "application/json"},
          "body": json.dumps({"err": "Unable to extract topics " + str(e)}),
       }
    return output_pims
