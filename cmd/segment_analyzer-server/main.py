try:
    import unzip_requirements
except ImportError:
    pass

import os
import logging
from log.logger import setup_server_logger
import json
from mind_utils import load_mind_features, load_entity_features, load_entity_graph
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
        mind_dict = load_mind_features(json_request["detail"]["mindId"].lower())
        if (json_request["detail"]["mindId"]).lower() in ["01daaqy88qzb19jqz5prjfr76y", "01daatanxnrqa35e6004hb7mbn", "01dadp74wfv607knpcb6vvxgtg", "01daaqyn9gbebc92aywnxedp0c", "01daatbc3ak1qwc5nyc5ahv2xz","01dsyjns6ky64jd9736yt0nfjz"] :
        # if (json_request["detail"]["mindId"]).lower() in ["01daaqy88qzb19jqz5prjfr76y"] :

            entity_dict_full = load_entity_features(json_request["detail"]["mindId"].lower())
            pg_scores, entity_community_map, entity_community_rank = load_entity_graph(json_request["detail"]["mindId"].lower())
            common_entities = entity_dict_full.keys() & entity_community_map.keys()
            entity_dict = {}
            for ent in common_entities:
                if True not in np.isnan([entity_dict_full[ent]]):
                    entity_dict[ent] = entity_dict_full[ent]

        if json_request["type"] == "segment_analyzer.extract_features":
            Request = tp_scorer.decode_json_request(json_request["detail"])
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
                        mind_dict,
                        s,
                        Request.context_id,
                        Request.instance_id,
                        for_pims=True,
                    ),
                    Request.segments,
                )
            )
            assert(len(vector_list) == len(Request.segments))
            if (json_request["detail"]["mindId"]).lower() in ["01daaqy88qzb19jqz5prjfr76y", "01daatanxnrqa35e6004hb7mbn", "01dadp74wfv607knpcb6vvxgtg", "01daaqyn9gbebc92aywnxedp0c", "01daatbc3ak1qwc5nyc5ahv2xz", "01dsyjns6ky64jd9736yt0nfjz"] :
           # if (json_request["detail"]["mindId"]).lower() in ["01daaqy88qzb19jqz5prjfr76y"] :
                if vector_list is not False:
                    similar_entities = list(
                        map(
                            lambda s: get_similar_entities(
                                entity_dict,
                                s[0],
                                s[1]
                            ),
                            zip(Request.segments, vector_list),
                        )
                    )

                    segment_rank = list(
                        map(
                            lambda ent_list: get_segment_rank_pc(
                                ent_list,
                                pg_scores,
                                entity_community_map,
                                entity_community_rank,
                                (Request.mind_id).lower()
                            ),
                            similar_entities,
                        )
                    )

                    result = upload_segment_rank(segment_rank, Request.instance_id, Request.context_id, Request.segments)
            output_pims = {
               "statusCode": 200,
               "headers": {"Content-Type": "application/json"},
               "body": json.dumps({"analyzedSegment": True}),
            }

        else:
            logger.info("POST request Recieved: ", extra={"Request": json_request['detail']})
            Request_obj = tp_gs.decode_json_request(json_request['detail'])
            mindId = str(Request_obj.mind_id).lower()
            lambda_function = "mind-" + mindId
            if not Request_obj.segments:
                return json({"msg": "No segments to process"})
            topics = {}
            pim = {}
            topics_extracted, pim = grouper.get_groups(
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
