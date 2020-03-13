import json
from dataclasses import dataclass
from group_segments.artifacts_downloader import load_entity_features, load_entity_graph
import numpy as np
import logging
logger = logging.getLogger()

@dataclass
class Request:
    mind_id: str
    context_id: str
    groups: dict

@dataclass
class Artifacts:
    ent_fv: dict
    kp_entity_graph: dict
    entity_community_map: dict
    label_dict: dict
    gc: dict
    lc: dict


def decode_json_request(event) -> Request:
    if isinstance(event["body"], str):
        req = json.loads(event["body"])
    else:
        req = event["body"]

    logger.info("Decoding Request: ", extra={"Request": req})

    def decode_segments(groups):
        new_groups = {}
        for groupid, groupobj in groups.items():
            new_groups[groupid] = groupobj['analyzedSegments']

        return new_groups

    try:
        groups = decode_segments(req['group'])
        entity_dict_full = load_entity_features(req["mindId"].lower(), req["contextId"].lower())
        kp_entity_graph, entity_community_map, label_dict, gc, lc = load_entity_graph(req["mindId"].lower(), req["contextId"].lower())
        common_entities = entity_dict_full.keys() & entity_community_map.keys()

        ent_fv = {}
        for ent in common_entities:
            if True not in np.isnan([entity_dict_full[ent]]):
                ent_fv[ent] = entity_dict_full[ent]

    except Exception as e:
        raise Exception("Decoding error: {}".format(e))

    return ( Request(
        req["mindId"].lower(),
        req["contextId"].lower(),
        groups
    ), Artifacts(ent_fv, kp_entity_graph, entity_community_map, label_dict, gc, lc))
