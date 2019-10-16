import json
from dataclasses import dataclass, asdict
from typing import List
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class Request:
    pim_result: list
    gs_result: dict
    gs_rec_map: dict
    pim_rec_map: dict

def decode_json_request(req) -> Request:
    if isinstance(req, str):
        req = json.load(req)

    gs_result = deepcopy(req['groups'])
    pim_result = deepcopy(req['pims'])
    gs_rec_map = {}
    pim_rec_map = {}

    for keys in gs_result.keys():
        for seg in gs_result[keys]:
            if gs_rec_map.get(seg['recordingId']) is False:
                if len(gs_result[keys]) > len(gs_result[gs_rec_map[seg['recordingId']]]):
                    gs_rec_map[seg['recordingId']] = keys
            else:
                gs_rec_map[seg['recordingId']] = keys
    for seg in pim_result["segments"]:
        pim_rec_map[seg["recordingId"]] =  seg['distance']
    logger.info("decoding results: ", extra={"gs_mapping": gs_rec_map, "pim_mapping": pim_rec_map})
    return Request(pim_result, gs_result, gs_rec_map, pim_rec_map)
