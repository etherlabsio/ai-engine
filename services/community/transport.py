import json
from dataclasses import dataclass, asdict
from typing import List
import logging
import text_preprocessing.preprocess as tp
from community import utility

logger = logging.getLogger()


@dataclass
class Request:
    segments: list
    segments_org: list

def decode_json_request(body) -> Request:
    req = body

    if isinstance(body, str):
        req = json.load(body)

    def decode_segments(seg):
        #segments_text = list(map(lambda x:tp.preprocess(x['originalText'], stop_words=False, remove_punct=False), seg['segments']))
        segments_text = list(map(lambda x:utility.preprocess_text(x['originalText']), seg['segments']))
        segments_data = seg['segments']
        for index, segment in  enumerate(segments_data):
            segments_data[index]['originalText'] = segments_text[index]
        return segments_data
    
    if req['segments'] is None:
        return False
    logger.info("segments", extra={"segments data":type(req['segments'])})
    segments_org = req
    segments = decode_segments(req)
    return Request(segments, segments_org)
