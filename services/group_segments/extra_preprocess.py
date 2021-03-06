import text_preprocessing.preprocess as tp
import nltk
import iso8601
from datetime import datetime
import json


def preprocess_text(text):
    mod_texts_unfiltered = tp.preprocess(text, stop_words=False, remove_punct=False)
    mod_texts = []
    if mod_texts_unfiltered is not None:
        for index, sent in enumerate(mod_texts_unfiltered):
            filtered_list = tp.st_get_candidate_phrases(sent)
            if len(filtered_list) == 0:
                continue
            elif True not in list(map(lambda x: len(x.split(" ")) > 1, filtered_list)):
                continue

            if len(sent.split(" ")) > 250:
                length = len(sent.split(" "))
                split1 = " ".join([i for i in sent.split(" ")[: round(length / 2)]])
                split2 = " ".join([i for i in sent.split(" ")[round(length / 2) :]])
                mod_texts.append(split1)
                mod_texts.append(split2)
                continue

            if len(sent.split(" ")) <= 10:
                continue

            mod_texts.append(sent)
        if len(mod_texts) == 1:
            if not (len(mod_texts[0].split(' ')) >= 20):
                return ""
        elif len(mod_texts) == 0:
            return ""
    else:
        return ""
    return mod_texts


def format_time(tz_time, datetime_object=False):
    isoTime = iso8601.parse_date(tz_time)
    ts = isoTime.timestamp()
    ts = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S:%f")

    if datetime_object:
        ts = datetime.fromisoformat(ts)
    return ts


def format_pims_output(pim, req, segmentsmap, mindId):
    pims = {}
    pims["group"] = {}
    for no in pim.keys():
        tmp_seg = []
        for seg in pim[no].keys():
            tmp_seg.append(segmentsmap[pim[no][seg][-1]])
            # uncomment the below to print the computed sentences instead of the original.
            # tmp_seg[-1]["analyzedText"] = pim[no][seg][0]
            tmp_seg[-1]["analyzedText"] = tmp_seg[-1]["originalText"]
        pims["group"][no] = tmp_seg
    pims["contextId"] = (req)["contextId"]
    pims["instanceId"] = (req)["instanceId"]
    pims["mindId"] = mindId
    response_output = {}
    response_output["statusCode"] = 200
    response_output["headers"] = {"Content-Type": "application/json"}
    response_output["body"] = json.dumps(pims)
    return response_output
