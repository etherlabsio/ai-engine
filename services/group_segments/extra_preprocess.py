import text_preprocessing.preprocess as tp
import nltk
import iso8601
from datetime import datetime
import json

def get_filtered_pos(filtered, pos_list=['NN', 'JJ']):
    filtered_list_temp = []
    filtered_list = []
    flag = False
    flag_JJ = False
    for word, pos in filtered:
        if pos == 'NN' or pos == 'JJ':
            flag=True
            if pos == 'JJ':
                flag_JJ = True
            else:
                flag_JJ = False
            filtered_list_temp.append((word, pos))
            continue
        if flag:
            if 'NN' in list(map(lambda x: x[1], filtered_list_temp)):
                if not flag_JJ:
                    filtered_list.append(list(map(lambda x:x[0], filtered_list_temp)))
                else:
                    filtered_list.append(list(map(lambda x:x[0], filtered_list_temp))[:-1])
                    flag_JJ = False
            filtered_list_temp = []
            flag=False
    return filtered_list

def preprocess_text(text):
    mod_texts_unfiltered = tp.preprocess(text, stop_words=False, remove_punct=False)
    mod_texts = []

    for index, sent in enumerate(mod_texts_unfiltered):

        filtered_list = tp.st_get_candidate_phrases(sent)
        if len(filtered_list)==0:
            continue

        flag = False
        for kp in filtered_list:
            if len(kp.split(" "))>1:
                flag = True
        if not flag:
            continue

        if len(sent.split(' ')) > 250:
            length = len(sent.split(' '))
            split1 = ' '.join([i for i in sent.split(' ')[:round(length / 2)]])
            split2 = ' '.join([i for i in sent.split(' ')[round(length / 2):]])
            mod_texts.append(split1)
            mod_texts.append(split2)
            continue

        if len(sent.split(' ')) <= 4:
                continue

        mod_texts.append(sent)
    if len(mod_texts) <=0:
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
