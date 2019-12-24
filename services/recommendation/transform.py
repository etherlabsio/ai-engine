import json as js
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class FileTransform(object):
    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

    def read_json(self, json_file):
        with open(json_file) as f_:
            meeting = js.load(f_)
        return meeting

    def load_pickle(self, byte_string=None, file_name=None):
        if file_name is not None:
            with open(os.path.join(os.getcwd(), file_name), "rb") as f_:
                data = pickle.load(f_)
        else:
            data = pickle.loads(byte_string)

        return data

    def format_reference_response(self, resp, function_name):
        user_dict = {}

        for info in resp[function_name]:
            context_obj = info["hasContext"]
            meeting_obj = context_obj["hasMeeting"]
            for m_info in meeting_obj:
                segment_obj = m_info["hasSegment"]
                for segment_info in segment_obj:
                    segment_kw = []
                    try:
                        user_id = segment_info.get("authoredBy")["xid"]
                        user_name = segment_info.get("authoredBy")["name"]

                        try:
                            user_dict[user_id]
                        except KeyError:
                            user_dict.update(
                                {
                                    user_id: {
                                        "name": user_name,
                                        "keywords": None,
                                    }
                                }
                            )

                        keyword_object = segment_info["hasKeywords"]
                        segment_kw.extend(list(set(keyword_object["values"])))

                        user_kw_list = user_dict[user_id].get("keywords")
                        if user_kw_list is not None:
                            user_kw_list.extend(segment_kw)
                            user_dict[user_id].update(
                                {"keywords": list(set(user_kw_list))}
                            )
                        else:
                            user_dict[user_id].update({"keywords": segment_kw})
                    except Exception as e:
                        logger.warning(e)
                        continue

        return user_dict

    def format_response(self, resp, function_name):
        user_id_dict = {}
        user_kw_dict = {}
        user_kw = []

        for info in resp[function_name]:
            context_obj = info["hasContext"]
            meeting_obj = context_obj["hasMeeting"]
            for m_info in meeting_obj:
                segment_obj = m_info["hasSegment"]
                for segment_info in segment_obj:
                    try:
                        user_id = segment_info.get("authoredBy")["xid"]
                        user_name = segment_info.get("authoredBy")["name"]
                        user_id_dict.update({user_id: user_name})

                        keyword_object = segment_info["hasKeywords"]
                        user_kw.extend(list(set(keyword_object["values"])))
                    except Exception as e:
                        logger.warning(e)
                        continue

                    user_kw_dict.update({user_id: user_kw})

        return user_kw_dict, user_id_dict
