import jsonlines
import logging
import json as js
import os
from pathlib import Path
from collections import OrderedDict
import requests
import numpy as np
import uuid
import hashlib

from object_def import (
    UserID,
    InputData,
    MetaData,
    UserVectorData,
    UserFeatureMap,
    UserMetaData,
    HashResult,
)

logger = logging.getLogger(__name__)


class Utils(object):
    def __init__(self, web_hook_url, s3_client=None):
        self.web_hook_url = web_hook_url
        self.s3_client = s3_client
        self.validation_dict = {}

    def hash_sha_object(self) -> str:
        uid = uuid.uuid4()
        uid = str(uid)
        hash_object = hashlib.sha1(uid.encode())
        hash_str = hash_object.hexdigest()
        return hash_str

    def make_validation_data(
        self,
        req_data,
        user_list,
        user_scores,
        suggested_user_list,
        word_list,
        segment_users,
        original_user_score,
        original_user_names,
        upload=False,
    ):
        segment_obj = req_data["segments"]
        instance_id = req_data["instanceId"]
        context_id = req_data["contextId"]
        input_keyphrase_list = req_data["keyphrases"]

        segment_ids = []
        segment_texts = []

        for i in range(len(segment_obj)):
            segment_id = segment_obj[i]["id"]
            segment_text = segment_obj[i]["originalText"]

            segment_ids.append(segment_id)
            segment_texts.append(segment_text)

        text = " \n ".join(segment_texts)
        self.validation_dict.update(
            {
                "text": text,
                "labels": user_list,
                "meta": {
                    "contextId": context_id,
                    "instanceId": instance_id,
                    "segmentId": segment_ids,
                    "originalUsers": original_user_names,
                    "originalUserScores": original_user_score,
                    "suggestedUsers": suggested_user_list,
                    "userScore": user_scores,
                    "keyphrases": input_keyphrase_list,
                    "relatedWords": word_list,
                    "positiveLabels": segment_users,
                },
            }
        )

        if upload:
            self._upload_validation_data(instance_id=instance_id, context_id=context_id)

    def _upload_validation_data(self, instance_id, context_id, prefix="watchers_"):
        validation_id = self.hash_sha_object()
        file_name = (
            prefix + context_id + "_" + instance_id + "_" + validation_id + ".jsonl"
        )
        with jsonlines.open(file_name, mode="w") as writer:
            writer.write(self.validation_dict)

        s3_path = "validation/recommendations/" + file_name

        try:
            self.s3_client.upload_to_s3(file_name=file_name, object_name=s3_path)
        except Exception as e:
            logger.warning(e)

        logger.info("Saved validation data", extra={"validationPath": s3_path})

        # Once uploading is successful, check if NPZ exists on disk and delete it
        local_path = Path(file_name).absolute()
        if os.path.exists(local_path):
            os.remove(local_path)

    def post_to_slack(
        self, req_data, user_list, user_scores, suggested_user_list, word_list
    ):
        instance_id = req_data["instanceId"]
        input_keyphrase_list = req_data["keyphrases"]

        service_name = "recommendation-service"
        msg_text = "*Recommended users for meeting: {}* \n *Segment summary*: ```{}```\n".format(
            instance_id, self._reformat_list_to_text(input_keyphrase_list)
        )

        msg_format = "[{}]: {} *Related Users*: ```{}```\n *User Confidence Scores*: ```{}```\n *Suggested Users*: ```{}```\n *Related Words*: ```{}```".format(
            service_name,
            msg_text,
            self._reformat_list_to_text(user_list),
            self._reformat_list_to_text(user_scores),
            self._reformat_list_to_text(suggested_user_list),
            self._reformat_list_to_text(word_list),
        )

        slack_payload = {"text": msg_format}
        requests.post(url=self.web_hook_url, data=js.dumps(slack_payload).encode())

    def _reformat_list_to_text(self, input_list: InputData):
        try:
            if type(input_list[0]) != str:
                formatted_text = ", ".join(["{:.2f}".format(i) for i in input_list])
            else:
                formatted_text = ", ".join([str(w) for w in input_list])
        except Exception as e:
            formatted_text = input_list
            logger.warning(e)

        return formatted_text

    def sort_dict_by_value(self, dict_var: HashResult, order="desc", key=None):
        """
        A utility function to sort lists by their value.
        Args:
            item_list:
            order:

        Returns:

        """
        item_list = dict_var.items()
        if order == "desc":
            if key is not None:
                sorted_list = sorted(
                    item_list, key=lambda x: (x[1][key], x[0]), reverse=True
                )
            else:
                sorted_list = sorted(
                    item_list, key=lambda x: (x[1], x[0]), reverse=True
                )
        else:
            if key is not None:
                sorted_list = sorted(
                    item_list, key=lambda x: (x[1][key], x[0]), reverse=False
                )
            else:
                sorted_list = sorted(
                    item_list, key=lambda x: (x[1], x[0]), reverse=False
                )

        return OrderedDict(sorted_list)

    def normalize(self, score, scores_list):
        normalized_score = (score - np.mean(scores_list)) / (
            np.max(scores_list) - np.min(scores_list)
        )
        return normalized_score
