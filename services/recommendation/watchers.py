import json as js
import os
import pickle
import logging
from nltk import word_tokenize, pos_tag
import jsonlines
from pathlib import Path

from .lsh import WordSearch, UserSearch

logger = logging.getLogger(__name__)


class RecWatchers(object):
    def __init__(
        self,
        reference_user_file,
        reference_user_kw_vector,
        vectorizer=None,
        s3_client=None,
    ):
        self.vectorizer = vectorizer
        self.s3_client = s3_client
        self.pos_list = ["NN", "NNS", "NNP"]

        # Load and read files for rec
        self.reference_user_dict = self.read_json(
            os.path.join(os.getcwd(), reference_user_file)
        )
        self.user_vector_data = self.load_pickle(
            file_name=reference_user_kw_vector
        )

        # Initialize User query

        self.ref_user_kw_dict = {
            k: self.reference_user_dict[k]["keywords"]
            for k in self.reference_user_dict.keys()
        }
        self.us = UserSearch(
            input_dict=self.ref_user_kw_dict,
            vectorizer=self.vectorizer,
            user_vector_data=self.user_vector_data,
        )

        self.validation_dict = {}

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

    def featurize_reference_users(self):
        # Featurize reference users
        self.us.featurize()

    def _query_similar_users(self, kw_list):
        result = self.us.query(kw_list=kw_list)
        top_similar_users = sorted(result, key=result.get, reverse=True)

        similar_user_score_dict = {
            user: result[user] for user in top_similar_users
        }

        return similar_user_score_dict

    def _explainability(self, similar_user_list, kw_list):

        similar_users_kw_dict = {
            k: self.reference_user_dict[k]["keywords"]
            for k in similar_user_list
        }
        similar_users_kw_list = list(
            set(
                [
                    words
                    for user_kw in similar_users_kw_dict.values()
                    for words in user_kw
                ]
            )
        )

        ws = WordSearch(
            input_list=similar_users_kw_list, vectorizer=self.vectorizer
        )
        ws.featurize()

        result = ws.query(kw_list=kw_list)
        top_similar_words = sorted(result, key=result.get, reverse=True)
        top_similar_words = self._filter_pos(top_similar_words)

        top_words = {
            word: result[word]
            for word in top_similar_words
            if word not in kw_list
        }

        return top_words

    def _filter_pos(self, word_list):
        filtered_word = []

        for word in word_list:
            pos_word = pos_tag(word_tokenize(word))
            counter = 0
            for tags in pos_word:
                p = tags[1]
                if p in self.pos_list:
                    counter += 1

            if counter == len(word_tokenize(word)):
                filtered_word.append(word)
        return filtered_word

    def get_recommended_watchers(self, kw_list, n_users=6, n_kw=4):
        similar_user_score_dict = self._query_similar_users(kw_list=kw_list)

        top_users = [user for user, score in similar_user_score_dict.items()]
        user_scores = [
            score for user, score in similar_user_score_dict.items()
        ]
        user_names = [
            self.reference_user_dict[u].get("name") for u in top_users
        ]

        logger.info(
            "Top recommended users found",
            extra={
                "totalSimilarUsers": len(top_users),
                "users": user_names[:n_users],
                "scores": user_scores[:n_users],
            },
        )

        related_word_score_dict = self._explainability(
            similar_user_list=top_users[:n_users], kw_list=kw_list
        )
        top_words = [words for words, score in related_word_score_dict.items()]
        top_scores = [
            score for words, score in related_word_score_dict.items()
        ]

        logger.info(
            "Similarity explanation",
            extra={
                "totalRelatedWords": len(top_words),
                "words": top_words[:4],
                "scores": top_scores[:4],
            },
        )

        return user_names[:n_users], top_words[:n_kw]

    def make_validation_data(self, req_data, user_list, word_list):
        segment_obj = req_data["segments"]
        instance_id = req_data["instanceId"]
        input_keyphrase_list = req_data["keyphrases"]

        for i in range(len(segment_obj)):
            segment_id = segment_obj[i]["id"]
            segment_text = segment_obj[i]["originalText"]
            self.validation_dict.update(
                {
                    "text": segment_text,
                    "labels": user_list,
                    "meta": {
                        "instanceId": instance_id,
                        "segmentId": segment_id,
                        "inputKeyphrases": input_keyphrase_list,
                        "relatedWords": word_list,
                    },
                }
            )

    def format_validation_data(
        self, instance_id, context_id, prefix="watchers_"
    ):
        file_name = prefix + instance_id + ".jsonl"
        with jsonlines.open(file_name, mode="w") as writer:
            writer.write(self.validation_dict)

        s3_path = self.upload_validation(
            context_id=context_id,
            instance_id=instance_id,
            validation_file_name=file_name,
        )

        logger.info("Saved validation data", extra={"validationPath": s3_path})

    def upload_validation(self, context_id, instance_id, validation_file_name):
        s3_path = (
            context_id
            + "/sessions/"
            + instance_id
            + "/validation/recommendations/"
            + validation_file_name
        )

        try:
            self.s3_client.upload_to_s3(
                file_name=validation_file_name, object_name=s3_path
            )
        except Exception as e:
            logger.warning(e)

        # Once uploading is successful, check if NPZ exists on disk and delete it
        local_path = Path(validation_file_name).absolute()
        if os.path.exists(local_path):
            os.remove(local_path)

        return s3_path
