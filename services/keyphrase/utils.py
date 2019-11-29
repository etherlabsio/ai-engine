from datetime import datetime
import iso8601
import itertools
import json
from collections import OrderedDict
import hashlib
from typing import List, Dict, Tuple, Union
import numpy as np
from io import BytesIO
import uuid


class KeyphraseUtils(object):
    def __init__(self):
        pass

    def hash_phrase(self, phrase: str) -> str:
        hash_object = hashlib.md5(phrase.encode())
        hash_str = hash_object.hexdigest()
        return hash_str

    def hash_sha_object(self) -> str:
        uid = uuid.uuid4()
        uid = str(uid)
        hash_object = hashlib.sha1(uid.encode())
        hash_str = hash_object.hexdigest()
        return hash_str

    def map_embeddings_to_phrase(
        self, phrase_list: List, embedding_list: List
    ) -> Tuple[Dict, Dict]:
        phrase_hash_dict = dict(
            zip(map(self.hash_phrase, phrase_list), phrase_list)
        )
        phrase_embedding_dict = dict(
            zip(map(self.hash_phrase, phrase_list), embedding_list)
        )

        return phrase_hash_dict, phrase_embedding_dict

    def serialize_to_npz(self, embedding_dict: Dict, file_name: str):
        np.savez_compressed(file_name, **embedding_dict)

        return file_name + ".npz"

    def deserialize_from_npz(self, file_name: Union[str, bytes]):
        if isinstance(file_name, bytes):
            file_name = BytesIO(file_name)

        npz_file = np.load(file_name)

        return npz_file

    def formatTime(self, tz_time, datetime_object=False):
        isoTime = iso8601.parse_date(tz_time)
        ts = isoTime.timestamp()
        ts = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S:%f")

        if datetime_object:
            ts = datetime.fromisoformat(ts)
        return ts

    def sort_by_value(self, item_list, key=1, order="desc"):
        """
        A utility function to sort lists by their value.
        Args:
            key:
            item_list:
            order:

        Returns:

        """

        if order == "desc":
            sorted_list = sorted(item_list, key=lambda x: x[key], reverse=True)
        else:
            sorted_list = sorted(
                item_list, key=lambda x: x[key], reverse=False
            )

        return sorted_list

    def sort_dict_by_value(self, dict_var, order="desc", key=None):
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

    def write_to_json(self, data, file_name="keyphrase_validation"):
        with open(file_name + ".json", "w", encoding="utf-8") as f_:
            json.dump(data, f_, ensure_ascii=False, indent=4)

        return file_name + ".json"

    def read_segments(self, segment_object):

        segment_list = []

        for i in range(len(segment_object)):
            segment_list.append(segment_object[i].get("originalText"))

        return segment_list

    def post_process_output(
        self,
        keyphrase_object,
        dict_key="descriptive",
        preserve_singlewords=False,
    ):

        for i, kp_item in enumerate(keyphrase_object):

            # Post-process entities
            entity_dict = kp_item["entities"]
            distinct_entities = list(entity_dict.keys())
            entity_scores = list(entity_dict.values())

            processed_entities = self.post_process_entities(distinct_entities)
            keyphrase_object[i]["entities"] = dict(
                zip(processed_entities, entity_scores)
            )

            # Post-process keyphrases
            keyphrase_dict = kp_item[dict_key]

            # Remove the first occurrence of entity in the list of keyphrases
            unwanted_kp_list = []
            for entities in distinct_entities:
                for keyphrase in keyphrase_dict.keys():
                    if keyphrase in entities:
                        # distinct_keyword_list.remove(keyphrase)
                        unwanted_kp_list.append(keyphrase)

            # Remove the unwanted keyphrases from dict
            for phrase in unwanted_kp_list:
                keyphrase_dict.pop(phrase, None)

            # Place the single keywords in the end of the list.
            multiphrase_dict = {
                words: values
                for words, values in keyphrase_dict.items()
                if len(words.split()) > 1
            }
            singleword_dict = {
                words: values
                for words, values in keyphrase_dict.items()
                if len(words.split()) == 1
            }

            if preserve_singlewords:
                multiphrase_dict.update(singleword_dict)

            keyphrase_object[i][dict_key] = multiphrase_dict

        return keyphrase_object

    def post_process_entities(self, entity_list):
        processed_entities = []

        # Remove duplicates from the single phrases which are occurring in multi-keyphrases
        multi_phrases = [
            phrases for phrases in entity_list if len(phrases.split()) > 1
        ]
        single_phrase = [
            phrases for phrases in entity_list if len(phrases.split()) == 1
        ]
        for kw in single_phrase:
            for kw_m in multi_phrases:
                r = kw_m.find(kw)
                if r > -1:
                    try:
                        single_phrase.remove(kw)
                    except Exception:
                        continue

        # Remove same word occurrences in a multi-keyphrase
        for multi_key in multi_phrases:
            kw_m = multi_key.split()
            unique_kp_list = list(dict.fromkeys(kw_m))
            multi_keyphrase = " ".join(unique_kp_list)
            if len(multi_keyphrase) > 0:
                processed_entities.append(multi_keyphrase)

        processed_entities.extend(single_phrase)

        return processed_entities

    def limit_phrase_list(
        self,
        entities_dict,
        keyphrase_dict,
        phrase_limit=10,
        entities_limit=5,
        entity_quality_score=0,
        keyphrase_quality_score=0,
        remove_phrases=False,
        rank_by="pagerank",
        sort_by="loc",
        final_sort=False,
    ):
        modified_entity_dict = {}
        modified_keyphrase_dict = {}

        rank_key_dict = {
            "pagerank": 0,
            "segment_relevance": 1,
            "boosted_score": 2,
            "norm_boosted_score": 3,
            "order": "desc",
        }

        sort_key_dict = {"loc": -1, "preference": 4, "order": "asc"}

        if remove_phrases:
            for entity, scores in entities_dict.items():
                boosted_score = scores[rank_key_dict.get("boosted_score")]
                norm_boosted_score = scores[
                    rank_key_dict.get("norm_boosted_score")
                ]

                entity_score = boosted_score
                if final_sort:
                    entity_score = norm_boosted_score

                if entity_score >= entity_quality_score:
                    modified_entity_dict[entity] = scores

            for phrase, scores in keyphrase_dict.items():
                boosted_score = scores[rank_key_dict.get("boosted_score")]
                norm_boosted_score = scores[
                    rank_key_dict.get("norm_boosted_score")
                ]

                keyphrase_score = boosted_score
                if final_sort:
                    keyphrase_score = norm_boosted_score

                if keyphrase_score > keyphrase_quality_score:
                    modified_keyphrase_dict[phrase] = scores

        else:
            modified_keyphrase_dict = keyphrase_dict
            modified_entity_dict = entities_dict

        ranked_entities_dict, ranked_keyphrase_dict = self._sort_phrase_dict(
            keyphrase_dict=modified_keyphrase_dict,
            entity_dict=modified_entity_dict,
            rank_by=rank_by,
            sort_by=sort_by,
            rank_key_dict=rank_key_dict,
            sort_key_dict=sort_key_dict,
            final_sort=final_sort,
        )

        if final_sort:
            (
                ranked_entities_dict,
                ranked_keyphrase_dict,
            ) = self._slice_phrase_dict(
                entities_dict=ranked_entities_dict,
                keyphrase_dict=ranked_keyphrase_dict,
                phrase_limit=phrase_limit,
                entities_limit=entities_limit,
            )

            # Combine entities and keyphrases
            final_result_dict = {
                **ranked_entities_dict,
                **ranked_keyphrase_dict,
            }

            # Sort chronologically
            sorted_keyphrase_dict = self.sort_dict_by_value(
                dict_var=final_result_dict,
                key=sort_key_dict[sort_by],
                order=sort_key_dict["order"],
            )

            return sorted_keyphrase_dict

        else:
            return ranked_entities_dict, ranked_keyphrase_dict

    def _sort_phrase_dict(
        self,
        keyphrase_dict,
        entity_dict,
        rank_key_dict,
        sort_key_dict,
        rank_by,
        sort_by,
        final_sort,
    ):

        # Sort by rank/scores
        ranked_keyphrase_dict = self.sort_dict_by_value(
            dict_var=keyphrase_dict,
            key=rank_key_dict[rank_by],
            order=rank_key_dict["order"],
        )

        # Sort Entities by preference
        ranked_entities_dict = self.sort_dict_by_value(
            dict_var=entity_dict,
            key=rank_key_dict["boosted_score"],
            order=rank_key_dict["order"],
        )
        if final_sort:
            ranked_entities_dict = self.sort_dict_by_value(
                dict_var=entity_dict,
                key=rank_key_dict["norm_boosted_score"],
                order=rank_key_dict["order"],
            )

        return ranked_entities_dict, ranked_keyphrase_dict

    def _slice_phrase_dict(
        self, entities_dict, keyphrase_dict, phrase_limit=10, entities_limit=5
    ):

        word_limit = phrase_limit - entities_limit
        if len(list(entities_dict.keys())) >= entities_limit:
            modified_entity_dict = dict(
                itertools.islice(entities_dict.items(), entities_limit)
            )

            limited_keyphrase_dict = dict(
                itertools.islice(keyphrase_dict.items(), word_limit)
            )
        else:
            num_of_entities = len(list(entities_dict.keys()))
            modified_entity_dict = dict(
                itertools.islice(entities_dict.items(), num_of_entities)
            )
            difference = phrase_limit - num_of_entities
            limited_keyphrase_dict = dict(
                itertools.islice(keyphrase_dict.items(), difference)
            )

        return modified_entity_dict, limited_keyphrase_dict
