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

from .objects import Phrase, Keyphrase, Entity, Segment

SegmentType = List[Segment]


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
        phrase_hash_dict = dict(zip(map(self.hash_phrase, phrase_list), phrase_list))
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
            sorted_list = sorted(item_list, key=lambda x: x[key], reverse=False)

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

    def read_segments(self, segment_object: SegmentType):

        segment_list = []

        for segment in segment_object:
            segment_list.append(segment.originalText)

        return segment_list

    def post_process_output(
        self, phrase_object: Phrase, preserve_singlewords=False,
    ) -> Phrase:

        # Post-process entities
        entity_object = phrase_object.entities
        keyphrase_object = phrase_object.keyphrases

        processed_entities = self.post_process_entities(entity_object, keyphrase_object)

        # Place the single keywords in the end of the list.
        multiphrase_object_list = [
            multi_keyphrase
            for multi_keyphrase in keyphrase_object
            if len(multi_keyphrase.originalForm.split()) > 1
        ]
        singlephrase_object_list = [
            single_keyphrase
            for single_keyphrase in keyphrase_object
            if len(single_keyphrase.originalForm.split()) == 1
        ]

        phrase_object.entities = processed_entities
        phrase_object.keyphrases = multiphrase_object_list

        if preserve_singlewords:
            phrase_object.keyphrases.extend(singlephrase_object_list)

        return phrase_object

    def post_process_entities(
        self, entity_list: List[Entity], keyphrase_list: List[Keyphrase]
    ) -> List[Entity]:
        processed_entities = []

        # Remove duplicates from the single phrases which are occurring in multi-keyphrases
        multi_phrases = [
            phrases for phrases in entity_list if len(phrases.originalForm.split()) > 1
        ]
        single_phrase = [
            phrases for phrases in entity_list if len(phrases.originalForm.split()) == 1
        ]
        for kw_obj in single_phrase:
            for kw_m_obj in multi_phrases:
                r = kw_m_obj.originalForm.find(kw_obj.originalForm)
                if r > -1:
                    try:
                        single_phrase.remove(kw_obj)
                    except Exception:
                        continue

        # Remove same word occurrences in a multi-keyphrase
        for multi_key_obj in multi_phrases:
            kw_m = multi_key_obj.originalForm.split()
            unique_kp_list = list(dict.fromkeys(kw_m))
            multi_keyphrase = " ".join(unique_kp_list)

            if len(multi_keyphrase) > 0:
                multi_key_obj.originalForm = multi_keyphrase
                processed_entities.extend(multi_key_obj)

        # Remove the entities which already occur in keyphrases
        processed_entities.extend(single_phrase)
        for keyphrase_obj in keyphrase_list:
            keyphrase = keyphrase_obj.originalForm

            for entities_obj in processed_entities:
                entities = entities_obj.originalForm

                if entities in keyphrase or entities.lower() in keyphrase:
                    entities_obj.related_to_keyphrase = True

        return processed_entities

    def limit_phrase_list(
        self,
        entities_object: List[Entity],
        keyphrase_object: List[Keyphrase],
        phrase_limit: int = 10,
        entities_limit: int = 5,
        entity_quality_score: int = 0,
        keyphrase_quality_score: int = 0,
        remove_phrases: bool = False,
        rank_by: str = "pagerank",
        sort_by: str = "loc",
        final_sort: bool = False,
    ) -> Union[
        Tuple[Dict[str, float], List[Entity], List[Keyphrase]],
        Tuple[List[Entity], List[Keyphrase]],
    ]:

        if remove_phrases:
            for entity_obj in entities_object:
                preference_value = entity_obj.preference
                boosted_score = entity_obj.score.boosted_sim
                norm_boosted_score = entity_obj.score.norm_boosted_sim

                entity_score = boosted_score
                # Use normalized boosted score for all phrases when highlight segments come in
                if final_sort:
                    entity_score = norm_boosted_score

                # Check for relevance scores if the entity type is other than Organization or Product
                if preference_value > 2:
                    if entity_score <= entity_quality_score:
                        entity_obj.to_remove = True
                else:
                    if entity_score > 0 and entity_obj.related_to_keyphrase is not True:
                        entity_obj.to_remove = False

            for kp_object in keyphrase_object:
                boosted_score = kp_object.score.boosted_sim
                norm_boosted_score = kp_object.score.norm_boosted_sim

                keyphrase_score = boosted_score
                if final_sort:
                    keyphrase_score = norm_boosted_score

                if keyphrase_score <= keyphrase_quality_score:
                    kp_object.to_remove = True

        ranked_entities_object, ranked_keyphrase_object = self._sort_phrase_dict(
            keyphrase_object=keyphrase_object,
            entities_object=entities_object,
            rank_by=rank_by,
            final_sort=final_sort,
        )

        if final_sort:
            ranked_entities_object, ranked_keyphrase_object = self._slice_phrase_dict(
                entities_object=ranked_entities_object,
                keyphrase_object=ranked_keyphrase_object,
                phrase_limit=phrase_limit,
                entities_limit=entities_limit,
            )

            # Convert List[Objects] to dictionary for uniform sorting
            ranked_entities_dict = {
                ent_obj.originalForm: ent_obj.score.loc
                for ent_obj in ranked_entities_object
            }
            ranked_keyphrase_dict = {
                kp_obj.originalForm: kp_obj.score.loc
                for kp_obj in ranked_keyphrase_object
            }

            final_result_dict = {**ranked_entities_dict, **ranked_keyphrase_dict}

            # Sort chronologically
            sorted_phrase_dict = self.sort_dict_by_value(
                dict_var=final_result_dict, order="asc"
            )

            return sorted_phrase_dict, ranked_entities_object, ranked_keyphrase_object

        else:
            return ranked_entities_object, ranked_keyphrase_object

    def _sort_phrase_dict(
        self,
        keyphrase_object: List[Keyphrase],
        entities_object: List[Entity],
        final_sort,
        rank_by="boosted_sim",
    ) -> Tuple[List[Entity], List[Keyphrase]]:

        # Sort by rank/scores
        ranked_keyphrase_object = sorted(
            keyphrase_object,
            key=lambda keyphrase: keyphrase.score.boosted_sim,
            reverse=True,
        )
        if rank_by == "pagerank":
            ranked_keyphrase_object = sorted(
                keyphrase_object,
                key=lambda keyphrase: keyphrase.score.pagerank,
                reverse=True,
            )

        # Sort Entities by preference
        ranked_entities_object = sorted(
            entities_object, key=lambda entity: entity.preference
        )

        if final_sort:
            ranked_entities_object = sorted(
                entities_object,
                key=lambda entity: entity.score.norm_boosted_sim,
                reverse=True,
            )

        return ranked_entities_object, ranked_keyphrase_object

    def _slice_phrase_dict(
        self,
        entities_object: List[Entity],
        keyphrase_object: List[Keyphrase],
        phrase_limit=10,
        entities_limit=5,
    ):

        word_limit = phrase_limit - entities_limit

        filtered_entities_object = [
            ent_obj
            for ent_obj in entities_object
            if ent_obj.related_to_keyphrase is not True and ent_obj.to_remove is False
        ]
        filtered_keyphrase_object = [
            kp_obj for kp_obj in keyphrase_object if kp_obj.to_remove is not True
        ]

        if len(filtered_entities_object) >= entities_limit:
            modified_entity_object = filtered_entities_object[:entities_limit]
            modified_keyphrase_object = filtered_keyphrase_object[:word_limit]

        else:
            num_of_entities = len(filtered_entities_object)
            modified_entity_object = filtered_entities_object[:num_of_entities]
            difference = phrase_limit - num_of_entities
            modified_keyphrase_object = filtered_keyphrase_object[:difference]

        return modified_entity_object, modified_keyphrase_object
