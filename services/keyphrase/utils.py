from datetime import datetime
import iso8601
import json
from collections import OrderedDict
import hashlib
from typing import List, Dict, Tuple, Union
import numpy as np
from io import BytesIO
import uuid
import logging

from .objects import Phrase, PhraseType, KeyphraseType, EntityType, SegmentType

logger = logging.getLogger(__name__)


class KeyphraseUtils(object):
    def __init__(self, mind_dir: str, graph_filter_object=None, mind_store=None):
        self.gfilter = graph_filter_object
        self.mind_store = mind_store
        self.mind_dir = mind_dir

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

    def format_time(self, tz_time, datetime_object=False):
        iso_time = iso8601.parse_date(tz_time)
        ts = iso_time.timestamp()
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
        self,
        session_id: str,
        phrase_object: Phrase,
        preserve_singlewords=False,
        dict_key="descriptive",
        filter_by_graph: bool = False,
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

        if filter_by_graph:
            try:
                phrase_object = self._graph_filtration(
                    phrase_object, session_id=session_id
                )
            except Exception:
                phrase_object = phrase_object
                logger.warning("No graph filtration is performed")

        return phrase_object

    def _graph_filtration(self, phrase_object: Phrase, session_id: str,) -> Phrase:
        filtered_entities = []
        final_dropped_entities = []
        filtered_keyphrases = []
        final_dropped_keyphrases = []

        # Filter entities by Entity graph
        entity_graph = self.mind_store.get_object(key=session_id)
        if entity_graph is None:
            mind_id = session_id.split(":")[-1]
            mind_graph_path = self.mind_dir + mind_id + "/kp_entity_graph.pkl"
            try:
                mind_filter_graph = self.gfilter.download_mind(
                    graph_file_path=mind_graph_path
                )
                try:
                    self.mind_store.set_object(key=session_id, object=mind_filter_graph)
                    entity_graph = self.mind_store.get_object(key=session_id)
                    mind_filter_graph.clear()

                except Exception as e:
                    logger.warning(
                        "Unable to set the entity graph to redis ... Retrying",
                        extra={"warn": e},
                    )
                    try:
                        self.mind_store.delete_key(key=session_id)
                        self.mind_store.set_object(
                            key=session_id, object=mind_filter_graph
                        )
                        entity_graph = self.mind_store.get_object(key=session_id)
                    except Exception as e:
                        logger.warning(
                            "Error while setting entity graph object", extra={"warn": e}
                        )
                        entity_graph = mind_filter_graph
                        mind_filter_graph.clear()
                        # raise

            except Exception as e:
                logger.error("Unable to download entity graph", extra={"err": e})
                raise

        segment_text = phrase_object.originalText
        entity_object = phrase_object.entities
        keyphrase_object = phrase_object.keyphrases

        keyphrases = [kp.originalForm for kp in keyphrase_object]
        entity_phrases = [entity.originalForm for entity in entity_object]

        processed_entities, dropped_entities = self.gfilter.filter_entities(
            phrase=entity_phrases,
            segment_text_list=[segment_text],
            kp_graph=entity_graph,
        )
        processed_keyphrases, dropped_keyphrases = self.gfilter.filter_keyphrases(
            phrase=keyphrases, segment_text_list=[segment_text], kp_graph=entity_graph
        )

        filtered_entities.extend([ent for ent in processed_entities])
        final_dropped_entities.extend([ent for ent in dropped_entities])
        filtered_keyphrases.extend([kp for kp in processed_keyphrases])
        final_dropped_keyphrases.extend([kp for kp in dropped_keyphrases])

        try:
            for phrase in entity_object:
                if (
                    phrase.originalForm in dropped_entities
                    or phrase.value in dropped_entities
                ):
                    phrase.to_remove = True

            for phrase in keyphrase_object:
                if (
                    phrase.originalForm in dropped_keyphrases
                    or phrase.value in dropped_keyphrases
                ):
                    phrase.to_remove = True

        except Exception as e:
            logger.warning(
                "Unable to post-process keyphrases and entities", extra={"warn": e}
            )

        finally:
            entity_graph.clear()

        logger.debug(
            "Processed keyphrases & entities by ENT-KP Graph",
            extra={
                "filteredKeyphrases": filtered_keyphrases,
                "droppedKeyphrases": final_dropped_keyphrases,
                "filteredEntities": filtered_entities,
                "droppedEntities": final_dropped_entities,
            },
        )

        return phrase_object

    def post_process_entities(
        self, entity_list: EntityType, keyphrase_list: KeyphraseType
    ) -> EntityType:

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
            original_kw_m_len = len(kw_m)
            unique_kp_list = list(dict.fromkeys(kw_m))
            unique_kp_len = len(unique_kp_list)

            if original_kw_m_len != unique_kp_len:
                multi_key_obj.to_remove = True

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
        entities_object: EntityType,
        keyphrase_object: KeyphraseType,
        phrase_limit: int = 10,
        entities_limit: int = 5,
        entity_quality_score: int = 0,
        keyphrase_quality_score: int = 0,
        remove_phrases: bool = False,
        rank_by: str = "pagerank",
        sort_by: str = "loc",
        final_sort: bool = False,
    ) -> Union[
        Tuple[Dict[str, float], EntityType, KeyphraseType],
        Tuple[EntityType, KeyphraseType],
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
                if preference_value > 2 and preference_value != 5:
                    if (
                        len(entities_object) >= 2
                        and entity_score <= entity_quality_score
                    ):
                        entity_obj.to_remove = True
                    else:
                        continue
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
        keyphrase_object: KeyphraseType,
        entities_object: EntityType,
        final_sort,
        rank_by="boosted_sim",
    ) -> Tuple[EntityType, KeyphraseType]:

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
        entities_object: EntityType,
        keyphrase_object: KeyphraseType,
        phrase_limit=10,
        entities_limit=5,
    ) -> Tuple[EntityType, KeyphraseType]:

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
