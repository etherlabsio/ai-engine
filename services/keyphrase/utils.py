from datetime import datetime
import iso8601
import itertools
import json
from collections import OrderedDict


class KeyphraseUtils(object):
    def __init__(self):
        pass

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

    def write_to_json(self, data, file_name="keyphrase_validation.json"):
        with open(file_name, "w", encoding="utf-8") as f_:
            json.dump(data, f_, ensure_ascii=False, indent=4)

    def read_segments(self, segment_object):

        segment_list = []

        for i in range(len(segment_object)):
            segment_list.append(segment_object[i].get("originalText"))

        return segment_list

    def post_process_output(
        self, keyphrase_object, dict_key="descriptive", preserve_singlewords=False
    ):

        # # Get distinct entities and keyphrases
        # distinct_entities = list(dict.fromkeys(entity_list))
        # # distinct_keyword_list = list(dict.fromkeys(keyphrase_list))
        #
        # # Post-process entities
        # distinct_entities = self.post_process_entities(distinct_entities)
        # # distinct_entities = self.post_process_entities(entity_dict=entity_list)

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
        multi_phrases = [phrases for phrases in entity_list if len(phrases.split()) > 1]
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

        # Remove single lettered entity that are coming up
        for entities in processed_entities:
            tmp_entitites = list(entities)
            if len(tmp_entitites) < 4 or ("*" in entities or "." in entities):
                try:
                    processed_entities.remove(entities)
                except Exception:
                    continue

        return processed_entities

    def limit_phrase_list(
        self,
        entities_dict,
        keyphrase_dict,
        phrase_limit=6,
        word_limit=3,
        keyphrase_object=None,
        remove_phrases=False,
    ):
        modified_entity_dict = {}
        modified_keyphrase_dict = {}
        if remove_phrases:
            for i, kp_dict in enumerate(keyphrase_object):
                entity_quality_score = kp_dict["entitiesQuality"]
                keyphrase_quality_score = kp_dict["medianKeyphraseQuality"]

                for entity, scores in entities_dict.items():
                    boosted_score = scores[2]
                    if boosted_score > entity_quality_score:
                        modified_entity_dict[entity] = scores

                for phrase, scores in keyphrase_dict.items():
                    segment_score = scores[1]
                    if segment_score > keyphrase_quality_score:
                        modified_keyphrase_dict[phrase] = scores

        else:
            modified_keyphrase_dict = keyphrase_dict
            modified_entity_dict = entities_dict

        if len(list(modified_entity_dict.keys())) >= phrase_limit:
            limited_keyphrase_dict = dict(
                itertools.islice(modified_keyphrase_dict.items(), word_limit)
            )
            # limited_keyphrase_list = keyphrase_list[:word_limit]
        else:
            num_of_entities = len(list(modified_entity_dict.keys()))
            difference = phrase_limit - num_of_entities
            limited_keyphrase_dict = dict(
                itertools.islice(modified_keyphrase_dict.items(), difference)
            )
            # limited_keyphrase_list = keyphrase_list[:difference]

        return modified_entity_dict, limited_keyphrase_dict
