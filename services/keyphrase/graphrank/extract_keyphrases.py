import pandas as pd
from datetime import datetime
import iso8601
import networkx as nx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import logging
from timeit import default_timer as timer
import traceback

from .graph_rank import GraphRank
from .utils import TextPreprocess, GraphUtils

logger = logging.getLogger(__name__)


class KeyphraseExtractor(object):
    def __init__(self):
        self.stop_words = list(STOP_WORDS)
        self.nlp = spacy.load("vendor/en_core_web_sm/en_core_web_sm-2.1.0")
        self.gr = GraphRank()
        self.tp = TextPreprocess()
        self.gutils = GraphUtils()
        self.meeting_graph = nx.Graph()
        self.syntactic_filter = [
            "JJ",
            "JJR",
            "JJS",
            "NN",
            "NNP",
            "NNS",
            "VB",
            "VBP",
            "NNPS",
            "FW",
        ]

    def formatTime(self, tz_time, datetime_object=False):
        isoTime = iso8601.parse_date(tz_time)
        ts = isoTime.timestamp()
        ts = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S:%f")

        if datetime_object:
            ts = datetime.fromisoformat(ts)
        return ts

    def reformat_input(self, input_json):

        json_df_ts = pd.DataFrame(input_json["segments"], index=None)
        json_df_ts["id"] = json_df_ts["id"].astype(str)
        json_df_ts["originalText"] = json_df_ts["originalText"].apply(lambda x: str(x))
        json_df_ts["createdAt"] = json_df_ts["createdAt"].apply(
            lambda x: self.formatTime(x)
        )
        json_df_ts["endTime"] = json_df_ts["endTime"].apply(
            lambda x: self.formatTime(x)
        )
        json_df_ts["startTime"] = json_df_ts["startTime"].apply(
            lambda x: self.formatTime(x)
        )
        json_df_ts["updatedAt"] = json_df_ts["updatedAt"].apply(
            lambda x: self.formatTime(x)
        )
        json_df_ts = json_df_ts.sort_values(["createdAt"], ascending=[1])

        return json_df_ts

    def sort_by_value(self, item_list, order="desc"):
        """
        A utility function to sort lists by their value.
        Args:
            item_list:
            order:

        Returns:

        """

        if order == "desc":
            sorted_list = sorted(item_list, key=lambda x: x[1], reverse=True)
        else:
            sorted_list = sorted(item_list, key=lambda x: x[1], reverse=False)

        return sorted_list

    def initialize_meeting_graph(self, context_id, instance_id):
        self.meeting_graph = nx.Graph(
            context_instance_id=instance_id, context_id=context_id
        )

    def read_segments(self, segment_df, node_attrs=False):
        text_list = []
        attrs = {
            "spokenBy": [],
            "id": [],
            "createdAt": [],
            "recordingId": [],
            "transcriber": [],
        }

        for i in range(len(segment_df)):
            segment_text = segment_df.iloc[i]["originalText"]
            text_list.append(segment_text)

            if node_attrs:
                attrs["spokenBy"].append(segment_df.iloc[i].get("spokenBy"))
                attrs["id"].append(segment_df.iloc[i].get("id"))
                attrs["createdAt"].append(segment_df.iloc[i].get("createdAt"))
                attrs["recordingId"].append(segment_df.iloc[i].get("recordingId"))
                attrs["transcriber"].append(segment_df.iloc[i].get("transcriber"))
            else:
                attrs = None
        return text_list, attrs

    def process_text(
        self, text, filter_by_pos=True, stop_words=False, syntactic_filter=None
    ):
        original_tokens, pos_tuple, filtered_pos_tuple = self.tp.preprocess_text(
            text,
            filter_by_pos=filter_by_pos,
            pos_filter=syntactic_filter,
            stop_words=stop_words,
        )

        return original_tokens, pos_tuple, filtered_pos_tuple

    def build_custom_graph(
        self,
        text_list,
        attrs,
        window=4,
        preserve_common_words=False,
        syntactic_filter=None,
        add_context=True,
    ):

        for i in range(len(text_list)):
            text = text_list[i]

            original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(text)
            self.meeting_graph = self.gr.build_word_graph(
                input_pos_text=pos_tuple,
                window=window,
                syntactic_filter=syntactic_filter,
                preserve_common_words=preserve_common_words,
                node_attributes=attrs,
                add_context=add_context,
            )

    def get_custom_keyphrases(
        self,
        graph,
        pos_tuple=None,
        token_list=None,
        window=4,
        normalize_nodes_fn="degree",
        preserve_common_words=False,
        normalize_score=False,
        descriptive=False,
    ):

        keyphrases = self.gr.get_keyphrases(
            graph_obj=graph,
            input_pos_text=pos_tuple,
            original_tokens=token_list,
            window=window,
            normalize_nodes=normalize_nodes_fn,
            preserve_common_words=preserve_common_words,
            normalize_score=normalize_score,
            descriptive=descriptive,
        )

        return keyphrases

    def get_entities(self, input_segment):
        spacy_list = ["PRODUCT", "EVENT", "LOC", "ORG", "PERSON", "WORK_OF_ART"]
        comprehend_list = [
            "COMMERCIAL_ITEM",
            "EVENT",
            "LOCATION",
            "ORGANIZATION",
            "PERSON",
            "TITLE",
        ]
        match_dict = dict(zip(spacy_list, comprehend_list))

        doc = self.nlp(input_segment)
        t_noun_chunks = list(set(list(doc.noun_chunks)))
        filtered_entities = []
        t_ner_type = []

        for ent in list(set(doc.ents)):
            t_type = ent.label_
            if t_type in list(match_dict) and str(ent) not in filtered_entities:
                t_ner_type.append(match_dict[t_type])
                filtered_entities.append(str(ent))
        t_noun_chunks = [str(item).strip().lower() for item in t_noun_chunks]

        # Post-process entities
        filtered_entities = self.post_process_entities(filtered_entities)

        # remove stop words from noun_chunks/NERs
        entity_dict = []
        for entt in list(zip(filtered_entities, t_ner_type)):
            entity_dict.append({"text": str(entt[0]), "type": entt[1]})

        logger.debug("Processing entities", extra={"entities": entity_dict})

        return filtered_entities

    def segment_search(
        self, input_json, keyphrase_list, top_n=None, preserve_singlewords=False
    ):
        """
        Search for keyphrases in the top-5 PIM segments and return them for each segment
        Args:
            input_json:
            keyphrase_list:
            top_n:
            preserve_singlewords:

        Returns:

        """
        result = {}
        for i in range(len(input_json["segments"])):
            sort_list = []

            entity_segment = input_json["segments"][i].get("originalText")
            input_segment = input_json["segments"][i].get("originalText").lower()
            keywords_list = []

            for tup in keyphrase_list:
                kw = tup[0]
                score = tup[1]

                result = input_segment.find(kw)
                if result > -1:
                    keywords_list.append((kw, score))

            sort_list = self.gutils.sort_by_value(keywords_list, order="desc")
            if top_n is not None:
                sort_list = sort_list[:top_n]

            segment_entity = self.get_entities(entity_segment)
            segment_keyword_list = [words for words, score in sort_list]

            # Get distinct entities and keyphrases
            segment_entity = list(dict.fromkeys(segment_entity))
            segment_keyword_list = list(dict.fromkeys(segment_keyword_list))

            # Post-process entities
            segment_entity = self.post_process_entities(segment_entity)

            # Remove the first occurrence of entity in the list of keyphrases
            for entities in segment_entity:
                for keyphrase in segment_keyword_list:
                    if keyphrase in entities.lower():
                        segment_keyword_list.remove(keyphrase)

            # Place the single keywords in the end of the list.
            segment_multiphrase_list = [
                words for words in segment_keyword_list if len(words.split()) > 1
            ]
            segment_singleword_list = [
                words for words in segment_keyword_list if len(words.split()) == 1
            ]

            if preserve_singlewords:
                segment_multiphrase_list.extend(segment_singleword_list)

            segment_entity.extend(segment_multiphrase_list)
            segment_output = segment_entity

            result = {"keyphrases": segment_output}

        return result

    def chapter_segment_search(
        self, input_json, keyphrase_list, top_n=None, preserve_singlewords=False
    ):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            input_json:
            keyphrase_list:
            top_n:
            preserve_singlewords:

        Returns:

        """
        chapter_keywords_list = []
        chapter_entities = []
        for i in range(len(input_json["segments"])):
            entity_segment = input_json["segments"][i].get("originalText")
            input_segment = input_json["segments"][i].get("originalText").lower()

            for tup in keyphrase_list:
                kw = tup[0]
                score = tup[1]

                result = input_segment.find(kw)
                if result > -1:
                    chapter_keywords_list.append((kw, score))

            entities = self.get_entities(entity_segment)
            chapter_entities.extend(entities)

        sort_list = self.sort_by_value(chapter_keywords_list, order="desc")
        if top_n is not None:
            sort_list = sort_list[:top_n]

        chapter_keyphrases = [phrases for phrases, score in sort_list]

        # Get distinct entities and keyphrases
        chapter_entities = list(dict.fromkeys(chapter_entities))
        chapter_keyphrases = list(dict.fromkeys(chapter_keyphrases))

        # Post-process entities
        chapter_entities = self.post_process_entities(chapter_entities)

        # Remove the first occurrence of entity in the list of keyphrases
        for entities in chapter_entities:
            for keyphrase in chapter_keyphrases:
                if keyphrase in entities.lower():
                    chapter_keyphrases.remove(keyphrase)

        # Place the single keywords in the end of the list.
        chapter_multiphrase_list = [
            words for words in chapter_keyphrases if len(words.split()) > 1
        ]
        chapter_singleword_list = [
            words for words in chapter_keyphrases if len(words.split()) == 1
        ]

        if preserve_singlewords:
            chapter_multiphrase_list.extend(chapter_singleword_list)

        chapter_entities.extend(chapter_multiphrase_list)
        chapter_output = chapter_entities

        result = {"keyphrases": chapter_output}

        return result

    def chapter_segment_offset_search(
        self,
        segments,
        relative_time,
        keyphrase_list,
        top_n=None,
        preserve_singlewords=False,
    ):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            segments:
            top_n:
            preserve_singlewords:
            relative_time:
            keyphrase_list:

        Returns:

        """
        chapter_entities = []
        chapter_keywords_list = []

        for i in range(len(segments)):
            entity_segment = segments[i].get("originalText")
            input_segment = segments[i].get("originalText").lower()

            # Set offset time for every keywords
            start_time = segments[i].get("startTime")
            start_time = self.formatTime(start_time, datetime_object=True)
            offset_time = float((start_time - relative_time).seconds)
            logger.debug("Checking offset", extra={"diffSec": offset_time})

            for tup in keyphrase_list:
                kw = tup[0]
                score = tup[1]

                result = input_segment.find(kw)
                if result > -1:
                    chapter_keywords_list.append(({kw: offset_time}, score))

            entities = self.get_entities(entity_segment)
            chapter_entities.extend([{entity: offset_time} for entity in entities])

        chapter_entities_list = [
            entities for ent in chapter_entities for entities, off in ent.items()
        ]

        sort_list = self.sort_by_value(chapter_keywords_list, order="desc")
        if top_n is not None:
            sort_list = sort_list[:top_n]

        chapter_keyphrases = [phrases for phrases, score in sort_list]

        # Get distinct entities and keyphrases
        chapter_entities_list = list(dict.fromkeys(chapter_entities_list))

        # Post-process entities
        chapter_entities_list = self.post_process_entities(chapter_entities_list)

        # Remove the first occurrence of entity in the list of keyphrases
        for entities in chapter_entities_list:
            for i, keyphrase_dict in enumerate(chapter_keyphrases):
                for keyphrase in keyphrase_dict.keys():
                    if keyphrase in entities.lower():
                        del chapter_keyphrases[i]

        # Place the single keywords in the end of the list.
        chapter_multiphrase_list = [
            {kw: offset}
            for words in chapter_keyphrases
            for kw, offset in words.items()
            if len(kw.split()) > 1
        ]
        chapter_singleword_list = [
            {kw: offset}
            for words in chapter_keyphrases
            for kw, offset in words.items()
            if len(kw.split()) == 1
        ]

        if preserve_singlewords:
            chapter_multiphrase_list.extend(chapter_singleword_list)

        chapter_entities.extend(chapter_multiphrase_list)

        # Reformat as per contract
        chapter_output = {
            words: offset
            for elements in chapter_entities
            for words, offset in elements.items()
        }

        result = {"keyphrases": chapter_output}

        return result

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
                    except Exception as e:
                        logger.warning("entity not found", extra={"warning": e})
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
            if len(tmp_entitites) < 4:
                try:
                    processed_entities.remove(entities)
                except Exception as e:
                    logger.debug("entity not found", extra={"warning": e})
                    continue

        return processed_entities

    def post_process_desc(self, keyphrases):

        # Join 2 similar sentences
        processed_keyphrase = []
        for index1, words in enumerate(keyphrases):
            for index2, words2 in enumerate(keyphrases):
                if index1 != index2:
                    word_set = set(list(dict.fromkeys(words.split(" "))))
                    word_set2 = set(list(dict.fromkeys(words2.split(" "))))
                    if len(word_set & word_set2) > 2:
                        new_set = word_set & word_set2

                        # for w in list(new_set)[:1]:
                        w = list(new_set)[0]
                        print(w)
                        word_index1 = words.split(" ").index(w)
                        word_index2 = words2.split(" ").index(w)
                        print(word_index1, word_index2)
                        if word_index1 > word_index2:
                            word3 = words.split(" ") + words2.split(" ")
                            word4 = " ".join(list(dict.fromkeys(word3)))
                            processed_keyphrase.append(word4)

        print(processed_keyphrase)

        if len(processed_keyphrase) == 0:
            processed_keyphrase = keyphrases

        return processed_keyphrase

    def populate_word_graph(self, req_data, add_context=True):
        start = timer()
        segment_df = self.reformat_input(req_data)

        try:
            text_list, attrs = self.read_segments(
                segment_df=segment_df, node_attrs=False
            )
            self.build_custom_graph(
                text_list=text_list, attrs=attrs, add_context=add_context
            )
        except Exception as e:
            end = timer()
            logger.error(
                "Error populating graph",
                extra={
                    "err": e,
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                },
            )

        end = timer()

        logger.info(
            "Populating graph",
            extra={
                "nodes": self.meeting_graph.number_of_nodes(),
                "edges": self.meeting_graph.number_of_edges(),
                "instanceId": req_data["instanceId"],
                "responseTime": end - start,
            },
        )

    def compute_keyphrases(self):
        keyphrase_list = self.get_custom_keyphrases(graph=self.meeting_graph)

        descriptive_keyphrase_list = []
        try:
            descriptive_keyphrase_list = self.get_custom_keyphrases(
                graph=self.meeting_graph, descriptive=True
            )
        except Exception:
            logger.debug(
                "error retrieving descriptive phrases",
                extra={"err": traceback.print_exc()},
            )

        return keyphrase_list, descriptive_keyphrase_list

    def _get_pim_keyphrases(self, req_data, n_kw=10, default_form="original"):
        start = timer()
        keyphrase_list, descriptive_kp = self.compute_keyphrases()

        segment_keyphrases = self.segment_search(
            input_json=req_data, keyphrase_list=keyphrase_list, top_n=n_kw
        )

        segment_desc_keyphrases = self.segment_search(
            input_json=req_data, keyphrase_list=descriptive_kp, top_n=n_kw
        )

        processed_descriptive_words = self.post_process_desc(
            segment_desc_keyphrases["keyphrases"]
        )
        segment_desc_keyphrases["keyphrases"] = processed_descriptive_words

        end = timer()
        logger.debug(
            "Comparing keyphrase output",
            extra={
                "responseTime": end - start,
                "instanceId": req_data["instanceId"],
                "segmentObj": req_data["segments"],
                "originalKeyphrase": segment_keyphrases,
                "descriptiveKeyphrase": segment_desc_keyphrases,
            },
        )

        if (
            default_form == "descriptive"
            and len(segment_desc_keyphrases["keyphrases"]) > 0
        ):
            keyphrase = segment_desc_keyphrases

        elif (
            default_form == "original"
            and len(segment_keyphrases["keyphrases"]) < 5
            and len(segment_desc_keyphrases["keyphrases"]) > 0
        ):
            keyphrase = segment_desc_keyphrases

        else:
            keyphrase = segment_keyphrases

        return keyphrase

    def _get_chapter_keyphrases(self, req_data, n_kw=10, default_form="original"):
        start = timer()
        keyphrase_list, descriptive_kp = self.compute_keyphrases()

        chapter_keyphrases = self.chapter_segment_search(
            input_json=req_data, keyphrase_list=keyphrase_list, top_n=n_kw
        )

        chapter_desc_keyphrases = self.chapter_segment_search(
            input_json=req_data, keyphrase_list=descriptive_kp, top_n=n_kw
        )

        processed_descriptive_words = self.post_process_desc(
            chapter_desc_keyphrases["keyphrases"]
        )
        chapter_desc_keyphrases["keyphrases"] = processed_descriptive_words

        end = timer()
        logger.debug(
            "Comparing keyphrase output",
            extra={
                "responseTime": end - start,
                "instanceId": req_data["instanceId"],
                "segmentObj": req_data["segments"],
                "originalKeyphrase": chapter_keyphrases,
                "descriptiveKeyphrase": chapter_desc_keyphrases,
            },
        )

        if (
            default_form == "descriptive"
            and len(chapter_desc_keyphrases["keyphrases"]) > 0
        ):
            keyphrase = chapter_desc_keyphrases

        elif (
            default_form == "original"
            and len(chapter_keyphrases["keyphrases"]) < 5
            and len(chapter_desc_keyphrases["keyphrases"]) > 0
        ):
            keyphrase = chapter_desc_keyphrases

        else:
            keyphrase = chapter_keyphrases

        return keyphrase

    def get_keyphrases(self, req_data, n_kw=10, words_limit=6):
        start = timer()
        segments_array = req_data["segments"]

        logger.info("Re-populating word graph on google segments")
        if segments_array[0].get("transcriber") == "google_speech_api":
            self.populate_word_graph(req_data, add_context=True)
        else:
            self.populate_word_graph(req_data, add_context=True)

        keyphrases = []
        try:
            # Decide between PIM or Chapter keyphrases
            if len(segments_array) > 1:
                keyphrases = self._get_chapter_keyphrases(
                    req_data, n_kw=n_kw, default_form="descriptive"
                )
                if words_limit:
                    keyphrases["keyphrases"] = keyphrases["keyphrases"][:words_limit]
            else:
                keyphrases = self._get_pim_keyphrases(
                    req_data, n_kw=n_kw, default_form="descriptive"
                )
        except Exception as e:
            end = timer()
            logger.error(
                "Error extracting keyphrases from segment",
                extra={
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                    "segmentObj": req_data["segments"],
                    "err": traceback.print_exc(),
                },
            )

        return keyphrases

    def get_instance_keyphrases(self, req_data, n_kw=10):
        keyphrase_list, descriptive_kp = self.compute_keyphrases()
        instance_keyphrases = [words for words, score in keyphrase_list]

        result = {"keyphrases": instance_keyphrases[:n_kw]}

        return result

    def get_chapter_offset_keyphrases(
        self, req_data, n_kw=10, default_form="descriptive"
    ):
        start = timer()

        logger.info(
            "Re-populating word graph on google segments for chapter offset phrases"
        )
        if req_data["segments"][0].get("transcriber") == "google_speech_api":
            self.populate_word_graph(req_data, add_context=True)
        else:
            self.populate_word_graph(req_data, add_context=False)

        keyphrase_list, descriptive_kp = self.compute_keyphrases()

        relative_time = self.formatTime(req_data["relativeTime"], datetime_object=True)
        chapter_keyphrases = []
        chapter_desc_keyphrases = []

        try:
            chapter_keyphrases = self.chapter_segment_offset_search(
                segments=req_data["segments"],
                relative_time=relative_time,
                keyphrase_list=keyphrase_list,
                top_n=n_kw,
            )

            chapter_desc_keyphrases = self.chapter_segment_offset_search(
                segments=req_data["segments"],
                relative_time=relative_time,
                keyphrase_list=descriptive_kp,
                top_n=n_kw,
            )

            # processed_descriptive_words = self.post_process_desc(chapter_desc_keyphrases["keyphrases"])
            # chapter_desc_keyphrases["keyphrases"] = processed_descriptive_words

        except Exception as e:
            end = timer()
            logger.error(
                "Error processing chapter keyphrases with offset",
                extra={
                    "err": e,
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                },
            )

        if (
            default_form == "descriptive"
            and len(chapter_desc_keyphrases["keyphrases"]) > 0
        ):
            keyphrase = chapter_desc_keyphrases

        elif (
            default_form == "original"
            and len(chapter_keyphrases["keyphrases"]) < 5
            and len(chapter_desc_keyphrases["keyphrases"]) > 0
        ):
            keyphrase = chapter_desc_keyphrases

        else:
            keyphrase = chapter_keyphrases

        return keyphrase

    def reset_keyphrase_graph(self, req_data):
        start = timer()

        self.gr.reset_graph()
        self.meeting_graph.clear()
        end = timer()
        logger.info(
            "Post-reset: Graph info",
            extra={
                "nodes": self.meeting_graph.number_of_nodes(),
                "edges": self.meeting_graph.number_of_edges(),
                "responseTime": end - start,
            },
        )

        return {"result": "done", "message": "reset successful"}
