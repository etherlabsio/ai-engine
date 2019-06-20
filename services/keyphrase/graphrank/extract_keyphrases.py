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
    def __init__(self, s3_client=None):
        self.s3_client = s3_client
        self.stop_words = list(STOP_WORDS)
        self.nlp = spacy.load("vendor/en_core_web_sm/en_core_web_sm-2.1.0")
        self.gr = GraphRank()
        self.tp = TextPreprocess()
        self.gutils = GraphUtils()
        self.graph_obj_dict = {}
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

    def get_graph_id(self, req_data):
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]
        graph_id = context_id + ":" + instance_id
        return graph_id

    def get_graph_instance_object(self, graph_id):
        if graph_id in list(self.graph_obj_dict.keys()):
            self.meeting_graph = self.graph_obj_dict.get(graph_id)
        else:
            logger.warning(
                "Graph object does not exist",
                extra={
                    "graphId": self.meeting_graph.graph.get("graphId"),
                    "graphServiceIdentifier": graph_id,
                },
            )

        if self.meeting_graph.graph.get("graphId") is None:
            logger.warning(
                "Null graphId",
                extra={
                    "graphId": self.meeting_graph.graph.get("graphId"),
                    "graphServiceIdentifier": graph_id,
                },
            )

    def initialize_meeting_graph(self, req_data):
        graph_id = self.get_graph_id(req_data=req_data)
        self.graph_obj_dict[graph_id] = nx.Graph(graphId=graph_id)

        # Set the current instance as the active one
        self.get_graph_instance_object(graph_id=graph_id)
        logger.info(
            "Meeting graph intialized and updated",
            extra={
                "currentGraphObjectList": list(self.graph_obj_dict.keys()),
                "currentGraphId": self.meeting_graph.graph.get("graphId"),
            },
        )

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
        graph,
        text_list,
        attrs,
        window=4,
        preserve_common_words=False,
        syntactic_filter=None,
        add_context=False,
    ):

        for i in range(len(text_list)):
            text = text_list[i]

            original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(text)
            graph = self.gr.build_word_graph(
                graph_obj=graph,
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
        original_pos_list=None,
        window=4,
        normalize_nodes_fn="degree",
        preserve_common_words=False,
        normalize_score=False,
        descriptive=False,
        post_process_descriptive=False,
    ):

        keyphrases = self.gr.get_keyphrases(
            graph_obj=graph,
            input_pos_text=pos_tuple,
            original_tokens=original_pos_list,
            window=window,
            normalize_nodes=normalize_nodes_fn,
            preserve_common_words=preserve_common_words,
            normalize_score=normalize_score,
            descriptive=descriptive,
            post_process_descriptive=post_process_descriptive,
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

        # remove stop words from noun_chunks/NERs
        entity_dict = []
        for entt in list(zip(filtered_entities, t_ner_type)):
            entity_dict.append({"text": str(entt[0]), "type": entt[1]})

        return filtered_entities

    def extract_pim_words(
        self,
        input_json,
        keyphrase_list,
        top_n=None,
        preserve_singlewords=False,
        limit_phrase=False,
    ):
        """
        Search for keyphrases in the top-5 PIM segments and return them for each segment
        Args:
            input_json:
            keyphrase_list:
            top_n:
            preserve_singlewords:
            limit_phrase(bool):

        Returns:

        """
        cleaned_keyphrase_list = []
        input_segment = input_json["segments"][0].get("originalText")

        for word, score in keyphrase_list:
            loc = input_segment.find(word)
            if loc > -1:
                cleaned_keyphrase_list.append((word, score))

        sorted_keyphrase_list = self.sort_by_value(cleaned_keyphrase_list, order="desc")
        if top_n is not None:
            sorted_keyphrase_list = sorted_keyphrase_list[:top_n]

        segment_entity = self.get_entities(input_segment)
        segment_keyword_list = [words for words, score in sorted_keyphrase_list]

        processed_entities, multiphrase_list = self.post_process_output(
            entity_list=segment_entity,
            keyphrase_list=segment_keyword_list,
            preserve_singlewords=preserve_singlewords,
        )

        if limit_phrase:
            processed_entities, multiphrase_list = self.limit_phrase_list(
                entities_list=processed_entities, keyphrase_list=multiphrase_list
            )

        processed_entities.extend(multiphrase_list)
        segment_output = processed_entities

        return segment_output

    def extract_chapter_words(
        self,
        input_json,
        keyphrase_list,
        top_n=None,
        preserve_singlewords=False,
        limit_phrase=True,
    ):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            limit_phrase:
            input_json:
            keyphrase_list:
            top_n:
            preserve_singlewords:

        Returns:

        """
        chapter_entities = []
        cleaned_keyphrase_list = []
        segments = input_json["segments"]
        for i in range(len(segments)):
            entity_segment = segments[i].get("originalText")
            input_segment = segments[i].get("originalText")

            # Get chapter entities
            entities = self.get_entities(entity_segment)
            chapter_entities.extend(entities)

            # Get cleaned chapter words
            for word, score in keyphrase_list:
                loc = input_segment.find(word)
                if loc > -1:
                    cleaned_keyphrase_list.append((word, score))

        sorted_keyphrase_list = self.sort_by_value(cleaned_keyphrase_list, order="desc")
        if top_n is not None:
            sorted_keyphrase_list = sorted_keyphrase_list[:top_n]

        chapter_keyphrases = [phrases for phrases, score in sorted_keyphrase_list]

        processed_entities, multiphrase_list = self.post_process_output(
            entity_list=chapter_entities,
            keyphrase_list=chapter_keyphrases,
            preserve_singlewords=preserve_singlewords,
        )

        if limit_phrase:
            processed_entities, multiphrase_list = self.limit_phrase_list(
                entities_list=processed_entities, keyphrase_list=multiphrase_list
            )

        processed_entities.extend(multiphrase_list)
        chapter_output = processed_entities

        return chapter_output

    def chapter_segment_offset_search(
        self,
        segments,
        relative_time,
        keyphrase_list,
        top_n=None,
        preserve_singlewords=False,
        limit_phrase=True,
    ):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            limit_phrase:
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
            input_segment = segments[i].get("originalText")

            # Set offset time for every keywords
            start_time = segments[i].get("startTime")
            start_time = self.formatTime(start_time, datetime_object=True)
            offset_time = float((start_time - relative_time).seconds)

            for kw, score in keyphrase_list:
                if kw in input_segment:
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
                    if keyphrase in entities:
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

        if limit_phrase:
            chapter_entities, chapter_multiphrase_list = self.limit_phrase_list(
                entities_list=chapter_entities, keyphrase_list=chapter_multiphrase_list
            )

        chapter_entities.extend(chapter_multiphrase_list)

        # Reformat as per api contract
        chapter_output = {
            words: offset
            for elements in chapter_entities
            for words, offset in elements.items()
        }

        result = {"keyphrases": chapter_output}

        return result

    def post_process_output(
        self, entity_list, keyphrase_list, preserve_singlewords=False
    ):

        # Get distinct entities and keyphrases
        distinct_entities = list(dict.fromkeys(entity_list))
        distinct_keyword_list = list(dict.fromkeys(keyphrase_list))

        # Post-process entities
        distinct_entities = self.post_process_entities(distinct_entities)

        # Remove the first occurrence of entity in the list of keyphrases
        for entities in distinct_entities:
            for keyphrase in distinct_keyword_list:
                if keyphrase in entities:
                    distinct_keyword_list.remove(keyphrase)

        # Place the single keywords in the end of the list.
        multiphrase_list = [
            words for words in distinct_keyword_list if len(words.split()) > 1
        ]
        singleword_list = [
            words for words in distinct_keyword_list if len(words.split()) == 1
        ]

        if preserve_singlewords:
            multiphrase_list.extend(singleword_list)

        return distinct_entities, multiphrase_list

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
                    except:
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
                except:
                    continue

        return processed_entities

    def populate_word_graph(self, req_data, add_context=False):
        start = timer()
        segment_df = self.reformat_input(req_data)
        graph_id = self.get_graph_id(req_data=req_data)

        self.get_graph_instance_object(graph_id=graph_id)

        try:
            text_list, attrs = self.read_segments(
                segment_df=segment_df, node_attrs=False
            )
            self.build_custom_graph(
                text_list=text_list,
                attrs=attrs,
                add_context=add_context,
                graph=self.meeting_graph,
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
                "graphId": self.meeting_graph.graph.get("graphId"),
                "graphServiceIdentifier": graph_id,
                "nodes": self.meeting_graph.number_of_nodes(),
                "edges": self.meeting_graph.number_of_edges(),
                "instanceId": req_data["instanceId"],
                "responseTime": end - start,
            },
        )

    def compute_keyphrases(self, req_data):
        # Select the right graph Id to extract keyphrases
        graph_id = self.get_graph_id(req_data=req_data)
        self.get_graph_instance_object(graph_id=graph_id)

        segment_df = self.reformat_input(req_data)
        text_list, attrs = self.read_segments(segment_df=segment_df, node_attrs=False)
        keyphrase_list = []
        descriptive_keyphrase_list = []
        try:
            for i in range(len(text_list)):
                text = text_list[i]

                original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(text)

                keyphrase_list.extend(
                    self.get_custom_keyphrases(
                        graph=self.meeting_graph, pos_tuple=pos_tuple
                    )
                )
                descriptive_keyphrase_list.extend(
                    self.get_custom_keyphrases(
                        graph=self.meeting_graph,
                        pos_tuple=pos_tuple,
                        descriptive=True,
                        post_process_descriptive=True,
                    )
                )
        except Exception:
            logger.error(
                "error retrieving descriptive phrases",
                extra={"err": traceback.print_exc()},
            )

        return keyphrase_list, descriptive_keyphrase_list

    def limit_phrase_list(
        self, entities_list, keyphrase_list, phrase_limit=6, word_limit=3
    ):

        if len(entities_list) >= phrase_limit:
            keyphrase_list = keyphrase_list[:word_limit]
        else:
            num_of_entities = len(entities_list)
            difference = phrase_limit - num_of_entities
            keyphrase_list = keyphrase_list[:difference]

        return entities_list, keyphrase_list

    def _get_pim_keyphrases(self, req_data, n_kw=10, default_form="original"):
        start = timer()
        keyphrase_list, descriptive_kp = self.compute_keyphrases(req_data=req_data)

        segment_keyphrases = self.extract_pim_words(
            input_json=req_data, keyphrase_list=keyphrase_list, top_n=n_kw
        )

        segment_desc_keyphrases = self.extract_pim_words(
            input_json=req_data, keyphrase_list=descriptive_kp, top_n=n_kw
        )

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

        if default_form == "descriptive" and len(segment_desc_keyphrases) > 0:
            keyphrase = segment_desc_keyphrases

        elif (
            default_form == "original"
            and len(segment_keyphrases) < 5
            and len(segment_desc_keyphrases) > 0
        ):
            keyphrase = segment_desc_keyphrases

        else:
            keyphrase = segment_keyphrases

        return keyphrase

    def _get_chapter_keyphrases(self, req_data, n_kw=10, default_form="original"):
        start = timer()
        keyphrase_list, descriptive_kp = self.compute_keyphrases(req_data=req_data)

        chapter_keyphrases = self.extract_chapter_words(
            input_json=req_data, keyphrase_list=keyphrase_list, top_n=n_kw
        )

        chapter_desc_keyphrases = self.extract_chapter_words(
            input_json=req_data, keyphrase_list=descriptive_kp, top_n=n_kw
        )

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

        if default_form == "descriptive" and len(chapter_desc_keyphrases) > 0:
            keyphrase = chapter_desc_keyphrases

        elif (
            default_form == "original"
            and len(chapter_keyphrases) < 5
            and len(chapter_desc_keyphrases) > 0
        ):
            keyphrase = chapter_desc_keyphrases

        else:
            keyphrase = chapter_keyphrases

        return keyphrase

    def get_keyphrases(self, req_data, n_kw=10):
        start = timer()
        segments_array = req_data["segments"]

        # Re-populate graph in case google transcripts are present
        self.populate_word_graph(req_data, add_context=False)

        keyphrases = []
        try:
            # Decide between PIM or Chapter keyphrases
            if len(segments_array) > 1:
                keyphrases = self._get_chapter_keyphrases(
                    req_data, n_kw=n_kw, default_form="descriptive"
                )
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

        result = {"keyphrases": keyphrases}

        return result

    def get_instance_keyphrases(self, req_data, n_kw=10):
        keyphrase_list, descriptive_kp = self.compute_keyphrases(req_data)
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

        # Re-populate graph for google transcripts
        self.populate_word_graph(req_data, add_context=False)

        keyphrase_list, descriptive_kp = self.compute_keyphrases(req_data)

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

        except Exception as e:
            end = timer()
            logger.error(
                "Error processing chapter keyphrases with offset",
                extra={
                    "err": traceback.print_exc(),
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
        graph_id = self.get_graph_id(req_data=req_data)
        self.get_graph_instance_object(graph_id=graph_id)

        deleted_graph = self.graph_obj_dict.pop(graph_id)
        deleted_graph_id = deleted_graph.graph.get("graphId")

        # Upload the graph object to s3 before resetting it
        logger.info("starting upload session")
        success = self.upload_s3(graph_obj=deleted_graph)
        if success:
            logger.info("Upload completed")

        self.gr.reset_graph()
        self.meeting_graph.clear()
        end = timer()
        logger.info(
            "Post-reset: Graph info",
            extra={
                "deletedGraphId": deleted_graph_id,
                "currentGraphObjectList": list(self.graph_obj_dict.keys()),
                "numOfGraphInstances": len(self.graph_obj_dict),
                "nodes": self.meeting_graph.number_of_nodes(),
                "edges": self.meeting_graph.number_of_edges(),
                "responseTime": end - start,
            },
        )

        return {"result": "done", "message": "reset successful"}

    # S3 storage utility functions

    def upload_s3(self, graph_obj):
        graph_id = graph_obj.graph.get("graphId")
        context_id = graph_id.split(":")[0]
        instance_id = graph_id.split(":")[1]

        if graph_id == context_id + ":" + instance_id:
            serialized_graph_string = self.gutils.write_to_pickle(graph_obj=graph_obj)
            s3_key = str(context_id) + "/keyphrase-graphs/" + graph_id

            self.s3_client.upload_object(body=serialized_graph_string, s3_key=s3_key)
            logger.info(
                "Uploaded graph object to s3",
                extra={"graphId": graph_obj.graph.get("graphId"), "fileName": s3_key},
            )
            return True
        else:
            logger.error(
                "graphId and context info not matching",
                extra={
                    "graphId": graph_id,
                    "contextInfo": context_id + ":" + instance_id,
                },
            )
            return False

    def download_s3(self, context_id, instance_id):
        graph_id = context_id + ":" + instance_id
        s3_path = str(context_id) + "/keyphrase-graphs/" + graph_id

        file_obj = self.s3_client.download_file(file_name=s3_path)
        file_obj_bytestring = file_obj["Body"].read()

        graph_obj = self.gutils.load_graph_from_pickle(byte_string=file_obj_bytestring)

        logger.info(
            "Downloaded graph object from s3",
            extra={
                "graphId": graph_obj.graph.get("graphId"),
                "nodes": graph_obj.number_of_nodes(),
                "edges": graph_obj.number_of_edges(),
            },
        )

        return graph_obj
