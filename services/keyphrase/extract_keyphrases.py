import networkx as nx
import logging
from timeit import default_timer as timer
import traceback
from typing import List, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from fuzzywuzzy import fuzz, process
import json

from graphrank.core import GraphRank
from graphrank.utils import TextPreprocess, GraphUtils

from .utils import KeyphraseUtils
from .ranker import KeyphraseRanker
from .s3io import S3IO
from .word_graph import WordGraphBuilder
from .queries import Queries

logger = logging.getLogger(__name__)

SegmentType = Dict


class KeyphraseExtractor(object):
    def __init__(
        self,
        s3_client=None,
        encoder_lambda_client=None,
        lambda_function=None,
        ner_lambda_function=None,
        nats_manager=None,
    ):
        self.context_dir = "/context-instance-graphs/"
        self.feature_dir = "/sessions/"
        self.s3_client = s3_client
        self.utils = KeyphraseUtils()
        self.io_util = S3IO(
            s3_client=s3_client, graph_utils_obj=GraphUtils(), utils=KeyphraseUtils(),
        )
        self.ranker = KeyphraseRanker(
            encoder_lambda_client=encoder_lambda_client,
            lambda_function=lambda_function,
            s3_io_util=self.io_util,
            context_dir=self.context_dir,
            utils=KeyphraseUtils(),
        )

        self.wg = WordGraphBuilder(
            graphrank_obj=GraphRank(),
            textpreprocess_obj=TextPreprocess(),
            graphutils_obj=GraphUtils(),
            keyphrase_utils_obj=self.utils,
            lambda_client=encoder_lambda_client,
            ner_lambda_function=ner_lambda_function,
        )

        self.query_client = Queries(nats_manager=nats_manager)

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
        self.meeting_keywords = []

    def get_graph_id(self, req_data):
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]
        graph_id = context_id + ":" + instance_id
        return graph_id

    def wake_up_lambda(self, req_data):
        # Start the encoder lambda to avoid cold start problem
        logger.info("Invoking lambda to reduce cold-start ...")
        test_segment = ["Wake up Sesame!"]
        self.ranker.get_embeddings(input_list=test_segment, req_data=req_data)
        self.wg.call_custom_ner(input_segment=test_segment[0])

    def initialize_meeting_graph(self, req_data: dict):
        graph_id = self.get_graph_id(req_data=req_data)
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        meeting_word_graph = nx.Graph(graphId=graph_id)

        logger.info(
            "Meeting word graph intialized",
            extra={"currentGraphId": meeting_word_graph.graph.get("graphId")},
        )
        logger.info("Uploading serialized graph object")
        self.io_util.upload_s3(
            graph_obj=meeting_word_graph,
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
        )

        # Start the encoder lambda to avoid cold start problem
        self.wake_up_lambda(req_data=req_data)

    def _retrieve_word_graph(self, req_data: SegmentType) -> nx.Graph:
        """
        Download meeting word graph from s3
        Args:
            req_data:

        Returns:

        """

        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        # Get graph object from S3
        meeting_word_graph = self.io_util.download_s3(
            context_id=context_id, instance_id=instance_id, s3_dir=self.context_dir
        )

        return meeting_word_graph

    def _update_word_graph(
        self, req_data: SegmentType, meeting_word_graph: nx.Graph, **kwargs
    ) -> nx.Graph:
        """
        Populate instance information, add meeting word graph to context graph and upload the context graph
        Args:
            req_data:
            context_graph:
            meeting_word_graph:

        Returns:

        """

        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        # Write back the graph object to S3
        self.io_util.upload_s3(
            graph_obj=meeting_word_graph,
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
        )

        return meeting_word_graph

    def populate_word_graph(self, req_data):
        start = timer()
        # Get graph objects
        meeting_word_graph = self._retrieve_word_graph(req_data=req_data)
        segment_object = req_data["segments"]

        # Populate word graph for the current instance
        try:
            text_list = self.utils.read_segments(segment_object=segment_object)
            meeting_word_graph = self.wg.build_custom_graph(
                text_list=text_list, graph=meeting_word_graph
            )

            end = timer()
            logger.info(
                "Populated graph and written to s3",
                extra={
                    "graphId": meeting_word_graph.graph.get("graphId"),
                    "nodes": meeting_word_graph.number_of_nodes(),
                    "edges": meeting_word_graph.number_of_edges(),
                    "instanceId": req_data["instanceId"],
                    "responseTime": end - start,
                },
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

        # Push updated meeting graph
        meeting_word_graph = self._update_word_graph(
            req_data=req_data, meeting_word_graph=meeting_word_graph
        )

        return meeting_word_graph

    def compute_embeddings(
        self,
        req_data,
        segment_object,
        meeting_word_graph=None,
        default_form="descriptive",
        **kwargs,
    ):
        """
        Compute embedding vectors for segments and segment-keyphrases and store them as node attributes in the knowledge
        graph.
        Args:
            default_form:
            segment_object:
            meeting_word_graph:
            req_data:

        Returns:

        """
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        populate_graph = kwargs.get("populate_graph", True)
        group_id = kwargs.get("group_id", "")
        keyphrase_attr = kwargs.get("keyphrase_attr")

        if meeting_word_graph is None:
            # Get graph objects
            meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

        keyphrase_object = self.extract_keywords(
            segment_object=segment_object,
            meeting_word_graph=meeting_word_graph,
            default_form=default_form,
        )

        # Get segment text
        for i, kp_dict in enumerate(keyphrase_object):
            keyphrase_attr_dict = deepcopy(keyphrase_attr)
            seg_text = kp_dict["segments"]
            segment_id = kp_dict["segmentId"]
            segment_keyphrase_dict = kp_dict[default_form]
            segment_entity_dict = kp_dict["entities"]

            keyphrase_list = list(segment_keyphrase_dict.keys())
            entities_list = list(segment_entity_dict.keys())

            input_phrases_list = entities_list
            input_phrases_list.extend(keyphrase_list)

            # Compute segment embedding vector
            segment_embedding = self.ranker.get_embeddings(
                input_list=[seg_text], req_data=req_data
            )

            # Compute keyphrase embedding vectors
            keyphrase_embeddings = self.ranker.get_embeddings(
                input_list=input_phrases_list, req_data=req_data
            )

            # Combine segment and keyphrase embeddings and serialize them
            f_name = segment_id
            if populate_graph is not True:
                f_name = segment_id + "_" + group_id

            segment_embedding_dict = {f_name: np.array(segment_embedding)}

            (
                phrase_hash_dict,
                phrase_embedding_dict,
            ) = self.utils.map_embeddings_to_phrase(
                phrase_list=input_phrases_list, embedding_list=keyphrase_embeddings,
            )

            segment_keyphrase_embeddings = {
                **segment_embedding_dict,
                **phrase_embedding_dict,
            }

            # Serialize the entire segment-keyphrase embedding dictionary to NPZ
            npz_file_name = self.utils.serialize_to_npz(
                embedding_dict=segment_keyphrase_embeddings, file_name=f_name
            )
            npz_s3_path = self.io_util.upload_npz(
                context_id=context_id,
                instance_id=instance_id,
                feature_dir=self.feature_dir,
                npz_file_name=npz_file_name,
            )

            # Update context graph with embedding vectors
            segment_attr_dict = {
                "text": segment_object[i]["originalText"],
                "embedding_vector_uri": npz_s3_path,
                "embedding_model": "use_v1",
            }
            if not populate_graph:
                segment_attr_dict = {
                    "analyzedText": segment_object[i]["originalText"],
                    "embedding_vector_group_uri": npz_s3_path,
                    "groupId": group_id,
                    "embedding_model": "use_v1",
                }

            attributed_segment_obj = self._form_segment_attr_object(
                segment=segment_object[i], segment_attr=segment_attr_dict
            )

            attributed_segment_obj = self._form_keyphrase_attr_object(
                segment=attributed_segment_obj,
                keyphrase_attr=keyphrase_attr_dict,
                keyphrase_list=input_phrases_list,
            )

            req_data["segments"][i] = attributed_segment_obj

            logger.info(
                "features embeddings computed and stored",
                extra={"embeddingUri": npz_s3_path},
            )

        modified_request_object = req_data

        return modified_request_object

    def populate_and_embed_graph(self, req_data, segment_object, **kwargs):
        meeting_word_graph = self.populate_word_graph(req_data)

        # Compute embeddings for segments and keyphrases
        modified_request_obj = self.compute_embeddings(
            req_data=req_data,
            segment_object=segment_object,
            meeting_word_graph=meeting_word_graph,
            **kwargs,
        )

        self.utils.write_to_json(modified_request_obj, file_name="segment_attr")

        return modified_request_obj, meeting_word_graph

    def _form_segment_attr_object(self, segment: dict, segment_attr: dict) -> dict:
        attr_object = {"attributes": segment_attr}
        attributed_segment = {**segment, **attr_object}

        return attributed_segment

    def _form_keyphrase_attr_object(self, segment, keyphrase_attr, **kwargs):
        keyphrase_list = kwargs.get("keyphrase_list")
        keyphrase_attr["values"] = keyphrase_list

        attr_object = {"keyphrases": keyphrase_attr}
        keyphrase_attr_segment = {**segment, **attr_object}

        return keyphrase_attr_segment

    def encode_word_graph(self, word_graph):
        word_graph = self.ranker.compute_edge_weights(word_graph)

        return word_graph

    def _compute_relevant_phrases(self, **kwargs):
        kp_dict = self.ranker.compute_local_relevance(dict_key="descriptive", **kwargs)

        # Compute the relevance of entities
        kp_dict = self.ranker.compute_local_relevance(dict_key="entities", **kwargs)

        return kp_dict

    def _make_validation(self, keyphrase_object, context_id, instance_id):
        validation_id = self.utils.hash_sha_object()
        validation_file_name = self.utils.write_to_json(
            keyphrase_object, file_name="keyphrase_validation_" + validation_id
        )
        self.io_util.upload_validation(
            context_id=context_id,
            feature_dir=self.feature_dir,
            instance_id=instance_id,
            validation_file_name=validation_file_name,
        )

    async def get_keyphrases(
        self,
        req_data,
        segment_object: SegmentType,
        meeting_word_graph=None,
        n_kw=10,
        default_form="descriptive",
        rank_by="segment_relevance",
        sort_by="loc",
        validate: bool = False,
        **kwargs,
    ):
        start = timer()

        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        # Use startTime of the first segment as relative starting point
        relative_time = self.utils.formatTime(
            segment_object[0]["startTime"], datetime_object=True
        )

        populate_graph = kwargs.get("populate_graph", True)
        group_id = kwargs.get("group_id")
        try:
            if meeting_word_graph is None:
                # Get graph objects
                meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

            logger.info("Adding segments before extracting keyphrases")
            # Repopulate the graphs
            modified_request_obj, meeting_word_graph = self.populate_and_embed_graph(
                req_data=req_data, segment_object=segment_object, **kwargs
            )

            keyphrase_object = self.extract_keywords(
                segment_object=segment_object,
                meeting_word_graph=meeting_word_graph,
                default_form=default_form,
                relative_time=relative_time,
            )

            try:
                for i, kp_dict in enumerate(keyphrase_object):
                    segment_id = kp_dict["segmentId"]
                    segment_query_obj = self.ranker.form_segment_query(
                        segment_id=segment_id, populate_graph=populate_graph
                    )
                    response = await self.query_client.query_graph(
                        query_object=segment_query_obj
                    )
                    npz_file = self.ranker.query_segment_embeddings(
                        response=response, populate_graph=populate_graph
                    )
                    kp_dict = self._compute_relevant_phrases(
                        npz_file=npz_file,
                        segment_id=segment_id,
                        kp_dict=kp_dict,
                        populate_graph=populate_graph,
                        group_id=group_id,
                    )
                    keyphrase_object[i] = kp_dict

                keyphrase_object = self.ranker.compute_normalized_boosted_rank(
                    keyphrase_object=keyphrase_object, dict_key=default_form
                )
                keyphrases, keyphrase_object = self.prepare_keyphrase_output(
                    keyphrase_object=keyphrase_object,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by=rank_by,
                    sort_by=sort_by,
                    remove_phrases=True,
                )
            except Exception as e:
                logger.warning(
                    "Error computing keyphrase relevance",
                    extra={"warnMsg": e, "trace": traceback.print_exc()},
                )

                keyphrases, keyphrase_object = self.prepare_keyphrase_output(
                    keyphrase_object=keyphrase_object,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by="pagerank",
                    sort_by=sort_by,
                    remove_phrases=False,
                )

            if validate:
                self._make_validation(
                    keyphrase_object=keyphrase_object,
                    context_id=context_id,
                    instance_id=instance_id,
                )

            logger.debug(
                "keyphrases extracted successfully",
                extra={"result": keyphrases, "output": keyphrase_object},
            )

            result = {"keyphrases": keyphrases}
            return result

        except Exception as e:
            end = timer()
            logger.error(
                "Error extracting keyphrases from segment",
                extra={
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                    "segmentsReceived": [seg_id["id"] for seg_id in segment_object],
                    "err": traceback.print_exc(),
                    "errMsg": e,
                },
            )
            raise

    async def get_keyphrases_with_offset(
        self,
        req_data,
        n_kw=10,
        default_form="descriptive",
        rank_by="segment_relevance",
        sort_by="loc",
        validate: bool = False,
        **kwargs,
    ):
        start = timer()
        keyphrase_offsets = []
        segment_object = req_data["segments"]
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        populate_graph = kwargs.get("populate_graph", True)
        group_id = kwargs.get("group_id")
        try:
            # Get graph objects
            meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

            logger.info("Adding segments before extracting keyphrases")
            # Repopulate the graphs
            modified_request_obj, meeting_word_graph = self.populate_and_embed_graph(
                req_data=req_data, segment_object=segment_object, **kwargs
            )

            relative_time = self.utils.formatTime(
                req_data["relativeTime"], datetime_object=True
            )
            keyphrase_object = self.extract_keywords(
                segment_object=segment_object,
                meeting_word_graph=meeting_word_graph,
                default_form=default_form,
                relative_time=relative_time,
            )

            try:
                for i, kp_dict in enumerate(keyphrase_object):
                    segment_id = kp_dict["segmentId"]
                    segment_query_obj = self.ranker.form_segment_query(
                        segment_id=segment_id, populate_graph=populate_graph
                    )
                    response = await self.query_client.query_graph(
                        query_object=segment_query_obj
                    )
                    npz_file = self.ranker.query_segment_embeddings(
                        response=response, populate_graph=populate_graph
                    )
                    kp_dict = self._compute_relevant_phrases(
                        npz_file=npz_file,
                        segment_id=segment_id,
                        kp_dict=kp_dict,
                        populate_graph=populate_graph,
                        group_id=group_id,
                    )
                    keyphrase_object[i] = kp_dict

                keyphrase_object = self.ranker.compute_normalized_boosted_rank(
                    keyphrase_object=keyphrase_object, dict_key=default_form
                )
                keyphrases, keyphrase_object = self.prepare_keyphrase_output(
                    keyphrase_object=keyphrase_object,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by=rank_by,
                    sort_by=sort_by,
                    remove_phrases=True,
                )
            except Exception as e:
                logger.warning(
                    "Error computing keyphrase relevance",
                    extra={"warnMsg": e, "trace": traceback.print_exc()},
                )

                keyphrases, keyphrase_object = self.prepare_keyphrase_output(
                    keyphrase_object=keyphrase_object,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by="pagerank",
                    sort_by=sort_by,
                    remove_phrases=False,
                )

            keyphrase_offsets = self.parse_keyphrase_offset(
                keyphrase_list=keyphrases, keyphrase_object=keyphrase_object
            )
            if validate:
                self._make_validation(
                    keyphrase_object=keyphrase_object,
                    context_id=context_id,
                    instance_id=instance_id,
                )

            end = timer()
            logger.debug(
                "Extracted keyphrases with offsets",
                extra={"output": keyphrase_object, "responseTime": end - start},
            )

        except Exception:
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

        result = {"keyphrases": keyphrase_offsets}

        return result

    def parse_keyphrase_offset(self, keyphrase_list, keyphrase_object):

        keyphrase_offset_list = []
        for i, segment_dict in enumerate(keyphrase_object):
            segment = segment_dict["segments"]
            offset_ts = segment_dict["offset"]
            for word in keyphrase_list:
                loc = segment.find(word)
                if loc > -1:
                    keyphrase_offset_list.append({word: offset_ts})

        # Reformat as per api contract
        keyphrase_offset_output = {
            words: offset
            for elements in keyphrase_offset_list
            for words, offset in elements.items()
        }

        return keyphrase_offset_output

    def extract_keywords(
        self,
        segment_object: SegmentType,
        meeting_word_graph: nx.Graph,
        relative_time=None,
        preserve_singlewords=False,
        default_form="descriptive",
    ):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            segment_object:
            relative_time:
            meeting_word_graph:
            req_data:
            preserve_singlewords:

        Returns: processed_entities (list) cleaned_keyphrase_list (list[Dict]): Returns list of dictionary -
        {
            "segmentId": str,
            "segments": str,
            "offset": 0.0,
            "original": {word: score},
            "descriptive": {word: score}
        }
        where, score = list(pagerank_score, segment_relevance_score, boosted_score, norm_boosted_score, location)
        """
        segment_entities = []
        cleaned_keyphrase_list = []
        cleaned_keyphrase_dict = {
            "segmentId": str,
            "segments": str,
            "offset": 0.0,
            "original": {},
            "descriptive": {},
            "entities": {},
        }

        segment_relevance_score = 0
        boosted_score = 0
        norm_boosted_score = 0
        keyphrase_list, descriptive_kp = self.wg.get_segment_keyphrases(
            segment_object=segment_object, word_graph=meeting_word_graph
        )
        for i in range(len(segment_object)):
            input_segment = segment_object[i].get("originalText")
            input_segment_id = segment_object[i].get("id")

            segment_dict = deepcopy(cleaned_keyphrase_dict)

            segment_dict["segmentId"] = input_segment_id
            segment_dict["segments"] = input_segment
            if relative_time is not None:
                # Set offset time for every keywords
                start_time = segment_object[i].get("startTime")
                start_time = self.utils.formatTime(start_time, datetime_object=True)
                offset_time = float((start_time - relative_time).seconds)
                segment_dict["offset"] = offset_time
            else:
                offset_time = 0.0

            # Get entities
            entities = self.wg.get_custom_entities(input_segment)
            segment_entities.extend(entities)

            # Get cleaned words
            for word, pagerank_score in keyphrase_list:
                loc = input_segment.find(word)
                if loc > -1 and ("*" not in word or "." not in word):
                    segment_dict["original"][word] = list(
                        (
                            pagerank_score,
                            segment_relevance_score,
                            boosted_score,
                            norm_boosted_score,
                            loc + offset_time,
                        )
                    )

            # Get cleaned descriptive phrases
            for word, pagerank_score in descriptive_kp:
                loc = input_segment.find(word)
                if loc > -1 and ("*" not in word or "." not in word):
                    segment_dict["descriptive"][word] = list(
                        (
                            pagerank_score,
                            segment_relevance_score,
                            boosted_score,
                            norm_boosted_score,
                            loc + offset_time,
                        )
                    )

            # Add entity scores in the object
            for item in segment_entities:
                word = item["text"]
                preference = item["preference"]
                loc = input_segment.find(word)
                loc_small = input_segment.find(word.lower())
                if (loc > -1 or loc_small > -1) and ("*" not in word):
                    try:
                        entity_pagerank_score = meeting_word_graph.nodes[word].get(
                            "pagerank"
                        )
                    except Exception:
                        try:
                            entity_pagerank_score = meeting_word_graph.nodes[
                                word.lower()
                            ].get("pagerank")
                        except Exception:
                            entity_pagerank_score = 0

                    final_loc = loc if loc > -1 else loc_small
                    segment_dict["entities"][word] = list(
                        (
                            entity_pagerank_score,
                            segment_relevance_score,
                            boosted_score,
                            norm_boosted_score,
                            preference,
                            final_loc + offset_time,
                        )
                    )

            cleaned_keyphrase_list.append(segment_dict)

        cleaned_keyphrase_list = self.utils.post_process_output(
            keyphrase_object=cleaned_keyphrase_list,
            preserve_singlewords=preserve_singlewords,
            dict_key=default_form,
        )

        return cleaned_keyphrase_list

    def prepare_keyphrase_output(
        self,
        keyphrase_object: List[dict],
        top_n: int = None,
        default_form: str = "descriptive",
        rank_by: str = "pagerank",
        sort_by: str = "loc",
        remove_phrases: bool = True,
        phrase_threshold=3,
    ) -> Tuple[list, List[dict]]:

        final_keyphrase_dict = OrderedDict()
        final_entity_dict = OrderedDict()

        for i, kp_dict in enumerate(keyphrase_object):
            keyphrase_dict = kp_dict[default_form]
            entity_dict = kp_dict["entities"]

            try:
                entity_quality_score = kp_dict["entitiesQuality"]
                keyphrase_quality_score = kp_dict["medianKeyphraseQuality"]
            except KeyError as e:
                logger.warning(
                    "Unable to compute entities and keyphrase quality score",
                    extra={"warnMsg": e},
                )
                entity_quality_score = 0
                keyphrase_quality_score = 0

            # # Sort by rank/scores
            # For chapters: Choose top-n from each segment for better diversity
            (
                ranked_entities_dict,
                ranked_keyphrase_dict,
            ) = self.utils.limit_phrase_list(
                entities_dict=entity_dict,
                keyphrase_dict=keyphrase_dict,
                phrase_limit=top_n,
                entities_limit=5,
                entity_quality_score=entity_quality_score,
                keyphrase_quality_score=keyphrase_quality_score,
                rank_by=rank_by,
                sort_by=sort_by,
                remove_phrases=False,
                final_sort=False,
            )

            final_keyphrase_dict = {
                **final_keyphrase_dict,
                **ranked_keyphrase_dict,
            }
            final_entity_dict = {**final_entity_dict, **ranked_entities_dict}

        logger.debug(
            "Keyphrase and entity list before limiting",
            extra={
                "entities": list(final_entity_dict.keys()),
                "keyphrases": list(final_keyphrase_dict.keys()),
                "count": len(final_entity_dict.keys())
                + len(final_keyphrase_dict.keys()),
            },
        )

        # Set a dynamic threshold for top-n number
        total_phrase_count = len(final_entity_dict.keys()) + len(
            final_keyphrase_dict.keys()
        )
        dynamic_top_n = int(total_phrase_count / phrase_threshold)
        if dynamic_top_n < top_n:
            dynamic_top_n = top_n

        if top_n == 10 or top_n == 5:
            dynamic_top_n = top_n

        if dynamic_top_n >= 10:
            dynamic_top_n = 10

        try:
            overall_entity_quality_score = np.mean(
                [scores[3] for entity, scores in final_entity_dict.items()]
            )
            overall_keyphrase_quality_score = np.mean(
                [scores[3] for phrase, scores in final_keyphrase_dict.items()]
            )
        except Exception as e:
            logger.warning(
                "Unable to compute overall quality scores", extra={"warnMsg": e},
            )

            overall_entity_quality_score = 0
            overall_keyphrase_quality_score = 0

        # Limit keyphrase list to top-n
        if top_n == 10:
            remove_phrases = False

        sorted_keyphrase_dict = self.utils.limit_phrase_list(
            entities_dict=final_entity_dict,
            keyphrase_dict=final_keyphrase_dict,
            phrase_limit=dynamic_top_n,
            entities_limit=5,
            entity_quality_score=overall_entity_quality_score,
            keyphrase_quality_score=overall_keyphrase_quality_score,
            rank_by=rank_by,
            sort_by=sort_by,
            remove_phrases=remove_phrases,
            final_sort=True,
        )

        logger.debug(
            "Keyphrase and entity list after limiting",
            extra={
                "keyphrases": list(sorted_keyphrase_dict.keys()),
                "count": len(sorted_keyphrase_dict.keys()),
                "threshold": dynamic_top_n,
                "keyphraseQuality": overall_keyphrase_quality_score,
                "entityQuality": overall_entity_quality_score,
            },
        )
        keyphrase = [phrases for phrases, scores in sorted_keyphrase_dict.items()]
        keyphrase = self._dedupe_phrases(keyphrase)

        return keyphrase, keyphrase_object

    def _dedupe_phrases(self, keyphrase_list):
        """
        Remove any duplicate phrases arising due to difference in cases.
        Args:
            keyphrase_list:

        Returns:

        """
        deduped_keyphrase_list = list(process.dedupe(keyphrase_list))

        return deduped_keyphrase_list

    def get_instance_keyphrases(self, req_data, n_kw=10):

        segment_object = req_data["segments"]
        # Get graph objects
        meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

        keyphrase_list, descriptive_kp = self.wg.get_segment_keyphrases(
            segment_object=segment_object, word_graph=meeting_word_graph
        )
        instance_keyphrases = [words for words, score in keyphrase_list]

        result = {"keyphrases": instance_keyphrases[:n_kw]}

        return result

    def reset_keyphrase_graph(self, req_data):
        start = timer()
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        # Download context graph from s3 and remove the word graph object upon reset

        word_graph = self._retrieve_word_graph(req_data=req_data)
        word_graph_id = word_graph.graph.get("graphId")

        # Write it back to s3
        self.io_util.upload_s3(
            graph_obj=word_graph,
            s3_dir=self.context_dir,
            context_id=context_id,
            instance_id=instance_id,
        )

        end = timer()
        logger.info(
            "Post-reset: Graph info",
            extra={
                "deletedGraphId": word_graph_id,
                "nodes": word_graph.number_of_nodes(),
                "edges": word_graph.number_of_edges(),
                "responseTime": end - start,
                "instanceId": req_data["instanceId"],
            },
        )
