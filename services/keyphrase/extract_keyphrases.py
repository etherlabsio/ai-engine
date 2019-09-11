import networkx as nx
import logging
from timeit import default_timer as timer
import traceback
from typing import List, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy
import numpy as np

from graphrank.core import GraphRank
from graphrank.utils import TextPreprocess, GraphUtils

from .utils import KeyphraseUtils
from .knowledge_graph import KnowledgeGraph
from .ranker import KeyphraseRanker
from .s3io import S3IO
from .word_graph import WordGraphBuilder

logger = logging.getLogger(__name__)

SegmentType = Dict


class KeyphraseExtractor(object):
    def __init__(
        self, s3_client=None, encoder_lambda_client=None, lambda_function=None
    ):
        self.context_dir = "/context-instance-graphs/"
        self.kg = KnowledgeGraph()
        self.utils = KeyphraseUtils()
        self.io_util = S3IO(s3_client=s3_client, graph_utils_obj=GraphUtils())
        self.ranker = KeyphraseRanker(
            encoder_lambda_client=encoder_lambda_client,
            lambda_function=lambda_function,
            s3_io_util=self.io_util,
            context_dir=self.context_dir,
            knowledge_graph_object=self.kg,
        )
        self.wg = WordGraphBuilder(
            graphrank_obj=GraphRank(),
            textpreprocess_obj=TextPreprocess(),
            graphutils_obj=GraphUtils(),
            keyphrase_utils_obj=self.utils,
        )

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

    def get_graph_id(self, req_data):
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]
        graph_id = context_id + ":" + instance_id
        return graph_id

    def initialize_meeting_graph(self, req_data: dict):
        graph_id = self.get_graph_id(req_data=req_data)
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        context_graph = nx.DiGraph(graphId=graph_id)
        meeting_word_graph = nx.Graph(graphId=graph_id)

        # Populate context information into meeting-knowledge-graph
        context_graph = self.kg.populate_context_info(request=req_data, g=context_graph)
        context_graph = self.kg.populate_word_graph_info(
            request=req_data, context_graph=context_graph, word_graph=meeting_word_graph
        )

        logger.info(
            "Meeting word graph intialized",
            extra={"currentGraphId": meeting_word_graph.graph.get("graphId")},
        )
        logger.info("Uploading serialized graph object")
        self.io_util.upload_s3(
            graph_obj=context_graph,
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
        )

        # Start the encoder lambda to avoid cold start problem
        logger.info("Invoking lambda to reduce cold-start ...")
        test_segment = ["Wake up Sesame!"]
        self.ranker.get_embeddings(input_list=test_segment, req_data=req_data)

    def _retrieve_context_graph(
        self, req_data: SegmentType
    ) -> Tuple[nx.DiGraph, nx.Graph]:
        """
        Download context graph and meeting word graph from s3
        Args:
            req_data:

        Returns:

        """

        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        # Get graph object from S3
        context_graph = self.io_util.download_s3(
            context_id=context_id, instance_id=instance_id, s3_dir=self.context_dir
        )

        # Get meeting word graph object from the context graph
        meeting_word_graph = self.kg.query_word_graph_object(
            context_graph=context_graph
        )

        return context_graph, meeting_word_graph

    def _populate_push_context_graph(
        self,
        req_data: SegmentType,
        context_graph: nx.DiGraph,
        meeting_word_graph: nx.Graph,
        attribute_dict=None,
    ) -> Tuple[nx.DiGraph, nx.Graph]:
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

        # Populate meeting-level knowledge graph
        context_graph = self.kg.populate_instance_info(
            request=req_data, g=context_graph, attribute_dict=attribute_dict
        )

        # Add word graph object to context graph to reduce population time in next requests
        context_graph = self.kg.populate_word_graph_info(
            context_graph=context_graph, word_graph=meeting_word_graph, request=req_data
        )

        # Write back the graph object to S3
        self.io_util.upload_s3(
            graph_obj=context_graph,
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
        )

        return context_graph, meeting_word_graph

    def _update_context_with_keyphrases(
        self,
        req_data: SegmentType,
        segment_keyphrases: list,
        context_graph: nx.DiGraph = None,
        attr_dict=None,
        is_pim=False,
        upload=False,
    ):
        """
        Get keyphrases for each input segment and add it to context graph, then upload it.
        Args:
            req_data:
            segment_keyphrases:
            context_graph:
            attr_dict:
            is_pim:
            upload:

        Returns:

        """
        start = timer()
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        if context_graph is None:
            # Download KG from s3
            context_graph = self.io_util.download_s3(
                context_id=context_id, instance_id=instance_id, s3_dir=self.context_dir
            )

        try:
            # Populate keyphrases in KG
            context_graph = self.kg.populate_keyphrase_info(
                request=req_data,
                keyphrase_list=segment_keyphrases,
                g=context_graph,
                keyphrase_attr_dict=attr_dict,
                is_pim=is_pim,
            )
        except Exception:
            logger.debug(traceback.print_exc())

        if upload:
            # Write it back to s3
            self.io_util.upload_s3(
                graph_obj=context_graph,
                s3_dir=self.context_dir,
                context_id=context_id,
                instance_id=instance_id,
            )

        end = timer()
        logger.info(
            "meeting knowledge graph updated with keywords",
            extra={
                "graphId": context_graph.graph.get("graphId"),
                "kgNodes": context_graph.number_of_nodes(),
                "kgEdges": context_graph.number_of_edges(),
                "instanceId": req_data["instanceId"],
                "responseTime": end - start,
            },
        )

        return context_graph

    def check_for_segment_id(self, segment_object, context_graph):
        # Check if segment_id already exists in context graph before populating

        check_status = 0
        context_segment_id_list = []
        for node, attr in context_graph.nodes.data("attribute"):
            if attr == "segmentId":
                context_segment_id_list.append(node)

        for i in range(len(segment_object)):
            if segment_object[i].get("id") in context_segment_id_list:
                # Re-populate graph in case google transcripts are present
                check_status += 1
            else:
                continue

        if check_status == len(segment_object):
            logger.debug("Segment ID already present in the context graph ...")
            return True
        else:
            logger.debug("Segment not found in existing context graph")
            return False

    def populate_word_graph(self, req_data):
        start = timer()
        # Get graph objects
        context_graph, meeting_word_graph = self._retrieve_context_graph(
            req_data=req_data
        )
        segment_object = req_data["segments"]

        # Populate word graph for the current instance
        try:
            text_list = self.utils.read_segments(segment_object=segment_object)
            meeting_word_graph = self.wg.build_custom_graph(
                text_list=text_list, graph=meeting_word_graph
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

        # Populate instance info and push context graphs
        context_graph, meeting_word_graph = self._populate_push_context_graph(
            req_data=req_data,
            context_graph=context_graph,
            meeting_word_graph=meeting_word_graph,
        )

        return context_graph, meeting_word_graph

    def populate_context_embeddings(
        self,
        req_data,
        segment_object,
        context_graph=None,
        meeting_word_graph=None,
        default_form="descriptive",
    ):
        """
        Compute embedding vectors for segments and segment-keyphrases and store them as node attributes in the knowledge
        graph.
        Args:
            default_form:
            segment_object:
            meeting_word_graph:
            context_graph:
            req_data:

        Returns:

        """
        if context_graph is None and meeting_word_graph is None:
            # Get graph objects
            context_graph, meeting_word_graph = self._retrieve_context_graph(
                req_data=req_data
            )

        entities_list, keyphrase_object = self.extract_keywords(
            segment_object=segment_object,
            context_graph=context_graph,
            meeting_word_graph=meeting_word_graph,
            default_form=default_form,
        )

        # Get segment text
        for i, kp_dict in enumerate(keyphrase_object):
            seg_text = kp_dict["segments"]
            segment_keyphrase_dict = kp_dict[default_form]

            keyphrase_list = list(segment_keyphrase_dict.keys())

            # Compute segment embedding vector and store it
            segment_embedding = self.ranker.get_embeddings(
                input_list=[seg_text], req_data=req_data
            )

            # Compute keyphrase embedding vectors and store it
            keyphrase_embeddings = self.ranker.get_embeddings(
                input_list=keyphrase_list, req_data=req_data
            )

            # Update context graph with embedding vectors
            context_graph, meeting_word_graph = self._populate_push_context_graph(
                req_data=req_data,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
                attribute_dict={"embedding_vector": np.array(segment_embedding)},
            )

            context_graph = self._update_context_with_keyphrases(
                req_data=req_data,
                segment_keyphrases=keyphrase_list,
                context_graph=context_graph,
                is_pim=False,
                upload=True,
                attr_dict={
                    "embedding_vector": np.array(keyphrase_embeddings),
                    "keyphraseType": "descriptive",
                },
            )

        return context_graph, meeting_word_graph

    def encode_word_graph(self, word_graph):
        word_graph = self.ranker.compute_edge_weights(word_graph)

        return word_graph

    def get_keyphrases(
        self,
        req_data,
        segment_object: SegmentType,
        context_graph=None,
        meeting_word_graph=None,
        n_kw=10,
        rank=True,
        default_form="descriptive",
        rank_by="segment_relevance",
        sort_by="loc",
        validate: bool = False,
    ):
        start = timer()

        keyphrases = []
        try:
            if context_graph is None and meeting_word_graph is None:
                # Get graph objects
                context_graph, meeting_word_graph = self._retrieve_context_graph(
                    req_data=req_data
                )

                # handle the situation when word graph is removed but gets request later
                if meeting_word_graph.number_of_nodes() == 0:
                    # Repopulate the graphs
                    context_graph, meeting_word_graph = self.populate_word_graph(
                        req_data
                    )

                    # Compute embeddings for segments and keyphrases
                    context_graph, meeting_word_graph = self.populate_context_embeddings(
                        req_data=req_data,
                        segment_object=segment_object,
                        context_graph=context_graph,
                        meeting_word_graph=meeting_word_graph,
                    )

            # Check if the segments are already present in the context graph
            status = self.check_for_segment_id(
                segment_object=segment_object, context_graph=context_graph
            )
            if status is not True:
                logger.info("Adding segments before extracting keyphrases")
                # Repopulate the graphs
                context_graph, meeting_word_graph = self.populate_word_graph(req_data)

                # Compute embeddings for segments and keyphrases
                context_graph, meeting_word_graph = self.populate_context_embeddings(
                    req_data=req_data,
                    segment_object=segment_object,
                    context_graph=context_graph,
                    meeting_word_graph=meeting_word_graph,
                )

            entities, keyphrase_object = self.extract_keywords(
                segment_object=segment_object,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
                default_form=default_form,
            )

            if rank:
                keyphrase_object = self.ranker.compute_local_relevance(
                    keyphrase_object=keyphrase_object,
                    context_graph=context_graph,
                    normalize=False,
                )

            keyphrases, keyphrase_object = self.prepare_keyphrase_output(
                processed_entities=entities,
                keyphrase_object=keyphrase_object,
                top_n=n_kw,
                default_form=default_form,
                rank_by=rank_by,
                sort_by=sort_by,
            )

            if validate:
                self.utils.write_to_json(keyphrase_object)

            logger.debug(
                "keyphrases extracted successfully",
                extra={"result": keyphrases, "output": keyphrase_object},
            )

            # TODO need to separate it and move to context-graph-constructor service
            # Populate PIM keyphrases to context graph
            if n_kw > 6:
                original_pim_keyphrases = list(keyphrase_object[0]["original"].keys())
                self._update_context_with_keyphrases(
                    req_data=req_data,
                    segment_keyphrases=original_pim_keyphrases,
                    context_graph=context_graph,
                    is_pim=True,
                    upload=True,
                    attr_dict={"keyphraseType": "original"},
                )
                logger.info("Updated context graph with PIM keyphrases")

        except Exception as e:
            end = timer()
            logger.error(
                "Error extracting keyphrases from segment",
                extra={
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                    "segmentObj": req_data["segments"],
                    "err": traceback.print_exc(),
                    "errMsg": e,
                },
            )

        result = {"keyphrases": keyphrases}

        return result

    def get_keyphrases_with_offset(
        self,
        req_data,
        n_kw=10,
        rank=True,
        default_form="descriptive",
        rank_by="segment_relevance",
        sort_by="loc",
        validate: bool = False,
    ):
        start = timer()
        keyphrase_offsets = []
        segment_object = req_data["segments"]
        try:
            # Get graph objects
            context_graph, meeting_word_graph = self._retrieve_context_graph(
                req_data=req_data
            )

            # Check if the segments are already present in the context graph
            status = self.check_for_segment_id(
                segment_object=segment_object, context_graph=context_graph
            )
            if status is not True:
                logger.info("Adding segments before extracting keyphrases")
                # Repopulate the graphs
                context_graph, meeting_word_graph = self.populate_word_graph(req_data)

                # Compute embeddings for segments and keyphrases
                context_graph, meeting_word_graph = self.populate_context_embeddings(
                    req_data=req_data,
                    segment_object=segment_object,
                    context_graph=context_graph,
                    meeting_word_graph=meeting_word_graph,
                )

            relative_time = self.utils.formatTime(
                req_data["relativeTime"], datetime_object=True
            )
            entities, keyphrase_object = self.extract_keywords(
                segment_object=segment_object,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
                relative_time=relative_time,
            )

            if rank:
                keyphrase_object = self.ranker.compute_local_relevance(
                    keyphrase_object=keyphrase_object, context_graph=context_graph
                )

            keyphrases, keyphrase_object = self.prepare_keyphrase_output(
                processed_entities=entities,
                keyphrase_object=keyphrase_object,
                top_n=n_kw,
                default_form=default_form,
                rank_by=rank_by,
                sort_by=sort_by,
            )

            keyphrase_offsets = self.parse_keyphrase_offset(
                keyphrase_list=keyphrases, keyphrase_object=keyphrase_object
            )
            if validate:
                self.utils.write_to_json(keyphrase_object)

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
        context_graph: nx.DiGraph,
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
            context_graph:
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
        where, score = list(pagerank_score, segment_relevance_score, boosted_score, location)
        """
        segment_entities = []
        cleaned_keyphrase_list = []
        cleaned_keyphrase_dict = {
            "segmentId": str,
            "segments": str,
            "offset": 0.0,
            "original": {},
            "descriptive": {},
        }

        segment_relevance_score = 0
        boosted_score = 0
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

            # Get entities
            entities = self.wg.get_entities(input_segment)
            segment_entities.extend(entities)

            # Get cleaned words
            for word, pagerank_score in keyphrase_list:
                loc = input_segment.find(word)
                if loc > -1 and ("*" not in word or "." not in word):
                    segment_dict["original"][word] = list(
                        (pagerank_score, segment_relevance_score, boosted_score, loc)
                    )

            # Get cleaned descriptive phrases
            for word, pagerank_score in descriptive_kp:
                loc = input_segment.find(word)
                if loc > -1 and ("*" not in word or "." not in word):
                    segment_dict["descriptive"][word] = list(
                        (pagerank_score, segment_relevance_score, boosted_score, loc)
                    )

            cleaned_keyphrase_list.append(segment_dict)

        processed_entities, cleaned_keyphrase_list = self.utils.post_process_output(
            entity_list=segment_entities,
            keyphrase_object=cleaned_keyphrase_list,
            preserve_singlewords=preserve_singlewords,
            dict_key="original",
        )

        processed_entities, cleaned_keyphrase_list = self.utils.post_process_output(
            entity_list=segment_entities,
            keyphrase_object=cleaned_keyphrase_list,
            preserve_singlewords=preserve_singlewords,
            dict_key="descriptive",
        )

        return processed_entities, cleaned_keyphrase_list

    def prepare_keyphrase_output(
        self,
        processed_entities: list,
        keyphrase_object: List[dict],
        top_n: int = None,
        default_form: str = "descriptive",
        rank_by: str = "pagerank",
        sort_by: str = "loc",
    ) -> Tuple[list, List[dict]]:

        rank_key_dict = {
            "pagerank": 0,
            "segment_relevance": 1,
            "boosted_score": 2,
            "order": "desc",
        }

        sort_key_dict = {"loc": 3, "order": "asc"}

        final_keyphrase_dict = OrderedDict()

        for i, kp_dict in enumerate(keyphrase_object):
            keyphrase_dict = kp_dict[default_form]

            # Sort by rank/scores
            ranked_keyphrase_dict = self.utils.sort_dict_by_value(
                dict_var=keyphrase_dict,
                key=rank_key_dict[rank_by],
                order=rank_key_dict["order"],
            )

            final_keyphrase_dict = {**ranked_keyphrase_dict, **final_keyphrase_dict}

        final_keyphrase_dict = self.utils.sort_dict_by_value(
            dict_var=final_keyphrase_dict,
            key=rank_key_dict[rank_by],
            order=rank_key_dict["order"],
        )

        # Limit keyphrase list to top-n
        processed_entities, final_keyphrase_dict = self.utils.limit_phrase_list(
            entities_list=processed_entities,
            keyphrase_dict=final_keyphrase_dict,
            phrase_limit=top_n,
        )

        # Sort only descriptive by time spoken
        if default_form == "descriptive":
            sorted_keyphrase_dict = self.utils.sort_dict_by_value(
                dict_var=final_keyphrase_dict,
                key=sort_key_dict[sort_by],
                order=sort_key_dict["order"],
            )
        else:
            sorted_keyphrase_dict = final_keyphrase_dict

        processed_entities.extend(
            [items for items, values in sorted_keyphrase_dict.items()]
        )
        keyphrase = processed_entities

        return keyphrase, keyphrase_object

    def get_instance_keyphrases(self, req_data, n_kw=10):

        segment_object = req_data["segments"]
        # Get graph objects
        context_graph, meeting_word_graph = self._retrieve_context_graph(
            req_data=req_data
        )

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

        context_graph, word_graph = self._retrieve_context_graph(req_data=req_data)
        context_graph.remove_node(word_graph)

        # Remove the embedding features from the context graph
        context_graph = self.kg.query_for_embedded_nodes(context_graph)
        context_graph = self.kg.query_for_embedded_segments(context_graph)

        word_graph_id = word_graph.graph.get("graphId")

        # Write it back to s3
        self.io_util.upload_s3(
            graph_obj=context_graph,
            s3_dir=self.context_dir,
            context_id=context_id,
            instance_id=instance_id,
        )

        word_graph.clear()

        end = timer()
        logger.info(
            "Post-reset: Graph info - Temporarily not resetting graph",
            extra={
                "deletedGraphId": word_graph_id,
                "nodes": word_graph.number_of_nodes(),
                "edges": word_graph.number_of_edges(),
                "kgNodes": context_graph.number_of_nodes(),
                "kgEdges": context_graph.number_of_edges(),
                "responseTime": end - start,
            },
        )
