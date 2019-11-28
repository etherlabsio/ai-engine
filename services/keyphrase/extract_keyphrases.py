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
        self,
        s3_client=None,
        encoder_lambda_client=None,
        lambda_function=None,
        ner_lambda_function=None,
    ):
        self.context_dir = "/context-instance-graphs/"
        self.feature_dir = "/sessions/"
        self.s3_client = s3_client
        self.kg = KnowledgeGraph()
        self.utils = KeyphraseUtils()
        self.io_util = S3IO(
            s3_client=s3_client,
            graph_utils_obj=GraphUtils(),
            utils=KeyphraseUtils(),
        )
        self.ranker = KeyphraseRanker(
            encoder_lambda_client=encoder_lambda_client,
            lambda_function=lambda_function,
            s3_io_util=self.io_util,
            context_dir=self.context_dir,
            knowledge_graph_object=self.kg,
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
        self.wg.call_custom_ner(input_segment=test_segment)

    def initialize_meeting_graph(self, req_data: dict):
        graph_id = self.get_graph_id(req_data=req_data)
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        context_graph = nx.DiGraph(graphId=graph_id)
        meeting_word_graph = nx.Graph(graphId=graph_id)

        # Populate context information into meeting-knowledge-graph
        context_graph = self.kg.populate_context_info(
            request=req_data, g=context_graph
        )
        context_graph = self.kg.populate_word_graph_info(
            request=req_data,
            context_graph=context_graph,
            word_graph=meeting_word_graph,
            state="processing",
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
        self.wake_up_lambda(req_data=req_data)

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
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
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
        segment_object: SegmentType,
        attribute_dict=None,
    ) -> nx.DiGraph:
        """
        Populate instance information, add meeting word graph to context graph and upload the context graph
        Args:
            req_data:
            context_graph::

        Returns:

        """

        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        # Populate meeting-level knowledge graph
        context_graph = self.kg.populate_instance_info(
            instance_id=instance_id,
            segment_object=segment_object,
            g=context_graph,
            attribute_dict=attribute_dict,
        )

        # Write back the graph object to S3
        self.io_util.upload_s3(
            graph_obj=context_graph,
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
        )

        return context_graph

    def _update_context_with_word_graph(
        self,
        req_data: SegmentType,
        context_graph: nx.DiGraph,
        meeting_word_graph: nx.Graph,
        **kwargs,
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

        # Add word graph object to context graph to reduce population time in next requests
        context_graph = self.kg.populate_word_graph_info(
            context_graph=context_graph,
            word_graph=meeting_word_graph,
            request=req_data,
            **kwargs,
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
        segment_object: SegmentType,
        context_graph: nx.DiGraph = None,
        attr_dict=None,
        is_pim=False,
        upload=False,
        phrase_hash_dict=None,
    ) -> nx.DiGraph:
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
                context_id=context_id,
                instance_id=instance_id,
                s3_dir=self.context_dir,
            )

        try:
            # Populate keyphrases in KG
            context_graph = self.kg.populate_keyphrase_info(
                request=req_data,
                keyphrase_list=segment_keyphrases,
                segment_object=segment_object,
                g=context_graph,
                keyphrase_attr_dict=attr_dict,
                is_pim=is_pim,
                phrase_hash_dict=phrase_hash_dict,
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

            end = timer()
            logger.info(
                "Populated graph and written to s3",
                extra={
                    "graphId": meeting_word_graph.graph.get("graphId"),
                    "nodes": meeting_word_graph.number_of_nodes(),
                    "edges": meeting_word_graph.number_of_edges(),
                    "kgNodes": context_graph.number_of_nodes(),
                    "kgEdges": context_graph.number_of_edges(),
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

        # Populate instance info and push context graphs
        context_graph, meeting_word_graph = self._update_context_with_word_graph(
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
        **kwargs,
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
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]

        populate_graph = kwargs.get("populate_graph", True)
        group_id = kwargs.get("group_id", "")

        if context_graph is None and meeting_word_graph is None:
            # Get graph objects
            context_graph, meeting_word_graph = self._retrieve_context_graph(
                req_data=req_data
            )

        keyphrase_object = self.extract_keywords(
            segment_object=segment_object,
            context_graph=context_graph,
            meeting_word_graph=meeting_word_graph,
            default_form=default_form,
        )

        # Get segment text
        for i, kp_dict in enumerate(keyphrase_object):
            seg_text = kp_dict["segments"]
            segment_id = kp_dict["segmentId"]
            segment_keyphrase_dict = kp_dict[default_form]
            segment_entity_dict = kp_dict["entities"]

            keyphrase_list = list(segment_keyphrase_dict.keys())
            entities_list = list(segment_entity_dict.keys())

            input_phrases_list = []
            input_phrases_list.extend(entities_list)
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

            phrase_hash_dict, phrase_embedding_dict = self.utils.map_embeddings_to_phrase(
                phrase_list=input_phrases_list,
                embedding_list=keyphrase_embeddings,
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
            if populate_graph:
                segment_attr_dict = {
                    "text": segment_object[i]["originalText"],
                    "embedding_vector_uri": npz_s3_path,
                    "embedding_model": "use_v1",
                }

                keyphrase_attr_dict = {
                    "keyphraseType": "descriptive",
                    "keyphraseOrigin": "meetingSummary",
                }
            else:
                segment_attr_dict = {
                    "analyzedText": segment_object[i]["originalText"],
                    "embedding_vector_group_uri": npz_s3_path,
                    "groupId": group_id,
                    "embedding_model": "use_v1",
                }

                keyphrase_attr_dict = {
                    "keyphraseType": "descriptive",
                    "keyphraseOrigin": "groupSummary",
                }

            context_graph = self._populate_push_context_graph(
                req_data=req_data,
                context_graph=context_graph,
                segment_object=segment_object[i],
                attribute_dict=segment_attr_dict,
            )

            context_graph = self._update_context_with_keyphrases(
                req_data=req_data,
                segment_keyphrases=input_phrases_list,
                segment_object=segment_object[i],
                context_graph=context_graph,
                is_pim=False,
                upload=True,
                attr_dict=keyphrase_attr_dict,
                phrase_hash_dict=phrase_hash_dict,
            )

            logger.info(
                "features embeddings computed and stored",
                extra={"embeddingUri": npz_s3_path},
            )

        return context_graph, meeting_word_graph

    def encode_word_graph(self, word_graph):
        word_graph = self.ranker.compute_edge_weights(word_graph)

        return word_graph

    def _compute_relevant_phrases(self, **kwargs):
        keyphrase_object = kwargs.get("keyphrase_object")
        context_graph = kwargs.get("context_graph")
        n_kw = kwargs.get("n_kw")
        default_form = kwargs.get("default_form")
        rank_by = kwargs.get("rank_by")
        sort_by = kwargs.get("sort_by")
        remove_phrases = kwargs.get("remove_phrases")

        keyphrase_object = self.ranker.compute_local_relevance(
            keyphrase_object=keyphrase_object,
            context_graph=context_graph,
            normalize=False,
        )

        # Compute the relevance of entities
        keyphrase_object = self.ranker.compute_local_relevance(
            keyphrase_object=keyphrase_object,
            context_graph=context_graph,
            normalize=False,
            dict_key="entities",
        )

        keyphrases, keyphrase_object = self.prepare_keyphrase_output(
            keyphrase_object=keyphrase_object,
            top_n=n_kw,
            default_form=default_form,
            rank_by=rank_by,
            sort_by=sort_by,
            remove_phrases=remove_phrases,
        )

        return keyphrases, keyphrase_object

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

        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]
        keyphrases = []
        try:
            if context_graph is None and meeting_word_graph is None:
                # Get graph objects
                context_graph, meeting_word_graph = self._retrieve_context_graph(
                    req_data=req_data
                )

            # handle the situation when word graph is removed but gets request later
            if meeting_word_graph.graph.get("state") == "reset":
                # Repopulate the graphs
                logger.info("re-populating graph since it is in reset state")
                context_graph, meeting_word_graph = self._update_context_with_word_graph(
                    req_data=req_data,
                    context_graph=context_graph,
                    meeting_word_graph=meeting_word_graph,
                    state="reset",
                )

                context_graph, meeting_word_graph = self.populate_word_graph(
                    req_data
                )

                # Check if the segments are already present in the context graph
                status = self.check_for_segment_id(
                    segment_object=segment_object, context_graph=context_graph
                )
                if status is not True:
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

            keyphrase_object = self.extract_keywords(
                segment_object=segment_object,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
                default_form=default_form,
            )

            if rank:
                try:
                    keyphrases, keyphrase_object = self._compute_relevant_phrases(
                        keyphrase_object=keyphrase_object,
                        context_graph=context_graph,
                        n_kw=n_kw,
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
                validation_id = self.utils.hash_sha_object()
                validation_file_name = self.utils.write_to_json(
                    keyphrase_object,
                    file_name="keyphrase_validation_" + validation_id,
                )
                self.io_util.upload_validation(
                    context_id=context_id,
                    feature_dir=self.feature_dir,
                    instance_id=instance_id,
                    validation_file_name=validation_file_name,
                )

            logger.debug(
                "keyphrases extracted successfully",
                extra={"result": keyphrases, "output": keyphrase_object},
            )

            # TODO need to separate it and move to context-graph-constructor service
            # Populate PIM keyphrases to context graph
            if n_kw == 10:
                original_pim_keyphrases = list(
                    keyphrase_object[0]["original"].keys()
                )
                pim_keyphrases_hash = dict(
                    zip(
                        map(self.utils.hash_phrase, original_pim_keyphrases),
                        original_pim_keyphrases,
                    )
                )
                self._update_context_with_keyphrases(
                    req_data=req_data,
                    segment_keyphrases=original_pim_keyphrases,
                    segment_object=segment_object[0],
                    context_graph=context_graph,
                    is_pim=True,
                    upload=True,
                    attr_dict={"keyphraseType": "original"},
                    phrase_hash_dict=pim_keyphrases_hash,
                )
                logger.info("Updated context graph with PIM keyphrases")

            self.meeting_keywords.extend(keyphrases)

            result = {"keyphrases": keyphrases}
            return result

        except Exception as e:
            end = timer()
            logger.error(
                "Error extracting keyphrases from segment",
                extra={
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                    "segmentsReceived": [
                        seg_id["id"] for seg_id in segment_object
                    ],
                    "err": traceback.print_exc(),
                    "errMsg": e,
                },
            )
            raise

    def get_summary_chapter_keyphrases(
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
        populate_graph=True,
        group_id="",
    ):
        start = timer()

        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]
        keyphrases = []
        try:
            if context_graph is None and meeting_word_graph is None:
                # Get graph objects
                context_graph, meeting_word_graph = self._retrieve_context_graph(
                    req_data=req_data
                )

            # handle the situation when word graph is removed but gets request later
            if meeting_word_graph.graph.get("state") == "reset":
                # Repopulate the graphs
                logger.info("re-populating graph since it is in reset state")
                context_graph, meeting_word_graph = self._update_context_with_word_graph(
                    req_data=req_data,
                    context_graph=context_graph,
                    meeting_word_graph=meeting_word_graph,
                    state="reset",
                )

                context_graph, meeting_word_graph = self.populate_word_graph(
                    req_data
                )

                # Compute embeddings for segments and keyphrases
                context_graph, meeting_word_graph = self.populate_context_embeddings(
                    req_data=req_data,
                    segment_object=segment_object,
                    context_graph=context_graph,
                    meeting_word_graph=meeting_word_graph,
                    populate_graph=populate_graph,
                    group_id=group_id,
                )

            else:
                logger.info("Adding segments for new summary...")
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
                    populate_graph=populate_graph,
                    group_id=group_id,
                )

            keyphrase_object = self.extract_keywords(
                segment_object=segment_object,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
                default_form=default_form,
            )

            if rank:
                try:
                    keyphrases, keyphrase_object = self._compute_relevant_phrases(
                        keyphrase_object=keyphrase_object,
                        context_graph=context_graph,
                        n_kw=n_kw,
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
                validation_id = self.utils.hash_sha_object()
                validation_file_name = self.utils.write_to_json(
                    keyphrase_object,
                    file_name="keyphrase_validation_" + validation_id,
                )
                self.io_util.upload_validation(
                    context_id=context_id,
                    feature_dir=self.feature_dir,
                    instance_id=instance_id,
                    validation_file_name=validation_file_name,
                )

            logger.debug(
                "keyphrases extracted successfully for new summary",
                extra={"result": keyphrases, "output": keyphrase_object},
            )

            self.meeting_keywords.extend(keyphrases)

            result = {"keyphrases": keyphrases}
            return result

        except Exception as e:
            end = timer()
            logger.error(
                "Error extracting keyphrases from grouped segments",
                extra={
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                    "segmentsReceived": [
                        seg_id["id"] for seg_id in segment_object
                    ],
                    "err": traceback.print_exc(),
                    "errMsg": e,
                },
            )
            raise

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

            relative_time = self.utils.formatTime(
                req_data["relativeTime"], datetime_object=True
            )
            keyphrase_object = self.extract_keywords(
                segment_object=segment_object,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
                relative_time=relative_time,
            )

            if rank:
                keyphrase_object = self.ranker.compute_local_relevance(
                    keyphrase_object=keyphrase_object,
                    context_graph=context_graph,
                )

            keyphrases, keyphrase_object = self.prepare_keyphrase_output(
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
                extra={
                    "output": keyphrase_object,
                    "responseTime": end - start,
                },
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
                start_time = self.utils.formatTime(
                    start_time, datetime_object=True
                )
                offset_time = float((start_time - relative_time).seconds)
                segment_dict["offset"] = offset_time

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
                            loc,
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
                            loc,
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
                        entity_pagerank_score = meeting_word_graph.nodes[
                            word
                        ].get("pagerank")
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
                            final_loc,
                        )
                    )

            cleaned_keyphrase_list.append(segment_dict)

        cleaned_keyphrase_list = self.utils.post_process_output(
            keyphrase_object=cleaned_keyphrase_list,
            preserve_singlewords=preserve_singlewords,
            dict_key="original",
        )

        cleaned_keyphrase_list = self.utils.post_process_output(
            keyphrase_object=cleaned_keyphrase_list,
            preserve_singlewords=preserve_singlewords,
            dict_key="descriptive",
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
            ranked_entities_dict, ranked_keyphrase_dict = self.utils.limit_phrase_list(
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
                "Unable to compute overall quality scores",
                extra={"warnMsg": e},
            )

            overall_entity_quality_score = 0
            overall_keyphrase_quality_score = 0

        # Limit keyphrase list to top-n
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

        keyphrase = [
            phrases for phrases, scores in sorted_keyphrase_dict.items()
        ]

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

        context_graph, word_graph = self._retrieve_context_graph(
            req_data=req_data
        )
        context_graph.remove_node(word_graph)

        # Remove the embedding features from the context graph
        # context_graph = self.kg.query_for_embedded_nodes(context_graph)
        # context_graph = self.kg.query_for_embedded_segments(context_graph)

        word_graph_id = word_graph.graph.get("graphId")

        # Write it back to s3
        self.io_util.upload_s3(
            graph_obj=context_graph,
            s3_dir=self.context_dir,
            context_id=context_id,
            instance_id=instance_id,
        )

        word_graph.clear()

        result = {"keyphrases": self.meeting_keywords}

        end = timer()
        logger.info(
            "Post-reset: Graph info",
            extra={
                "deletedGraphId": word_graph_id,
                "nodes": word_graph.number_of_nodes(),
                "edges": word_graph.number_of_edges(),
                "kgNodes": context_graph.number_of_nodes(),
                "kgEdges": context_graph.number_of_edges(),
                "responseTime": end - start,
                "instanceId": req_data["instanceId"],
            },
        )

        return result
