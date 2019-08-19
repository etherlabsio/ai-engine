import networkx as nx
import logging
from timeit import default_timer as timer
import traceback
from typing import List, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy

from .utils import KeyphraseUtils
from .knowledge_graph import KnowledgeGraph
from .ranker import KeyphraseRanker
from .s3io import S3IO
from .word_graph import WordGraphBuilder

logger = logging.getLogger(__name__)

SegmentType = Dict


class KeyphraseExtractor(object):
    def __init__(self, s3_client=None):
        self.kg = KnowledgeGraph()
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

        self.context_dir = "/context-instance-graphs/"
        self.io_util = S3IO(s3_client=s3_client)
        self.ranker = KeyphraseRanker(
            s3_io_util=self.io_util, context_dir=self.context_dir
        )
        self.utils = KeyphraseUtils()
        self.wg = WordGraphBuilder()

    def get_graph_id(self, req_data):
        context_id = req_data["contextId"]
        instance_id = req_data["instanceId"]
        graph_id = context_id + ":" + instance_id
        return graph_id

    def initialize_meeting_graph(self, req_data: dict):
        graph_id = self.get_graph_id(req_data=req_data)

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
            graph_obj=context_graph, req_data=req_data, s3_dir=self.context_dir
        )

    def retrieve_context_graph(
        self, req_data: SegmentType
    ) -> Tuple[nx.DiGraph, nx.Graph]:
        # Get graph object from S3
        context_graph = self.io_util.download_s3(
            req_data=req_data, s3_dir=self.context_dir
        )

        # Get meeting word graph object from the context graph
        meeting_word_graph = self.kg.query_word_graph_object(
            context_graph=context_graph
        )

        return context_graph, meeting_word_graph

    def populate_push_context_graph(
        self,
        req_data: SegmentType,
        context_graph: nx.DiGraph,
        meeting_word_graph: nx.Graph,
    ) -> Tuple[nx.DiGraph, nx.Graph]:
        # Populate meeting-level knowledge graph
        context_graph = self.kg.populate_instance_info(
            request=req_data, g=context_graph
        )

        # Add word graph object to context graph to reduce population time in next requests
        context_graph = self.kg.populate_word_graph_info(
            context_graph=context_graph, word_graph=meeting_word_graph, request=req_data
        )

        # Write back the graph object to S3
        self.io_util.upload_s3(
            graph_obj=context_graph, req_data=req_data, s3_dir=self.context_dir
        )

        return context_graph, meeting_word_graph

    def update_context_with_word_graph(
        self, req_data: SegmentType, segment_keyphrases: list
    ):
        start = timer()
        # Download KG from s3
        context_graph = self.io_util.download_s3(
            req_data=req_data, s3_dir=self.context_dir
        )

        # Populate keyphrases in KG
        context_graph = self.kg.populate_keyphrase_info(
            request=req_data, keyphrase_list=segment_keyphrases, g=context_graph
        )

        # Write it back to s3
        self.io_util.upload_s3(
            graph_obj=context_graph, s3_dir=self.context_dir, req_data=req_data
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

    def check_for_segment_id(self, req_data, context_graph):
        # Check if segment_id already exists in context graph before populating
        segment_object = req_data["segments"]

        check_status = 0
        for node, attr in context_graph.nodes.data("attribute"):
            if attr == "segment_id":
                for i in range(len(segment_object)):
                    if segment_object[i].get("id") in node:
                        # Re-populate graph in case google transcripts are present
                        logger.debug(
                            "Segment ID already present in the context graph ..."
                        )
                        check_status += 0
                    else:
                        logger.debug("Segment not found in existing context graph")

        if check_status == len(segment_object):
            return True
        else:
            return False

    def populate_word_graph(self, req_data, add_context=False):
        start = timer()
        graph_id = self.get_graph_id(req_data=req_data)

        # Get graph objects
        context_graph, meeting_word_graph = self.retrieve_context_graph(
            req_data=req_data
        )

        # Populate word graph for the current instance
        try:
            text_list = self.utils.read_segments(req_data=req_data)
            meeting_word_graph = self.wg.build_custom_graph(
                text_list=text_list, add_context=add_context, graph=meeting_word_graph
            )

            # # Add similarity embeddings
            # meeting_word_graph = self.encode_word_graph(word_graph=meeting_word_graph)
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

        # Populate and push context graphs
        context_graph, meeting_word_graph = self.populate_push_context_graph(
            req_data=req_data,
            context_graph=context_graph,
            meeting_word_graph=meeting_word_graph,
        )
        end = timer()
        logger.info(
            "Populated graph and written to s3",
            extra={
                "graphId": meeting_word_graph.graph.get("graphId"),
                "graphServiceIdentifier": graph_id,
                "nodes": meeting_word_graph.number_of_nodes(),
                "edges": meeting_word_graph.number_of_edges(),
                "kgNodes": context_graph.number_of_nodes(),
                "kgEdges": context_graph.number_of_edges(),
                "instanceId": req_data["instanceId"],
                "responseTime": end - start,
            },
        )

        return context_graph, meeting_word_graph

    # TODO test chapter offset logic
    # TODO add keyphrase quality score to keyphrase object
    def get_keyphrases(
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

        keyphrases = []
        try:
            # Get graph objects
            context_graph, meeting_word_graph = self.retrieve_context_graph(
                req_data=req_data
            )

            # Check if the segments are already present in the context graph
            status = self.check_for_segment_id(
                req_data=req_data, context_graph=context_graph
            )
            if status is not True:
                # Repopulate the graphs
                context_graph, meeting_word_graph = self.populate_word_graph(
                    req_data, add_context=False
                )

            entities, keyphrase_object = self.extract_keywords(
                req_data,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
            )

            if rank:
                keyphrase_object = self.ranker.compute_local_relevance(
                    keyphrase_object=keyphrase_object
                )

            keyphrases, keyphrase_object = self.prepare_keyphrase_output(
                processed_entities=entities,
                keyphrase_object=keyphrase_object,
                top_n=n_kw,
                default_form=default_form,
                limit_phrase=True,
                rank_by=rank_by,
                sort_by=sort_by,
            )
            if validate:
                self.utils.write_to_json(keyphrase_object)

            logger.debug(
                "keyphrases extracted successfully", extra={"output": keyphrase_object}
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

    def encode_word_graph(self, word_graph):
        word_graph = self.ranker.compute_edge_weights(word_graph)

        return word_graph

    def extract_keywords(
        self,
        req_data: SegmentType,
        context_graph: nx.DiGraph,
        meeting_word_graph: nx.Graph,
        preserve_singlewords=False,
    ):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            meeting_word_graph:
            context_graph:
            req_data:
            preserve_singlewords:

        Returns:

        """
        start = timer()
        segment_entities = []
        cleaned_keyphrase_list = []
        cleaned_keyphrase_dict = {"segments": str, "original": {}, "descriptive": {}}

        segment_relevance_score = 0
        boosted_score = 0
        keyphrase_list, descriptive_kp = self.wg.get_segment_keyphrases(
            req_data=req_data, word_graph=meeting_word_graph
        )
        segment_object = req_data["segments"]
        for i in range(len(segment_object)):
            input_segment = segment_object[i].get("originalText")

            # Get entities
            entities = self.wg.get_entities(input_segment)
            segment_entities.extend(entities)

            segment_dict = deepcopy(cleaned_keyphrase_dict)
            segment_dict["segments"] = input_segment

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
        limit_phrase: bool = True,
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

            # final_keyphrase_list.extend(ranked_keyphrase_list)
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

        # Log final keyphrase list to validation set
        for i, kp_dict in enumerate(keyphrase_object):
            kp_dict["keyphrases"] = keyphrase

        return keyphrase, keyphrase_object

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
            input_segment = segments[i].get("originalText")

            # Set offset time for every keywords
            start_time = segments[i].get("startTime")
            start_time = self.utils.formatTime(start_time, datetime_object=True)
            offset_time = float((start_time - relative_time).seconds)

            for kw, score in keyphrase_list:
                loc = input_segment.find(kw)
                if loc > -1 and ("*" not in kw or "." not in kw):
                    chapter_keywords_list.append(({kw: offset_time}, score))

            entities = self.wg.get_entities(input_segment)
            chapter_entities.extend([{entity: offset_time} for entity in entities])

        chapter_entities_list = [
            entities for ent in chapter_entities for entities, off in ent.items()
        ]

        sort_list = self.utils.sort_by_value(chapter_keywords_list, order="desc")
        if top_n is not None:
            sort_list = sort_list[:top_n]

        chapter_keyphrases = [phrases for phrases, score in sort_list]

        # Get distinct entities and keyphrases
        chapter_entities_list = list(dict.fromkeys(chapter_entities_list))

        # Post-process entities
        chapter_entities_list = self.utils.post_process_entities(chapter_entities_list)

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
            chapter_entities, chapter_multiphrase_list = self.utils.limit_phrase_list(
                entities_list=chapter_entities,
                keyphrase_dict=chapter_multiphrase_list,
                phrase_limit=top_n,
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

    def get_instance_keyphrases(self, req_data, n_kw=10):
        keyphrase_list, descriptive_kp = self.compute_keyphrases(req_data)
        instance_keyphrases = [words for words, score in keyphrase_list]

        result = {"keyphrases": instance_keyphrases[:n_kw]}

        return result

    def get_chapter_offset_keyphrases(
        self, req_data, n_kw=10, default_form="descriptive"
    ):
        start = timer()
        keyphrase_list, descriptive_kp = self.compute_keyphrases(req_data)

        relative_time = self.utils.formatTime(
            req_data["relativeTime"], datetime_object=True
        )
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

        # Download context graph from s3 and remove the word graph object upon reset

        context_graph, word_graph = self.retrieve_context_graph(req_data=req_data)
        context_graph.remove_node(word_graph)

        word_graph_id = word_graph.graph.get("graphId")

        # Write it back to s3
        self.io_util.upload_s3(
            graph_obj=context_graph, s3_dir=self.context_dir, req_data=req_data
        )

        # self.gr.reset_graph()
        word_graph.clear()

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
            },
        )
