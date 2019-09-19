import networkx as nx
import json as js
import logging
from copy import deepcopy
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeGraph(object):
    def __init__(self):
        self.context_label = {"attribute": "contextId"}
        self.instance_label = {"attribute": "instanceId"}
        self.context_instance_rel = {"relation": "hasMeeting"}

        self.instance_segment_rel = {"relation": "hasSegment"}

        self.user_label = {"attribute": "authorId"}
        self.transcriber_label = {"attribute": "segmentProvider"}

        self.segment_user_rel = {"relation": "authoredBy"}
        self.segment_transcriber_rel = {"relation": "providedBy"}

        self.recording_label = {"attribute": "sourceId", "type": "recording"}
        self.segment_recording_rel = {"relation": "hasSource"}

        self.pim_keyphrase_label = {"attribute": "importantKeywords"}
        self.keyphrase_label = {"attribute": "segmentKeywords"}
        self.segment_keyphrase_rel = {"relation": "hasKeywords"}

        self.mind_label = {"attribute": "mindId"}
        self.context_mind_rel = {"relation": "associatedMind"}

    def read_json(self, json_file):
        with open(json_file) as f:
            meeting = js.load(f)

        return meeting

    def populate_context_info(self, request, g=None):
        if g is None:
            g = nx.DiGraph()

        context_id = request["contextId"]
        instance_id = request["instanceId"]

        g.add_nodes_from(
            [(context_id, self.context_label), (instance_id, self.instance_label)]
        )
        g.add_edges_from([(context_id, instance_id, self.context_instance_rel)])

        return g

    def populate_instance_info(self, request, g=None, attribute_dict=None):
        if g is None:
            g = nx.DiGraph()

        instance_id = request["instanceId"]
        segment_list = request["segments"]

        segment_attrs_list = []
        user_list = []
        transcriber_list = []
        recording_list = []
        segment_user_edge_list = []
        segment_transcriber_edge_list = []
        segment_recording_edge_list = []

        for segment in segment_list:
            # Add segment node and its attributes
            segment_node = segment["id"]

            segment_node_attrs = {
                "attribute": "segmentId",
                "text": segment["originalText"],
                "confidence": segment["confidence"],
                "startTime": segment["startTime"],
                "endTime": segment["endTime"],
                "duration": segment["duration"],
                "language": segment["languageCode"],
            }

            if attribute_dict is not None:
                segment_node_attrs.update(attribute_dict)

            segment_attrs_list.append((segment_node, segment_node_attrs))

            # Add userId node and its attributes
            user_node = segment["spokenBy"]
            user_list.append((user_node, self.user_label))

            # Add transcriber node and its attributes
            transcriber_node = segment["transcriber"]
            transcriber_list.append((transcriber_node, self.transcriber_label))

            # Add recording node and its attributes
            recording_node = segment["recordingId"]
            recording_list.append((recording_node, self.recording_label))

            # Create edge tuple list
            segment_user_edge_list.append(
                (segment_node, user_node, self.segment_user_rel)
            )
            segment_transcriber_edge_list.append(
                (segment_node, transcriber_node, self.segment_transcriber_rel)
            )
            segment_recording_edge_list.append(
                (segment_node, recording_node, self.segment_recording_rel)
            )

        # Add instance -> segment nodes
        g.add_nodes_from([(instance_id, self.instance_label)])
        g.add_nodes_from(segment_attrs_list)

        # Add segment -> other nodes
        g.add_nodes_from(user_list)
        g.add_nodes_from(transcriber_list)
        g.add_nodes_from(recording_list)

        # Add edges for the above nodes
        g.add_edges_from(segment_user_edge_list)
        g.add_edges_from(segment_transcriber_edge_list)
        g.add_edges_from(segment_recording_edge_list)

        # Create edge between instanceId -> segmentId and add to graph
        instance_segment_edges = [
            (instance_id, seg_id, self.instance_segment_rel)
            for seg_id, seg_attrs in segment_attrs_list
        ]
        g.add_edges_from(instance_segment_edges)

        return g

    def populate_keyphrase_info(
        self, request, keyphrase_list, g=None, is_pim=True, keyphrase_attr_dict=None
    ):
        if g is None:
            g = nx.DiGraph()

        segment_list = request["segments"]
        mind_id = request.get("mindId", "undefinedMind")
        context_id = request["contextId"]

        for segment in segment_list:
            segment_node = segment["id"]
            segment_keyphrase_edge_list = [
                (segment_node, words, self.segment_keyphrase_rel)
                for words in keyphrase_list
            ]

            g.add_edges_from(segment_keyphrase_edge_list)

        # Unload list and add the words individually in the graph
        # Check if keyphrase_attr_dict contains keyword embedding attribute. If yes, unpack it
        keyphrase_node_list = []
        if is_pim:
            keyphrase_attr_dict.update(self.pim_keyphrase_label)
        else:
            keyphrase_attr_dict.update(self.keyphrase_label)

        if (
            keyphrase_attr_dict is not None
            and keyphrase_attr_dict.get("embedding_vector") is not None
        ):
            embedding_array = keyphrase_attr_dict.get("embedding_vector")
            embedding_array = np.asarray(embedding_array)

            if len(keyphrase_list) == embedding_array.shape[0]:
                keyphrase_vector_zip = zip(keyphrase_list, embedding_array)
                for i, (word, word_vector) in enumerate(keyphrase_vector_zip):
                    attr_dict = deepcopy(keyphrase_attr_dict)
                    attr_dict["embedding_vector"] = word_vector
                    attr_dict["word"] = word
                    keyphrase_node_list.append((word, attr_dict))

                g.add_nodes_from(keyphrase_node_list)
        else:
            keyphrase_node_list = [
                (word, keyphrase_attr_dict) for word in keyphrase_list
            ]
            g.add_nodes_from(keyphrase_node_list)

        g.add_nodes_from([(mind_id, self.mind_label)])
        g.add_edges_from([(context_id, mind_id, self.context_mind_rel)])

        return g

    def populate_word_graph_info(self, request, context_graph, word_graph):
        instance_id = request["instanceId"]

        # Add word graph as a node in the context graph
        context_graph.add_node(
            word_graph,
            attribute="wordGraph",
            type="graphObject",
            graphId=word_graph.graph.get("graphId"),
            state="processing",
        )

        # Add edge between instanceId and word graph
        context_graph.add_edge(instance_id, word_graph, relation="hasWordGraph")

        return context_graph

    def query_word_graph_object(self, context_graph):
        # Get instance id for faster search
        instance_id = ""
        for n, n_attr in context_graph.nodes.data("attribute"):
            if n_attr == "instanceId":
                instance_id = n

        # Use instance Id only to search from
        for (n1, n2, e_attr) in context_graph.edges(
            data="relation", nbunch=[instance_id]
        ):
            print(n1, n2)
            if e_attr == "hasWordGraph":
                if isinstance(n2, nx.Graph):
                    logger.info("retrieved word graph object")

                    return n2
            else:
                if context_graph.nodes[n2]["attribute"] != "wordGraph":
                    continue
        logger.warning(
            "graphId does not exist or does not match context info. Returning empty graph with a reset state"
        )
        return nx.Graph(graphId=context_graph.graph.get("graphId"), state="reset")

    def query_for_embedded_nodes(self, context_graph):
        """
        Query and remove embedding vectors from keyphrase nodes
        Args:
            context_graph:

        Returns:

        """
        for node, n_attr in context_graph.nodes.data("attribute"):
            if n_attr == "segmentKeywords" or n_attr == "importantKeywords":
                context_graph.nodes[node]["embedding_vector"] = 0

        logger.info("removed embeddings vectors from keyword nodes")
        return context_graph

    def query_for_embedded_segments(self, context_graph):
        """
        Query and remove embeddings from segments
        Args:
            context_graph:

        Returns:

        """

        for node, n_attr in context_graph.nodes.data("attribute"):
            if n_attr == "segmentId":
                context_graph.nodes[node]["embedding_vector"] = 0

        logger.info("removed embeddings vectors from segment nodes")
        return context_graph
