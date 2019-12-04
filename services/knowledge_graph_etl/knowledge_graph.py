import networkx as nx
import json as js
import logging

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

        self.keyphrase_label = {"attribute": "importantKeywords"}
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
            [(context_id, self.context_label), (instance_id, self.instance_label),]
        )
        g.add_edges_from([(context_id, instance_id, self.context_instance_rel)])

        return g

    def populate_instance_info(self, request, g=None):
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

    def populate_keyphrase_info(self, request, keyphrase_list, g=None):

        if g is None:
            g = nx.DiGraph()

        segment_list = request["segments"]
        mind_id = request["mindId"]
        context_id = request["contextId"]

        for segment in segment_list:
            segment_node = segment["id"]
            segment_keyphrase_edge_list = [
                (segment_node, words, self.segment_keyphrase_rel)
                for words in keyphrase_list
            ]

            g.add_edges_from(segment_keyphrase_edge_list)

        # Unload list and add the words individually in the graph
        keyphrase_node_list = [
            (words, self.keyphrase_label) for words in keyphrase_list
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
        )

        # Add edge between instanceId and word graph
        context_graph.add_edge(instance_id, word_graph, relation="hasWordGraph")

        return context_graph

    def query_word_graph_object(self, context_graph):
        for (n1, n2, e_attr) in context_graph.edges.data("relation"):
            if e_attr == "hasWordGraph":
                return n2
