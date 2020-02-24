import networkx as nx
import json as js
from typing import List, Dict
import pydgraph
import hashlib
import os
import logging
import traceback
import simplejson as sjson

from backfill import BackFillCleanupJob
from graphio import GraphIO

logger = logging.getLogger(__name__)


class DgraphETL(object):
    def __init__(self):
        self.list_type_edges = [
            "hasMeeting",
            "hasSegment",
            "hasUser",
            "hasMember",
            "hasMarker",
        ]
        self.keyword_node = ["importKeywords"]
        self.schema_type = {
            "contextId": "Context",
            "instanceId": "ContextSession",
            "segmentId": "TranscriptionSegment",
            "userId": "User",
            "mindId": "Mind",
            "workspaceId": "Workspace",
            "channelId": "Channel",
            "sourceId": "Source",
            "segmentProvider": "TranscriptionProvider",
            "importantKeywords": "Keyphrase",
            "segmentKeywords": "Keyphrase",
            "customerId": "Customer",
            "markerId": "Marker",
        }

        self.context_label = {"attribute": "contextId"}
        self.instance_label = {"attribute": "instanceId"}
        self.segment_label = {"attribute": "segmentId"}
        self.user_label = {"attribute": "userId"}
        self.transcriber_label = {"attribute": "segmentProvider"}
        self.recording_label = {"attribute": "sourceId", "type": "recording"}
        self.pim_keyphrase_label = {"attribute": "importantKeywords"}
        self.keyphrase_label = {"attribute": "segmentKeywords"}
        self.mind_label = {"attribute": "mindId"}
        self.marker_label = {"attribute": "markerId"}

    def to_json(self, data, filename):

        try:
            with open(
                os.path.join(os.getcwd(), filename + ".json"), "w", encoding="utf-8",
            ) as f_:
                sjson.dump(data, f_, ensure_ascii=False, indent=4, ignore_nan=True)
        except Exception as e:
            print(e)
            print(traceback.print_exc())

        return filename + ".json"

    def create_client_stub(self):
        return pydgraph.DgraphClientStub("localhost:9080")

    def create_client(self, client_stub):
        return pydgraph.DgraphClient(client_stub)

    def drop_all(self, client):
        return client.alter(pydgraph.Operation(drop_all=True))

    def set_schema(self):
        pass

    def _hash_sha_object(self, data: str) -> str:
        hash_object = hashlib.sha1(data.encode())
        hash_str = hash_object.hexdigest()
        return hash_str

    def nx_dgraph(self, g: nx.DiGraph):
        node_list = []
        adj_dict = dict(g.adjacency())
        for i, (source, target) in enumerate(adj_dict.items()):
            if len(g[source]) != 0:
                source_attr = g.nodes[source]

                try:
                    dgraph_type = self.schema_type[source_attr.get("attribute")]
                except KeyError:
                    dgraph_type = "null"

                source_node = {
                    "dgraph.type": dgraph_type,
                    "uid": "_:" + source,
                    "xid": source,
                    **source_attr,
                }

                keyword_list = []
                list_type_nodes = []
                for target_node, relation_dict in target.items():
                    target_attr = g.nodes[target_node]
                    relation = relation_dict.get("relation")

                    try:
                        target_dgraph_type = self.schema_type[
                            target_attr.get("attribute")
                        ]
                    except KeyError as e:
                        target_dgraph_type = "null"
                        logger.warning(e)
                        print(traceback.print_exc())

                    if relation == "hasKeywords":
                        # keyword_list.append(target_node)
                        segment_text = source_attr.get("text", "analyzedText")
                        if target_node is not None and target_node in segment_text:
                            target_node_dict = {
                                "dgraph.type": target_dgraph_type,
                                "uid": "_:" + self._hash_sha_object(target_node),
                                "xid": self._hash_sha_object(target_node),
                                "originalForm": target_node,
                                "value": target_node.lower(),
                                **target_attr,
                                "type": "descriptive",
                            }
                            keyword_list.append(target_node_dict)
                        source_node.update({relation: keyword_list})
                    elif relation in self.list_type_edges:
                        target_node_dict = {
                            "dgraph.type": target_dgraph_type,
                            "uid": "_:" + target_node,
                            "xid": target_node,
                            **target_attr,
                        }
                        list_type_nodes.append(target_node_dict)
                        source_node.update({relation: list_type_nodes})
                    else:
                        target_node_dict = {
                            "dgraph.type": target_dgraph_type,
                            "uid": "_:" + target_node,
                            "xid": target_node,
                            **target_attr,
                        }
                        source_node.update({relation: target_node_dict})
                    # source_node.update({relation: target_node_dict})

                node_list.append(source_node.copy())

        return node_list


if __name__ == "__main__":
    dgraph = DgraphETL()
    nx_data = "meta_graph_prod.pickle"
    g = nx.read_gpickle(nx_data)

    backfill_obj = BackFillCleanupJob()
    gio = GraphIO(backfill_obj=backfill_obj)
    g = gio.cleanup_graph(graph_obj=g)

    dgraph_compat_data = dgraph.nx_dgraph(g)
    dgraph.to_json(dgraph_compat_data, filename="meta_graph_prod_2")
