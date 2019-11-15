import networkx as nx
import json as js
from typing import List, Dict
import pydgraph
import hashlib
import os
import logging
import uuid

from backfill import BackFillCleanupJob
from graphio import GraphIO

logger = logging.getLogger(__name__)


class DgraphETL(object):
    def __init__(self):
        self.list_type_edges = ["hasMeeting", "hasSegment", "hasKeywords"]
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

    def to_json(self, data, filename):
        with open(
            os.getcwd() + filename + ".json", "w", encoding="utf-8"
        ) as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

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

    def nx_to_dgraph(self, g: nx.DiGraph) -> List[dict]:

        dgraph_compat_data = []
        node_dict = dict(g.nodes())
        for source_node, node_attr in node_dict.items():
            if node_attr.get("attribute") in self.keyword_node:
                dgraph_source_node_dict = {"keyphrase": source_node}
            else:
                dgraph_source_node_dict = {
                    "uid": "_:" + source_node,
                    "name": source_node,
                }
            dgraph_source_node_dict.update(node_attr)

            # Add `dgraph.type` for every uid node type
            if dgraph_source_node_dict["attribute"] in self.schema_type.keys():
                dgraph_source_node_dict.update(
                    {
                        "dgraph.type": self.schema_type[
                            dgraph_source_node_dict["attribute"]
                        ]
                    }
                )

            edge_rel_list = []
            target_spec_dict = {}
            source_node_neighbours = dict(g[source_node]).items()
            for target, edge_attr in list(source_node_neighbours):

                relation_key = edge_attr.get("relation", None)
                target_spec = {relation_key: []}
                if relation_key in self.list_type_edges:
                    if relation_key == "hasKeywords":
                        edge_rel_list.append({"keyphrase": target})
                    else:
                        edge_rel_list.append(
                            {"uid": "_:" + target, "name": target}
                        )

                    target_spec[relation_key] = edge_rel_list
                else:
                    target_spec = {
                        relation_key: {"uid": "_:" + target, "name": target}
                    }

                target_spec_dict = {**target_spec, **target_spec_dict}

            dgraph_node_edge_dict = {
                **dgraph_source_node_dict,
                **target_spec_dict,
            }
            dgraph_compat_data.append(dgraph_node_edge_dict)

        return dgraph_compat_data

    def nx_dgraph(self, g: nx.DiGraph):
        node_list = []
        adj_dict = dict(g.adjacency())
        for i, (source, target) in enumerate(adj_dict.items()):
            if len(g[source]) != 0:
                source_attr = g.nodes[source]

                try:
                    dgraph_type = self.schema_type[
                        source_attr.get("attribute")
                    ]
                except KeyError:
                    dgraph_type = "null"

                source_node = {
                    "dgraph.type": dgraph_type,
                    "uid": "_:" + source,
                    "xid": source,
                    **source_attr,
                }

                keyword_list = []
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

                    if relation == "hasKeywords":
                        keyword_list.append(target_node)
                        target_node_dict = {
                            "dgraph.type": target_dgraph_type,
                            "uid": "_:" + self._hash_sha_object(source),
                            "xid": self._hash_sha_object(source),
                            "values": keyword_list,
                            **target_attr,
                        }
                    else:
                        target_node_dict = {
                            "dgraph.type": target_dgraph_type,
                            "uid": "_:" + target_node,
                            "xid": target_node,
                            **target_attr,
                        }
                    source_node.update({relation: target_node_dict})

                node_list.append(source_node.copy())

        return node_list


if __name__ == "__main__":
    dgraph = DgraphETL()
    nx_data = "meta_graph_s2.pickle"
    g = nx.read_gpickle(nx_data)

    backfill_obj = BackFillCleanupJob()
    gio = GraphIO(backfill_obj=backfill_obj)
    g = gio.cleanup_graph(graph_obj=g)

    dgraph_compat_data = dgraph.nx_dgraph(g)
    dgraph.to_json(dgraph_compat_data, filename="meta_graph_s2")
