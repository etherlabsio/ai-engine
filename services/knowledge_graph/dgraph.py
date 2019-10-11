import networkx as nx
import json as js
from typing import List, Dict
import pydgraph


class DgraphETL(object):
    def __init__(self):
        self.list_type_edges = ["hasMeeting", "hasSegment", "hasKeywords"]

    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

    def create_client_stub(self):
        return pydgraph.DgraphClientStub("localhost:9080")

    def create_client(self, client_stub):
        return pydgraph.DgraphClient(client_stub)

    def drop_all(self, client):
        return client.alter(pydgraph.Operation(drop_all=True))

    def set_schema(self):
        pass

    def nx_to_dgraph(self, g: nx.DiGraph) -> List[dict]:

        dgraph_compat_data = []
        node_dict = dict(g.nodes())
        for source_node, node_attr in node_dict.items():
            dgraph_source_node_dict = {"uid": "_:" + source_node, "name": source_node}
            dgraph_source_node_dict.update(node_attr)

            edge_rel_list = []
            target_spec_dict = {}
            source_node_neighbours = dict(g[source_node]).items()
            for target, edge_attr in list(source_node_neighbours):

                relation_key = edge_attr.get("relation", None)
                target_spec = {relation_key: []}
                if relation_key in self.list_type_edges:
                    edge_rel_list.append({"uid": "_:" + target, "name": target})
                    target_spec[relation_key] = edge_rel_list
                else:
                    target_spec = {relation_key: {"uid": "_:" + target, "name": target}}

                target_spec_dict = {**target_spec, **target_spec_dict}

            dgraph_node_edge_dict = {**dgraph_source_node_dict, **target_spec_dict}
            dgraph_compat_data.append(dgraph_node_edge_dict)

        return dgraph_compat_data
