import networkx as nx
import json as js
from typing import List, Dict
import pydgraph


class DgraphETL(object):
    def __init__(self):
        self.list_type_edges = ["hasMeeting", "hasSegment", "hasKeywords"]
        self.keyword_node = ["importKeywords"]
        self.schema_type = {
            "contextId": "Context",
            "instanceId": "Instance",
            "segmentId": "Segment",
            "authorId": "User",
            "mindId": "Mind",
            "workspaceId": "Workspace",
            "channelId": "Channel",
            "sourceId": "Source",
            "segmentProvider": "Provider",
            "importantKeywords": "Keyword",
        }

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
                        edge_rel_list.append({"uid": "_:" + target, "name": target})

                    target_spec[relation_key] = edge_rel_list
                else:
                    target_spec = {relation_key: {"uid": "_:" + target, "name": target}}

                target_spec_dict = {**target_spec, **target_spec_dict}

            dgraph_node_edge_dict = {**dgraph_source_node_dict, **target_spec_dict}
            dgraph_compat_data.append(dgraph_node_edge_dict)

        return dgraph_compat_data


class DgraphSchema(object):
    def set_schema(self):

        ether_schema = """
        
        type Context {
            name: string
            attribute: string
            hasMeeting: [Instance]
            associatedMind: Mind
        }
        
        type Mind {
            name: string
            attribute: string
        }
        
        type Instance {
            name: string
            attribute: string
            hasSegment: [Segment]
        }
        
        type Segment {
            name: string
            attribute: string
            text: string
            confidence: float
            language: string
            startTime: datetime
            endTime: datetime
            duration: int
            authoredBy: [User]
            hasKeywords: [Keyword]
            hasSource: [Source]
            providedBy: [Provider]    
        }
        
        type User {
            name: string
            attribute: string
        }
        
        type Source {
            name: string
            attribute: string
            type: string
        }
        
        type Provider {
            name: string
            attribute: string
        }
        
        type Keyword {
            keyphrase: string
            attribute: string
            important: bool
        }
        
        name: string @index(exact) .
        attribute: string @index(hash) .
        text: string @index(fulltext) .
        keyphrase: string @index(term) .
        associatedMind: uid .
        hasMeeting: [uid] @reverse .
        hasSegment: [uid] @reverse .
        hasKeywords: [uid] .
        authoredBy: uid @reverse .
        
        """
