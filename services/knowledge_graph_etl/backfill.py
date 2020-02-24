import networkx as nx
import ciso8601
import datetime
import math

# import pandas as pd
import json as js
import os
from copy import deepcopy
import logging
import uuid

user_metadata = os.getenv("USER_META", "ws-ch-relations-staging2.csv")
workspace_metadata = os.getenv("WORKSPACE_META", "user_context_associations_s2")


logger = logging.getLogger(__name__)


class BackFillCleanupJob(object):
    def __init__(self):
        # self.user_meta_df = pd.read_csv(user_metadata)
        # self.workspace_meta_df = pd.read_csv(workspace_metadata)

        self.user_attribute_namespace = [
            "name",
            "email",
            "source",
            "status",
            "avatars",
            "deleted",
            "sourceId",
            "createdAt",
            "deletedAt",
            "updatedAt",
            "accessToken",
            "mentionName",
            "teamId",
        ]

        self.user_node_namespace = ["workspaceId"]

        self.user_label = {"attribute": "userId"}

        self.workspace_label = {"attribute": "workspaceId", "name": ""}
        self.channel_label = {"attribute": "channelId", "name": ""}
        self.context_label = {"attribute": "contextId"}
        self.mind_label = {"attribute": "mindId", "name": ""}
        self.customer_label = {"attribute": "customerId"}
        self.mind_dict = {
            "01DAAPWR6W051Q9WWQY99JSGFY": {"name": "generic", "type": "domain"},
            "01DAAQY88QZB19JQZ5PRJFR76Y": {
                "name": "Software Engineering",
                "type": "domain",
            },
            "01DAAQYN9GBEBC92AYWNXEDP0C": {"name": "HR", "type": "domain"},
            "01DAATANXNRQA35E6004HB7MBN": {"name": "Marketing", "type": "domain"},
            "01DAATBC3AK1QWC5NYC5AHV2XZ": {"name": "Product", "type": "domain"},
            "01DADP74WFV607KNPCB6VVXGTG": {"name": "AI", "type": "domain"},
            "01DAAYHEKY5F4E02QVRJPTFTXV": {
                "name": "Ether Engineering",
                "type": "custom",
            },
        }

        self.user_context_rel = {"relation": "belongsTo"}
        self.context_channel_rel = {"relation": "belongsTo"}
        self.channel_workspace_rel = {"relation": "belongsTo"}
        self.workspace_customer_rel = {"relation": "belongsTo"}
        self.workspace_user_rel = {"relation": "hasMember"}
        self.context_mind_rel = {"relation": "associatedMind"}

        self.tz_attributes = [
            "createdAt",
            "deletedAt",
            "updatedAt",
            "startTime",
            "endTime",
        ]

    def format_old_labels(self, g: nx.DiGraph):
        old_label_name = "label"
        old_user_attr = "authorId"
        old_keyword_attr = "importantKeywords"

        # Change node attribute - label to attribute
        for n, attr in g.nodes.data(old_label_name):
            if attr is not None:
                g.nodes[n]["attribute"] = g.nodes[n].pop(old_label_name)

        for n, attr in g.nodes.data("attribute"):
            if attr == old_user_attr:
                g.nodes[n]["attribute"] = "userId"

        for n, attr in g.nodes.data("attribute"):
            if attr == old_keyword_attr:
                g.nodes[n]["attribute"] = "segmentKeywords"

        return g

    def convert_id_to_uuid_format(self, g: nx.DiGraph):

        user_id_dict = {}
        instance_id_dict = {}
        workspace_id_dict = {}
        segment_id_dict = {}
        recording_id_dict = {}

        for n, attr in g.nodes.data("attribute"):
            if attr == "userId":
                user_id_dict[n] = str(uuid.UUID(n))

            if attr == "instanceId":
                instance_id_dict[n] = str(uuid.UUID(n))

            if attr == "segmentId":
                segment_id_dict[n] = str(uuid.UUID(n))

            if attr == "sourceId":
                recording_id_dict[n] = str(uuid.UUID(n))

            if attr == "workspaceId":
                workspace_id_dict[n] = str(uuid.UUID(n))
            elif attr is None:
                for k, v in dict(g.adj[n]).items():
                    rel = v.get("relation")
                    if rel == "hasMember":
                        workspace_id_dict[n] = str(uuid.UUID(n))

        # Relabel user and instance nodes to UUID format
        nx.relabel_nodes(g, user_id_dict, copy=False)
        nx.relabel_nodes(g, instance_id_dict, copy=False)
        nx.relabel_nodes(g, workspace_id_dict, copy=False)
        nx.relabel_nodes(g, segment_id_dict, copy=False)
        nx.relabel_nodes(g, recording_id_dict, copy=False)

        return g

    @staticmethod
    def reformat_deprecated_attributes(g: nx.DiGraph):
        for n, attr in g.nodes.data("attribute"):
            # Remove "phraseId" and "phrase" attributes.
            # Rename "keyphraseType"
            if attr == "segmentKeywords":
                g.nodes[n]["type"] = g.nodes[n].pop("keyphraseType", "")
                try:
                    del g.nodes[n]["phraseId"]
                    del g.nodes[n]["phrase"]
                    del g.nodes[n]["word"]
                except Exception:
                    continue

            # Flatten and normalize user attributes in meta graph
            if attr == "userId":
                if g.nodes[n].get("avatars") is not None:
                    g.nodes[n]["avatars.image192"] = g.nodes[n]["avatars"].pop(
                        "image192", ""
                    )
                    g.nodes[n]["avatars.imageOriginal"] = g.nodes[n]["avatars"].pop(
                        "imageOriginal", ""
                    )
                    try:
                        del g.nodes[n]["avatars"]
                    except Exception:
                        continue

        # Remove mindId with "undefinedMind" value
        try:
            g.remove_node("undefinedMind")
            logger.warning("undefinedMind present")
        except Exception:
            pass
            # logger.warning("No undefinedMind present")

        return g

    def prepare_meta_graph(self, context_graph: nx.DiGraph = None):
        if context_graph is None:
            context_graph = nx.DiGraph(type="meta")

        user_node_list = self._prepare_user_nodes()
        (user_workspace_edge_list, user_context_edge_list,) = self._prepare_user_edges()

        N, E = self._prepare_workspace_nodes()
        workspace_node_list = N[0]
        channel_node_list = N[1]
        context_node_list = N[2]
        mind_node_list = N[3]
        customer_node_list = N[4]

        context_channel_edge_list = E[0]
        channel_workspace_edge_list = E[1]
        workspace_customer_edge_list = E[2]
        context_mind_edge_list = E[3]

        context_graph.add_nodes_from(user_node_list)
        context_graph.add_nodes_from(workspace_node_list)
        context_graph.add_nodes_from(channel_node_list)
        context_graph.add_nodes_from(context_node_list)
        context_graph.add_nodes_from(mind_node_list)
        context_graph.add_nodes_from(customer_node_list)

        context_graph.add_edges_from(user_workspace_edge_list)
        context_graph.add_edges_from(user_context_edge_list)
        context_graph.add_edges_from(context_channel_edge_list)
        context_graph.add_edges_from(channel_workspace_edge_list)
        context_graph.add_edges_from(workspace_customer_edge_list)
        context_graph.add_edges_from(context_mind_edge_list)

        return context_graph

    def _prepare_user_nodes(self):
        user_node_list = []
        for i in range(len(self.user_meta_df)):
            uinfo_attr = js.loads(self.user_meta_df["user_attributes"][i])
            u_attr = {x: uinfo_attr.pop(x, None) for x in self.user_attribute_namespace}
            u_attr.update(self.user_label)
            user_id = self.user_meta_df["user_id"][i]
            user_node_list.append((user_id, u_attr))

        return user_node_list

    def _prepare_user_edges(self):
        user_workspace_edge_list = []
        user_context_edge_list = []
        for i in range(len(self.user_meta_df)):
            uinfo_attr = js.loads(self.user_meta_df["user_attributes"][i])
            u_edge_nodes = {
                x: uinfo_attr.pop(x, None) for x in self.user_node_namespace
            }

            workspace_id = u_edge_nodes["workspaceId"]
            context_id = self.user_meta_df["context_id"][i]
            user_id = self.user_meta_df["user_id"][i]

            user_workspace_edge_list.append(
                (workspace_id, user_id, self.workspace_user_rel)
            )
            user_context_edge_list.append((user_id, context_id, self.user_context_rel))

        return user_workspace_edge_list, user_context_edge_list

    def _prepare_workspace_nodes(self):
        workspace_node_list = []
        channel_node_list = []
        context_node_list = []
        mind_node_list = []
        customer_node_list = []

        context_channel_edge_list = []
        channel_workspace_edge_list = []
        workspace_customer_edge_list = []
        context_mind_edge_list = []

        N, E = (), ()

        for i in range(len(self.workspace_meta_df)):
            w_label = deepcopy(self.workspace_label)
            c_label = deepcopy(self.channel_label)
            cont_label = deepcopy(self.context_label)
            cust_label = deepcopy(self.customer_label)
            m_label = deepcopy(self.mind_label)

            workspace_id = self.workspace_meta_df["workspace_id"][i]
            workspace_name = self.workspace_meta_df["workspace_name"][i]
            workspace_name_dict = {"name": workspace_name}

            channel_id = self.workspace_meta_df["channel_id"][i]
            channel_name = self.workspace_meta_df["channel_name"][i]
            channel_name_dict = {"name": channel_name}

            context_id = self.workspace_meta_df["context_id"][i]
            mind_id = self.workspace_meta_df["mind_id"][i]
            customer_id = self.workspace_meta_df["customer_id"][i]

            w_label.update(workspace_name_dict)
            workspace_node_list.append((workspace_id, w_label))

            c_label.update(channel_name_dict)
            channel_node_list.append((channel_id, c_label))

            context_node_list.append((context_id, cont_label))
            customer_node_list.append((customer_id, cust_label))

            current_mind_info = self.mind_dict[mind_id]
            m_label.update(current_mind_info)
            mind_node_list.append((mind_id, m_label))

            # Prepare Edge list

            context_channel_edge_list.append(
                (context_id, channel_id, self.context_channel_rel)
            )
            channel_workspace_edge_list.append(
                (channel_id, workspace_id, self.channel_workspace_rel)
            )
            workspace_customer_edge_list.append(
                (workspace_id, customer_id, self.workspace_customer_rel)
            )
            context_mind_edge_list.append((context_id, mind_id, self.context_mind_rel))

            N = (
                workspace_node_list,
                channel_node_list,
                context_node_list,
                mind_node_list,
                customer_node_list,
            )
            E = (
                context_channel_edge_list,
                channel_workspace_edge_list,
                workspace_customer_edge_list,
                context_mind_edge_list,
            )

        return N, E

    def cleanup_nx_data_types(self, graph):
        """
        Checks if there is any numpy array or dict or any
        other data types are present in the graph object and
        removes it before converting to GraphML
        Returns:
            graph (nx.DiGraph)

        """
        graph = BackFillCleanupJob.reformat_deprecated_attributes(g=graph)
        graph = BackFillCleanupJob.remove_word_graph_object(context_graph=graph)
        graph = BackFillCleanupJob.remove_embedded_nodes(context_graph=graph)
        graph = BackFillCleanupJob.remove_embedded_segments(context_graph=graph)
        graph = self.reformat_datetime(context_graph=graph)
        graph = self.handle_nonetype_values(graph)

        return graph

    @staticmethod
    def remove_word_graph_object(context_graph):
        # Get instance id for faster search
        instance_id = ""
        for n, n_attr in context_graph.nodes.data("attribute"):
            if n_attr == "instanceId":
                instance_id = n

        # Use instance Id only to search from
        for (n1, n2, e_attr) in context_graph.edges(
            data="relation", nbunch=[instance_id]
        ):
            if e_attr == "hasWordGraph":
                if isinstance(n2, nx.Graph):

                    context_graph.remove_node(n2)
                    return context_graph
            else:
                if context_graph.nodes[n2]["attribute"] != "wordGraph":
                    continue
        return context_graph

    @staticmethod
    def remove_embedded_nodes(context_graph):
        """
        Query and remove embedding vectors from keyphrase nodes
        Args:
            context_graph:

        Returns:

        """
        for node, n_attr in context_graph.nodes.data("attribute"):
            if n_attr == "segmentKeywords" or n_attr == "importantKeywords":
                try:
                    del context_graph.nodes[node]["embedding_vector"]
                except Exception:
                    continue

        return context_graph

    @staticmethod
    def remove_embedded_segments(context_graph):
        """
        Query and remove embeddings from segments
        Args:
            context_graph:

        Returns:

        """

        for node, n_attr in context_graph.nodes.data("attribute"):
            if n_attr == "segmentId":
                try:
                    del context_graph.nodes[node]["embedding_vector"]
                except Exception:
                    continue

        return context_graph

    def reformat_datetime(self, context_graph):
        """
        Query and remove embeddings from segments
        Args:
            context_graph:

        Returns:

        """

        for node, n_attr in context_graph.nodes.data():
            for k, v in n_attr.items():
                if k in self.tz_attributes:
                    try:
                        t = context_graph.nodes[node][k]

                        if t is None:
                            context_graph.nodes[node][k] = t
                        else:
                            ts = ciso8601.parse_datetime(t)
                            ts_iso = ts.isoformat()
                            context_graph.nodes[node][k] = ts_iso
                    except Exception as e:
                        logger.warning(e)
                        continue

        return context_graph

    def handle_nonetype_values(self, g: nx.DiGraph):

        # Remove null or None values in attributes
        for n, attr in g.nodes.data():
            for k, v in attr.items():
                if v == "":
                    g.nodes[n][k] = None
                elif v in ["nan", "NaN"]:
                    print(k, v)
                    nan_bool = math.isnan(v)
                    if nan_bool:
                        g.nodes[n][k] = None

        return g
