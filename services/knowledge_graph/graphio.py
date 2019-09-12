import networkx as nx
import sys
import logging

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)


class GraphIO(object):
    def __init__(self, s3_client=None):
        self.s3_client = s3_client

    def write_to_pickle(
        self, graph_obj, filename=None, protocol=pickle.HIGHEST_PROTOCOL
    ) -> str:
        """

        Args:
            graph_obj: A NetworkX graph object
            filename (str): Filename in .pickle or .gz, .bz2 format if needed to be stored locally. Defaults to None
            protocol: pickle's data stream format.

        Returns:
            If `filename` is None: pickled representation of the object as a string, instead of writing it to a file.
            Else writes to a file.
        """
        if filename is not None:
            pickle.dump(obj=graph_obj, file=filename, protocol=protocol)

        s = pickle.dumps(obj=graph_obj, protocol=protocol)

        return s

    def load_graph_from_pickle(self, byte_string: bytes, filename=None):
        """

        Args:
            filename (str): Filename ending in `.pkl`, `.gpickle` or `.gz, .bz2`. Defaults to None
            byte_string (bytearray): Pickled bytes stream object

        Returns:
            graph_obj: Returns a NetworkX graph
        """
        try:
            if filename is not None:
                graph_obj = nx.read_gpickle(path=filename)
            else:
                graph_obj = pickle.loads(byte_string)

            return graph_obj

        except Exception as e:
            logger.error("Could not load graph object from file", extra={"err": e})
            graph_obj = nx.DiGraph()
            return graph_obj

    def cleanup_graph(self, graph_obj):
        graph_obj = GraphTransforms.cleanup_nx_data_types(graph=graph_obj)

        return graph_obj

    def convert_pickle_to_graphml(self, graph_pickle: bytes, output_filename: str):
        graph_obj = self.load_graph_from_pickle(byte_string=graph_pickle)

        try:
            processed_graph = self.cleanup_graph(graph_obj=graph_obj)
            nx.write_graphml_lxml(processed_graph, output_filename)

        except Exception as e:
            logger.error(
                "Could not convert graph object to GraphML format",
                extra={"err": e, "filename": output_filename},
            )

        return output_filename

    # S3 storage utility functions

    def upload_s3(self, file_name, s3_path):

        resp = self.s3_client.upload_object(body=file_name, s3_key=s3_path)
        if resp:
            return True
        else:
            return False

    def download_s3(self, s3_path):

        file_obj = self.s3_client.download_file(file_name=s3_path)
        file_obj_bytestring = file_obj["Body"].read()

        logger.info("Downloaded graph object from s3")

        return file_obj_bytestring


class GraphTransforms(object):
    def __init__(self):
        try:
            import xml.etree.ElementTree as et
        except ImportError:
            msg = "GraphML transformer requires xml.elementtree.ElementTree"
            raise ImportError(msg)

        self.et = et

    @staticmethod
    def remove_word_graph_object(context_graph):
        for (n1, n2, e_attr) in context_graph.edges.data("relation"):
            if e_attr == "hasWordGraph":
                if isinstance(n2, nx.Graph):
                    logger.debug("retrieved word graph object")

                    context_graph.remove_node(n2)
                    return context_graph
                else:
                    logger.error(
                        "graphId does not exist or does not match context info"
                    )
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
                    assert context_graph.nodes[node]["embedding_vector"] == 0
                except AssertionError:
                    context_graph.nodes[node]["embedding_vector"] = 0
                    continue

        logger.info("removed embeddings vectors from keyword nodes")
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
                    assert context_graph.nodes[node]["embedding_vector"] == 0
                except AssertionError:
                    context_graph.nodes[node]["embedding_vector"] = 0
                    continue

        logger.info("removed embeddings vectors from segment nodes")
        return context_graph

    @staticmethod
    def cleanup_nx_data_types(graph):
        """
        Checks if there is any numpy array or dict or any other data types are present in the graph object and
        removes it before converting to GraphML
        Returns:
            graph (nx.DiGraph)

        """
        graph = GraphTransforms.remove_word_graph_object(context_graph=graph)
        graph = GraphTransforms.remove_embedded_nodes(context_graph=graph)
        graph = GraphTransforms.remove_embedded_segments(context_graph=graph)

        return graph

    @staticmethod
    def fixtag(ns, tag):
        return "{" + ns + "}" + tag

    @staticmethod
    def graphml_tag(tag):
        graphml_ns = "http://graphml.graphdrawing.org/xmlns"
        if tag.startswith("{" + graphml_ns + "}"):
            return tag
        else:
            return GraphTransforms.fixtag(graphml_ns, tag)

    @staticmethod
    def py_compat_str(encoding, data):
        if sys.hexversion >= 0x3000000:
            return data.encode(encoding).decode("utf-8")
        else:
            return data.encode(encoding)

    def get_key_dict(self, f, tag="key"):
        key_dict = {}
        context = iter(self.et.iterparse(f, events=("start", "end")))
        _, root = next(context)  # get root element
        for event, elem in context:
            if event == "end" and GraphTransforms.graphml_tag(
                elem.tag
            ) == GraphTransforms.graphml_tag(tag):
                # Get a map of original id's to transformed ids

                new_id = elem.attrib["attr.name"]
                key_dict[elem.attrib["id"]] = new_id
                elem.set("id", new_id)

        return key_dict

    def get_key_elements(self, elem, tag="key"):
        if GraphTransforms.graphml_tag(elem.tag) == GraphTransforms.graphml_tag(tag):
            # Replace elem's id with its name
            new_id = elem.attrib["attr.name"]
            elem.set("id", new_id)

    def modify_graph_attribute_keys(self, elem, key_dict, tag="graph"):
        if GraphTransforms.graphml_tag(elem.tag) == GraphTransforms.graphml_tag(tag):
            for data in elem:
                if GraphTransforms.graphml_tag(data.tag) == GraphTransforms.graphml_tag(
                    "data"
                ):
                    original_attr_val = data.attrib.get("key")
                    new_graph_key = key_dict[original_attr_val]
                    data.set("key", new_graph_key)

    def modify_node_attribute_keys(self, elem, key_dict, tag="node"):
        if GraphTransforms.graphml_tag(elem.tag) == GraphTransforms.graphml_tag(tag):
            for data in elem:
                original_attr_val = data.attrib.get("key")
                data.set("key", key_dict[original_attr_val])

    def modify_edge_attribute_keys(self, elem, key_dict, tag="edge"):
        if GraphTransforms.graphml_tag(elem.tag) == GraphTransforms.graphml_tag(tag):
            for data in elem:
                original_attr_val = data.attrib.get("key")
                data.set("key", key_dict[original_attr_val])

    def graphml_transformer(self, graphml_file, out_graphml_file=None):

        if out_graphml_file is None:
            out_graphml_file = graphml_file + ".graphml"

        # Get key maps
        key_dict = self.get_key_dict(graphml_file)
        with open(graphml_file, "r") as f:
            context = iter(self.et.iterparse(f, events=("start", "end")))

            # get root element
            _, root = next(context)
            for event, elem in context:
                if event == "end":
                    # Set new key-ids
                    self.get_key_elements(elem)

                    # Set new graph id
                    self.modify_graph_attribute_keys(elem, key_dict)

                    # Set new keys for node attributes
                    self.modify_node_attribute_keys(elem, key_dict)

                    # Set new keys for edge attributes
                    self.modify_edge_attribute_keys(elem, key_dict)

                with open(out_graphml_file, "wb") as f_:
                    f_.write(self.et.tostring(elem, encoding="utf-8"))

            # Free up memory for large graphs
            root.clear()

        return out_graphml_file
