import logging
import json as js
import pydgraph
from os import getenv

from context_parser import ContextSessionParser
from schema import Schema

logger = logging.getLogger(__name__)
DGRAPH_URL = getenv("DGRAPH_URL", "localhost:9080")


class GraphHandler(object):
    def __init__(self, dgraph_client):
        self.dgraph = dgraph_client
        self.context_parser = ContextSessionParser()
        self.context_schema = Schema()

        self.context_instance_rel = "hasMeeting"
        self.instance_segment_rel = "hasSegment"
        self.segment_user_rel = "authoredBy"
        self.segment_transcriber_rel = "providedBy"
        self.segment_recording_rel = "hasSource"
        self.segment_keyphrase_rel = "hasKeywords"
        self.context_mind_rel = "associatedMind"

        client_stub = pydgraph.DgraphClientStub(DGRAPH_URL)
        client = pydgraph.DgraphClient(client_stub)

        self.dgraph = client

    # For testing purposes
    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

    def read_json(self, json_file):
        with open(json_file) as f_:
            meeting = js.load(f_)
        return meeting

    def query_transform_node(self, xid, node_obj, extra_field=None):
        """
        Given an xid a query request for UID is made and given a node object, transform the node object to use the UID
        Args:
            response:
            node:

        Returns:

        """
        response = self._query_uid(xid=xid)

        # This is useful for predicates that do not have the standard `xid` attribute
        if extra_field is not None:
            ext_query = """query q($i: string) {
                    q(func: eq(name, $i)) {
                        uid
                        attribute
                        xid
                    }
                }"""
            response = self._query_uid(xid=extra_field, ext_query=ext_query)

        try:
            node_uid = response["q"][0].get("uid")
            logger.info(
                "Received response", extra={"response": response, "uid": node_uid}
            )
            node_obj["uid"] = node_uid
        except IndexError:
            logger.warning("No UID found", extra={"response": response})

        return node_obj

    def set_schema(self):
        schema = self.context_schema.fetch_schema()
        return self.dgraph.alter(pydgraph.Operation(schema=schema))

    def populate_context_info(self, req_data, **kwargs):
        context_node, instance_node, mind_node = self.context_parser.parse_context_info(
            req_data=req_data
        )

        context_id = context_node["xid"]
        instance_id = instance_node["xid"]
        mind_id = mind_node["xid"]

        context_node = self.query_transform_node(xid=context_id, node_obj=context_node)
        instance_node = self.query_transform_node(
            xid=instance_id, node_obj=instance_node
        )
        mind_node = self.query_transform_node(xid=mind_id, node_obj=mind_node)

        context_node.update({self.context_instance_rel: instance_node})
        context_node.update({self.context_mind_rel: mind_node})
        mutation_query_obj = context_node

        resp = self._mutate_info(mutation_query=mutation_query_obj)

        # To check how the JSON looks
        # self.to_json(context_node, "context")
        return resp

    def populate_instance_segment_info(self, req_data, **kwargs):
        instance_node, segment_node = self.context_parser.parse_instance_segment_info(
            req_data=req_data
        )

        instance_id = instance_node["xid"]
        segment_id = segment_node["xid"]
        instance_node = self.query_transform_node(
            xid=instance_id, node_obj=instance_node
        )
        segment_node = self.query_transform_node(xid=segment_id, node_obj=segment_node)

        instance_node.update({self.instance_segment_rel: segment_node})
        mutation_query_obj = instance_node

        resp = self._mutate_info(mutation_query_obj)

        # To check how the JSON looks
        # self.to_json(instance_node, "instance_seg")

        return resp

    def populate_segment_info(self, req_data, **kwargs):
        segment_object = req_data["segments"]
        segment_node, user_node, provider_node, recorder_node = self.context_parser.parse_segment_info(
            segment_object=segment_object, **kwargs
        )

        segment_id = segment_node["xid"]
        user_id = user_node["xid"]
        recorder_id = recorder_node["xid"]
        provider_name = provider_node["xid"]

        segment_node = self.query_transform_node(xid=segment_id, node_obj=segment_node)
        user_node = self.query_transform_node(xid=user_id, node_obj=user_node)
        recorder_node = self.query_transform_node(
            xid=recorder_id, node_obj=recorder_node
        )
        provider_node = self.query_transform_node(
            xid=provider_name, node_obj=provider_node, extra_field=None
        )

        segment_node.update(
            {
                self.segment_user_rel: user_node,
                self.segment_recording_rel: recorder_node,
                self.segment_transcriber_rel: provider_node,
            }
        )

        mutation_query_obj = segment_node
        resp = self._mutate_info(mutation_query=mutation_query_obj)

        return resp

    def populate_keyphrase_info(self, req_data, **kwargs):
        segment_object = req_data["segments"]
        segment_node, user_node, provider_node, recorder_node = self.context_parser.parse_segment_info(
            segment_object=segment_object, **kwargs
        )

        keyphrase_node = self.context_parser.parse_keyphrase_info(
            segment_object=segment_object
        )
        segment_id = segment_node["xid"]
        keyphrase_id = keyphrase_node["xid"]

        segment_node = self.query_transform_node(xid=segment_id, node_obj=segment_node)
        keyphrase_node = self.query_transform_node(
            xid=keyphrase_id, node_obj=keyphrase_node
        )
        segment_node.update({self.segment_keyphrase_rel: keyphrase_node})

        mutation_query_obj = segment_node
        resp = self._mutate_info(mutation_query=mutation_query_obj)

        # self.to_json(segment_node, "seg")

        return resp

    def populate_marker_info(self, req_data):
        pass

    def _query_uid(self, xid, ext_query=None):
        txn = self.dgraph.txn()
        try:
            query = """query q($i: string) {
                    q(func: eq(xid, $i)) {
                        uid
                        attribute
                        xid
                    }
                }"""
            if ext_query is not None:
                query = ext_query

            variables = {"$i": xid}
            res = self.dgraph.txn(read_only=True).query(query, variables=variables)
            response = js.loads(res.json)

            return response
        finally:
            txn.discard()

    def _mutate_info(self, mutation_query):
        txn = self.dgraph.txn()
        try:
            # Run mutation.
            response = txn.mutate(set_obj=mutation_query)

            # Commit transaction.
            txn.commit()

            return response
        finally:
            # Clean up. Calling this after txn.commit() is a no-op and hence safe.
            txn.discard()

    def perform_queries(self, query_text, variables=None):
        txn = self.dgraph.txn()
        try:
            query = query_text

            if variables is not None:
                res = self.dgraph.txn(read_only=True).query(query, variables=variables)
            else:
                res = self.dgraph.txn(read_only=True).query(query)

            response = js.loads(res.json)
            return response
        finally:
            txn.discard()


# For testing locally
if __name__ == "__main__":
    gh = GraphHandler(dgraph_client="")

    req_data = gh.read_json("meeting_test.json")

    # Execute one-by-one in sequence

    gh.set_schema()
    # gh.populate_context_info(req_data)
    # gh.populate_instance_segment_info(req_data)
    # gh.populate_segment_info(req_data)
