import logging
import json as js

from .context_parser import ContextSessionParser

logger = logging.getLogger(__name__)


class GraphHandler(object):
    def __init__(self, dgraph_client):
        self.dgraph = dgraph_client
        self.context_parser = ContextSessionParser()

    # For testing purposes
    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

    def read_json(self, json_file):
        with open(json_file) as f_:
            meeting = js.load(f_)
        return meeting

    def populate_context_info(self, req_data, **kwargs):
        context_node = self.context_parser.parse_context_info(req_data=req_data)
        self.to_json(context_node, "context")

    def populate_segment_info(self, req_data, **kwargs):
        instance_node, instance_segment_relation, segment_node = self.context_parser.parse_instance_segment_info(
            req_data=req_data
        )

        instance_node.update({instance_segment_relation: segment_node})
        self.to_json(instance_node, "instance_seg")

    def populate_keyphrase_info(self, req_data):
        pass

    def populate_marker_info(self, req_data):
        pass

    def query_instance_uid(self, instance_id):
        pass

    def query_context_uid(self, context_id):
        pass

    def query_segment_uid(self, segment_id):
        pass
