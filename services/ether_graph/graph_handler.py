import logging

from .context_parser import ContextSessionParser

logger = logging.getLogger(__name__)


class GraphHandler(object):
    def __init__(self, dgraph_client):
        self.dgraph = dgraph_client
        self.context_parser = ContextSessionParser()

    def populate_context_info(self, req_data):
        pass

    def populate_segment_info(self, req_data):
        pass

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
