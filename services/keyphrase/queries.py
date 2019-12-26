import json


class Queries(object):
    def __init__(self, nats_manager=None):
        self.nats_manager = nats_manager
        self.eg_query_topic = "ether_graph_service.perform_query"
        self.TIMEOUT = 20

    async def query_graph(self, query_object):
        msg = await self.nats_manager.conn.request(
            self.eg_query_topic, json.dumps(query_object).encode(), timeout=self.TIMEOUT
        )
        resp = json.loads(msg.data.decode())

        return resp
