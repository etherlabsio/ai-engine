import json


class Queries(object):
    def __init__(self, nats_manager=None):
        self.nats_manager = nats_manager

    async def query_graph(self, query_object):
        eg_query_topic = "ether_graph_service.perform_query"
        TIMEOUT = 20

        msg = await self.nats_manager.conn.request(
            eg_query_topic, json.dumps(query_object).encode(), timeout=TIMEOUT
        )
        resp = msg.data.decode()

        return resp
