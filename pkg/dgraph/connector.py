import logging
import pydgraph

logger = logging.getLogger(__name__)


class Connector(object):
    def __init__(self, url="localhost:9080"):
        self.url = url
        self.client_stub = self.create_client_stub()

    # Create a client stub.
    def create_client_stub(self):
        logger.info("Connecting to client...", extra={"url": self.url})
        return pydgraph.DgraphClientStub(self.url)

    # Create a client.
    def create_client(self):
        client = pydgraph.DgraphClient(self.client_stub)

        try:
            logger.info("Created client...")
            return client
        finally:
            logger.info("Client stub connection is closed.")
            self.client_stub.close()

    def close_client(self):
        logger.info("Client stub connection is closed.")
        self.client_stub.close()
