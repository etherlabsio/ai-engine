import asyncio
import uvloop
import signal
import logging
from dotenv import load_dotenv, find_dotenv
from os import getenv

from log.logger import setup_server_logger
from dgraph.client import DgraphClient
from dgraph.connector import Connector

from ether_graph.context_parser import ContextSessionParser
from ether_graph.manager import GraphHandler


logger = logging.getLogger()
load_dotenv(find_dotenv())


if __name__ == "__main__":

    # Setup logger
    setup_server_logger(debug=True)  # default False for disabling debug mode

    # Load ENV variables
    dgraph_client_url = getenv("DGRAPH_URL", "localhost:9080")

    # Initialize dgraph client
    connector = Connector(url=dgraph_client_url)
    dgraph_client = DgraphClient(connector=connector)

    # Initialize graph parser
    context_parser = ContextSessionParser()

    # Initialize graph handler
    graph_handler = GraphHandler(dgraph_client=dgraph_client, parser=context_parser)

    # Initialize event loop and transport layers
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    def shutdown():
        logger.info("received interrupt; shutting down")
        loop.create_task(connector.close_client())

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, shutdown)
    try:
        loop.run_forever()
    finally:
        loop.close()
