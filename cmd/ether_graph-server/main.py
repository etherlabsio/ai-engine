import asyncio
import uvloop
import signal
import logging
from dotenv import load_dotenv, find_dotenv
from os import getenv

from log.logger import setup_server_logger

# from dgraph.client import DgraphClient
# from dgraph.connector import Connector
from nats.manager import Manager

from ether_graph.transport.nats import NATSTransport
from ether_graph.context_parser import GraphPopulator


logger = logging.getLogger()
load_dotenv(find_dotenv())


if __name__ == "__main__":

    # Setup logger
    setup_server_logger(debug=True)  # default False for disabling debug mode

    # Load ENV variables
    nats_url = getenv("NATS_URL", "nats://localhost:4222")
    dgraph_client_url = getenv(
        "DGRAPH_ENDPOINT", "dgraph-0.staging2.internal.etherlabs.io:9080"
    )

    # # Initialize dgraph client
    # connector = Connector(url=dgraph_client_url)
    # dgraph_client = DgraphClient(connector=connector)

    # Initialize event loop and transport layers
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    nats_manager = Manager(
        loop=loop, url=nats_url, queue_name="io.etherlabs.ether_graph_service"
    )

    # Initialize graph handler
    graph_populator = GraphPopulator(dgraph_url=dgraph_client_url)
    nats_transport = NATSTransport(
        nats_manager=nats_manager, eg_service=graph_populator
    )

    def shutdown():
        logger.info("received interrupt; shutting down")
        loop.create_task(graph_populator.close_client())
        loop.create_task(nats_manager.close())

    loop.run_until_complete(nats_manager.connect())
    loop.run_until_complete(nats_transport.subscribe_context())

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, shutdown)
    try:
        loop.run_forever()
    finally:
        loop.close()
