import asyncio
import uvloop
import argparse
import signal
import structlog

import settings
from graphrank.extract_keyphrases import ExtractKeyphrase
from transport.nats_service import NATSTransport
from transport.manager import Manager
# from pkg.log import logger


log = structlog.getLogger(__name__)

NATS_URL = settings.NATS_URL
ACTIVE_ENV = settings.ACTIVE_ENV
DEFAULT_ENV = settings.DEFAULT_ENV


def run_nats_listener(args):
    loop = asyncio.get_event_loop()
    queueName = "io.etherlabs.ether.keyphrase_service"
    keyphrase_object = ExtractKeyphrase()
    nats_manager = Manager(loop=loop, url=args.nats_url, queueName=queueName)
    n = NATSTransport(nats_manager=nats_manager, keyphrase_object=keyphrase_object, logger=log)

    def shutdown():
        log.info("received interrupt; shutting down")
        loop.create_task(nats_manager.close())

    loop.run_until_complete(nats_manager.connect())
    loop.run_until_complete(n.subscribe_context())

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, shutdown)

    try:
        loop.run_forever()
    finally:
        loop.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for keyphrase_service')
    parser.add_argument("--nats_url", type=str, default=NATS_URL, help="nats server url")
    args = parser.parse_args()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # setup_logger()

    run_nats_listener(args)
