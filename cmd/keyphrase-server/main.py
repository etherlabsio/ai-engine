from transport.http import app
from transport.nats_service import NATSHandler
import threading
import asyncio
import uvloop
import os
import argparse
import signal
import structlog

import settings


log = structlog.getLogger(__name__)

NATS_URL = settings.NATS_URL
ACTIVE_ENV = settings.ACTIVE_ENV
DEFAULT_ENV = settings.DEFAULT_ENV


def run_nats_listener(args):
    loop = asyncio.get_event_loop()
    # n = NATSTransport(loop, url=args.nats_url)
    n = NATSHandler(loop, args.nats_url)

    def shutdown():
        log.info("received interrupt; shutting down")
        loop.create_task(n.nats_manager.close())

    loop.run_until_complete(n.nats_manager.connect())
    loop.run_until_complete(n.subscribe_context())

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, shutdown)

    try:
        loop.run_forever()
    finally:
        loop.close()


def run_http_server():
    loop = asyncio.get_event_loop()
    app.run(host='0.0.0.0', port=7070)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for keyphrase_service')
    parser.add_argument("--nats_url", type=str, default=NATS_URL, help="nats server url")
    args = parser.parse_args()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # setup_logger()

    if ACTIVE_ENV == DEFAULT_ENV:
        run_http_server()
    else:
        run_nats_listener(args)
