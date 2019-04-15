from transport.http import app
from transport.nats import NATSTransport
import threading
import asyncio
import uvloop
import os
import argparse
import signal
import structlog

log = structlog.getLogger(__name__)


def run_nats_listener(args):
    loop = asyncio.get_event_loop()
    n = NATSTransport(loop, url=args.nats_url)

    def shutdown():
        log.info("received interrupt; shutting down")
        loop.create_task(n.close())

    loop.run_until_complete(n.connect())
    loop.run_until_complete(n.subscribe())

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, shutdown)

    try:
        loop.run_forever()
    finally:
        loop.close()


def run_http_server():
    loop = asyncio.get_event_loop()
    app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for keyphrase_service')
    parser.add_argument("--nats_url", type=str, default="nats://localhost:4222", help="nats server url")
    args = parser.parse_args()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    devEnv = "dev"
    active_env = os.environ.get("ACTIVE_ENV", devEnv)

    # setup_logger()

    if active_env == devEnv:
        run_http_server()
    else:
        run_nats_listener(args)
