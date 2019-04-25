
import asyncio
import signal
import uvloop
import logging
from os import getenv
from keyphrase.graphrank.extract_keyphrases import KeyphraseExtractor
from keyphrase.transport.nats import NATSTransport
from keyphrase.transport.manager import Manager
from dotenv import load_dotenv

log = logging.getLogger(__name__)

if __name__ == '__main__':
    load_dotenv()
    print("inside nats server")
    active_env = getenv("ACTIVE_ENV", "development")
    nats_url = getenv("NATS_URL", "nats://docker.for.mac.localhost:4222")
    print(nats_url)
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    keyphrase_extractor = KeyphraseExtractor()

    nats_manager = Manager(loop=loop,
                           url=nats_url,
                           queue_name="io.etherlabs.keyphrase.ether_service")
    nats_transport = NATSTransport(
        nats_manager=nats_manager,
        keyphrase_service=keyphrase_extractor,
    )

    def shutdown():
        log.info("received interrupt; shutting down")
        loop.create_task(nats_manager.close())

    loop.run_until_complete(nats_manager.connect())
    loop.run_until_complete(nats_transport.subscribe_context())

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, shutdown)
    try:
        loop.run_forever()
    finally:
        loop.close()
