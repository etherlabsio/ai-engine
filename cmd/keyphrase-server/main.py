import asyncio
import signal
import uvloop
import logging
from dotenv import load_dotenv, find_dotenv
from os import getenv

from keyphrase.graphrank.extract_keyphrases import KeyphraseExtractor
from keyphrase.transport.nats import NATSTransport
from nats.manager import Manager
from log.logger import setup_server_logger
from s3client.s3 import S3Manager

logger = logging.getLogger()


if __name__ == "__main__":
    # Setup logger
    setup_server_logger(debug=True)  # default False for disabling debug mode
    load_dotenv(find_dotenv())

    active_env = getenv("ACTIVE_ENV", "development")
    nats_url = getenv("NATS_URL", "nats://localhost:4222")
    bucket_store = getenv("STORAGE_BUCKET", "io.etherlabs.staging2.contexts")

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    s3_client = S3Manager(bucket_name=bucket_store)
    keyphrase_extractor = KeyphraseExtractor(s3_client=s3_client)

    nats_manager = Manager(
        loop=loop, url=nats_url, queue_name="io.etherlabs.keyphrase_service"
    )
    nats_transport = NATSTransport(
        nats_manager=nats_manager, keyphrase_service=keyphrase_extractor
    )

    def shutdown():
        logger.info("received interrupt; shutting down")
        loop.create_task(nats_manager.close())

    loop.run_until_complete(nats_manager.connect())
    loop.run_until_complete(nats_transport.subscribe_context())

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, shutdown)
    try:
        loop.run_forever()
    finally:
        loop.close()
