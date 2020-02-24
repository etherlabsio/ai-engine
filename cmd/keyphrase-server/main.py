import asyncio
import signal
import uvloop
import logging
from dotenv import load_dotenv, find_dotenv
import os
from os import getenv
from boto3 import client
from botocore.client import Config


from keyphrase.extract_keyphrases import KeyphraseExtractor
from keyphrase.transport.nats import NATSTransport


from nats.manager import Manager
from log.logger import setup_server_logger
from s3client.s3 import S3Manager

logger = logging.getLogger()
load_dotenv(find_dotenv())


if __name__ == "__main__":

    # Setup logger
    setup_server_logger(debug=True)  # default False for disabling debug mode

    # Load ENV variables
    nats_url = getenv("NATS_URL", "nats://localhost:4222")
    bucket_store = getenv("STORAGE_BUCKET", "io.etherlabs.staging2.contexts")
    encoder_lambda_function = getenv("FUNCTION_NAME", "sentence-encoder-lambda")
    ner_lambda_function = getenv("NER_FUNCTION_NAME", "ner")
    aws_region = getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Initialize Boto session for aws services
    aws_config = Config(
        connect_timeout=180,
        read_timeout=300,
        retries={"max_attempts": 2},
        region_name=aws_region,
    )
    s3_client = S3Manager(bucket_name=bucket_store)
    lambda_client = client("lambda", config=aws_config)

    # Initialize event loop and transport layers
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    nats_manager = Manager(
        loop=loop, url=nats_url, queue_name="io.etherlabs.keyphrase_service"
    )

    # Initialize keyphrase-service client
    keyphrase_extractor = KeyphraseExtractor(
        s3_client=s3_client,
        encoder_lambda_client=lambda_client,
        lambda_function=encoder_lambda_function,
        ner_lambda_function=ner_lambda_function,
        nats_manager=nats_manager,
    )
    logger.debug("download complete")

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
