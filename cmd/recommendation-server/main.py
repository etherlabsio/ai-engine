import asyncio
import signal
import uvloop
import logging
from dotenv import load_dotenv, find_dotenv
import os
from os import getenv
from boto3 import client
from botocore.client import Config

from recommendation.transport.nats import NATSTransport
from recommendation.watchers import RecWatchers
from recommendation.vectorize import Vectorizer

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
    active_env = None
    encoder_lambda_function = getenv(
        "FUNCTION_NAME", "sentence-encoder-lambda"
    )
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
        loop=loop,
        url=nats_url,
        queue_name="io.etherlabs.recommendation_service",
    )

    reference_user_file = "reference_prod_user.json"
    reference_user_vector_data = "reference_user_text_vector.pickle"
    reference_user_kw_vector_data = "reference_user_kw_vector.pickle"

    ether_demo_reference_user_file = "ether_demo_reference_users.json"
    ether_demo_reference_text_vector_data = (
        "reference_ether_demo_user_text_vector.pickle"
    )
    ether_demo_reference_kw_vector_data = (
        "reference_ether_demo_user_vector.pickle"
    )

    vectorizer = Vectorizer(
        lambda_client=lambda_client, lambda_function=encoder_lambda_function
    )

    if active_env is None:
        active_env = bucket_store.split(".")[2]

    logger.debug("Active env", extra={"activeEnv": active_env})

    if active_env == "test":
        web_hook_url = "https://hooks.slack.com/services/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"
    elif active_env == "staging2":
        web_hook_url = "https://hooks.slack.com/services/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"
    elif active_env == "production":
        web_hook_url = "https://hooks.slack.com/services/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"
    else:
        web_hook_url = "https://hooks.slack.com/services/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"

    if active_env == "test":
        rec_object = RecWatchers(
            ether_demo_reference_user_file,
            ether_demo_reference_text_vector_data,
            ether_demo_reference_kw_vector_data,
            vectorizer=vectorizer,
            s3_client=s3_client,
            web_hook_url=web_hook_url,
            active_env_ab_test=active_env,
        )
    else:
        rec_object = RecWatchers(
            reference_user_file,
            reference_user_vector_data,
            reference_user_kw_vector_data,
            vectorizer=vectorizer,
            s3_client=s3_client,
            web_hook_url=web_hook_url,
            active_env_ab_test=active_env,
        )

    nats_transport = NATSTransport(
        nats_manager=nats_manager,
        watcher_service=rec_object,
        meeting_service=None,
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
