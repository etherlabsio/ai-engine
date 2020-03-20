import asyncio
import json
from nats.aio.client import Client as NATS
import argparse
import os
import uvloop
import logging

NATS_URL = os.getenv("NATS_URL")
TIMEOUT = os.getenv("TIMEOUT", 20)

logger = logging.getLogger(__name__)


async def test_user_addition():
    nc = NATS()
    topic = "context.user_added"
    await nc.connect(servers=[nats_url])
    request = {
        "accID": "01E3S3V0EYXWN45R7FD142MQF7",
        "accType": "context.context",
        "at": "2020-03-20T12:17:45.679158337Z",
        "data": {
            "contextId": "01E3S3V0EYXWN45R7FD142MQF7",
            "userId": "a67e29e0-1f0e-4010-b220-7f110e22d905",
        },
    }

    await nc.publish(topic, json.dumps(request).encode())
    await nc.close()


async def test_user_deletion():
    nc = NATS()
    topic = "context.user_removed"
    await nc.connect(servers=[nats_url])
    request = {
        "accID": "01E3S3V0EYXWN45R7FD142MQF7",
        "accType": "context.context",
        "at": "2020-03-20T12:17:45.679158337Z",
        "data": {
            "contextId": "01E3S3V0EYXWN45R7FD142MQF7",
            "userId": "a67e29e0-1f0e-4010-b220-7f110e22d905",
        },
    }

    await nc.publish(topic, json.dumps(request).encode())
    await nc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="topic arguments for keyphrase_service"
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=str,
        default="add",
        help="define nats topics for the ether-graph service to listen to",
    )
    parser.add_argument(
        "-n", "--nats_url", type=str, default=NATS_URL, help="nats server url address",
    )
    parser.add_argument(
        "-ti",
        "--timeout",
        type=int,
        default=TIMEOUT,
        help="specify NATS reply timeout in sec",
    )

    args = parser.parse_args()

    nats_url = args.nats_url
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    if args.topics == "add":
        t1 = loop.run_until_complete(test_user_addition())
    elif args.topics == "remove":
        loop.run_until_complete(test_user_deletion())
    else:
        loop.close()
