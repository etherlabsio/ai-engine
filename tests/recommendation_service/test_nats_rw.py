import asyncio
import json
from nats.aio.client import Client as NATS
import argparse
import os
import uvloop
import logging
from copy import deepcopy

NATS_URL = os.getenv("NATS_URL")
TIMEOUT = os.getenv("TIMEOUT", 120)

logger = logging.getLogger(__name__)


async def get_recommendations():
    nc = NATS()
    topic = "recommendation.service.get_watchers"
    await nc.connect(servers=[nats_url])
    test_json = read_json(test_json_file)
    topic, resp = replace_ids(
        test_json["contextId"], test_json["instanceId"], topic, resp={}
    )
    msg = await nc.request(topic, json.dumps(test_json).encode(), timeout=TIMEOUT)
    data = msg.data.decode()
    print("Received a message: {data}".format(data=data))


async def create_context():
    nc = NATS()
    topic = "context.instance.created"
    await nc.connect(servers=[nats_url])
    resp = {"contextId": "*", "instanceId": "*", "state": "created"}

    topic, resp = replace_ids(topic=topic, resp=resp)
    await nc.publish(topic, json.dumps(resp).encode())
    pass


async def start_context():
    nc = NATS()
    topic = "context.instance.started"
    await nc.connect(servers=[nats_url])
    resp = {"instanceId": "*", "state": "started", "contextId": "*"}
    topic, resp = replace_ids(topic=topic, resp=resp)
    await nc.publish(topic, json.dumps(resp).encode())
    pass


async def end_context():
    nc = NATS()
    topic = "context.instance.ended"
    await nc.connect(servers=[nats_url])
    resp = {"instanceId": "*", "state": "ended"}
    topic, resp = replace_ids(topic=topic, resp=resp)
    await nc.publish(topic, json.dumps(resp).encode())
    await nc.flush()
    await nc.close()
    pass


def read_json(json_file):
    with open(json_file) as f_:
        meeting = json.load(f_)
    return meeting


def replace_ids(context_id=None, instance_id=None, topic=None, resp=dict()):

    if context_id is None and instance_id is None:
        context_id = "01DB8DEW0YFYK0ZBP2Q3XR2YT1"
        instance_id = "b5d4"

    resp["instanceId"] = instance_id
    resp["contextId"] = context_id

    if "*" in topic:
        formatted_topic = topic.replace("*", instance_id)
    else:
        formatted_topic = topic

    return formatted_topic, resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="topic arguments for keyphrase_service"
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=str,
        default="def",
        help="define nats topics for the recommendation service to listen to",
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
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="data/keyphrase_struct.json",
        help="specify filename for meeting transcript file for population",
    )
    args = parser.parse_args()

    test_json_file = os.path.join(os.getcwd(), args.file)

    nats_url = args.nats_url
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    if args.topics == "def":
        t1 = loop.run_until_complete(create_context())
    elif args.topics == "start":
        loop.run_until_complete(start_context())
    elif args.topics == "recommend":
        loop.run_until_complete(get_recommendations())
    else:
        loop.run_until_complete(end_context())
