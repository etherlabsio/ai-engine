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
    test_json = get_slack_keyphrases(test_json, v2=use_version2)
    msg = await nc.request(topic, json.dumps(test_json).encode(), timeout=TIMEOUT)
    data = msg.data.decode()
    print("Received a message: {data}".format(data=data))


async def create_context():
    nc = NATS()
    topic = "context.instance.created"
    test_json = read_json(test_json_file)
    await nc.connect(servers=[nats_url])
    if use_version2:
        test_json["instanceId"] = test_json["instanceId"] + "_v2"
    resp = {
        "contextId": test_json["contextId"],
        "instanceId": test_json["instanceId"],
        "state": "created",
    }

    await nc.publish(topic, json.dumps(resp).encode())
    pass


async def start_context():
    nc = NATS()
    topic = "context.instance.started"
    await nc.connect(servers=[nats_url])
    test_json = read_json(test_json_file)
    if use_version2:
        test_json["instanceId"] = test_json["instanceId"] + "_v2"
        topic = topic + ".v2"
    resp = {
        "instanceId": test_json["instanceId"],
        "state": "started",
        "contextId": test_json["contextId"],
        "extra_options": test_json["extra_options"],
    }
    await nc.publish(topic, json.dumps(resp).encode())
    pass


async def end_context():
    nc = NATS()
    topic = "context.instance.ended"
    await nc.connect(servers=[nats_url])
    test_json = read_json(test_json_file)
    if use_version2:
        test_json["instanceId"] = test_json["instanceId"] + "_v2"
    resp = {
        "instanceId": test_json["instanceId"],
        "state": "ended",
        "contextId": test_json["contextId"],
    }
    await nc.publish(topic, json.dumps(resp).encode())
    await nc.flush()
    pass


def read_json(json_file):
    with open(json_file) as f_:
        meeting = json.load(f_)
    return meeting


def get_slack_keyphrases(test_json: dict, v2=False):
    query_keywords = [w for w in slack_input.split(", ")]
    test_json["keyphrases"] = query_keywords
    if v2:
        test_json["instanceId"] = test_json["instanceId"] + "_v2"

    return test_json


def replace_ids(context_id=None, instance_id=None, topic=None, resp=dict()):

    if context_id is None and instance_id is None:
        context_id = "01DBB3SN99AVJ8ZWJDQ57X9TGX"
        instance_id = "b5d4"

    resp["instanceId"] = instance_id
    resp["contextId"] = context_id

    if "*" in topic:
        formatted_topic = topic.replace("*", instance_id)
    else:
        formatted_topic = topic

    return formatted_topic, resp


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    parser.add_argument(
        "--version2",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use alt version 2",
    )
    args = parser.parse_args()

    test_json_file = os.path.join(os.getcwd(), args.file)
    use_version2 = args.version2

    nats_url = args.nats_url
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()

    slack_input = "call at SRI, KP, further open build, domain mind Etc"

    if args.topics == "def":
        t1 = loop.run_until_complete(create_context())
    elif args.topics == "start":
        loop.run_until_complete(start_context())
    elif args.topics == "recommend":
        loop.run_until_complete(get_recommendations())
    else:
        loop.run_until_complete(end_context())
