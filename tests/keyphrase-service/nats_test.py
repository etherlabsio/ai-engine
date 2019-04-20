import asyncio
import json
from nats.aio.client import Client as NATS
from nats.aio.utils import new_inbox
import argparse
import os
import uvloop
from dotenv import load_dotenv

load_dotenv()

ACTIVE_ENV = os.getenv("ACTIVE_ENV")
NATS_URL = os.getenv("NATS_URL")
DEFAULT_ENV = os.getenv("DEF_ENV")


async def publish_keyphrase():
    nc = NATS()
    topic = "keyphrase_service.in5627.extract_keyphrases"
    await nc.connect(servers=[nats_url])
    test_json = read_json(single_json_file)
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    # await nc.close()


async def publish_chapter_keyphrase():
    nc = NATS()
    topic = "keyphrase_service.in5627.extract_keyphrases"
    await nc.connect(servers=[nats_url])
    test_json = read_json(multi_json_file)
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def publish_instance_keyphrase():
    nc = NATS()
    topic = "keyphrase_service.in5627.keyphrases_for_context_instance"
    await nc.connect(servers=[nats_url])
    test_json = read_json(multi_json_file)
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def reset_keyphrase():
    nc = NATS()
    topic = "io.etherlabs.ether.keyphrase_service.reset_keyphrases"
    await nc.connect(servers=[nats_url])
    test_json = read_json(multi_json_file)
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def populate_graph():
    nc = NATS()
    topic = "context.instance.in5627.add_segments"
    await nc.connect(servers=[nats_url])
    test_json = read_json(multi_json_file)
    single_test = read_json(single_json_file)
    await nc.publish(topic, json.dumps(single_test).encode())
    # await nc.flush()
    await nc.close()


async def create_context():
    nc = NATS()
    topic = "context.instance.created"
    await nc.connect(servers=[nats_url])
    resp = {
        "contextId": "567238",
        "id": "in5627",
        "state": "created"
    }
    await nc.publish(topic, json.dumps(resp).encode())
    # await start_context()
    # await nc.flush()
    pass


async def start_context():
    nc = NATS()
    topic = "context.instance.in5627.started"
    await nc.connect(servers=[nats_url])
    resp = {
        "id": "in5627",
        "state": "started"
    }
    await nc.publish(topic, json.dumps(resp).encode())
    # await asyncio.sleep(10, loop=loop)
    # await nc.flush()
    pass


async def end_context():
    nc = NATS()
    topic = "context.instance.in5627.ended"
    await nc.connect(servers=[nats_url])
    resp = {
        "id": "in5627",
        "state": "ended"
    }
    await nc.publish(topic, json.dumps(resp).encode())
    await nc.flush()
    await nc.close()
    pass


def read_json(json_file):
    with open(json_file) as f_:
        meeting = json.load(f_)
    return meeting


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='topic arguments for keyphrase_service')
    parser.add_argument("--topics", type=str, default="def", help="publish keyphrase graph")
    parser.add_argument("--nats_url", type=str, default=NATS_URL, help="nats server url")
    args = parser.parse_args()
    nats_url = args.nats_url

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    single_json_file = os.path.join(os.getcwd(), "single_segment_test.json")
    multi_json_file = os.path.join(os.getcwd(), "multi_segment_test.json")

    if args.topics == 'def':
        t1 = loop.run_until_complete(create_context())
    elif args.topics == 'populate':
        loop.run_until_complete(populate_graph())
    elif args.topics == 'pub_chapter':
        loop.run_until_complete(publish_chapter_keyphrase())
    elif args.topics == 'pub_pim':
        loop.run_until_complete(publish_keyphrase())
    elif args.topics == 'pub_instance':
        loop.run_until_complete(publish_instance_keyphrase())
    else:
        # loop.run_until_complete(reset_keyphrase())
        loop.run_until_complete(end_context())
