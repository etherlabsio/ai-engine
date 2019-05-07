import asyncio
import json
from nats.aio.client import Client as NATS
import argparse
import os
import uvloop
from dotenv import load_dotenv
from nltk import word_tokenize, WordNetLemmatizer
import logging

load_dotenv()

ACTIVE_ENV = os.getenv("ACTIVE_ENV")
NATS_URL = os.getenv("NATS_URL")
DEFAULT_ENV = os.getenv("DEF_ENV")

logger = logging.getLogger(__name__)


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


async def publish_chapter_offset_keyphrase():
    nc = NATS()
    topic = "keyphrase_service.in5627.extract_keyphrases_with_offset"
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
    # test_json = read_json(multi_json_file)
    single_test = read_json(single_json_file)
    await nc.publish(topic, json.dumps(single_test).encode())
    # await nc.flush()
    await nc.close()


async def create_context():
    nc = NATS()
    topic = "context.instance.created"
    await nc.connect(servers=[nats_url])
    resp = {"contextId": "567238", "id": "in5627", "state": "created"}
    await nc.publish(topic, json.dumps(resp).encode())
    # await start_context()
    # await nc.flush()
    pass


async def start_context():
    nc = NATS()
    topic = "context.instance.in5627.started"
    await nc.connect(servers=[nats_url])
    resp = {"id": "in5627", "state": "started"}
    await nc.publish(topic, json.dumps(resp).encode())
    # await asyncio.sleep(10, loop=loop)
    # await nc.flush()
    pass


async def end_context():
    nc = NATS()
    topic = "context.instance.in5627.ended"
    await nc.connect(servers=[nats_url])
    resp = {"id": "in5627", "state": "ended"}
    await nc.publish(topic, json.dumps(resp).encode())
    await nc.flush()
    await nc.close()
    pass


def read_json(json_file):
    with open(json_file) as f_:
        meeting = json.load(f_)
    return meeting


def test_keyphrase_quality():
    keyphrase_list = [
        ("story story", 0.3),
        ("epic epics", 0.38),
        ("basic basic", 0.44),
        ("meetings id created", 0.5),
        ("mindfulness emotional intelligence decision making", 0.76),
        ("emotional intelligence decision making", 0.7),
        ("meeting meeting", 0.2),
        ("meeting id created", 0.3),
        ("key phrase", 0.3),
        ("key phrases", 0.3),
        ("key phases", 0.2),
        ("slack apps", 0.4),
        ("marketplaces", 0.5),
    ]

    return keyphrase_list


def post_process():
    keyphrases = test_keyphrase_quality()
    processed_keyphrases = []

    # Remove same word occurrences in a multi-keyphrase
    for multi_key, multi_score in keyphrases:
        kw_m = multi_key.split()
        unique_kp_list = list(dict.fromkeys(kw_m))
        multi_keyphrase = " ".join(unique_kp_list)
        processed_keyphrases.append((multi_keyphrase, multi_score))

    single_phrase = [
        phrases for phrases in processed_keyphrases if len(phrases[0].split()) == 1
    ]
    multi_proc_phrases = [
        phrases for phrases in processed_keyphrases if len(phrases[0].split()) > 1
    ]
    # Remove duplicates from the single phrases which are occurring in multi-keyphrases
    for tup in single_phrase:
        kw = tup[0]
        for tup_m in multi_proc_phrases:
            kw_m = tup_m[0]
            r = kw_m.find(kw)
            if r > -1:
                try:
                    processed_keyphrases.remove(tup)
                except Exception as e:
                    logger.warning("keyword not found", extra={"warning": e})
                    continue

    # Remove duplicates from multi-phrases
    proc_phrase = processed_keyphrases
    for tup in proc_phrase:
        kw = tup[0]
        for tup_m in processed_keyphrases:
            kw_m = tup_m[0]
            if kw_m in kw or kw in kw_m:
                print(kw, kw_m)
                if kw != kw_m:
                    processed_keyphrases.remove(tup_m)
                else:
                    continue

    # Singular list
    processed_keyphrases = _lemmatize_sentence(processed_keyphrases)

    return processed_keyphrases


def _lemmatize_sentence(keyphrase_list):
    tmp_check_list = keyphrase_list
    result = []
    lemma = WordNetLemmatizer()

    for tup in tmp_check_list:
        phrase = tup[0]
        score = tup[1]
        tokenize_phrase = word_tokenize(phrase)
        singular_tokens = [lemma.lemmatize(word) for word in tokenize_phrase]
        singular_sentence = " ".join(singular_tokens)
        if len(singular_sentence) > 0:
            if singular_sentence in result:
                keyphrase_list.remove(tup)
            else:
                result.append((phrase, score))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="topic arguments for keyphrase_service"
    )
    parser.add_argument(
        "--topics", type=str, default="def", help="publish keyphrase graph"
    )
    parser.add_argument(
        "--nats_url", type=str, default=NATS_URL, help="nats server url"
    )
    args = parser.parse_args()
    nats_url = args.nats_url

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    single_json_file = os.path.join(os.getcwd(), "single_segment_test.json")
    multi_json_file = os.path.join(os.getcwd(), "staging_meeting_deepgram.json")

    if args.topics == "def":
        t1 = loop.run_until_complete(create_context())
    elif args.topics == "populate":
        loop.run_until_complete(populate_graph())
    elif args.topics == "pub_chapter":
        loop.run_until_complete(publish_chapter_keyphrase())
    elif args.topics == "pub_chapter_offset":
        loop.run_until_complete(publish_chapter_offset_keyphrase())
    elif args.topics == "pub_pim":
        loop.run_until_complete(publish_keyphrase())
    elif args.topics == "pub_instance":
        loop.run_until_complete(publish_instance_keyphrase())
    elif args.topics == "quality":
        post_process()
    else:
        loop.run_until_complete(end_context())
