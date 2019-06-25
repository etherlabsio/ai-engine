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
    topic = "keyphrase_service.extract_keyphrases"
    await nc.connect(servers=[nats_url])
    test_json = read_json(single_json_file)
    topic, resp = replace_ids(
        test_json["contextId"], test_json["instanceId"], topic, resp={}
    )
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    # await nc.close()


async def publish_chapter_keyphrase():
    nc = NATS()
    topic = "keyphrase_service.extract_keyphrases"
    await nc.connect(servers=[nats_url])
    test_json = read_json(multi_json_file)
    topic, resp = replace_ids(
        test_json["contextId"], test_json["instanceId"], topic, resp={}
    )
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def publish_chapter_offset_keyphrase():
    nc = NATS()
    topic = "keyphrase_service.extract_keyphrases_with_offset"
    await nc.connect(servers=[nats_url])
    test_json = read_json(multi_json_file)
    topic, resp = replace_ids(
        test_json["contextId"], test_json["instanceId"], topic, resp={}
    )
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def publish_instance_keyphrase():
    nc = NATS()
    topic = "keyphrase_service.keyphrases_for_context_instance"
    await nc.connect(servers=[nats_url])
    test_json = read_json(meeting_json_file)
    topic, resp = replace_ids(
        test_json["contextId"], test_json["instanceId"], topic, resp={}
    )
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def reset_keyphrase():
    nc = NATS()
    topic = "io.etherlabs.ether.keyphrase_service.reset_keyphrases"
    await nc.connect(servers=[nats_url])
    test_json = read_json(meeting_json_file)
    topic, resp = replace_ids(
        test_json["contextId"], test_json["instanceId"], topic, resp={}
    )
    await nc.request(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def populate_graph():
    nc = NATS()
    topic = "context.instance.add_segments"
    await nc.connect(servers=[nats_url])
    # test_json = read_json(multi_json_file)
    test_json = read_json(meeting_json_file)
    topic, resp = replace_ids(
        test_json["contextId"], test_json["instanceId"], topic, resp={}
    )
    await nc.publish(topic, json.dumps(test_json).encode())
    # await nc.flush()
    await nc.close()


async def create_context():
    nc = NATS()
    topic = "context.instance.created"
    await nc.connect(servers=[nats_url])
    resp = {"contextId": "*", "instanceId": "*", "state": "created"}

    topic, resp = replace_ids(topic=topic, resp=resp)
    await nc.publish(topic, json.dumps(resp).encode())
    # await start_context()
    # await nc.flush()
    pass


async def start_context():
    nc = NATS()
    topic = "context.instance.started"
    await nc.connect(servers=[nats_url])
    resp = {"instanceId": "*", "state": "started", "contextId": "*"}
    topic, resp = replace_ids(topic=topic, resp=resp)
    await nc.publish(topic, json.dumps(resp).encode())
    # await asyncio.sleep(10, loop=loop)
    # await nc.flush()
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


def desc_keyphrase_quality():
    keyphrase_list = [
        ("ten on the google processing", 0.3),
        ("space and zoom is ron", 0.38),
        ("speakers this morning was sheet", 0.44),
        ("morning was sheet people", 0.5),
        ("heavy complete wise", 0.76),
        ("slack box space", 0.7),
        ("behavior with zoom going public", 0.2),
        ("kind of develop the conversation", 0.3),
        ("explain her behavior with zoom", 0.3),
        ("meetings to progress the discussions", 0.3),
        ("set up meetings to progress", 0.2),
        ("development of a training partner", 0.4),
        ("help them compete and penetrate", 0.5),
        ("compete and penetrate accounts", 0.5),
        ("custom professional services opportunities", 0.4),
        ("vertical or custom professional services", 0.4),
        ("san francisco", 0.3),
        ("high forms", 0.2),
        ("runs their whole digital practice", 0.6),
        ("open up the windows command", 0.3),
        ("standard windows command", 0.5),
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


def post_process_desc():
    keyphrases = desc_keyphrase_quality()

    # Join 2 similar sentences
    processed_keyphrase = []
    for index1, (words, score) in enumerate(keyphrases):
        for index2, (words2, score2) in enumerate(keyphrases):
            if index1 != index2:
                word_set = set(list(dict.fromkeys(words.split(" "))))
                word_set2 = set(list(dict.fromkeys(words2.split(" "))))
                if len(word_set & word_set2) > 2:
                    new_set = word_set & word_set2

                    # for w in list(new_set)[:1]:
                    w = list(new_set)[0]
                    word_index1 = words.split(" ").index(w)
                    word_index2 = words2.split(" ").index(w)
                    if word_index1 > word_index2:
                        word3 = words.split(" ") + words2.split(" ")
                        word4 = " ".join(list(dict.fromkeys(word3)))
                        processed_keyphrase.append(word4)

    logger.info(processed_keyphrase)


def replace_ids(context_id=None, instance_id=None, topic=None, resp=dict()):

    if context_id is None and instance_id is None:
        context_id = "new6baa3490-69d6-48fc-b5d4-3994e3e8fae0"
        instance_id = "newb5d4-3994e3e8fae0"

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
        "--topics", type=str, default="def", help="publish keyphrase graph"
    )
    parser.add_argument(
        "--nats_url", type=str, default=NATS_URL, help="nats server url"
    )
    args = parser.parse_args()
    nats_url = args.nats_url

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    single_json_file = os.path.join(os.getcwd(), "pim_test.json")
    multi_json_file = os.path.join(os.getcwd(), "chapter_test.json")
    meeting_json_file = os.path.join(os.getcwd(), "meeting_test2.json")

    if args.topics == "def":
        t1 = loop.run_until_complete(create_context())
        loop.run_until_complete(start_context())
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
        post_process_desc()
    else:
        loop.run_until_complete(end_context())
