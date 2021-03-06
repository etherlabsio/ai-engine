try:
    import unzip_requirements
except ImportError:
    pass

import os
import logging
import json
from pims.scorer import get_score
from pims.pre_processors import preprocess_segments
from log.logger import setup_server_logger

logger = logging.getLogger()
setup_server_logger(debug=True)


def handler(event, context):
    logger.info("POST request recieved", extra={"event['body']:": event["body"]})
    if isinstance(event["body"], str):
        json_request = json.loads(event["body"])
    else:
        json_request = event["body"]

    mindId = str(json_request["mindId"]).lower()
    lambda_function = "mind-" + mindId

    transcript_text = json_request["segments"][0]["originalText"]
    pre_processed_input = preprocess_segments(transcript_text)

    if len(pre_processed_input) != 0:
        mind_input = json.dumps({"text": pre_processed_input})
        mind_input = json.dumps({"body": mind_input})
        logger.info("sending request to mind service")
        transcript_score = get_score(mind_input, lambda_function)
    else:
        transcript_score = 0.00001
        logger.warn("processing transcript: {}".format(transcript_text))
        logger.warn("transcript too small to process. Returning default score")

    # hack to penalize out of domain small transcripts coming as PIMs - word level
    if len(pre_processed_input.split(" ")) < 40:
        transcript_score = 0.1 * transcript_score

    out_response = json.dumps(
        {
            "text": transcript_text,
            "distance": 1 / transcript_score,
            "id": json_request["segments"][0]["id"],
            "conversationLength": 1000,
            "speaker": json_request["segments"][0]["spokenBy"],
        }
    )
    logger.info("response", extra={"out_response": out_response})
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "d2vResult": [
                    {
                        "text": transcript_text,
                        "distance": 1 / transcript_score,
                        "id": json_request["segments"][0]["id"],
                        "conversationLength": 1000,
                        "speaker": json_request["segments"][0]["spokenBy"],
                    }
                ]
            }
        ),
    }
