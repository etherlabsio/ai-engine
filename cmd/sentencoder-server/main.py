# try:
#     import unzip_requirements
# except ImportError:
#     pass

import json
import logging
import os

from log.logger import setup_server_logger
from sentence_encoder.encoder import SentenceEncoder, NumpyEncoder

logger = logging.getLogger(__name__)

# Setup logger
setup_server_logger(debug=True)  # default False for disabling debug mode

bucket_store = os.getenv("STORAGE_BUCKET", "io.etherlabs.staging2.contexts")
model_loc = os.getenv("MODEL", "MODELS/sentence_enc")
model_path = "s3://" + bucket_store + "/" + model_loc

encoder_object = SentenceEncoder(model_path=model_path)
logger.info("Instantiated model")


def process_input(json_request):
    if isinstance(json_request, str):
        json_request = json.loads(json_request)
    text_request = json_request["text_input"]
    return text_request


def lambda_handler(event, context):
    logger.info(event)
    if isinstance(event["body"], str):
        json_request = json.loads(event["body"])
    else:
        json_request = event["body"]

    input_list = process_input(json_request=json_request)
    embeddings = encoder_object.get_embedding_vector(input_list=input_list)
    logger.info("embedding shape", extra={"shape": embeddings.shape})
    response = json.dumps({"embeddings": embeddings}, cls=NumpyEncoder)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": response,
    }
