try:
    import unzip_requirements
except ImportError:
    print("unable to import unzip_requirements")
    pass

import json
import logging
import os
import numpy as np
from timeit import default_timer as timer

from log.logger import setup_server_logger
from sentence_encoder.encoder import SentenceEncoder, NumpyEncoder

logger = logging.getLogger(__name__)
setup_server_logger(debug=True)  # default False for disabling debug mode

tf_log_level = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_log_level

model_path = os.getenv("MODEL")

start = timer()

# Initialize model
logger.info("Loading model ...")
encoder_object = SentenceEncoder(model_path=model_path)

end = timer()
logger.info(
    "Model loaded and TF Graph initialized",
    extra={"path": model_path, "responseTime": end - start},
)


def process_input(json_request):
    if isinstance(json_request, str):
        json_request = json.loads(json_request)
    text_request = json_request["text_input"]
    return text_request


def handler(event, context):
    if isinstance(event["body"], str):
        json_request = json.loads(event["body"])
    else:
        json_request = event["body"]

    try:
        start = timer()
        input_list = process_input(json_request=json_request)
        embeddings = encoder_object.get_embedding_vector(input_list=input_list)

        response = json.dumps({"embeddings": embeddings}, cls=NumpyEncoder)

        end = timer()
        logger.info(
            "Features extracted successfully",
            extra={
                "embeddingShape": embeddings.shape,
                "request": json_request,
                "responseTime": end - start,
            },
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": response,
        }
    except Exception as e:
        logger.error(
            "Error processing request", extra={"err": e, "request": json_request}
        )
        placeholder_embeddings = np.zeros(shape=(1, 512))
        response = json.dumps({"embeddings": placeholder_embeddings}, cls=NumpyEncoder)
        return {
            "statusCode": 404,
            "headers": {"Content-Type": "application/json"},
            "body": response,
        }
