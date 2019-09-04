import json
import logging
import os
from timeit import default_timer as timer

from log.logger import setup_server_logger
from sentence_encoder.encoder import SentenceEncoder, NumpyEncoder

logger = logging.getLogger(__name__)
setup_server_logger(debug=True)  # default False for disabling debug mode

try:
    import unzip_requirements
except ImportError:
    logger.warning("unable to import unzip_requirements")
    pass

bucket_store = os.getenv("STORAGE_BUCKET", "io.etherlabs.staging2.contexts")
model_loc = os.getenv("MODEL", "MODELS/sentence_enc")
model_path = "s3://" + bucket_store + "/" + model_loc

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


def lambda_handler(event, context):
    logger.info(event)
    if isinstance(event["body"], str):
        json_request = json.loads(event["body"])
    else:
        json_request = event["body"]

    try:
        start = timer()
        input_list = process_input(json_request=json_request)
        embeddings = encoder_object.get_embedding_vector(input_list=input_list)

        end = timer()
        logger.info(
            "embedding shape",
            extra={"shape": embeddings.shape, "responseTime": end - start},
        )
        response = json.dumps({"embeddings": embeddings}, cls=NumpyEncoder)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": response,
        }
    except Exception as e:
        logger.error(
            "Error processing request", extra={"err": e, "request": json_request}
        )
        return {
            "statusCode": 404,
            "headers": {"Content-Type": "application/json"},
            "body": e,
        }
