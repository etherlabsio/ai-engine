import numpy as np
import logging
import json
from timeit import default_timer as timer
from boto3 import client as client
from botocore.client import Config

logger = logging.getLogger(__name__)


class Vectorizer(object):
    def __init__(self):
        aws_config = Config(
            connect_timeout=180,
            read_timeout=300,
            retries={"max_attempts": 2},
            region_name="us-east-1",
        )
        self.encoder_lambda_client = client("lambda", config=aws_config)
        self.lambda_function = "sentence-encoder-lambda"

    def get_embeddings(self, input_list):

        start = timer()
        lambda_payload = {"body": {"text_input": input_list}}

        try:
            logger.info("Invoking lambda function")
            invoke_response = self.encoder_lambda_client.invoke(
                FunctionName=self.lambda_function,
                InvocationType="RequestResponse",
                Payload=json.dumps(lambda_payload),
            )

            lambda_output = (
                invoke_response["Payload"]
                .read()
                .decode("utf8")
                .replace("'", '"')
            )
            response = json.loads(lambda_output)
            status_code = response["statusCode"]
            response_body = response["body"]

            end = timer()
            if status_code == 200:
                embedding_vector = np.asarray(
                    json.loads(response_body)["embeddings"]
                )
                logger.info(
                    "Received response from encoder lambda function",
                    extra={
                        "featureShape": embedding_vector.shape,
                        "lambdaResponseTime": end - start,
                    },
                )

            else:
                embedding_vector = np.asarray(
                    json.loads(response_body)["embeddings"]
                )
                logger.warning(
                    "Invalid response from encoder lambda function",
                    extra={
                        "warnMsg": "null embeddings",
                        "featureShape": embedding_vector.shape,
                        "lambdaResponseTime": end - start,
                    },
                )

            return embedding_vector

        except Exception as e:
            logger.error("Invoking failed", extra={"err": e})
            embedding_vector = np.zeros((1, 512))

            return embedding_vector
