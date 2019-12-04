import numpy as np
import logging
import json
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


class Vectorizer(object):
    def __init__(
        self, lambda_client=None, lambda_function="sentence-encoder-lambda"
    ):
        self.encoder_lambda_client = lambda_client
        self.lambda_function = lambda_function

    @staticmethod
    def batchify(l, n=30):
        n = max(1, n)
        return (l[i : i + n] for i in range(0, len(l), n))

    def get_embeddings(self, input_list, batch_size=30):

        start = timer()

        batched_list = Vectorizer.batchify(input_list, batch_size)
        embedding_vector = np.zeros((1, 512))

        try:
            logger.info("Invoking lambda function")
            for l in batched_list:
                lambda_payload = {"body": {"text_input": l}}

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
                response_body = response["body"]
                vec = json.loads(response_body)["embeddings"]
                embedding_vector = np.vstack((embedding_vector, vec))

            end = timer()
            logger.info(
                "Received response from encoder lambda function",
                extra={
                    "featureShape": embedding_vector.shape,
                    "lambdaResponseTime": end - start,
                },
            )
            mod_embebdding_vector = np.delete(embedding_vector, [0], axis=0)
            return mod_embebdding_vector

        except Exception as e:
            logger.error("Invoking failed", extra={"err": e})

            return embedding_vector
