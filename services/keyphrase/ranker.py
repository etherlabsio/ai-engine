from timeit import default_timer as timer
import logging
from scipy.spatial.distance import cosine
import numpy as np
from copy import deepcopy
import json

logger = logging.getLogger(__name__)


class KeyphraseRanker(object):
    def __init__(
        self,
        s3_io_util,
        context_dir,
        knowledge_graph_object,
        encoder_lambda_client,
        lambda_function,
    ):
        self.encoder_lambda_client = encoder_lambda_client
        self.lambda_function = lambda_function
        self.kg = knowledge_graph_object
        self.context_dir = context_dir
        self.io_util = s3_io_util

    def _get_pairwise_embedding_distance(self, embedding_array):

        dist = cosine(np.array(embedding_array[0]), np.array(embedding_array[1]))

        return dist

    def get_embeddings(self, input_list, req_data=None):

        start = timer()
        if req_data is None:
            lambda_payload = {"body": {"text_input": input_list}}
        else:
            lambda_payload = {"body": {"request": req_data, "text_input": input_list}}

        logger.info("Invoking lambda function")
        invoke_response = self.encoder_lambda_client.invoke(
            FunctionName=self.lambda_function,
            InvocationType="RequestResponse",
            Payload=json.dumps(lambda_payload),
        )

        lambda_output = (
            invoke_response["Payload"].read().decode("utf8").replace("'", '"')
        )
        response = json.loads(lambda_output)
        status_code = response["statusCode"]
        response_body = response["body"]

        end = timer()
        if status_code == 200:
            embedding_vector = np.asarray(json.loads(response_body)["embeddings"])
            logger.info(
                "Received response from encoder lambda function",
                extra={
                    "featureShape": embedding_vector.shape,
                    "lambdaResponseTime": end - start,
                },
            )

        else:
            embedding_vector = np.asarray(json.loads(response_body)["embeddings"])
            logger.warning(
                "Invalid response from encoder lambda function",
                extra={
                    "warn": "null embeddings",
                    "featureShape": embedding_vector.shape,
                    "lambdaResponseTime": end - start,
                },
            )

        return embedding_vector

    def compute_edge_weights(self, word_graph):

        start = timer()
        counter = 0
        for node1, node2, attr in word_graph.edges.data("edge_emb_wt"):
            if attr is None:
                node_list = [node1, node2]
                node_embedding_array = self.get_embeddings(input_list=node_list)
                emb_sim_edge_weight = 1 - self._get_pairwise_embedding_distance(
                    embedding_array=node_embedding_array
                )

                word_graph.add_edge(node1, node2, edge_emb_wt=emb_sim_edge_weight)
                counter += 1
            else:
                continue

        end = timer()

        logger.debug(
            "Computed word embeddings",
            extra={"totalComputed": counter, "responseTime": end - start},
        )

        return word_graph

    def compute_local_relevance(
        self,
        keyphrase_object,
        context_graph,
        dict_key="descriptive",
        normalize: bool = False,
        norm_limit: int = 4,
    ):

        segment_embedding_list = self._query_segment_phrase_embeddings(
            context_graph=context_graph,
            keyphrase_object=keyphrase_object,
            dict_key=dict_key,
        )
        for i, kp_dict in enumerate(keyphrase_object):
            segment_embedding = segment_embedding_list[i].get("segment_embedding")
            segment_keyphrase_embedding_dict = segment_embedding_list[i].get(
                "keyphrase_embedding"
            )

            keyphrase_dict = kp_dict[dict_key]

            segment_relevance_score_list = []
            for j, (phrase, values) in enumerate(keyphrase_dict.items()):
                phrase_len = len(phrase.split())

                try:
                    keyphrase_embedding = segment_keyphrase_embedding_dict[phrase]
                    seg_score = 1 - cosine(segment_embedding, keyphrase_embedding)
                except KeyError:
                    logger.warning(
                        "Keyphrase does not exist in the graph... Setting score to 0",
                        extra={"phrase": phrase},
                    )
                    seg_score = 0

                if normalize:
                    if phrase_len > norm_limit:
                        norm_seg_score = seg_score / (phrase_len - (norm_limit - 1))
                    else:
                        norm_seg_score = seg_score / phrase_len
                else:
                    norm_seg_score = seg_score

                keyphrase_dict[phrase][1] = norm_seg_score
                segment_relevance_score_list.append(norm_seg_score)

            if len(segment_relevance_score_list) > 0:
                segment_confidence_score = np.mean(segment_relevance_score_list)
            else:
                segment_confidence_score = 0
            kp_dict["quality"] = segment_confidence_score

        logger.info("Computed segment relevance score")

        return keyphrase_object

    def compute_boosted_rank(self, ranked_keyphrase_list):
        boosted_rank_list = []

        for i, items in enumerate(ranked_keyphrase_list):
            keyphrase = items[0]
            pagerank_score = items[1]
            segment_score = items[2]
            loc = items[3]

            boosted_score = pagerank_score + segment_score
            boosted_rank_list[i] = (
                keyphrase,
                pagerank_score,
                segment_score,
                boosted_score,
                loc,
            )

        assert len(ranked_keyphrase_list) == len(boosted_rank_list)

        logger.info("Computed pagerank boosted score")

        return boosted_rank_list

    def _query_segment_phrase_embeddings(
        self, context_graph, keyphrase_object, dict_key="descriptive"
    ):
        """

        Args:
            context_graph:
            keyphrase_object:
            dict_key:

        Returns:
            segment_embedding_list (list[Dict]): Returns list[Dict{"segmentId": str, "original": {}, "descriptive": {}, "keyphrase_embedding": {phrase: phrase_vector}}]
        """

        embedding_dict = {
            "segmentId": str,
            "original": {},
            "descriptive": {},
            "keyphrase_embedding": {},
        }
        segment_embedding_list = []
        for i, kp_dict in enumerate(keyphrase_object):
            embedding_dict_copy = deepcopy(embedding_dict)
            segment_id = kp_dict["segmentId"]
            embedding_dict_copy["segmentId"] = segment_id

            # Get segment embedding vector from context graph
            for node, nattr in context_graph.nodes(data=True):
                if nattr.get("attribute") == "segmentId" and node == segment_id:
                    segment_vector = nattr.get("embedding_vector")
                    embedding_dict_copy["segment_embedding"] = segment_vector

                    for neighbour_nodes, values in dict(context_graph[node]).items():
                        if values.get("relation") == "hasKeywords":
                            embedding_dict_copy["keyphrase_embedding"][
                                neighbour_nodes
                            ] = context_graph.nodes[neighbour_nodes].get(
                                "embedding_vector"
                            )

                    segment_embedding_list.append(embedding_dict_copy)

        return segment_embedding_list
