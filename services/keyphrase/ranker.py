from timeit import default_timer as timer
import logging
from scipy.spatial.distance import cosine
import numpy as np
import json
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class KeyphraseRanker(object):
    def __init__(
        self, s3_io_util, utils, context_dir, encoder_lambda_client, lambda_function
    ):
        self.encoder_lambda_client = encoder_lambda_client
        self.lambda_function = lambda_function
        self.context_dir = context_dir
        self.io_util = s3_io_util
        self.utils = utils

    def _get_pairwise_embedding_distance(self, embedding_array):

        dist = cosine(np.array(embedding_array[0]), np.array(embedding_array[1]))

        return dist

    def get_embeddings(self, input_list, req_data=None):

        start = timer()
        if req_data is None:
            lambda_payload = {"body": {"text_input": input_list}}
        else:
            lambda_payload = {"body": {"request": req_data, "text_input": input_list}}

        try:
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
                        "payload": input_list,
                    },
                )

            else:
                embedding_vector = np.asarray(json.loads(response_body)["embeddings"])
                logger.warning(
                    "Invalid response from encoder lambda function",
                    extra={
                        "warnMsg": "null embeddings",
                        "featureShape": embedding_vector.shape,
                        "lambdaResponseTime": end - start,
                        "payload": input_list,
                    },
                )

            return embedding_vector

        except Exception as e:
            logger.error("Invoking failed", extra={"err": e})
            embedding_vector = np.zeros((1, 512))
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
        kp_dict,
        segment_id,
        npz_file,
        dict_key="descriptive",
        normalize: bool = False,
        norm_limit: int = 4,
        populate_graph=True,
        group_id="",
    ):

        if populate_graph is not True:
            segment_embedding = npz_file[segment_id + "_" + group_id]
        else:
            segment_embedding = npz_file[segment_id]

        keyphrase_dict = kp_dict[dict_key]
        segment_relevance_score_list = []
        for j, (phrase, values) in enumerate(keyphrase_dict.items()):
            phrase_len = len(phrase.split())

            try:
                phrase_id = self.utils.hash_phrase(phrase)
                keyphrase_embedding = npz_file[phrase_id]
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

            # Compute boosted score
            keyphrase_dict = self.compute_boosted_rank(
                ranked_keyphrase_dict=keyphrase_dict
            )
            segment_relevance_score_list.append(keyphrase_dict[phrase][2])

        if len(segment_relevance_score_list) > 0:
            segment_confidence_score = np.mean(segment_relevance_score_list)
            median_score = np.median(segment_relevance_score_list)
        else:
            segment_confidence_score = 0
            median_score = 0

        if dict_key == "descriptive" or dict_key == "original":
            kp_dict["keyphraseQuality"] = segment_confidence_score
            kp_dict["medianKeyphraseQuality"] = median_score

        else:
            kp_dict["entitiesQuality"] = segment_confidence_score
            kp_dict["medianEntitiesQuality"] = median_score

        logger.info("Computed segment relevance score")

        return kp_dict

    def compute_boosted_rank(self, ranked_keyphrase_dict):

        for i, (phrase, items) in enumerate(ranked_keyphrase_dict.items()):
            pagerank_score = items[0]
            segment_score = items[1]

            boosted_score = pagerank_score + segment_score
            ranked_keyphrase_dict[phrase][2] = boosted_score

        return ranked_keyphrase_dict

    def compute_normalized_boosted_rank(self, keyphrase_object, dict_key="descriptive"):

        total_keyphrase_quality = [
            kp_dict["medianKeyphraseQuality"] for kp_dict in keyphrase_object
        ]
        total_keyphrase_quality_score = np.sum(total_keyphrase_quality)

        total_entities_quality = [
            kp_dict["entitiesQuality"] for kp_dict in keyphrase_object
        ]
        total_entities_quality_score = np.sum(total_entities_quality)

        for i, kp_dict in enumerate(keyphrase_object):
            keyphrase_dict = kp_dict[dict_key]
            entity_dict = kp_dict["entities"]
            segment = kp_dict["segments"]
            segment_sentence_len = len(sent_tokenize(segment))

            ent_quality_score = kp_dict["entitiesQuality"]
            kp_quality_score = kp_dict["medianKeyphraseQuality"]

            for phrase, scores in keyphrase_dict.items():
                boosted_score = scores[2]

                norm_boosted_score = (
                    boosted_score * kp_quality_score * segment_sentence_len
                ) / (total_keyphrase_quality_score)
                keyphrase_dict[phrase][3] = norm_boosted_score

            for phrase, scores in entity_dict.items():
                boosted_score = scores[2]

                norm_boosted_score = (
                    boosted_score * ent_quality_score * segment_sentence_len
                ) / (total_entities_quality_score)
                entity_dict[phrase][3] = norm_boosted_score

        return keyphrase_object

    def query_segment_embeddings(self, response, populate_graph=True):

        # Get segment embedding vector from context graph
        if populate_graph:
            embedding_uri = response["q"][0].get("embedding_vector_uri")
        else:
            embedding_uri = response["q"][0].get("embedding_vector_group_uri")

        npz_file = self.io_util.download_npz(npz_file_path=embedding_uri)
        return npz_file

    def form_segment_query(self, segment_id, populate_graph=True):
        query = """query q($i: string) {
                                q(func: eq(xid, $i)) {
                                    uid
                                    attribute
                                    xid
                                    embedding_vector_uri
                                }
                            }"""

        if populate_graph is not True:
            query = """query q($i: string) {
                                    q(func: eq(xid, $i)) {
                                        uid
                                        attribute
                                        xid
                                        embedding_vector_group_uri
                                    }
                                }"""

        variables = {"$i": segment_id}
        query_object = {"query": query, "variables": variables}

        return query_object
