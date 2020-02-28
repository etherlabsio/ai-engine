from timeit import default_timer as timer
import logging
from scipy.spatial.distance import cosine
import numpy as np
import json
from nltk.tokenize import sent_tokenize
from typing import List

from .objects import (
    Phrase,
    Keyphrase,
    Request,
    GraphQueryRequest,
    GraphSegmentResponse,
    ContextRequest,
)

logger = logging.getLogger(__name__)


class KeyphraseRanker(object):
    def __init__(
        self,
        s3_io_util,
        utils,
        context_dir,
        encoder_lambda_client,
        lambda_function,
        query_client,
    ):
        self.encoder_lambda_client = encoder_lambda_client
        self.lambda_function = lambda_function
        self.context_dir = context_dir
        self.io_util = s3_io_util
        self.utils = utils
        self.query_client = query_client

    async def compute_relevance(
        self,
        phrase_object_list: List[Phrase],
        group_id: str,
        highlight: bool = False,
        default_form="descriptive",
    ) -> List[Phrase]:

        for i, phrase_object in enumerate(phrase_object_list):
            segment_id = phrase_object.segmentId
            segment_query_obj = self.form_segment_query(segment_id=segment_id)
            response = await self.query_client.query_graph(
                query_object=GraphQueryRequest.get_dict(segment_query_obj)
            )
            npz_file = self.query_segment_embeddings(
                response=GraphSegmentResponse.get_object(response),
                highlight=highlight,
                group_id=group_id,
            )
            phrase_object = self._compute_relevant_phrases(
                npz_file=npz_file,
                segment_id=segment_id,
                phrase_object=phrase_object,
                highlight=highlight,
                group_id=group_id,
            )

        phrase_object_list = self.compute_normalized_boosted_rank(
            phrase_object=phrase_object_list, dict_key=default_form
        )

        return phrase_object_list

    def _compute_relevant_phrases(self, phrase_object, **kwargs) -> Phrase:
        phrase_object = self.compute_local_relevance(
            dict_key="descriptive", phrase_object=phrase_object, **kwargs
        )

        # Compute the relevance of entities
        phrase_object = self.compute_local_relevance(
            dict_key="entities", phrase_object=phrase_object, **kwargs
        )

        return phrase_object

    def _get_pairwise_embedding_distance(self, embedding_array):

        dist = cosine(np.array(embedding_array[0]), np.array(embedding_array[1]))

        return dist

    def get_embeddings(self, input_list: List[str], req_data: ContextRequest = None):

        start = timer()
        if req_data is None:
            lambda_payload = {"body": {"text_input": input_list}}
        else:
            lambda_payload = {
                "body": {"request": Request.to_dict(req_data), "text_input": input_list}
            }

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
        phrase_object: Phrase,
        segment_id,
        npz_file,
        dict_key="descriptive",
        normalize: bool = False,
        norm_limit: int = 4,
        highlight: bool = False,
        group_id: str = None,
    ):

        if highlight:
            segment_embedding = npz_file[segment_id + "_" + group_id]
        else:
            segment_embedding = npz_file[segment_id]

        if dict_key == "descriptive" or dict_key == "original":
            keyphrase_object = [
                kp_obj for kp_obj in phrase_object.keyphrases if kp_obj.type == dict_key
            ]
        else:
            keyphrase_object = phrase_object.entities

        for j, kp_obj in enumerate(keyphrase_object):
            phrase = kp_obj.originalForm
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

            kp_obj.score.segsim = norm_seg_score

            # Compute boosted score
            kp_obj = self.compute_boosted_rank(ranked_keyphrase_object=kp_obj)

        segment_relevance_score_list = [
            kp_obj.score.boosted_sim for kp_obj in keyphrase_object
        ]

        if len(segment_relevance_score_list) > 0:
            segment_confidence_score = np.mean(segment_relevance_score_list)
            median_score = np.median(segment_relevance_score_list)
        else:
            segment_confidence_score = 0
            median_score = 0

        if dict_key == "descriptive" or dict_key == "original":
            phrase_object.keyphraseQuality = segment_confidence_score
            phrase_object.medianKeyphraseQuality = median_score
        else:
            phrase_object.entitiesQuality = segment_confidence_score
            phrase_object.medianEntitiesQuality = median_score

        logger.info("Computed segment relevance score")

        return phrase_object

    def compute_boosted_rank(self, ranked_keyphrase_object: Keyphrase):

        keyphrase_score_obj = ranked_keyphrase_object.score
        pagerank_score = keyphrase_score_obj.pagerank
        segment_score = keyphrase_score_obj.segsim

        boosted_score = pagerank_score + segment_score
        keyphrase_score_obj.boosted_sim = boosted_score

        return ranked_keyphrase_object

    def compute_normalized_boosted_rank(
        self, phrase_object: List[Phrase], dict_key="descriptive"
    ):

        total_keyphrase_quality = [
            phrase_obj.medianKeyphraseQuality for phrase_obj in phrase_object
        ]
        total_keyphrase_quality_score = np.sum(total_keyphrase_quality)

        total_entities_quality = [
            phrase_obj.entitiesQuality for phrase_obj in phrase_object
        ]
        total_entities_quality_score = np.sum(total_entities_quality)

        for i, phrase_obj in enumerate(phrase_object):
            keyphrase_object = [
                kp_obj for kp_obj in phrase_obj.keyphrases if kp_obj.type == dict_key
            ]
            entity_object = phrase_obj.entities
            segment = phrase_obj.originalText
            segment_sentence_len = len(sent_tokenize(segment))

            ent_quality_score = phrase_obj.entitiesQuality
            kp_quality_score = phrase_obj.medianKeyphraseQuality

            for kp_obj in keyphrase_object:
                keyphrase_score = kp_obj.score
                boosted_score = keyphrase_score.boosted_sim

                norm_boosted_score = (
                    boosted_score * kp_quality_score * segment_sentence_len
                ) / (total_keyphrase_quality_score)

                keyphrase_score.norm_boosted_sim = norm_boosted_score

            for ent_obj in entity_object:
                entity_score = ent_obj.score
                boosted_score = entity_score.boosted_sim

                norm_boosted_score = (
                    boosted_score * ent_quality_score * segment_sentence_len
                ) / (total_entities_quality_score)

                entity_score.norm_boosted_sim = norm_boosted_score

        return phrase_object

    def query_segment_embeddings(
        self,
        response: GraphSegmentResponse,
        highlight: bool = False,
        group_id: str = None,
    ) -> str:

        # Get segment embedding vector from context graph
        vector_location = response.q[0].embedding_vector_uri
        if not highlight:
            embedding_uri = vector_location
        else:
            f_name = vector_location.split(".")[0]
            f_format = vector_location.split(".")[1]
            embedding_uri = f_name + "_" + group_id + "." + f_format

        npz_file = self.io_util.download_npz(npz_file_path=embedding_uri)
        return npz_file

    def form_segment_query(self, segment_id: str) -> GraphQueryRequest:
        query = """
        query q($i: string) {
            q(func: eq(xid, $i)) {
                uid
                attribute
                xid
                embedding_vector_uri
                embedding_vector_group_uri
            }
        }
        """

        variables = {"$i": segment_id}
        query_object = GraphQueryRequest(query=query, variables=variables)

        return query_object
