import logging
import json as js
import pickle
import math
import uuid
from typing import Iterable, List, Dict

logger = logging.getLogger(__name__)


class WatcherUtils(object):
    def __init__(
        self,
        query_handler,
        s3_client,
        vectorizer,
        hash_session_object,
        hash_size: int = 16,
        num_buckets: int = 200,
        feature_dir: str = "/features/recommendation/",
    ):

        self.query_handler = query_handler
        self.hash_session_object = hash_session_object
        self.s3_client = s3_client
        self.vectorizer = vectorizer
        self.hash_size = hash_size
        self.num_buckets = num_buckets

        self.feature_dir = feature_dir

    def initialize_reference_objects(
        self,
        context_id: str,
        top_n: int = 100,
        perform_query: bool = True,
        tag: str = "v1",
        query_by: str = "keywords",
    ):
        total_features = 0

        reference_user_json_path = (
            context_id + self.feature_dir + context_id + "_" + tag + ".json"
        )
        features_path = (
            context_id + self.feature_dir + context_id + "_" + tag + ".pickle"
        )

        if perform_query:
            (
                reference_user_json_path,
                features_path,
                total_features,
            ) = self.query_handler.form_user_contexts_query(
                context_id=context_id, top_n_result=top_n, tag=tag, query_by=query_by
            )

        reference_user_meta_dict, reference_features = self.download_reference_objects(
            context_id=context_id,
            reference_user_file_path=reference_user_json_path,
            reference_user_vector_data_path=features_path,
            tag=tag,
        )

        return reference_user_meta_dict, reference_features, total_features

    def download_reference_objects(
        self,
        context_id: str,
        reference_user_file_path: str = None,
        reference_user_vector_data_path: str = None,
        tag="v1",
    ):
        try:
            logger.info("Downloading required objects...")
            if reference_user_file_path is None:
                reference_user_file_path = (
                    context_id + self.feature_dir + context_id + "_" + tag + ".json"
                )

            if reference_user_vector_data_path is None:
                reference_user_vector_data_path = (
                    context_id + self.feature_dir + context_id + "_" + tag + ".pickle"
                )

            reference_user_meta = self.s3_client.download_file(
                file_name=reference_user_file_path
            )
            reference_user_meta_str = reference_user_meta["Body"].read().decode("utf8")
            reference_user_meta_dict = js.loads(reference_user_meta_str)

            reference_user_vector_object = self.s3_client.download_file(
                file_name=reference_user_vector_data_path
            )
            reference_user_vector_str = reference_user_vector_object["Body"].read()
            reference_user_vector = pickle.loads(reference_user_vector_str)

            logger.info("Downloaded required objects")

            return reference_user_meta_dict, reference_user_vector

        except Exception as e:
            logger.error("Error downloading objects", extra={"err": e})
            raise

    def _next_power_of_2(self, x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def compute_parameters(self, num_features: int = 0):
        if num_features == 0:
            hash_size = self.hash_size
            num_buckets = self.num_buckets
        else:
            hash_size = self._next_power_of_2(math.ceil(math.log2(num_features)))
            logger.debug("Using hash size of {}".format(hash_size))
            num_buckets = self.num_buckets

        return hash_size, num_buckets

    def create_hash_session(self, session_id, num_features: int = 0):
        hash_size, num_buckets = self.compute_parameters(num_features)

        logger.info(
            "Adaptive hash size computed",
            extra={
                "hashSize": hash_size,
                "buckets": num_buckets,
                "session": session_id,
            },
        )
        hs = self.hash_session_object(num_buckets=num_buckets, hash_size=hash_size,)
        return hs

    def get_user_names(self, user_id):
        query, var = self.query_handler.query_client.get_user_name_query(
            user_id=user_id
        )
        response = self.query_handler.query_client.perform_query(query, var)
        user_name = self.query_handler.get_user_name(response=response)

        return user_name

    def get_active_participants(self, response: List[Dict]) -> List:
        try:
            participants = [resp_obj["sourceUserId"] for resp_obj in response]
            participants = [str(uuid.UUID(u)) for u in participants]
        except Exception as e:
            logger.warning(
                "unable to get participant list",
                extra={"warn": e, "participantResponse": response},
            )
            participants = []

        return participants
