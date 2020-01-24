import json as js
import pickle
import logging
from typing import List, Dict, Mapping
import numpy as np
import uuid
import itertools
from fuzzywuzzy import process
import traceback

from lsh import HashSession
from explain import Explainability
from graph_query import QueryHandler, GraphQuery
from utils import Utils

logger = logging.getLogger(__name__)


class RecWatchers(object):
    def __init__(
        self,
        dgraph_url: str,
        vectorizer=None,
        s3_client=None,
        web_hook_url=None,
        active_env_ab_test=None,
        num_buckets=200,
        hash_size=16,
    ):
        self.vectorizer = vectorizer
        self.s3_client = s3_client

        self.query_client = GraphQuery(dgraph_url=dgraph_url)

        self.query_handler = QueryHandler(vectorizer=vectorizer, s3_client=s3_client)

        self.active_env_ab_test = active_env_ab_test
        self.web_hook_url = web_hook_url
        self.num_buckets = num_buckets
        self.hash_size = hash_size

        self.utils = Utils(web_hook_url=self.web_hook_url, s3_client=self.s3_client,)

        self.exp = Explainability(
            vectorizer=self.vectorizer,
            utils_obj=self.utils,
            num_buckets=self.num_buckets,
            hash_size=self.hash_size,
        )

        self.feature_dir = "/features/recommendation/"

    def initialize_reference_objects(
        self, context_id: str, top_n: int = 50, perform_query: bool = True
    ):

        if perform_query:
            query_text, variables = self.query_client.form_user_contexts_query(
                context_id=context_id, top_n_result=top_n
            )
            response = self.query_client.perform_query(
                query=query_text, variables=variables
            )

            reference_user_dict = self.query_handler.format_user_contexts_reference_response(
                response
            )
            (
                reference_user_json_path,
                features_path,
            ) = self.query_handler.form_reference_features(
                reference_user_dict=reference_user_dict,
                context_id=context_id,
                ref_key="keywords",
            )
        else:
            reference_user_json_path = (
                context_id + self.feature_dir + context_id + ".json"
            )
            features_path = context_id + self.feature_dir + context_id + ".pickle"

        reference_user_meta_dict, reference_features = self.download_reference_objects(
            context_id=context_id,
            reference_user_file_path=reference_user_json_path,
            reference_user_vector_data_path=features_path,
        )

        return reference_user_meta_dict, reference_features

    def download_reference_objects(
        self,
        context_id: str,
        reference_user_file_path: str = None,
        reference_user_vector_data_path: str = None,
    ):
        try:
            if reference_user_file_path is None:
                reference_user_file_path = (
                    context_id + self.feature_dir + context_id + ".json"
                )

            if reference_user_vector_data_path is None:
                reference_user_vector_data_path = (
                    context_id + self.feature_dir + context_id + ".pickle"
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

    def create_hash_session(self):
        hs = HashSession(
            vectorizer=self.vectorizer,
            num_buckets=self.num_buckets,
            hash_size=self.hash_size,
        )
        return hs

    async def featurize_reference_users(
        self,
        reference_user_dict: Dict,
        reference_features: Dict,
        user_feature_map: Dict[str, int],
        hash_session_object,
    ):
        session = hash_session_object
        # Featurize reference users
        updated_user_feature_map = session.hs.featurize(
            input_dict=reference_user_dict,
            user_vector_data=reference_features,
            user_feature_map=user_feature_map,
        )

        return updated_user_feature_map

    def perform_hash_query(
        self, input_list, user_feature_map: Dict[str, int], hash_session_object
    ):
        session = hash_session_object
        hash_result = session.hs.query(
            input_list=input_list, user_feature_map=user_feature_map
        )
        norm_hash_result, cut_off_score = self._normalize_lsh_score(
            similar_users_dict=hash_result
        )

        return norm_hash_result

    def get_recommended_watchers(
        self,
        input_query_list,
        input_kw_query,
        user_feature_map: Dict,
        hash_session_object=None,
        reference_user_meta_dict: Dict = None,
        hash_result=None,
        segment_obj=None,
        segment_user_ids=None,
        n_users=6,
        n_kw=6,
        check_relevancy=False,
        user_vector_data=None,
    ):
        top_n_user_object = {}
        top_user_words = []
        suggested_users = []
        try:
            if segment_user_ids is None:
                segment_user_ids = []

            if hash_result is None or len(list(hash_result.keys())) < 1:
                hash_result = self.perform_hash_query(
                    input_list=input_kw_query,
                    user_feature_map=user_feature_map,
                    hash_session_object=hash_session_object,
                )

            if len(hash_result.keys()) == 0 or hash_result is None:
                logger.info(
                    "No recommendations available... couldn't find or less user data available"
                )
                return top_n_user_object, top_user_words, suggested_users

            similar_user_scores_dict = self.query_similar_users(
                hash_result=hash_result, input_list=input_kw_query
            )

            if len(similar_user_scores_dict.keys()) == 0:
                logger.info(
                    "No recommendations available... couldn't find or less user data available"
                )
                return top_n_user_object, top_user_words, suggested_users

            if check_relevancy:
                relevant_recommendation = self.check_recommendation_relevancy(
                    hash_result=hash_result, relevance_threshold=0.40
                )
                if not relevant_recommendation:
                    logger.info("Low relevance score... No recommendations")

                    return top_n_user_object, top_user_words, suggested_users

            logger.info("Computing explainability...")
            top_n_user_object, top_related_words = self.exp.get_explanation(
                similar_user_scores_dict=similar_user_scores_dict,
                reference_user_dict=reference_user_meta_dict,
                input_query=input_query_list,
                input_kw_query=input_kw_query,
                query_key="keywords",
                user_vector_data=user_vector_data,
            )

            user_scores = list(top_n_user_object.values())

            # Include only those users that are not part of segment request
            top_n_user_object = {
                u: s for u, s in top_n_user_object.items() if u not in segment_user_ids
            }
            suggested_users = self.post_process_users(
                segment_obj=segment_obj, user_dict=top_n_user_object, percentile_val=60,
            )
            top_user_words = [
                w for u in suggested_users for w, score in top_related_words[u].items()
            ]
            top_user_words = list(process.dedupe(top_user_words))

            top_n_user_names = [
                self.get_user_names(u_id) for u_id in top_n_user_object.keys()
            ]
            suggested_user_names = [
                self.get_user_names(u_id) for u_id in suggested_users
            ]

            logger.info(
                "Top recommended users found",
                extra={
                    "users": list(top_n_user_object.keys()),
                    "userName": top_n_user_names,
                    "scores": user_scores,
                    "suggestedUsers": suggested_users,
                    "suggestedUserNames": suggested_user_names,
                },
            )

            logger.info(
                "Similarity explanation",
                extra={
                    "totalRelatedWords": len(top_user_words),
                    "words": top_user_words[:n_kw],
                },
            )

            return top_n_user_object, top_user_words[:n_kw], suggested_users
        except Exception as e:
            logger.warning("Unable to get recommendation", extra={"err": e})
            print(traceback.print_exc())

            return top_n_user_object, top_user_words, suggested_users

    def query_similar_users(self, hash_result, input_list: List, n_retries=3) -> Dict:
        top_similar_users = self.utils.sort_dict_by_value(hash_result)

        similar_users_dict, cutoff_score = self._normalize_lsh_score(
            top_similar_users, normalize_by="percentile", percentile_val=70
        )
        filtered_similar_users_dict = self._threshold_user_info(
            similar_users_dict, cutoff_score
        )

        user_scores = list(similar_users_dict.values())
        # user_names = [
        #     self.reference_user_dict[u].get("name") for u in similar_users_dict.keys()
        # ]
        # filtered_user_names = [
        #     self.reference_user_dict[u].get("name")
        #     for u in filtered_similar_users_dict.keys()
        # ]

        logger.debug(
            "Top recommended users",
            extra={
                "totalSimilarUsers": len(similar_users_dict),
                "scores": user_scores,
                "numFilteredUsers": len(filtered_similar_users_dict),
                "cutOffScore": cutoff_score,
            },
        )

        return filtered_similar_users_dict

    def get_user_names(self, user_id):
        query, var = self.query_client.get_user_name_query(user_id=user_id)
        response = self.query_client.perform_query(query, var)
        user_name = self.query_handler.get_user_name(response=response)

        return user_name

    def re_hash_users(self, input_list):
        # self.featurize_reference_users()
        hash_result = self.perform_hash_query(input_list=input_list)

        return hash_result

    def check_recommendation_relevancy(
        self, hash_result: Dict, relevance_threshold: float = 0.40
    ) -> bool:
        # Hash relevancy
        # norm_hash_result, cut_off_score = self._normalize_lsh_score(hash_result)
        sorted_hash_result = self.utils.sort_dict_by_value(hash_result)
        user_hash_scores = list(sorted_hash_result.values())

        logger.debug(
            "Verifying Relevancy",
            extra={
                "relevanceScore": user_hash_scores[0],
                "relevanceThreshold": relevance_threshold,
            },
        )

        # Check if the top user's hash score is greater than the threshold
        if user_hash_scores[0] >= relevance_threshold:
            return True
        else:
            return False

    def _check_high_user_deviation(self, similar_user_scores_dict, limit=3):
        user_scores = list(similar_user_scores_dict.values())
        std_dev = np.std(user_scores)

        try:
            if (
                int((user_scores[0] - user_scores[1]) / std_dev) >= limit
                or (user_scores[0] - user_scores[1]) >= std_dev
            ):
                return True
            else:
                return False
        except Exception as e:
            logger.warning(e)
            return False

    def _normalize_lsh_score(
        self, similar_users_dict: Dict, normalize_by="mean", percentile_val=60
    ) -> [Dict, float]:
        cutoff_score = 0
        user_score_list = list(similar_users_dict.values())
        similar_users_dict = {
            r: self.utils.normalize(score, user_score_list)
            for r, score in similar_users_dict.items()
        }

        try:
            if normalize_by == "mean":
                cutoff_score = np.mean(list(similar_users_dict.values()))
            elif normalize_by == "median":
                cutoff_score = np.median(list(similar_users_dict.values()))
            elif normalize_by == "percentile":
                cutoff_score = np.percentile(
                    list(similar_users_dict.values()), percentile_val
                )
        except Exception as e:
            logger.warning(e)
            cutoff_score = 0

        return similar_users_dict, cutoff_score

    def _threshold_user_info(
        self, similar_users_dict: Dict, cutoff_score: float
    ) -> Dict:
        filtered_similar_users = {
            u: score for u, score in similar_users_dict.items() if score >= cutoff_score
        }

        return filtered_similar_users

    def post_process_users(self, segment_obj=None, user_dict=None, percentile_val=70):
        try:
            percentile_cutoff = np.percentile(list(user_dict.values()), percentile_val)
            suggested_users = [
                user
                for user, scores in user_dict.items()
                if scores >= percentile_cutoff
            ]
        except Exception as e:
            logger.warning(e)
            suggested_users = [user for user, scores in user_dict.items()]

        return suggested_users

    def prepare_slack_validation(
        self,
        req_data,
        user_dict,
        word_list,
        suggested_users,
        segment_users,
        upload=False,
    ):
        user_list = list(user_dict.keys())
        user_scores = list(user_dict.values())
        segment_user_ids = [str(uuid.UUID(u)) for u in segment_users]
        try:
            segment_user_names = [
                self.reference_user_dict[u]["name"] for u in segment_user_ids
            ]
        except Exception:
            segment_user_names = ["NA"]

        self.utils.make_validation_data(
            req_data=req_data,
            user_list=user_list,
            user_scores=user_scores,
            suggested_user_list=suggested_users,
            word_list=word_list,
            segment_users=segment_user_names,
            upload=upload,
        )
        self.utils.post_to_slack(
            req_data=req_data,
            user_list=user_list,
            user_scores=user_scores,
            suggested_user_list=suggested_users,
            word_list=word_list,
        )
