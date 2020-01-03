import json as js
import pickle
import logging
from typing import List, Dict
import numpy as np
import uuid
import itertools
from fuzzywuzzy import process

from lsh import UserSearch
from explain import Explainability
from transform import FileTransform
from utils import Utils

logger = logging.getLogger(__name__)


class RecWatchers(object):
    def __init__(
        self,
        reference_user_file=None,
        reference_user_vector_data=None,
        reference_user_kw_vector_data=None,
        reference_user_dict=None,
        user_vector_data=None,
        vectorizer=None,
        s3_client=None,
        web_hook_url=None,
        active_env_ab_test=None,
        num_buckets=200,
        hash_size=16,
    ):
        self.vectorizer = vectorizer
        self.s3_client = s3_client

        self.reference_user_file = reference_user_file
        self.reference_user_vector_data = reference_user_vector_data
        self.reference_user_kw_vector_data = reference_user_kw_vector_data
        self.active_env_ab_test = active_env_ab_test
        self.web_hook_url = web_hook_url
        self.num_buckets = num_buckets
        self.hash_size = hash_size
        self.user_vector_data = user_vector_data
        self.reference_user_dict = reference_user_dict

        if user_vector_data is None and reference_user_dict is None:
            self.initialize_downloads()

        self.ref_user_info_dict = {
            k: self.reference_user_dict[k]["keywords"]
            for k in self.reference_user_dict.keys()
        }

        self.utils = Utils(
            web_hook_url=self.web_hook_url,
            s3_client=self.s3_client,
            reference_user_dict=self.reference_user_dict,
        )
        self.exp = Explainability(
            reference_user_dict=self.reference_user_dict,
            vectorizer=self.vectorizer,
            utils_obj=self.utils,
            num_buckets=self.num_buckets,
            hash_size=self.hash_size,
        )
        self.us = UserSearch(
            input_dict=self.ref_user_info_dict,
            vectorizer=self.vectorizer,
            user_vector_data=self.user_vector_data,
            num_buckets=self.num_buckets,
            hash_size=self.hash_size,
        )

    def initialize_downloads(self):
        # Download and transform recommendation objects
        logger.info("Downloading reference objects from s3")
        (
            self.reference_user_dict,
            self.user_vector_data,
        ) = self.download_reference_objects(
            self.reference_user_file, self.reference_user_kw_vector_data
        )
        self.ref_user_info_dict = {
            k: self.reference_user_dict[k]["keywords"]
            for k in self.reference_user_dict.keys()
        }

        logger.info("loaded reference features")

    def download_reference_objects(
        self, reference_user_file, reference_user_vector_data
    ):
        reference_user_meta = self.s3_client.download_file(
            file_name=reference_user_file
        )
        reference_user_meta_str = reference_user_meta["Body"].read().decode("utf8")
        reference_user_meta_dict = js.loads(reference_user_meta_str)

        reference_user_vector_object = self.s3_client.download_file(
            file_name=reference_user_vector_data
        )
        reference_user_vector_str = reference_user_vector_object["Body"].read()
        reference_user_vector = pickle.loads(reference_user_vector_str)

        return reference_user_meta_dict, reference_user_vector

    def featurize_reference_users(self):
        # Featurize reference users
        self.us.featurize()

    def perform_hash_query(self, input_list):
        hash_result = self.us.query(input_list=input_list)
        norm_hash_result, cut_off_score = self._normalize_lsh_score(
            similar_users_dict=hash_result
        )

        return norm_hash_result

    def get_recommended_watchers(
        self,
        input_query_list,
        input_kw_query,
        hash_result=None,
        segment_obj=None,
        n_users=6,
        n_kw=6,
    ):
        if hash_result is None or len(list(hash_result.keys())) < 1:
            hash_result = self.perform_hash_query(input_list=input_kw_query)

        relevant_recommendation = self.check_recommendation_relevancy(
            hash_result=hash_result, relevance_threshold=0.40
        )
        if not relevant_recommendation:
            logger.info("Low relevance score... No recommendations")
            top_n_user_object = {}
            top_user_words = []
            suggested_users = []
            return top_n_user_object, top_user_words, suggested_users

        similar_user_scores_dict = self.query_similar_users(
            hash_result=hash_result, input_list=input_kw_query
        )

        logger.info("Computing explainability...")
        top_n_user_object, top_related_words = self.exp.get_explanation(
            similar_user_scores_dict=similar_user_scores_dict,
            input_query=input_query_list,
            input_kw_query=input_kw_query,
            query_key="keywords",
        )

        suggested_users, filtered_top_user_object = self.post_process_users(
            segment_obj=segment_obj, user_dict=top_n_user_object, percentile_val=60,
        )
        top_user_words = self.get_top_user_words(
            suggested_users=suggested_users,
            top_related_words=top_related_words,
            n_kw=n_kw,
        )

        logger.info(
            "Final recommended users & Explanation",
            extra={
                "users": list(filtered_top_user_object.keys()),
                "scores": list(filtered_top_user_object.values()),
                "totalRelatedWords": len(top_user_words),
                "words": top_user_words,
            },
        )
        return filtered_top_user_object, top_user_words, suggested_users

    def query_similar_users(self, hash_result, input_list: List, n_retries=3) -> Dict:
        top_similar_users = self.utils.sort_dict_by_value(hash_result)

        # # Logic for handling random cases where standard deviation between users is very high
        # for i in range(n_retries):
        #     high_user_dev = self._check_high_user_deviation(
        #         similar_user_scores_dict=top_similar_users
        #     )
        #     if high_user_dev:
        #         logger.debug(
        #             "Recomputing hashes - Re-try {}".format(i),
        #             extra={"currentScore": top_similar_users},
        #         )
        #         hash_result = self.re_hash_users(input_list=input_list)
        #         top_similar_users = self.utils.sort_dict_by_value(hash_result)
        #
        #     else:
        #         logger.debug("Appropriate search found...")
        #         break

        similar_users_dict, cutoff_score = self._normalize_lsh_score(top_similar_users)
        filtered_similar_users_dict = self._threshold_user_info(
            similar_users_dict, cutoff_score
        )

        user_scores = list(similar_users_dict.values())
        user_names = [
            self.reference_user_dict[u].get("name") for u in similar_users_dict.keys()
        ]
        filtered_user_names = [
            self.reference_user_dict[u].get("name")
            for u in filtered_similar_users_dict.keys()
        ]

        logger.debug(
            "Top recommended users",
            extra={
                "totalSimilarUsers": len(user_names),
                "users": user_names,
                "scores": user_scores,
                "numFilteredUsers": len(filtered_user_names),
                "filteredUsers": filtered_user_names,
                "cutOffScore": cutoff_score,
            },
        )

        return filtered_similar_users_dict

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

    def re_hash_users(self, input_list):
        self.featurize_reference_users()
        hash_result = self.perform_hash_query(input_list=input_list)

        return hash_result

    def post_process_users(self, segment_obj=None, user_dict=None, percentile_val=70):
        if segment_obj is None:
            filtered_user_dict = user_dict
        else:
            segment_user_ids = [
                str(uuid.UUID(seg_ids["spokenBy"])) for seg_ids in segment_obj
            ]
            try:
                segment_user_names = [
                    self.reference_user_dict[u]["name"] for u in segment_user_ids
                ]
                filtered_user_dict = {
                    user: score
                    for user, score in user_dict.items()
                    if user not in segment_user_names
                }
            except Exception as e:
                logger.warning(e)
                filtered_user_dict = user_dict

        percentile_cutoff = np.percentile(
            list(filtered_user_dict.values()), percentile_val
        )
        suggested_users = [
            user
            for user, score in filtered_user_dict.items()
            if score >= percentile_cutoff
        ]
        return suggested_users, filtered_user_dict

    def get_top_user_words(
        self, suggested_users, top_related_words: Dict, n_kw: int = 6
    ):
        top_user_words = [
            w for u in suggested_users for w, score in top_related_words[u].items()
        ]
        top_user_words = list(process.dedupe(top_user_words))

        return top_user_words[:n_kw]

    def _check_high_user_deviation(self, similar_user_scores_dict, limit=3):
        user_scores = list(similar_user_scores_dict.values())
        std_dev = np.std(user_scores)

        try:
            if (
                int((user_scores[0] - abs(user_scores[1])) / std_dev) >= limit
                or (user_scores[0] - abs(user_scores[1])) >= std_dev
            ):
                return True
            else:
                return False
        except Exception as e:
            logger.warning(e)
            return False

    def _normalize_lsh_score(
        self, similar_users_dict: Dict, threshold_by="percentile", percentile_val=70
    ) -> [Dict, float]:
        cutoff_score = 0
        user_score_list = list(similar_users_dict.values())
        similar_users_dict = {
            r: self.utils.normalize(score, user_score_list)
            for r, score in similar_users_dict.items()
        }

        if threshold_by == "mean":
            cutoff_score = np.mean(list(similar_users_dict.values()))
        elif threshold_by == "median":
            cutoff_score = np.median(list(similar_users_dict.values()))
        elif threshold_by == "percentile":
            cutoff_score = np.percentile(
                list(similar_users_dict.values()), percentile_val
            )

        return similar_users_dict, cutoff_score

    def _threshold_user_info(
        self, similar_users_dict: Dict, cutoff_score: float
    ) -> Dict:
        filtered_similar_users = {
            u: score for u, score in similar_users_dict.items() if score >= cutoff_score
        }

        return filtered_similar_users

    def prepare_slack_validation(
        self,
        req_data,
        user_dict,
        word_list,
        suggested_users,
        segment_users,
        upload=False,
    ):
        rec_user_list = list(user_dict.keys())
        rec_user_scores = list(user_dict.values())
        segment_user_ids = [str(uuid.UUID(u)) for u in segment_users]
        try:
            segment_user_names = [
                self.reference_user_dict[u]["name"] for u in segment_user_ids
            ]
            segment_user_names = list(set(segment_user_names))
        except Exception:
            segment_user_names = ["NA"]

        self.utils.make_validation_data(
            req_data=req_data,
            user_list=rec_user_list,
            user_scores=rec_user_scores,
            suggested_user_list=suggested_users,
            word_list=word_list,
            segment_users=segment_user_names,
            upload=upload,
        )
        self.utils.post_to_slack(
            req_data=req_data,
            user_list=rec_user_list,
            user_scores=rec_user_scores,
            suggested_user_list=suggested_users,
            segment_user_names=segment_user_names,
            word_list=word_list,
        )
