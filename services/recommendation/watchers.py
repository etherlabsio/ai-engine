import logging
from typing import List, Dict, MutableMapping, Sequence, Tuple, Union
import numpy as np
from fuzzywuzzy import process
import traceback

from lsh import HashSession
from explain import Explainability
from graph_query import QueryHandler
from utils import Utils
from watcher_utils import WatcherUtils
from redis_store import RedisStore
from object_def import (
    UserID,
    InputData,
    MetaData,
    UserVectorData,
    UserFeatureMap,
    UserMetaData,
    HashResult,
)

logger = logging.getLogger(__name__)


USER_STORE = "recommendations/watchers/users"
VECTOR_STORE = "recommendations/watchers/vectors"
USER_FEATURE_NUM_STORE = "recommendations/watchers/user_feature_map"
HASH_STORE = "recommendations/watchers/hash_session"
REDIS_EXPIRE = 120


class RecWatchers(object):
    def __init__(
        self,
        dgraph_url: str,
        vectorizer=None,
        s3_client=None,
        web_hook_url=None,
        active_env_ab_test=None,
        num_buckets=500,
        hash_size=16,
        redis_host: str = "localhost",
    ):
        self.feature_dir = "/features/recommendation/"
        self.user_store = RedisStore(id=USER_STORE, host=redis_host)
        self.vector_store = RedisStore(id=VECTOR_STORE, host=redis_host)
        self.user_feature_store = RedisStore(id=USER_FEATURE_NUM_STORE, host=redis_host)
        self.hash_store = RedisStore(id=HASH_STORE, host=redis_host)

        self.vectorizer = vectorizer
        self.s3_client = s3_client

        self.query_handler = QueryHandler(
            vectorizer=vectorizer,
            s3_client=s3_client,
            dgraph_url=dgraph_url,
            feature_dir=self.feature_dir,
        )

        self.wu = WatcherUtils(
            query_handler=self.query_handler,
            s3_client=s3_client,
            vectorizer=vectorizer,
            hash_session_object=HashSession,
            hash_size=hash_size,
            num_buckets=num_buckets,
            feature_dir=self.feature_dir,
        )

        self.active_env_ab_test = active_env_ab_test
        self.web_hook_url = web_hook_url
        self.num_buckets = num_buckets
        self.hash_size = hash_size

        self.utils = Utils(web_hook_url=self.web_hook_url, s3_client=self.s3_client)

        self.exp = Explainability(
            vectorizer=self.vectorizer,
            utils_obj=self.utils,
            num_buckets=self.num_buckets,
            hash_size=self.hash_size,
        )

    def initialize_objects(
        self,
        context_id: str,
        session_id: str,
        top_n: int = 30,
        perform_query: bool = True,
        tag: str = "v1",
        query_by: str = "keywords",
        store_redis: bool = True,
    ):
        if tag == "v2":
            top_n = 10
        (
            reference_users_metadata,
            reference_vectors,
            total_features,
        ) = self.wu.initialize_reference_objects(
            context_id=context_id,
            top_n=top_n,
            perform_query=perform_query,
            query_by=query_by,
            tag=tag,
        )

        if store_redis:
            self.user_store.set_object(key=session_id, object=reference_users_metadata)
            self.vector_store.set_object(key=session_id, object=reference_vectors)

        return reference_users_metadata, reference_vectors, total_features

    def featurize_reference_users(
        self,
        session_id: str,
        reference_user_dict: UserMetaData,
        reference_features: UserVectorData,
        user_feature_map: UserFeatureMap = None,
        hash_session_object: HashSession = None,
        num_features: int = 0,
    ) -> UserVectorData:

        if hash_session_object is None:
            hash_session = self.wu.create_hash_session(
                session_id=session_id, num_features=num_features
            )
            self.hash_store.set_object(key=session_id, object=hash_session)
        else:
            hash_session = hash_session_object

        if user_feature_map is None:
            try:
                user_feature_map = self.user_feature_store.get_object(key=session_id)
            except KeyError:
                user_feature_map = {}

        # Featurize reference users
        updated_user_feature_map = hash_session.hash_features(
            input_dict=reference_user_dict,
            user_vector_data=reference_features,
            user_feature_map=user_feature_map,
        )

        # Store the updated user-feature num map in redis for future reference
        self.hash_store.set_object(key=session_id, object=hash_session)
        self.user_feature_store.set_object(
            key=session_id, object=updated_user_feature_map
        )

        return updated_user_feature_map

    def perform_hash_query(
        self,
        input_list: InputData,
        user_feature_map: UserFeatureMap,
        hash_session,
        tag: str = "v1",
    ) -> HashResult:

        input_features = self.vectorizer.get_embeddings(input_list)
        hash_result = hash_session.hash_query(
            kw_features=input_features, user_feature_map=user_feature_map, tag=tag
        )

        return hash_result

    def get_recommended_watchers(
        self,
        context_id: str,
        instance_id: str,
        session_id: str,
        input_query_list: InputData,
        input_kw_query: InputData,
        participant_response: List[Dict] = None,
        user_feature_map: UserFeatureMap = None,
        hash_session=None,
        reference_user_meta_dict: UserMetaData = None,
        user_vector_data: UserVectorData = None,
        hash_result: HashResult = None,
        segment_user_ids: Sequence[UserID] = None,
        n_users: int = 6,
        n_kw: int = 6,
        check_relevancy: bool = False,
        query_by: str = "keywords",
        tag: str = "v1",
    ):
        original_rec_users = {}
        top_n_user_object = {}
        top_user_words = []
        suggested_users = []
        suggested_user_names = []
        top_n_user_names = []
        try:
            if segment_user_ids is None:
                segment_user_ids = []

            if hash_session is None:
                hash_session_object = self.hash_store.get_object(key=session_id)
                user_feature_map = self.user_feature_store.get_object(session_id)
                reference_user_meta_dict = self.user_store.get_object(session_id)
                user_vector_data = self.vector_store.get_object(session_id)
            else:
                hash_session_object = hash_session

            # This condition is reequired for cases where context.instance.ended is received before get_watchers
            if hash_session_object is None:
                (
                    reference_user_meta_dict,
                    user_vector_data,
                    total_features,
                ) = self.initialize_objects(
                    context_id=context_id,
                    session_id=session_id,
                    perform_query=False,
                    tag=tag,
                )

                user_feature_map = self.featurize_reference_users(
                    session_id=session_id,
                    reference_user_dict=reference_user_meta_dict,
                    reference_features=user_vector_data,
                    num_features=total_features,
                )

                hash_session_object = self.hash_store.get_object(key=session_id)

            if hash_result is None or len(list(hash_result.keys())) < 1:
                hash_result = self.perform_hash_query(
                    input_list=input_kw_query,
                    user_feature_map=user_feature_map,
                    hash_session=hash_session_object,
                    tag=tag,
                )

            original_rec_users = self.utils.sort_dict_by_value(hash_result)

            if len(hash_result.keys()) == 0 or hash_result is None:
                logger.info(
                    "No recommendations available... couldn't find or less user data available"
                )
                return (
                    original_rec_users,
                    top_n_user_object,
                    top_user_words,
                    suggested_users,
                    suggested_user_names,
                    top_n_user_names,
                )

            if check_relevancy:
                relevant_recommendation = self.check_recommendation_relevancy(
                    hash_result=hash_result, relevance_threshold=0.55
                )
                if not relevant_recommendation:
                    logger.info("Low relevance score... No recommendations")

                    return (
                        original_rec_users,
                        top_n_user_object,
                        top_user_words,
                        suggested_users,
                        suggested_user_names,
                        top_n_user_names,
                    )

            participant_list = self.wu.get_active_participants(
                response=participant_response
            )
            similar_user_scores_dict = self.query_similar_users(
                hash_result=hash_result,
                participants=participant_list,
                instance_id=instance_id,
            )

            if len(similar_user_scores_dict.keys()) == 0:
                logger.info(
                    "No recommendations available... couldn't find or less user data available"
                )
                return (
                    original_rec_users,
                    top_n_user_object,
                    top_user_words,
                    suggested_users,
                    suggested_user_names,
                    top_n_user_names,
                )

            logger.info("Computing explainability...")
            top_n_user_object, top_related_words = self.exp.get_explanation(
                similar_user_scores_dict=similar_user_scores_dict,
                reference_user_dict=reference_user_meta_dict,
                input_query=input_query_list,
                input_kw_query=input_kw_query,
                query_by=query_by,
                user_vector_data=user_vector_data,
            )

            (
                top_n_user_object,
                top_user_words,
                suggested_users,
                suggested_user_names,
                top_n_user_names,
            ) = self.post_process_users(
                user_dict=top_n_user_object,
                top_related_words=top_related_words,
                segment_user_ids=segment_user_ids,
                n_kw=n_kw,
                percentile_val=75,
            )

            logger.info(
                "Top recommended users found",
                extra={
                    "users": list(top_n_user_object.keys()),
                    "userName": top_n_user_names,
                    "scores": list(top_n_user_object.values()),
                    "suggestedUsers": suggested_users,
                    "suggestedUserNames": suggested_user_names,
                },
            )

            logger.info(
                "Similarity explanation",
                extra={
                    "totalRelatedWords": len(top_user_words),
                    "words": top_user_words,
                },
            )

            return (
                original_rec_users,
                top_n_user_object,
                top_user_words,
                suggested_users,
                suggested_user_names,
                top_n_user_names,
            )
        except Exception as e:
            logger.warning("Unable to get recommendation", extra={"err": e})
            print(traceback.print_exc())

            return (
                original_rec_users,
                top_n_user_object,
                top_user_words,
                suggested_users,
                suggested_user_names,
                top_n_user_names,
            )

    def query_similar_users(
        self, hash_result, instance_id: str, participants: List
    ) -> Dict:
        top_similar_users = self.utils.sort_dict_by_value(hash_result)

        if len(participants) > 0:
            # Restrict top similar users object to non-participants
            top_similar_users = {
                u: s for u, s in top_similar_users.items() if u not in participants
            }

            logger.info(
                "Modified search space after removing meeting participants",
                extra={
                    "instanceId": instance_id,
                    "totalParticipants": len(participants),
                    "totalNonParticipants": len(top_similar_users.keys()),
                    "participants": [self.wu.get_user_names(p) for p in participants],
                    "modSimilarUsers": [
                        self.wu.get_user_names(u) for u in top_similar_users.keys()
                    ],
                },
            )

        similar_users_dict, cutoff_score = self._normalize_lsh_score(
            top_similar_users, normalize_by="percentile", percentile_val=75
        )
        filtered_similar_users_dict = self._threshold_user_info(
            similar_users_dict, cutoff_score
        )

        user_scores = list(similar_users_dict.values())

        logger.debug(
            "Top recommended users",
            extra={
                "totalSimilarUsers": len(hash_result),
                "scores": user_scores,
                "numFilteredUsers": len(filtered_similar_users_dict),
                "cutOffScore": cutoff_score,
            },
        )

        return filtered_similar_users_dict

    def check_recommendation_relevancy(
        self, hash_result: HashResult, relevance_threshold: float = 0.50
    ) -> bool:
        # Hash relevancy
        norm_hash_result, cut_off_score = self._normalize_lsh_score(hash_result)
        sorted_hash_result = self.utils.sort_dict_by_value(norm_hash_result)
        user_hash_scores = list(sorted_hash_result.values())

        logger.debug(
            "Verifying Relevancy",
            extra={
                "relevanceScore": user_hash_scores[0],
                "relevanceThreshold": relevance_threshold,
                "users": list(sorted_hash_result.keys()),
                "scores": user_hash_scores,
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
        self, similar_users_dict: Dict, normalize_by="mean", percentile_val=75
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
            u: score
            for u, score in similar_users_dict.items()
            if score >= cutoff_score and score > 0
        }

        return filtered_similar_users

    def post_process_users(
        self,
        user_dict: Dict,
        top_related_words: MutableMapping[str, Dict[str, float]],
        segment_user_ids: List[UserID],
        percentile_val=60,
        n_kw: int = 6,
    ):

        # Include only those users that are not part of segment request
        top_n_user_object = {
            u: s for u, s in user_dict.items() if u not in segment_user_ids
        }

        try:
            percentile_cutoff = np.percentile(
                list(top_n_user_object.values()), percentile_val
            )
            suggested_users = [
                user
                for user, scores in top_n_user_object.items()
                if scores >= percentile_cutoff
            ]
        except Exception as e:
            logger.warning(e)
            suggested_users = [user for user, scores in top_n_user_object.items()]

        top_user_words = [
            w for u in suggested_users for w, score in top_related_words[u].items()
        ]
        top_user_words = list(process.dedupe(top_user_words))

        top_n_user_names = [
            self.wu.get_user_names(u_id) for u_id in top_n_user_object.keys()
        ]
        suggested_user_names = [
            self.wu.get_user_names(u_id) for u_id in suggested_users
        ]

        return (
            top_n_user_object,
            top_user_words[:n_kw],
            suggested_users,
            suggested_user_names,
            top_n_user_names,
        )

    def prepare_slack_validation(
        self,
        req_data,
        original_user_dict,
        user_dict,
        word_list,
        suggested_users,
        segment_users,
        upload=False,
        post_to_slack=False,
    ):
        original_user_list = list(original_user_dict.keys())
        original_user_score = list(original_user_dict.values())
        user_list = list(user_dict.keys())
        user_scores = list(user_dict.values())
        # segment_user_ids = [str(uuid.UUID(u)) for u in segment_users]
        try:
            segment_user_names = [self.wu.get_user_names(uid) for uid in segment_users]
            original_user_names = [
                self.wu.get_user_names(uid) for uid in original_user_list
            ]
        except Exception:
            segment_user_names = ["NA"]
            original_user_names = ["NA"]

        self.utils.make_validation_data(
            req_data=req_data,
            user_list=user_list,
            user_scores=user_scores,
            suggested_user_list=suggested_users,
            word_list=word_list,
            segment_users=segment_user_names,
            original_user_score=original_user_score,
            original_user_names=original_user_names,
            upload=upload,
        )

        if post_to_slack:
            self.utils.post_to_slack(
                req_data=req_data,
                user_list=user_list,
                user_scores=user_scores,
                suggested_user_list=suggested_users,
                word_list=word_list,
            )

    async def cleanup_stores(self, session_id: str):
        logger.info("Clearing store for session", extra={"sessionId": session_id})
        self.vector_store.delete_key(key=session_id)
        self.user_store.delete_key(key=session_id)
        self.user_feature_store.delete_key(key=session_id)
        self.hash_store.delete_key(key=session_id)

    async def clear_redis_db(self):
        logger.info("clearing objects from redis")
        self.user_store.expire_store(time=5)
        self.vector_store.expire_store(time=5)
        self.user_feature_store.expire_store(time=5)
        self.hash_store.expire_store(time=5)
