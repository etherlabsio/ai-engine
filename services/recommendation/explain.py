from nltk import word_tokenize, pos_tag
from lsh import WordSearch
from typing import List, Dict, Tuple
from fuzzywuzzy import process, fuzz
from scipy.spatial.distance import cosine
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Explainability(object):
    def __init__(
        self,
        reference_user_dict,
        vectorizer,
        num_buckets=8,
        hash_size=4,
        utils_obj=None,
    ):
        self.reference_user_dict = reference_user_dict
        self.vectorizer = vectorizer
        self.utils = utils_obj
        self.num_buckets = num_buckets
        self.hash_size = hash_size
        self.pos_list = [
            "JJ",
            "JJR",
            "JJS",
            "NN",
            "NNP",
            "NNS",
            "NNPS",
            "FW",
        ]

        self.ws = WordSearch(
            vectorizer=self.vectorizer,
            num_buckets=self.num_buckets,
            hash_size=self.hash_size,
        )

    def get_explanation(
        self,
        similar_user_scores_dict: Dict,
        input_query: List[str],
        input_kw_query: List[str],
        query_key="keywords",
    ) -> Tuple[Dict, Dict]:

        input_query_text = self._form_query_text(query_list=input_query)

        similar_users_info_dict = {
            k: self.reference_user_dict[k][query_key]
            for k in similar_user_scores_dict.keys()
        }
        filtered_sim_user_dict = self._filter_user_info(
            similar_users_info_dict=similar_users_info_dict,
            input_query=input_query_text,
        )

        query_feature_vector = self.ws.featurize_input(input_list=input_query)

        user_meta_dict = self.rerank_users(
            similar_users_dict=filtered_sim_user_dict,
            similar_user_scores_dict=similar_user_scores_dict,
            query_feature_vector=query_feature_vector,
            input_query_text=input_query_text,
        )
        sorted_user_meta = self.utils.sort_dict_by_value(
            user_meta_dict, key="confidence"
        )

        try:
            top_user_object = {
                sorted_user_meta[u]["name"]: sorted_user_meta[u]["confidence"]
                for u in filtered_sim_user_dict.keys()
            }
            top_user_object = self.utils.sort_dict_by_value(top_user_object)

            top_words = {
                sorted_user_meta[u]["name"]: sorted_user_meta[u]["topPhrases"][:50]
                for u in filtered_sim_user_dict.keys()
            }

            input_kw_query_text = self._form_query_text(query_list=input_kw_query)
            top_words_dict = self.filter_related_words(
                top_words, input_query_text=input_kw_query_text
            )

            logger.debug(
                "Top users after phrase scoring",
                extra={
                    "users": list(top_user_object.keys()),
                    "userScore": list(top_user_object.values()),
                    "relatedWords": top_words_dict,
                },
            )

            return top_user_object, top_words_dict
        except KeyError as e:
            logger.warning(e)

    def _filter_pos(self, word_list):
        filtered_word = []
        # multi_phrase = []

        for word in word_list:
            single_phrases = []
            pos_word = pos_tag(word_tokenize(word))
            counter = 0
            for tags in pos_word:
                p = tags[1]
                if p in self.pos_list:
                    counter += 1
                    single_phrases.append(tags[0])
                else:
                    if len(single_phrases) > 1:
                        # multi_phrase = [" ".join(single_phrases)]
                        pass

                    single_phrases = []

            if counter == len(word_tokenize(word)):
                filtered_word.append(word)

        # filtered_word.extend(single_phrases)
        return filtered_word

    def filter_related_words(self, top_words: Dict, input_query_text: str) -> Dict:

        filtered_top_words = {u: self._filter_pos(w) for u, w in top_words.items()}
        filtered_top_words = {
            u: list(process.dedupe(w)) for u, w in filtered_top_words.items()
        }
        filtered_top_words = {
            u: self._get_best_fuzzy_match(input_query_text, w)
            for u, w in filtered_top_words.items()
        }

        top_user_words_dict = {
            u: self.utils.sort_dict_by_value(word_dict)
            for u, word_dict in filtered_top_words.items()
        }

        return top_user_words_dict

    def _form_query_text(self, query_list: List[str]):
        input_query_text = ""

        if len(query_list) > 1:
            kw_query_text = " ".join([w for w in query_list])
            kw_query_text_list = [
                v
                for v, i in pos_tag(word_tokenize(kw_query_text))
                if i in self.pos_list
            ]
            input_query_text = " ".join([w for w in kw_query_text_list])

        elif len(query_list) == 1:
            input_query_text = query_list[0]

        return input_query_text

    def _get_best_fuzzy_match(self, input_query, input_list, limit=20):
        try:
            best_words = dict(
                process.extractBests(
                    input_query,
                    input_list,
                    limit=limit,
                    scorer=fuzz.partial_token_set_ratio,
                    score_cutoff=30,
                )
            )
            sorted_best_words = self.utils.sort_dict_by_value(best_words)

            return sorted_best_words
        except Exception as e:
            logger.warning(e)

            return dict()

    def _filter_user_info(
        self, similar_users_info_dict: Dict, input_query: str
    ) -> [List, Dict]:
        """
        Dedupe repeating phrases
        Args:
            similar_users_info_dict:
            input_query:

        Returns:

        """
        filtered_sim_user_info = similar_users_info_dict.copy()
        for u, words in similar_users_info_dict.items():
            filtered_sim_user_info[u] = list(process.dedupe(filtered_sim_user_info[u]))

        return filtered_sim_user_info

    def _compute_cosine_similarity(
        self, reference_feature_vector, query_feature_vector
    ):
        dist = cosine(query_feature_vector, reference_feature_vector)
        cos_sim = 1 - dist

        return cos_sim

    def post_process_related_words(
        self, related_word_list, input_query_text, limit
    ) -> List:
        processed_related_words = process.extractBests(
            input_query_text, related_word_list, limit=limit
        )
        processed_related_words = [word for word, score in processed_related_words]

        return processed_related_words

    def prepare_keyword_meta_object(
        self,
        reference_feature_vector,
        query_feature_vector,
        filtered_sim_user_info_list,
        sim_keyword_result,
    ):
        keyword_meta_dict = {}
        for i in range(reference_feature_vector.shape[0]):
            ranked_w = filtered_sim_user_info_list[i]
            if ranked_w in sim_keyword_result.keys():
                lsh_score = sim_keyword_result[ranked_w]
            else:
                lsh_score = np.min(list(sim_keyword_result.values()))

            cos_sim = self._compute_cosine_similarity(
                query_feature_vector, reference_feature_vector[i]
            )
            keyword_meta_dict.update(
                {
                    ranked_w: {
                        "similarity": cos_sim,
                        "hashSimilarity": lsh_score,
                        "confidence": (cos_sim * lsh_score),
                    }
                }
            )

        return keyword_meta_dict

    def prepare_user_meta_object(self, similar_users_dict, similar_user_kp):
        user_meta_dict = {}
        for user, user_score in similar_users_dict.items():
            user_meta_dict.update(
                {
                    user: {
                        "hashScore": user_score,
                        "phraseScore": np.mean(similar_user_kp),
                        "confidence": (user_score * np.mean(similar_user_kp)),
                    }
                }
            )

        return user_meta_dict

    def rerank_users(
        self,
        similar_users_dict,
        similar_user_scores_dict,
        query_feature_vector,
        input_query_text,
    ):
        user_meta_dict = {}

        if query_feature_vector.shape[0] > 1:
            query_feature_vector = self.ws.featurize_input(
                input_list=[input_query_text]
            )

        for users, user_kw in similar_users_dict.items():
            if len(user_kw) > 0:
                user_kw_features = self.ws.featurize_input(user_kw)
                similarity_dict = {
                    user_kw[i]: self._compute_cosine_similarity(
                        user_kw_features[i], query_feature_vector
                    )
                    for i in range(user_kw_features.shape[0])
                }
                sorted_similarity_dict = self.utils.sort_dict_by_value(similarity_dict)

                phrase_score = np.mean(list(sorted_similarity_dict.values())[:10])
                top_words = list(sorted_similarity_dict.keys())[:10]
                hash_score = similar_user_scores_dict[users]

                user_meta_dict.update(
                    {
                        users: {
                            "name": self.reference_user_dict[users]["name"],
                            "topPhrases": top_words,
                            "phraseScore": phrase_score,
                            "hashScore": hash_score,
                            "confidence": phrase_score + hash_score,
                        }
                    }
                )
            else:
                phrase_score = 0
                hash_score = similar_user_scores_dict[users]
                user_meta_dict.update(
                    {
                        users: {
                            "name": self.reference_user_dict[users]["name"],
                            "topPhrases": list(),
                            "phraseScore": phrase_score,
                            "hashScore": hash_score,
                            "confidence": phrase_score + hash_score,
                        }
                    }
                )

        return user_meta_dict
