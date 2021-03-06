import numpy as np
import logging
from typing import List, MutableMapping

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


class MinHash(object):
    @staticmethod
    def hash_func(vecs, projections):
        bools = np.dot(vecs, projections.T) > 0
        return [MinHash.bool2int(bool_vec) for bool_vec in bools]

    @staticmethod
    def bool2int(x):
        y = 0
        for i, j in enumerate(x):
            if j:
                y += 1 << i
        return y


class Table(object):
    def __init__(self, hash_size, dim):
        self.table = dict()
        self.hash_size = hash_size
        self.projections = np.random.randn(self.hash_size, dim)

    def add(self, vecs, label):
        entry = {"label": label}
        hashes = MinHash.hash_func(vecs, self.projections)
        for h in hashes:
            if h in self.table.keys():
                self.table[h].append(entry)
            else:
                self.table[h] = [entry]

    def query(self, vecs):
        hashes = MinHash.hash_func(vecs, self.projections)
        results = list()
        for h in hashes:
            if h in self.table.keys():
                results.extend(self.table[h])
        return results


class LSH(object):
    def __init__(self, dim, num_buckets, hash_size):
        self.num_tables = num_buckets
        self.hash_size = hash_size
        self.tables = list()
        for i in range(self.num_tables):
            self.tables.append(Table(self.hash_size, dim))

    def add(self, vecs, label):
        for table in self.tables:
            table.add(vecs, label)

    def query(self, vecs):
        results = list()
        for table in self.tables:
            results.extend(table.query(vecs))
        return results

    def describe(self):
        for table in self.tables:
            yield (table)


class WordSearch(object):
    def __init__(
        self, vectorizer=None, num_buckets: int = 8, hash_size: int = 4, dim: int = 512,
    ):
        self.dim_size = dim
        self.lsh = LSH(self.dim_size, num_buckets=num_buckets, hash_size=hash_size)
        self.vectorizer = vectorizer
        self.num_features_in_input = dict()

    def featurize(self, reference_features, reference_input_list):
        for kw in reference_input_list:
            self.num_features_in_input[kw] = 0

        for i in range(len(reference_features)):
            self.lsh.add([reference_features[i]], reference_input_list[i])
            self.num_features_in_input[reference_input_list[i]] += len(
                reference_features[i]
            )

    def featurize_input(self, input_list: list):
        feature_object = self.vectorizer.get_embeddings(input_list=input_list)
        return feature_object

    def query(self, feature_object):
        # kw_features = self.vectorizer.get_embeddings(input_list=input_list)

        results = self.lsh.query(feature_object)

        counts = dict()
        for r in results:
            if r["label"] in counts.keys():
                counts[r["label"]] += 1
            else:
                counts[r["label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.num_features_in_input[k]
        return counts

    def describe(self):
        for t in self.lsh.describe():
            yield (t)


class UserSearch(object):
    def __init__(
        self, vectorizer=None, num_buckets: int = 8, hash_size: int = 4, dim: int = 512,
    ):
        self.dim_size = dim
        self.lsh = LSH(self.dim_size, num_buckets=num_buckets, hash_size=hash_size)

    def featurize(
        self,
        input_dict: UserMetaData,
        user_vector_data: UserVectorData,
        user_feature_map: UserFeatureMap,
    ) -> UserFeatureMap:
        num_features_in_input = (
            user_feature_map if user_feature_map is not None else dict()
        )
        for user, kw in input_dict.items():
            try:
                kw_features = user_vector_data[user]
            except KeyError as e:
                logger.warning(
                    "could not find feature vector", extra={"warn": e, "userId": user}
                )
                continue

            self.lsh.add(kw_features, user)
            num_features_in_input[user] = len(kw_features)

        logger.debug(
            "hashed users",
            extra={
                "users": list(num_features_in_input.keys()),
                "featureLen": list(num_features_in_input.values()),
                "totalFeat": len(list(num_features_in_input.keys())),
            },
        )

        return num_features_in_input

    def query(
        self, kw_features: InputData, user_feature_map: UserFeatureMap, tag: str = "v1"
    ):
        # kw_features = self.vectorizer.get_embeddings(input_list=input_list)

        results = self.lsh.query(kw_features)
        logger.info("num results", extra={"totalMatches": len(results)})

        counts = {}
        for r in results:
            if r["label"] in counts.keys():
                counts[r["label"]] += 1
            else:
                counts[r["label"]] = 1

        # total_features = np.sum([user_feature_map[k] for k in counts])
        if tag == "v1":
            for k in counts:
                counts[k] = float(counts[k]) / user_feature_map[k]
        else:
            try:
                norm_user_feature_num = self.norm(
                    lower=np.mean(list(user_feature_map.values())),
                    upper=np.max(list(user_feature_map.values())),
                    d=user_feature_map,
                )
            except Exception:
                norm_user_feature_num = user_feature_map
            for k in counts:
                counts[k] = float(counts[k]) / (len(results) * norm_user_feature_num[k])

        return counts

    def describe(self):
        for t in self.lsh.describe():
            yield (t)

    def norm(self, lower, upper, d: dict):
        l_norm = {x: lower + (upper - lower) * v for x, v in d.items()}
        return l_norm


class HashSession(UserSearch):
    def __init__(
        self, vectorizer=None, num_buckets: int = 8, hash_size: int = 4, dim: int = 512
    ):
        super().__init__(vectorizer, num_buckets, hash_size, dim)

    def hash_features(
        self,
        input_dict: UserMetaData,
        user_vector_data: UserVectorData,
        user_feature_map: UserFeatureMap,
    ):
        num_features = self.featurize(
            input_dict=input_dict,
            user_vector_data=user_vector_data,
            user_feature_map=user_feature_map,
        )

        return num_features

    def hash_query(
        self, kw_features: InputData, user_feature_map: UserFeatureMap, tag: str = "v1"
    ):
        counts = self.query(kw_features, user_feature_map, tag=tag)
        return counts
