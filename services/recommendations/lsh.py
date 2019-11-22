import numpy as np
from vectorize import Vectorizer


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
            # print(table.table, table.projections)
            yield (table)


class Search(object):
    def __init__(
        self, input, vectorizer=Vectorizer(), num_buckets=4, hash_size=8
    ):
        self.dim_size = 512
        self.lsh = LSH(
            self.dim_size, num_buckets=num_buckets, hash_size=hash_size
        )
        self.vectorizer = vectorizer
        self.input = input
        self.num_features_in_input = dict()
        for f in self.input:
            self.num_features_in_input[f] = 0

    def featurize(self):
        # for kw_list in self.input:
        kw_features = self.vectorizer.get_embeddings(input_list=self.input)

        for i in range(len(kw_features)):
            self.lsh.add([kw_features[i]], self.input[i])
            self.num_features_in_input[self.input[i]] += len(kw_features[i])

    def query(self, kw_list):
        # kw_list = user_dict[user]
        kw_features = self.vectorizer.get_embeddings(input_list=kw_list)

        results = self.lsh.query(kw_features)
        print("num results", len(results))

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
