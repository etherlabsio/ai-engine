import tensorflow.compat.v1 as tf
from timeit import default_timer as timer
import logging
from scipy.spatial.distance import cosine
import numpy as np
import os

# from .encoder import SentenceEncoder
# from .knowledge_graph import KnowledgeGraph

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


class KeyphraseRanker(object):
    def __init__(self, s3_io_util, context_dir, encoder_object, knowledge_graph_object):
        self.sentence_encoder = encoder_object
        self.kg = knowledge_graph_object
        self.context_dir = context_dir
        self.io_util = s3_io_util

    def compute_edge_weights(self, word_graph):

        start = timer()
        counter = 0
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            for node1, node2, attr in word_graph.edges.data("edge_emb_wt"):
                if attr is None:
                    node_list = [node1, node2]
                    emb_edge_weight = 1 - self.sentence_encoder.get_embedding_distance(
                        session, node_list
                    )
                    word_graph.add_edge(node1, node2, edge_emb_wt=emb_edge_weight)
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
        dict_key="descriptive",
        normalize: bool = False,
        norm_limit: int = 4,
    ):

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            for i, kp_dict in enumerate(keyphrase_object):
                segment = kp_dict["segments"]
                keyphrase_dict = kp_dict[dict_key]
                input_list = [segment]
                input_list.extend([words for words, values in keyphrase_dict.items()])
                input_embedding_vector = self.sentence_encoder.get_embeddings(
                    session, input_list
                )
                segment_vector = input_embedding_vector[0]

                segment_relevance_score_list = []
                for j, (phrase, values) in enumerate(keyphrase_dict.items()):
                    phrase_len = len(phrase.split())
                    seg_score = 1 - cosine(
                        segment_vector, input_embedding_vector[j + 1]
                    )

                    if normalize:
                        if phrase_len > norm_limit:
                            norm_seg_score = seg_score / (phrase_len - (norm_limit - 1))
                        else:
                            norm_seg_score = seg_score / phrase_len
                    else:
                        norm_seg_score = seg_score

                    assert keyphrase_dict[phrase][1] == 0
                    keyphrase_dict[phrase][1] = norm_seg_score
                    segment_relevance_score_list.append(norm_seg_score)

                segment_confidence_score = np.mean(segment_relevance_score_list)
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
