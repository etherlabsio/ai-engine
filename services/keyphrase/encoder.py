import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)


class SentenceEncoder(object):
    def __init__(self):
        large_module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.embed = hub.Module(module_url)

        self.input_placeholder = tf.placeholder(tf.string, shape=(None))
        self.encoder = self.embed(self.input_placeholder)

    def get_embeddings(self, session, word_list):
        embeddings = session.run(
            self.encoder, feed_dict={self.input_placeholder: word_list}
        )

        return embeddings

    def get_embedding_distance(self, session, word_list):
        embeddings = session.run(
            self.encoder, feed_dict={self.input_placeholder: word_list}
        )
        dist = cosine(np.array(embeddings[0]), np.array(embeddings[1]))

        return dist

    def get_segment_word_similarity(self, session, segment, word):
        input_list = [segment, word]
        cosine_dist = self.get_embedding_distance(session, input_list)

        return 1 - cosine_dist
