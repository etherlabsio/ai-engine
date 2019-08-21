import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy as np
from scipy.spatial.distance import cosine
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


class SentenceEncoder(object):
    def __init__(self):
        module_name = "USE_model_v1"
        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), module_name)
        logger.info("Loading model from path", extra={
            "path": module_path
        })
        self.embed = hub.Module(module_path)

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
