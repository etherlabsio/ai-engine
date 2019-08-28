import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


class SentenceEncoder(object):
    def __init__(self):
        module_name = "USE_model_v1"
        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), module_name)
        logger.info("Loading model from path", extra={"path": module_path})
        self.embed = hub.Module(module_path)

        self.input_placeholder = tf.placeholder(tf.string, shape=(None))
        self.encoder = self.embed(self.input_placeholder)

    def get_embedding_vector(self, input_list):
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embeddings = session.run(self.encoder, feed_dict={self.input_placeholder: input_list})

        return embeddings
