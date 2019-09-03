import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.saved_model import simple_save, load, load_v2, tag_constants
import logging
import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


# TODO Migrate tf.v1 code to v2 in next release
class SentenceEncoder(object):
    def __init__(self, model_path=None):
        if model_path is None:
            model_name = "SentEnL"
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), model_name
            )

        g = tf.Graph()
        with g.as_default():
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module(model_path)
            self.encoder = embed(self.text_input)
            init_op = tf.group(
                [tf.global_variables_initializer(), tf.tables_initializer()]
            )
        g.finalize()

        # Create session and initialize.
        self.session = tf.Session(graph=g)
        self.session.run(init_op)

    def get_embedding_vector(self, input_list):
        embeddings = self.session.run(
            self.encoder, feed_dict={self.text_input: input_list}
        )

        return embeddings


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SaveTFModel(object):
    def save_model(self):

        model_name = "test_sent"
        import_model_name = "SentEnL"
        export_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), model_name
        )

        import_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), import_model_name
        )
        with tf.Session(graph=tf.Graph()) as sess:
            module = hub.Module(import_model_path)
            input_params = module.get_input_info_dict()
            # take a look at what tensor does the model accepts - 'text' is input tensor name

            text_input = tf.placeholder(
                name="text",
                dtype=input_params["text"].dtype,
                shape=input_params["text"].get_shape(),
            )
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            embeddings = module(text_input)

            simple_save(
                sess,
                export_model_path,
                inputs={"text": text_input},
                outputs={"embeddings": embeddings},
                legacy_init_op=tf.tables_initializer(),
            )


if __name__ == "__main__":
    saveTF = SaveTFModel()
    saveTF.save_model()
