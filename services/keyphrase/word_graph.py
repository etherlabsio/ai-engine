import networkx as nx
import logging
import traceback
import json
from timeit import default_timer as timer
from typing import Tuple

logger = logging.getLogger(__name__)


class WordGraphBuilder(object):
    def __init__(
        self,
        graphrank_obj,
        textpreprocess_obj,
        graphutils_obj,
        keyphrase_utils_obj,
        lambda_client,
        ner_lambda_function,
    ):
        self.gr = graphrank_obj
        self.tp = textpreprocess_obj
        self.gutils = graphutils_obj
        self.utils = keyphrase_utils_obj
        self.lambda_client = lambda_client
        self.ner_lambda_function = ner_lambda_function

    def process_text(
        self, text, filter_by_pos=True, stop_words=False, syntactic_filter=None
    ):
        original_tokens, pos_tuple, filtered_pos_tuple = self.tp.preprocess_text(
            text,
            filter_by_pos=filter_by_pos,
            pos_filter=syntactic_filter,
            stop_words=stop_words,
        )

        return original_tokens, pos_tuple, filtered_pos_tuple

    def build_custom_graph(
        self,
        graph,
        text_list,
        window=4,
        preserve_common_words=False,
        syntactic_filter=None,
        add_context=False,
    ):
        meeting_word_graph = graph
        for text in text_list:
            original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(
                text
            )
            meeting_word_graph = self.gr.build_word_graph(
                graph_obj=graph,
                input_pos_text=pos_tuple,
                window=window,
                syntactic_filter=syntactic_filter,
                preserve_common_words=preserve_common_words,
                add_context=add_context,
            )
        return meeting_word_graph

    def get_custom_keyphrases(
        self,
        graph,
        pos_tuple=None,
        original_pos_list=None,
        window=4,
        normalize_nodes_fn="degree",
        preserve_common_words=False,
        normalize_score=False,
        descriptive=False,
        post_process_descriptive=False,
    ):

        keyphrases = self.gr.get_keyphrases(
            graph_obj=graph,
            input_pos_text=pos_tuple,
            original_tokens=original_pos_list,
            window=window,
            normalize_nodes=normalize_nodes_fn,
            preserve_common_words=preserve_common_words,
            normalize_score=normalize_score,
            descriptive=descriptive,
            post_process_descriptive=post_process_descriptive,
        )

        return keyphrases

    def get_custom_entities(self, input_segment):
        entity_list = []

        entity_dict = self.call_custom_ner(input_segment=input_segment)

        for entity, conf_score in entity_dict.items():
            entity_list.append(
                {"text": entity, "preference": 1, "conf_score": conf_score}
            )

        return entity_list

    def call_custom_ner(self, input_segment):

        start = timer()
        lambda_payload = {"body": {"originalText": input_segment}}

        try:
            logger.info("Invoking NER lambda")
            invoke_response = self.lambda_client.invoke(
                FunctionName=self.ner_lambda_function,
                InvocationType="RequestResponse",
                Payload=json.dumps(lambda_payload),
            )

            lambda_output = (
                invoke_response["Payload"]
                .read()
                .decode("utf8")
                .replace("'", '"')
            )
            response = json.loads(lambda_output)
            status_code = response["statusCode"]
            response_body = response["body"]

            end = timer()
            if status_code == 200:
                entity_dict = json.loads(response_body)["entities"]
                logger.info(
                    "Received response from encoder lambda function",
                    extra={
                        "num": len(entity_dict.keys()),
                        "lambdaResponseTime": end - start,
                    },
                )

            else:
                entity_dict = {}
                logger.warning(
                    "Invalid response from encoder lambda function",
                    extra={"lambdaResponseTime": end - start},
                )

            return entity_dict

        except Exception as e:
            logger.error("Invoking failed", extra={"err": e})

            entity_dict = {}
            return entity_dict

    def get_segment_keyphrases(
        self, segment_object: dict, word_graph: nx.Graph
    ) -> Tuple[list, list]:

        keyphrase_list = []
        descriptive_keyphrase_list = []
        segment_list = self.utils.read_segments(segment_object=segment_object)
        try:
            for text in segment_list:
                original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(
                    text
                )

                keyphrase_list.extend(
                    self.get_custom_keyphrases(
                        graph=word_graph, pos_tuple=pos_tuple
                    )
                )
                descriptive_keyphrase_list.extend(
                    self.get_custom_keyphrases(
                        graph=word_graph,
                        pos_tuple=pos_tuple,
                        descriptive=True,
                        post_process_descriptive=True,
                    )
                )
        except Exception:
            logger.error(
                "error retrieving descriptive phrases",
                extra={"err": traceback.print_exc()},
            )

        return keyphrase_list, descriptive_keyphrase_list
