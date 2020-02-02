import networkx as nx
import logging
import traceback
import json
from timeit import default_timer as timer
from typing import Tuple, List, Text, Dict, Union

from .objects import Entity, Score, Keyphrase

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
        (original_tokens, pos_tuple, filtered_pos_tuple,) = self.tp.preprocess_text(
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
    ) -> nx.Graph:
        meeting_word_graph = graph
        for text in text_list:
            original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(text)
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
    ) -> Tuple[Text, float]:

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

    def get_custom_entities(self, input_segment: Text) -> List[Entity]:
        entity_list = []
        entity_preference_map = {
            "MISC": 1,
            "ORG": 2,
            "PER": 3,
            "LOC": 4,
            "O": 5,
        }

        entity_dict, entity_label_dict = self.call_custom_ner(
            input_segment=input_segment
        )

        for entity, conf_score in entity_dict.items():
            entity_label = entity_label_dict[entity]
            entity_list.append(
                Entity(
                    originalForm=entity,
                    label=entity_label,
                    preference=entity_preference_map[entity_label],
                    score=Score(),
                    confidence_score=conf_score,
                )
            )

            logger.debug(
                "Obtained entities",
                extra={"entities": entity, "label": entity_label, "score": conf_score},
            )

        return entity_list

    def call_custom_ner(
        self, input_segment: Text
    ) -> Tuple[Union[Dict, str], Union[Dict, str]]:

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
                invoke_response["Payload"].read().decode("utf8").replace("'", '"')
            )
            response = json.loads(lambda_output)
            status_code = response["statusCode"]
            response_body = response["body"]

            end = timer()
            if status_code == 200:
                entity_dict = json.loads(response_body)["entities"]
                entity_label_dict = json.loads(response_body)["labels"]
                logger.info(
                    "Received response from encoder lambda function",
                    extra={
                        "num": len(entity_dict.keys()),
                        "lambdaResponseTime": end - start,
                    },
                )

            else:
                entity_dict = {}
                entity_label_dict = {}
                logger.warning(
                    "Invalid response from encoder lambda function",
                    extra={"lambdaResponseTime": end - start},
                )

            return entity_dict, entity_label_dict

        except Exception as e:
            logger.error("Invoking failed", extra={"err": e})

            entity_dict = {}
            entity_label_dict = {}
            return entity_dict, entity_label_dict

    def get_segment_keyphrases(
        self, segment_text: str, word_graph: nx.Graph
    ) -> List[Keyphrase]:

        keyphrase_list = []
        try:
            original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(
                segment_text
            )

            descriptive_kp = self.get_custom_keyphrases(
                graph=word_graph,
                pos_tuple=pos_tuple,
                descriptive=True,
                post_process_descriptive=True,
            )
            non_descriptive_kp = self.get_custom_keyphrases(
                graph=word_graph, pos_tuple=pos_tuple
            )

            descriptive_kp_obj = [
                Keyphrase(
                    originalForm=word, type="descriptive", score=Score(pagerank=pg_rank)
                )
                for word, pg_rank in descriptive_kp
            ]
            non_descriptive_kp_obj = [
                Keyphrase(
                    originalForm=word, type="original", score=Score(pagerank=pg_rank)
                )
                for word, pg_rank in non_descriptive_kp
            ]
            keyphrase_list.extend(descriptive_kp_obj)
            keyphrase_list.extend(non_descriptive_kp_obj)

        except Exception:
            logger.error(
                "error retrieving descriptive phrases",
                extra={"err": traceback.print_exc()},
            )

        return keyphrase_list
