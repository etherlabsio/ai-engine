import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import networkx as nx
import logging
import traceback
from typing import Tuple

logger = logging.getLogger(__name__)


class WordGraphBuilder(object):
    def __init__(self, graphrank_obj, textpreprocess_obj, graphutils_obj, keyphrase_utils_obj):
        self.nlp = spacy.load("vendor/en_core_web_sm/en_core_web_sm-2.1.0")
        self.stop_words = list(STOP_WORDS)
        self.gr = graphrank_obj
        self.tp = textpreprocess_obj
        self.gutils = graphutils_obj
        self.utils = keyphrase_utils_obj

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

    def get_entities(self, input_segment):
        spacy_list = ["PRODUCT", "EVENT", "LOC", "ORG", "PERSON", "WORK_OF_ART"]
        comprehend_list = [
            "COMMERCIAL_ITEM",
            "EVENT",
            "LOCATION",
            "ORGANIZATION",
            "PERSON",
            "TITLE",
        ]
        match_dict = dict(zip(spacy_list, comprehend_list))

        doc = self.nlp(input_segment)
        t_noun_chunks = list(set(list(doc.noun_chunks)))
        filtered_entities = []
        t_ner_type = []

        for ent in list(set(doc.ents)):
            t_type = ent.label_
            if t_type in list(match_dict) and str(ent) not in filtered_entities:
                t_ner_type.append(match_dict[t_type])
                filtered_entities.append(str(ent))
        t_noun_chunks = [str(item).strip().lower() for item in t_noun_chunks]

        # remove stop words from noun_chunks/NERs
        entity_dict = []
        for entt in list(zip(filtered_entities, t_ner_type)):
            entity_dict.append({"text": str(entt[0]), "type": entt[1]})

        return filtered_entities

    def get_segment_keyphrases(
        self, req_data: dict, word_graph: nx.Graph
    ) -> Tuple[list, list]:
        keyphrase_list = []
        descriptive_keyphrase_list = []
        segment_list = self.utils.read_segments(req_data=req_data)
        try:
            for text in segment_list:
                original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(text)

                keyphrase_list.extend(
                    self.get_custom_keyphrases(graph=word_graph, pos_tuple=pos_tuple)
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
