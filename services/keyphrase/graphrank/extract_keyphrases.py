import pandas as pd
from datetime import datetime
import iso8601
import time
import networkx as nx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import logging

from .graph_rank import GraphRank
from .utils import TextPreprocess, GraphUtils

logger = logging.getLogger(__name__)


class KeyphraseExtractor(object):
    def __init__(self):
        self.stop_words = list(STOP_WORDS)
        self.nlp = spacy.load('vendor/en_core_web_sm/en_core_web_sm-2.1.0')
        self.gr = GraphRank()
        self.tp = TextPreprocess()
        self.gutils = GraphUtils()
        self.meeting_graph = nx.Graph()
        self.syntactic_filter = ['JJ', 'JJR', 'JJS',
                                 'NN', 'NNP', 'NNS', 'VB', 'VBP', 'NNPS', 'FW']

    def formatTime(self, tz_time):
        isoTime = iso8601.parse_date(tz_time)
        ts = isoTime.timestamp()
        ts = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S:%f")
        return ts

    def reformat_input(self, input_json):
        systime = time.strftime('%d-%m-%Y-%H-%M')

        json_df_ts = pd.DataFrame(input_json['segments'], index=None)
        json_df_ts['id'] = json_df_ts['id'].astype(str)
        json_df_ts['filteredText'] = json_df_ts['filteredText'].apply(
            lambda x: str(x))
        json_df_ts['originalText'] = json_df_ts['originalText'].apply(
            lambda x: str(x))
        json_df_ts['createdAt'] = json_df_ts['createdAt'].apply(
            lambda x: self.formatTime(x))
        json_df_ts['endTime'] = json_df_ts['endTime'].apply(
            lambda x: self.formatTime(x))
        json_df_ts['startTime'] = json_df_ts['startTime'].apply(
            lambda x: self.formatTime(x))
        json_df_ts['updatedAt'] = json_df_ts['updatedAt'].apply(
            lambda x: self.formatTime(x))
        json_df_ts = json_df_ts.sort_values(['createdAt'], ascending=[1])

        return json_df_ts

    def read_segments(self, segment_df):
        text_list = []
        for i in range(len(segment_df)):
            segment_text = segment_df.iloc[i]['originalText']
            text_list.append(segment_text)
        return text_list

    def process_text(self, text, filter_by_pos=True, stop_words=False, syntactic_filter=None):
        original_tokens, pos_tuple, filtered_pos_tuple = self.tp.preprocess_text(text,
                                                                                 filter_by_pos=filter_by_pos,
                                                                                 pos_filter=syntactic_filter,
                                                                                 stop_words=stop_words)

        return original_tokens, pos_tuple, filtered_pos_tuple

    def initialize_meeting_graph(self, context_id, instance_id):
        self.meeting_graph = nx.Graph(
            context_instance_id=instance_id, context_id=context_id)
        pass

    def build_custom_graph(self, text_list, window=4, preserve_common_words=False, syntactic_filter=None):

        for i in range(len(text_list)):
            text = text_list[i]
            try:
                original_tokens, pos_tuple, filtered_pos_tuple = self.process_text(
                    text)
                self.meeting_graph = self.gr.build_word_graph(input_pos_text=filtered_pos_tuple,
                                                              original_tokens=original_tokens,
                                                              window=window,
                                                              syntactic_filter=syntactic_filter,
                                                              preserve_common_words=preserve_common_words)
            except Exception as e:
                logger.error(
                    "Could not process the sentence: ErrorMsg: {}".format(e))

    def get_custom_keyphrases(self, graph,
                              pos_tuple=None,
                              token_list=None,
                              window=4,
                              normalize_nodes_fn='degree',
                              preserve_common_words=False,
                              normalize_score=False):

        keyphrases = self.gr.get_keyphrases(graph_obj=graph,
                                            input_pos_text=pos_tuple,
                                            original_tokens=token_list,
                                            window=window,
                                            normalize_nodes=normalize_nodes_fn,
                                            preserve_common_words=preserve_common_words,
                                            normalize_score=normalize_score)

        return keyphrases

    def get_entities(self, input_segment):
        spacy_list = ["PRODUCT", "EVENT", "LOC",
                      "ORG", "PERSON", "WORK_OF_ART"]
        comprehend_list = ["COMMERCIAL_ITEM", "EVENT",
                           "LOCATION", "ORGANIZATION", "PERSON", "TITLE"]
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
            entity_dict.append({'text': str(entt[0]), 'type': entt[1]})

        return filtered_entities

    def segment_search(self, input_json, keyphrase_list, top_n=None):
        """
        Search for keyphrases in the top-5 PIM segments and return them for each segment
        Args:
            input_json:
            keyphrase_list:

        Returns:

        """
        result = {}
        for i in range(len(input_json['segments'])):
            sort_list = []

            entity_segment = input_json['segments'][i].get('originalText')
            input_segment = input_json['segments'][i].get(
                'originalText').lower()
            keywords_list = []

            for tup in keyphrase_list:
                kw = tup[0]
                score = tup[1]

                result = input_segment.find(kw)
                if result > -1:
                    keywords_list.append((kw, score))

            sort_list = self.gutils.sort_by_value(keywords_list, order='desc')
            if top_n is not None:
                sort_list = sort_list[:-1]

            segment_entity = self.get_entities(entity_segment)
            segment_keyword_list = [words for words, score in sort_list]

            # Remove the first occurrence of entity in the list of keyphrases
            for entities in segment_entity:
                if entities.lower() in segment_keyword_list:
                    segment_keyword_list.remove(entities.lower)

            segment_entity.extend(segment_keyword_list)
            segment_output = segment_entity

            result = {
                "keyphrases": segment_output
            }

        return result

    def chapter_segment_search(self, input_json, keyphrase_list, top_n=None):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            input_json:
            keyphrase_list:

        Returns:

        """
        chapter_keywords_list = []
        chapter_entities = []
        for i in range(len(input_json['segments'])):
            entity_segment = input_json['segments'][i].get('originalText')
            input_segment = input_json['segments'][i].get(
                'originalText').lower()

            for tup in keyphrase_list:
                kw = tup[0]
                score = tup[1]

                result = input_segment.find(kw)
                if result > -1:
                    chapter_keywords_list.append((kw, score))

            chapter_entities.extend(self.get_entities(entity_segment))

        sort_list = self.gutils.sort_by_value(
            chapter_keywords_list, order='desc')
        if top_n is not None:
            sort_list = sort_list[:top_n]

        chapter_keyphrases = [phrases for phrases, score in sort_list]

        # Remove the first occurrence of entity in the list of keyphrases
        for entities in chapter_entities:
            if entities.lower() in chapter_keyphrases:
                chapter_keyphrases.remove(entities.lower)

        chapter_entities.extend(chapter_keyphrases)
        chapter_output = chapter_entities

        result = {
            "keyphrases": chapter_output
        }

        return result

    def populate_word_graph(self, req_data):
        segment_df = self.reformat_input(req_data)

        text_list = self.read_segments(segment_df=segment_df)
        self.build_custom_graph(text_list=text_list)

        logger.info("Number of nodes: {}; Number of edges: {}".format(self.meeting_graph.number_of_nodes(),
                                                                      self.meeting_graph.number_of_edges()))

    def compute_keyphrases(self):
        keyphrase_list = []
        try:
            keyphrase_list = self.get_custom_keyphrases(
                graph=self.meeting_graph)
            # logger.debug("Complete keyphrases: {}".format(keyphrase_list))

        except Exception as e:
            logger.debug("ErrorMsg: {}".format(e))

        return keyphrase_list

    def _get_pim_keyphrases(self, req_data, n_kw=5):
        keyphrase_list = self.compute_keyphrases()
        segment_keyphrases = []
        try:
            segment_keyphrases = self.segment_search(
                input_json=req_data, keyphrase_list=keyphrase_list, top_n=n_kw)
        except Exception as e:
            logger.debug("ErrorMsg: {}".format(e))

        return segment_keyphrases

    def _get_chapter_keyphrases(self, req_data, n_kw=5):
        keyphrase_list = self.compute_keyphrases()
        chapter_keyphrases = []
        try:
            chapter_keyphrases = self.chapter_segment_search(
                input_json=req_data, keyphrase_list=keyphrase_list, top_n=n_kw)
        except Exception as e:
            logger.debug("ErrorMsg: {}".format(e))

        return chapter_keyphrases

    def get_keyphrases(self, req_data, n_kw=10):
        segments_array = req_data['segments']

        # Re-populate the graph if google transcripts are coming in
        self.populate_word_graph(req_data)

        # Decide between PIM or Chapter keyphrases
        if len(segments_array) > 1:
            logger.info("Publishing Chapter Keyphrases")
            keyphrases = self._get_chapter_keyphrases(req_data, n_kw=n_kw)
        else:
            logger.info("Publishing PIM Keyphrases")
            keyphrases = self._get_pim_keyphrases(req_data, n_kw=n_kw)

        return keyphrases

    def get_instance_keyphrases(self, req_data, n_kw=10):
        keyphrase_list = self.compute_keyphrases()
        instance_keyphrases = [words for words, score in keyphrase_list]

        result = {
            "keyphrases": instance_keyphrases[:n_kw]
        }

        return result

    def reset_keyphrase_graph(self, req_data):
        logger.debug(
            "Before Reset - Number of nodes: {}; Number of edges: {}".format(self.meeting_graph.number_of_nodes(),
                                                                             self.meeting_graph.number_of_edges()))

        self.gr.reset_graph()
        self.meeting_graph.clear()
        logger.info("Number of nodes: {}; Number of edges: {}".format(self.meeting_graph.number_of_nodes(),
                                                                      self.meeting_graph.number_of_edges()))

        return {'result': 'done', 'message': 'reset successful'}
