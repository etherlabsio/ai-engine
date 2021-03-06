import logging
import nltk
from nltk.corpus import stopwords
import pickle
from typing import List
import networkx as nx

logger = logging.getLogger(__name__)


stop_words = set(stopwords.words("english"))
stop_words_spacy = list(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere n't

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves yeah okay
""".split()
)
stop_words = set(list(stop_words) + list(stop_words_spacy))


class GraphFilter(object):
    def __init__(self, s3_client=None):
        self.s3_client = s3_client
        # self.mind_store = mind_store

    def download_mind(self, graph_file_path=None) -> nx.Graph:
        graph_object = self.s3_client.download_file(file_name=graph_file_path)
        graph_object_str = graph_object["Body"].read()
        mind_graph = pickle.loads(graph_object_str)

        logger.info("Downloaded ENT-KP graph")

        return mind_graph

    def get_segment_nouns(self, segment_text_list):
        segment_noun_list = []
        for text in list(segment_text_list):
            tagged_sents = nltk.pos_tag_sents(
                nltk.word_tokenize(text) for sent in nltk.sent_tokenize(text)
            )
            text_kps = []
            for tagged_sent in tagged_sents:
                text_kps.extend(
                    [ele[0] for ele in list(tagged_sents[0]) if ele[1].startswith("NN")]
                )
            segment_noun_list.append(text_kps)

        return segment_noun_list

    def get_ent(self, ent):
        return set(ent.lower().split(" "))

    def get_kp_nouns(self, kp, sent_nouns):
        return set(kp.lower().split(" ")) & set(sent_nouns)

    def filter_keyphrases(
        self, phrase: List, segment_text_list: List, kp_graph: nx.Graph
    ):
        filtered_kp_list = []
        dropped_kp_list = []

        segment_noun_list = self.get_segment_nouns(segment_text_list=segment_text_list)

        for text, nouns in zip(segment_text_list, segment_noun_list):
            sent_nouns = [ele.lower() for ele in nouns]

            for kps in phrase:
                kp_nouns = self.get_kp_nouns(kps, sent_nouns)
                if set(kp_nouns) & set(kp_graph) == set(kp_nouns):
                    filtered_kp_list.append(kps)
                else:
                    dropped_kp_list.append(kps)

        return filtered_kp_list, dropped_kp_list

    def filter_entities(
        self, phrase: List, segment_text_list: List, kp_graph: nx.Graph
    ):
        filtered_ent_list = []
        dropped_ent_list = []

        for text in segment_text_list:
            for ents in phrase:
                ent = self.get_ent(ents)
                if set(ent) & set(kp_graph) == set(ent):
                    filtered_ent_list.append(ents)
                else:
                    dropped_ent_list.append(ents)

        return filtered_ent_list, dropped_ent_list
