from __future__ import absolute_import, division, print_function

import math
import os
import networkx as nx
import logging
from nltk import WordNetLemmatizer, word_tokenize

from keyphrase.graphrank.metrics import GraphSolvers, WeightMetrics
from keyphrase.graphrank.utils import GraphUtils, TextPreprocess
from keyphrase.graphrank.dgraph import update_graph

logger = logging.getLogger(__name__)


class GraphRank(object):
    def __init__(self):
        self.graph = nx.Graph(type="keyphrases")
        self.graph_utils = GraphUtils()
        self.graph_solver = GraphSolvers()
        self.metric_object = WeightMetrics()
        self.preprocess_text = TextPreprocess()

        self.lemma = WordNetLemmatizer()

        # Store the original text and maintain the context flow to extend the graph.
        self.context = []

        # Load pkg word list
        root_dir = os.getcwd()
        local_file = os.path.realpath(
            os.path.join(root_dir, os.path.dirname(__file__))
        )

        stop_word_file = os.path.join(local_file, "long_stopwords.txt")
        text_file = open(stop_word_file, "r")
        self.common_words = text_file.read().split()

    def build_word_graph(
        self,
        input_pos_text,
        window=2,
        syntactic_filter=None,
        reset_graph_context=False,
        preserve_common_words=False,
        node_attributes=None,
        edge_attributes=None,
        preserve_plurals=False,
    ):
        """
        Build co-occurrence of words graph based on the POS tags and the window of occurrence
        Args:
            edge_attributes:
            preserve_plurals:
            node_attributes:
            preserve_common_words:
            reset_graph_context:
            input_pos_text: List of list of tuple(word_token, POS)
            window:
            syntactic_filter: POS tag filter

        Returns:
            cooccurrence_graph (Networkx graph obj): Graph of co-occurring keywords
        """
        if syntactic_filter is None:
            syntactic_filter = [
                "JJ",
                "JJR",
                "JJS",
                "NN",
                "NNP",
                "NNS",
                "VB",
                "VBP",
                "NNPS",
                "FW",
            ]

        original_token_pos_list = [
            (word.lower(), pos)
            for sent in input_pos_text
            for word, pos in sent
        ]

        # Extend the context of the graph
        if reset_graph_context:
            self.reset_graph()
        self.context.extend(original_token_pos_list)

        if preserve_common_words:
            common_words = []
        else:
            common_words = self.common_words

        # Filter input based on common words and Flatten it
        if preserve_plurals:
            filtered_pos_list = [
                (word.lower(), pos)
                for sent in input_pos_text
                for word, pos in sent
                if pos in syntactic_filter and word.lower() not in common_words
            ]
        else:
            filtered_pos_list = [
                (word.lower(), pos)
                for sent in input_pos_text
                for word, pos in sent
                if pos in syntactic_filter and word.lower() not in common_words
            ]

            filtered_pos_list = [
                (self.lemma.lemmatize(word), pos)
                if pos == "NNS"
                else (word, pos)
                for word, pos in filtered_pos_list
            ]

        # Add nodes
        if node_attributes is not None:
            self.graph.add_nodes_from(
                [(word, node_attributes) for word, pos in filtered_pos_list]
            )
        else:
            self.graph.add_nodes_from(
                [word for word, pos in filtered_pos_list]
            )

        # Add edges
        # TODO Consider unfiltered token list to build cooccurrence edges.
        for i, (node1, pos) in enumerate(filtered_pos_list):
            if node1 in self.graph.nodes():

                for j in range(i + 1, min(i + window, len(filtered_pos_list))):
                    node2, pos2 = filtered_pos_list[j]
                    if node2 in self.graph.nodes() and node1 != node2:
                        self.graph.add_edge(node1, node2, weight=1.0)
            else:
                continue

        cooccurence_graph = self.graph

        return cooccurence_graph

    def node_weighting(
        self,
        graph_obj,
        input_pos_text=None,
        window=2,
        top_t_percent=None,
        solver="pagerank_scipy",
        syntactic_filter=None,
        normalize_nodes=None,
    ):
        """
        Computes the weights of the vertices/nodes of the graph based on the `solver` algorithm.
        Args:
            normalize_nodes:
            syntactic_filter:
            graph_obj:
            input_pos_text:
            window:
            top_t_percent:
            solver: solver function to compute node scores. Defaults to `pagerank_scipy`

        Returns:
            node_weights (dict): Dict of nodes (keys) and their weights (values)
            top_words (list): List of tuple of (top nodes, scores)
        """

        # Build word graph
        if graph_obj is None and input_pos_text is not None:
            graph_obj = self.build_word_graph(
                input_pos_text=input_pos_text,
                window=window,
                syntactic_filter=syntactic_filter,
            )
        elif graph_obj is None and input_pos_text is None:
            raise SyntaxError(
                "Both `graph_obj` and `input_pos_text` cannot be `None`"
            )

        # Compute node scores using unweighted pagerank implementation
        # TODO Extend to other solvers
        node_weights = self.graph_solver.get_graph_algorithm(
            graph_obj=graph_obj, solver_fn=solver
        )

        # Normalize node weights using graph properties
        normalized_node_weights = self.graph_solver.normalize_nodes(
            graph_obj=graph_obj,
            node_weights=node_weights,
            normalize_fn=normalize_nodes,
        )
        # sorting the nodes by decreasing scores
        top_words = self.graph_utils.sort_by_value(
            normalized_node_weights.items(), order="desc"
        )

        if top_t_percent is not None:
            # warn user
            logging.warning(
                "Candidates are generated using {}-top".format(top_t_percent)
            )

            # computing the number of top keywords
            n_nodes = self.graph.number_of_nodes()
            nodes_to_keep = min(math.floor(n_nodes * top_t_percent), n_nodes)

            # creating keyphrases from the T-top words
            top_words = top_words[: int(nodes_to_keep)]

        return normalized_node_weights, top_words

    def retrieve_multi_keyterms(
        self,
        graph_obj,
        input_pos_text=None,
        original_tokens=None,
        window=2,
        syntactic_filter=None,
        top_t_percent=None,
        preserve_common_words=False,
        normalize_nodes=None,
    ):
        """
        Search for co-occurring keyword terms and place them together as multi-keyword terms.
        Args:
            original_tokens:
            normalize_nodes:
            preserve_common_words:
            graph_obj:
            input_pos_text:
            window:
            syntactic_filter:
            top_t_percent:

        Returns:
            multi_terms (list): List of unique tuples of (list[co-occurring keyterms], list[node scores])
        """

        node_weights, top_weighted_words = self.node_weighting(
            graph_obj=graph_obj,
            input_pos_text=input_pos_text,
            window=window,
            top_t_percent=top_t_percent,
            syntactic_filter=syntactic_filter,
            normalize_nodes=normalize_nodes,
        )

        tmp_keywords = [word for word, we in node_weights.items()]

        # Check if the graph is to be extended or created from smaller context (segments)
        if original_tokens is None:
            original_tokens = self.context

        unfiltered_word_tokens = [
            token.lower() for token, pos in original_tokens
        ]

        plural_word_tokens = [
            token.lower() for token, pos in original_tokens if pos == "NNS"
        ]

        marked_text_tokens = self._tag_text_for_keywords(
            original_token_list=unfiltered_word_tokens,
            keyword_list=tmp_keywords,
            plural_word_list=plural_word_tokens,
        )

        multi_terms = []
        current_term_units = []
        scores_list = []

        if preserve_common_words:
            common_words = []
        else:
            common_words = self.common_words

        # use space to construct multi-word term later
        for token, marker in marked_text_tokens:
            # Don't include stopwords in post-processing
            # TODO Add better ways to combine words to make phrases: grammar rules, n-grams etc.
            if marker == "k" and token not in common_words:
                current_term_units.append(token)
                scores_list.append(node_weights[token])
            elif marker == "p":
                current_term_units.append(token)
                # Use its singular form's score in keyphrase weighting
                root_token = self.lemma.lemmatize(token)
                scores_list.append(node_weights[root_token])
            else:
                # Get unique nodes
                if (
                    current_term_units
                    and (current_term_units, scores_list) not in multi_terms
                ):
                    multi_terms.append((current_term_units, scores_list))
                # reset for next term candidate
                current_term_units = []
                scores_list = []

        return multi_terms

    def compute_multiterm_score(
        self,
        graph_obj,
        original_tokens=None,
        input_pos_text=None,
        window=2,
        top_t_percent=None,
        weight_metrics="sum",
        normalize_score=False,
        syntactic_filter=None,
        preserve_common_words=True,
        normalize_nodes=None,
    ):
        """
        Compute aggregated scores for multi-keyword terms. The scores are computed based on the weight metrics.
        The final scores for a keyword term determines its relative importance in the list of phrases.
        Args:
            normalize_nodes:
            preserve_common_words:
            original_tokens:
            graph_obj:
            input_pos_text:
            window:
            syntactic_filter:
            top_t_percent:
            weight_metrics:
            normalize_score (bool):

        Returns:
            multi_keywords (list[list])): List of list of keywords and/or multiple keyword terms
            multi_term_scores (list): Weighted scores for each of the list of multi-keyword terms

        """

        multi_keyterms = self.retrieve_multi_keyterms(
            graph_obj=graph_obj,
            input_pos_text=input_pos_text,
            original_tokens=original_tokens,
            window=window,
            top_t_percent=top_t_percent,
            syntactic_filter=syntactic_filter,
            preserve_common_words=preserve_common_words,
            normalize_nodes=normalize_nodes,
        )

        # TODO extend to more weighting metrics
        # TODO add support for normalization of scores based on word length, degree, betweenness or other factors
        # Decide the criteria to score the multi-word terms
        multi_term_scores = [
            self.metric_object.compute_weight_fn(
                weight_metrics=weight_metrics,
                key_terms=key_terms,
                score_list=scores,
                normalize=normalize_score,
            )
            for key_terms, scores in multi_keyterms
        ]
        multi_keywords = [key_terms for key_terms, scores in multi_keyterms]

        return multi_keywords, multi_term_scores

    def get_keyphrases(
        self,
        graph_obj,
        original_tokens=None,
        input_pos_text=None,
        window=2,
        top_t_percent=None,
        weight_metrics="sum",
        normalize_score=False,
        top_n=None,
        syntactic_filter=None,
        preserve_common_words=False,
        normalize_nodes=None,
        post_process=True,
    ):
        """
        Get `top_n` keyphrases from the word graph.
        Args:
            post_process:
            normalize_nodes:
            preserve_common_words:
            syntactic_filter:
            original_tokens:
            graph_obj:
            input_pos_text:
            window:
            top_t_percent:
            weight_metrics:
            normalize_score:
            top_n:

        Returns:
            sorted_keyphrases (list): Keyphrases in descending order of their weighted scores.
        """

        multi_keywords, multi_term_score = self.compute_multiterm_score(
            graph_obj=graph_obj,
            input_pos_text=input_pos_text,
            original_tokens=original_tokens,
            window=window,
            top_t_percent=top_t_percent,
            weight_metrics=weight_metrics,
            normalize_score=normalize_score,
            syntactic_filter=syntactic_filter,
            preserve_common_words=preserve_common_words,
            normalize_nodes=normalize_nodes,
        )

        # Convert list of keywords to form keyphrase/multi-phrases
        keyphrases = [" ".join(terms) for terms in multi_keywords]

        # Create a list of tuples of (keyphrases, weighted_scores)
        scored_keyphrases = list(zip(keyphrases, multi_term_score))

        # Sort the list in a decreasing order
        sorted_keyphrases = self.graph_utils.sort_by_value(
            scored_keyphrases, order="desc"
        )

        if post_process:
            sorted_keyphrases = self.post_process(sorted_keyphrases)

        # Choose `top_n` number of keyphrases, if given
        if top_n is not None:
            sorted_keyphrases = sorted_keyphrases[:top_n]

        return sorted_keyphrases

    def post_process(self, keyphrases):
        """
        Post process to remove duplicate words from single phrases.
        Args:
            keyphrases (list): list of tuple of keyphrases and scores

        Returns:
            processed_keyphrases (list): list of post-processed keyphrases
        """
        processed_keyphrases = []

        # Remove duplicates from the single phrases which are occurring in multi-keyphrases
        multi_phrases = [
            phrases for phrases in keyphrases if len(phrases[0].split()) > 1
        ]
        single_phrase = [
            phrases for phrases in keyphrases if len(phrases[0].split()) == 1
        ]
        for tup in single_phrase:
            kw = tup[0]
            for tup_m in multi_phrases:
                kw_m = tup_m[0]
                r = kw_m.find(kw)
                if r > -1:
                    try:
                        single_phrase.remove(tup)
                    except Exception as e:
                        logger.debug(
                            "No duplicate single-word in a phrase: ",
                            extra={"err": e},
                        )
                        continue

        # Remove duplicates from multi-phrases
        twoplus_multi_phrase = [
            phrases for phrases in keyphrases if len(phrases[0].split()) > 2
        ]
        two_phrase = [
            phrases for phrases in keyphrases if len(phrases[0].split()) == 2
        ]
        for tup in two_phrase:
            kw = tup[0]
            for tup_m in twoplus_multi_phrase:
                kw_m = tup_m[0]
                r = kw_m.find(kw)
                if r > -1:
                    try:
                        two_phrase.remove(tup)
                    except Exception as e:
                        logger.debug(
                            "No multi-words in a phrase: ", extra={"err": e}
                        )
                        continue

        # Remove same word occurrences in a multi-keyphrase
        for multi_key, multi_score in twoplus_multi_phrase:
            kw_m = multi_key.split()
            unique_kp_list = list(dict.fromkeys(kw_m))
            multi_keyphrase = " ".join(unique_kp_list)
            processed_keyphrases.append((multi_keyphrase, multi_score))

        processed_keyphrases.extend(two_phrase)

        # Sort the multi-keyphrases first and then append the single keywords to the tail of the list.
        processed_keyphrases = self.graph_utils.sort_by_value(
            processed_keyphrases, order="desc"
        )

        processed_keyphrases.extend(single_phrase)

        # Remove occurrences of Plurals if their singular form is existing
        new_processed_keyphrases = self._lemmatize_sentence(
            processed_keyphrases
        )

        return new_processed_keyphrases

    def reset_graph(self):
        self.context = []
        self.graph.clear()
        self.graph = nx.Graph(type="keyphrases")

    def populate_dgraph(self, graph_obj, meeting_id):
        update_graph(graph_obj=graph_obj, meetingid=meeting_id)

    def _tag_text_for_keywords(
        self, original_token_list, keyword_list, plural_word_list
    ):
        marked_text_tokens = []
        keyword_tag = "k"
        plural_tag = "p"
        for token in original_token_list:
            if token in plural_word_list:
                stem_form = self.lemma.lemmatize(token)
                if stem_form in keyword_list:
                    marked_text_tokens.append((token, plural_tag))
            elif token in keyword_list:
                marked_text_tokens.append((token, keyword_tag))
            else:
                marked_text_tokens.append((token, ""))

        return marked_text_tokens

    def _lemmatize_sentence(self, keyphrase_list):
        tmp_check_list = keyphrase_list
        result = []

        for tup in tmp_check_list:
            phrase = tup[0]
            score = tup[1]
            tokenize_phrase = word_tokenize(phrase)
            singular_tokens = [
                self.lemma.lemmatize(word) for word in tokenize_phrase
            ]
            if len(singular_tokens) > 1:
                singular_sentence = " ".join(singular_tokens)
            else:
                singular_sentence = ""
                singular_sentence = singular_sentence.join(singular_tokens)
            if singular_sentence in result:
                keyphrase_list.remove(tup)
            else:
                result.append((phrase, score))

        return result
