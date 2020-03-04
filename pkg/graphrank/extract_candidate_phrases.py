import itertools
import nltk
import string


class CandidateKPExtractor(object):
    def __init__(self, stop_words=[], filter_small_sents=True):
        self.punct = set(string.punctuation)
        self.filter_small_sents = filter_small_sents
        self.stop_words = stop_words

    def get_candidate_phrases(self, text, pos_search_pattern_list=None):
        if pos_search_pattern_list is None:
            pos_search_pattern_list = [
                r"""base: {(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}""",
                # r"""nounverb:{<NN.+>+<.+>{0,3}<VB+>{1}}""",
                r"""verbnoun:{<VB+>{1}<.+>{0,2}<NN.+>+}""",
                r""" nounnoun:{<NN.+>+<.+>{0,2}<NN.+>+}""",
            ]
        all_chunks = []

        for pattern in pos_search_pattern_list:
            all_chunks += self.get_regex_chunks(text, pattern)

        candidates_tokens = [
            " ".join(word for word, pos, chunk in group)
            for key, group in itertools.groupby(
                all_chunks, self.lambda_unpack(lambda word, pos, chunk: chunk != "O")
            )
            if key
        ]

        candidate_phrases = [
            cand
            for cand in candidates_tokens
            if cand not in self.stop_words
            and not all(char in self.punct for char in cand)
        ]

        return candidate_phrases

    def get_regex_chunks(self, text, grammar):
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = nltk.pos_tag_sents(
            nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)
        )
        all_chunks = list(
            itertools.chain.from_iterable(
                nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                for tagged_sent in tagged_sents
            )
        )
        return all_chunks

    def lambda_unpack(self, f):
        return lambda args: f(*args)
