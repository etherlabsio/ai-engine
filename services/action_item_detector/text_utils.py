import string
import itertools
import nltk


class CandidateKPExtractor(object):
    def __init__(self, stop_words, filter_small_sents=True):

        self.punct = set(string.punctuation)
        self.filter_small_sents = filter_small_sents
        self.stop_words = stop_words

    def get_candidate_phrases(
        self, text, pos_search_pattern_list=[r"""verbnoun:{<VB>+<.+>{0,2}<NN.*>+}"""]
    ):
        all_chunks = []

        for pattern in pos_search_pattern_list:
            all_chunks += self.getregexChunks(text, pattern)

        candidates_tokens = [
            " ".join(word for word, pos, chunk in group).lower()
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

    def get_ai_subjects(self, text, prop_pattern=[r"""prpvb:{<PRP><MD><VB>+}"""]):
        ai_candidates = self.get_candidate_phrases(text)
        prop_candidates = self.get_candidate_phrases(text, prop_pattern)
        if len(ai_candidates) == 0 and len(prop_candidates) > 0:
            # search for ai subject with new candidates
            ai_candidates = self.get_candidate_phrases(
                text, pos_search_pattern_list=[r"""verbnoun:{<VB>+<.+>{0,5}<NN.*>+}"""]
            )

        return ai_candidates

    def getregexChunks(self, text, grammar):

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
