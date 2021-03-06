import re
import torch
import torch.nn as nn
from bert_utils.tokenization_bert import BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import tldextract

nltk.data.path.append("/tmp/nltk_data")
nltk.download("punkt", download_dir="/tmp/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="/tmp/nltk_data")


class BERT_NER:
    def __init__(self, model):
        if model.config.num_labels == 9:
            self.labels = [
                "O",
                "MISC",
                "MISC",
                "PER",
                "PER",
                "ORG",
                "ORG",
                "LOC",
                "LOC",
            ]
        elif model.config.num_labels == 5:
            self.labels = [
                "O",
                "MISC",
                "PER",
                "ORG",
                "LOC",
            ]
        else:
            self.labels = ["O", "MISC"]
        self.model = model
        self.tokenizer = BertTokenizer("vocab.txt")
        self.sm = nn.Softmax(dim=1)
        self.conf = 0.995
        self.contractions = {
            "[sep]": "separator",
            "[cls]": "classify",
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "oughtn't": "ought not",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there had",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
        }
        self.stop_words = {
            "",
            "hi",
            "hey",
            "welcome",
            "bye",
            "goodbye",
            "i",
            "aha",
            "ugh",
            "ah",
            "oh",
            "uh",
            "um",
            "huh",
            "like",
            "right",
            "yeah",
            "okay",
            "s",
            "ourselves",
            "hers",
            "between",
            "yourself",
            "but",
            "again",
            "there",
            "about",
            "once",
            "during",
            "out",
            "very",
            "having",
            "with",
            "they",
            "own",
            "an",
            "be",
            "some",
            "for",
            "do",
            "its",
            "yours",
            "such",
            "into",
            "of",
            "most",
            "itself",
            "other",
            "off",
            "is",
            "am",
            "or",
            "who",
            "as",
            "from",
            "him",
            "each",
            "the",
            "themselves",
            "until",
            "below",
            "are",
            "we",
            "these",
            "your",
            "his",
            "through",
            "don",
            "nor",
            "me",
            "were",
            "her",
            "more",
            "himself",
            "this",
            "down",
            "should",
            "our",
            "their",
            "while",
            "above",
            "both",
            "up",
            "to",
            "ours",
            "had",
            "she",
            "all",
            "no",
            "when",
            "at",
            "any",
            "before",
            "them",
            "same",
            "and",
            "been",
            "have",
            "in",
            "will",
            "on",
            "does",
            "yourselves",
            "then",
            "that",
            "because",
            "what",
            "over",
            "why",
            "so",
            "can",
            "did",
            "not",
            "now",
            "under",
            "he",
            "you",
            "herself",
            "has",
            "just",
            "where",
            "too",
            "only",
            "myself",
            "which",
            "those",
            "after",
            "few",
            "whom",
            "t",
            "being",
            "if",
            "theirs",
            "my",
            "against",
            "a",
            "by",
            "doing",
            "it",
            "how",
            "further",
            "was",
            "here",
            "than",
        }
        self.url_regex = r"""(?i)\b((?:https?:(?:(/| (forward )?slash ){1,3}|[a-z0-9%])|([a-z0-9\-]+|([.]| dot ))([.]| dot )(?:com|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:([\-]|([.]| dot ))[a-z0-9]+)*([.]| dot )(?:com|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b(/| (forward )?slash )?(?!@)))"""

    def clean_url(self, url):
        url = url.replace(" dot ", ".")
        url = url.replace(" forward slash ", "/")
        url = url.replace(" slash ", "/")
        return url

    def get_domain_name_from_url(self, url):
        return tldextract.extract(url).domain

    def replace_contractions(self, text):
        # Handle acronyms
        text = re.sub(
            " (\w{1} \w{1}) (\w{1}[ .])*",
            lambda x: " " + x.group(0).replace(" ", "").upper() + " ",
            text,
        )
        text = re.sub(
            "[A-Z]\. ", lambda mobj: mobj.group(0)[0] + mobj.group(0)[1], text
        )
        text = re.sub(
            "([A-Z])\.(\w{2,})", lambda mobj: mobj.group(1) + ". " + mobj.group(2), text
        )
        # Handle contractions
        for word in text.split(" "):
            if self.contractions.get(word.lower()):
                text = text.replace(word, self.contractions[word.lower()])

            if "." in word.strip("."):
                if not re.match(self.url_regex, word) and any(
                    [len(ini) > 1 for ini in word.split(".")]
                ):
                    text = text.replace(word, word.replace(".", " "))

        # Handle URLs with words
        text = re.sub(self.url_regex, lambda m: self.clean_url(m.group(0)), text)
        return text

    def get_entities(self, text):
        segment_entities = []
        segment_scores = []
        segment_labels = []

        text = text.strip()
        if len(text) > 1 and text[-1] not in [".", "?", "!"]:
            text += "."
        text = self.replace_contractions(text) + " "
        for sent in sent_tokenize(text):
            if len(sent.split()) > 3:
                (sent_ent, sent_score, sent_labels,) = self.get_entities_from_sentence(
                    sent
                )

                segment_entities.extend(sent_ent)
                segment_scores.extend(sent_score)
                segment_labels.extend(sent_labels)

        # removing duplicate entities
        seg_entities = dict(zip(segment_entities, segment_scores))
        seg_labels = dict(zip(segment_entities, segment_labels))
        return seg_entities, seg_labels

    def get_entities_from_sentence(self, clean_text):
        # splitting text, preserving punctuation

        split_text = list(
            filter(
                lambda word: word not in ["", None],
                re.split(
                    "[\s]|([?,!/()]+)|(\.)([A-Z][a-z]+)|(\w{2,}[*]*\.?\w{2,})(\.)\s",
                    clean_text + " ",
                ),
            )
        )
        pos_text = nltk.pos_tag(split_text)
        input_ids, token_to_word = self.prepare_input_for_model(pos_text)

        entities = self.extract_entities(input_ids, token_to_word)
        sent_entity_list, sent_scores, sent_labels = self.concat_entities(entities)
        if len(sent_entity_list) > 0:
            sent_entity_list = self.capitalize_entities(sent_entity_list)
            sent_labels = self.prioritize_labels(sent_labels)
        return sent_entity_list, sent_scores, sent_labels

    def prioritize_labels(self, sent_labels):
        preference_labels = ["MISC", "ORG", "PER", "LOC", "O"]
        sent_labels = [
            min(label_list, key=lambda l: preference_labels.index(l))
            for label_list in sent_labels
        ]
        return sent_labels

    def capitalize_entities(self, entity_list):
        def capitalize_entity(ent):
            if "." in ent:
                if re.match(self.url_regex, ent):
                    ent = self.get_domain_name_from_url(ent).capitalize()
                else:
                    ent = ent.title()
                return ent
            if ent.lower() in self.stop_words:
                ent = ent.lower()
                return ent
            if not ent[0].isupper():
                ent = ent.capitalize()

            return ent

        entity_list = list(
            map(
                lambda entities: " ".join(
                    list(map(lambda ent: capitalize_entity(ent), entities.split(),))
                ),
                entity_list,
            )
        )

        return entity_list

    def prepare_input_for_model(self, pos_text):
        input_ids = []
        token_to_word = []

        for w_index, (word, tag) in enumerate(pos_text):
            toks = self.tokenizer.encode(word)
            # removing characters that usually do not appear within text
            clean_word = re.sub(r"[^a-zA-Z0-9_\'*-.]+", "", word).strip(" .,'\"")
            if clean_word != "":
                token_to_word.extend([(w_index, clean_word, tag)] * len(toks))
                input_ids.extend(toks)
        return input_ids, token_to_word

    def extract_entities(self, input_ids, token_to_word):
        # Calculating batch size based on nearest "." from mid-point of text if length exceeds 512
        if len(input_ids) > 512:
            batch_size = (
                510 - 1 - input_ids[:510][::-1].index(self.tokenizer.encode(".")[0])
            )
        else:
            batch_size = 510

        entities = []
        for i in range(0, len(input_ids), batch_size):
            encoded_text_sp = (
                self.tokenizer.encode("[CLS]")
                + input_ids[i : i + batch_size]
                + self.tokenizer.encode("[SEP]")
            )
            input_tensor = torch.tensor(encoded_text_sp).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(input_tensor)[0][0, 1:-1]

            scores = self.sm(outputs).detach().numpy().max(-1)
            labels = [
                self.labels[ind] for ind in self.sm(outputs).argmax(-1).detach().numpy()
            ]
            batch_tok_word = token_to_word[i : i + batch_size]
            for j, (w_ind, tok, tag) in enumerate(batch_tok_word):
                # Consider Entities, and Non-Entities with low confidence (false negatives)
                if labels[j] not in ["O"] or (
                    labels[j] == "O" and scores[j] < self.conf
                ):
                    entities.append((tok, scores[j], tag, labels[j], w_ind))

        return entities

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def concat_entities(self, entities):
        sent_entity_list = []
        sent_scores = []
        sent_labels = []
        seen = []
        # remove consecutive duplicate entities from list(tuple(word, score, pos_tag, label))
        grouped_scores = {}
        grouped_labels = {}
        grouped_words = []
        prev = -1
        for i, (tok, sc, tag, lab, w_ind) in enumerate(entities):
            grouped_scores[(tok, tag, w_ind)] = min(
                grouped_scores.get((tok, tag, w_ind), 2), sc
            )
            grouped_labels[(tok, tag, w_ind)] = grouped_labels.get(
                (tok, tag, w_ind), []
            ) + [lab]
            if prev != w_ind:
                grouped_words.append((tok, tag, w_ind))
            prev = w_ind
        for i in range(len(grouped_words)):
            if i in seen or grouped_words[i][0].lower() in self.stop_words:
                continue

            word_fragments = [grouped_words[i][0]]

            check = grouped_words[i][-1]
            score = grouped_scores[grouped_words[i]]
            label = grouped_labels[grouped_words[i]]
            seen += [i]
            k = i + 1
            while k < len(grouped_words) and grouped_words[k][-1] - check == 1:
                word = grouped_words[k][0]
                word_fragments += [word]
                check = grouped_words[k][-1]
                score += grouped_scores[grouped_words[k]]
                label += grouped_labels[grouped_words[k]]
                seen += [k]
                k += 1
            # remove numbers, single verb, punct, pronoun and adjective entities
            if len(word_fragments) == 1:
                if label[0] == "O" and (grouped_words[i][1][0] not in {"N", "F"}):
                    continue
            if "'" in word_fragments[-1]:
                word_fragments[-1] = word_fragments[-1].split("'")[0]

            # r - stripping stop_words
            while word_fragments and word_fragments[-1].lower() in self.stop_words:
                word_fragments.pop(-1)

            if word_fragments:
                clean_entities = [
                    " ".join(word_fragments).strip(".").replace(" . ", ".")
                ]
                sent_entity_list += clean_entities

                sent_scores += [score / (k - i)]
                sent_labels += [list(set(label))]

        return sent_entity_list, sent_scores, sent_labels
