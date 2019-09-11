import os
import re

import torch
import torch.nn as nn
from bert_utils.modeling_bert import BertPreTrainedModel, BertModel
from bert_utils.tokenization_bert import BertTokenizer
from itertools import groupby
import nltk
from nltk.tokenize import sent_tokenize


nltk.data.path.append("/tmp/nltk_data")
nltk.download("punkt", download_dir="/tmp/nltk_data")


class BertForTokenClassification_custom(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification_custom, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        return outputs  # (scores)


class BERT_NER:
    def __init__(self, model):

        self.labels = ["O", "E"]
        self.model = model
        self.tokenizer = BertTokenizer("vocab.txt")
        self.sm = nn.Softmax(dim=1)
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

    def replace_contractions(self, text):
        for word in text.split(" "):
            if self.contractions.get(word.lower()):
                text = text.replace(word, self.contractions[word.lower()])
        return text

    def get_entities(self, text):
        segment_entities = []
        segment_scores = []

        text = text + " "
        for sent in sent_tokenize(text):
            if len(sent.split()) > 1:
                sent_ent, sent_score = self.get_entities_from_sentence(sent)

                segment_entities.extend(sent_ent)
                segment_scores.extend(sent_score)

        # removing duplicate entities
        seg_entities = dict(zip(segment_entities, segment_scores))
        # seg_words = list(seg_entities.keys())
        # seg_scores = list(seg_entities.values())
        return seg_entities

    def get_entities_from_sentence(self, text):

        # cleaning and splitting text, preserving punctuation
        clean_text = self.replace_contractions(text)
        split_text = re.split("[\s]|([?.,!/]+)", clean_text)

        input_ids, token_to_word = self.prepare_input_for_model(split_text)

        entities = self.extract_entities(input_ids, token_to_word)

        sent_entity_list, sent_scores = self.concat_entities(clean_text, entities)
        return sent_entity_list, sent_scores

    def prepare_input_for_model(self, split_text):
        input_ids = []
        token_to_word = []

        for word in split_text:
            if word not in ["", None]:
                toks = self.tokenizer.encode(word)
                # removing characters that usually do not appear within text
                clean_word = re.sub(r"[^a-zA-Z0-9_\'*-]+", "", word).strip()
                token_to_word.extend([clean_word] * len(toks))
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

            for j, (tok, embed) in enumerate(
                zip(token_to_word[i : i + batch_size], list(outputs))
            ):
                embed = embed.unsqueeze(0)
                score = self.sm(embed).detach().numpy().max(-1)[0]
                label = self.labels[self.sm(embed).argmax().detach().numpy()]
                # Consider Entities and Non-Entities with low confidence (false negatives)
                if tok != "":
                    if label != "O" or (label == "O" and score < 0.995):
                        entities.append((tok, score))
        return entities

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def concat_entities(self, text, entities):
        sent_entity_list = []
        sent_scores = []
        seen = []
        # handling abbreviations such as U.S.
        text = re.sub(
            "[A-Z][.]\s?", lambda mobj: mobj.group(0)[0] + " ", text
        ).casefold()

        # remove consecutive duplicate entities
        entity_words = [
            grouped_entity[0]
            for grouped_entity in groupby(
                map(lambda ent_score_tuple: ent_score_tuple[0], entities)
            )
        ]
        entity_scores = {
            ent_score_tuple[0]: ent_score_tuple[1] for ent_score_tuple in entities
        }
        for i in range(len(entity_words)):
            if i in seen:
                continue
            if entity_words[i].casefold() in text:
                conc = entity_words[i].strip("'\"") + " "
                conc = conc if conc[0].isupper() else conc.capitalize()

                check = entity_words[i] + " "
                score = entity_scores[entity_words[i]]
                seen += [i]
                k = i + 1

                while k < len(entity_words) and (
                    check.casefold() + entity_words[k].casefold() in text
                ):
                    conc_word = entity_words[k].strip("'\"") + " "
                    conc += (
                        conc_word if conc_word[0].isupper() else conc_word.capitalize()
                    )

                    check += entity_words[k] + " "
                    score += entity_scores[entity_words[k]]
                    seen += [k]
                    k += 1
                sent_entity_list += [conc.strip(" ,.")]
                sent_scores += [score / (k - i)]

        return sent_entity_list, sent_scores