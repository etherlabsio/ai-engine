import torch
import torch.nn as nn
from bert_utils.modeling_bert import BertConfig,BertPreTrainedModel,BertModel
from bert_utils.tokenization_bert import BertTokenizer

import nltk
from nltk.tokenize import sent_tokenize
import os

nltk.data.path.append("/tmp/nltk_data")
nltk.download("stopwords", download_dir="/tmp/nltk_data")
nltk.download("punkt", download_dir="/tmp/nltk_data")
nltk.download("averaged_perceptron_tagger",download_dir="/tmp/nltk_data")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
stop_words.add('hear')
stop_words.add('see')

import logging
from log.logger import setup_server_logger

logger = logging.getLogger(__name__)
setup_server_logger(debug=True)
tokenizer = BertTokenizer('bert-base-uncased-vocab.txt')

class BertForActionItemDetection(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForActionItemDetection, self).__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sfmax = nn.Softmax()
        self.apply(self.init_weights)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return self.sfmax(logits)

def get_ai_probability(model,input_sent):
	if input_sent[-1]=='.' or input_sent[-1]=='?':
		input_sent = input_sent[:-1]
	input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)
	ai_scores = model(input_ids)
	return ai_scores.detach().numpy()[0][1] #[0,1] - [non_ai, ai] scores respectively

def post_process_ai_check(candidate_text):
    is_ai_flag = False
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(candidate_text))[0]
    token_list = [ele[0] for ele in tagged_sents]
    tag_list = [ele[1] for ele in tagged_sents]
    if 'VB' in tag_list:
        #get VB locations and see if they all are in stop_words
        vb_loc = [i for i, e in enumerate(tag_list) if e == 'VB']
        vb_tokens = [token_list[i] for i in vb_loc]
        if len(set(vb_tokens)&set(stop_words))<len(set(vb_tokens)):
            is_ai_flag = True
    return is_ai_flag

def get_ai_sentences(model,transcript_text, ai_confidence_threshold = 0.5):

	detected_ai_list = []
	if type(transcript_text)!=str:
		logger.warn(
            "Invalid transcript. Returning empty list",
            extra={"input text": transcript_text},
        )
	else:
		sent_list = sent_tokenize(transcript_text)
		for sent in sent_list:
			sent_ai_prob = get_ai_probability(model,sent)
			if sent_ai_prob>=ai_confidence_threshold and post_process_ai_check(sent):
				detected_ai_list.append(sent)
	return detected_ai_list