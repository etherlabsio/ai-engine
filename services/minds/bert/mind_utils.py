from . import BertTokenizer, BertModel
from .modeling import BertPreTrainedModel, BertPreTrainingHeads
import torch
import numpy as np
import io
import zlib
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CustomBertPreTrainedModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBertPreTrainedModel, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        output_all_encoded_layers = True
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers)
        if output_all_encoded_layers:
            sequence_output_pred = sequence_output[-1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output_pred, pooled_output)
        return prediction_scores, seq_relationship_score, sequence_output, pooled_output

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-vocab.txt')

def getBERTFeatures(model, text,attn_head_idx=-1):  # attn_head_idx - index o[]
    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > 200:
        tokenized_text = tokenized_text[0:200]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    _, _, seq_out, pool_out = model(tokens_tensor)
    seq_out = list(getPooledFeatures(seq_out[attn_head_idx]).T)
    return seq_out

def getPooledFeatures(np_array):
    np_array = np_array.reshape(np_array.shape[1], np_array.shape[2]).detach().numpy()
    np_array_mp = np.mean(np_array, axis=0).reshape(1, -1)
    return np_array_mp

def getNSPScore(sample_text, model,tokenizer):
    m = torch.nn.Softmax()

    tokenized_text = tokenizer.tokenize(sample_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [0] * tokenized_text.index('[SEP]') + [1] * (len(tokenized_text) - tokenized_text.index('[SEP]'))

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    pred_score, seq_rel, seq_out, pool_out = model(tokens_tensor, segments_tensors)
    return m(seq_rel).detach().numpy()[0][0]

def predict(model, mind_dict, input_text, get_nsp='True'):
  #split text into sentences and return sentence feature vector list
    sent_feat_list = []
    sent_list = []
    for sent in list(input_text.split('.')):
        if len(sent)>0:
            sent_feats = getBERTFeatures(model, sent)
            sent_feat_list.append(sent_feats)
            sent_list.append(sent)

    #calculate cluster NSP score for each of the filtered sentence
    segment_nsp_list = []
    if get_nsp=='True':
        for sent in sent_list:
            curr_sent_nsp = []
            for clust_sent in list(mind_dict['sentence'].values()):
                nsp_input = sent + ' [SEP] ' + clust_sent
                curr_sent_nsp.append(getNSPScore(nsp_input,model,tokenizer))
            segment_nsp_list.append(curr_sent_nsp)

    if len(sent_feat_list)>0:
        sent_feat_list = np.array(sent_feat_list).reshape(len(sent_feat_list),-1)
    feats = list(mind_dict['feature_vector'].values())
    mind_feats_nparray = np.array(feats).reshape(len(feats),-1)
    if len(segment_nsp_list)>0:
        segment_nsp_list = np.array(segment_nsp_list).reshape(len(segment_nsp_list),-1)

    json_out = {'sent_feats': [sent_feat_list],
                          'mind_feats': [mind_feats_nparray],
                          'sent_nsp_scores':[segment_nsp_list]}
    return json_out