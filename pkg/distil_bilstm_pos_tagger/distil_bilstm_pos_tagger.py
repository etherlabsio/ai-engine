try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import boto3
import requests
from distilbert_utils.tokenization_distilbert import DistilBertTokenizer
import nltk
import numpy as np

class BiLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores,tag_space
        
class DistilBiLstmPosTagger:
    def __init__(self,model_path=None):
        self.tokenizer  = DistilBertTokenizer("bert-base-uncased-vocab.txt")
        self.EMBEDDING_DIM = 256
        self.HIDDEN_DIM = 512
        self.lab_list = ['NNS', 'CD', 'TO', 'VBD', 'WP$', 'LS', 'RP', 'SYM', 'VBN', 'NNPS', 'RBR', 'JJS', 'VBP', 'MD', 'JJ', 'CC', 'VBG', 'IN', 'WP', 'PRP', 'PUNC', 'POS', 'FW', 'JJR', 'EX', 'WRB', 'DT', 'UH', 'VB', 'VBZ', 'RB', 'RBS', 'NN', 'WDT', 'NNP', 'PRP$', 'PDT']
        self.label_map = {label:i for i, label in enumerate(self.lab_list)}
        self.ix_to_tag = {i:label for i, label in enumerate(self.lab_list)}
        self.model = BiLSTMTagger(self.EMBEDDING_DIM, self.HIDDEN_DIM, self.tokenizer.vocab_size, len(self.label_map))
        if (model_path != None):
            state_dict = torch.load(model_path,map_location='cpu')
        else:
            state_dict = self.load_model()

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def load_model(self):
        bucket = os.getenv("BUCKET_NAME")
        model_path = os.getenv("POS_MODEL")
        s3 = boto3.resource("s3")

        modelObj = s3.Object(bucket_name=bucket, key=model_path)
        state_dict = torch.load(
            io.BytesIO(modelObj.get()["Body"].read()), map_location="cpu"
        )
        return state_dict

    def label_selector(self,predicted_label_list,tokenized_sent):
        predicted_list =[]
        start = 0 
        end = 1 
        for token in tokenized_sent:
            window_len = len(self.tokenizer.tokenize(token))
            if window_len ==1:
                predicted_list.append(predicted_label_list[start])
                start=end
                end+=1
            elif window_len >1:
                end = window_len +start
                if (predicted_label_list[start] == 'PUNC'):
                    lab_check = False
                    for lab in predicted_label_list[start:end+1]:
                        if (lab != 'PUNC'):
                            predicted_list.append(lab)
                            lab_check = True
                            break
                    if (lab_check==False):
                        predicted_list.append(predicted_label_list[start])
                    else:
                        pass
                else:       
                    predicted_list.append(predicted_label_list[start])
                start=end
                end+=1
        return predicted_list
            
    def get_sent_pos_tags(self, text):
        batch_size = 128
        predicted_label_list = []
        tokenized_sent = nltk.word_tokenize(text)
        text = " ".join(tokenized_sent)
        input_ids = self.tokenizer.encode(text,add_special_tokens=False)
        for i in range(0,len(input_ids),batch_size):
            batch_input_ids = input_ids[i:i+batch_size]
            batch_input_ids = torch.tensor(batch_input_ids)    
            with torch.no_grad(): 
                tag_scores,_ = self.model(batch_input_ids)
            for sc in tag_scores:
                predicted_label_list.append(self.ix_to_tag[int(np.argmax(sc.cpu().detach().numpy()))])

        predicted_list = self.label_selector(predicted_label_list,tokenized_sent)
        sent_tag = []
        for i in zip(tokenized_sent,predicted_list):
            sent_tag.append(i)
        return sent_tag
