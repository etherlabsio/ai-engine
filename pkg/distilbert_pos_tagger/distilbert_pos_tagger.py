import torch
import torch.nn as nn
import os
import boto3
import requests
from distilbert_utils.modeling_distilbert import DistilBertPreTrainedModel,  DistilBertModel, DistilBertConfig
from distilbert_utils.tokenization_distilbert import DistilBertTokenizer
import nltk
import numpy as np

class DistilBertForTokenClassificationCustom(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilBertForTokenClassificationCustom, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, head_mask=None,
                inputs_embeds=None, labels=None):

        outputs = self.distilbert(input_ids,
                            attention_mask=None,
                            head_mask=None,
                            inputs_embeds=None)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits)

        return outputs
        
class DistilBertPosTagger:
    def __init__(self,model_path=None):
        self.tokenizer  = DistilBertTokenizer("bert-base-uncased-vocab.txt")
        config = DistilBertConfig(num_labels=37)
        self.model = DistilBertForTokenClassificationCustom(config)
        if (model_path != None):
            state_dict = torch.load(model_path,map_location='cpu')
        else:
            state_dict = self.load_model()

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.labels = ['NNS', 'CD', 'TO', 'VBD', 'WP$', 'LS', 'RP', 'SYM', 'VBN', 'NNPS', 'RBR', 'JJS', 'VBP', 'MD', 'JJ', 'CC', 'VBG', 'IN', 'WP', 'PRP', 'PUNC', 'POS', 'FW', 'JJR', 'EX', 'WRB', 'DT', 'UH', 'VB', 'VBZ', 'RB', 'RBS', 'NN', 'WDT', 'NNP', 'PRP$', 'PDT']
        self.label_map = {i: label for i, label in enumerate(self.labels)}

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
        preds = None
        batch_size = 510
        predicted_label_list = []
        tokenized_sent = nltk.word_tokenize(text)
        text = " ".join(tokenized_sent)
        input_ids = self.tokenizer.encode(text)
        for i in range(0,len(input_ids),batch_size):
            batch_input_ids = input_ids[i:i+batch_size] 
            batch_input_ids = torch.tensor(batch_input_ids)    
            with torch.no_grad(): 
                outputs = self.model(batch_input_ids.unsqueeze(0))
            logits = outputs
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=2)[0]
            for i in preds[1:-1]:
                predicted_label_list.append(self.label_map[i])
        predicted_list = self.label_selector(predicted_label_list,tokenized_sent)
        sent_tag = []
        for i in zip(tokenized_sent,predicted_list):
            sent_tag.append(i)
        return sent_tag
