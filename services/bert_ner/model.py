import os
import io
import boto3
import torch
import torch.nn as nn
import pickle
import logging
from bert_utils.modeling_bert import BertPreTrainedModel, BertModel
from bert_utils import BertConfig
from log.logger import setup_server_logger
s3 = boto3.resource("s3")

logger = logging.getLogger()
setup_server_logger(debug=True)

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

def download_model():
    bucket = os.getenv("BUCKET_NAME")
    model_path = os.getenv("MODEL")

    modelObj = s3.Object(bucket_name=bucket, key=model_path)
    state_dict = torch.load(
        io.BytesIO(modelObj.get()["Body"].read()), map_location="cpu"
    )
    return state_dict

def load_model():
	# load the model when lambda execution context is created
	state_dict = download_model()
	config = BertConfig()
	config.num_labels = state_dict["classifier.weight"].shape[0]
	model = BertForTokenClassification_custom(config)
	model.load_state_dict(state_dict)
	model.eval()
	logger.info(f"Model loaded for evaluation")
	return model