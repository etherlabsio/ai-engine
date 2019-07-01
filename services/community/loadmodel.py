import torch
from community import bert
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertPreTrainingHeads
from community.utility import getBERTFeatures, formatTime
import logging
import community.utility 

logger = logging.getLogger()

def loadmodel(model_config, mind):
   tokenizer = BertTokenizer.from_pretrained(model_config['tokenizer'])
   config_file = BertConfig.from_json_file(model_config['config'])
   logger.debug("config file", extra={"contains": model_config})
   bert_model = model_config['bert_model']
   load_file = model_config['load_file'] + mind
  # config_file = BertConfig.from_json_file('services/community/bert_config.json')
   model1 = bert.BertForPreTraining_custom(config_file)
   state_dict_1 = torch.load(load_file, map_location='cpu')
   model1.load_state_dict(state_dict_1)
   return model1

def selectmodel(MindId):
    return "mind-"+str(MindId)+".bin"
