import os
import re
import zipfile

import torch
import torch.nn as nn
from bert_utils.modeling_bert import BertConfig, BertPreTrainedModel, BertModel
from bert_utils.tokenization_bert import BertTokenizer

import logging
from log.logger import setup_server_logger

import nltk
from nltk.tokenize import sent_tokenize
from text_utils import CandidateKPExtractor

# import spacy
# #download and unzip spacy
# #client = boto3.client('s3')
# s3 = boto3.client('s3')
# bucket = os.getenv('BUCKET_NAME')
# spacy_en_model_path = os.getenv('SPACY_MODEL_PATH')
# spacy.util.set_data_path('/opt')
# s3.download_file(bucket,spacy_en_model_path,"/opt")
# #unzip the file
# with zipfile.ZipFile("/opt/en_core_web_sm.zip","r") as zip_ref:
#     zip_ref.extractall("/opt/")
# nlp = spacy.load("en_core_web_sm")
# print('Spacy NLP Loaded')

logger = logging.getLogger(__name__)
setup_server_logger(debug=True)

# if os.path.isdir("/tmp/nltk_data"):
#     logger.info('Using existing nltk download files')
# else:
nltk.data.path.append("/tmp/nltk_data")
logger.info('Downloading nltk data files to /tmp/nltk_data')
nltk.download("stopwords", download_dir="/tmp/nltk_data")
nltk.download("punkt", download_dir="/tmp/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="/tmp/nltk_data")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
stop_words.add('hear')
stop_words.add('see')

stop_words_spacy = list(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere n't

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves

'd 'll 'm 're 's 've
""".split()
)

stop_words = set(list(stop_words) + stop_words_spacy)
stop_words = stop_words-set(['get','give','go','do','make','please'])
stop_words = set(list(stop_words)+list(stop_words_spacy))
action_marker_list = ["we","i","you","let's","i'll","we'll"]

tokenizer = BertTokenizer('bert-base-uncased-vocab.txt')

c_kp = CandidateKPExtractor(stop_words)


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


def get_ai_probability(model, input_sent):
    if input_sent[-1] == '.' or input_sent[-1] == '?':
        input_sent = input_sent[:-1]
    input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)
    ai_scores = model(input_ids)
    return ai_scores.detach().numpy()[0][1]   # [0,1] - [non_ai, ai] scores respectively


def post_process_ai_check(candidate_text):  #returns is_ai flag and candidate action item
    is_ai_flag = 0
    ret_candidate = ''
    candidate_ais = c_kp.get_ai_subjects(candidate_text)

    drop_ctr = 0
    #stop_drop_list = []
    for ai_sub in candidate_ais: #remove action items that are starting with stop_words
        if ai_sub.split(' ')[0] in stop_words:
            drop_ctr+=1
        if drop_ctr==len(candidate_ais):
            #stop_drop_list.append(ai_sent)
            candidate_ais = []
    
    #check if there are any subjects and drop sentences without any of action_marker
    if len(candidate_ais)>=1 and (len(set(candidate_text.lower().split(' ')) & set(action_marker_list))>0):
        is_ai_flag = 1 #whether the sentence is an AI candidate
        ret_candidate = candidate_ais[0]
    return [is_ai_flag],ret_candidate


def matcher(matchObj):
    return matchObj.group(0)[0]+matchObj.group(0)[1]+" "+matchObj.group(0)[2]    


def get_ai_sentences(model, transcript_text, ai_confidence_threshold=0.5):

    #detected_ai_list = []
    action_item_candidate = []
    if type(transcript_text) != str:
        logger.warn(
            "Invalid transcript. Returning empty list",
            extra={"input text": transcript_text},)
    else:
        transcript_text = re.sub("[a-z][.?][A-Z]",matcher,transcript_text)
        sent_list = sent_tokenize(transcript_text)
        for sent in sent_list:
            if len(sent.split(' '))>2:
                # if (sent[-1]!="?" and sent[-2]!="?"):
                sent_ai_prob = get_ai_probability(model, sent)
                if sent_ai_prob >= ai_confidence_threshold and post_process_ai_check(sent)[0]:
                    action_item_candidate.append(post_process_ai_check(sent)[1])

    return action_item_candidate

def get_ai_users(ai_sent_list):
    first_person_list = ["i","we","we'll","i'll"]
    second_person_list = ["you"]
    combine_list = ["let's"]
    

    ai_assignee_list = []
    for sent in ai_sent_list:
        assign_flag = 0 #default to first person

        fp_list = set(sent.lower().split(' ')) & set(first_person_list)
        sp_list = set(sent.lower().split(' ')) & set(second_person_list)
        com_list = set(sent.lower().split(' ')) & set(combine_list)

        if len(com_list)>0 and len(fp_list)==0 and len(sp_list)==0:
            assign_flag = 2
        else:
            if len(fp_list)>0 and len(sp_list)>0:
                assign_flag = 2
            else:
                if len(sp_list)>1:
                    assign_flag = 1
        ai_assignee_list.append(assign_flag)
    return ai_assignee_list

        #map action_items to users
        # 0 - current segment speaker
        # 1 - previous segment speaker
        # 2 - current and previous user


