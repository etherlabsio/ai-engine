import os
import re
import zipfile

import torch
import torch.nn as nn
from bert_utils.modeling_bert import BertConfig, BertPreTrainedModel, BertModel
from bert_utils.tokenization_bert import BertTokenizer

import uuid

import nltk
from nltk.tokenize import sent_tokenize
from text_utils import CandidateKPExtractor


nltk.data.path.append("/tmp/nltk_data")
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
stop_words = stop_words-set(['get', 'give', 'go', 'do', 'make', 'please'])
stop_words = set(list(stop_words)+list(stop_words_spacy))
action_marker_list = ["we", "i", "you", "let's", "i'll", "we'll"]


class BertForActionItemDetection(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForActionItemDetection, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sfmax = nn.Softmax()
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sfmax(logits)


class ActionItemDetector():

    def __init__(self, segment_object_list, model):
        self.segment_object_list = segment_object_list
        self.model = model
        self.tokenizer = BertTokenizer('bert-base-uncased-vocab.txt')
        self.c_kp = CandidateKPExtractor(stop_words)
        self.first_person_list = ["i", "we", "we'll", "i'll"]
        self.second_person_list = ["you"]
        self.combine_list = ["let's"]

    def get_ai_probability(self, input_sent):
        input_ids = torch.tensor(self.tokenizer.encode(input_sent))
        input_ids = input_ids.unsqueeze(0)
        ai_scores = self.model(input_ids)
        # [0,1] - [non_ai, ai] scores respectively
        return ai_scores.detach().numpy()[0][1]

    def post_process_ai_check(self, candidate_text):
        is_ai_flag = 0
        ret_candidate = ''
        candidate_ais = self.c_kp.get_ai_subjects(candidate_text)

        drop_ctr = 0
        # remove action items that are starting with stop_words
        for ai_sub in candidate_ais:
            if ai_sub.split(' ')[0] in stop_words:
                drop_ctr += 1
            if drop_ctr == len(candidate_ais):
                candidate_ais = []
        if len(candidate_ais) >= 1:
            if (len(set(candidate_text.lower().split(' ')) & set(action_marker_list)) > 0):
                is_ai_flag = 1
        return is_ai_flag, candidate_ais

    def matcher(self, matchObj):
        return matchObj.group(0)[0]+matchObj.group(0)[1]+" "+matchObj.group(0)[2]

    def get_ai_candidates(self, transcript_text, ai_confidence_threshold=0.5):

        action_item_subjects = []
        action_item_sentences = []
        if type(transcript_text) != str:
            return [], []
        else:
            transcript_text = re.sub("[a-z][.?][A-Z]", self.matcher, transcript_text)
            sent_list = sent_tokenize(transcript_text)
            for sent in sent_list:
                if len(sent.split(' ')) > 2:
                    # if (sent[-1]!="?" and sent[-2]!="?"):
                    sent_ai_prob = self.get_ai_probability(sent)

                    if sent_ai_prob >= ai_confidence_threshold and self.post_process_ai_check(sent)[0]:
                        curr_ai_subjects = self.post_process_ai_check(sent)[1]
                        print(sent)
                        if len(curr_ai_subjects) > 1:
                            # merge action items
                            start_idx = sent.lower().find(curr_ai_subjects[0].lower())
                            end_idx = sent.lower().find(curr_ai_subjects[-1].lower())+len(curr_ai_subjects[-1].lower())
                            ai_subject = sent[start_idx:end_idx]
                        else:
                            ai_subject = curr_ai_subjects[0]
                        if len(ai_subject) > 0:
                            ai_subject = ai_subject[0].upper() + ai_subject[1:]
                        action_item_subjects.append(ai_subject)
                        action_item_sentences.append(sent)
        return action_item_subjects, action_item_sentences

    def get_ai_users(self, ai_sent_list):
        ai_assignee_list = []
        for sent in ai_sent_list:
            assign_flag = 0  # default to first person

            fp_list = set(sent.lower().split(' ')) & set(self.first_person_list)
            sp_list = set(sent.lower().split(' ')) & set(self.second_person_list)
            com_list = set(sent.lower().split(' ')) & set(self.combine_list)

            if len(com_list) > 0 and len(fp_list) == 0 and len(sp_list) == 0:
                assign_flag = 2
            else:
                if len(fp_list) > 0 and len(sp_list) > 0:
                    assign_flag = 2
                else:
                    if len(sp_list) > 1:
                        assign_flag = 1
            ai_assignee_list.append(assign_flag)
        return ai_assignee_list

    def get_action_decision_subjects_list(self):

        ai_subject_list = []
        ai_user_list = []
        segment_id_list = []
        assignees_list = []
        isAssigneePrevious_list = []
        isAssigneeBoth_list = []

        for seg_object in self.segment_object_list:

            curr_assignees_list = []
            curr_isAssigneePrevious_list = []
            curr_isAssigneeBoth_list = []

            transcript_text = seg_object['originalText']
            # get the AI probabilities for each sentence in the transcript
            curr_ai_list, curr_ai_sents = self.get_ai_candidates(transcript_text)
            curr_ai_user_list = self.get_ai_users(curr_ai_sents)
            curr_segment_id_list = [seg_object['id']]*len(curr_ai_list)

            for ai_user in curr_ai_user_list:
                if ai_user == 0:
                    curr_assignees_list += [seg_object['spokenBy']]
                else:
                    curr_assignees_list += ['NA']
                if ai_user == 1:
                    curr_isAssigneePrevious_list.append(True)
                else:
                    curr_isAssigneePrevious_list.append(False)
                if ai_user == 2:
                    curr_isAssigneeBoth_list.append(True)
                else:
                    curr_isAssigneeBoth_list.append(False)

            ai_subject_list += curr_ai_list
            ai_user_list += curr_ai_user_list
            segment_id_list += curr_segment_id_list
            assignees_list += curr_assignees_list
            isAssigneePrevious_list += curr_isAssigneePrevious_list
            isAssigneeBoth_list += curr_isAssigneeBoth_list

        uuid_list = []
        ai_response_list = []
        for i in range(len(ai_subject_list)):
            uuid_list.append(str(uuid.uuid1()))
        for uuid_, segment, action_item, assignee, is_prev_user, is_both in zip(uuid_list, segment_id_list, ai_subject_list, assignees_list, isAssigneePrevious_list, isAssigneeBoth_list):
            ai_response_list.append({"id": uuid_,
                                    "subject": action_item,
                                    "segment_ids": [segment],
                                    "assignees": assignee,
                                    "is_assignee_previous": is_prev_user,
                                    "is_assignee_both": is_both})

        # placeholder decision list
        decision_response_list = [{'id': str(str(uuid.uuid1())),
                                    'segment_ids': ['seg1'],
                                        'subject': 'decision_text1'},
                                {'id': str(str(uuid.uuid1())),
                                    'segment_ids': ['seg2'],
                                        'subject': 'decision_text2'}]
        return ai_response_list, decision_response_list