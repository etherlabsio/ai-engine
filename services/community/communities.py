import json as js
import text_preprocessing.preprocess as tp
import torch
import logging
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertPreTrainingHeads
from community.utility import getBERTFeatures, formatTime
from community import bert
import networkx as nx
import math
from scipy.spatial.distance import cosine
import community
from datetime import datetime

logger = logging.getLogger()
class community_detection():
    segments_list = []
    model1 = None
    def __init__(self, Request, model1):
        self.segments_list = Request.segments
        self.model1 = model1
    #def parse_meeting(self, segments):
    #    segments_data = list(map(lambda x: tp.preprocess(x['originalText'], stop_words=False, remove_punct=False), segments['segments']))
    #    self.segments_list = segments['segments']
    #    for index,seg in enumerate(self.segments_list):
    #        self.segments_list[index]['originalText'] = segments_data[index]

    def compute_feature_vector(self):
        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #config = BertConfig.from_json_file('services/community/bert_config.json')
        #bert_model = 'bert-base-uncased'
        #model1 = bert.BertForPreTraining_custom(config)
        #state_dict_1 = torch.load('services/community/bert_10epc_se_1e-6_sl40.bin')
        #state_dict_1 = torch.load('services/community/bert_10epc_inc_se+etherdata_1e-6_sl40_bt64.bin')
        #model1.load_state_dict(state_dict_1)
        #model1.eval()
        graph_list = {}
        fv = {}
        index = 0
        for segment in self.segments_list:
            for sent in segment['originalText']:
                graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                fv[index] = getBERTFeatures(self.model1, sent, attn_head_idx=-1)
                index+=1
        return fv, graph_list

    def construct_graph(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for indexa, nodea in enumerate(graph_list.values()):
            for indexb, nodeb in enumerate(graph_list.values()):
                if indexb>indexa:
                    c_weight = cosine(fv[indexa], fv[indexb])
                    meeting_graph.add_edge(indexa, indexb, weight=c_weight)
                    yetto_prune.append((indexa, indexb, c_weight))
        return meeting_graph, yetto_prune

    def prune_edges(self, meeting_graph, graph_list, yetto_prune,v=0.01):
        yetto_prune = sorted(yetto_prune, key=lambda kv : kv[2], reverse=True)
        yetto_prune = yetto_prune[:math.ceil(len(yetto_prune)*v)+1]
        logger.info("pruning value", extra={"v is : ": v})
        meeting_graph_pruned = nx.Graph()
        for indexa, indexb, c_score in yetto_prune:
            meeting_graph_pruned.add_edge(indexa, indexb)
        return meeting_graph_pruned

    def compute_louvian_community(self, meeting_graph_pruned, community_set):
        #community_set = community.best_partition(meeting_graph_pruned)
        #modularity_score = community.modularity(community_set, meeting_graph_pruned)
        #logger.info("Community results", extra={"modularity score":modularity_score})
        community_set_sorted = sorted(community_set.items(), key = lambda kv: kv[1], reverse=False)

        return community_set_sorted

    def refine_community(self, community_set_sorted, graph_list):
        clusters = []
        temp = []
        prev_com = 0
        for word,cluster in community_set_sorted:
            if prev_com==cluster:
                temp.append(word)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append(word)
        timerange = []
        temp = []
        for cluster in clusters:
            temp= []
            for sent in cluster:
                #temp.append(graph_list[sent])
                #logger.info("segment values", extra={"segment":self.segments_list})
                temp.append(graph_list[sent])
            if len(temp)!=0:
                temp = list(set(temp))
                temp = sorted(temp,key=lambda kv: kv[1], reverse=False)
                timerange.append(temp)

        return timerange

    def group_community_by_time(self, timerange):
       timerange_detailed = []
       temp = []
       flag = False
       pims = {}
       index_pim = 0
       index_segment = 0
       for index,com in enumerate(timerange):
           temp = []
           flag = False
           #print ("-----community-----", index)
           for (index1,(sent1,time1,user1, id1)), (index2,(sent2,time2,user2, id2)) in zip(enumerate(com[0:]),enumerate(com[1:])):
               if id1!=id2:
                   if ((formatTime( time2, True)-formatTime(time1, True)).seconds<=240):
                       if (not flag):
                           pims[index_pim] = {'segment'+str(index_segment):[sent1,time1,user1, id1]}
                           index_segment+=1
                           temp.append((sent1,time1,user1, id1))
                       pims[index_pim]['segment'+str(index_segment)] = [sent2,time2,user2, id2]
                       index_segment+=1
                       temp.append((sent2,time2,user2,id2))
                       flag=True
                   else:
                       if flag==True:
                           index_pim+=1
                           index_segment=0
                       flag=False
           if flag==True:
               index_pim+=1
               index_segment=0
           timerange_detailed.append(temp)
       return pims

    def wrap_community_by_time(self, pims):
        yet_to_combine = []
        need_to_remove = []
        inverse_dangling_pims = []
        for index1,i in enumerate(pims.keys()):
            for index2,j in enumerate(pims.keys()):
                if index1!=index2:
                    if pims[i]['segment0'][1] >= pims[j]['segment0'][1] and pims[i]['segment0'][1] <= pims[j]['segment'+str(len(pims[j].values())-1)][1]:
                        if (j,i) not in yet_to_combine and i not in need_to_remove and j not in need_to_remove:
                            yet_to_combine.append((i,j))
                            need_to_remove.append(i)
        for i,j in yet_to_combine:
            for k in pims[i]:
                if pims[i][k] not in pims[j].values():
                    pims[j]['segment'+str(len(pims[j].values())-1)] = pims[i][k]
                    continue
        for i in need_to_remove:
            pims.pop(i)

        for index, p in enumerate(pims.keys()):
            for seg in pims[p].keys():
                pims[p][seg][0] = [' '.join(text for text in segment['originalText']) for segment in self.segments_list if segment['id']==pims[p][seg][3]]
                inverse_dangling_pims.append(pims[p][seg][3])

        c_len = 0
        for segment in self.segments_list:
            if segment['id'] not in inverse_dangling_pims:
                while c_len in pims.keys():
                    c_len+=1
                pims[c_len] = {"segment0": [' '.join(text for text in segment['originalText']), segment['startTime'], segment['spokenBy'], segment['id']]}
        return pims

    def get_communities(self):
        segments_data = ' '.join([sentence for segment in self.segments_list for sentence in segment['originalText']])
        fv, graph_list = self.compute_feature_vector()
        logger.info("No of sentences is", extra={"sentence": len(fv.keys())})
        meeting_graph, yetto_prune = self.construct_graph(fv, graph_list)
        for v in [0.15, 0.1, 0.05, 0.01]:
            flag = False
            for count in range(5):
                meeting_graph_pruned =  self.prune_edges(meeting_graph, graph_list, yetto_prune, v)
                community_set = community.best_partition(meeting_graph_pruned)
                mod = community.modularity(community_set, meeting_graph_pruned)
                logger.info("Meeting Graph results", extra={"edges before prunning":meeting_graph.number_of_edges(), "edges after prunning": meeting_graph_pruned.number_of_edges()})
                if mod>0.3:
                    flag = True
                    break
                elif mod==0:
                    meeting_graph_pruned = self.prune_edges(meeting_graph, graph_list, yetto_prune, 0.15)
                    flag = True
                    break
            if flag:
                break
        #logger.info("Meeting Graph results", extra={"edges before prunning":meeting_graph.number_of_edges(), "edges after prunning": meeting_graph_pruned.number_of_edges()})
        community_set_sorted = self.compute_louvian_community(meeting_graph_pruned, community_set)
        community_timerange = self.refine_community(community_set_sorted, graph_list)
       #logger.info("commnity timerange", extra={"timerange": community_timerange})
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time(pims)

        logger.info("Final PIMs", extra={"PIMs": pims})
        return pims
