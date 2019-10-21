import numpy as np
import json
import text_preprocessing.preprocess as tp
from group_segments import extra_preprocess
from copy import deepcopy
import networkx as nx
import math
from group_segments.scorer import cosine
import community
from datetime import datetime
from group_segments import scorer
import logging
import math
from log.logger import setup_server_logger
logger = logging.getLogger()


class community_detection():
    segments_list = []
    segments_org = []
    segments_map = {}
    segments_order = {}
    lambda_function = None

    def __init__(self, Request, lambda_function):
        self.segments_list = Request.segments
        self.segments_org = Request.segments_org
        self.segments_order = Request.segments_order
        self.segments_map = Request.segments_map
        self.lambda_function = lambda_function

    def compute_feature_vector_gpt(self):
        graph_list = {}
        input_list = []
        fv = {}
        index = 0
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    input_list.append(sent)

        transcript_score = scorer.get_feature_vector(input_list, self.lambda_function)
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                    fv[index] = transcript_score[index]
                    index += 1
        return fv, graph_list

    def construct_graph(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                c_weight = cosine(fv[nodea], fv[nodeb])
                meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                yetto_prune.append((nodea, nodeb, c_weight))
        return meeting_graph, yetto_prune

    def prune_edges_outlier(self, meeting_graph, graph_list, yetto_prune):
        meeting_graph_pruned = nx.Graph()
        weights = []
        for nodea, nodeb, weight in meeting_graph.edges.data():
            meeting_graph_pruned.add_nodes_from([nodea, nodeb])
            weights.append(weight["weight"])

        q3 = np.percentile(weights, 75)
        logger.info("Outlier Score", extra={"outlier threshold is : ": q3})

        for indexa, indexb, c_score in meeting_graph.edges.data():
            if c_score["weight"]>=q3:
                meeting_graph_pruned.add_edge(indexa, indexb, weight=c_score["weight"])
        return meeting_graph_pruned

    def compute_louvian_community(self, meeting_graph_pruned, community_set):
        # community_set = community.best_partition(meeting_graph_pruned)
        # modularity_score = community.modularity(community_set, meeting_graph_pruned)
        # logger.info("Community results", extra={"modularity score":modularity_score})
        community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
        return community_set_sorted

    def refine_community(self, community_set_sorted, graph_list):
        clusters = []
        temp = []
        prev_com = 0
        for index, (word, cluster) in enumerate(community_set_sorted):
            if prev_com == cluster:
                temp.append(word)
                if index == len(community_set_sorted) - 1:
                    clusters.append(temp)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append(word)
        timerange = []
        temp = []
        for cluster in clusters:
            temp = []
            for sent in cluster:
                # temp.append(graph_list[sent])
                # logger.info("segment values", extra={"segment":self.segments_list})
                temp.append(graph_list[sent])
            if len(temp) != 0:
                temp = list(set(temp))
                temp = sorted(temp, key=lambda kv: kv[1], reverse=False)
                timerange.append(temp)

        return timerange

    def group_community_by_time(self, timerange):
        timerange_detailed = []
        temp = []
        flag = False
        pims = {}
        index_pim = 0
        index_segment = 0

        for index, com in enumerate(timerange):
            temp = []
            flag = False

            if com[1:] == []:
                pims[index_pim] = {'segment0': [com[0][0], com[0][1], com[0][2], com[0][3]]}
                index_pim += 1

            for (index1, (sent1, time1, user1, id1)), (index2, (sent2, time2, user2, id2)) in zip(enumerate(com[0:]), enumerate(com[1:])):
                if id1 != id2:
                    if ((extra_preprocess.format_time(time2, True) - extra_preprocess.format_time(time1, True)).seconds <= 120):
                        if (not flag):
                            pims[index_pim] = {'segment' + str(index_segment): [sent1, time1, user1, id1]}
                            index_segment += 1
                            temp.append((sent1, time1, user1, id1))
                        pims[index_pim]['segment' + str(index_segment)] = [sent2, time2, user2, id2]
                        index_segment += 1
                        temp.append((sent2, time2, user2, id2))
                        flag = True
                    else:
                        if flag is True:
                            index_pim += 1
                            index_segment = 0
                        elif flag is False and index2 == len(com) - 1:
                            pims[index_pim] = {'segment0' : [sent1, time1, user1, id1]}
                            index_pim += 1
                            temp.append((sent1, time1, user1, id1))
                            pims[index_pim] = {'segment0' : [sent2, time2, user2, id2]}
                            index_pim += 1
                            temp.append((sent2, time2, user2, id2))
                        else:
                            pims[index_pim] = {'segment0' : [sent1, time1, user1, id1]}
                            index_pim += 1
                            temp.append((sent1, time1, user1, id1))
                        flag = False
            if flag is True:
                index_pim += 1
                index_segment = 0
            timerange_detailed.append(temp)
        return pims

    def wrap_community_by_time_refined(self, pims):
        inverse_dangling_pims = []
        pims_keys = list(pims.keys())
        i = 0
        j = 0
        while i != len(pims_keys):
            j = 0
            while j != len(pims_keys):
                if i != j and pims_keys[i] in pims and pims_keys[j] in pims:
                    if (pims[pims_keys[i]]['segment0'][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):
                        for seg in pims[pims_keys[i]].values():
                            pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()))] = seg
                        del pims[pims_keys[i]]

                        sorted_j = sorted(pims[pims_keys[j]].values(), key=lambda kv: kv[1], reverse=False)
                        temp_pims = {}
                        new_index = 0
                        for new_seg in sorted_j:
                            temp_pims['segment' + str(new_index)] = new_seg
                            new_index += 1
                        pims[pims_keys[j]] = temp_pims
                        j = -1
                        i = 0
                    # elif (pims[pims_keys[i]]['segment0'][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):

                    #     for seg in pims[pims_keys[i]].values():
                    #         pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()))] = seg
                    #     del pims[pims_keys[i]]

                    #     sorted_j = sorted(pims[pims_keys[j]].values(), key=lambda kv: kv[1], reverse=False)
                    #     temp_pims = {}
                    #     new_index = 0
                    #     for new_seg in sorted_j:
                    #         temp_pims['segment' + str(new_index)] = new_seg
                    #         new_index += 1
                    #     pims[pims_keys[j]] = temp_pims
                    #     j = -1
                    #     i = 0
                    # elif (pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):
                    #     for seg in pims[pims_keys[i]].values():
                    #         pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()))] = seg
                    #     del pims[pims_keys[i]]

                    #     sorted_j = sorted(pims[pims_keys[j]].values(), key=lambda kv: kv[1], reverse=False)
                    #     temp_pims = {}
                    #     new_index = 0
                    #     for new_seg in sorted_j:
                    #         temp_pims['segment' + str(new_index)] = new_seg
                    #         new_index += 1
                    #     pims[pims_keys[j]] = temp_pims
                    #     j = -1
                    #     i = 0
                j += 1
            i += 1
        for index, p in enumerate(pims.keys()):
            for seg in pims[p].keys():
                # pims[p][seg][0] = [' '.join(text for text in segment['originalText']) for segment in self.segments_list if segment['id'] == pims[p][seg][3]]
                pims[p][seg][0] = [segment['originalText'] for segment in self.segments_org["segments"] if segment['id'] == pims[p][seg][3]]
                inverse_dangling_pims.append(pims[p][seg][3])

        # c_len = 0
        # for segment in self.segments_list:
        #    if (segment['id'] not in inverse_dangling_pims):
        #        while c_len in pims.keys():
        #            c_len += 1
        #        pims[c_len] = {"segment0": [' '.join(text for text in segment['originalText']), segment['startTime'], segment['spokenBy'], segment['id']]}

        # Remove Redundent PIMs in a group and also for single segment as a topic accept it as a topic only if it has duration greater than 30 sec.
        new_pim = {}
        track_single_seg = []
        for pim in list(pims.keys()):
            if len(pims[pim]) == 1:
                if self.segments_map[pims[pim]["segment0"][3]]["duration"]>30:
                    if pims[pim]["segment0"][3] in track_single_seg:
                        continue
                    track_single_seg.append(pims[pim]["segment0"][3])
                    pass
                else:
                    continue
            seen = []
            new_pim[pim] = {}
            index = 0
            for seg in list(pims[pim]):
                if pims[pim][seg][3] in seen:
                    pass
                else:
                    new_pim[pim]['segment' + str(index)] = {}
                    new_pim[pim]['segment' + str(index)] = pims[pim][seg]
                    index += 1
                    seen.append(pims[pim][seg][3])

        return new_pim

    def h_communities(self, h_flag=False):
        fv, graph_list = self.compute_feature_vector_gpt()
        meeting_graph, yetto_prune = self.construct_graph(fv, graph_list)
        meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list, yetto_prune)
        community_set = community.best_partition(meeting_graph_pruned)
        community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
        clusters = []
        temp = []
        prev_com = 0
        for index,(word,cluster) in enumerate(community_set_sorted):
            if prev_com==cluster:
                temp.append(word)
                if index==len(community_set_sorted)-1:
                    clusters.append(temp)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append(word)
        if (h_flag):
            community_set_collection = []
            old_cluster = []
            for cluster in clusters:
                if len(cluster) >= 2:
                    graph_list_pruned = deepcopy(graph_list)
                    for k in graph_list.keys():
                        if k not in cluster:
                            del graph_list_pruned[k]

                    meeting_graph, yetto_prune = self.construct_graph(fv, graph_list_pruned)
                    meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list_pruned, yetto_prune)
                    community_set = community.best_partition(meeting_graph_pruned)
                    community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
                    i = 0
                    prev_cluster = 9999999999999999
                    for (sent, cls) in community_set_sorted:
                        if cls not in old_cluster:
                            community_set_collection.append((sent, cls))
                            old_cluster.append(cls)
                            prev_cluster = cls
                            i = cls
                        else:
                            if cls == prev_cluster:
                                community_set_collection.append((sent, i))
                                continue
                            while i in old_cluster:
                                i += 1
                            prev_cluster = cls
                            community_set_collection.append((sent, i))
                            old_cluster.append(i)
                    for (sent, cls) in community_set_sorted:
                        old_cluster.append(cls)
                else:
                    i = 0
                    while i in old_cluster:
                        i += 1
                    community_set_collection.append((cluster[0], i))
                    old_cluster.append(i)
            community_set_collection = sorted(community_set_collection, key = lambda x: x[1], reverse=False)
            community_timerange = self.refine_community(community_set_collection, graph_list)
            # logger.info("commnity timerange", extra={"timerange": community_timerange})
            pims = self.group_community_by_time(community_timerange)
            pims = self.wrap_community_by_time_refined(pims)
            logger.info("Final PIMs", extra={"PIMs": pims})
        else:
            community_set_collection = deepcopy(community_set_sorted)
            community_set_collection = sorted(community_set_collection, key = lambda x: x[1], reverse=False)
            community_timerange = self.refine_community(community_set_collection, graph_list)
            # logger.info("commnity timerange", extra={"timerange": community_timerange})
            pims = self.group_community_by_time(community_timerange)
            pims = self.wrap_community_by_time_refined(pims)
            logger.info("Final PIMs", extra={"PIMs": pims})
        return pims

    def get_communities_without_outlier(self):
        fv, graph_list = self.compute_feature_vector()
        logger.info("No of sentences is", extra={"sentence": len(fv.keys())})
        meeting_graph, yetto_prune = self.construct_graph(fv, graph_list)
        max_meeting_grap_pruned = None
        max_community_set = None
        v = 0.10
        i = 0
        edge_count = meeting_graph.number_of_edges()
        meeting_graph_pruned = meeting_graph
        while(i!=3):
            meeting_graph_pruned = self.prune_edges_outlier(meeting_graph_pruned, graph_list, yetto_prune, v)
            community_set = community.best_partition(meeting_graph_pruned)
            mod = community.modularity(community_set, meeting_graph_pruned)
            logger.info("Meeting Graph results", extra={"edges before prunning": edge_count, "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity": mod})
            i +=1
        community_set_sorted = self.compute_louvian_community(meeting_graph_pruned, community_set)
        community_timerange = self.refine_community(community_set_sorted, graph_list)
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time_refined(pims)
        logger.info("Final PIMs", extra={"PIMs": pims})
        return pims
