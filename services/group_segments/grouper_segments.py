import pickle
import numpy as np
import json
import text_preprocessing.preprocess as tp
from group_segments import extra_preprocess
from collections import Counter
from copy import deepcopy
import networkx as nx
import math
from group_segments.scorer import cosine
import community
from datetime import datetime
from group_segments import scorer
import logging
import math
import os
from s3client.s3 import S3Manager
from group_segments import scorer
from log.logger import setup_server_logger

logger = logging.getLogger()


class community_detection:
    segments_list = []
    segments_org = []
    segments_map = {}
    segments_order = {}
    lambda_function = None
    mind_features = None
    mind_id = None
    context_id = None
    instance_id = None
    compute_fv = True

    def __init__(self, Request, lambda_function, mind_f, compute_fv):
        self.segments_list = Request.segments
        self.segments_org = Request.segments_org
        self.segments_order = Request.segments_order
        self.segments_map = Request.segments_map
        self.lambda_function = lambda_function
        self.mind_features = mind_f
        self.compute_fv = compute_fv
        self.mind_id = Request.mind_id
        self.context_id = Request.context_id
        self.instance_id = Request.instance_id

    def compute_feature_vector_gpt(self):
        graph_list = {}
        fv_mapped_score = {}
        input_list = []
        fv = {}
        index = 0
        for segment in self.segments_list:
            for sent in segment["originalText"]:
                if sent != "":
                    input_list.append(sent)

        transcript_score, mind_score = scorer.get_feature_vector(
            input_list, self.lambda_function, self.mind_features
        )
        for segment in self.segments_list:
            for sent in segment["originalText"]:
                if sent != "":
                    graph_list[index] = (
                        sent,
                        segment["startTime"],
                        segment["spokenBy"],
                        segment["id"],
                    )
                    fv[index] = transcript_score[index]
                    if segment["id"] in fv_mapped_score.keys():
                        fv_mapped_score[segment["id"]].append(mind_score[index])
                    else:
                        fv_mapped_score[segment["id"]] = [mind_score[index]]
                    # fv_mapped_score[index] = (segment['id'], mind_score[index])
                    index += 1
        for segi in fv_mapped_score.keys():
            fv_mapped_score[segi] = np.mean(fv_mapped_score[segi])
        return fv, graph_list, fv_mapped_score

    def get_computed_feature_vector_gpt(self):
        graph_list = {}
        fv_mapped_score = {}
        input_list = []
        fv = {}
        index = 0
        bucket = "io.etherlabs." + os.getenv("ACTIVE_ENV", "staging2") + ".contexts"
        s3_obj = S3Manager(bucket_name=bucket)
        for segment in self.segments_list:
            if segment["originalText"] != []:
                s3_path = (
                    self.context_id
                    + "/feature_vectors/"
                    + self.instance_id
                    + "/"
                    + segment["id"]
                    + ".pkl"
                )
                bytestream = (s3_obj.download_file(file_name=s3_path))["Body"].read()
                segment_fv = pickle.loads(bytestream)
                # with open("/tmp/" + s3_path, "rb") as f:
                #    segment_fv = pickle.load(f)
                mind_score = scorer.get_mind_score(segment_fv, self.mind_features)
                assert len(mind_score) == len(segment_fv)
                for ind, sent in enumerate(segment["originalText"]):
                    if sent != "":
                        graph_list[index] = (
                            sent,
                            segment["startTime"],
                            segment["spokenBy"],
                            segment["id"],
                        )
                        fv[index] = segment_fv[ind]
                        fv_mapped_score[segment["id"]] = mind_score[ind]
                        index += 1
        for segi in fv_mapped_score.keys():
            fv_mapped_score[segi] = np.mean(fv_mapped_score[segi])
        return fv, graph_list, fv_mapped_score

    def construct_graph(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                c_weight = cosine(fv[nodea], fv[nodeb])
                meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                yetto_prune.append((nodea, nodeb, c_weight))

        X = nx.to_numpy_array(meeting_graph)

        for i in range(len(X)):
            X[i][i] = X[i].mean()

        norm_mat = (X - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1))
        norm_mat = (np.transpose(np.tril(norm_mat)) + np.triu(norm_mat)) / 2
        norm_mat = norm_mat + np.transpose(norm_mat)
        meeting_graph = nx.from_numpy_array(norm_mat)
        logger.info(
            "Completed Normalization",
            extra={
                "nodes: ": meeting_graph.number_of_nodes(),
                "edges: ": meeting_graph.number_of_edges(),
            },
        )

        for index in range(meeting_graph.number_of_nodes()):
            meeting_graph[index][index]["weight"] = 1

        logger.info(
            "Completed Normalization and after removing diagonal values",
            extra={
                "nodes: ": meeting_graph.number_of_nodes(),
                "edges: ": meeting_graph.number_of_edges(),
            },
        )
        yetto_prune = []
        for nodea, nodeb, weight in meeting_graph.edges.data():
            yetto_prune.append((nodea, nodeb, weight["weight"]))
        return meeting_graph, yetto_prune

    def construct_graph_next_segment(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                if self.segments_order[graph_list[nodeb][-1]] - self.segments_order[
                    graph_list[nodea][-1]
                ] in [0, 1]:
                    c_weight = cosine(fv[nodea], fv[nodeb])
                    meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                    yetto_prune.append((nodea, nodeb, c_weight))

        # X = nx.to_numpy_array(meeting_graph)

        # for i in range(len(X)):
        #     X[i][i] = X[i].mean()

        # norm_mat = (X - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1))
        # norm_mat = (np.transpose(np.tril(norm_mat)) + np.triu(norm_mat)) / 2
        # norm_mat = norm_mat + np.transpose(norm_mat)
        # meeting_graph = nx.from_numpy_array(norm_mat)
        # logger.info(
        #     "Completed Normalization",
        #     extra={
        #         "nodes: ": meeting_graph.number_of_nodes(),
        #         "edges: ": meeting_graph.number_of_edges(),
        #     },
        # )

        # for index in range(meeting_graph.number_of_nodes()):
        #     meeting_graph[index][index]["weight"] = 1

        # logger.info(
        #     "Completed Normalization and after removing diagonal values",
        #     extra={
        #         "nodes: ": meeting_graph.number_of_nodes(),
        #         "edges: ": meeting_graph.number_of_edges(),
        #     },
        # )
        # yetto_prune = []
        # for nodea, nodeb, weight in meeting_graph.edges.data():
        #     yetto_prune.append((nodea, nodeb, weight["weight"]))
        return meeting_graph, yetto_prune

    def prune_edges_outlier(self, meeting_graph, graph_list, yetto_prune, v):
        meeting_graph_pruned = nx.Graph()
        weights = []
        for nodea, nodeb, weight in meeting_graph.edges.data():
            meeting_graph_pruned.add_nodes_from([nodea, nodeb])
            weights.append(weight["weight"])

        q3 = np.percentile(weights, v)
        logger.info("Outlier Score", extra={"outlier threshold is : ": q3})

        for indexa, indexb, c_score in meeting_graph.edges.data():
            if c_score["weight"] >= q3:
                meeting_graph_pruned.add_edge(indexa, indexb, weight=c_score["weight"])

        return meeting_graph_pruned

    def compute_louvain_community(self, meeting_graph_pruned, t):
        community_set = community.best_partition(meeting_graph_pruned, resolution=t)
        modularity_score = community.modularity(community_set, meeting_graph_pruned)
        logger.info("Community results", extra={"modularity score": modularity_score})
        community_set_sorted = sorted(
            community_set.items(), key=lambda kv: kv[1], reverse=False
        )

        return community_set_sorted, modularity_score

    def refine_community(self, community_set_sorted, graph_list):
        clusters = []
        temp = []
        prev_com = 0
        seg_cls = {}
        seg_max = {}
        for index, (word, cluster) in enumerate(community_set_sorted):
            if cluster not in seg_cls.keys():
                seg_cls[cluster] = {}
            if prev_com == cluster:
                temp.append((word, graph_list[word][-1]))
                if index == len(community_set_sorted) - 1:
                    clusters.append(temp)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append((word, graph_list[word][-1]))

        for index, cluster in enumerate(clusters):
            seg_cls[index] = Counter(seg for sent, seg in cluster)
            seg_count = {}
            for segid, count in seg_cls[index].items():
                seg_count[segid] = count
            for segid in seg_count.keys():
                if segid not in seg_max.keys():
                    seg_max[segid] = (seg_count[segid], index)
                elif seg_count[segid] >= seg_max[segid][0]:
                    seg_max[segid] = (seg_count[segid], index)

        new_clusters = deepcopy(clusters)
        for index, cluster in enumerate(new_clusters):
            for sent, seg in cluster:
                if seg_max[seg][1] != index:
                    clusters[index].remove((sent, seg))

        timerange = []
        temp = []
        for cluster in clusters:
            temp = []
            for sent, seg in cluster:
                temp.append(graph_list[sent])
            if len(temp) != 0:
                temp = list(set(temp))
                temp = sorted(temp, key=lambda kv: kv[1], reverse=False)
                timerange.append(temp)
        return timerange

    def remove_preprocessed_segments(self, graph_list):
        graph_list_id = list(map(lambda x: x[-1], graph_list.values()))
        temp_segments_order = deepcopy(list(self.segments_order.items()))
        temp_segments_order = sorted(
            temp_segments_order, key=lambda kv: kv[1], reverse=False
        )
        sudo_index = 0
        for segid, index in temp_segments_order:
            if segid not in graph_list_id:
                del self.segments_order[segid]
            else:
                self.segments_order[segid] = sudo_index
                sudo_index += 1
        return True

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
                pims[index_pim] = {
                    "segment0": [com[0][0], com[0][1], com[0][2], com[0][3]]
                }
                index_pim += 1
                continue

            for (
                (index1, (sent1, time1, user1, id1)),
                (index2, (sent2, time2, user2, id2)),
            ) in zip(enumerate(com[0:]), enumerate(com[1:])):
                if id1 != id2:
                    # if ((extra_preprocess.format_time(time2, True) - extra_preprocess.format_time(time1, True)).seconds <= 120):
                    if (self.segments_order[id2] - self.segments_order[id1]) in [0, 1]:
                        if not flag:
                            pims[index_pim] = {
                                "segment"
                                + str(index_segment): [sent1, time1, user1, id1,]
                            }
                            index_segment += 1
                            temp.append((sent1, time1, user1, id1))
                        pims[index_pim]["segment" + str(index_segment)] = [
                            sent2,
                            time2,
                            user2,
                            id2,
                        ]
                        index_segment += 1
                        temp.append((sent2, time2, user2, id2))
                        flag = True
                    else:
                        if flag is True:
                            index_pim += 1
                            index_segment = 0
                        elif flag is False and index2 == len(com) - 1:
                            pims[index_pim] = {"segment0": [sent1, time1, user1, id1]}
                            index_pim += 1
                            temp.append((sent1, time1, user1, id1))
                            pims[index_pim] = {"segment0": [sent2, time2, user2, id2]}
                            index_pim += 1
                            temp.append((sent2, time2, user2, id2))
                        else:
                            pims[index_pim] = {"segment0": [sent1, time1, user1, id1]}
                            index_pim += 1
                            temp.append((sent1, time1, user1, id1))
                        flag = False
            if flag is True:
                index_pim += 1
                index_segment = 0
            timerange_detailed.append(temp)

        return pims

    def wrap_community_by_time_refined(self, pims):
        # Add segments which were removed while pre-processing.
        c_len = 0
        for segment in self.segments_org["segments"]:
            if segment["id"] not in self.segments_order.keys():
                while c_len in pims.keys():
                    c_len += 1
                pims[c_len] = {
                    "segment0": [
                        " ".join(text for text in segment["originalText"]),
                        segment["startTime"],
                        segment["spokenBy"],
                        segment["id"],
                    ]
                }

        # If one group can be placed in-between an another group, with a factor of time, then combine them into single group.
        inverse_dangling_pims = []
        pims_keys = list(pims.keys())
        i = 0
        j = 0
        while i != len(pims_keys):
            j = 0
            while j != len(pims_keys):
                if (
                    i != j
                    and pims_keys[i] in pims
                    and pims_keys[j] in pims
                    and (len(pims[pims_keys[i]]) != 1 or len(pims[pims_keys[j]]) != 1)
                ):
                    if (
                        pims[pims_keys[i]]["segment0"][1]
                        >= pims[pims_keys[j]]["segment0"][1]
                        and pims[pims_keys[i]]["segment0"][1]
                        <= pims[pims_keys[j]][
                            "segment" + str(len(pims[pims_keys[j]].values()) - 1)
                        ][1]
                    ) and (
                        pims[pims_keys[i]][
                            "segment" + str(len(pims[pims_keys[i]].values()) - 1)
                        ][1]
                        >= pims[pims_keys[j]]["segment0"][1]
                        and pims[pims_keys[i]][
                            "segment" + str(len(pims[pims_keys[i]].values()) - 1)
                        ][1]
                        <= pims[pims_keys[j]][
                            "segment" + str(len(pims[pims_keys[j]].values()) - 1)
                        ][1]
                    ):
                        for seg in pims[pims_keys[i]].values():
                            pims[pims_keys[j]][
                                "segment" + str(len(pims[pims_keys[j]].values()))
                            ] = seg
                        del pims[pims_keys[i]]

                        sorted_j = sorted(
                            pims[pims_keys[j]].values(),
                            key=lambda kv: kv[1],
                            reverse=False,
                        )
                        temp_pims = {}
                        new_index = 0
                        for new_seg in sorted_j:
                            temp_pims["segment" + str(new_index)] = new_seg
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

        # replace segment pre-processed text with orginal Text.
        for index, p in enumerate(pims.keys()):
            for seg in pims[p].keys():
                # pims[p][seg][0] = [' '.join(text for text in segment['originalText']) for segment in self.segments_list if segment['id'] == pims[p][seg][3]]
                pims[p][seg][0] = [
                    segment["originalText"]
                    for segment in self.segments_org["segments"]
                    if segment["id"] == pims[p][seg][3]
                ]
                if len(pims[p].keys()) != 1:
                    inverse_dangling_pims.append(pims[p][seg][3])

        # if a segment wasn't present in a group and can be placed right next to the last segment of a group, based on time, add it.
        # for segmentid in self.segments_order.keys():
        #     if segmentid not in inverse_dangling_pims:
        #         order = self.segments_order[segmentid]
        #         for pim in pims.keys():
        #             if len(pims[pim].keys()) != 1:
        #                 if (
        #                     self.segments_order[
        #                         pims[pim]["segment" + str(len(pims[pim].values()) - 1)][
        #                             -1
        #                         ]
        #                     ]
        #                     == order - 1
        #                 ):
        #                     print(
        #                         "appending extra segment based on order: ",
        #                         self.segments_map[segmentid],
        #                         pim,
        #                     )
        #                     pims[pim]["segment" + str(len(pims[pim].values()))] = (
        #                         self.segments_map[segmentid]["originalText"],
        #                         self.segments_map[segmentid]["spokenBy"],
        #                         self.segments_map[segmentid]["startTime"],
        #                         self.segments_map[segmentid]["id"],
        #                     )
        #                     break

        # Remove Redundent PIMs in a group and also for single segment as a topic accept it as a topic only if the word count is greater than 120.
        flag = False
        index = 0
        for pim in list(pims.keys()):
            if len(pims[pim]) > 1:
                flag = True
        if not flag:
            return pims

        index = 0
        for pim in list(pims.keys()):
            if len(pims[pim]) == 1:
                # if (
                #     len(
                #         self.segments_map[pims[pim]["segment0"][-1]][
                #             "originalText"
                #         ].split(" ")
                #     )
                #     < 120
                # ):
                del pims[pim]
        return pims

    def order_groups_by_score(self, pims, fv_mapped_score):
        new_pims = {}
        group_score_mapping = {}
        for key in list(pims.keys()):
            group_score = []
            for segi in pims[key].keys():
                if pims[key][segi][3] in fv_mapped_score.keys():
                    group_score.append(fv_mapped_score[pims[key][segi][3]])
                else:
                    group_score.append(0)
            group_score_mapping[key] = np.mean(group_score)

        sorted_groups = sorted(
            group_score_mapping.items(), key=lambda kv: kv[1], reverse=True
        )
        index = 0
        for groupid, score in sorted_groups:
            new_pims[index] = pims[groupid]
            # new_pims[index]['distance'] = score
            index += 1
        return new_pims

    def h_communities(self, h_flag):
        v = 0  # percentile pruning value
        t = 0.9
        if self.compute_fv:
            fv, graph_list, fv_mapped_score = self.compute_feature_vector_gpt()
        else:
            (fv, graph_list, fv_mapped_score,) = self.get_computed_feature_vector_gpt()
        # _ = self.remove_preprocessed_segments(graph_list)

        meeting_graph, yetto_prune = self.construct_graph_next_segment(fv, graph_list)
        meeting_graph_pruned = self.prune_edges_outlier(
            meeting_graph, graph_list, yetto_prune, v
        )
        # meeting_graph_pruned = deepcopy(meeting_graph)
        l_mod = 1
        flag = False
        community_set_sorted = None
        for itr in range(5):
            community_set, mod = self.compute_louvain_community(meeting_graph_pruned, t)
            if mod < l_mod:
                l_mod = mod
                community_set_sorted = community_set
                flag = True
        if not flag:
            community_set_sorted = community_set
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
        if h_flag:
            v = 75
            community_set_collection = []
            old_cluster = []
            for cluster in clusters:
                if len(cluster) >= 2:
                    graph_list_pruned = deepcopy(graph_list)
                    for k in graph_list.keys():
                        if k not in cluster:
                            del graph_list_pruned[k]

                    meeting_graph, yetto_prune = self.construct_graph(
                        fv, graph_list_pruned
                    )
                    meeting_graph_pruned = self.prune_edges_outlier(
                        meeting_graph, graph_list_pruned, yetto_prune, v
                    )
                    community_set = community.best_partition(meeting_graph_pruned)
                    community_set_sorted = sorted(
                        community_set.items(), key=lambda kv: kv[1], reverse=False,
                    )
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
            community_set_collection = sorted(
                community_set_collection, key=lambda x: x[1], reverse=False
            )
            community_timerange = self.refine_community(
                community_set_collection, graph_list
            )
            # logger.info("commnity timerange", extra={"timerange": community_timerange})
            pims = self.group_community_by_time(community_timerange)
            pims = self.wrap_community_by_time_refined(pims)
            logger.info("Final PIMs", extra={"PIMs": pims})
        else:
            community_set_collection = deepcopy(community_set_sorted)
            community_set_collection = sorted(
                community_set_collection, key=lambda x: x[1], reverse=False
            )
            community_timerange = self.refine_community(
                community_set_collection, graph_list
            )
            # logger.info("commnity timerange", extra={"timerange": community_timerange})
            pims = self.group_community_by_time(community_timerange)
            pims = self.wrap_community_by_time_refined(pims)
            pims = self.order_groups_by_score(pims, fv_mapped_score)
            logger.info("Final PIMs", extra={"PIMs": pims})
        return pims

    def fallback_pims(self):
        print("Unable to compute Groups, falling back to PIMs approach.")
        if self.compute_fv:
            fv, graph_list, fv_mapped_score = self.compute_feature_vector_gpt()
        else:
            (fv, graph_list, fv_mapped_score,) = self.get_computed_feature_vector_gpt()
        pims = {}
        for index, segment in enumerate(self.segments_org["segments"]):
            pims[index] = {}
            pims[index]["segment0"] = (
                segment["originalText"],
                segment["spokenBy"],
                segment["createdAt"],
                segment["id"],
            )
        pims = self.order_groups_by_score(pims, fv_mapped_score)
        new_pims = {}
        for key in pims.keys()[:5]:
            new_pims = deepcopy(pims[key])
        logger.info("Final PIMs", extra={"PIMs": new_pims})

        return new_pims

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
        while i != 3:
            meeting_graph_pruned = self.prune_edges_outlier(
                meeting_graph_pruned, graph_list, yetto_prune, v
            )
            community_set = community.best_partition(meeting_graph_pruned)
            mod = community.modularity(community_set, meeting_graph_pruned)
            logger.info(
                "Meeting Graph results",
                extra={
                    "edges before prunning": edge_count,
                    "edges after prunning": meeting_graph_pruned.number_of_edges(),
                    "modularity": mod,
                },
            )
            i += 1
        community_set_sorted = self.compute_louvian_community(
            meeting_graph_pruned, community_set
        )
        community_timerange = self.refine_community(community_set_sorted, graph_list)
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time_refined(pims)
        logger.info("Final PIMs", extra={"PIMs": pims})
        return pims
