import statistics
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
from collections import Counter
from sklearn.cluster import KMeans
from group_segments.publish_info import post_to_slack, post_to_slack_topic
from group_segments.summarise import ClusterFeatures
from group_segments.extract_topic import get_topics
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
    segment_fv = {}
    segid_index = {}

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
        segments_fv = {}
        segid_index = {}
        index = 0
        bucket = "io.etherlabs." + os.getenv("ACTIVE_ENV", "staging2") + ".contexts"
        s3_obj = S3Manager(bucket_name=bucket)
        for segment in self.segments_list:
            if segment["originalText"] != "":
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
                        if segment["id"] in segments_fv.keys():
                            segments_fv[segment["id"]].append(fv[index])
                            segid_index[segment["id"]].append((sent, fv[index]))
                        else:
                            segments_fv[segment["id"]] = [fv[index]]
                            segid_index[segment["id"]] = [(sent, fv[index])]
                        fv_mapped_score[segment["id"]] = mind_score[ind]
                        index += 1
        self.segid_index = segid_index
        for key in segments_fv.keys():
            segments_fv[key] = np.mean(segments_fv[key], axis=0)
        self.segment_fv = segments_fv
        for segi in fv_mapped_score.keys():
            fv_mapped_score[segi] = np.mean(fv_mapped_score[segi])
        return fv, graph_list, fv_mapped_score

    def get_segment_scores(self):
        segments_score = {}
        bucket = "io.etherlabs." + os.getenv("ACTIVE_ENV", "staging2") + ".contexts"
        s3_obj = S3Manager(bucket_name=bucket)
        for segid in self.segments_order.keys():
            segments_score_dict = {}
            s3_path = (
                self.context_id
                + "/segment_rank/"
                + self.instance_id
                + "/"
                + segid
                + ".pkl"
            )
            bytestream = (s3_obj.download_file(file_name=s3_path))["Body"].read()
            segments_score_dict = pickle.loads(bytestream)
            for keys in segments_score_dict.keys():
                segments_score[keys] = segments_score_dict[keys]
        print (s3_path, bucket)
        print (segments_score)
        return segments_score

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

    def construct_graph_ns_max(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0

        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                c_weight = cosine(fv[nodea], fv[nodeb])
                meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                yetto_prune.append((nodea, nodeb, c_weight))

        max_connection = {}
        max_score = {}
        outlier_score = {}
        for node in meeting_graph.nodes():
            closest_connection_n = sorted(dict(meeting_graph[node]).items(), key=lambda kv:kv[1]["weight"], reverse=True)
            max_score_current = []
            max_connection[node] = closest_connection_n
            weights_n = list(map(lambda kv: (kv[1]["weight"]).tolist(), closest_connection_n))
            q3 = np.percentile(weights_n, 75)
            iqr = np.subtract(*np.percentile(weights_n, [75, 25]))
            outlier_score[node] = {}
            outlier_score[node]["outlier"] = q3 + 1 * iqr
            outlier_score[node]["iqr"] = iqr
            outlier_score[node]["q3"] = q3
            outlier_score[node]["weights_n"] = closest_connection_n
            outlier_score[node]["avg+pstd"] = statistics.mean(weights_n)+statistics.pstdev(weights_n)


        graph_data = deepcopy(meeting_graph.edges.data())
        for nodea, nodeb, weight in graph_data:
            if weight["weight"] > outlier_score[nodea]["outlier"] or (((self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [0])):
                pass
            elif (self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [2, -1, 1, 2] and weight["weight"] >= outlier_score[nodea]["q3"] :
                    #print (graph_list[nodea], graph_list[nodeb])
                pass
            else:
                meeting_graph.remove_edge(nodea, nodeb)
        for nodea, nodeb, weight in graph_data:
            if (self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [0]:
                #meeting_graph[nodea][nodeb]["Weight"] = outlier_score[nodea]["weights_n"][1][1]["weight"]
                meeting_graph[nodea][nodeb]["Weight"] = 1

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
                #if id1 != id2:
                if True:
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
        # c_len = 0
        # for segment in self.segments_org["segments"]:
        #     if segment["id"] not in self.segments_order.keys():
        #         while c_len in pims.keys():
        #             c_len += 1
        #         pims[c_len] = {
        #             "segment0": [
        #                 " ".join(text for text in segment["originalText"]),
        #                 segment["startTime"],
        #                 segment["spokenBy"],
        #                 segment["id"],
        #             ]
        #         }

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
        #                         ]n
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

        new_pim = {}
        track_single_seg = []
        for pim in list(pims.keys()):
            #if len(pims[pim]) == 1:
            #    track_single_seg.append(pims[pim]["segment0"][3])
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
        # Remove Redundent PIMs in a group and also for single segment as a topic accept it as a topic only if the word count is greater than 120.
        flag = False
        index = 0
        for pim in list(new_pim.keys()):
            if len(new_pim[pim]) > 1:
                flag = True
        if not flag:
            return new_pim

        index = 0
        for pim in list(new_pim.keys()):
            if len(new_pim[pim]) == 1:
                if (
                    len(
                        self.segments_map[new_pim[pim]["segment0"][-1]][
                            "originalText"
                        ].split(" ")
                    )
                    < 30
                ):
                    del new_pim[pim]
        return new_pim

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
            (fv, graph_list, fv_mapped_score) = self.get_computed_feature_vector_gpt()
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
            logger.info("Final PIMs", extra={"PIMs": pims})
        return pims

    def fallback_pims(self):
        logger.info("Unable to compute Groups, falling back to PIMs approach.")
        if self.compute_fv:
            fv, graph_list, fv_mapped_score = self.compute_feature_vector_gpt()
        else:
            fv, graph_list, fv_mapped_score = self.get_computed_feature_vector_gpt()
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
        for key in list(pims.keys())[:5]:
            new_pims[key] = deepcopy(pims[key])
        logger.info("Final PIMs", extra={"PIMs": new_pims})

        return new_pims

    def combine_pims_by_time(self, pims, group_info):
        print("Before Merging", len(pims.keys()))
        sorted_pims = sorted(pims.items(), key=lambda kv: kv[1]["segment0"][1], reverse=False)
        new_pims = {}
        merge_group = []
#         for pos, (index, pim) in enumerate(sorted_pims):
#             for pos1 ,(index1, pim1) in enumerate(sorted_pims):
#                 if index != index1 and pos1 - pos == 1:
#                     if self.segments_order[pim["segment" + str(len(pim.keys())-1)][-1]] - self.segments_order[pim1["segment0"][-1]] != -1:
#                         merge_group.append((index, index1))
        for pos, (index, pim) in enumerate(sorted_pims):
            for pos1 ,(index1, pim1) in enumerate(sorted_pims):
                if index != index1 and pos1 - pos == 1:
                    if group_info[pim["segment" + str(len(pim.keys())-1)][-1]] == group_info[pim1["segment0"][-1]]:
                        merge_group.append((index, index1))
        tracking_changes = {}
        pim_seg = {}
        for group1, group2 in merge_group:
            seg_update = []
            if group1 in tracking_changes.keys():
                already_appended_group = tracking_changes[group1]
                for seg in list(pims[group2].values()):
                    seg_update.append(seg)
                tracking_changes[group2] = already_appended_group
                pim_seg[already_appended_group].append(seg_update)
            else:
                for seg in list(pims[group1].values()) + list(pims[group2].values()):
                    seg_update.append(seg)
                tracking_changes[group2] = group1
                if group1 in pim_seg.keys():
                    pim_seg[group1].append(seg_update)
                else:
                    pim_seg[group1] = [seg_update]

        for index, (groupno, group) in enumerate(pim_seg.items()):
            index_n = 0
            new_pims[index] = {}
            for seg in [i for j in group for i in j]:
                new_pims[index]["segment" + str(index_n)] = seg
                index_n += 1
        inverse_merge_group = [group for group in pims.keys() if group not in [i for j in merge_group for i in j]]
        index = len(new_pims)
        for group in inverse_merge_group:
            new_pims[index] = pims[group]
            index +=1
        print("After Merging", len(new_pims.keys()))
        return new_pims

    def get_group_scores(self, group, segment_scores):
        group_scores = {}
        for groupid in group.keys():
            group_scores[groupid] = [segment_scores[segid] for segid in [group[groupid][s][-1] for s in group[groupid]] if segid in segment_scores.keys()]
            if group_scores[groupid] == []:
                group_scores[groupid] = 0
                print ("group which had no score: ", group[groupid])
            else:
                group_scores[groupid] = np.median(group_scores[groupid])
        print ("computed Group Scores")
        print ("group_scores:", group_scores)
        for groupid, position in sorted(group_scores.items(), key=lambda kv: kv[1], reverse=False):
            print ("Group ID: ", groupid, "   Group Ranking: ", position ,"\n\n")
            print (*[group[groupid][segkey][0] for segkey in group[groupid].keys()], sep="\n\n", end="\n\n")
        return group_scores

    def filter_groups(self, group, group_scores, segid_score):
        filtered_group = {}
        seg_list_fv = list(self.segment_fv.values())
        seg_list_id = list(self.segment_fv.keys())
        kmeans = KMeans(n_clusters=2, random_state=0).fit(seg_list_fv)
        s_map = {}
        for index, assigned in enumerate(kmeans.labels_):
            s_map[index] = assigned
        ###
        prev = 0
        print ("------------cluster  1--------------")
        for seg, cls in sorted(s_map.items(), key=lambda kv:kv[1]):
            if prev!=cls:
                print ("------------cluster  2--------------")
                prev=cls
            print (self.segments_map[seg_list_id[seg]]["originalText"], "\n\n")
        ###
        clusters = []
        temp = []
        prev_com = 0
        for index,(word,cluster) in enumerate(sorted(s_map.items(), key=lambda kv:kv[1])):
            if prev_com==cluster:
                temp.append(word)
                if index==len(s_map.items())-1:
                    clusters.append(temp)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append(word)

        cluster_score = []
        for cls in clusters:
            temp = []
            for cluster in cls:
                if seg_list_id[cluster] in segid_score.keys():
                    temp.append(segid_score[seg_list_id[cluster]])
            cluster_score.append(temp)
        print ("clusters ", clusters)
        if len(clusters) != 2:
            return group

        cluster1 = cluster_score[0]
        cluster2 = cluster_score[1]
        final_score = []
        for index, cluster in enumerate(cluster_score):
            temp = []
            for cls in cluster:
                # True indicates the element is the lower than all the elements in the next cluster
                if index == 0:
                    temp.append([False if cls>score else True for score in cluster2])
                else:
                    temp.append([False if cls>score else True for score in cluster1])
            final_score.append(temp)

        final = []
        for cluster in final_score:
            final_temp = []
            for cls in cluster:
                res = Counter(cls)
                final_temp.append( [True if res[True]>=res[False] else False][0])
            final.append(final_temp)

        print ("final score:", final)
        out = []
        prob = []

        final_score[0] = final[0]
        final_score[1] = final[1]
        for itr in [0,1]:
            result = dict(Counter(final_score[itr]))
            if True not in result.keys():
                result[True] = 0
            if False not in result.keys():
                result[False] = 0
            prob.append(result[True]/(result[False] + result[True]))
        print ("prob:", prob)
        threshold = 0.75
        out = []
        flag = False
        if prob[0] <= 0.75 and prob[1] <= 0.75:
            out = cluster[0] + cluster[1]
            flag = True
        if not flag and prob[0] >= threshold:
            out += clusters[0]
        if not flag and prob[1] >= threshold:
            out += clusters[1]
        if out == []:
            out = clusters[0] + clusters[1]
        print ("out: ", out)
        filtered_seg = [seg_list_id[x] for x in out]
        removed_seg = []
        for groupid in group.keys():
            count = Counter([True if seg in filtered_seg else False for seg in [group[groupid][x][-1] for x in group[groupid].keys()]])
            if True in count.keys():
                true_count = count[True]
            else:
                true_count = 0
            if False in count.keys():
                false_count = count[False]
            else:
                false_count = 0

            if (true_count)/(true_count+false_count) > 0.75:
                filtered_group[groupid] = group[groupid]
            else:
                for seg in [group[groupid][x][-1] for x in group[groupid]]:
                    removed_seg.append(seg)

        res = post_to_slack(self.instance_id, [self.segments_map[segid]["originalText"] for segid in removed_seg])
        return filtered_group

    def order_groups_by_group_scores(self, segment_scores, pims):
        ordered_groups = {}
        ss_sorted = sorted(segment_scores.items(), key = lambda kv: kv[1],reverse=False)
        index = 0
        for groupid, _ in ss_sorted:
            ordered_groups[index] = pims[groupid]
            index+=1
            if index==7:
                break
        return ordered_groups

    def summarise_groups(self, graph_list, sent_fv, group):
        summarised_text = []
        potential_topics = []
        for groupid in group.keys():
            group_info = []
            group_info = [self.segid_index[segid] for segid in [group[groupid][x][-1] for x in group[groupid].keys()]][0]
            text_list = [info[0] for info in group_info]
            fv_list = [info[1] for info in group_info]
            cf = ClusterFeatures(np.asarray(fv_list))
            res = cf.cluster(ratio=0.3)
            summarised_text.append([text_list[s] for s in res])
        potential_topics.append([tp.st_get_candidate_phrases(x[0]) for x in summarised_text])
        #post_to_slack_topic(self.instance_id, potential_topics)
        return True

    def itr_communities(self):
        v = 0
        t = 1
        if self.compute_fv:
            fv, graph_list, fv_mapped_score = self.compute_feature_vector_gpt()
        else:
            (fv, graph_list, fv_mapped_score) = self.get_computed_feature_vector_gpt()
            if (self.mind_id).lower() in ["01daaqy88qzb19jqz5prjfr76y", "01daatanxnrqa35e6004hb7mbn"] :
                segment_scores = self.get_segment_scores()
        print ("Number of sentences: ", len(graph_list))
        meeting_graph_pruned, yetto_prune = self.construct_graph_ns_max(fv, graph_list)
        #meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list, yetto_prune, v)
        print ("Graph population done.")
        l_mod = 1
        flag = False
        community_set = None
        for itr in range(5):
            community_set, mod = self.compute_louvain_community(meeting_graph_pruned, t)
            if mod < l_mod:
                l_mod = mod
                community_set_collection = community_set
                flag = True
        if not flag:
            community_set_collection = community_set

        print ("Computed Community.")
        community_timerange = self.refine_community(community_set_collection, graph_list)
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time_refined(pims)
        pims = self.order_groups_by_score(pims, fv_mapped_score)
        print ("Computed Groups.", len(pims.keys()))
        #pims = self.combine_pims_by_time(pims)
        graph_list_index = {}
        for index, g in enumerate(graph_list.values()):
            if g[-1] not in graph_list_index.keys():
                graph_list_index[g[-1]] = [index]
            else:
                graph_list_index[g[-1]].append(index)
        group = []
        pim_fv = []
        for pim in pims.keys():
            pim_seg = []
            fv_index= []
            for seg in pims[pim].keys():
                if pims[pim][seg][-1] in self.segments_order.keys():
                    pim_seg.append(pims[pim][seg])
                    fv_index.append([fv[x] for x in graph_list_index[pims[pim][seg][-1]]])
            group.append(pim_seg)
            pim_fv.append(np.mean([ i for j in fv_index for i in j], axis=0))
        print ("Getting para features for phase 2.")
        G2 = nx.Graph()
        for index1 in range(len(pim_fv)):
            for index2 in range(len(pim_fv)):
                G2.add_edge(index1, index2, weight = cosine(pim_fv[index1], pim_fv[index2]))
        #G3 = deepcopy(self.construct_graph_para_new(G2))
        print ("Populated phase 2 graph.")
        cs2 = community.best_partition(G2, resolution=1.0)
        cs2_sorted = sorted(cs2.items(), key = lambda x: x[1], reverse=False)
        prev = 0
        group_seg_list = {}
        for seg, cluster in cs2_sorted:
            if prev !=cluster:
                prev=cluster
            for segi in list(map(lambda kv: kv[-1], group[seg])):
                group_seg_list[segi] = cluster
        print ("Computed phase 2 community.")
        pims = self.combine_pims_by_time(pims, group_seg_list)
        print ("Computed phase 2 Groups")
        if (self.mind_id).lower() not in ["01daaqy88qzb19jqz5prjfr76y", "01daatanxnrqa35e6004hb7mbn"] :

            logger.info("Final PIMs", extra={"PIMs": pims})
            return pims
        logger.info("Intermediate PIMs", extra={"PIMs": pims})
        print ("Ranking and filtering groups based on domain mind.")
        group_score = self.get_group_scores(pims, segment_scores)
        filtered_groups = self.filter_groups(pims, group_score, segment_scores)
        group_score = self.get_group_scores(filtered_groups, segment_scores)
        print ("Ordering based on group scores.")
        filtered_groups = self.order_groups_by_group_scores(group_score, filtered_groups)
        print ("Summarizing groups")
        self.summarise_groups(graph_list, fv, filtered_groups)
        print ("Extracting Topics")
        topics_extracted = get_topics(filtered_groups)
        # post_to_slack_topic(self.instance_id, topics_extracted)
        logger.info("Final PIMs", extra={"PIMs": filtered_groups})
        return filtered_groups


    def get_communities_without_outlier(self):
        fv, graph_list = self.compute_feature_vector()
        logger.info("No of sentences is", extra={"sentence": len(fv.keys())})
        meeting_graph, yetto_prune = self.construct_graph(fv, graph_list)
        max_meeting_grap_pruned = None
        max_community_set = None
        v = 0
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
