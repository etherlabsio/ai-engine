# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from networkx import pagerank
from copy import deepcopy
from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config
from dataclasses import dataclass
from scorer.pre_process import preprocess_text
from s3client.s3 import S3Manager
from group_segments.extra_preprocess import preprocess_text as pt
import os
import pickle
from collections import Counter

logger = logging.getLogger()

config = Config(connect_timeout=60, read_timeout=240, retries={"max_attempts": 0},)
lambda_client = boto3_client("lambda", config=config)


@dataclass
class TextSegment:
    id: str
    text: str
    speaker: str


@dataclass
class Score(TextSegment):
    score: float


def upload_fv(fv_list, Request, context_id, instance_id):
    try:
        bucket = "io.etherlabs." + os.getenv("ACTIVE_ENV", "staging2") + ".contexts"
        s3_path = (
            context_id + "/feature_vectors/" + instance_id + "/" + Request.id + ".pkl"
        )
        logger.info("The path used for s3.", extra={"S3": s3_path, "bucket": bucket})
        s3_obj = S3Manager(bucket_name=bucket)
        s3_obj.upload_object(pickle.dumps(fv_list), s3_path)
    except Exception as e:
        logger.info("Uploading failed ", extra={"exception:": e})
        return False
    return True


def get_score(
    mind_id: str,
    mind_dict,
    Request: TextSegment,
    context_id,
    instance_id,
    for_pims=False,
) -> Score:
    score = []
    pre_processed_input = pt(Request.text, scorer=True)
    lambda_function = "mind-" + mind_id
    transcript_text = Request.text
    if len(pre_processed_input) != 0:
        mind_input = json.dumps({"text": pre_processed_input})
        mind_input = json.dumps({"body": mind_input})
        # logger.info("sending request to mind service")
        if for_pims is False:
            transcript_score = get_feature_vector(
                mind_input,
                lambda_function,
                mind_dict,
                Request,
                context_id,
                instance_id,
            )
        else:
            response = get_feature_vector(
                mind_input,
                lambda_function,
                mind_dict,
                Request,
                context_id,
                instance_id,
                store_features=True,
            )
            if response is not False:
                return response
            else:
                return False
    else:
        return True
        transcript_score = 0.00001
        logger.warn("processing transcript: {}".format(transcript_text))
        logger.warn("transcript too small to process. Returning default score")
    # hack to penalize out of domain small transcripts coming as PIMs - word level
    if len(transcript_text.split(" ")) < 40:
        transcript_score = 0.1 * transcript_score
    score = 1 / transcript_score
    return score

def cosine(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def getClusterScore(mind_vec, sent_vec):
    n1 = norm(mind_vec, axis=1).reshape(1, -1)
    n2 = norm(sent_vec, axis=1).reshape(-1, 1)
    dotp = dot(sent_vec, mind_vec).squeeze(2)
    segment_scores = dotp / (n2 * n1)
    return segment_scores


def get_feature_vector(
    mind_input,
    lambda_function,
    mind_dict,
    Request,
    context_id,
    instance_id,
    store_features=False,
):
    invoke_response = lambda_client.invoke(
        FunctionName=lambda_function,
        InvocationType="RequestResponse",
        Payload=mind_input,
    )
    out_json = invoke_response["Payload"].read().decode("utf8").replace("'", '"')
    data = json.loads(json.loads(out_json)["body"])
    response = json.loads(out_json)["statusCode"]
    if store_features is True:
        vector_list = data["sent_feats"][0]
        upload_fv(vector_list, Request, context_id, instance_id)
        return vector_list

    feats = list(mind_dict["feature_vector"].values())
    mind_vector = np.array(feats).reshape(len(feats), -1)
    transcript_score = 0.00001
    transcript_mind_list = []
    transcript_score_list = []
    if response == 200:
        logger.info("got {} from mind server".format(response))
        feature_vector = np.array(data["sent_feats"][0])
        if len(feature_vector) > 0:
            # For paragraphs, uncomment below LOC
            # feature_vector = np.mean(np.array(feature_vector),0).reshape(1,-1)
            batch_size = min(10, feature_vector.shape[0])
            for i in range(0, feature_vector.shape[0], batch_size):
                mind_vec = np.expand_dims(np.array(mind_vector), 2)
                sent_vec = feature_vector[i : i + batch_size]

                cluster_scores = getClusterScore(mind_vec, sent_vec)

                batch_scores = cluster_scores.max(1)
                transcript_score_list.extend(batch_scores)

                minds_selected = cluster_scores.argmax(1)
                transcript_mind_list.extend(minds_selected)
            transcript_score = np.mean(transcript_score_list)
            logger.info(
                "Mind Selected is {}".format(
                    {
                        ele: transcript_mind_list.count(ele)
                        for ele in set(transcript_mind_list)
                    }
                )
            )
    else:
        logger.debug(
            "Invalid response from mind service for input: {}".format(mind_input)
        )
        logger.debug("Returning default score")

    return transcript_score

def get_similar_entities(ent_fv, segment_info, sent_fv ):
    ent_score = []
    segment_fv = np.mean(sent_fv, axis=0)
    for ent in ent_fv.keys():
        ent_score.append((ent, cosine(segment_fv, ent_fv[ent])))
    ent_score_sorted = (sorted(ent_score, key=lambda kv: kv[1], reverse=True))[:10]
    return (segment_info.id, ent_score_sorted)

def get_segment_rank(ent_list, ent_graph):
    current_ent_list = [ent for ent, score in ent_list[1]]
    subgraph = deepcopy(ent_graph.subgraph(current_ent_list))
    mod_graph = deepcopy(ent_graph)
    mod_graph.add_node('-1')
    nodes_list_modgraph = mod_graph.nodes()
    for nodea, nodeb, param in subgraph.edges.data():

        if nodea in nodes_list_modgraph:
            mod_graph.remove_node(nodea)
            for node1, param1 in dict(ent_graph[nodea]).items():
                mod_graph.add_edge(node1,  '-1', edge_freq = param1['edge_freq'], edge_ctr = param1['edge_ctr'])

        if nodeb in nodes_list_modgraph:
            mod_graph.remove_node(nodeb)
            for node1, param1 in dict(ent_graph[nodeb]).items():
                mod_graph.add_edge(node1,  '-1', edge_freq = param1['edge_freq'], edge_ctr = param1['edge_ctr'])

    if dict(mod_graph['-1']) != {}:
        pg = pagerank(mod_graph, weight = 'edge_freq')

        for index, (node, score) in enumerate(sorted(pg.items(), key=lambda kv: kv[1], reverse=True)):
            if node == '-1':
                return (ent_list[0], index)
    else:
        return (ent_list[0], 10**6)

def get_segment_rank_pc(ent_list, pg_scores, com_map, ranked_com, mind_id):
    print ("\n\n Segid: ", ent_list[0])
    current_ent_list = [ent for ent, score in ent_list[1]]
    print (current_ent_list)
    degree_score_filtered = [com_map[ent] for ent in current_ent_list]
    print (degree_score_filtered)
    degree_score_filtered = [cls for cls in degree_score_filtered if cls in ranked_com.keys()]
    print (degree_score_filtered)
    degree_map_filtered = [ranked_com[ent] for ent in degree_score_filtered]
    print (degree_map_filtered)
    degree_map_filtered = sorted(dict(Counter(degree_map_filtered)).items(), key=lambda kv:kv[1], reverse=True)[0][0]
    print (degree_map_filtered)
    return (ent_list[0], degree_map_filtered)

def upload_segment_rank(segment_rank, instance_id, context_id, segment):
    try:
        bucket = "io.etherlabs." + os.getenv("ACTIVE_ENV", "staging2") + ".contexts"
        s3_obj = S3Manager(bucket_name=bucket)
        for segid, rank in segment_rank:
            segment_rank_dict = {}
            segment_rank_dict[segid] = rank
            s3_path = (
                context_id + "/segment_rank/" + instance_id + "/" + segid + ".pkl"
            )
            logger.info("The path used for s3.", extra={"S3": s3_path, "bucket": bucket})
            s3_obj.upload_object(pickle.dumps(segment_rank_dict), s3_path)
    except Exception as e:
        logger.info("Uploading failed ", extra={"exception:": e})
        return False
    return True
