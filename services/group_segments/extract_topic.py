from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config
import math
from copy import deepcopy
from scorer.scorer import getClusterScore
import numpy as np
from networkx import pagerank
import networkx as nx
import statistics
from text_preprocessing import preprocess as tp

logger = logging.getLogger(__name__)

config = Config(connect_timeout=240, read_timeout=240, retries={"max_attempts": 0},)
lambda_client = boto3_client("lambda", config=config)


def cosine(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def get_feature_vector(input_list, lambda_function):
    batches_count = 300
    feature_vector = []
    count = math.ceil(len(input_list) / batches_count)
    logger.info(
        "computing in batches",
        extra={"batches count": count, "number of sentences": len(input_list)},
    )
    for itr in range(count):
        extra_input = deepcopy(
            input_list[itr * batches_count : (itr + 1) * batches_count]
        )
        mind_input = json.dumps({"text_input": extra_input})
        mind_input = json.dumps({"body": mind_input})
        logger.info(
            "getting feature vector from mind service", extra={"iteration count:": itr},
        )
        invoke_response = lambda_client.invoke(
            FunctionName=lambda_function,
            InvocationType="RequestResponse",
            Payload=mind_input,
        )
        logger.info("Request Sent", extra={"iteration count": itr})
        out_json = invoke_response["Payload"].read().decode("utf8").replace("'", '"')
        data = json.loads(json.loads(out_json)["body"])
        response = json.loads(out_json)["statusCode"]

        if response == 200:
            feature_vector.extend(data["embeddings"][0])
            logger.info("Response Recieved")

        else:
            logger.error("Invalid response from  mind service")
    return feature_vector

def prune_edge(graph):
        c_weight = 0
        max_connection = {}
        max_score = {}
        outlier_score = {}
        for node in graph.nodes():
            closest_connection_n = sorted(dict(graph[node]).items(), key=lambda kv:kv[1]["weight"], reverse=True)
            weights_n = list(map(lambda kv: (kv[1]["weight"]).tolist(), closest_connection_n))
            q3 = np.percentile(weights_n, 75)
            iqr = np.subtract(*np.percentile(weights_n, [75, 25]))
            outlier_score[node] = {}
            outlier_score[node]["outlier"] = q3 + 1 * iqr
            outlier_score[node]["iqr"] = iqr
            outlier_score[node]["q3"] = q3
            outlier_score[node]["weights_n"] = closest_connection_n
            outlier_score[node]["avg+pstd"] = statistics.mean(weights_n)+statistics.pstdev(weights_n)


        graph_data = deepcopy(graph.edges.data())
        for nodea, nodeb, weight in graph_data:
            if weight["weight"] >= outlier_score[nodea]["q3"] :
                pass
            else:
                graph.remove_edge(nodea, nodeb)
        return graph

def get_topics(groups):
    result = {
    }
    text_list = []
    for groupid in groups.keys():
        text_list.append(" ".join([groups[groupid][segkey][0][0] for segkey in groups[groupid].keys()]))
    # text_list = [" ".join([groups[id][segkey][0] for segkey in groups[id].keys()]) for id in groups.keys()]

    for pos, group in enumerate(groups.keys()):
        text = text_list[pos]
        candidate_topics = tp.st_get_candidate_phrases(text)
        kg = nx.Graph()
        kg.add_nodes_from(candidate_topics)
        kp_fv = {}
        #kp_fv_raw = [gpt_model.get_text_feats(k+".") for k in candidate_topics]
        ct_fv = get_feature_vector([i+'.' for i in candidate_topics], "sentence-encoder-lambda")
        kp_fv_raw = [ct_fv[index] for index, k in enumerate(candidate_topics)]
        for index, kp in enumerate(candidate_topics):
            kp_fv[kp] = kp_fv_raw[index]
        for index1, nodea in enumerate(kg.nodes()):
            for index2, nodeb in enumerate(kg.nodes()):
                if index2 >= index1:
                    kg.add_edge(nodea, nodeb, weight = 1 - cosine(kp_fv[nodea], kp_fv[nodeb]))
        kg = deepcopy(prune_edge(kg))
        pg = pagerank(kg, weight="weight")
        pg_sorted = (sorted(pg.items(), key=lambda kv:kv[1], reverse=True))[:10]
        #print (pg_sorted)
        #text_fv = gpt_model.get_text_feats(text)
        text_fv = get_feature_vector([text], "sentence-encoder-lambda")[0]
        ranked = [(kp, 1-cosine(text_fv, kp_fv[kp])) for kp in [i[0] for i in pg_sorted]]
        pg_sorted = (sorted(ranked, key=lambda kv:kv[1], reverse=True))[:5]
        result[group] = [i for i, j in pg_sorted[:2]]
        print (group, pg_sorted[:2])
        #groups[groupid]["segment0"]["topics"] = [topic for topic, score in pg_sorted[:2]]
    return result
