import os
import pickle
import boto3

s3 = boto3.resource("s3")

def load_entity_features(mind_id):
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")
    entity_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/entity.pkl"
    entity_dl_path = os.path.join(os.sep, "tmp", "entity.pkl")
    s3.Bucket(bucket).download_file(entity_path, entity_dl_path)
    entity_dict = pickle.load(open(entity_dl_path, "rb"))
    return entity_dict

def load_entity_graph(mind_id):
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")

    noun_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/noun_graph.pkl"
    noun_dl_path = os.path.join(os.sep, "tmp", "noun_graph.pkl")
    s3.Bucket(bucket).download_file(noun_path, noun_dl_path)
    noun_graph = pickle.load(open(noun_dl_path, "rb"))

    kp_entity_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/kp_entity_graph.pkl"
    kp_entity_dl_path = os.path.join(os.sep, "tmp", "kp_entity_graph.pkl")
    s3.Bucket(bucket).download_file(kp_entity_path, kp_entity_dl_path)
    kp_entity_graph = pickle.load(open(kp_entity_dl_path, "rb"))

    comm_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/entity_community_map.pkl"
    comm_dl_path = os.path.join(os.sep, "tmp", "entity_community_map.pkl")
    s3.Bucket(bucket).download_file(comm_path, comm_dl_path)
    entity_community_map = pickle.load(open(comm_dl_path, "rb"))

    comm_rank_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/entity_community_rank.pkl"
    comm_rank_dl_path = os.path.join(os.sep, "tmp", "entity_community_rank.pkl")
    s3.Bucket(bucket).download_file(comm_rank_path, comm_rank_dl_path)
    entity_community_rank = pickle.load(open(comm_rank_dl_path, "rb"))

    return noun_graph, kp_entity_graph, entity_community_map, entity_community_rank
