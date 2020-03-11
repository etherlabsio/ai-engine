import os
import pickle
import boto3

s3 = boto3.resource("s3")

def load_entity_features(mind_id, context_id):
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")
    entity_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/entity.pkl"
    entity_dl_path = os.path.join(os.sep, "tmp", "entity.pkl")
    s3.Bucket(bucket).download_file(entity_path, entity_dl_path)
    entity_dict = pickle.load(open(entity_dl_path, "rb"))
    return entity_dict

def load_entity_graph(mind_id, context_id):
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")

    kp_entity_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/kp_entity_graph.pkl"
    kp_entity_dl_path = os.path.join(os.sep, "tmp", "kp_entity_graph.pkl")
    s3.Bucket(bucket).download_file(kp_entity_path, kp_entity_dl_path)
    kp_entity_graph = pickle.load(open(kp_entity_dl_path, "rb"))

    comm_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/entity_community_map.pkl"
    comm_dl_path = os.path.join(os.sep, "tmp", "entity_community_map.pkl")
    s3.Bucket(bucket).download_file(comm_path, comm_dl_path)
    entity_community_map = pickle.load(open(comm_dl_path, "rb"))

    comm_rank_path_gc = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/gc.pkl"
    comm_rank_dl_path_gc = os.path.join(os.sep, "tmp", "gc.pkl")
    s3.Bucket(bucket).download_file(comm_rank_path_gc, comm_rank_dl_path_gc)
    entity_community_rank_gc = pickle.load(open(comm_rank_dl_path_gc, "rb"))

    comm_rank_path_lc = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/lc.pkl"
    comm_rank_dl_path_lc = os.path.join(os.sep, "tmp", "lc.pkl")
    s3.Bucket(bucket).download_file(comm_rank_path_lc, comm_rank_dl_path_lc)
    entity_community_rank_lc = pickle.load(open(comm_rank_dl_path_lc, "rb"))

    label_dict_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/label_dict.pkl"
    label_dict_dl_path = os.path.join(os.sep, "tmp", "label_dict.pkl")
    s3.Bucket(bucket).download_file(label_dict_path, label_dict_dl_path)
    label_dict = pickle.load(open(label_dict_dl_path, "rb"))


    return kp_entity_graph, entity_community_map, label_dict, entity_community_rank_gc, entity_community_rank_lc
