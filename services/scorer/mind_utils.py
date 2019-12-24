import os
import pickle
import boto3

s3 = boto3.resource("s3")


def load_mind_features(mind_id):
    # BUCKET_NAME = io.etherlabs.artifacts
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")
    # MINDS = staging2/minds/
    mind_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/mind.pkl"
    mind_dl_path = os.path.join(os.sep, "tmp", "mind.pkl")
    s3.Bucket(bucket).download_file(mind_path, mind_dl_path)
    mind_dict = pickle.load(open(mind_dl_path, "rb"))

    return mind_dict

def load_entity_features(mind_id):
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")
    entity_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/entity.pkl"
    entity_dl_path = os.path.join(os.sep, "tmp", "entity.pkl")
    s3.Bucket(bucket).download_file(entity_path, entity_dl_path)
    entity_dict = pickle.load(open(entity_dl_path, "rb"))
    return entity_dict

def load_entity_graph(mind_id):
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")
    entity_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/pg_scores.pkl"
    entity_dl_path = os.path.join(os.sep, "tmp", "pg_scores.pkl")
    s3.Bucket(bucket).download_file(entity_path, entity_dl_path)
    entity_pg_scores = pickle.load(open(entity_dl_path, "rb"))

    entity_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/community_map.pkl"
    entity_dl_path = os.path.join(os.sep, "tmp", "entity_community_map.pkl")
    s3.Bucket(bucket).download_file(entity_path, entity_dl_path)
    entity_community_map = pickle.load(open(entity_dl_path, "rb"))


    entity_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/community_rank.pkl"
    entity_dl_path = os.path.join(os.sep, "tmp", "entity_community_rank.pkl")
    s3.Bucket(bucket).download_file(entity_path, entity_dl_path)
    entity_community_rank = pickle.load(open(entity_dl_path, "rb"))

    return entity_pg_scores, entity_community_map, entity_community_rank
