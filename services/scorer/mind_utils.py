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
    entity_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/entity_graph.pkl"
    entity_dl_path = os.path.join(os.sep, "tmp", "entity_graph.pkl")
    s3.Bucket(bucket).download_file(entity_path, entity_dl_path)
    entity_graph = pickle.load(open(entity_dl_path, "rb"))
    return entity_graph

def load_pg_scores(mind_id):
    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")
    pg_path = os.getenv("ACTIVE_ENV") + "/minds/" + mind_id + "/pg_scores.pkl"
    pg_dl_path = os.path.join(os.sep, "tmp", "pg_scores.pkl")
    s3.Bucket(bucket).download_file(pg_path, pg_dl_path)
    pg_scores = pickle.load(open(pg_dl_path, "rb"))
    return pg_scores
