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
