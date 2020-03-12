import os
import pickle
import boto3
from s3client.s3 import S3Manager

bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")
s3c = S3Manager(bucket_name=bucket)

def upload_mind_artifacts(mind_id, context_id, gc, lc):

    gc_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/gc.pkl"
    serialized_gc = pickle.dumps(gc)
    res = s3c.upload_object(serialized_gc, gc_path)

    lc_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/lc.pkl"
    serialized_lc = pickle.dumps(lc)
    res = s3c.upload_object(serialized_lc, lc_path)

    return True

def upload_graph(mind_id, context_id, kp_entity_graph):
    for n,d in kp_entity_graph.copy().nodes(data=True):
        if kp_entity_graph.nodes()[n].get('meet_freq_list'):
            kp_entity_graph.nodes()[n].pop('meet_freq_list')
        if kp_entity_graph.nodes()[n].get('art_freq_list'):
            kp_entity_graph.nodes()[n].pop('art_freq_list')

    kp_graph_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/kp_entity_graph.pkl"
    serialized_kp_graph = pickle.dumps(kp_entity_graph)
    res = s3c.upload_object(serialized_kp_graph, kp_graph_path)

    return


def upload_all_mind_artifacts(entity_community_map, gc, lc, entity_dict, mind_id, context_id):
    upload_mind_artifacts(mind_id, context_id, gc, lc)


    com_map_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/entity_community_map.pkl"
    serialized_com = pickle.dumps(entity_community_map)
    res = s3c.upload_object(serialized_com, com_map_path)

    ent_path = os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/entity_dict.pkl"
    serialized_ent = pickle.dumps(entity_dict)
    res = s3c.upload_object(serialized_ent, ent_path)

    return

