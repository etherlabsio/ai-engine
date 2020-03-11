import os
import pickle
import boto3

s3 = boto3.resource("s3")

def upload_mind_artifacts(mind_id, context_id, gc, lc):
    pickle.dump(gc, open("/tmp/gc.pkl","wb"))
    pickle.dump(lc, open("/tmp/lc.pkl","wb"))

    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")

    gc_path = "artifacts/" + os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/gc.pkl"
    s3.Bucket(bucket).upload_fileobj(open("/tmp/gc.pkl","rb"), gc_path)

    lc_path = "artifacts/" + os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/lc.pkl"
    s3.Bucket(bucket).upload_fileobj(open("/tmp/lc.pkl","rb"), gc_path)

    return True

def upload_graph(mind_id, context_id, kp_entity_graph):
    for n,d in kp_entity_graph.copy().nodes(data=True):
        if kp_entity_graph.nodes()[n].get('meet_freq_list'):
            kp_entity_graph.nodes()[n].pop('meet_freq_list')
        if kp_entity_graph.nodes()[n].get('art_freq_list'):
            kp_entity_graph.nodes()[n].pop('art_freq_list')

    pickle.dump(kp_entity_graph, open("/tmp/kp_entity_graph.pkl","wb"))

    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")

    graph_path = "artifacts/" + os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/kp_entity_graph.pkl"
    s3.Bucket(bucket).upload_fileobj(open("/tmp/kp_entity_graph.pkl","rb"), graph_path)

    return


def upload_all_mind_artifacts(entity_community_map, gc, lc, entity_dict, mind_id, context_id):
    print ("uploading all mind artifacts.")
    pickle.dump(gc, open("/tmp/gc.pkl","wb"))
    pickle.dump(lc, open("/tmp/lc.pkl","wb"))
    pickle.dump(entity_community_map, open("/tmp/com_map.pkl","wb"))
    pickle.dump(entity_dict, open("/tmp/entity.pkl","wb"))

    bucket = os.getenv("BUCKET_NAME", "io.etherlabs.artifacts")

    gc_path = "artifacts/" + os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/gc.pkl"
    s3.Bucket(bucket).upload_fileobj(open("/tmp/gc.pkl","rb"), gc_path)

    lc_path = "artifacts/" + os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/lc.pkl"
    res = s3.Bucket(bucket).upload_fileobj(open("/tmp/lc.pkl","rb"), gc_path)
    print("juploading res", res)
    com_map_path = "artifacts/" + os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/entity_community_map.pkl"
    s3.Bucket(bucket).upload_fileobj(open("/tmp/com_map.pkl","rb"), com_map_path)

    ent_path = "artifacts/" + os.getenv("ACTIVE_ENV") + "/contexts/" + context_id + "/" + mind_id + "/entity.pkl"
    s3.Bucket(bucket).upload_fileobj(open("/tmp/entity.pkl","rb"), ent_path)

    return

