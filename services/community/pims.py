from community import communities
from community import s3upload
import json
from os import getenv
import logging

logger = logging.getLogger()


def computetopics(pims):
    topics = {}
    topics['topics']=[]
    for i in pims:
        #if (len(pims[i]))>=2:
        new_topic={}
        new_topic['id']=pims[i]['segment0'][3]
        #new_topic['text']=pims[i]['segment0'][0]
        new_topic['no_of_segment']=len(pims[i])
        new_topic['authors']=pims[i]['segment0'][2]
        new_topic['authoredAt']=pims[i]['segment0'][1]
        topics['topics'].append(new_topic)
    return topics
        

def uploadtos3(pims, topics, instanceId):
    s3Obj = s3upload.S3Manager(bucket_name="io.etherlabs.staging2.contexts")
    s3_key = "topics-extractor/v1/"+instanceId+".json"
    s3Obj.upload_object(topics, s3_key)
    s3_key = "community-extractor/v1/"+instanceId+".json"
    s3Obj.upload_object(pims, s3_key)
    return True

def computepims(segments, model1):
    community_extraction = communities.community_detection(segments, model1)
    pims = community_extraction.get_communities()
    topics = computetopics(pims)
    #s3_connection = boto.connect_s3()
    #bucket = s3_connection.get_bucket('io.etherlabs.staging2.contexts')
    #key = boto.s3.key.Key(bucket, 'some_file.zip')
    #with open('some_file.zip') as f:
    #    key.send_file(f)
    logger.info("Topics identified", extra={"Topics are": topics})
    return topics, pims
    #return pims
