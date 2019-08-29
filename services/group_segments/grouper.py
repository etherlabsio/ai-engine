import json
from group_segments.grouper_topics import gettopics
from group_segments import grouper_segments

def getgroups(segments, model1):
    community_extraction = grouper_segments.community_detection(segments, model1)
    pims = community_extraction.get_communities()
    topics = gettopics(pims)

    return topics, pims
