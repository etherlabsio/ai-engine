import json
from group_segments.grouper_topics import get_topics
from group_segments import grouper_segments


def get_groups(segments, model1):
    community_extraction = grouper_segments.community_detection(segments, model1)
    pims = community_extraction.get_communities()
    topics = gettopics(pims)

    return topics, pims
