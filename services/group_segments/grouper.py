import json
from group_segments.grouper_topics import get_topics
from group_segments import grouper_segments


def get_groups(segments, model1, mind_dict, for_pims=False):
    community_extraction = grouper_segments.community_detection(
        segments, model1, mind_dict, compute_fv=(not for_pims)
    )
    try:
        pims = community_extraction.h_communities(h_flag=False)  # get hierarchy community
        print (not pims, pims)
        if not pims:
            raise Exception
    except Exception as e:
        pims = community_extraction.fallback_pims()

    topics = get_topics(pims)

    return topics, pims
