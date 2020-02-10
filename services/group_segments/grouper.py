import json
from group_segments.grouper_topics import get_topics
from group_segments import grouper_segments


def get_groups(Request, lambda_function, for_pims=False):
    community_extraction = grouper_segments.community_detection(
        Request, lambda_function, compute_fv=(not for_pims)
    )
    try:
        pims, topics_extracted = community_extraction.itr_communities()
        if not pims:
            raise Exception
    except Exception as e:
        print ("Error while forming groups: ", e )
        topics_extracted, pims = community_extraction.fallback_pims()

    #topics = get_topics(pims)

    return topics_extracted, pims
