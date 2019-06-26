from community import communities



def computepims(segments, model1):
    community_extraction = communities.community_detection(segments, model1)
    pims = community_extraction.get_communities()
    return pims
