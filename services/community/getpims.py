from community import getcommunity, utility, bert
import torch



def computerpims(request):
    cd = getcommunity.community_detection(request)
    graph_list = cd.get_communities()
    return graph_list
