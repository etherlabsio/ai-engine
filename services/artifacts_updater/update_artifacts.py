import sys
import logging
import json
import numpy as np
from copy import deepcopy
from artifacts_updater.graph_updater import get_grouped_segments, form_sentence_graph, get_base_graph, update_entity_nodes, update_kp_nodes, update_edges, update_kp_tokens, update_entity_feat_dict, get_common_entities, get_most_similar_entities, get_agreable_communities, update_communitiy_artifacts, combine_sent_dicts
from group_segments.artifacts_downloader import load_entity_features, load_entity_graph
from group_segments.artifacts_uploader import  upload_graph, upload_all_mind_artifacts

def update_artifacts(json_request):
    print ("INside update_artifacts")
    groups = json_request['group']
    new_groups = {}
    for groupid, groupobj in groups.items():
        new_groups[groupid] = groupobj['analyzedSegments']

    master_paragraphs, master_ids = get_grouped_segments(new_groups)
    entity_dict_full = load_entity_features((json_request['mindId']).lower(), json_request['contextId'])
    kp_entity_graph, entity_community_map, label_dict, gc, lc = load_entity_graph((json_request['mindId']).lower(), json_request['contextId'])

    common_entities = entity_dict_full.keys() & entity_community_map.keys()
    ent_fv = {}
    for ent in common_entities:
        if True not in np.isnan([entity_dict_full[ent]]):
            ent_fv[ent] = entity_dict_full[ent]

    ent_sent_dict, kp_sent_dict, label_dict, noun_list, entity_dict = form_sentence_graph(master_paragraphs, master_ids, label_dict)
    all_sent_dict = combine_sent_dicts(ent_sent_dict, kp_sent_dict)
    kp_entity_graph = get_base_graph(kp_entity_graph)
    kp_entity_graph, node_list = update_entity_nodes(kp_entity_graph, ent_sent_dict, label_dict)
    kp_entity_graph, node_list = update_kp_nodes(kp_entity_graph, ent_sent_dict, node_list, kp_sent_dict)
    kp_entity_graph = update_edges(kp_entity_graph, node_list, all_sent_dict)
    kp_entity_graph = update_kp_tokens(kp_entity_graph, node_list)
    upload_graph((json_request['mindId']).lower(), (json_request['contextId']).lower(), kp_entity_graph)
    entity_dict = update_entity_feat_dict(entity_dict, ent_fv, (json_request['mindId']).lower())

    fv_new, fv = get_common_entities(entity_community_map, entity_dict)
    if fv!={}:
        new_ent_placement = get_most_similar_entities(fv_new, fv)
        agreed_communities = get_agreable_communities(new_ent_placement, entity_community_map)
        entity_community_map, gc, lc = update_communitiy_artifacts(agreed_communities, entity_community_map, gc, lc)
        upload_all_mind_artifacts(entity_community_map, gc, lc, entity_dict, (json_request['mindId']).lower(), json_request['contextId'])
    return True

