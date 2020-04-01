import sys
import logging
import json
import numpy as np
from copy import deepcopy
from mind_enricher.graph_updater import get_grouped_segments, extract_information_from_groups, get_base_graph, update_entity_nodes, update_kp_nodes, update_edges, update_kp_tokens, update_entity_feat_dict, get_common_entities, get_most_similar_entities, get_agreable_communities, update_communitiy_artifacts, combine_sent_dicts
from group_segments.artifacts_uploader import  upload_graph, upload_all_mind_artifacts

logger = logging.getLogger()

def update_artifacts(Request, Artifacts):
    logger.info("Updating Artifacts:")
    try:
        master_paragraphs, master_ids = get_grouped_segments(Request.groups)
        ent_sent_dict, kp_sent_dict, label_dict, noun_list, entity_dict = extract_information_from_groups(master_paragraphs, master_ids, Artifacts.label_dict)
        all_sent_dict = combine_sent_dicts(ent_sent_dict, kp_sent_dict)
        logger.info("Updating Graph related Artifacts:")
        Artifacts.kp_entity_graph = get_base_graph(Artifacts.kp_entity_graph)
        Artifacts.kp_entity_graph, node_list = update_entity_nodes(Artifacts.kp_entity_graph, ent_sent_dict, label_dict)
        Artifacts.kp_entity_graph, node_list = update_kp_nodes(Artifacts.kp_entity_graph, ent_sent_dict, node_list, kp_sent_dict)
        Artifacts.kp_entity_graph = update_edges(Artifacts.kp_entity_graph, node_list, all_sent_dict)
        Artifacts.kp_entity_graph = update_kp_tokens(Artifacts.kp_entity_graph, node_list)
        upload_graph(Request.mind_id, Request.context_id, Artifacts.kp_entity_graph)
        logger.info("Updating Entity related Artifacts:")
        entity_dict = update_entity_feat_dict(entity_dict, Artifacts.ent_fv, Request.mind_id)

        fv_new, fv = get_common_entities(Artifacts.entity_community_map, entity_dict)
        if fv!={}:
            new_ent_placement = get_most_similar_entities(fv_new, fv)
            agreed_communities = get_agreable_communities(new_ent_placement, Artifacts.entity_community_map)
            Artifacts.entity_community_map, gc, lc = update_communitiy_artifacts(agreed_communities, Artifacts.entity_community_map, Artifacts.gc, Artifacts.lc)
            logger.info("Uploading the updated Artifacts:")
            upload_all_mind_artifacts(Artifacts.entity_community_map, gc, lc, entity_dict, Request.mind_id, Request.context_id)

    except Exception as e:
        raise Exception("Unable to update Artifacts: {}".format(e))

    return True

