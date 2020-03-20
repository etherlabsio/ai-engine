from artifacts_updater.kp_extractor import cleanText, CandidateKPExtractor
from artifacts_updater.scorer import get_ner
from artifacts_updater.scorer import get_feature_vector, cosine
import nltk, itertools
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
import numpy as np

kp_model = CandidateKPExtractor()

def get_grouped_segments(groups):
    paragraphs = []
    meeting_groups = []
    for groupid, groupobj in groups.items():
        paragraphs.append(" ".join([seg["originalText"] for seg in groupobj]))
    meeting_groups.append(" <p_split>".join(paragraphs))
    ids = ["0"]
    return meeting_groups, ids


def extract_information_from_groups(master_paragraphs, master_ids, multi_label_dict):
    ent_sent_dict = {}
    kp_sent_dict = {}
    label_dict = multi_label_dict
    noun_list = []
    for meeting_ctr in range(len(master_paragraphs)):
        clean_text= cleanText(master_paragraphs[meeting_ctr]).lower()
        tok_sents = nltk.sent_tokenize(clean_text)
        tok_sents = [ele for ele in tok_sents if len(nltk.word_tokenize(ele))>5]
        merged_text = ' '.join(tok_sents)

        paragraph_split = merged_text.split('<p_split>')
        paragraph_sets = []
        process_ctr = 0
        for i in range(len(paragraph_split)):
            if process_ctr>=len(paragraph_split):
                break
            curr_para = paragraph_split[process_ctr]
            if len(curr_para.split(' '))>5:
                if curr_para[-1]!='.':
                    curr_para = curr_para.strip()+'.'
                para_len = len(nltk.sent_tokenize(curr_para))
                if para_len>=4:
                    paragraph_sets.append(curr_para.replace('..','. ').replace(':.','.'))
                    process_ctr+=1
                else:
                    while para_len<4 and process_ctr<len(paragraph_split)-1:
                        process_ctr+=1
                        text_to_add = paragraph_split[process_ctr].strip()
                        curr_para = curr_para+' '+text_to_add
                        if curr_para[-1]!='.':
                            curr_para = curr_para.strip()+'.'
                        para_len = len(nltk.sent_tokenize(curr_para))
                    curr_para = curr_para.replace('..','. ').replace(':.','.')
                    paragraph_sets.append(curr_para)
                    process_ctr+=1
            else:
                process_ctr+=1

        meet_id = master_ids[meeting_ctr]
        for p_no, para in enumerate(paragraph_sets):
            sent_list = nltk.sent_tokenize(para)
            master_sent_list = []
            master_entity_list = []
            master_kp_list = []
            sent_to_index = {}
            ner_result = get_ner(sent_list, lambda_function="ner")
            for idx, sent in enumerate(sent_list):
                sent_kps = kp_model.get_candidate_phrases(sent)
                _, sent_ent_labels = (ner_result[idx]['entities'], ner_result[idx]['labels'])
                if len(sent_ent_labels)>0:
                    master_sent_list.append(sent)
                    master_entity_list.append(sent_ent_labels)
                if len(sent_kps)>0:
                    master_kp_list.append(sent_kps)

            meet_id1 = str(meet_id)+"_"+str(p_no)
            for sent,ents in zip(master_sent_list,master_entity_list):
                for ent, lab in ents.items():
                    if ent in ent_sent_dict:
                        #update the value and assign
                        updated_sents = [sent]
                        if meet_id1 in ent_sent_dict[ent]:
                            curr_sents = list(ent_sent_dict[ent][meet_id1])
                            updated_sents = curr_sents+[sent]
                        ent_sent_dict[ent][meet_id1] = updated_sents
                        label_dict[ent].update([lab])
                    else:
                        ent_sent_dict[ent] = {meet_id1:[sent]}
                        label_counter = Counter(dict([(lbl, 0) for lbl in ents]))
                        label_counter.update([lab])
                        label_dict[ent] = label_counter
            for sent,kps in zip(master_sent_list,master_kp_list):
                for kp in kps:
                    if kp in kp_sent_dict:
                        #update the value and assign
                        updated_sents = [sent]
                        if meet_id1 in kp_sent_dict[kp]:
                            curr_sents = list(kp_sent_dict[kp][meet_id1])
                            updated_sents = list(kp_sent_dict[kp][meet_id1])+[sent]
                        kp_sent_dict[kp][meet_id1] = updated_sents
                    else:
                        kp_sent_dict[kp] = {meet_id1:[sent]}
            single_nouns = list(set([ele for ele in kp_model.get_candidate_phrases(para, [r"""singular_nn: {<NN>{1}}"""]) if len(ele.split(' '))==1]))
            single_nouns = [ele.lower() for ele in single_nouns]
            noun_list.extend(single_nouns)
    ent_single_label_dict = {node:max(counter,key=lambda x: counter[x]) for node,counter in label_dict.items()}
    entity_dict = {e:sum(dcts.values(),[]) for e,dcts in ent_sent_dict.items()}
    return ent_sent_dict, kp_sent_dict, label_dict, noun_list, entity_dict

def combine_sent_dicts(ent_sent_dict, kp_sent_dict):
    all_sent_dict = ent_sent_dict.copy()
    all_sent_dict.update(dict(map(lambda x: (x[0].lower(),x[1]),kp_sent_dict.items())))
    return all_sent_dict

def get_base_graph(ent_kp_graph):
    if any([d.get("is_ether_node","missing")=="missing" for n,d in ent_kp_graph.nodes(data=True)]):
        nodes_list = list(ent_kp_graph.nodes())
        for i,node in enumerate(nodes_list):
            ent_kp_graph.nodes[node]['is_ether_node'] = False
            ent_kp_graph.nodes[node]['ether_meet_ctr'] = 0
            ent_kp_graph.nodes[node]['ether_grp_ctr'] = 0
            ent_kp_graph.nodes[node]['ether_sent_ctr'] = 0
            ent_kp_graph.nodes[node]['ether_meet_freq_list'] = 0
            for j,node1 in enumerate(nodes_list):
                if j>i:
                    if ent_kp_graph.has_edge(node,node1):
                        ent_kp_graph[node][node1]['ether_meet_ctr'] = 0
                        ent_kp_graph[node][node1]['ether_grp_ctr'] = 0
                        ent_kp_graph[node][node1]['ether_sent_ctr'] = 0
    return ent_kp_graph

def update_entity_nodes(ent_kp_graph, ent_sent_dict, multi_label_dict):
    node_list = []
    for ent in ent_sent_dict:
        if ent.isdigit():
            continue
        node_list.append(ent)
        meet_dict = dict()
        for p,s in ent_sent_dict[ent].items():
            meet_dict[p.split("_")[0]] = meet_dict.get(p.split("_")[0],[]) + s
        meet_freq = list(map(lambda sent_list: len(sent_list),meet_dict.values()))
        if ent in ent_kp_graph:
            ent_kp_graph.nodes()[ent]['is_ether_node'] = True
            ent_kp_graph.nodes()[ent]['node_type'] = "entity"
            #ent_kp_graph.nodes()[ent]['node_label'] = ent_single_label_dict.get(ent,"N/A")
            ent_kp_graph.nodes()[ent]['node_label'] = multi_label_dict.get(ent, "N/A")
            ent_kp_graph.nodes()[ent]['ether_grp_ctr'] = ent_kp_graph.nodes()[ent].get('ether_grp_ctr',0) + len(ent_sent_dict[ent])
            ent_kp_graph.nodes()[ent]['ether_meet_ctr'] = ent_kp_graph.nodes()[ent].get('ether_meet_ctr',0) + len(meet_dict)
            ent_kp_graph.nodes()[ent]['ether_sent_ctr'] = ent_kp_graph.nodes()[ent].get('ether_sent_ctr',0) + sum(meet_freq)
            ent_kp_graph.nodes()[ent]['ether_meet_freq_list'] = ent_kp_graph.nodes()[ent].get('ether_meet_freq_list',[]) + list(map(lambda sent_list: len(sent_list),ent_sent_dict[ent].values()))
        else:
            ent_kp_graph.add_node(ent,
                                  node_type = "entity",
                                  is_ether_node = True,
#                                   node_label = ent_single_label_dict.get(ent,"N/A"),
                                  node_label = multi_label_dict.get(ent, "N/A"),
                                  ether_grp_ctr = len(ent_sent_dict[ent]),
                                  ether_meet_ctr = len(meet_dict),
                                  ether_meet_freq_list = meet_freq,
                                  ether_sent_ctr = sum(meet_freq),
                                  art_ctr = 0,
                                  para_ctr = 0,
                                  sent_ctr = 0)
    return ent_kp_graph, node_list

def update_kp_nodes(ent_kp_graph, ent_sent_dict, node_list, kp_sent_dict):
    all_ent_sents = []
    x = [all_ent_sents.extend(sents)  for p_dict in ent_sent_dict.values() for sents in p_dict.values() ]
    all_ent_sents = set(all_ent_sents)
    for kp in kp_sent_dict:
        node_list.append(kp.lower())
        meet_dict = dict()
        for p,s in kp_sent_dict[kp].items():
            meet_dict[p.split("_")[0]] = meet_dict.get(p.split("_")[0],[]) + s
        meet_freq = list(map(lambda sent_list: len(sent_list),meet_dict.values()))
        lower_kp = kp.lower()
        if lower_kp in ent_kp_graph:
            ent_kp_graph.nodes()[lower_kp]['is_ether_node'] = True
            ent_kp_graph.nodes()[lower_kp]['node_type'] = "key_phrase"
            ent_kp_graph.nodes()[lower_kp]['node_label'] = "N/A"
            ent_kp_graph.nodes()[lower_kp]['ether_grp_ctr'] = ent_kp_graph.nodes()[lower_kp].get('ether_grp_ctr',0) + len(kp_sent_dict[kp])
            ent_kp_graph.nodes()[lower_kp]['ether_meet_ctr'] = ent_kp_graph.nodes()[lower_kp].get('ether_meet_ctr',0) + len(meet_dict)
            ent_kp_graph.nodes()[lower_kp]['ether_meet_freq_list'] = ent_kp_graph.nodes()[lower_kp].get('ether_meet_freq_list',[]) + meet_freq
            ent_kp_graph.nodes()[lower_kp]['ether_sent_ctr'] = ent_kp_graph.nodes()[lower_kp].get('ether_sent_ctr',0) + sum(meet_freq)
        else:
            ent_kp_graph.add_node(lower_kp,
                                  node_type = "key_phrase",
                                  node_label = "N/A",
                                  is_ether_node = True,
                                  ether_grp_ctr = len(kp_sent_dict[kp]),
                                  ether_meet_ctr = len(meet_dict),
                                  ether_meet_freq_list = meet_freq,
                                  ether_sent_ctr = sum(meet_freq),
                                  art_ctr = 0,
                                  para_ctr = 0,
                                  sent_ctr = 0)
    return ent_kp_graph, node_list

def update_edges(ent_kp_graph, node_list, all_sent_dict):
    node_type_map = {"entity":"ent","key_phrase":"kp"}
    for a, node_a in enumerate(node_list):
        for b, node_b in enumerate(node_list):
            if b>a:
                node_type_a = ent_kp_graph.nodes()[node_a]['node_type']
                node_type_b = ent_kp_graph.nodes()[node_b]['node_type']
                node_typestring_a = node_type_map[node_type_a]
                node_typestring_b = node_type_map[node_type_b]
                grp_set_a = set(all_sent_dict[node_a])
                grp_set_b = set(all_sent_dict[node_b])
                grp_intersection = grp_set_a & grp_set_b
                if len(grp_intersection)<1:
                    continue
                meet_set_a = set(list(map(lambda x: x.split("_")[0],grp_set_a)))
                meet_set_b = set(list(map(lambda x: x.split("_")[0],grp_set_b)))
                meet_intersection = meet_set_a & meet_set_b
                sent_set_a = set(list(itertools.chain(*list(all_sent_dict[node_a].values()))))
                sent_set_b = set(list(itertools.chain(*list(all_sent_dict[node_b].values()))))
                sent_intersection = sent_set_a & sent_set_b
                if node_typestring_b=="kp" and len(sent_intersection)<1:
                    continue
                if ent_kp_graph.has_edge(node_a,node_b):
                    ent_kp_graph[node_a][node_b]['ether_meet_ctr'] = ent_kp_graph[node_a][node_b].get('ether_meet_ctr',0) + len(meet_intersection)
                    ent_kp_graph[node_a][node_b]['ether_grp_ctr'] = ent_kp_graph[node_a][node_b].get('ether_grp_ctr',0) + len(grp_intersection)
                    ent_kp_graph[node_a][node_b]['ether_sent_ctr'] = ent_kp_graph[node_a][node_b].get('ether_sent_ctr',0) + len(sent_intersection)
                else:
                    ent_kp_graph.add_edge(node_a,
                                          node_b,
                                          edge_type = node_typestring_a + "_to_" + node_typestring_b,
                                          ether_meet_ctr = len(meet_intersection),
                                          ether_grp_ctr = len(grp_intersection),
                                          ether_sent_ctr = len(sent_intersection),
                                          art_ctr = 0,
                                          para_ctr = 0,
                                          sent_ctr = 0)
    return ent_kp_graph

def update_kp_tokens(ent_kp_graph, noun_list):
    entity_list = [node for node, d in ent_kp_graph.nodes(data=True) if d['node_type']=='entity']
    entity_list_lower = entity_list_lower = [ele.lower() for ele in entity_list]
    kp_nodes = [node for node, d in ent_kp_graph.nodes(data=True) if d['node_type']=='key_phrase']
    multi_tok_kps = [ele for ele in kp_nodes if len(ele.split(' '))>1]
    single_tok_kps = [ele for ele in kp_nodes if len(ele.split(' '))==1]
    multi_tok_kps = list(set(multi_tok_kps)-set(entity_list_lower))
    multi_kp_tokens = []
    for kp in multi_tok_kps:
        multi_kp_tokens.extend(kp.split(' '))
    multi_kp_tokens = list(set(multi_kp_tokens) - set(single_tok_kps) - set(entity_list_lower))
    noun_graph_tokens = list(set(multi_kp_tokens)&set(noun_list))
    nouns_to_update = []
    for noun_token in noun_graph_tokens:
        if noun_token in ent_kp_graph:
            ent_kp_graph.nodes()[noun_token]['is_ether_node'] = True
            ent_kp_graph.nodes()[noun_token]['node_label'] = "N/A"
            ent_kp_graph.nodes()[noun_token]['ether_meet_ctr'] = ent_kp_graph.nodes()[noun_token].get('ether_meet_ctr',0)
            ent_kp_graph.nodes()[noun_token]['ether_grp_ctr'] = ent_kp_graph.nodes()[noun_token].get('ether_grp_ctr',0)
            ent_kp_graph.nodes()[noun_token]['ether_sent_ctr'] = ent_kp_graph.nodes()[noun_token].get('ether_sent_ctr',0)
        else:
            nouns_to_update.append(noun_token)
            ent_kp_graph.add_node(noun_token,
                                  is_ether_node=True,
                                  node_type = 'kp_token',
                                  node_label="N/A",
                                  art_ctr = 0,
                                  para_ctr = 0,
                                  sent_ctr = 0)

    for kp in multi_tok_kps:
        kp_nouns = set(kp.split(' '))&set(noun_graph_tokens)
        for noun in kp_nouns:
            if ent_kp_graph.has_edge(kp,noun):
                ent_kp_graph[kp][noun]['edge_type'] = "kp_to_tok"
                ent_kp_graph[kp][noun]['ether_meet_ctr'] = ent_kp_graph[kp][noun].get('ether_meet_ctr',0) + ent_kp_graph.nodes[kp]['ether_meet_ctr']
                ent_kp_graph[kp][noun]['ether_grp_ctr'] = ent_kp_graph[kp][noun].get('ether_grp_ctr',0) + ent_kp_graph.nodes[kp]['ether_grp_ctr']
                ent_kp_graph[kp][noun]['ether_sent_ctr'] = ent_kp_graph[kp][noun].get('ether_sent_ctr',0) + ent_kp_graph.nodes[kp]['ether_sent_ctr']
            else:
                ent_kp_graph.add_edge(kp,
                                      noun,
                                      edge_type='kp_to_tok',
                                      ether_grp_ctr = ent_kp_graph.nodes[kp]['ether_grp_ctr'],
                                      ether_sent_ctr = ent_kp_graph.nodes[kp]['ether_sent_ctr'],
                                      ether_meet_ctr = ent_kp_graph.nodes[kp]['ether_meet_ctr'],
                                      art_ctr = 0,
                                      para_ctr = 0,
                                      sent_ctr = 0)

    for noun_token in nouns_to_update:
        ent_kp_graph.nodes[noun_token]['ether_meet_ctr'] = sum([d['ether_meet_ctr'] for n,d in ent_kp_graph[noun_token].items()])
        ent_kp_graph.nodes[noun_token]['ether_grp_ctr'] = sum([d['ether_grp_ctr'] for n,d in ent_kp_graph[noun_token].items()])
        ent_kp_graph.nodes[noun_token]['ether_sent_ctr'] = sum([d['ether_sent_ctr'] for n,d in ent_kp_graph[noun_token].items()])

    return ent_kp_graph

def update_entity_feat_dict(ent_sent_dict, ent_feat_dict, mind_id):
    for ent in ent_sent_dict:
        ent_feat = np.sum(get_feature_vector(ent_sent_dict[ent] , lambda_function="mind-"+mind_id), axis=0)
        if "<ETHER>-"+ent in ent_feat_dict:
            ent_feat_dict["<ETHER>-"+ent] += ent_feat
        else:
            ent_feat_dict["<ETHER>-"+ent] = ent_feat
    return ent_feat_dict

def get_common_entities(com_map, entity_dict):
    fv_new = {}
    common_entities = set(com_map.keys()) & set(entity_dict.keys())
    fv_new = dict([(ent, entity_dict[ent]) for ent in entity_dict.keys() if ent not in common_entities and "<ETHER>-" in ent])
    fv = dict([(ent, entity_dict[ent]) for ent in common_entities])
    return fv_new, fv

def get_most_similar_entities(fv_new, fv):
    placement = {}
    for ent, fv_ent in fv_new.items():
        most_similar = [(e, cosine(fv_ent, fv_old_ent)) for e, fv_old_ent in fv.items()]
        most_similar = list(sorted(most_similar, key=lambda kv:kv[1], reverse=True))[:10]
        ent_list = list(map(lambda kv:kv[0], most_similar))
        placement[ent] = ent_list
    return placement

def get_agreable_communities(new_ent_placement, com_map):
    agreed_communities = {}
    for ent, ent_list in new_ent_placement.items():
        agreed_communities[ent] = [com_map[e] for e in ent_list if e in com_map.keys()]
        agg = Counter(agreed_communities[ent]).most_common()[0]
        if agg[1]>2:
            agreed_communities[ent] = agg[0]
        else:
            agreed_communities[ent] = -1
    return agreed_communities

def update_communitiy_artifacts(agreed_communities, com_map, gc, lc):
    updated_comm_list = []
    for new_ent, comm in agreed_communities.items():
        if comm!=-1:
            com_map[new_ent] = comm
            if comm in gc.keys():
                gc[comm] += 1
            else:
                gc[comm] = 1
            if comm in lc.keys():
                if len(lc[comm])!=5:
                    if comm not in updated_comm_list:
                        lc[comm].append(1)
                    else:
                        lc[comm].append(lc[comm].pop()+1)
                else:
                    if comm not in updated_comm_list:
                        del lc[comm][0]
                        lc[comm].append(1)
                    else:
                        lc[comm].append(lc[comm].pop()+1)
            else:
                lc[comm] = [1]
            updated_comm_list.append(comm)
    return com_map, gc, lc
