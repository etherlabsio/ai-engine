import logging

logger = logging.getLogger()


def get_pims(Request):
    used_topics = []
    group_no = None
    index = 0
    topic_pim = {}
    ranked_pims = sorted(
        [(k, v) for (k, v) in Request.pim_result.items()], key=lambda kv: kv[1]
    )

    for (rec_id, distance) in ranked_pims:
        if rec_id in Request.gs_result.keys():
            group_no = Request.gs_result[rec_id]

            if group_no not in used_topics:
                topic_pim[index] = group_no
                used_topics.append(group_no)
                index += 1

    final_output = list(map(lambda x: Request.group["group"][x], topic_pim.values()))
    return final_output
