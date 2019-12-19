import json as js
import requests

def post_to_slack( instance_id, seg, web_hook_url="https://hooks.slack.com/services/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"):

     service_name = "segment_analyser"
     msg_text = "Filtered Groups for meeting: {} \n\n segments: {}".format(
         instance_id, seg
     )

     msg_format = "[{}]: {}".format(
         service_name, msg_text
     )

     slack_payload = {"text": msg_format}
     result = requests.post(
         url=web_hook_url, data=js.dumps(slack_payload).encode()
        )
     return result


def post_to_slack_topic( instance_id, potential_topic, web_hook_url="https://hooks.slack.com/services/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"):

     service_name = "segment_analyser"
     msg_text = "Potential Topics for meeting: {} \n\n Topics: {}".format(
         instance_id, potential_topic
     )

     msg_format = "[{}]: {}".format(
         service_name, msg_text
     )

     slack_payload = {"text": msg_format}
     result = requests.post(
         url=web_hook_url, data=js.dumps(slack_payload).encode()
        )
     return result
