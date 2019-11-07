import json as js
import logging

logger = logging.getLogger(__name__)


class ContextSessionParser(object):
    def __init__(self):
        self.context_label = {"attribute": "contextId"}
        self.instance_label = {"attribute": "instanceId"}
        self.context_instance_rel = {"relation": "hasMeeting"}

        self.instance_segment_rel = {"relation": "hasSegment"}

        self.segment_label = {"attribute": "segmentId"}
        self.user_label = {"attribute": "userId"}
        self.transcriber_label = {"attribute": "segmentProvider"}

        self.segment_user_rel = {"relation": "authoredBy"}
        self.segment_transcriber_rel = {"relation": "providedBy"}

        self.recording_label = {"attribute": "sourceId", "type": "recording"}
        self.segment_recording_rel = {"relation": "hasSource"}

        self.pim_keyphrase_label = {"attribute": "importantKeywords"}
        self.keyphrase_label = {"attribute": "segmentKeywords"}
        self.segment_keyphrase_rel = {"relation": "hasKeywords"}

        self.mind_label = {"attribute": "mindId"}
        self.context_mind_rel = {"relation": "associatedMind"}

        self.list_type_edges = ["hasMeeting", "hasSegment", "hasKeywords"]
        self.keyword_node = ["importKeywords"]
        self.schema_type = {
            "contextId": "Context",
            "instanceId": "ContextSession",
            "segmentId": "TranscriptionSegment",
            "userId": "User",
            "mindId": "Mind",
            "workspaceId": "Workspace",
            "channelId": "Channel",
            "sourceId": "Source",
            "segmentProvider": "TranscriptionProvider",
            "importantKeywords": "Keyphrase",
        }

    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

    def read_json(self, json_file):
        with open(json_file) as f_:
            meeting = js.load(f_)
        return meeting

    def meeting_definition(self):
        meeting_def = """
        type Context {
            xid: string
            attribute: string
            hasMeeting: [Instance]
            associatedMind: Mind
        }
                
        type Mind {
            xid: string
            attribute: string
        }
        
        type ContextSession {
            xid: string
            attribute: string
            hasSegment: [Segment]
        }
        
        type TranscriptionSegment {
            xid: string
            attribute: string
            text: string
            analyzedText: string
            confidence: float
            language: string
            startTime: datetime
            endTime: datetime
            duration: int
            authoredBy: [User]
            hasKeywords: [Keyword]
            hasSource: [Source]
            providedBy: [Provider]    
        }
        
        type User {
            xid: string
            attribute: string
        }
        
        type Source {
            xid: string
            attribute: string
            type: string
        }
        
        type TranscriptionProvider {
            name: string
            attribute: string
        }
        
        type Keyphrase {
            value: string
            attribute: string
            important: bool
            type: string
            origin: string
        }
        """

        return meeting_def

    def parse_context_info(self, req_data, **kwargs):
        instance_node = self.parse_instance_segment_info(req_data=req_data, **kwargs)
        self._context_instance_info(req_data=req_data, instance_node=instance_node)

    def parse_instance_segment_info(self, req_data, **kwargs):
        instance_id = req_data["instanceId"]
        segment_node = self.parse_segment_info(req_data=req_data, **kwargs)
        instance_node = self._instance_segment_info(
            instance_id=instance_id, segment_node=segment_node
        )

        return instance_node

    def parse_segment_info(self, req_data, **kwargs):
        external_segment_attr = kwargs.get("segment_attr", dict())
        external_user_attr = kwargs.get("user_attr", dict())
        external_provider_attr = kwargs.get("provider_attr", dict())
        external_recorder_attr = kwargs.get("recorder_attr", dict())

        segment_attr_list = [
            "originalText",
            "confidence",
            "startTime",
            "endTime",
            "duration",
            "languageCode",
            "transcriptId",
            "createdAt",
            "tenantId",
        ]

        segment_object = req_data["segments"]
        segment_node = {}

        for i, segment in enumerate(segment_object):
            user_node = self._user_info(segment=segment, user_attr=external_user_attr)
            provider_node = self._provider_info(
                segment=segment, provider_attr=external_provider_attr
            )
            recoder_node = self._recorder_info(
                segment=segment, recorder_attr=external_recorder_attr
            )

            segment_attr_dict = {k: segment[k] for k in segment_attr_list}
            segment_attr_dict.update(external_segment_attr)

            # Rename `originalText` field
            segment_attr_dict["text"] = segment_attr_dict.pop("originalText")

            segment_node = self._segment_info(
                segment=segment,
                segment_attr=segment_attr_dict,
                user_node=user_node,
                provider_node=provider_node,
                recorder_node=recoder_node,
            )

        return segment_node

    def _context_instance_info(self, req_data, **kwargs):
        context_attr = kwargs.get("context_attr", dict())
        mind_attr = kwargs.get("mind_attr", dict())
        instance_node = kwargs.get("instance_node")

        context_id = req_data["contextId"]
        mind_id = req_data["mindId"]

        # Update context info
        self.context_label.update(context_attr)
        context_node = {
            "uid": "_:" + context_id,
            "xid": context_id,
            "dgraph.type": self.schema_type[self.context_label.get("attribute")],
            **self.context_label,
        }

        # Update mind info
        self.mind_label.update(mind_attr)
        mind_node = {
            "uid": "_:" + mind_id,
            "xid": mind_id,
            "dgraph.type": self.schema_type[self.mind_label.get("attribute")],
            **self.mind_label,
        }

        # Define Context relations
        context_relations = {
            self.context_mind_rel.get("relation"): mind_node,
            self.context_instance_rel.get("relation"): instance_node,
        }

        # Join Context node and relations
        context_node.update(context_relations)

        self.to_json(context_node, "context_info")
        return context_node

    def _instance_segment_info(self, instance_id, **kwargs):
        instance_attr = kwargs.get("instance_attr", dict())
        segment_node = kwargs.get("segment_node")

        # Update instance info
        self.instance_label.update(instance_attr)
        instance_node = {
            "uid": "_:" + instance_id,
            "xid": instance_id,
            "dgraph.type": self.schema_type[self.instance_label.get("attribute")],
            **self.instance_label,
        }

        # Define Instance relations
        instance_relations = {self.instance_segment_rel.get("relation"): segment_node}

        # Join Instance node and relations
        instance_node.update(instance_relations)

        return instance_node

    def _segment_info(self, segment, **kwargs):
        segment_attr = kwargs.get("segment_attr")
        user_node = kwargs.get("user_node")
        provider_node = kwargs.get("provider_node")
        recorder_node = kwargs.get("recorder_node")

        # Update segment info
        self.segment_label.update(segment_attr)
        segment_node = {
            "uid": "_:" + segment["id"],
            "xid": segment["id"],
            "dgraph.type": self.schema_type[self.segment_label.get("attribute")],
            **self.segment_label,
        }

        # Define Segment relations
        segment_relations = {
            self.segment_user_rel.get("relation"): user_node,
            self.segment_transcriber_rel.get("relation"): provider_node,
            self.segment_recording_rel.get("relation"): recorder_node,
        }

        # Join Segment node and relations
        segment_node.update(segment_relations)

        return segment_node

    def _user_info(self, segment, **kwargs):
        user_attr = kwargs.get("user_attr", dict())
        user_id = segment["spokenBy"]

        self.user_label.update(user_attr)
        user_node = {
            "uid": "_:" + user_id,
            "xid": user_id,
            "dgraph.type": self.schema_type[self.user_label.get("attribute")],
            **self.user_label,
        }

        return user_node

    def _provider_info(self, segment, **kwargs):
        provider_attr = kwargs.get("provider_attr", dict())
        provider_name = segment["transcriber"]

        self.transcriber_label.update(provider_attr)
        provider_node = {
            "name": provider_name,
            "dgraph.type": self.schema_type[self.transcriber_label.get("attribute")],
            **self.transcriber_label,
        }

        return provider_node

    def _recorder_info(self, segment, **kwargs):
        recorder_attr = kwargs.get("recorder_attr", dict())
        recorder_id = segment["recordingId"]

        self.recording_label.update(recorder_attr)
        recorder_node = {
            "uid": "_:" + recorder_id,
            "xid": recorder_id,
            "dgraph.type": self.schema_type[self.recording_label.get("attribute")],
            **self.recording_label,
        }

        return recorder_node


if __name__ == "__main__":
    parser = ContextSessionParser()
    test_json = parser.read_json("meeting_test.json")

    parser.parse_context_info(req_data=test_json)
