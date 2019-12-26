import logging
import hashlib

logger = logging.getLogger(__name__)


class ContextSessionParser(object):
    """
    Parse meeting events and send nodes only as JSON objects. Relations are attached before populating to graph
    """

    def __init__(self):
        self.context_label = {"attribute": "contextId"}
        self.instance_label = {"attribute": "instanceId"}
        self.segment_label = {"attribute": "segmentId"}
        self.user_label = {"attribute": "userId"}
        self.transcriber_label = {"attribute": "segmentProvider"}
        self.recording_label = {"attribute": "sourceId", "type": "recording"}
        self.pim_keyphrase_label = {"attribute": "importantKeywords"}
        self.keyphrase_label = {"attribute": "segmentKeywords"}
        self.mind_label = {"attribute": "mindId"}

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
            "segmentKeywords": "Keyphrase",
        }

    def parse_context_info(self, req_data, **kwargs):
        context_id = req_data["contextId"]
        mind_id = req_data.get("mindId", "NA")
        instance_id = req_data["instanceId"]

        context_node, instance_node, mind_node = self._context_instance_info(
            context_id=context_id, mind_id=mind_id, instance_id=instance_id, **kwargs,
        )
        return context_node, instance_node, mind_node

    def parse_instance_segment_info(self, req_data, **kwargs):
        instance_id = req_data["instanceId"]
        segment_object = req_data["segments"]

        (
            segment_node,
            user_node,
            provider_node,
            recorder_node,
        ) = self.parse_segment_info(segment_object=segment_object, **kwargs)
        instance_node = self._instance_info(instance_id=instance_id)

        # return individual nodes for population using upsert operation
        return instance_node, segment_node

    def parse_segment_info(self, segment_object, **kwargs):
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
        ]

        segment_node = {}
        user_node = {}
        provider_node = {}
        recoder_node = {}

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
            extra_segment_attributes = segment.get("attributes", None)
            if extra_segment_attributes is not None:
                segment_attr_dict.update(extra_segment_attributes)

            # Rename `originalText` field
            segment_attr_dict["text"] = segment_attr_dict.pop("originalText")

            segment_node = self._segment_info(
                segment=segment, segment_attr=segment_attr_dict
            )

        return segment_node, user_node, provider_node, recoder_node

    def parse_keyphrase_info(self, segment_object, **kwargs):
        keyphrase_node = {}
        ext_keyphrase_attr = kwargs.get("keyphrase_attr", dict())

        for i, segment in enumerate(segment_object):
            keyphrase_node = self._keyphrase_info(
                segment=segment, keyphrase_attr=ext_keyphrase_attr
            )

        return keyphrase_node

    def parse_topic_marker(self):
        pass

    def parse_action_marker(self):
        pass

    def parse_decision_marker(self):
        pass

    def _context_instance_info(self, context_id, mind_id, instance_id, **kwargs):
        context_attr = kwargs.get("context_attr", dict())
        instance_attr = kwargs.get("instance_attr", dict())
        mind_attr = kwargs.get("mind_attr", dict())

        # Update context info
        self.context_label.update(context_attr)
        context_node = {
            "dgraph.type": self.schema_type[self.context_label.get("attribute")],
            "uid": "_:" + context_id,
            "xid": context_id,
            **self.context_label,
        }

        # Update mind info
        self.mind_label.update(mind_attr)
        mind_node = {
            "dgraph.type": self.schema_type[self.mind_label.get("attribute")],
            "uid": "_:" + mind_id,
            "xid": mind_id,
            **self.mind_label,
        }

        # Update instance info
        self.instance_label.update(instance_attr)
        instance_node = {
            "dgraph.type": self.schema_type[self.instance_label.get("attribute")],
            "uid": "_:" + instance_id,
            "xid": instance_id,
            **self.instance_label,
        }

        return context_node, instance_node, mind_node

    def _instance_info(self, instance_id, **kwargs):
        instance_attr = kwargs.get("instance_attr", dict())

        # Update instance info
        self.instance_label.update(instance_attr)
        instance_node = {
            "dgraph.type": self.schema_type[self.instance_label.get("attribute")],
            "uid": "_:" + instance_id,
            "xid": instance_id,
            **self.instance_label,
        }

        return instance_node

    def _segment_info(self, segment, **kwargs):
        segment_attr = kwargs.get("segment_attr")

        # Update segment info
        self.segment_label.update(segment_attr)
        segment_node = {
            "dgraph.type": self.schema_type[self.segment_label.get("attribute")],
            "uid": "_:" + segment["id"],
            "xid": segment["id"],
            **self.segment_label,
        }

        return segment_node

    def _user_info(self, segment, **kwargs):
        user_attr = kwargs.get("user_attr", dict())
        user_id = segment["spokenBy"]

        self.user_label.update(user_attr)
        user_node = {
            "dgraph.type": self.schema_type[self.user_label.get("attribute")],
            "uid": "_:" + user_id,
            "xid": user_id,
            **self.user_label,
        }

        return user_node

    def _provider_info(self, segment, **kwargs):
        provider_attr = kwargs.get("provider_attr", dict())
        provider_name = segment["transcriber"]

        self.transcriber_label.update(provider_attr)
        provider_node = {
            "dgraph.type": self.schema_type[self.transcriber_label.get("attribute")],
            "uid": "_:" + provider_name,
            "xid": provider_name,
            "name": provider_name,
            **self.transcriber_label,
        }

        return provider_node

    def _recorder_info(self, segment, **kwargs):
        recorder_attr = kwargs.get("recorder_attr", dict())
        recorder_id = segment["recordingId"]

        self.recording_label.update(recorder_attr)
        recorder_node = {
            "dgraph.type": self.schema_type[self.recording_label.get("attribute")],
            "uid": "_:" + recorder_id,
            "xid": recorder_id,
            **self.recording_label,
        }

        return recorder_node

    def _keyphrase_info(self, segment, **kwargs):
        keyphrase_attributes = segment["keyphrases"]
        segment_id = segment["id"]
        keyphrase_attr = kwargs.get("keyphrase_attr", dict())

        keyphrase_attributes.update(keyphrase_attr)
        self.keyphrase_label.update(keyphrase_attributes)
        keyphrase_node = {
            "dgraph.type": self.schema_type[self.keyphrase_label.get("attribute")],
            "uid": "_:" + self._hash_sha_object(data=segment_id),
            "xid": self._hash_sha_object(data=segment_id),
            **self.keyphrase_label,
        }

        return keyphrase_node

    def _hash_sha_object(self, data: str) -> str:
        hash_object = hashlib.sha1(data.encode())
        hash_str = hash_object.hexdigest()
        return hash_str
