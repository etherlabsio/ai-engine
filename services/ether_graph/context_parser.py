import logging
import hashlib
import json as js
from typing import Tuple, List, Dict, Text, Union

from graph_definitions import (
    Context,
    ContextSession,
    TranscriptionSegment,
    TranscriptionProvider,
    Keyphrase,
    Entity,
    Mind,
    User,
    Source,
    SummaryGroup,
)
from service_definitions import (
    ContextRequest,
    SessionRequest,
    SegmentRequest,
    SummaryRequest,
)
from schema_generator import SchemaGenerator
from graph_handler import GraphHandler


logger = logging.getLogger(__name__)

KeyphraseObject = List[Keyphrase]
EntityObject = List[Entity]
TranscriptionSegmentObject = List[TranscriptionSegment]


class GraphPopulator(object):
    def __init__(self, dgraph_url=None):
        self.schema_generator = SchemaGenerator()
        self.gh = GraphHandler(
            dgraph_url=dgraph_url, schema_generator_object=self.schema_generator
        )
        self.parser = ContextSessionParser(graph_handler_object=self.gh)

    async def populate_context_info(self, req_data: ContextRequest):
        context_node = self.parser.parse_context_info(req_data=req_data)
        response = self.gh.mutate_info(Context.get_dict_from_object(context_node))

        return response

    async def populate_context_instance_segment_info(self, req_data: SessionRequest):
        context_node = self.parser.parse_context_instance_segment_info(
            req_data=req_data
        )
        response = self.gh.mutate_info(Context.get_dict_from_object(context_node))

        return response

    async def populate_summary_info(self, req_data: SummaryRequest):
        instance_node = self.parser.parse_summary_info(req_data=req_data)
        response = self.gh.mutate_info(
            ContextSession.get_dict_from_object(instance_node)
        )

        return response

    async def perform_query(self, query: str, variables: Dict):
        response = self.gh.perform_queries(query_text=query, variables=variables)

        return response

    async def set_schema(self):
        self.gh.set_schema()

    async def close_client(self):
        logger.info("Closing dgraph client connection...")
        await self.gh.close_client()


class ContextSessionParser(object):
    """
    Parse meeting events and send nodes only as JSON objects. Relations are attached before populating to graph
    """

    def __init__(self, graph_handler_object):
        self.gh = graph_handler_object

    # For testing purposes
    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

    def read_json(self, json_file):
        with open(json_file) as f_:
            meeting = js.load(f_)
        return meeting

    def parse_context_info(self, req_data: ContextRequest):
        context_id = req_data.contextId
        instance_id = req_data.instanceId
        mind_id = req_data.mindId
        start_ts = req_data.at

        mind_node = Mind(mindId=mind_id)
        instance_node = ContextSession(instanceId=instance_id, startTime=start_ts)
        context_node = Context(contextId=context_id)

        mind_node = self.gh.query_transform_node(node_obj=mind_node)
        instance_node = self.gh.query_transform_node(node_obj=instance_node)
        context_node = self.gh.query_transform_node(node_obj=context_node)

        instance_node.associatedMind = mind_node
        context_node.hasMeeting = instance_node

        return context_node

    def parse_context_instance_segment_info(self, req_data: SessionRequest) -> Context:

        # return individual nodes for population using upsert operation
        context_obj = Context(contextId=req_data.contextId)
        instance_id = req_data.instanceId
        segment_object = req_data.segments
        segment_object_dict = SegmentRequest.get_dict_from_object(segment_object)

        segment_node, context_node = self._parse_segment_info(
            segment_object=segment_object_dict, context_object=context_obj
        )
        instance_node = ContextSession(instanceId=instance_id, hasSegment=segment_node)
        instance_node = self.gh.query_transform_node(instance_node)
        context_node.hasMeeting = instance_node

        return context_node

    def _parse_segment_info(
        self, segment_object: Dict, context_object: Context = None
    ) -> Tuple[TranscriptionSegmentObject, Context]:

        segment_node_list = []
        for i, segment in enumerate(segment_object):
            provider_node = self._parse_provider_info(segment)
            recoder_node = self._parse_recorder_info(segment)

            keyphrase_node_object = self._keyphrase_info(segment)
            entity_node_object = self._entity_info(segment)

            user_node, context_node = self._parse_user_info(
                segment, context_object=context_object, entity_object=entity_node_object
            )
            context_object = context_node

            segment_node = TranscriptionSegment(
                **segment,
                authoredBy=user_node,
                providedBy=provider_node,
                hasSource=recoder_node,
                hasKeywords=keyphrase_node_object,
                hasEntities=entity_node_object,
                belongsTo=None,
            )
            segment_node = self.gh.query_transform_node(segment_node)
            segment_node_list.append(segment_node)

        return segment_node_list, context_object

    def parse_summary_info(self, req_data: SummaryRequest) -> ContextSession:
        # return individual nodes for population using upsert operation
        instance_id = req_data.instanceId
        segment_object = req_data.segments
        segment_object_dict = SegmentRequest.get_dict_from_object(segment_object)

        summary_keyphrases = req_data.keyphrases
        summary_entities = req_data.entities

        segment_node = self._parse_summary_segment_info(
            segment_object=segment_object_dict,
            summary_keyphrases=summary_keyphrases,
            summary_entities=summary_entities,
        )
        instance_node = ContextSession(instanceId=instance_id, hasSegment=segment_node)
        instance_node = self.gh.query_transform_node(instance_node)

        return instance_node

    def _parse_summary_segment_info(
        self,
        segment_object: List[Dict],
        summary_keyphrases: KeyphraseObject,
        summary_entities: EntityObject,
    ) -> TranscriptionSegmentObject:

        segment_node_list = []
        group_id = segment_object[0].get("groupId")
        group_user_nodes = [self._get_user_node(segment) for segment in segment_object]

        # Set keyphrase attribute to Summary type
        for kw_node in summary_keyphrases:
            kw_node.attribute = "summaryKeywords"

        summary_node = SummaryGroup(groupId=group_id)
        summary_node.hasKeywords = summary_keyphrases
        summary_node.hasEntities = summary_entities
        summary_node.hasUser = group_user_nodes

        summary_node = self.gh.query_transform_node(summary_node)

        for i, segment in enumerate(segment_object):
            provider_node = self._parse_provider_info(segment)
            recoder_node = self._parse_recorder_info(segment)
            user_node = self._get_user_node(segment)

            user_summary_info = self._parse_group_user_info(
                user_node=user_node, group_user_nodes=group_user_nodes
            )
            segment_node = TranscriptionSegment(
                **segment,
                authoredBy=user_summary_info,
                providedBy=provider_node,
                hasSource=recoder_node,
                belongsTo=summary_node,
            )
            segment_node = self.gh.query_transform_node(segment_node)
            segment_node_list.append(segment_node)

        return segment_node_list

    def parse_topic_marker(self):
        pass

    def parse_action_marker(self):
        pass

    def parse_decision_marker(self):
        pass

    def _parse_provider_info(self, segment: Dict) -> TranscriptionProvider:
        provider_node = TranscriptionProvider.get_object_from_dict(segment)
        provider_node = self.gh.query_transform_node(provider_node)

        return provider_node

    def _parse_recorder_info(self, segment: Dict) -> Source:
        recoder_node = Source.get_object_from_dict(segment)
        recoder_node = self.gh.query_transform_node(recoder_node)

        return recoder_node

    def _get_user_node(self, segment: Dict):
        user_node = User.get_object_from_dict(segment)
        user_node = self.gh.query_transform_node(user_node)

        return user_node

    def _parse_user_info(
        self, segment: Dict, context_object: Context, entity_object: EntityObject
    ) -> User:
        user_node = self._get_user_node(segment)

        # Context-User association
        context_node = self._parse_user_context_info(
            context_node=context_object, user_node=user_node
        )

        # User-Entity association
        user_node = self._parse_user_entity_info(
            entity_object=entity_object, user_node=user_node
        )

        return user_node, context_node

    def _parse_user_context_info(
        self, context_node: Context, user_node: User
    ) -> Context:
        context_node = self.gh.query_transform_node(context_node)

        # Make connection to Context's uid to avoid redundancy
        context_node.hasMember = {
            "dgraph.type": user_node.dgraphType,
            "uid": user_node.uid,
            "attribute": user_node.attribute,
        }

        return context_node

    def _parse_user_entity_info(
        self, entity_object: EntityObject, user_node: User
    ) -> User:
        user_node.userEntities = entity_object

        return user_node

    def _parse_group_user_info(
        self, user_node: User, group_user_nodes: List[User]
    ) -> User:
        grouped_user_node_object = [
            grp_user_node
            for grp_user_node in group_user_nodes
            if grp_user_node.uid != user_node.uid
        ]

        # Make connection to Grouped user's uid to avoid redundancy
        user_node.groupedWith = grouped_user_node_object

        return user_node

    def _keyphrase_info(self, segment: Dict) -> KeyphraseObject:
        keyphrase_obj = segment["keyphrases"]
        keyphrase_node = Keyphrase.get_object_from_dict(keyphrase_obj)
        keyphrase_node = [
            self.gh.query_transform_node(kp_node) for kp_node in keyphrase_node
        ]

        return keyphrase_node

    def _entity_info(self, segment: Dict) -> EntityObject:
        entity_obj = segment["entities"]
        entity_node = Entity.get_object_from_dict(entity_obj)
        entity_node = [
            self.gh.query_transform_node(ent_node) for ent_node in entity_node
        ]

        return entity_node

    def _hash_sha_object(self, data: str) -> str:
        hash_object = hashlib.sha1(data.encode())
        hash_str = hash_object.hexdigest()
        return hash_str


# # For testing locally
# if __name__ == "__main__":
#     gh = GraphHandler("localhost:9080")
#     gp = ContextSessionParser(graph_handler_object=gh)
#
#     # req_data = gp.read_json(
#     #     "/Users/shashank/Workspace/Orgs/Ether/ai-engine/tests/ether_graph_service/data/meeting_test.json"
#     # )
#     # summary_session_req = gp.read_json(
#     #     "/Users/shashank/Workspace/Orgs/Ether/ai-engine/tests/ether_graph_service/data/summary_test.json"
#     # )
#     # context_req = ContextRequest.get_object_from_dict(req_data)
#     # session_req = SessionRequest.get_object_from_dict(req_data)
#     # summary_session_req = SummaryRequest.get_object_from_dict(summary_session_req)
#     # Execute one-by-one in sequence
#
#     gp.gh.set_schema()
#     # gp.parse_context_info(context_req)
#     # gp.parse_context_instance_segment_info(session_req)
#     # gp.parse_summary_info(summary_session_req)
