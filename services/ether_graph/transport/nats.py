import json
import logging
from timeit import default_timer as timer
import traceback

from ether_graph.service_definitions import (
    SessionRequest,
    ContextRequest,
    SummaryRequest,
    UserMembershipRequest,
)

logger = logging.getLogger(__name__)


class NATSTransport(object):
    def __init__(self, nats_manager, eg_service):
        self.nats_manager = nats_manager
        self.eg_service = eg_service

    async def subscribe_context(self):
        context_created_topic = "context.instance.created"
        logger.info(
            "Subscribing to context instance event",
            extra={"topic": context_created_topic},
        )
        await self.nats_manager.subscribe(
            context_created_topic, handler=self.context_created_handler, queued=True,
        )

    async def context_created_handler(self, msg):
        msg_data = json.loads(msg.data)
        context_id = msg_data["contextId"]
        instance_id = msg_data["instanceId"]
        logger.info(
            "instance created",
            extra={"contextId": context_id, "instanceId": instance_id},
        )
        await self.subscribe_context_events()
        logger.info(
            "topics subscribed",
            extra={"topics": list(self.nats_manager.subscriptions.keys())},
        )

    async def subscribe_context_events(self):
        await self.nats_manager.subscribe(
            topic="context.instance.started",
            handler=self.context_start_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance.ended",
            handler=self.context_end_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="ether_graph_service.add_segments",
            handler=self.populate_segment_data,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="ether_graph_service.populate_summary",
            handler=self.populate_summary_data,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="ether_graph_service.perform_query",
            handler=self.perform_query,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.user_added", handler=self.add_user_membership, queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.user_removed",
            handler=self.delete_user_membership,
            queued=True,
        )

    async def unsubscribe_lifecycle_events(self):
        await self.nats_manager.unsubscribe(topic="context.instance.started")
        await self.nats_manager.unsubscribe(topic="context.instance.ended")
        await self.nats_manager.unsubscribe(topic="ether_graph_service.add_segments")
        await self.nats_manager.unsubscribe(
            topic="ether_graph_service.populate_summary"
        )
        await self.nats_manager.unsubscribe(topic="ether_graph_service.perform_query")
        await self.nats_manager.unsubscribe(topic="context.user_added")
        await self.nats_manager.unsubscribe(topic="context.user_removed")

    # NATS context handlers

    async def context_start_handler(self, msg):
        request = json.loads(msg.data)
        try:
            req_data = ContextRequest.get_object_from_dict(request)
            resp = await self.eg_service.populate_context_info(req_data=req_data)

            logger.info(
                "Populated context and instance info to dgraph",
                extra={"response": resp.uids, "latency": resp.latency, "success": True},
            )
        except Exception as e:
            logger.error("Error adding context info to dgraph", extra={"err": e})
            raise

    async def context_end_handler(self, msg):
        pass

    # Topic Handler functions

    async def populate_segment_data(self, msg):
        request = json.loads(msg.data)

        try:
            req_data = SessionRequest.get_object_from_dict(request)
            resp = await self.eg_service.populate_context_instance_segment_info(
                req_data=req_data
            )

            logger.info(
                "Populated segment info to dgraph",
                extra={"response": resp.uids, "latency": resp.latency, "success": True},
            )
        except Exception as e:
            logger.error("Error adding segment to dgraph", extra={"err": e})
            print(traceback.print_exc())
            raise

    async def populate_summary_data(self, msg):
        request = json.loads(msg.data)

        try:
            req_data = SummaryRequest.get_object_from_dict(request)
            resp = await self.eg_service.populate_summary_info(req_data=req_data)

            logger.info(
                "Populated summary info to dgraph",
                extra={"response": resp.uids, "latency": resp.latency, "success": True},
            )
        except Exception as e:
            logger.error("Error adding summary info to dgraph", extra={"err": e})
            print(traceback.print_exc())
            raise

    async def perform_query(self, msg):
        request = json.loads(msg.data)
        query_text = request["query"]
        variables = request["variables"]

        try:
            resp = await self.eg_service.perform_query(query_text, variables)

            logger.info("Successfully queried dgraph", extra={"success": True})
            await self.nats_manager.conn.publish(msg.reply, json.dumps(resp).encode())
        except Exception as e:
            logger.error("Error querying dgraph", extra={"err": e})
            raise

    async def add_user_membership(self, msg):
        request = json.loads(msg.data)

        try:
            membership_req = UserMembershipRequest.get_object_from_dict(request)
            resp = await self.eg_service.update_user_membership(
                req_data=membership_req, status="added"
            )

            logger.info(
                "Added user-context membership info",
                extra={"response": resp.uids, "latency": resp.latency},
            )
        except Exception as e:
            logger.error("Error updating user-context membership", extra={"err": e})
            print(traceback.print_exc())
            raise

    async def delete_user_membership(self, msg):
        request = json.loads(msg.data)

        try:
            membership_req = UserMembershipRequest.get_object_from_dict(request)
            resp = await self.eg_service.update_user_membership(
                req_data=membership_req, status="deleted"
            )

            logger.info(
                "Deleted user-context membership info",
                extra={"response": resp.uids, "latency": resp.latency},
            )
        except Exception as e:
            logger.error("Error deleting user-context membership", extra={"err": e})
            print(traceback.print_exc())
            raise
