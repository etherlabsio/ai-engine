import json
import logging
from timeit import default_timer as timer
import traceback

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
            topic="context.instance.add_segments",
            handler=self.populate_segments,
            queued=True,
        )

    async def unsubscribe_lifecycle_events(self):
        await self.nats_manager.unsubscribe(topic="context.instance.started")
        await self.nats_manager.unsubscribe(topic="context.instance.ended")
        await self.nats_manager.unsubscribe(topic="context.instance.add_segments")

    # NATS context handlers

    async def context_start_handler(self, msg):
        msg_data = json.loads(msg.data)
        try:
            self.eg_service.populate_context_info(req_data=msg_data)
        except Exception:
            raise

    async def context_end_handler(self, msg):
        pass

    # Topic Handler functions

    async def populate_segments(self, msg):
        start = timer()
        request = json.loads(msg.data)

        try:
            self.eg_service.populate_segment_info(req_data=request)
        except Exception as e:
            end = timer()
            logger.error(
                "Error adding segment to dgraph",
                extra={
                    "err": e,
                    "trace": traceback.print_exc(),
                    "responseTime": end - start,
                },
            )
            raise
