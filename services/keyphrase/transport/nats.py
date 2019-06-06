import json
import logging
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


class NATSTransport(object):
    def __init__(self, nats_manager, keyphrase_service):
        self.nats_manager = nats_manager
        self.keyphrase_service = keyphrase_service

    async def subscribe_context(self):
        context_created_topic = "context.instance.created"
        logger.info(
            "Subscribing to context instance event",
            extra={"topic": context_created_topic},
        )
        await self.nats_manager.subscribe(
            context_created_topic, handler=self.context_created_handler, queued=True
        )

    async def context_created_handler(self, msg):
        msg_data = json.loads(msg.data)
        context_id = msg_data["contextId"]
        context_instance_id = msg_data["instanceId"]
        logger.info(
            "instance created",
            extra={"contextId": context_id, "instanceId": context_instance_id},
        )
        await self.subscribe_context_events(context_instance_id)
        logger.info(
            "topics subscribed", extra={"topics": self.nats_manager.subscriptions}
        )

        self.keyphrase_service.initialize_meeting_graph(
            context_id=context_id, context_instance_id=context_instance_id
        )
        logger.info("Initialized word graph")

    async def subscribe_context_events(self, instance_id):
        await self.nats_manager.subscribe(
            topic="context.instance." + instance_id + ".started",
            handler=self.context_start_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance." + instance_id + ".context_changed",
            handler=self.context_change_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance." + instance_id + ".ended",
            handler=self.context_end_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="keyphrase_service." + instance_id + ".extract_keyphrases",
            handler=self.extract_segment_keyphrases,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="keyphrase_service."
            + instance_id
            + ".keyphrases_for_context_instance",
            handler=self.extract_instance_keyphrases,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance." + instance_id + ".add_segments",
            handler=self.populate_graph,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="keyphrase_service."
            + instance_id
            + ".extract_keyphrases_with_offset",
            handler=self.chapter_offset_handler,
            queued=True,
        )

    async def unsubscribe_lifecycle_events(self, instance_id):
        await self.nats_manager.unsubscribe(
            topic="context.instance." + instance_id + ".started"
        )
        await self.nats_manager.unsubscribe(
            topic="context.instance." + instance_id + ".context_changed"
        )
        await self.nats_manager.unsubscribe(
            topic="context.instance." + instance_id + ".ended"
        )
        await self.nats_manager.unsubscribe(
            topic="keyphrase_service." + instance_id + ".extract_keyphrases"
        )
        await self.nats_manager.unsubscribe(
            topic="keyphrase_service."
            + instance_id
            + ".keyphrases_for_context_instance"
        )
        await self.nats_manager.unsubscribe(
            topic="context.instance." + instance_id + ".add_segments"
        )
        await self.nats_manager.unsubscribe(
            topic="keyphrase_service." + instance_id + ".extract_keyphrases_with_offset"
        )

    # NATS context handlers

    async def context_start_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data["state"] == "started":
            logger.info("Instance started")
        pass

    async def context_change_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data["state"] == "context_changed":
            # Update contextId when change is notified
            context_id = msg_data["contextId"]
            return context_id
        pass

    async def context_end_handler(self, msg):
        msg_data = json.loads(msg.data)
        instance_id = msg_data["instanceId"]
        # Close, drain and unsubscribe connections to keyphrase topics
        await self.unsubscribe_lifecycle_events(instance_id)
        # Reset graph
        await self.reset_keyphrases(msg)

    # Topic Handler functions

    async def populate_graph(self, msg):
        request = json.loads(msg.data)

        self.keyphrase_service.populate_word_graph(request)

    async def extract_segment_keyphrases(self, msg):
        start = timer()
        request = json.loads(msg.data)

        output = self.keyphrase_service.get_keyphrases(request)
        end = timer()

        if len(request["segments"]) > 1:
            logger.info(
                "Publishing chapter keyphrases",
                extra={
                    "chapterKeyphraseList": output,
                    "instanceId": request["instanceId"],
                    "numOfSegments": len(request["segments"]),
                    "responseTime": end - start,
                    "requestReceived": request,
                },
            )
        else:
            logger.info(
                "Publishing PIM keyphrases",
                extra={
                    "pimKeyphraseList": output,
                    "instanceId": request["instanceId"],
                    "numOfSegments": len(request["segments"]),
                    "responseTime": end - start,
                    "requestReceived": request,
                },
            )
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def extract_instance_keyphrases(self, msg):
        start = timer()
        request = json.loads(msg.data)
        output = self.keyphrase_service.get_instance_keyphrases(request)
        end = timer()

        logger.info(
            "Publishing instance keyphrases",
            extra={
                "instanceList": output,
                "instanceId": request["instanceId"],
                "numOfSegments": len(request["segments"]),
                "responseTime": end - start,
                "requestReceived": request,
            },
        )
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def chapter_offset_handler(self, msg):
        start = timer()
        request = json.loads(msg.data)
        output = self.keyphrase_service.get_chapter_offset_keyphrases(request)
        end = timer()

        logger.info(
            "Publishing chapter keyphrases with offset",
            extra={
                "chapterOffsetList": output,
                "instanceId": request["instanceId"],
                "numOfSegments": len(request["segments"]),
                "responseTime": end - start,
                "requestReceived": request,
            },
        )

        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def reset_keyphrases(self, msg):
        request = json.loads(msg.data)
        logger.info("Resetting keyphrases graph")
        output = self.keyphrase_service.reset_keyphrase_graph(request)
        await self.nats_manager.conn.publish(msg, json.dumps(output).encode())
