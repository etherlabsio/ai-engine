import json
import logging
from timeit import default_timer as timer
import traceback

logger = logging.getLogger(__name__)


class NATSTransport(object):
    def __init__(self, nats_manager, watcher_service, meeting_service):
        self.nats_manager = nats_manager
        self.watcher_service = watcher_service
        self.meeting_service = meeting_service

    async def subscribe_context(self):
        context_created_topic = "context.instance.created"
        logger.info(
            "Subscribing to context instance event",
            extra={"topic": context_created_topic},
        )
        await self.nats_manager.subscribe(
            context_created_topic,
            handler=self.context_created_handler,
            queued=True,
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
            topic="recommendation.service.get_watchers",
            handler=self.get_watchers,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="recommendation.service.get_meetings",
            handler=self.get_meetings,
            queued=True,
        )

    async def unsubscribe_lifecycle_events(self):
        await self.nats_manager.unsubscribe(topic="context.instance.started")
        await self.nats_manager.unsubscribe(
            topic="recommendation.service.get_watchers"
        )
        await self.nats_manager.unsubscribe(
            topic="recommendation.service.get_meetings"
        )

    # NATS context handlers

    async def context_start_handler(self, msg):
        msg_data = json.loads(msg.data)
        try:
            self.watcher_service.featurize_reference_users(req_data=msg_data)
        except Exception:
            raise

    # Topic Handler functions

    async def get_watchers(self, msg):
        msg_data = json.loads(msg.data)
        keyphrase_list = msg_data["keyphrases"]

        try:
            rec_users, related_words = self.watcher_service.get_recommended_watchers(
                kw_list=keyphrase_list
            )
            watcher_response = {"users": rec_users, "words": related_words}

            await self.nats_manager.conn.publish(
                msg.reply, json.dumps(watcher_response).encode()
            )
        except Exception:
            raise

    async def get_meetings(self, msg):
        msg_data = json.loads(msg.data)
        keyphrase_list = msg_data["keyphrases"]

        try:
            self.meeting_service.recommend_meetings(kw_list=keyphrase_list)
        except Exception:
            raise

        pass
