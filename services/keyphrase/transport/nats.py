import json
import logging

logger = logging.getLogger(__name__)


class NATSTransport(object):
    def __init__(self, nats_manager, keyphrase_service):
        self.nats_manager = nats_manager
        self.keyphrase_service = keyphrase_service

    async def subscribe_context(self):
        context_created_topic = "context.instance.created"
        logger.info("Subscribing to context instance event",
                    topic=context_created_topic)
        await self.nats_manager.subscribe(context_created_topic,
                                          handler=self.context_created_handler,
                                          queued=True)

    async def context_created_handler(self, msg):
        msg_data = json.loads(msg.data)
        context_id = msg_data['contextId']
        context_instance_id = msg_data['id']
        logger.debug("instance created",
                     cid=context_id,
                     ciid=context_instance_id)
        await self.subscribe_context_events(context_instance_id)

    async def subscribe_context_events(self, instance_id):
        await self.nats_manager.subscribe(topic="context.instance." +
                                          instance_id + ".started",
                                          handler=self.context_start_handler,
                                          queued=True)
        await self.nats_manager.subscribe(topic="context.instance." +
                                          instance_id + ".context_changed",
                                          handler=self.context_change_handler,
                                          queued=True)
        await self.nats_manager.subscribe(topic="context.instance." +
                                          instance_id + ".ended",
                                          handler=self.context_end_handler,
                                          queued=True)
        await self.nats_manager.subscribe(
            topic="keyphrase_service." + instance_id + ".extract_keyphrases",
            handler=self.extract_segment_keyphrases,
            queued=True)
        await self.nats_manager.subscribe(
            topic="keyphrase_service." + instance_id +
            ".keyphrases_for_context_instance",
            handler=self.extract_instance_keyphrases,
            queued=True)
        await self.nats_manager.subscribe(topic="context.instance." +
                                          instance_id + ".add_segments",
                                          handler=self.populate_graph,
                                          queued=True)

    async def unsubscribe_lifecycle_events(self, instance_id):
        await self.nats_manager.unsubscribe(topic="context.instance." +
                                            instance_id + ".started")
        await self.nats_manager.unsubscribe(topic="context.instance." +
                                            instance_id + ".context_changed")
        await self.nats_manager.unsubscribe(topic="context.instance." +
                                            instance_id + ".ended")
        await self.nats_manager.unsubscribe(topic="keyphrase_service." +
                                            instance_id + ".extract_keyphrases"
                                            )
        await self.nats_manager.unsubscribe(topic="keyphrase_service." +
                                            instance_id +
                                            ".keyphrases_for_context_instance")
        await self.nats_manager.unsubscribe(topic="context.instance." +
                                            instance_id + ".add_segments")

    # NATS context handlers

    async def context_start_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'started':
            logger.info("Instance started")
        pass

    async def context_change_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'context_changed':
            # Update contextId when change is notified
            context_id = msg_data['contextId']
        pass

    async def context_end_handler(self, msg):
        msg_data = json.loads(msg.data)
        instance_id = msg_data['id']
        # Close, drain and unsubscribe connections to keyphrase topics
        await self.unsubscribe_lifecycle_events(instance_id)
        # Reset graph
        await self.reset_keyphrases(msg)

    # Topic Handler functions

    async def populate_graph(self, msg):
        request = json.loads(msg.data)

        logger.info("Populating word graph ...")
        self.keyphrase_service.populate_word_graph(request)

    async def extract_segment_keyphrases(self, msg):
        request = json.loads(msg.data)

        output = self.keyphrase_service.get_keyphrases(request)
        logger.debug("Output : {}".format(output))
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def extract_instance_keyphrases(self, msg):
        request = json.loads(msg.data)
        logger.info("Publishing Instance Keyphrases")
        output = self.keyphrase_service.get_instance_keyphrases(request)
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def reset_keyphrases(self, msg):
        request = json.loads(msg.data)
        logger.info("Resetting keyphrases graph ...")
        output = self.keyphrase_service.reset_keyphrase_graph(request)
        await self.nats_manager.conn.publish(msg, json.dumps(output).encode())
