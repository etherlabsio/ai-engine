import json


class NATSTransport(object):
    def __init__(self, nats_manager, keyphrase_object, logger=None):
        self.nats_manager = nats_manager
        self.kpe = keyphrase_object
        self.log = logger

        self.url = self.nats_manager.url
        self.loop = self.nats_manager.loop
        self.queueName = self.nats_manager.queueName
        self.nc = self.nats_manager.conn

        self.context_id = []
        self.context_instance_id = []
        self.context_created_topic = "context.instance.created"

    # NATS handler functions

    async def subscribe_context(self):
        self.log.info("Subscribing to context instance event", topic=self.context_created_topic)
        await self.nats_manager.subscribe(self.context_created_topic, handler=self.context_created_handler, queued=True)

    async def context_created_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'created':
            self.context_id = msg_data['contextId']
            self.context_instance_id = msg_data['id']
            self.log.debug("instance created", cid=self.context_id, ciid=self.context_instance_id)

            await self.subscribe_context_events()

    async def subscribe_context_events(self):
        await self.nats_manager.subscribe(topic=self._reformat_topic("context.instance.*.started"),
                                          handler=self.context_start_handler, queued=True)
        await self.nats_manager.subscribe(topic=self._reformat_topic("context.instance.*.context_changed"),
                                          handler=self.context_change_handler, queued=True)
        await self.nats_manager.subscribe(topic=self._reformat_topic("context.instance.*.ended"),
                                          handler=self.context_end_handler, queued=True)
        await self.nats_manager.subscribe(topic=self._reformat_topic("keyphrase_service.*.extract_keyphrases"),
                                          handler=self.extract_segment_keyphrases, queued=True)
        await self.nats_manager.subscribe(
            topic=self._reformat_topic("keyphrase_service.*.keyphrases_for_context_instance"),
            handler=self.extract_instance_keyphrases, queued=True)
        await self.nats_manager.subscribe(topic=self._reformat_topic("context.instance.*.add_segments"),
                                          handler=self.populate_graph, queued=True)

    async def unsubscribe_lifecycle_events(self, instance_id):
        await self.nats_manager.unsubscribe(topic=self._reformat_topic("context.instance.*.started",
                                                                       instance_id))
        await self.nats_manager.unsubscribe(topic=self._reformat_topic("context.instance.*.context_changed",
                                                                       instance_id))
        await self.nats_manager.unsubscribe(topic=self._reformat_topic("context.instance.*.ended",
                                                                       instance_id))
        await self.nats_manager.unsubscribe(topic=self._reformat_topic("keyphrase_service.*.extract_keyphrases",
                                                                       instance_id))
        await self.nats_manager.unsubscribe(
            topic=self._reformat_topic("keyphrase_service.*.keyphrases_for_context_instance",
                                       instance_id))
        await self.nats_manager.unsubscribe(topic=self._reformat_topic("context.instance.*.add_segments",
                                                                       instance_id))

    # NATS context handlers

    async def context_start_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'started':
            self.log.info("Start keyphrase subscriptions")
        pass

    async def context_change_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'context_changed':
            # Update contextId when change is notified
            self.context_id = msg_data['contextId']
        pass

    async def context_end_handler(self, msg):
        msg_data = json.loads(msg.data)
        instance_id = msg_data['id']
        if msg_data['state'] == 'ended':
            # Close, drain and unsubscribe connections to keyphrase topics
            await self.unsubscribe_lifecycle_events(instance_id=instance_id)
            # Reset graph
            await self.reset_keyphrases(msg)
        pass

    # Topic Handler functions

    async def populate_graph(self, msg):
        request = json.loads(msg.data)

        self.log.info("Populating word graph ...")
        self.kpe.populate_word_graph(request)
        pass

    async def extract_segment_keyphrases(self, msg):
        request = json.loads(msg.data)

        output = self.kpe.get_keyphrases(request)
        self.log.debug("Output : {}".format(output))
        await self.nc.publish(msg.reply, json.dumps(output).encode())
        pass

    async def extract_instance_keyphrases(self, msg):
        request = json.loads(msg.data)

        self.log.info("Publishing Instance Keyphrases")
        output = self.kpe.get_instance_keyphrases(request)
        await self.nc.publish(msg.reply, json.dumps(output).encode())
        pass

    async def reset_keyphrases(self, msg):
        request = json.loads(msg.data)

        self.log.info("Resetting keyphrases graph ...")
        output = self.kpe.reset_keyphrase_graph(request)
        await self.nc.publish(msg, json.dumps(output).encode())
        pass

    # Utility function
    def _reformat_topic(self, topic, instance_id=None):
        topic_list = topic.split('.')

        if instance_id is None:
            context_instance_topic = [str(self.context_instance_id) if x == '*' else x for x in topic_list]
        else:
            context_instance_topic = [str(instance_id) if x == '*' else x for x in topic_list]

        context_instance_topic = ".".join(context_instance_topic)
        return context_instance_topic
