import json
import asyncio
import structlog

from graphrank.extract_keyphrases import ExtractKeyphrase
from .manager import Manager

log = structlog.getLogger(__name__)


class CallbackHandler(object):
    def __init__(self, loop, nc, subscribe_lifecycle, unsubscribe_lifecycle):
        self.loop = loop
        self.nc = nc
        self.context_id = []
        self.context_instance_id = []
        self.subscribe_lifecycle = subscribe_lifecycle
        self.unsubscribe_lifecycle = unsubscribe_lifecycle
        self.kpe = ExtractKeyphrase()

        self.context_lifecycle_topics = {
            "context.instance.*.started": self.context_start_handler,
            "context.instance.*.context_changed": self.context_change_handler,
            "context.instance.*.ended": self.context_end_handler,
            "keyphrase_service.*.extract_keyphrases": self.extract_segment_keyphrases,
            "keyphrase_service.*.keyphrases_for_context_instance": self.extract_instance_keyphrases,
            "context.instance.*.add_segments": self.populate_graph
        }
        self.context_subscriptions = {}

    async def context_created_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'created':
            self.context_id = msg_data['contextId']
            self.context_instance_id = msg_data['id']
            log.info("instance created", cid=self.context_id, ciid=self.context_instance_id)
            await self.rebuild_context_topics()
            await asyncio.sleep(0.1)
            await self.subscribe_lifecycle()

    async def context_start_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'started':
            log.info("Start keyphrase subscriptions")
        pass

    async def context_change_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'context_changed':
            # Update contextId when change is notified
            self.context_id = msg_data['contextId']
        pass

    async def context_end_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data['state'] == 'ended':
            # Close, drain and unsubscribe connections to keyphrase topics
            await self.unsubscribe_lifecycle()
            # Reset graph
            await self.reset_keyphrases(msg)
        pass

    # Topic Handler functions

    async def populate_graph(self, msg):
        request = json.loads(msg.data)

        log.info("Populating word graph ...")
        self.kpe.populate_word_graph(request)
        pass

    async def extract_segment_keyphrases(self, msg):
        request = json.loads(msg.data)

        output = self.kpe.get_keyphrases(request)
        log.info("Output : {}".format(output))
        output_json = json.dumps(output).encode()
        log.info("Output in json: {}".format(output_json))
        await self.nc.publish(msg.reply, json.dumps(output).encode())
        pass

    async def extract_instance_keyphrases(self, msg):
        request = json.loads(msg.data)

        log.info("Publishing Instance Keyphrases")
        output = self.kpe.get_instance_keyphrases(request)
        log.info("Output : {}".format(output))
        await self.nc.publish(msg.reply, json.dumps(output).encode())
        pass

    async def reset_keyphrases(self, msg):
        request = json.loads(msg.data)

        log.info("Resetting keyphrases graph ...")
        output = self.kpe.reset_keyphrase_graph(request)
        log.info("output", out=json.dumps(output).encode())
        await self.nc.publish(msg, json.dumps(output).encode())
        pass

    # Utility function
    async def _reformat_topic(self, topic):
        topic_list = topic.split('.')

        context_instance_topic = [str(self.context_instance_id) if x == '*' else x for x in topic_list]
        context_instance_topic = ".".join(context_instance_topic)
        await self._update_subscription_handlers(topic_placeholder=topic, subscribed_topic=context_instance_topic)
        return context_instance_topic

    async def _update_subscription_handlers(self, topic_placeholder, subscribed_topic):
        try:
            self.context_subscriptions[subscribed_topic] = self.context_lifecycle_topics[topic_placeholder]
        except Exception as e:
            log.error("Cannot update subscription handler", err=e)
        pass

    async def rebuild_context_topics(self):
        for topic in self.context_lifecycle_topics.keys():
            await self._reformat_topic(topic)

        # log.info("new context", sub=self.context_subscriptions)


class NATSHandler(CallbackHandler):
    def __init__(self, loop, url="nats://docker.for.mac.localhost:4222"):
        self.loop = loop
        self.url = url
        self.queueName = "io.etherlabs.ether.keyphrase_service"
        self.nats_manager = Manager(loop=self.loop, url=self.url, queueName=self.queueName)
        self.nc = self.nats_manager.conn
        super().__init__(loop,
                         nc=self.nc,
                         subscribe_lifecycle=self.subscribe_context_lifecycle,
                         unsubscribe_lifecycle=self.unsubscribe_lifecycle_events)

        self.context_topic = ["context.instance.created"]

    async def subscribe_context(self):
        for topic in self.context_topic:
            log.info("Subscribing to context instance event", topic=topic)
            await self.nats_manager.subscribe(topic, handler=self.context_created_handler, queued=True)

    async def subscribe_context_lifecycle(self):
        for topic, func in self.context_subscriptions.items():
            await self.nats_manager.subscribe(topic, handler=self.context_subscriptions[topic],
                                              queued=True)
        log.info('subscriptions', sub=self.nats_manager.subscriptions)

    async def unsubscribe_lifecycle_events(self):
        for topic, func in self.context_subscriptions.items():
            # topic = self.reformat_topic(topic)
            log.info("Unsubscribing:", topic=topic)
            await self.nats_manager.unsubscribe(topic)
