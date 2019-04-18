import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers
from nats.aio.utils import new_inbox
import signal
import json
import logging

from graphrank import extract_keyphrases as kpe

log = logging.getLogger(__name__)

class NATSTransport:
    def __init__(self, loop, subHandlers={}, url="nats://localhost:4222"):
        self.nc = NATS()
        self.loop = loop
        self.url = url
        self.context_subscriptions = []
        self.subscriptions = []
        self.context_instance_id = []
        self.context_id = []
        self.queueName = "io.etherlabs.ether.keyphrase_service"
        self.topics = [
            "io.etherlabs.ether.keyphrase_service.*.extract_keyphrases",
            "io.etherlabs.ether.keyphrase_service.*.keyphrases_for_context_instance",
        ]
        self.context_event = ["context.instance.created"]
        self.context_lifecycle_topics = [
            "context.instance.*.started",
            "context.instance.*.ended",
            "context.instance.*.context_changed"
        ]
        self.subscription_handlers = {
            "extract_keyphrases": self.extract_segment_keyphrases,
            "keyphrases_for_context_instance": self.extract_instance_keyphrases,
            "reset_keyphrases": self.reset_keyphrases,
        }

    async def close_connection(self):
        if self.nc.is_connected:
            log.info("Closing connection ...")
            await self.nc.drain()

    async def close_context_life(self):
        for sid in self.subscriptions:
            log.info("flushing nats context sub", id=sid)
            await self.nc.unsubscribe(sid)

    async def connect(self):
        loop = self.loop

        async def closed_cb():
            if self.nc.is_closed:
                log.info("Connection to NATS is closed.")
                await asyncio.sleep(0.1, loop=loop)
                loop.stop()

        async def error_cb(e):
            log.error("There was an error:", err=e)

        async def reconnected_cb():
            log.info("Connected to NATS at {}...".format(self.nc.connected_url.netloc))

        options = {
            "io_loop": loop,
            "closed_cb": closed_cb,
            "reconnected_cb": reconnected_cb,
            "error_cb": error_cb
        }

        try:
            # Setting explicit list of servers in a cluster.
            log.info("Connecting ...")
            await self.nc.connect(servers=[self.url], loop=loop, **options, verbose=True)
        except ErrNoServers as e:
            log.error("no nats servers to connect", err=e)

        log.info("connected to nats server", url=self.url)

    async def context_subscribe(self):
        for create_event in self.context_event:
            sid = await self.nc.subscribe(create_event, self.queueName, self.ctx_lifecycle_handler)
            self.context_subscriptions.append(sid)

    async def context_lifecycle_events(self):
        log.info("subscribing to instance events")

        for context_topic in self.context_lifecycle_topics:
            context_instance_topic = await self.reformat_topic(context_topic)
            log.info("subscribing to topics", topic=context_instance_topic)
            sid = await self.nc.subscribe(context_instance_topic, self.queueName, self.ctx_lifecycle_handler)
            self.subscriptions.append(sid)

        await self.subscribe()

    async def subscribe(self):
        for topic in self.topics:
            keyphrase_instance_topic = await self.reformat_topic(topic)
            log.info("subscribing to keyphrase topics", topic=keyphrase_instance_topic)
            sid = await self.nc.subscribe(keyphrase_instance_topic, self.queueName, self.message_handler)
            self.subscriptions.append(sid)

    # Handler Callbacks

    async def ctx_lifecycle_handler(self, msg):
        try:
            subject = msg.subject
            reply = msg.reply
            log.info("received nats message", subject=subject, reply=reply, data=msg.data)
            msg_data = json.loads(msg.data)

            # Start listening to keyphrase events when instance starts
            if msg_data['state'] == 'created':
                self.context_id = msg_data['contextId']
                self.context_instance_id = msg_data['id']
                log.info("instance created", cid=self.context_id, ciid=self.context_instance_id)
                self.loop.run_until_complete(self.context_lifecycle_events())

            elif msg_data['state'] == 'started':
                log.info("Start keyphrase subscriptions")

            elif msg_data['state'] == 'context_changed':
                # Update contextId when change is notified
                self.context_id = msg_data['contextId']

            elif msg_data['state'] == 'ended':
                # Close, drain and unsubscribe connections to keyphrase topics
                await self.close_context_life()

                # Reset graph
                await self.reset_keyphrases(msg)
            else:
                pass
        except Exception as e:
            await self.nc.publish(msg.reply, json.dumps({
                "error": {
                    "code": 0,
                    "message": "process message failure",
                    "cause": str(e)
                }
            }).encode())
            log.error("failed to process message", subject=msg.subject, data=msg.data, err=e)
            pass

    async def message_handler(self, msg):
        try:
            subject = msg.subject
            reply = msg.reply
            log.info("received nats message", subject=subject, reply=reply, data=msg.data)
            topic_function = subject.split('.')[-1]

            handle = self.subscription_handlers[topic_function]
            await handle(msg)
        except Exception as e:
            await self.nc.publish(msg.reply, json.dumps({
                "error": {
                    "code": 0,
                    "message": "process message failure",
                    "cause": str(e)
                }
            }).encode())
            log.error("failed to process message", subject=msg.subject, data=msg.data, err=e)
            pass

    # Utility function
    async def reformat_topic(self, topic):
        topic = topic.split('.')
        context_instance_topic = [str(self.context_instance_id) if x == '*' else x for x in topic]
        context_instance_topic = ".".join(context_instance_topic)
        return context_instance_topic

    # Topic Handler functions
    async def extract_segment_keyphrases(self, msg):
        request = json.loads(msg.data)

        segments_array = request['segments']

        # Decide between PIM or Chapter keyphrases
        if len(segments_array) > 1:
            log.info("Publishing Chapter Keyphrases")
            output = kpe.get_chapter_keyphrases(request)
        else:
            log.info("Publishing PIM Keyphrases")
            output = kpe.get_pim_keyphrases(request)
        log.info("Output : {}".format(output))
        await self.nc.publish(msg.reply, json.dumps(output).encode())
        pass

    async def extract_instance_keyphrases(self, msg):
        request = json.loads(msg.data)

        log.info("Publishing Instance Keyphrases")
        output = kpe.get_instance_keyphrases(request)
        log.info("Output : {}".format(output))
        await self.nc.publish(msg.reply, json.dumps(output).encode())
        pass

    async def reset_keyphrases(self, msg):
        request = json.loads(msg.data)

        log.info("Resetting keyphrases graph ...")
        output = kpe.reset_keyphrase_graph(request)
        log.info("output", out=json.dumps(output).encode())
        await self.nc.publish(msg, json.dumps(output).encode())
        pass
