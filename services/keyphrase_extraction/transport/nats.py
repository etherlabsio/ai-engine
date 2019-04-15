import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers
from nats.aio.utils import new_inbox
import signal
import json
import structlog

from graphrank import extract_keyphrases as kpe

log = structlog.getLogger(__name__)


class NATSTransport:
    def __init__(self, loop, subHandlers={}, url="nats://localhost:4222"):
        self.nc = NATS()
        self.loop = loop
        self.url = url
        self.subscriptions = []
        self.queueName = "io.etherlabs.ether.keyphrase_service"
        self.subscription_handlers = {
            "io.etherlabs.ether.keyphrase_service.extract_keyphrases": self.extract_segment_keyphrases,
            "io.etherlabs.ether.keyphrase_service.keyphrases_for_context_instance": self.extract_instance_keyphrases,
            "io.etherlabs.ether.keyphrase_service.reset_keyphrases": self.reset_keyphrases,
        }

    async def close(self):
        for sid in self.subscriptions:
            log.info("flushing nats sub", id=sid)
            await self.nc.unsubscribe(sid)
        await self.nc.drain()

    async def connect(self):
        loop = self.loop

        async def closed_cb():
            log.info("Connection to NATS is closed.")
            await asyncio.sleep(0.001, loop=loop)
            loop.stop()

        async def reconnected_cb():
            log.info("Connected to NATS at {}...".format(self.nc.connected_url.netloc))

        options = {
            "io_loop": loop,
            "closed_cb": closed_cb,
            "reconnected_cb": reconnected_cb
        }

        try:
            # Setting explicit list of servers in a cluster.
            log.info("Connecting ...")
            await self.nc.connect(servers=[self.url], loop=loop, **options, verbose=True)
        except ErrNoServers as e:
            log.error("no nats servers to connect", err=e)

        log.info("connected to nats server", url=self.url)

    async def subscribe(self):
        for topic in self.subscription_handlers:
            sid = await self.nc.subscribe(topic, self.queueName, self.message_handler)
            self.subscriptions.append(sid)

    async def message_handler(self, msg):
        try:
            subject = msg.subject
            reply = msg.reply
            log.info("received nats message", subject=subject, reply=reply, data=msg.data)
            handle = self.subscription_handlers[subject]
            await handle(msg)
            # log.info("Reached here ...")
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
        await self.nc.publish(msg.reply, json.dumps(output).encode())
        pass