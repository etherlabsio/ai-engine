import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers
import json
import logging
import traceback

logger = logging.getLogger(__name__)


class Manager:
    def __init__(
        self, loop, queue_name, url="nats://docker.for.mac.localhost:4222", nc=NATS()
    ):
        self.conn = nc
        self.loop = loop
        self.url = url
        self.queue_name = queue_name
        self.subscriptions = {}

    async def close(self):
        for subject, sid in self.subscriptions.items():
            logger.info("flushing nats sub", extra={"sid": sid})
            if self.conn.is_connected:
                await self.conn.unsubscribe(sid)
        await self.conn.drain()

    async def connect(self):
        loop = self.loop

        async def closed_cb():
            logger.info("connection to NATS is closed.")
            await asyncio.sleep(0.1, loop=loop)
            loop.stop()

        async def reconnected_cb():
            logger.info(
                "connected to NATS at {}...".format(self.conn.connected_url.netloc)
            )

        options = {
            "io_loop": loop,
            "closed_cb": closed_cb,
            "reconnected_cb": reconnected_cb,
        }

        try:
            # Setting explicit list of servers in a cluster.
            await self.conn.connect(servers=[self.url], loop=loop, **options)
        except ErrNoServers as e:
            logger.error("no nats servers to connect ", extra={"err": e})

        logger.info("connected to nats server ", extra={"url": self.url})

    async def subscribe(self, topic, handler, queued=True):
        sid = None
        if queued is True:
            sid = await self.conn.subscribe(
                topic, queue=self.queue_name, cb=self.message_handler(cb=handler)
            )
        else:
            sid = await self.conn.subscribe(topic, cb=self.message_handler(handler))
        self.subscriptions[topic] = sid

    async def unsubscribe(self, topic):
        if topic in self.subscriptions.keys():
            sid = self.subscriptions.get(topic)
            await self.conn.unsubscribe(sid)
            self.subscriptions.pop(topic)
        else:
            logger.debug(
                "Topic not found in the subscription list ", extra={"topic": topic}
            )

    def message_handler(self, cb):
        async def handle(msg):
            try:
                subject = msg.subject
                reply = msg.reply
                logger.info(
                    "received nats message ",
                    extra={"subject": subject, "reply": reply, "data": msg.data},
                )
                await cb(msg)
            except Exception as e:
                send = self.conn.publish
                await send(
                    msg.reply,
                    json.dumps(
                        {
                            "error": {
                                "code": 0,
                                "message": "process message failure",
                                "cause": str(e),
                            }
                        }
                    ).encode(),
                )
                logger.error(
                    "failed to process message:",
                    extra={
                        "subject": msg.subject,
                        "data": msg.data,
                        "err": traceback.print_exc(limit=3),
                    },
                )

        return handle
