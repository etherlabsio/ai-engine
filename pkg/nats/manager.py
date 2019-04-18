import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers
import signal
import json
import structlog

log = structlog.getLogger(__name__)

class Manager:
    def __init__(self, loop, queueName, url="nats://docker.for.mac.localhost:4222", nc=NATS()):
        self.conn = nc
        self.loop = loop
        self.url = url
        self.queueName = queueName
        self.subscriptions = {}

    async def close(self):
        for subject, sid in self.subscriptions.items():
            log.info("flushing nats sub", id=sid)
            await self.conn.unsubscribe(sid)
        await self.conn.drain()

    async def connect(self):
        loop = self.loop

        async def closed_cb():
            log.info("connection to NATS is closed.")
            await asyncio.sleep(0.1, loop=loop)

        async def reconnected_cb():
            log.info("connected to NATS at {}...".format(self.conn.connected_url.netloc))

        options = {
            "io_loop": loop,
            "closed_cb": closed_cb,
            "reconnected_cb": reconnected_cb
        }

        try:
            # Setting explicit list of servers in a cluster.
            await self.conn.connect(servers=[self.url], loop=loop, **options)
        except ErrNoServers as e:
            log.error("no nats servers to connect", err=e)

        log.info("connected to nats server", url=self.url)

    async def subscribe(self, topic, handler, queued=True):
        sid = None
        if queued is True:
            sid = await self.conn.subscribe(topic, self.queueName, self.message_handler(handler))
        else:
            sid = await self.conn.subscribe(topic, cb=self.message_handler(handler))
        self.subscriptions[topic] = sid

    async def unsubscribe(self, topic):
        if not self.subscriptions.has_key(topic):
            pass
        sid = self.subscriptions.get(topic)
        await self.conn.unsubscribe(sid)
        self.subscriptions.pop(topic)

    async def message_handler(self, cb):
        async def handle(msg):
            try:
                subject = msg.subject
                reply = msg.reply
                log.info("received nats message", subject=subject, reply=reply, data=msg.data)
                await cb(self, msg)
            except Exception as e:
                send = self.conn.publish
                if  reply:
                    send = self.conn.reply
                await send(msg.reply, json.dumps({
                    "error": {
                        "code": 0,
                        "message": "process message failure",
                        "cause": str(e)
                    }
                }).encode())
                log.error("failed to process message", subject=msg.subject, data=msg.data, err=e)

        return handle
