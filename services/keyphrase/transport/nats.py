import json
import logging
from timeit import default_timer as timer
import traceback

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
        await self.subscribe_context_events()
        logger.info(
            "topics subscribed",
            extra={"topics": list(self.nats_manager.subscriptions.keys())},
        )
        self.keyphrase_service.wake_up_lambda(req_data=msg_data)

    async def subscribe_context_events(self):
        await self.nats_manager.subscribe(
            topic="context.instance." + "started",
            handler=self.context_start_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance." + "context_changed",
            handler=self.context_change_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance." + "ended",
            handler=self.context_end_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="keyphrase_service." + "extract_keyphrases",
            handler=self.extract_segment_keyphrases,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="keyphrase_service." + "keyphrases_for_context_instance",
            handler=self.extract_instance_keyphrases,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance." + "add_segments",
            handler=self.populate_graph,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="keyphrase_service." + "extract_keyphrases_with_offset",
            handler=self.chapter_offset_handler,
            queued=True,
        )

    async def unsubscribe_lifecycle_events(self):
        await self.nats_manager.unsubscribe(topic="context.instance." + "started")
        await self.nats_manager.unsubscribe(
            topic="context.instance." + "context_changed"
        )
        await self.nats_manager.unsubscribe(topic="context.instance." + "ended")
        await self.nats_manager.unsubscribe(
            topic="keyphrase_service." + "extract_keyphrases"
        )
        await self.nats_manager.unsubscribe(
            topic="keyphrase_service." + "keyphrases_for_context_instance"
        )
        await self.nats_manager.unsubscribe(topic="context.instance." + "add_segments")
        await self.nats_manager.unsubscribe(
            topic="keyphrase_service." + "extract_keyphrases_with_offset"
        )

    # NATS context handlers

    async def context_start_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data["state"] == "started":
            logger.info("Instance started")
            self.keyphrase_service.initialize_meeting_graph(req_data=msg_data)
        pass

    async def context_change_handler(self, msg):
        msg_data = json.loads(msg.data)
        if msg_data["state"] == "context_changed":
            # Update contextId when change is notified
            context_id = msg_data["contextId"]
            return context_id
        pass

    async def context_end_handler(self, msg):
        # Reset graph
        await self.reset_keyphrases(msg)

    # Topic Handler functions

    async def populate_graph(self, msg):
        start = timer()
        request = json.loads(msg.data)
        segment_object = request["segments"]

        try:
            context_graph, meeting_word_graph = self.keyphrase_service.populate_word_graph(
                request
            )

            # Compute embeddings for segments and keyphrases
            context_graph, meeting_word_graph = self.keyphrase_service.populate_context_embeddings(
                req_data=request,
                segment_object=segment_object,
                context_graph=context_graph,
                meeting_word_graph=meeting_word_graph,
            )

            end = timer()
            logger.info(
                "Populated graph and written to s3",
                extra={
                    "graphId": meeting_word_graph.graph.get("graphId"),
                    "nodes": meeting_word_graph.number_of_nodes(),
                    "edges": meeting_word_graph.number_of_edges(),
                    "kgNodes": context_graph.number_of_nodes(),
                    "kgEdges": context_graph.number_of_edges(),
                    "instanceId": request["instanceId"],
                    "responseTime": end - start,
                },
            )
        except Exception:
            end = timer()
            logger.error(
                "Error populating graph",
                extra={
                    "err": traceback.print_exc(),
                    "responseTime": end - start,
                    "instanceId": request["instanceId"],
                },
            )

    async def extract_segment_keyphrases(self, msg):
        start = timer()
        request = json.loads(msg.data)
        segment_object = request["segments"]
        context_info = request["contextId"] + ":" + request["instanceId"]

        limit = request.get("limit", 10)
        output = self.keyphrase_service.get_keyphrases(
            request, segment_object=segment_object, n_kw=limit, validate=False
        )
        end = timer()

        deadline_time = end - start
        if deadline_time > 5:
            timeout_msg = "-Context deadline is exceeding: {}; {}".format(
                deadline_time, 5
            )
        else:
            timeout_msg = ""

        if limit == 6:
            logger.info(
                "Publishing chapter keyphrases" + timeout_msg,
                extra={
                    "graphId": context_info,
                    "chapterKeyphraseList": output,
                    "instanceId": request["instanceId"],
                    "numOfSegments": len(request["segments"]),
                    "limit": limit,
                    "responseTime": end - start,
                    "requestReceived": request,
                },
            )
        else:
            logger.info(
                "Publishing PIM keyphrases" + timeout_msg,
                extra={
                    "graphId": context_info,
                    "pimKeyphraseList": output,
                    "instanceId": request["instanceId"],
                    "numOfSegments": len(request["segments"]),
                    "limit": limit,
                    "responseTime": end - start,
                    "requestReceived": request,
                },
            )
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def extract_instance_keyphrases(self, msg):
        start = timer()
        request = json.loads(msg.data)
        context_info = request["contextId"] + ":" + request["instanceId"]

        limit = request.get("limit", 10)
        output = self.keyphrase_service.get_instance_keyphrases(request, n_kw=limit)
        end = timer()

        deadline_time = end - start
        if deadline_time > 5:
            timeout_msg = "-Context deadline is exceeding: {}; {}".format(
                deadline_time, 5
            )
        else:
            timeout_msg = ""

        logger.info(
            "Publishing instance keyphrases" + timeout_msg,
            extra={
                "graphId": context_info,
                "instanceList": output,
                "instanceId": request["instanceId"],
                "numOfSegments": len(request["segments"]),
                "limit": limit,
                "responseTime": end - start,
                "requestReceived": request,
            },
        )
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def chapter_offset_handler(self, msg):
        start = timer()
        request = json.loads(msg.data)
        context_info = request["contextId"] + ":" + request["instanceId"]

        limit = request.get("limit", 10)
        output = self.keyphrase_service.get_keyphrases_with_offset(request, n_kw=limit)
        end = timer()

        deadline_time = end - start
        if deadline_time > 5:
            timeout_msg = "-Context deadline is exceeding: {}; {}".format(
                deadline_time, 5
            )
        else:
            timeout_msg = ""

        logger.info(
            "Publishing chapter keyphrases with offset" + timeout_msg,
            extra={
                "graphId": context_info,
                "chapterOffsetList": output,
                "instanceId": request["instanceId"],
                "numOfSegments": len(request["segments"]),
                "limit": limit,
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
