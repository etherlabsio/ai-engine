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

        keyphrase_attr_dict = {"type": "descriptive", "important": False}

        try:
            modified_request_obj, meeting_word_graph = self.keyphrase_service.populate_and_embed_graph(
                req_data=request,
                segment_object=segment_object,
                keyphrase_attr=keyphrase_attr_dict,
            )

            await self.populate_segment_keyphrase_info(
                modified_req_data=modified_request_obj
            )
            end = timer()
            logger.info(
                "Populated to dgraph and written to s3",
                extra={
                    "graphId": meeting_word_graph.graph.get("graphId"),
                    "nodes": meeting_word_graph.number_of_nodes(),
                    "edges": meeting_word_graph.number_of_edges(),
                    "instanceId": request["instanceId"],
                    "responseTime": end - start,
                },
            )
        except Exception:
            end = timer()
            logger.error(
                "Error populating to dgraph",
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
        validation = request.get("validate", True)
        populate_graph = request.get("populateGraph", True)

        segment_ids = [seg_ids["id"] for seg_ids in segment_object]
        context_info = request["contextId"] + ":" + request["instanceId"]

        limit = request.get("limit", 10)

        keyphrase_attr_dict = {"type": "descriptive", "important": False}

        if populate_graph:
            output = self.keyphrase_service.get_keyphrases(
                request,
                segment_object=segment_object,
                n_kw=limit,
                validate=validation,
                keyphrase_attr=keyphrase_attr_dict,
            )
        else:
            keyphrase_attr_dict = {"type": "descriptive", "important": True}
            group_id = self.keyphrase_service.utils.hash_sha_object()
            output = self.keyphrase_service.get_summary_chapter_keyphrases(
                request,
                segment_object=segment_object,
                n_kw=limit,
                validate=validation,
                populate_graph=populate_graph,
                group_id=group_id,
                keyphrase_attr=keyphrase_attr_dict,
            )

        end = timer()

        deadline_time = end - start
        if deadline_time > 15:
            timeout_msg = "-Context deadline is exceeding: {}; {}".format(
                deadline_time, 15
            )
        else:
            timeout_msg = ""

        if populate_graph is not True:
            logger.info(
                "Publishing summary chapter keyphrases" + timeout_msg,
                extra={
                    "graphId": context_info,
                    "topicKeyphraseList": output,
                    "instanceId": request["instanceId"],
                    "numOfSegments": len(request["segments"]),
                    "limit": limit,
                    "responseTime": end - start,
                    "segmentsReceived": segment_ids,
                },
            )

        elif limit == 6:
            logger.info(
                "Publishing chapter keyphrases" + timeout_msg,
                extra={
                    "graphId": context_info,
                    "chapterKeyphraseList": output,
                    "instanceId": request["instanceId"],
                    "numOfSegments": len(request["segments"]),
                    "limit": limit,
                    "responseTime": end - start,
                    "segmentsReceived": segment_ids,
                    "dynamicThreshold": len(output["keyphrases"]),
                },
            )
        elif limit == 10:
            logger.info(
                "Publishing PIM keyphrases" + timeout_msg,
                extra={
                    "graphId": context_info,
                    "pimKeyphraseList": output,
                    "instanceId": request["instanceId"],
                    "numOfSegments": len(request["segments"]),
                    "limit": limit,
                    "responseTime": end - start,
                    "segmentsReceived": segment_ids,
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
        if deadline_time > 15:
            timeout_msg = "-Context deadline is exceeding: {}; {}".format(
                deadline_time, 15
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
        segment_object = request["segments"]
        segment_ids = [seg_ids["id"] for seg_ids in segment_object]

        limit = request.get("limit", 10)
        output = self.keyphrase_service.get_keyphrases_with_offset(request, n_kw=limit)
        end = timer()

        deadline_time = end - start
        if deadline_time > 15:
            timeout_msg = "-Context deadline is exceeding: {}; {}".format(
                deadline_time, 15
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
                "segmentsReceived": segment_ids,
            },
        )

        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def reset_keyphrases(self, msg):
        request = json.loads(msg.data)
        logger.info("Resetting keyphrases graph")
        output = self.keyphrase_service.reset_keyphrase_graph(request)
        await self.nats_manager.conn.publish(msg, json.dumps(output).encode())

    async def populate_segment_keyphrase_info(self, modified_req_data):
        eg_segment_topic = "ether_graph_service.populate_segments"
        ether_graph_request = json.dumps(modified_req_data).encode()

        msg = await self.nats_manager.conn.publish(
            eg_segment_topic, ether_graph_request
        )
        resp = msg.data.decode()

        return resp

    async def query_ether_graph(self, query_text, variables=None):
        eg_query_topic = "ether_graph_service.perform_query"
        TIMEOUT = 20
        query_request = {"query": query_text, "variables": variables}

        msg = await self.nats_manager.conn.request(
            eg_query_topic, json.dumps(query_request).encode(), timeout=TIMEOUT
        )
        resp = msg.data.decode()

        return resp
