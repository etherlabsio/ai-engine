import json
import logging
from timeit import default_timer as timer
import traceback

from keyphrase.objects import Context, Segment, Request, SummaryRequest

logger = logging.getLogger(__name__)


class NATSTransport(object):
    def __init__(self, nats_manager, keyphrase_service):
        self.nats_manager = nats_manager
        self.keyphrase_service = keyphrase_service
        self.request_schema = Request.schema()
        self.segment_schema = Segment.schema()
        self.context_schema = Context.schema()

    async def subscribe_context(self):
        context_created_topic = "context.instance.created"
        logger.info(
            "Subscribing to context instance event",
            extra={"topic": context_created_topic},
        )
        await self.nats_manager.subscribe(
            context_created_topic, handler=self.context_created_handler, queued=True,
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
        logger.info("Instance started")

        request = json.loads(msg.data)
        request_object = Request.get_object(request)
        self.keyphrase_service.initialize_meeting_graph(req_data=request_object)

    async def context_change_handler(self, msg):
        request = json.loads(msg.data)
        request_object = Request.get_object(request)

        # Update contextId when change is notified
        context_id = request_object.contextId
        return context_id

    async def context_end_handler(self, msg):
        # Reset graph
        await self.reset_keyphrases(msg)

    # Topic Handler functions

    async def populate_graph(self, msg):
        start = timer()
        request = json.loads(msg.data)
        try:
            request_object = Request.get_object(request)
            populate_graph = request_object.populateGraph

            if populate_graph is True:
                highlight = False
            else:
                highlight = True

            segment_object = request_object.segments

            (
                modified_request_obj,
                meeting_word_graph,
            ) = self.keyphrase_service.populate_and_embed_graph(
                req_data=request_object,
                segment_object=segment_object,
                highlight=highlight,
            )

            # Forward the modified Request-Segment object to graph-service
            modified_request_obj_dict = Request.get_dict(modified_request_obj)
            await self.populate_segment_keyphrase_info(
                modified_req_data=modified_request_obj_dict
            )

            end = timer()
            logger.info(
                "Populated to dgraph and written to s3",
                extra={
                    "graphId": meeting_word_graph.graph.get("graphId"),
                    "nodes": meeting_word_graph.number_of_nodes(),
                    "edges": meeting_word_graph.number_of_edges(),
                    "instanceId": request_object.instanceId,
                    "responseTime": end - start,
                },
            )
        except Exception as e:
            end = timer()
            logger.error(
                "Error populating to dgraph",
                extra={
                    "err": e,
                    "responseTime": end - start,
                    # "instanceId": request_object.instanceId,
                },
            )

    async def extract_segment_keyphrases(self, msg):
        start = timer()
        request = json.loads(msg.data)

        request_object = Request.get_object(request)
        segment_object = request_object.segments

        summary_object = SummaryRequest.get_object(request)

        validation = request_object.validate
        populate_graph = request_object.populateGraph

        segment_ids = [seg_obj.id for seg_obj in segment_object]
        context_info = request_object.contextId + ":" + request_object.instanceId

        limit = request_object.limit
        group_id = None

        highlight = False
        if populate_graph is not True:
            highlight = True
            group_id = self.keyphrase_service.utils.hash_sha_object()

        output, summary_request_object = await self.keyphrase_service.get_keyphrases(
            req_data=request_object,
            segment_object=segment_object,
            summary_object=summary_object,
            n_kw=limit,
            validate=validation,
            group_id=group_id,
            highlight=highlight,
        )

        end = timer()

        if highlight:
            # Forward the modified Request-Segment-Keyphrase object to graph-service
            summary_request_dict = SummaryRequest.get_dict(summary_request_object)

            await self.populate_summary_info(summary_req_data=summary_request_dict)
            logger.info(
                "Publishing summary keyphrases",
                extra={
                    "graphId": context_info,
                    "topicKeyphraseList": output,
                    "instanceId": request_object.instanceId,
                    "numOfSegments": len(segment_object),
                    "limit": limit,
                    "responseTime": end - start,
                    "segmentsReceived": segment_ids,
                },
            )

        else:
            logger.info(
                "Publishing chapter keyphrases",
                extra={
                    "graphId": context_info,
                    "chapterKeyphraseList": output,
                    "instanceId": request_object.instanceId,
                    "numOfSegments": len(segment_object),
                    "limit": limit,
                    "responseTime": end - start,
                    "segmentsReceived": segment_ids,
                    "dynamicThreshold": len(output["keyphrases"]),
                },
            )
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def extract_instance_keyphrases(self, msg):
        start = timer()
        request = json.loads(msg.data)
        request_object = Request.get_object(request)
        context_info = request_object.contextId + ":" + request_object.instanceId

        limit = request_object.limit
        output = self.keyphrase_service.get_instance_keyphrases(
            request_object, n_kw=limit
        )
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
                "instanceId": request_object.instanceId,
                "numOfSegments": len(request_object.segments),
                "limit": limit,
                "responseTime": end - start,
                "requestReceived": request,
            },
        )
        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def chapter_offset_handler(self, msg):
        start = timer()
        request = json.loads(msg.data)
        request_object = Request.get_object(request)
        context_info = request_object.contextId + ":" + request_object.instanceId
        segment_object = request_object.segments
        segment_ids = [seg_obj.id for seg_obj in segment_object]

        limit = request_object.limit
        output = await self.keyphrase_service.get_keyphrases_with_offset(
            request_object, n_kw=limit
        )
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
                "instanceId": request_object.instanceId,
                "numOfSegments": len(segment_object),
                "limit": limit,
                "responseTime": end - start,
                "segmentsReceived": segment_ids,
            },
        )

        await self.nats_manager.conn.publish(msg.reply, json.dumps(output).encode())

    async def reset_keyphrases(self, msg):
        request = json.loads(msg.data)
        request_object = Request.get_object(request)
        logger.info("Resetting keyphrases graph")
        self.keyphrase_service.reset_keyphrase_graph(request_object)

    async def populate_segment_keyphrase_info(self, modified_req_data: dict):
        eg_segment_topic = "ether_graph_service.add_segments"
        ether_graph_request = json.dumps(modified_req_data).encode()

        await self.nats_manager.conn.publish(eg_segment_topic, ether_graph_request)

    async def populate_summary_info(self, summary_req_data: dict):
        eg_segment_topic = "ether_graph_service.populate_summary"
        ether_graph_request = json.dumps(summary_req_data).encode()

        await self.nats_manager.conn.publish(eg_segment_topic, ether_graph_request)
