import json
import logging
from timeit import default_timer as timer
import traceback
import uuid
from typing import List, Dict, Mapping
from nats.aio.errors import ErrTimeout

logger = logging.getLogger(__name__)


class NATSTransport(object):
    def __init__(self, nats_manager, watcher_service=None, meeting_service=None):
        self.nats_manager = nats_manager
        self.watcher_service = watcher_service
        self.meeting_service = meeting_service
        self.whitelist_contexts = [
            "01DJSFMQ5MP8AX83Y89QC6T39E",
            "01DBB3SN99AVJ8ZWJDQ57X9TGX",
            "01DBB3SN874B4V18DCP4ATMRXA",
        ]
        self.test_contexts = [
            "01DBCR0S832GDJS7NGF9YW11WH",
            "01DBB3SNC86R968E1ESMQFCYK7",
            "01DBCSCC4YWZD0D1E1KT4A4RJC",
            "01DBCS96Y867PTEH9Z5B6TAACM",
        ]
        self.production_testing = ["01DBB3SN6EVW6Y4CZ6ETFC9Y9X"]

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
        instance_id = msg_data["instanceId"]
        session_id = context_id + ":" + instance_id

        logger.info(
            "instance created",
            extra={
                "contextId": context_id,
                "instanceId": instance_id,
                "sessionId": session_id,
            },
        )
        await self.subscribe_context_events()
        logger.info(
            "topics subscribed",
            extra={"topics": list(self.nats_manager.subscriptions.keys())},
        )

    async def subscribe_context_events(self):
        await self.nats_manager.subscribe(
            topic="context.instance.started",
            handler=self.context_start_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="context.instance.ended",
            handler=self.context_end_handler,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="recommendation.service.get_watchers",
            handler=self.get_watchers,
            queued=True,
        )
        await self.nats_manager.subscribe(
            topic="recommendation.service.get_meetings",
            handler=self.get_meetings,
            queued=True,
        )

        # For A/B testing: Alternate test
        await self.nats_manager.subscribe(
            topic="context.instance.started.v2",
            handler=self.context_start_handler,
            queued=True,
        )

    async def unsubscribe_lifecycle_events(self):
        await self.nats_manager.unsubscribe(topic="context.instance.started")
        await self.nats_manager.unsubscribe(topic="context.instance.ended")
        await self.nats_manager.unsubscribe(topic="recommendation.service.get_watchers")
        await self.nats_manager.unsubscribe(topic="recommendation.service.get_meetings")

    # NATS context handlers

    async def context_start_handler(self, msg):
        request = json.loads(msg.data)

        context_id = request["contextId"]
        instance_id = request["instanceId"]
        session_id = context_id + ":" + instance_id
        extra_options = request.get("extra_options", None)
        try:
            start = timer()
            kwargs = {
                "perform_query": True,
                "tag": "v1",
                "top_n": 30,
                "query_by": "keywords",
            }

            if context_id in self.test_contexts:
                kwargs = {
                    "perform_query": True,
                    "tag": "v2",
                    "top_n": 10,
                    "query_by": "keywords",
                }

            if context_id in self.production_testing:
                logger.info("using engineering context instead ...")
                context_id = "01DBB3SN874B4V18DCP4ATMRXA"
            if extra_options is not None:
                kwargs.update(extra_options)
            (
                reference_user_dict,
                reference_features,
                total_features,
            ) = self.watcher_service.initialize_objects(
                context_id=context_id, session_id=session_id, store_redis=True, **kwargs
            )

            end = timer()
            logger.info(
                "Vectorized reference users",
                extra={
                    "instanceId": instance_id,
                    "contextId": context_id,
                    "responseTime": end - start,
                },
            )

            updated_user_feature_map = self.watcher_service.featurize_reference_users(
                session_id=session_id,
                reference_user_dict=reference_user_dict,
                reference_features=reference_features,
                num_features=total_features,
            )

            end = timer()
            logger.info(
                "Formed LSH Buckets for reference users",
                extra={
                    "instanceId": instance_id,
                    "contextId": context_id,
                    "featureMap": updated_user_feature_map,
                    "sessionId": session_id,
                    "responseTime": end - start,
                },
            )
        except Exception as e:
            logger.error(
                "Error computing features for reference users", extra={"err": e},
            )
            print(traceback.print_exc())
            raise

    async def context_end_handler(self, msg):
        request = json.loads(msg.data)

        context_id = request["contextId"]
        instance_id = request["instanceId"]
        session_id = context_id + ":" + instance_id

        await self.watcher_service.cleanup_stores(session_id=session_id)

        logger.info("Meeting ended")

    # Topic Handler functions

    async def get_watchers(self, msg):
        start = timer()
        request = json.loads(msg.data)
        context_id = request["contextId"]
        instance_id = request["instanceId"]
        segment_object = request["segments"]
        segment_ids = [seg_ids["id"] for seg_ids in segment_object]
        keyphrase_list = request["keyphrases"]
        segment_user_ids = [
            str(uuid.UUID(seg_ids["spokenBy"])) for seg_ids in segment_object
        ]
        session_id = context_id + ":" + instance_id

        tag = "v1"
        if context_id in self.test_contexts:
            tag = "v2"

        try:
            participant_response = await self.get_meeting_attendees(
                instance_id=instance_id
            )
            attendees_response_object = participant_response.get("attendees", None)
            (
                original_rec_users,
                rec_users_dict,
                related_words,
                suggested_user_list,
                suggested_user_names,
                rec_user_names,
            ) = self.watcher_service.get_recommended_watchers(
                context_id=context_id,
                instance_id=instance_id,
                session_id=session_id,
                participant_response=attendees_response_object,
                input_query_list=keyphrase_list,
                input_kw_query=keyphrase_list,
                segment_user_ids=segment_user_ids,
                tag=tag,
                check_relevancy=True,
                query_by="keywords",
            )
            rec_users = list(rec_users_dict.keys())
            watcher_response = {
                "recommendedWatchers": suggested_user_list,
                "suggestedNames": suggested_user_names,
                "recommendedWatcherNames": rec_user_names,
            }

            end = timer()
            logger.info(
                "Recommended watchers computed",
                extra={
                    "originalRecWatchers": list(original_rec_users.keys()),
                    "recWatchers": rec_users,
                    "recWatcherNames": rec_user_names,
                    "userScores": list(rec_users_dict.values()),
                    "suggestedWatchers": suggested_user_list,
                    "suggestedWatchersNames": suggested_user_names,
                    "segmentWords": keyphrase_list,
                    "relatedWords": related_words,
                    "instanceId": request["instanceId"],
                    "sessionId": session_id,
                    "segmentsReceived": segment_ids,
                    "responseTime": end - start,
                },
            )

            await self.nats_manager.conn.publish(
                msg.reply, json.dumps(watcher_response).encode()
            )

            # Logic for posting to slack
            if context_id in (self.whitelist_contexts + self.production_testing):
                self.watcher_service.prepare_slack_validation(
                    req_data=request,
                    original_user_dict=original_rec_users,
                    user_dict=rec_users_dict,
                    word_list=related_words,
                    suggested_users=suggested_user_names,
                    segment_users=segment_user_ids,
                    upload=True,
                    post_to_slack=False,
                )
        except Exception as e:
            logger.error(
                "Error computing recommended watchers", extra={"err": e},
            )
            print(traceback.print_exc())
            raise

    async def get_meeting_attendees(self, instance_id: str) -> Mapping[str, List[Dict]]:
        topic = "ether.meeting.attendees"
        request_obj = {"meetingID": instance_id}

        try:
            msg = await self.nats_manager.conn.request(
                topic, json.dumps(request_obj).encode(), timeout=20
            )
            resp = json.loads(msg.data.decode())
        except ErrTimeout as e:
            logger.warning(e)
            resp = json.loads('{"attendees": null}')

        logger.debug("Response received", extra={"attendeesResp": resp})

        return resp

    async def get_meetings(self, msg):
        msg_data = json.loads(msg.data)
        keyphrase_list = msg_data["keyphrases"]

        try:
            self.meeting_service.recommend_meetings(kw_list=keyphrase_list)
        except Exception:
            raise

        pass
