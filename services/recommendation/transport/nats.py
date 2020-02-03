import json
import logging
from timeit import default_timer as timer
import traceback
import uuid

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

        self.context_user_feature_map = {}
        self.context_hash_session = {}

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

        self.context_user_feature_map.update({session_id: {}})
        self.context_hash_session.update(
            {session_id: self.watcher_service.create_hash_session()}
        )
        logger.info(
            "instance created",
            extra={
                "contextId": context_id,
                "instanceId": instance_id,
                "sessionId": session_id,
                "contextHashSessions": self.context_hash_session,
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
        try:
            start = timer()
            (
                reference_user_dict,
                reference_features,
            ) = self.watcher_service.initialize_reference_objects(
                context_id=context_id, perform_query=True
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
            try:
                user_feature_map = self.context_user_feature_map[session_id]
                context_hash_session = self.context_hash_session[session_id]
            except KeyError as e:
                logger.warning(
                    "could not find the hash session ... creating new session",
                    extra={"warn": e},
                )
                self.context_user_feature_map.update({session_id: {}})
                self.context_hash_session.update(
                    {session_id: self.watcher_service.create_hash_session()}
                )

                user_feature_map = self.context_user_feature_map[session_id]
                context_hash_session = self.context_hash_session[session_id]

            updated_user_feature_map = await self.watcher_service.featurize_reference_users(
                reference_user_dict=reference_user_dict,
                reference_features=reference_features,
                user_feature_map=user_feature_map,
                hash_session_object=context_hash_session,
            )
            self.context_user_feature_map[session_id] = updated_user_feature_map

            end = timer()
            logger.info(
                "Formed LSH Buckets for reference users",
                extra={
                    "instanceId": instance_id,
                    "contextId": context_id,
                    "featureMap": updated_user_feature_map,
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
        logger.info("Meeting ended")
        pass

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

        try:
            try:
                user_feature_map_for_context = self.context_user_feature_map[session_id]
                context_hash_session = self.context_hash_session[session_id]
            except KeyError as e:
                logger.warning(
                    "could not find the hash session ... creating new session",
                    extra={"warn": e},
                )
                self.context_user_feature_map.update({session_id: {}})
                self.context_hash_session.update(
                    {session_id: self.watcher_service.create_hash_session()}
                )

                user_feature_map_for_context = self.context_user_feature_map[session_id]
                context_hash_session = self.context_hash_session[session_id]

            (
                reference_user_dict,
                reference_features,
            ) = self.watcher_service.download_reference_objects(context_id=context_id)
            (
                rec_users_dict,
                related_words,
                suggested_user_list,
            ) = self.watcher_service.get_recommended_watchers(
                input_query_list=keyphrase_list,
                input_kw_query=keyphrase_list,
                segment_obj=segment_object,
                user_feature_map=user_feature_map_for_context,
                hash_result=None,
                segment_user_ids=segment_user_ids,
                reference_user_meta_dict=reference_user_dict,
                hash_session_object=context_hash_session,
                user_vector_data=reference_features,
            )
            rec_users = list(rec_users_dict.keys())
            watcher_response = {"recommendedWatchers": suggested_user_list}

            end = timer()
            logger.info(
                "Recommended watchers computed",
                extra={
                    "recWatchers": rec_users,
                    "suggestedWatchers": suggested_user_list,
                    "relatedWords": related_words,
                    "instanceId": request["instanceId"],
                    "segmentsReceived": segment_ids,
                    "responseTime": end - start,
                },
            )

            await self.nats_manager.conn.publish(
                msg.reply, json.dumps(watcher_response).encode()
            )

            # Logic for posting to slack
            # if context_id in self.whitelist_contexts:
            #     self.watcher_service.prepare_slack_validation(
            #         req_data=request,
            #         user_dict=rec_users_dict,
            #         word_list=related_words,
            #         suggested_users=suggested_user_list,
            #         segment_users=segment_user_ids,
            #         upload=True,
            #     )
        except Exception as e:
            logger.error(
                "Error computing recommended watchers", extra={"err": e},
            )
            print(traceback.print_exc())
            raise

    async def get_meetings(self, msg):
        msg_data = json.loads(msg.data)
        keyphrase_list = msg_data["keyphrases"]

        try:
            self.meeting_service.recommend_meetings(kw_list=keyphrase_list)
        except Exception:
            raise

        pass
