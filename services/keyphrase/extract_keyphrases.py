import networkx as nx
import logging
from timeit import default_timer as timer
import traceback
from typing import List, Dict, Tuple, Text
import numpy as np
from fuzzywuzzy import fuzz, process

from graphrank.core import GraphRank
from graphrank.utils import TextPreprocess, GraphUtils

from .utils import KeyphraseUtils
from .ranker import KeyphraseRanker
from .s3io import S3IO
from .word_graph import WordGraphBuilder
from .queries import Queries
from .objects import Score, Keyphrase, Entity, Phrase, Segment, Request

logger = logging.getLogger(__name__)

SegmentType = List[Segment]
PhraseType = List[Phrase]


class KeyphraseExtractor(object):
    def __init__(
        self,
        s3_client=None,
        encoder_lambda_client=None,
        lambda_function=None,
        ner_lambda_function=None,
        nats_manager=None,
    ):
        self.context_dir = "/context-instance-graphs/"
        self.feature_dir = "/sessions/"
        self.s3_client = s3_client
        self.utils = KeyphraseUtils()
        self.io_util = S3IO(
            s3_client=s3_client, graph_utils_obj=GraphUtils(), utils=KeyphraseUtils(),
        )
        self.query_client = Queries(nats_manager=nats_manager)
        self.ranker = KeyphraseRanker(
            encoder_lambda_client=encoder_lambda_client,
            lambda_function=lambda_function,
            s3_io_util=self.io_util,
            context_dir=self.context_dir,
            utils=KeyphraseUtils(),
            query_client=self.query_client,
        )

        self.wg = WordGraphBuilder(
            graphrank_obj=GraphRank(),
            textpreprocess_obj=TextPreprocess(),
            graphutils_obj=GraphUtils(),
            keyphrase_utils_obj=self.utils,
            lambda_client=encoder_lambda_client,
            ner_lambda_function=ner_lambda_function,
        )

        self.syntactic_filter = [
            "JJ",
            "JJR",
            "JJS",
            "NN",
            "NNP",
            "NNS",
            "VB",
            "VBP",
            "NNPS",
            "FW",
        ]
        self.meeting_keywords = []
        self.phrase_schema = Phrase.schema()

    def get_graph_id(self, req_data: Request) -> Text:
        context_id = req_data.contextId
        instance_id = req_data.instanceId
        graph_id = context_id + ":" + instance_id
        return graph_id

    def wake_up_lambda(self, req_data: Request):
        # Start the encoder lambda to avoid cold start problem
        logger.info("Invoking lambda to reduce cold-start ...")
        test_segment = ["Wake up Sesame!"]
        self.ranker.get_embeddings(input_list=test_segment, req_data=req_data)
        self.wg.call_custom_ner(input_segment="<IGN>")

    def initialize_meeting_graph(self, req_data: Request):
        graph_id = self.get_graph_id(req_data=req_data)
        context_id = req_data.contextId
        instance_id = req_data.instanceId

        meeting_word_graph = nx.Graph(graphId=graph_id)

        logger.info(
            "Meeting word graph intialized",
            extra={"currentGraphId": meeting_word_graph.graph.get("graphId")},
        )
        logger.info("Uploading serialized graph object")
        self.io_util.upload_s3(
            graph_obj=meeting_word_graph,
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
        )

        # Start the encoder lambda to avoid cold start problem
        self.wake_up_lambda(req_data=req_data)

    def _retrieve_word_graph(self, req_data: Request) -> nx.Graph:
        """
        Download meeting word graph from s3
        Args:
            req_data:

        Returns:

        """

        context_id = req_data.contextId
        instance_id = req_data.instanceId

        # Get graph object from S3
        meeting_word_graph = self.io_util.download_s3(
            context_id=context_id, instance_id=instance_id, s3_dir=self.context_dir
        )

        return meeting_word_graph

    def _update_word_graph(
        self, req_data: Request, meeting_word_graph: nx.Graph, **kwargs
    ) -> nx.Graph:
        """
        Populate instance information, add meeting word graph to context graph and upload the context graph
        Args:
            req_data:
            context_graph:
            meeting_word_graph:

        Returns:

        """

        context_id = req_data.contextId
        instance_id = req_data.instanceId

        # Write back the graph object to S3
        self.io_util.upload_s3(
            graph_obj=meeting_word_graph,
            context_id=context_id,
            instance_id=instance_id,
            s3_dir=self.context_dir,
        )

        return meeting_word_graph

    def populate_and_embed_graph(
        self, req_data: Request, segment_object: SegmentType, **kwargs
    ) -> Tuple[Request, nx.Graph]:
        meeting_word_graph = self.populate_word_graph(
            req_data=req_data, segment_object=segment_object
        )

        # Compute embeddings for segments and keyphrases
        modified_request_obj = self.compute_embeddings(
            req_data=req_data,
            segment_object=segment_object,
            meeting_word_graph=meeting_word_graph,
            **kwargs,
        )

        return modified_request_obj, meeting_word_graph

    def populate_word_graph(
        self, req_data: Request, segment_object: SegmentType
    ) -> nx.Graph:
        start = timer()
        # Get graph objects
        meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

        # Populate word graph for the current instance
        try:
            text_list = self.utils.read_segments(segment_object=segment_object)
            meeting_word_graph = self.wg.build_custom_graph(
                text_list=text_list, graph=meeting_word_graph
            )

            end = timer()
            logger.info(
                "Populated graph and written to s3",
                extra={
                    "graphId": meeting_word_graph.graph.get("graphId"),
                    "nodes": meeting_word_graph.number_of_nodes(),
                    "edges": meeting_word_graph.number_of_edges(),
                    "instanceId": req_data.instanceId,
                    "responseTime": end - start,
                },
            )

        except Exception as e:
            end = timer()
            logger.error(
                "Error populating graph",
                extra={
                    "err": e,
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                },
            )

        # Push updated meeting graph
        meeting_word_graph = self._update_word_graph(
            req_data=req_data, meeting_word_graph=meeting_word_graph
        )

        return meeting_word_graph

    def compute_embeddings(
        self,
        req_data: Request,
        segment_object: SegmentType,
        meeting_word_graph: nx.Graph = None,
        default_form: Text = "descriptive",
        **kwargs,
    ) -> Request:
        """
        Compute embedding vectors for segments and segment-keyphrases and store them as node attributes in the knowledge
        graph.
        Args:
            default_form:
            segment_object:
            meeting_word_graph:
            req_data:

        Returns:

        """
        context_id = req_data.contextId
        instance_id = req_data.instanceId

        populate_graph = req_data.populateGraph
        group_id = kwargs.get("group_id", None)

        if meeting_word_graph is None:
            # Get graph objects
            meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

        phrase_object_list = self.extract_keywords(
            segment_object=segment_object, meeting_word_graph=meeting_word_graph,
        )

        # Get segment text
        for i, phrase_object in enumerate(phrase_object_list):
            seg_text = segment_object[i].originalText
            segment_id = phrase_object.segmentId

            segment_keyphrase_obj = [
                kp_obj
                for kp_obj in phrase_object.keyphrases
                if kp_obj.type == default_form
            ]
            segment_entity_obj = phrase_object.entities

            keyphrase_list = [kp_obj.originalForm for kp_obj in segment_keyphrase_obj]
            entities_list = [ent_obj.originalForm for ent_obj in segment_entity_obj]

            input_phrases_list = entities_list
            input_phrases_list.extend(keyphrase_list)

            # Compute segment embedding vector
            segment_embedding = self.ranker.get_embeddings(
                input_list=[seg_text], req_data=req_data
            )

            # Compute keyphrase embedding vectors
            keyphrase_embeddings = self.ranker.get_embeddings(
                input_list=input_phrases_list, req_data=req_data
            )

            # Combine segment and keyphrase embeddings and serialize them
            f_name = segment_id
            if populate_graph is not True:
                f_name = segment_id + "_" + group_id

            npz_s3_path = self._form_update_embeddings(
                context_id=context_id,
                instance_id=instance_id,
                f_name=f_name,
                segment_embedding=segment_embedding,
                keyphrase_embeddings=keyphrase_embeddings,
                input_phrases_list=input_phrases_list,
            )

            # # Update context graph with embedding vectors
            attributed_segment_obj = self._form_segment_attr_object(
                segment=segment_object[i],
                populate_graph=populate_graph,
                npz_file_path=npz_s3_path,
                group_id=group_id,
            )

            attributed_segment_obj = self._form_keyphrase_attr_object(
                segment=attributed_segment_obj,
                keyphrase_object=segment_keyphrase_obj,
                entity_object=segment_entity_obj,
            )

            req_data.segments[i] = attributed_segment_obj

            logger.info(
                "features embeddings computed and stored",
                extra={"embeddingUri": npz_s3_path},
            )

        modified_request_object = req_data

        return modified_request_object

    def _form_update_embeddings(
        self,
        context_id: Text,
        instance_id: Text,
        f_name: Text,
        segment_embedding: np.ndarray,
        keyphrase_embeddings: List[np.ndarray],
        input_phrases_list: List[Text],
    ) -> Text:
        # Combine segment and keyphrase embeddings and serialize them
        segment_embedding_dict = {f_name: np.array(segment_embedding)}

        (
            phrase_hash_dict,
            phrase_embedding_dict,
        ) = self.utils.map_embeddings_to_phrase(
            phrase_list=input_phrases_list, embedding_list=keyphrase_embeddings,
        )

        segment_keyphrase_embeddings = {
            **segment_embedding_dict,
            **phrase_embedding_dict,
        }

        # Serialize the entire segment-keyphrase embedding dictionary to NPZ
        npz_file_name = self.utils.serialize_to_npz(
            embedding_dict=segment_keyphrase_embeddings, file_name=f_name
        )
        npz_s3_path = self.io_util.upload_npz(
            context_id=context_id,
            instance_id=instance_id,
            feature_dir=self.feature_dir,
            npz_file_name=npz_file_name,
        )

        return npz_s3_path

    def _form_segment_attr_object(
        self,
        segment: Segment,
        populate_graph: bool,
        npz_file_path: Text,
        group_id: Text,
    ) -> Segment:

        # Update context graph with embedding vectors
        segment_attr_dict = {
            "text": segment.originalText,
            "embedding_vector_uri": npz_file_path,
            "embedding_model": "USE_v1",
            "highlight": False,
        }
        if not populate_graph:
            segment_attr_dict = {
                "text": segment.originalText,
                "embedding_vector_group_uri": npz_file_path,
                "groupId": group_id,
                "embedding_model": "USE_v1",
                "highlight": True,
            }

        segment.attributes = {**segment_attr_dict}

        return segment

    def _form_keyphrase_attr_object(
        self,
        segment: Segment,
        keyphrase_object: List[Keyphrase],
        entity_object: List[Entity],
    ) -> Segment:

        segment.keyphrases = keyphrase_object
        segment.entities = entity_object

        return segment

    def encode_word_graph(self, word_graph: nx.Graph) -> nx.Graph:
        word_graph = self.ranker.compute_edge_weights(word_graph)

        return word_graph

    async def get_keyphrases(
        self,
        req_data: Request,
        segment_object: SegmentType,
        meeting_word_graph: nx.Graph = None,
        n_kw: int = 10,
        default_form: Text = "descriptive",
        rank_by: Text = "norm_boosted_sim",
        sort_by: Text = "loc",
        validate: bool = False,
        highlight: bool = False,
        group_id: str = None,
        **kwargs,
    ) -> Dict:
        start = timer()

        context_id = req_data.contextId
        instance_id = req_data.instanceId

        # Use startTime of the first segment as relative starting point
        relative_time = self.utils.formatTime(
            segment_object[0].startTime, datetime_object=True
        )

        try:
            if meeting_word_graph is None:
                # Get graph objects
                meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

            logger.info("Adding segments before extracting keyphrases")
            # Repopulate the graphs
            modified_request_obj, meeting_word_graph = self.populate_and_embed_graph(
                req_data=req_data, segment_object=segment_object, **kwargs
            )

            phrase_object_list = self.extract_keywords(
                segment_object=segment_object,
                meeting_word_graph=meeting_word_graph,
                relative_time=relative_time,
            )

            try:
                ranked_phrase_object_list = await self.ranker.compute_relevance(
                    phrase_object_list=phrase_object_list,
                    highlight=highlight,
                    group_id=group_id,
                )

                keyphrases, phrase_object = self.prepare_keyphrase_output(
                    phrase_object=ranked_phrase_object_list,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by=rank_by,
                    sort_by=sort_by,
                    remove_phrases=True,
                )
            except Exception as e:
                logger.warning(
                    "Error computing keyphrase relevance",
                    extra={"warnMsg": e, "trace": traceback.print_exc()},
                )

                keyphrases, phrase_object = self.prepare_keyphrase_output(
                    phrase_object=phrase_object_list,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by="pagerank",
                    sort_by=sort_by,
                    remove_phrases=False,
                )

            # validation_id = "test"
            # validation_data = self.phrase_schema.dump(phrase_object, many=True)
            # validation_file_name = self.utils.write_to_json(
            #     validation_data, file_name="keyphrase_validation_" + validation_id
            # )

            if validate:
                self._make_validation(
                    phrase_object=phrase_object,
                    context_id=context_id,
                    instance_id=instance_id,
                )

            logger.debug(
                "keyphrases extracted successfully",
                extra={"result": keyphrases, "output": phrase_object},
            )

            result = {"keyphrases": keyphrases}
            return result

        except Exception as e:
            end = timer()
            logger.error(
                "Error extracting keyphrases from segment",
                extra={
                    "responseTime": end - start,
                    "instanceId": req_data["instanceId"],
                    "segmentsReceived": [seg_id["id"] for seg_id in segment_object],
                    "err": traceback.print_exc(),
                    "errMsg": e,
                },
            )
            raise

    def _make_validation(self, phrase_object: PhraseType, context_id, instance_id):
        validation_id = self.utils.hash_sha_object()

        # Convert Class object to python List[Dict]
        validation_data = self.phrase_schema.dump(phrase_object, many=True)
        validation_file_name = self.utils.write_to_json(
            validation_data, file_name="keyphrase_validation_" + validation_id
        )
        self.io_util.upload_validation(
            context_id=context_id,
            feature_dir=self.feature_dir,
            instance_id=instance_id,
            validation_file_name=validation_file_name,
        )

    async def get_keyphrases_with_offset(
        self,
        req_data: Request,
        segment_object: SegmentType,
        meeting_word_graph: nx.Graph = None,
        n_kw: int = 10,
        default_form: Text = "descriptive",
        rank_by: Text = "segment_relevance",
        sort_by: Text = "loc",
        validate: bool = False,
        highlight: bool = False,
        **kwargs,
    ):
        start = timer()
        keyphrase_offsets = []
        context_id = req_data.contextId
        instance_id = req_data.instanceId

        group_id = kwargs.get("group_id")

        relative_time = self.utils.formatTime(
            req_data.relativeTime, datetime_object=True
        )
        try:
            if meeting_word_graph is None:
                # Get graph objects
                meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

            logger.info("Adding segments before extracting keyphrases")
            # Repopulate the graphs
            modified_request_obj, meeting_word_graph = self.populate_and_embed_graph(
                req_data=req_data, segment_object=segment_object, **kwargs
            )

            phrase_object_list = self.extract_keywords(
                segment_object=segment_object,
                meeting_word_graph=meeting_word_graph,
                relative_time=relative_time,
            )

            try:
                ranked_phrase_object_list = await self.ranker.compute_relevance(
                    phrase_object_list=phrase_object_list,
                    highlight=highlight,
                    group_id=group_id,
                )

                keyphrases, phrase_object = self.prepare_keyphrase_output(
                    phrase_object=ranked_phrase_object_list,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by=rank_by,
                    sort_by=sort_by,
                    remove_phrases=True,
                )
            except Exception as e:
                logger.warning(
                    "Error computing keyphrase relevance",
                    extra={"warnMsg": e, "trace": traceback.print_exc()},
                )

                keyphrases, phrase_object = self.prepare_keyphrase_output(
                    phrase_object=phrase_object_list,
                    top_n=n_kw,
                    default_form=default_form,
                    rank_by="pagerank",
                    sort_by=sort_by,
                    remove_phrases=False,
                )

            if validate:
                self._make_validation(
                    phrase_object=phrase_object,
                    context_id=context_id,
                    instance_id=instance_id,
                )

            keyphrase_offsets = self.parse_keyphrase_offset(
                keyphrase_list=keyphrases, phrase_object=phrase_object
            )

            end = timer()
            logger.debug(
                "Extracted keyphrases with offsets",
                extra={"output": phrase_object, "responseTime": end - start},
            )

        except Exception:
            end = timer()
            logger.error(
                "Error extracting keyphrases from segment",
                extra={
                    "responseTime": end - start,
                    "instanceId": req_data.instanceId,
                    "segmentObj": req_data.segments,
                    "err": traceback.print_exc(),
                },
            )

        result = {"keyphrases": keyphrase_offsets}

        return result

    def parse_keyphrase_offset(
        self, keyphrase_list: List[Text], phrase_object: List[Phrase]
    ):

        keyphrase_offset_list = []
        for i, phrase_obj in enumerate(phrase_object):
            segment = phrase_obj.originalText
            offset_ts = phrase_obj.offset
            for word in keyphrase_list:
                loc = segment.find(word)
                if loc > -1:
                    keyphrase_offset_list.append({word: offset_ts})

        # Reformat as per api contract
        keyphrase_offset_output = {
            words: offset
            for elements in keyphrase_offset_list
            for words, offset in elements.items()
        }

        return keyphrase_offset_output

    def extract_keywords(
        self,
        segment_object: SegmentType,
        meeting_word_graph: nx.Graph,
        relative_time: str = None,
        highlight: bool = False,
        preserve_singlewords=False,
    ):
        """
        Search for keyphrases in an array of N segments and return them as one list of keyphrases
        Args:
            segment_object:
            relative_time:
            meeting_word_graph:
            req_data:
            preserve_singlewords:

        Returns: processed_entities (list) cleaned_keyphrase_list (list[Dict]): Returns list of dictionary -
        {
            "segmentId": str,
            "segments": str,
            "offset": 0.0,
            "original": {word: score},
            "descriptive": {word: score}
        }
        where, score = list(pagerank_score, segment_relevance_score, boosted_score, norm_boosted_score, location)
        """

        phrase_obj_list = []

        for i, segment_obj in enumerate(segment_object):
            input_segment = segment_obj.originalText
            input_segment_id = segment_obj.id

            if relative_time is not None:
                # Set offset time for every keywords
                start_time = segment_obj.startTime
                start_time = self.utils.formatTime(start_time, datetime_object=True)
                offset_time = float((start_time - relative_time).seconds)
            else:
                offset_time = 0.0

            # Get entities
            entity_object_list = self.wg.get_custom_entities(input_segment)

            # Get segment keyphrases
            keyphrase_object_list = self.wg.get_segment_keyphrases(
                segment_text=input_segment, word_graph=meeting_word_graph
            )

            # Get cleaned words
            for keyphrase_object in keyphrase_object_list:
                word = keyphrase_object.originalForm
                pagerank_score = keyphrase_object.score.pagerank
                loc = input_segment.find(word)
                if loc > -1 and ("*" not in word or "." not in word):
                    keyphrase_score_obj = Score(
                        pagerank=pagerank_score, loc=loc + offset_time,
                    )
                    keyphrase_object.score = keyphrase_score_obj

            # Add entity scores in the object
            for entity_object in entity_object_list:
                entity_phrase = entity_object.originalForm
                loc = input_segment.find(entity_phrase)
                loc_small = input_segment.find(entity_phrase.lower())
                if (loc > -1 or loc_small > -1) and ("*" not in entity_phrase):
                    try:
                        entity_pagerank_score = meeting_word_graph.nodes[
                            entity_phrase
                        ].get("pagerank")
                    except Exception:
                        try:
                            entity_pagerank_score = meeting_word_graph.nodes[
                                entity_phrase.lower()
                            ].get("pagerank")
                        except Exception:
                            entity_pagerank_score = 0

                    final_loc = loc if loc > -1 else loc_small

                    entity_score = Score(
                        pagerank=entity_pagerank_score, loc=final_loc + offset_time,
                    )
                    entity_object.score = entity_score

            phrase_obj = Phrase(
                segmentId=input_segment_id,
                originalText=input_segment,
                highlight=highlight,
                offset=offset_time,
                keyphrases=keyphrase_object_list,
                entities=entity_object_list,
            )

            # Post-process phrases
            phrase_obj = self.utils.post_process_output(phrase_object=phrase_obj)

            phrase_obj_list.append(phrase_obj)

        return phrase_obj_list

    def prepare_keyphrase_output(
        self,
        phrase_object: PhraseType,
        top_n: int = None,
        default_form: str = "descriptive",
        rank_by: str = "pagerank",
        sort_by: str = "loc",
        remove_phrases: bool = True,
        phrase_threshold: int = 3,
    ) -> Tuple[List[Text], PhraseType]:

        for i, phrase_obj in enumerate(phrase_object):
            keyphrase_obj = [
                kp_obj
                for kp_obj in phrase_obj.keyphrases
                if kp_obj.type == default_form
            ]
            entity_obj = phrase_obj.entities

            try:
                entity_quality_score = phrase_obj.entitiesQuality
                keyphrase_quality_score = phrase_obj.medianKeyphraseQuality
            except KeyError as e:
                logger.warning(
                    "Unable to compute entities and keyphrase quality score",
                    extra={"warnMsg": e},
                )
                entity_quality_score = 0
                keyphrase_quality_score = 0

            # # Sort by rank/scores
            # For chapters: Choose top-n from each segment for better diversity
            (
                ranked_entities_object,
                ranked_keyphrase_object,
            ) = self.utils.limit_phrase_list(
                entities_object=entity_obj,
                keyphrase_object=keyphrase_obj,
                phrase_limit=top_n,
                entities_limit=5,
                entity_quality_score=entity_quality_score,
                keyphrase_quality_score=keyphrase_quality_score,
                rank_by=rank_by,
                sort_by=sort_by,
                remove_phrases=False,
                final_sort=False,
            )

            phrase_obj.keyphrases = ranked_keyphrase_object
            phrase_obj.entities = ranked_entities_object

        entities_object = [
            ent_obj for phrase_obj in phrase_object for ent_obj in phrase_obj.entities
        ]
        keyphrase_object = [
            kp_obj for phrase_obj in phrase_object for kp_obj in phrase_obj.keyphrases
        ]
        logger.debug(
            "Keyphrase and entity list before limiting",
            extra={
                "entities": entities_object,
                "keyphrases": keyphrase_object,
                "count": len(entities_object) + len(keyphrase_object),
            },
        )

        # Set a dynamic threshold for top-n number
        total_phrase_count = len(entities_object) + len(keyphrase_object)
        dynamic_top_n = int(total_phrase_count / phrase_threshold)
        if dynamic_top_n <= top_n:
            dynamic_top_n = top_n

        if top_n == 10 or top_n == 5:
            dynamic_top_n = top_n

        if dynamic_top_n >= 10:
            dynamic_top_n = 10

        try:
            overall_entity_quality_score = np.mean(
                [entity_obj.score.norm_boosted_sim for entity_obj in entities_object]
            )
            overall_keyphrase_quality_score = np.mean(
                [kp_obj.score.norm_boosted_sim for kp_obj in keyphrase_object]
            )
        except Exception as e:
            logger.warning(
                "Unable to compute overall quality scores", extra={"warnMsg": e},
            )

            overall_entity_quality_score = 0
            overall_keyphrase_quality_score = 0

        # Limit keyphrase list to top-n
        if top_n == 10:
            remove_phrases = False

        sorted_keyphrase_dict = self.utils.limit_phrase_list(
            entities_object=entities_object,
            keyphrase_object=keyphrase_object,
            phrase_limit=dynamic_top_n,
            entities_limit=5,
            entity_quality_score=overall_entity_quality_score,
            keyphrase_quality_score=overall_keyphrase_quality_score,
            rank_by=rank_by,
            sort_by=sort_by,
            remove_phrases=remove_phrases,
            final_sort=True,
        )

        logger.debug(
            "Keyphrase and entity list after limiting",
            extra={
                "keyphrases": list(sorted_keyphrase_dict.keys()),
                "count": len(sorted_keyphrase_dict.keys()),
                "threshold": dynamic_top_n,
                "keyphraseQuality": overall_keyphrase_quality_score,
                "entityQuality": overall_entity_quality_score,
            },
        )
        keyphrase = [phrases for phrases, loc in sorted_keyphrase_dict.items()]
        keyphrase = self._dedupe_phrases(keyphrase)

        return keyphrase, phrase_object

    def _dedupe_phrases(self, keyphrase_list):
        """
        Remove any duplicate phrases arising due to difference in cases.
        Args:
            keyphrase_list:

        Returns:

        """
        deduped_keyphrase_list = list(process.dedupe(keyphrase_list))

        return deduped_keyphrase_list

    def get_instance_keyphrases(self, req_data: Request, n_kw: int = 10) -> Dict:

        segment_object = req_data.segments
        # Get graph objects
        meeting_word_graph = self._retrieve_word_graph(req_data=req_data)

        keyphrase_list, descriptive_kp = self.wg.get_segment_keyphrases(
            segment_text=segment_object[0].originalText, word_graph=meeting_word_graph
        )
        instance_keyphrases = [words for words, score in keyphrase_list]

        result = {"keyphrases": instance_keyphrases[:n_kw]}

        return result

    def reset_keyphrase_graph(self, req_data: Request):
        start = timer()
        context_id = req_data.contextId
        instance_id = req_data.instanceId

        # Download context graph from s3 and remove the word graph object upon reset

        word_graph = self._retrieve_word_graph(req_data=req_data)
        word_graph_id = word_graph.graph.get("graphId")

        # Write it back to s3
        self.io_util.upload_s3(
            graph_obj=word_graph,
            s3_dir=self.context_dir,
            context_id=context_id,
            instance_id=instance_id,
        )

        end = timer()
        logger.info(
            "Post-reset: Graph info",
            extra={
                "deletedGraphId": word_graph_id,
                "nodes": word_graph.number_of_nodes(),
                "edges": word_graph.number_of_edges(),
                "responseTime": end - start,
                "instanceId": req_data["instanceId"],
            },
        )
