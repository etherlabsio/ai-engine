import logging
import os
import json
from timeit import default_timer as timer

from knowledge_graph.graphio import GraphTransforms, GraphIO
from knowledge_graph.etl import GraphETL
from knowledge_graph.graphml2csv import GraphML2CSV
from knowledge_graph.backfill import BackFillCleanupJob


from log.logger import setup_server_logger
from s3client.s3 import S3Manager

logger = logging.getLogger()
setup_server_logger(debug=True)

bucket_store = os.getenv("STORAGE_BUCKET", "io.etherlabs.production.contexts")
s2_bucket_store = os.getenv("S2_STORAGE_BUCKET", "io.etherlabs.staging2.contexts")

neptune_store_dir = "NeptuneData/"
s2_neptune_store_dir = "NeptuneDataS2/"

s3_client = S3Manager(bucket_name=bucket_store, profile_name="default")
staging2_s3_client = S3Manager(bucket_name=s2_bucket_store, profile_name="staging2")

backfill_object = BackFillCleanupJob()
gio = GraphIO(s3_client=s3_client, backfill_obj=backfill_object)
gtransform = GraphTransforms()
neptune_obj = GraphML2CSV()

etl_obj = GraphETL(
    s3_client=s3_client,
    graph_io_obj=gio,
    graph_transform_obj=gtransform,
    neptune_util_obj=neptune_obj,
    staging2_s3_client=staging2_s3_client,
)


def process_input(json_request):
    if isinstance(json_request, str):
        json_request = json.loads(json_request)
    context_id = json_request["contextId"]
    return context_id


def handler(event, context):
    if isinstance(event["body"], str):
        json_request = json.loads(event["body"])
    else:
        json_request = event["body"]

    try:
        start = timer()
        context_id = process_input(json_request=json_request)

        if bucket_store.split(".")[-2] == "staging2":
            upload_path = s2_neptune_store_dir
        else:
            upload_path = neptune_store_dir

        if context_id == "__all__":
            etl_obj.process_all_objects(
                context_id_prefix="01", s3_upload_path=upload_path
            )
        else:
            etl_obj.process_graph_object(
                context_id=context_id, s3_upload_path=upload_path
            )

        response = json.dumps({"contextId": context_id})

        end = timer()
        logger.info(
            "Prepared & processed graphs for Neptune",
            extra={
                "request": json_request,
                "responseTime": end - start,
                "contextId": context_id,
            },
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": response,
        }
    except Exception as e:
        logger.error(
            "Error processing request", extra={"err": e, "request": json_request}
        )
        return {
            "statusCode": 404,
            "headers": {"Content-Type": "application/json"},
            "body": e,
        }
