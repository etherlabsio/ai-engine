import logging
import os
import json
from timeit import default_timer as timer

from knowledge_graph.graphio import GraphTransforms, GraphIO
from knowledge_graph.etl import GraphETL
from knowledge_graph.graphml2csv import GraphML2CSV

from log.logger import setup_server_logger
from s3client.s3 import S3Manager

logger = logging.getLogger()
setup_server_logger(debug=True)

bucket_store = os.getenv("STORAGE_BUCKET", "io.etherlabs.staging2.contexts")

s3_client = S3Manager(bucket_name=bucket_store)
gio = GraphIO(s3_client=s3_client)
gtransform = GraphTransforms()
neptune_obj = GraphML2CSV()

etl_obj = GraphETL(
    s3_client=s3_client,
    graph_io_obj=gio,
    graph_transform_obj=gtransform,
    neptune_util_obj=neptune_obj,
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

        if context_id == "__all__":
            etl_obj.process_all_objects(context_id_prefix="01")
        else:
            etl_obj.process_graph_object(context_id=context_id)

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
