import logging
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


class S3IO(object):
    # S3 storage utility functions
    def __init__(self, s3_client, graph_utils_obj):
        self.s3_client = s3_client
        self.gutils = graph_utils_obj

    def upload_s3(self, graph_obj, context_id, instance_id, s3_dir):
        graph_id = graph_obj.graph.get("graphId")

        if graph_id == context_id + ":" + instance_id:
            serialized_graph_string = self.gutils.write_to_pickle(graph_obj=graph_obj)
            s3_key = context_id + s3_dir + graph_id + ".pickle"

            resp = self.s3_client.upload_object(
                body=serialized_graph_string, s3_key=s3_key
            )
            if resp:
                return True
            else:
                return False
        else:
            logger.error(
                "graphId and context info not matching",
                extra={
                    "graphId": graph_id,
                    "contextInfo": context_id + ":" + instance_id,
                },
            )
            return False

    def download_s3(self, context_id, instance_id, s3_dir):
        start = timer()

        graph_id = context_id + ":" + instance_id
        s3_path = context_id + s3_dir + graph_id + ".pickle"

        file_obj = self.s3_client.download_file(file_name=s3_path)
        file_obj_bytestring = file_obj["Body"].read()

        graph_obj = self.gutils.load_graph_from_pickle(byte_string=file_obj_bytestring)

        end = timer()
        logger.info(
            "Downloaded graph object from s3",
            extra={
                "graphId": graph_obj.graph.get("graphId"),
                "kgNodes": graph_obj.number_of_nodes(),
                "kgEdges": graph_obj.number_of_edges(),
                "responseTime": end - start,
            },
        )

        return graph_obj
