import logging
from timeit import default_timer as timer
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class S3IO(object):
    # S3 storage utility functions
    def __init__(self, s3_client, graph_utils_obj, utils):
        self.s3_client = s3_client
        self.gutils = graph_utils_obj
        self.utils = utils

    def upload_s3(
        self, graph_obj, context_id, instance_id, s3_dir, file_format=".pickle"
    ):
        graph_id = graph_obj.graph.get("graphId")

        if graph_id == context_id + ":" + instance_id:
            serialized_graph_string = self.gutils.write_to_pickle(graph_obj=graph_obj)
            s3_key = context_id + s3_dir + graph_id + file_format

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

    def download_s3(self, context_id, instance_id, s3_dir, file_format=".pickle"):
        start = timer()

        graph_id = context_id + ":" + instance_id
        s3_path = context_id + s3_dir + graph_id + file_format

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

    def upload_npz(self, context_id, instance_id, s3_dir, npz_file_name):
        s3_path = context_id + s3_dir + instance_id + "/" + npz_file_name
        self.s3_client.upload_to_s3(file_name=npz_file_name, object_name=s3_path)

        # Once uploading is successful, check if NPZ exists on disk and delete it
        local_npz_path = Path(npz_file_name).absolute()
        logger.debug("local path", extra={
            "npz_local_path": local_npz_path,
            "npz_s3_path": s3_path
        })
        if os.path.exists(local_npz_path):
            os.remove(local_npz_path)

        return s3_path

    def download_npz(self, npz_file_path):
        npz_file_obj = self.s3_client.download_file(file_name=npz_file_path)
        npz_file_string = npz_file_obj["Body"].read()

        npz_file = self.utils.deserialize_from_npz(npz_file_string)
        return npz_file
