import logging

logger = logging.getLogger(__name__)


class GraphETL(object):
    def __init__(
        self,
        s3_client=None,
        graph_io_obj=None,
        graph_transform_obj=None,
        s3_bucket=None,
    ):
        self.s3_client = s3_client
        self.gio = graph_io_obj
        self.gtransform = graph_transform_obj
        self.context_instance_dir = "/context-instance-graphs/"
        self.context_dir = "/context-graphs/"
        self.s3_bucket = s3_bucket

    def list_s3_files(self, context_id):
        s3_path = context_id + self.context_instance_dir
        bucket_list = self.s3_client.get_s3_results(dir_name=s3_path)

        return bucket_list

    def process_graph_object(self, context_id):
        s3_path = context_id + self.context_instance_dir
        s3_upload_path = context_id + self.context_dir

        bucket_list = self.list_s3_files(context_id=context_id)
        logger.info(
            "Listing graph objects from s3",
            extra={
                "fileCount": len(bucket_list),
                "fileList": bucket_list,
                "s3Path": s3_path,
            },
        )

        for graph_obj_file in bucket_list:
            s3_obj_path = s3_path + graph_obj_file
            graph_obj_string = self.gio.download_s3(s3_path=s3_obj_path)
            graphml_file = self.gio.convert_pickle_to_graphml(
                graph_pickle=graph_obj_string
            )
            transformed_graphml_file = self.gtransform.graphml_transformer(
                graphml_file=graphml_file
            )

            # Upload the modified GraphML file to s3 under context bucket
            self.gio.upload_to_s3(
                file_name=transformed_graphml_file, s3_path=s3_upload_path
            )

            logger.info(
                "Uploaded GraphML file to s3",
                extra={"fileName": transformed_graphml_file, "location": s3_path},
            )
