import logging
import traceback

logger = logging.getLogger(__name__)


class GraphETL(object):
    def __init__(
        self,
        s3_client=None,
        graph_io_obj=None,
        graph_transform_obj=None,
        neptune_util_obj=None,
        staging2_s3_client=None,
    ):
        self.s3_client = s3_client
        self.staging2_s3_client = staging2_s3_client
        self.gio = graph_io_obj
        self.gtransform = graph_transform_obj
        self.context_instance_dir = "/context-instance-graphs/"
        self.context_dir = "/context-graphs/"

        self.neptune_util = neptune_util_obj
        self.delimiter = ","
        self.encoding = "utf-8"

    def list_context_id_folders(self, folder_prefix):
        context_id_list = self.s3_client.list_toplevel_folders(
            folder_prefix=folder_prefix
        )

        return context_id_list

    def list_s3_files(self, context_id):
        s3_path = context_id + self.context_instance_dir
        bucket_list = self.s3_client.get_s3_results(dir_name=s3_path)

        return bucket_list

    def process_graph_object(self, context_id, s3_upload_path=None):
        s3_path = context_id + self.context_instance_dir

        if s3_upload_path is None:
            s3_upload_path = context_id + self.context_dir

        bucket_list = self.list_s3_files(context_id=context_id)
        logger.info(
            "Listing graph objects from s3",
            extra={
                "fileCount": len(bucket_list),
                "fileNames": bucket_list,
                "s3Path": s3_path,
            },
        )

        self.perform_etl(
            bucket_list=bucket_list, s3_path=s3_path, s3_upload_path=s3_upload_path
        )

    def process_all_objects(self, context_id_prefix, s3_upload_path=None):
        context_id_list = self.list_context_id_folders(folder_prefix=context_id_prefix)

        logger.info(
            "Listing contexts from s3", extra={"contextCount": len(context_id_list)}
        )
        for i, context_id in enumerate(context_id_list):
            s3_path = context_id + self.context_instance_dir

            if s3_upload_path is None:
                s3_upload_path = context_id + self.context_dir

            bucket_list = self.list_s3_files(context_id=context_id)

            logger.info(
                "Processing graph objects from s3",
                extra={"fileCount": len(bucket_list), "s3Path": s3_path},
            )

            self.perform_etl(
                bucket_list=bucket_list, s3_path=s3_path, s3_upload_path=s3_upload_path
            )

            logger.info(
                "Processed graph objects in s3 Key",
                extra={
                    "processedCount": i,
                    "totalCount": len(context_id_list),
                    "remainingKeys": len(context_id_list) - i,
                    "contextId": context_id,
                },
            )

    def perform_etl(self, bucket_list, s3_path, s3_upload_path):
        for graph_obj_file in bucket_list:

            s3_obj_path = s3_path + graph_obj_file
            graph_file_name = graph_obj_file.split(".")[0]
            file_format = graph_obj_file.split(".")[-1]

            if file_format == "graphml" or file_format == "csv":
                logger.warning(
                    "Unsupported file format. Only .pickle and .pkl are currently supported",
                    extra={"fileName": graph_obj_file},
                )
                continue
            try:
                # Download graph pickle object
                graph_obj_string = self.gio.download_s3(s3_path=s3_obj_path)

                # Convert pickled object to GraphML
                graphml_file = self.gio.convert_pickle_to_graphml(
                    graph_pickle=graph_obj_string, output_filename=graph_file_name
                )

                # Modify GraphML to support Neptune
                transformed_graphml_filename = self.gtransform.graphml_transformer(
                    graphml_file=graphml_file
                )

                # Convert GraphML file to CSV
                node_file, edge_file = self.neptune_util.graphml_to_csv(
                    transformed_graphml_filename,
                    outfname=transformed_graphml_filename,
                    delimiter=self.delimiter,
                    encoding=self.encoding,
                )

                # Upload the modified GraphML file to s3 under context bucket
                with open(transformed_graphml_filename, "rb") as graphml_string:
                    self.gio.upload_s3(
                        file_name=graphml_string,
                        s3_path=s3_upload_path
                        + "GraphMLData/"
                        + transformed_graphml_filename,
                        s3_client=self.staging2_s3_client,
                    )

                # Upload CSVs to s3
                with open(node_file, "rb") as nf, open(edge_file, "rb") as ef:
                    self.gio.upload_s3(
                        file_name=nf,
                        s3_path=s3_upload_path + node_file,
                        s3_client=self.staging2_s3_client,
                    )
                    self.gio.upload_s3(
                        file_name=ef,
                        s3_path=s3_upload_path + edge_file,
                        s3_client=self.staging2_s3_client,
                    )
                    logger.info(
                        "Uploaded CSV file to s3",
                        extra={
                            "graphMLFile": transformed_graphml_filename,
                            "location": s3_path,
                            "nodeCSV": node_file,
                            "edgeCSV": edge_file,
                        },
                    )
            except Exception as e:
                logger.error(
                    "Error uploading file",
                    extra={"err": e, "stack": traceback.print_exc()},
                )
