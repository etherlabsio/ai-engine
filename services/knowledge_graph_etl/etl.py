import logging
import json as js
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class GraphETL(object):
    def __init__(
        self,
        s3_client=None,
        graph_io_obj=None,
        graph_transform_obj=None,
        neptune_util_obj=None,
        dgraph_etl_obj=None,
    ):
        self.s3_client = s3_client
        self.gio = graph_io_obj
        self.gtransform = graph_transform_obj
        self.dg_etl = dgraph_etl_obj
        self.context_instance_dir = "/context-instance-graphs/"
        self.context_dir = "/context-graphs/"

        self.neptune_util = neptune_util_obj
        self.delimiter = ","
        self.encoding = "utf-8"

        self.sink = "dgraph"
        self.download_loc = Path(
            "/Users/shashank/Workspace/Orgs/Ether/ai-engine"
            + "/tests/knowledge_graph-service/dgraph_prod_archive/"
        ).absolute()

    def process_graph_objects(
        self, context_id=None, context_id_prefix=None, s3_upload_path=None
    ):

        if context_id_prefix is not None:
            context_id_list = self._list_context_id_folders(
                folder_prefix=context_id_prefix
            )

            logger.info(
                "Listing contexts from s3",
                extra={"contextCount": len(context_id_list)},
            )
        else:
            context_id_list = [context_id]

        for i, c_id in enumerate(context_id_list):
            s3_path = c_id + self.context_instance_dir
            bucket_list = self._list_s3_files(context_id=c_id)

            if s3_upload_path is None:
                s3_upload_path = c_id + self.context_dir

            logger.info(
                "Processing graph objects in s3 Key",
                extra={
                    "processedCount": i + 1,
                    "totalCount": len(context_id_list),
                    "remainingKeys": len(context_id_list) - i,
                    "fileCount": len(bucket_list),
                    "s3Path": s3_path,
                    "contextId": c_id,
                },
            )

            try:
                self._perform_etl(
                    etl_dest=self.sink,
                    bucket_list=bucket_list,
                    s3_path=s3_path,
                    s3_upload_path=s3_upload_path,
                )

            except Exception as e:
                logger.error(
                    "Error processing graph objects",
                    extra={"err": e, "contextId": c_id},
                )
                raise

    def _perform_etl(self, etl_dest, bucket_list, s3_path, s3_upload_path):

        for graph_obj_file in bucket_list:
            s3_obj_path = s3_path + graph_obj_file
            graph_file_name = graph_obj_file.split(".")[0]
            file_format = graph_obj_file.split(".")[-1]

            if (
                file_format == "graphml"
                or file_format == "csv"
                or file_format == "npz"
                or file_format == "json"
            ):
                continue
            else:
                if etl_dest == "dgraph":
                    self._dgraph_etl(
                        graph_file_name=graph_file_name,
                        s3_path=s3_obj_path,
                        s3_upload_path=s3_upload_path,
                    )
                elif etl_dest == "neptune":
                    self._neptune_etl(
                        graph_file_name=graph_file_name,
                        s3_path=s3_obj_path,
                        s3_upload_path=s3_upload_path,
                    )

    def _list_context_id_folders(self, folder_prefix):
        context_id_list = self.s3_client.list_toplevel_folders(
            folder_prefix=folder_prefix
        )

        return context_id_list

    def _list_s3_files(self, context_id):
        s3_path = context_id + self.context_instance_dir
        bucket_list = self.s3_client.get_s3_results(dir_name=s3_path)

        return bucket_list

    def _neptune_etl(self, graph_file_name, s3_path, s3_upload_path):

        # Download and process graph object
        graph_obj = self.gio.extract_process_graph(s3_path=s3_path)

        # Convert pickled object to GraphML
        graphml_file = self.gio.convert_to_graphml(
            graph_obj=graph_obj, output_filename=graph_file_name
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
                s3_path=s3_upload_path + "GraphMLData/" + transformed_graphml_filename,
            )

        # Upload CSVs to s3
        with open(node_file, "rb") as nf, open(edge_file, "rb") as ef:
            self.gio.upload_s3(file_name=nf, s3_path=s3_upload_path + node_file)
            self.gio.upload_s3(file_name=ef, s3_path=s3_upload_path + edge_file)

    def _dgraph_etl(self, graph_file_name, s3_path, s3_upload_path):

        # Download and process graph object
        graph_obj = self.gio.extract_process_graph(s3_path=s3_path)

        # Convert NX graph to Dgraph-compatible JSON
        dgraph_compat_data = self.dg_etl.nx_dgraph(g=graph_obj)

        with open(graph_file_name + ".json", "w", encoding="utf-8") as f_:
            js.dump(dgraph_compat_data, f_, ensure_ascii=False, indent=4)

        # Upload the JSON file to s3 under context bucket
        with open(graph_file_name + ".json", "rb") as js_string:
            self.gio.upload_s3(
                file_name=js_string,
                s3_path=s3_upload_path + "DgraphData/" + graph_file_name + ".json",
            )

        self.s3_client.download_file(
            file_name=s3_upload_path + "DgraphData/" + graph_file_name + ".json",
            download_dir=str(self.download_loc),
        )

        logger.info(
            "processed graph file",
            extra={
                "graphName": graph_file_name,
                "nodes": graph_obj.number_of_nodes(),
                "edges": graph_obj.number_of_edges(),
            },
        )
