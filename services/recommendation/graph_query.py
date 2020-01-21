import pydgraph
import json as js
from fuzzywuzzy import process, fuzz
from typing import Tuple, List, Dict
import logging
import pickle
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class GraphQuery(object):
    def __init__(self, dgraph_url: str, dgraph_client=None):
        self.client_stub = pydgraph.DgraphClientStub(dgraph_url)
        self.client = pydgraph.DgraphClient(self.client_stub)
        # self.client = dgraph_client

    def perform_query(self, query, variables):
        txn = self.client.txn()
        try:
            res = self.client.txn(read_only=True).query(query, variables=variables)
            response = js.loads(res.json)

            return response
        finally:
            # Clean up. Calling this after txn.commit() is a no-op and hence safe.
            txn.discard()

    def form_user_keyword_query(self, user_name, top_n_result):
        user_kw_query = """
        query mlChannelUserKw($n: string, $t: int) {
            mlChannelUserKw(func: type("Channel"))
          @filter(eq(name, "ml-ai")) @cascade{
                uid
                xid
                hasContext {
                  xid
                  attribute
                  associatedMind {
                    name
                    type
                  }
                  hasMeeting (first: $t){
                    xid
                    hasSegment {
                      authoredBy @filter(anyofterms(name, $n)) {
                        name
                        xid
                      }
                      hasKeywords {
                        values
                      }
                    }
                  }
                }
           }
        }
        """

        variables = {"$n": user_name, "$t": str(top_n_result)}

        return user_kw_query, variables

    def form_reference_users_query(
        self, context_id: str, top_n_result: int = 10
    ) -> Tuple[str, Dict]:
        ref_user_query = """
        query contextUserInfo($c: string, $t: int) {
          contextUserInfo(func: eq(xid, $c)) @cascade{
            attribute
            xid
            associatedMind {
              xid
            }
          hasMeeting (first: $t) {
            xid
            attribute
            hasSegment {
              attribute
              xid
              text
              authoredBy {
                attribute
                xid
              }
              hasKeywords {
                values
              }
            }
          }
         }
        }
        """
        variables = {"$c": context_id, "$t": str(top_n_result)}

        return ref_user_query, variables


class QueryHandler(object):
    def __init__(self, vectorizer=None, s3_client=None):
        self.vectorizer = vectorizer
        self.s3_client = s3_client
        self.feature_dir = "/features/recommendation/"

    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

        return filename + ".json"

    def format_reference_response(
        self, query_response: Dict, function_name: str = "contextUserInfo"
    ) -> Dict:
        user_dict = {}
        user_kw = []

        for info in query_response[function_name]:
            meeting_obj = info["hasMeeting"]
            for m_info in meeting_obj:
                segment_obj = m_info["hasSegment"]
                for segment_info in segment_obj:
                    segment_kw = []
                    seg_ids = []
                    seg_texts = []
                    try:
                        user_id = segment_info.get("authoredBy")["xid"]
                        # user_name = segment_info.get("authoredBy")["name"]

                        try:
                            u_id = user_dict[user_id]
                        except KeyError:
                            # Intialize
                            user_dict.update(
                                {
                                    user_id: {
                                        # "name": user_name,
                                        "keywords": None,
                                        "text": None,
                                        "segment_id": None,
                                    }
                                }
                            )

                        keyword_object = segment_info["hasKeywords"]
                        segment_kw.extend(
                            list(process.dedupe(keyword_object["values"]))
                        )
                        seg_texts.append(segment_info["text"])
                        seg_ids.append(segment_info["xid"])

                        user_kw_list = user_dict[user_id].get("keywords")
                        user_text_list = user_dict[user_id].get("text")
                        user_id_list = user_dict[user_id].get("segment_id")
                        if user_kw_list is not None:
                            user_kw_list.extend(segment_kw)
                            user_text_list.extend(seg_texts)
                            user_id_list.extend(seg_ids)
                            user_dict[user_id].update(
                                {
                                    "keywords": list(set(user_kw_list)),
                                    "text": user_text_list,
                                    "segment_id": user_id_list,
                                }
                            )
                        else:
                            user_dict[user_id].update(
                                {
                                    "keywords": segment_kw,
                                    "text": seg_texts,
                                    "segment_id": seg_ids,
                                }
                            )
                    except Exception as e:
                        print(e)
                        continue

        return user_dict

    def form_reference_features(
        self,
        reference_user_dict: Dict,
        context_id: str,
        ref_key: str = "keywords",
        write: bool = True,
    ):
        ref_user_info_dict = {
            u: reference_user_dict[u][ref_key] for u in reference_user_dict.keys()
        }

        user_vector_data = {}
        num_features_in_input = {u: 0 for u in ref_user_info_dict.keys()}

        for user, info in ref_user_info_dict.items():
            kw_features = self.vectorizer.get_embeddings(input_list=info)
            user_vector_data.update({user: kw_features})
            num_features_in_input[user] = len(kw_features)

        if write:
            file_name = context_id
            with open(file_name + ".pickle", "wb") as f_:
                pickle.dump(user_vector_data, f_)

            features_s3_path = self._upload_file(
                context_id=context_id,
                feature_dir=self.feature_dir,
                file_name=file_name + ".pickle",
            )
            reference_user_json_name = self.to_json(
                reference_user_dict, filename=context_id
            )
            reference_user_json_path = self._upload_file(
                context_id=context_id,
                feature_dir=self.feature_dir,
                file_name=reference_user_json_name,
            )

            logger.info(
                "Reference user features formed & uploaded",
                extra={
                    "contextId": context_id,
                    "fileName": context_id,
                    "featuresPath": features_s3_path,
                    "metaPath": reference_user_json_path,
                    "users": list(num_features_in_input.keys()),
                    "numUsers": len(num_features_in_input.keys()),
                    "numFeatures": np.sum(list(num_features_in_input.values())),
                },
            )

            return reference_user_json_path, features_s3_path

    def _upload_file(self, context_id, feature_dir, file_name):
        s3_path = context_id + feature_dir + file_name
        self.s3_client.upload_to_s3(file_name=file_name, object_name=s3_path)

        # Once uploading is successful, check if feature file exists on disk and delete it
        local_path = Path(file_name).absolute()
        if os.path.exists(local_path):
            os.remove(local_path)

        return s3_path
