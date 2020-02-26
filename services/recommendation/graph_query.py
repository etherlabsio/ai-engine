import pydgraph
import json as js
from fuzzywuzzy import process, fuzz
from typing import Tuple, List, Dict
import logging
import pickle
import os
from pathlib import Path
import numpy as np
import traceback

logger = logging.getLogger(__name__)


class GraphQuery(object):
    def __init__(self, dgraph_url: str):
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

    # Tag - v1
    def form_reference_users_query(
        self, context_id: str, top_n_result: int = 10
    ) -> Tuple[str, Dict]:
        logger.info("Using v1 for querying")
        ref_user_query = """
        query contextUserInfo($c: string, $t: int) {
          contextUserInfo(func: eq(xid, $c)) @cascade @normalize {
            attribute
            xid
          hasMeeting(first: $t) {
            xid
            attribute
            hasSegment {
              attribute
              xid
              text: text
              startTime: startTime
              authoredBy {
                user_id: xid
              }
              hasKeywords {
                value: originalForm
              }
            }
          }
         }
        }
        """
        variables = {"$c": context_id, "$t": str(top_n_result)}

        return ref_user_query, variables

    # Tag - v2
    def form_user_contexts_query(self, context_id, top_n_result):
        logger.info("Using v2 for querying")
        alt_query = """
        query userContextKw($c: string, $t: int) {
            var(func: eq(xid, $c)) {
              xid 
              attribute
              ~hasContext {
                xid
                attribute
                hasMember {
                 user as uid
                }
                belongsTo {
                  attribute
                  c_id as uid
                }
              }
            }
        
        
            userContextKw(func: uid(c_id)) @normalize @cascade{
            channels: ~belongsTo @filter(eq(attribute, "channelId") AND NOT anyofterms(name, "test testing vtemp only")) {
            contexts: hasContext {
              context_id: xid
                instances: hasMeeting (first: $t){
                 xid
                segments: hasSegment {
                  segment_id: xid
                  text: text
                  time: startTime
                  authors: authoredBy @filter(uid(user)){
                    user_id: xid
                  }
                  keywords: hasKeywords {
                    value: value
                  }
                }
              }
            }   
            }
          }
        }
        """
        variables = {"$c": context_id, "$t": str(top_n_result)}

        return alt_query, variables

    def get_user_name_query(self, user_id):
        query = """
           query userName($u: string) {
              userName(func: eq(xid, $u)) {
                uid
                xid
                attribute
                name
              }
            }
        """
        variables = {"$u": user_id}
        return query, variables


class QueryHandler(object):
    def __init__(
        self,
        dgraph_url,
        vectorizer=None,
        s3_client=None,
        feature_dir: str = "/features/recommendation/",
    ):
        self.vectorizer = vectorizer
        self.s3_client = s3_client
        self.query_client = GraphQuery(dgraph_url=dgraph_url)
        self.feature_dir = feature_dir

        self.blacklist_context = ["01DBB3SN6EVW6Y4CZ6ETFC9Y9X"]

    def to_json(self, data, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f_:
            js.dump(data, f_, ensure_ascii=False, indent=4)

        return filename + ".json"

    def format_reference_response(
        self, query_response: Dict, function_name: str = "contextUserInfo"
    ):
        user_dict = {}

        try:
            for info in query_response[function_name]:
                segment_kw = []
                seg_texts = []

                text = info["text"]
                keywords = info["value"]
                user_id = info.get("authoredBy")["user_id"]

                try:
                    u_id = user_dict[user_id]
                except KeyError:
                    # Intialize
                    user_dict.update({user_id: {"keywords": None, "text": None,}})

                segment_kw.append(keywords)
                seg_texts.append(text)

                user_kw_list = user_dict[user_id].get("keywords")
                user_text_list = user_dict[user_id].get("text")

                if user_kw_list is not None:
                    user_kw_list.extend(segment_kw)
                    user_text_list.extend(seg_texts)
                    user_dict[user_id].update(
                        {
                            "keywords": list(set(user_kw_list)),
                            "text": list(set(user_text_list)),
                        }
                    )
                else:
                    user_dict[user_id].update(
                        {"keywords": segment_kw, "text": seg_texts,}
                    )
        except Exception as e:
            logger.error("Unable to format response", extra={"err": e})

        return user_dict

    def format_user_contexts_reference_response(
        self, query_response: Dict, function_name: str = "userContextKw"
    ) -> Dict:
        user_dict = {}

        for info in query_response[function_name]:
            context_list = []
            context_object = info["contexts"]
            context_id = context_object["context_id"]

            if context_id in self.blacklist_context:
                continue
            else:
                context_list.append(context_id)

            instance_object = context_object["instances"]

            for instance_info in instance_object:
                segment_object = instance_info["segments"]
                for segment_info in segment_object:
                    segment_kw = []
                    seg_ids = []
                    seg_texts = []

                    segment_id = segment_info["segment_id"]
                    text = segment_info["text"]
                    keyword_object = segment_info["keywords"]
                    user_object = segment_info["authors"]

                    user_id = user_object["user_id"]

                    try:
                        u_id = user_dict[user_id]
                    except KeyError:
                        # Intialize
                        user_dict.update(
                            {
                                user_id: {
                                    "keywords": None,
                                    "text": None,
                                    "segment_id": None,
                                    "context_id": context_list,
                                }
                            }
                        )

                    try:
                        keyword_values = [kw["value"] for kw in keyword_object]
                    except KeyError:
                        keyword_values = []
                        continue
                    segment_kw.extend(list(process.dedupe(keyword_values)))
                    seg_texts.append(text)
                    seg_ids.append(segment_id)

                    user_kw_list = user_dict[user_id].get("keywords")
                    user_text_list = user_dict[user_id].get("text")
                    user_id_list = user_dict[user_id].get("segment_id")
                    user_context_list = user_dict[user_id].get("context_id")

                    if user_kw_list is not None:
                        user_kw_list.extend(segment_kw)
                        user_text_list.extend(seg_texts)
                        user_id_list.extend(seg_ids)
                        user_context_list.extend(context_list)
                        user_dict[user_id].update(
                            {
                                "keywords": list(set(user_kw_list)),
                                "text": list(set(user_text_list)),
                                "segment_id": list(set(user_id_list)),
                                "context_id": list(set(user_context_list)),
                            }
                        )
                    else:
                        user_dict[user_id].update(
                            {
                                "keywords": segment_kw,
                                "text": seg_texts,
                                "segment_id": seg_ids,
                                "context_id": context_list,
                            }
                        )

        return user_dict

    def get_user_name(self, response: Dict, function_name: str = "userName"):

        for u_info in response[function_name]:
            user_id = u_info["xid"]
            try:
                user_name = u_info["name"]
            except Exception:
                logger.warning("Could not get user name ... using ID instead")
                user_name = user_id

            return user_name

    def form_reference_features(
        self,
        reference_user_dict: Dict,
        context_id: str,
        query_by: str = "keywords",
        write: bool = True,
        tag: str = "v1",
    ):
        ref_user_info_dict = {
            u: reference_user_dict[u][query_by] for u in reference_user_dict.keys()
        }

        user_vector_data = {}
        num_features_in_input = {u: 0 for u in ref_user_info_dict.keys()}

        for user, info in ref_user_info_dict.items():
            kw_features = self.vectorizer.get_embeddings(input_list=info)
            user_vector_data.update({user: kw_features})
            num_features_in_input[user] = len(kw_features)

        total_features = np.sum(list(num_features_in_input.values()))

        if write:
            file_name = context_id
            with open(file_name + "_" + tag + ".pickle", "wb") as f_:
                pickle.dump(user_vector_data, f_)

            features_s3_path = self._upload_file(
                context_id=context_id,
                feature_dir=self.feature_dir,
                file_name=file_name + "_" + tag + ".pickle",
            )
            reference_user_json_name = self.to_json(
                reference_user_dict, filename=context_id + "_" + tag
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

            return reference_user_json_path, features_s3_path, total_features

    def _upload_file(self, context_id, feature_dir, file_name):
        s3_path = context_id + feature_dir + file_name
        self.s3_client.upload_to_s3(file_name=file_name, object_name=s3_path)

        # Once uploading is successful, check if feature file exists on disk and delete it
        local_path = Path(file_name).absolute()
        if os.path.exists(local_path):
            os.remove(local_path)

        return s3_path

    def form_user_contexts_query(
        self, context_id: str, top_n_result: int, tag: str, query_by: str = "keywords"
    ):
        if tag == "v1":
            query, variables = self.query_client.form_reference_users_query(
                context_id=context_id, top_n_result=top_n_result
            )

            response = self.query_client.perform_query(query=query, variables=variables)

            reference_user_dict = self.format_reference_response(response)
        else:
            query, variables = self.query_client.form_user_contexts_query(
                context_id=context_id, top_n_result=top_n_result
            )
            response = self.query_client.perform_query(query=query, variables=variables)

            reference_user_dict = self.format_user_contexts_reference_response(response)

        (
            reference_user_json_path,
            features_path,
            total_features,
        ) = self.form_reference_features(
            reference_user_dict=reference_user_dict,
            context_id=context_id,
            query_by=query_by,
            tag=tag,
        )

        return reference_user_json_path, features_path, total_features


if __name__ == "__main__":
    test_file = "eg.json"
    with open(test_file, "rb") as f_:
        data = js.load(f_)

    q = QueryHandler(dgraph_url="localhost:9080")
    dg = GraphQuery("localhost:9080")
    query, var = dg.form_user_contexts_query(
        "01DBB3SN99AVJ8ZWJDQ57X9TGX", top_n_result=15
    )
    resp = dg.perform_query(query, var)
    q.to_json(resp, "v2_normalized_response")

    user_info = q.format_user_contexts_reference_response(resp)

    print(user_info.keys(), len(user_info.keys()))
    print()
    kw_list = []
    for k, v in user_info.items():
        kw_list.extend(v["keywords"])

    q.to_json(user_info, "v2_normalize")

    q.form_reference_features(user_info, context_id="01DBB3SN99AVJ8ZWJDQ57X9TGX")
