import pydgraph
import json as js


class DgraphClient(object):
    def __init__(self, url="111.93.155.194:9080"):
        client_stub = pydgraph.DgraphClientStub(url)
        self.client = pydgraph.DgraphClient(client_stub)

    def perform_query(self, query, variables):
        txn = self.client.txn()
        try:
            res = self.client.txn(read_only=True).query(
                query, variables=variables
            )
            response = js.loads(res.json)

            return response
        finally:
            # Clean up. Calling this after txn.commit() is a no-op and hence safe.
            txn.discard()

    def form_user_keyword_query(self, email, top_n_result):
        user_kw_query = """
        query userKw($n: string, $t: int) {
            userKw(func: eq(email, $n)) {
                uid
                xid
                name
                mentionName
                ~hasMember{
                    name
                }
                ~authoredBy (first: $t) {
                    hasKeywords {
                        values
                    }
                }
            }
        }
        """

        variables = {"$n": email, "$t": str(top_n_result)}

        return user_kw_query, variables

    def form_reference_users_keyword_query(self, top_n_result=5):
        ref_user_kw_query = """
        query userKw($t: int) {
            userKw(func: type("User")) {
                uid
                xid
                name
                mentionName
                ~hasMember{
                    name
                }
                ~authoredBy (first: $t) {
                    hasKeywords {
                        values
                    }
                }
            }
        }
        """
        variables = {"$t": str(top_n_result)}

        return ref_user_kw_query, variables

    def format_response(self, resp):
        user_id_dict = {}
        user_kw_dict = {}
        for info in resp["userKw"]:
            try:
                user_id = info["xid"]
                user_name = info["name"]

                user_id_dict.update({user_id: user_name})

                keyword_object = info["~authoredBy"]
                user_kw_list = []
                for obj in keyword_object:
                    kw = obj["hasKeywords"]
                    user_kw_list.extend([words for words in kw[0]["values"]])

                user_kw_dict.update({user_id: user_kw_list})
            except Exception:
                continue

        return user_kw_dict, user_id_dict
