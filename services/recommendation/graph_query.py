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

            return response, res
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

    def form_reference_users_keyword_query(self, top_n_result=5):
        ref_user_kw_query = """
        query mlChannelUserKw($t: int) {
            mlChannelUserKw(func: type("Channel"))
          @filter(eq(name, "ml-ai")) @cascade{
                uid
                xid
                belongsTo @filter(eq(name, "ether-labs")) {
                    name
                    attribute
                }
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
                      authoredBy {
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
        variables = {"$t": str(top_n_result)}

        return ref_user_kw_query, variables
