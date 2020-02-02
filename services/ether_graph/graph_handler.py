import logging
import json as js
import pydgraph

from graph_schema import Schema

logger = logging.getLogger(__name__)


class GraphHandler(object):
    def __init__(self, dgraph_url=None):
        self.context_schema = Schema()

        self.client_stub = pydgraph.DgraphClientStub(dgraph_url)
        client = pydgraph.DgraphClient(self.client_stub)

        self.dgraph = client

    async def close_client(self):
        logger.info("Client stub connection is closed.")
        self.client_stub.close()

    # Drop All - discard all data and start from a clean slate.
    async def drop_all(self):
        return self.dgraph.alter(pydgraph.Operation(drop_all=True))

    def query_transform_node(self, node_obj, extra_field=None):
        """
        Given an xid a query request for UID is made and given a node object, transform the node object to use the UID
        Args:
            response:
            node:

        Returns:

        """
        xid = node_obj.xid
        response = self._query_uid(xid=xid)

        # This is useful for predicates that do not have the standard `xid` attribute
        if extra_field is not None:
            ext_query = """query q($i: string) {
                    q(func: eq(name, $i)) {
                        uid
                        attribute
                        xid
                    }
                }"""
            response = self._query_uid(xid=extra_field, ext_query=ext_query)

        try:
            node_uid = response["q"][0].get("uid")
            logger.info(
                "Received response", extra={"response": response, "uid": node_uid},
            )
            node_obj.uid = node_uid
        except IndexError:
            logger.debug("No UID found", extra={"response": response})

        return node_obj

    def set_schema(self):
        schema = self.context_schema.fetch_schema()
        return self.dgraph.alter(pydgraph.Operation(schema=schema))

    def _query_uid(self, xid, ext_query=None):
        txn = self.dgraph.txn()
        try:
            query = """query q($i: string) {
                    q(func: eq(xid, $i)) {
                        uid
                        attribute
                        xid
                    }
                }"""
            if ext_query is not None:
                query = ext_query

            variables = {"$i": xid}
            res = self.dgraph.txn(read_only=True).query(query, variables=variables)
            response = js.loads(res.json)

            return response
        finally:
            txn.discard()

    def mutate_info(self, mutation_query):
        txn = self.dgraph.txn()
        try:
            # Run mutation.
            response = txn.mutate(set_obj=mutation_query)

            # Commit transaction.
            txn.commit()

            return response
        finally:
            # Clean up. Calling this after txn.commit() is a no-op and hence safe.
            txn.discard()

    def perform_queries(self, query_text, variables=None):
        txn = self.dgraph.txn()
        try:
            query = query_text

            if variables is not None:
                res = self.dgraph.txn(read_only=True).query(query, variables=variables)
            else:
                res = self.dgraph.txn(read_only=True).query(query)

            response = js.loads(res.json)
            return response
        finally:
            txn.discard()
