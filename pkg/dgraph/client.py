import pydgraph


class DgraphClient(object):
    def __init__(self, connector):
        self.client = connector.create_client()

    # Drop All - discard all data and start from a clean slate.
    def drop_all(self):
        return self.client.alter(pydgraph.Operation(drop_all=True))

    def set_schema(self, schema=""):
        """
        Set a schema

        Args:
            schema:

        Returns:

        Usage:
            schema = '''
                name: string @index(exact) .
                friend: [uid] @reverse .
                age: int .
                married: bool .
                loc: geo .
                dob: datetime .
                type Person {
                    name
                    friend
                    age
                    married
                    loc
                    dob
                }
            '''

        """
        return self.client.alter(pydgraph.Operation(schema=schema))

    def create_data(self, data):
        """
        Create data using JSON.

        Args:
            data: JSON object or JSON compatible format

        Returns:

        Usage:
            # Create data.
            data = {
                'uid': '_:alice',
                'dgraph.type': 'Person',
                'name': 'Alice',
                'age': 26,
                'married': True,
                'loc': {
                    'type': 'Point',
                    'coordinates': [1.1, 2],
                },
                'dob': datetime.datetime(1980, 1, 1, 23, 0, 0, 0).isoformat(),
                'friend': [
                    {
                        'uid': '_:bob',
                        'dgraph.type': 'Person',
                        'name': 'Bob',
                        'age': 24,
                    }
                ],
                'school': [
                    {
                        'name': 'Crown Public School',
                    }
                ]
            }

        """
        # Create a new transaction.
        txn = self.client.txn()
        try:
            # Run mutation.
            response = txn.mutate(set_obj=data)

            # Commit transaction.
            txn.commit()

            return response
        finally:
            # Clean up. Calling this after txn.commit() is a no-op and hence safe.
            txn.discard()

    def perform_upsert(self, query, mutate_obj, **kwargs):
        pass

    def query_data(self, data, **kwargs):
        pass

    def delete_data(self):
        """
        Deleting a data

        Returns:

        Usage:
        #     query1 = '''query all($a: string) {
        #         all(func: eq(name, $a)) {
        #            uid
        #         }
        #     }'''
        #     variables1 = {'$a': 'Bob'}
        #     res1 = client.txn(read_only=True).query(query1, variables=variables1)
        #     ppl1 = js.loads(res1.json)
        #     for person in ppl1['all']:
        #         print("Bob's UID: " + person['uid'])
        #         txn.mutate(del_obj=person)
        #         print('Bob deleted')
        #     txn.commit()
        #
        # finally:
        #     txn.discard()

        """
        # Create a new transaction.
        txn = self.client.txn()
        pass
