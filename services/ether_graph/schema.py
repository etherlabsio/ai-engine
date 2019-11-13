class Schema(object):
    def fetch_schema(self):
        meeting_def = """
                type Context {
                    xid: string
                    attribute: string
                    hasMeeting: [ContextSession]
                    associatedMind: Mind
                }

                type Mind {
                    xid: string
                    attribute: string
                }

                type ContextSession {
                    xid: string
                    attribute: string
                    hasSegment: [TranscriptionSegment]
                }

                type TranscriptionSegment {
                    xid: string
                    attribute: string
                    text: string
                    analyzedText: string
                    embedding_vector_uri: string
                    embedding_vector_group_uri: string
                    groupId: string
                    confidence: float
                    language: string
                    startTime: datetime
                    endTime: datetime
                    duration: int
                    authoredBy: User
                    hasKeywords: Keyphrase
                    hasSource: [Source]
                    providedBy: TranscriptionProvider    
                }

                type User {
                    xid: string
                    attribute: string
                }

                type Source {
                    xid: string
                    attribute: string
                    type: string
                }

                type TranscriptionProvider {
                    xid: string
                    name: string
                    attribute: string
                }

                type Keyphrase {
                    xid: string
                    values: string
                    attribute: string
                    important: bool
                    type: string
                    origin: string
                }

                xid: string @index(exact) @upsert .
                name: string @index(exact) .
                values: [string] .
                attribute: string @index(hash) .
                text: string @index(fulltext) .
                embedding_vector_uri: string .
                embedding_vector_group_uri: string .
                groupId: string @index(hash) .
                associatedMind: uid .
                hasMeeting: [uid] @reverse .
                hasSegment: [uid] @reverse .
                hasKeywords: uid .
                authoredBy: uid @reverse .
                providedBy: uid .

                """
        return meeting_def
