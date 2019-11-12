class Schema(object):
    def __init__(self):
        self.meeting_def = """
                type Context {
                    xid: string
                    attribute: string
                    hasMeeting: [Instance]
                    associatedMind: Mind
                }

                type Mind {
                    xid: string
                    attribute: string
                }

                type ContextSession {
                    xid: string
                    attribute: string
                    hasSegment: [Segment]
                }

                type TranscriptionSegment {
                    xid: string
                    attribute: string
                    text: string
                    analyzedText: string
                    embedding_vector_uri: string
                    embedding_vector_group_uri: string
                    confidence: float
                    language: string
                    startTime: datetime
                    endTime: datetime
                    duration: int
                    authoredBy: User
                    hasKeywords: [Keyphrase]
                    hasSource: [Source]
                    providedBy: Provider    
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
                    name: string
                    attribute: string
                }

                type Keyphrase {
                    values: [string]
                    attribute: string
                    important: bool
                    type: string
                    origin: string
                }

                xid: string @index(exact) @upsert .
                name: string @index(exact) .
                value: string @index(term) .
                attribute: string @index(hash) .
                text: string @index(fulltext) .
                embedding_vector_uri: string .
                embedding_vector_group_uri: string .
                associatedMind: uid .
                hasMeeting: [uid] @reverse .
                hasSegment: [uid] @reverse .
                hasKeywords: [uid] .
                authoredBy: uid @reverse .

                """
