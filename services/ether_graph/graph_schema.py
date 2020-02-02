class Schema(object):
    def fetch_schema(self):
        meeting_def = """
type Customer {
    xid: string
    attribute: string
    hasUser: [User]
}

type Workspace {
    xid: string
    attribute: string
    name: string
    belongsTo: Customer
    hasMember: [User]
}

type Channel {
    xid: string
    attribute: string
    name: string
    belongsTo: Workspace
    hasContext: Context
    hasMember: [User]
}

type Context {
    xid: string
    attribute: string
    associatedMind: Mind
    hasMeeting: [ContextSession]
    hasMember: [User]
}

type Mind {
    xid: string
    attribute: string
    name: string
    type: string
}

type User {
    xid: string
    attribute: string
    email: string
    name: string
    deleted: bool
    createdAt: dateTime
    deletedAt: dateTime
    updatedAt: dateTime
    userEntities: [Entity]
    groupedWith: [User]
}

type ContextSession {
    xid: string
    attribute: string
    hasSegment: [TranscriptionSegment]
    hasMarker: [Marker]
}

type Marker {
    xid: string
    attribute: string
    isSuggested: bool
    type: string
    description: string
    createdAt: dateTime
    createdBy: User
}

type TranscriptionSegment {
    xid: string
    attribute: string
    text: string
    analyzedText: string
    embedding_vector_uri: string
    embedding_model: string
    embedding_vector_group_uri: string
    groupId: string
    language: string
    startTime: dateTime
    endTime: dateTime
    duration: int
    highlight: bool
    authoredBy: User
    hasSource: Source
    providedBy: TranscriptionProvider
    hasKeywords: [Keyphrase]
    hasEntities: [Entity]
    belongsTo: SummaryGroup
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
    originalForm: string
    value: string
    attribute: string
    type: string
}

type Entity {
    xid: string
    originalForm: string
    value: string
    label: string
    related_to_keyphrase: bool
    attribute: string
}

type SummaryGroup {
    xid: string
    attribute: string
    hasKeywords: [Keyphrase]
    hasEntities: [Entity]
    hasUser: [User]
}

xid: string @index(exact) @upsert .
name: string @index(term) .
email: string @index(exact) .
value: string @index(term, fulltext) .
originalForm: string @index(term, fulltext) .
attribute: string @index(hash) .
label: string @index(exact) .
text: string @index(fulltext) .
embedding_vector_uri: string .
embedding_vector_group_uri: string .
associatedMind: uid @reverse .
hasMeeting: [uid] @reverse .
hasSegment: [uid] @reverse .
hasKeywords: [uid] @reverse .
hasEntities: [uid] @reverse .
authoredBy: uid @reverse .
providedBy: uid .
hasSource: uid .
belongsTo: uid @reverse .
hasMember: [uid] @reverse .
hasContext: uid @reverse .
hasMarker: [uid] @reverse .
hasUser: [uid] @reverse .
createdBy: uid @reverse .
description: string @index(term, fulltext) .
type: string @index(term) .
groupedWith: [uid] .
userEntities: [uid] .

"""
        return meeting_def
