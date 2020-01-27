from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, Undefined, CatchAll
from typing import List, Dict, Tuple, Text, Any, Mapping, Union, Optional
import uuid
from datetime import datetime

from graph_definitions import Keyphrase, Entity, ObjectConversions


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class ContextRequest(ObjectConversions):
    contextId: str
    instanceId: str
    mindId: str
    unknown_fields: CatchAll

    def __post_init__(self):
        self.instanceId = str(uuid.UUID(self.instanceId))


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SegmentRequest(ObjectConversions):
    id: str
    originalText: str
    startTime: str
    endTime: str
    duration: int
    recordingId: str
    languageCode: str
    transcriber: str
    transcriptId: str
    createdAt: str
    updatedAt: str
    spokenBy: str

    unknown_fields: CatchAll

    embedding_vector_uri: str = field(default="")
    embedding_vector_group_uri: str = field(default="")
    embedding_model: str = field(default="")
    text: str = field(default="")
    groupId: str = field(default="")
    highlight: bool = field(default=False)

    keyphrases: List[Keyphrase] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)

    def __post_init__(self):
        self.id = str(uuid.UUID(self.id))
        self.recordingId = str(uuid.UUID(self.recordingId))
        self.transcriptId = str(uuid.UUID(self.transcriptId))
        self.spokenBy = str(uuid.UUID(self.spokenBy))


# Request objects from Keyphrase-service
@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SessionRequest(ContextRequest):
    unknown_fields: CatchAll
    segments: List[SegmentRequest] = field(default_factory=list)


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SummaryRequest(ContextRequest):
    unknown_fields: CatchAll
    limit: int = field(init=False, default=10)
    populateGraph: bool = field(init=False, default=True)
    validate: bool = field(init=False, default=False)
    relativeTime: str = field(init=False, default="")
    segments: List[SegmentRequest] = field(default_factory=list)
    keyphrases: List[Keyphrase] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)


# Graph query requests
@dataclass_json
@dataclass
class GraphQueryRequest:
    query: str
    variables: Mapping[str, Any] = field(default=None)
