from dataclasses import dataclass, asdict, field
from dataclasses_json import dataclass_json, Undefined, CatchAll, DataClassJsonMixin
from typing import List, Dict, Tuple, Text, Any, Mapping, Union, Optional
import uuid
from datetime import datetime


class ObjectConversions(DataClassJsonMixin):
    """
    Helper methods to marhall and unMarshall from class objects to Python's dicts or List[dict] and vice-versa
    """

    @classmethod
    def get_object(cls, object_dict: Union[Dict, List[Dict]]):
        if type(object_dict) == list:
            return cls.schema().load(object_dict, many=True)
        else:
            return cls.from_dict(object_dict)

    @classmethod
    def get_dict(cls, class_object) -> Dict:
        if type(class_object) == list:
            return cls.schema().dump(class_object, many=True)
        else:
            return class_object.to_dict()


@dataclass(init=False)
class ArgHolder:
    kwargs: Mapping[Any, Any]

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@dataclass_json
@dataclass
class Score:
    pagerank: float = field(default=0.0)
    segsim: float = field(default=0.0)
    boosted_sim: float = field(default=0.0)
    norm_boosted_sim: float = field(default=0.0)
    loc: float = field(default=0.0)


@dataclass_json
@dataclass(order=True)
class Keyphrase(ObjectConversions):
    originalForm: str
    type: str = field(compare=False)
    score: Score
    to_remove: bool = field(init=False, default=False, compare=False)
    value: str = field(init=False, compare=True)

    def __post_init__(self):
        self.value = self.originalForm.lower()


@dataclass_json
@dataclass(order=True)
class Entity(ObjectConversions):
    originalForm: str
    label: str = field(compare=False)
    preference: int
    score: Score
    confidence_score: float
    type: str = field(init=False, default="entity")
    to_remove: bool = field(init=False, default=False, compare=False)
    related_to_keyphrase: bool = field(init=False, default=False, compare=False)
    value: str = field(init=False)

    def __post_init__(self):
        self.value = self.originalForm.lower()


@dataclass_json
@dataclass
class Phrase(ObjectConversions):
    segmentId: str
    originalText: str
    highlight: bool = field(default=False, compare=False)
    offset: float = field(default=0.0)
    keyphrases: List[Keyphrase] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    entitiesQuality: float = 0.0
    medianEntitiesQuality: float = 0.0
    keyphraseQuality: float = 0.0
    medianKeyphraseQuality: float = 0.0

    def __post_init__(self):
        self.segmentId = str(uuid.UUID(self.segmentId))


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Context(ObjectConversions):
    contextId: str
    instanceId: str
    mindId: str
    unknown_fields: CatchAll

    def __post_init__(self):
        self.instanceId = str(uuid.UUID(self.instanceId))


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Segment(ObjectConversions):
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

    embedding_vector_uri: str = ""
    embedding_vector_group_uri: str = ""
    embedding_model: str = ""
    text: str = ""
    groupId: str = None
    highlight: bool = False

    keyphrases: List[Keyphrase] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)

    def __post_init__(self):
        self.id = str(uuid.UUID(self.id))
        self.recordingId = str(uuid.UUID(self.recordingId))
        self.transcriptId = str(uuid.UUID(self.transcriptId))
        self.spokenBy = str(uuid.UUID(self.spokenBy))


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Request(Context):
    unknown_fields: CatchAll
    limit: int = field(init=False, default=10)
    populateGraph: bool = field(default=True)
    validate: bool = field(init=False, default=False)
    relativeTime: str = field(init=False, default="")
    segments: List[Segment] = field(default_factory=list)


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SummaryRequest(Request):
    unknown_fields: CatchAll
    segments: List[Segment] = field(default_factory=list)
    keyphrases: List[Keyphrase] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)


@dataclass_json
@dataclass
class GraphQueryRequest(ObjectConversions):
    query: str
    variables: Mapping[str, Any] = field(default=None)


@dataclass_json
@dataclass
class GraphResponse:
    uid: str
    xid: str
    attribute: str
    embedding_vector_uri: str
    embedding_vector_group_uri: str = field(default="")


@dataclass_json
@dataclass
class GraphSegmentResponse(ObjectConversions):
    q: List[GraphResponse] = field(default_factory=list)
