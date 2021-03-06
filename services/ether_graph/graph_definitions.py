from dataclasses import dataclass, field
from dataclasses_json import (
    dataclass_json,
    Undefined,
    DataClassJsonMixin,
)
from typing import List, Dict, Any, Mapping, Union, Optional
import uuid
from datetime import datetime
import hashlib

from meta_config import dgconfig


def hash_sha_object(data: str) -> str:
    hash_object = hashlib.sha1(data.encode())
    hash_str = hash_object.hexdigest()
    return hash_str


@dataclass(init=False)
class ArgHolder:
    kwargs: Mapping[Any, Any]

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ObjectConversions(DataClassJsonMixin):
    """
    Helper methods to marhall and unMarshall from class objects to Python's dicts or List[dict] and vice-versa
    """

    @classmethod
    def get_object_from_dict(cls, object_dict: Union[Dict, List[Dict]]):
        if type(object_dict) == list:
            return cls.schema().load(object_dict, many=True)
        else:
            return cls.from_dict(object_dict)

    @classmethod
    def get_dict_from_object(cls, class_object) -> Dict:
        if type(class_object) == list:
            return cls.schema().dump(class_object, many=True)
        else:
            return class_object.to_dict()


@dataclass_json
@dataclass
class DgraphTypeLister(ObjectConversions):
    """
    Helper methods to marhall and unMarshall from class objects to Python's dicts or List[dict] and vice-versa
    """

    @classmethod
    def get_class_name(cls):
        return cls.__name__


@dataclass_json
@dataclass
class DgraphAttributes(DgraphTypeLister):
    dgraphType: str = field(default="", metadata=dgconfig(field_name="dgraph.type"))
    xid: str = field(default="")
    uid: str = field(default="", metadata=dgconfig(ignore_schema=True))
    attribute: str = field(default="")

    def __post_init__(self):
        self.uid = "_:" + self.xid


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(order=True)
class Keyphrase(DgraphAttributes):
    originalForm: str = field(default="")
    type: str = field(compare=False, default=None)
    value: str = field(init=False, compare=True, default=None)
    attribute: str = field(default="segmentKeywords")

    def __post_init__(self):
        self.value = self.originalForm.lower()
        self.xid = hash_sha_object(self.value)
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json
@dataclass(order=True)
class Entity(DgraphAttributes):
    originalForm: str = field(default="")
    value: str = field(init=False, compare=True, default=None)
    label: str = field(compare=False, default=False)
    related_to_keyphrase: bool = field(
        init=False, default=False, compare=False, metadata=dgconfig(index=False)
    )
    attribute: str = field(default="entities")

    def __post_init__(self):
        self.value = self.originalForm.lower()
        self.xid = hash_sha_object(self.value)
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class User(DgraphAttributes):

    spokenBy: str = field(default="", repr=False, metadata=dgconfig(field_name="xid"))
    email: str = field(init=False, default=None)
    name: str = field(init=False, default=None)
    deleted: bool = field(init=False, default=False)
    createdAt: str = field(
        init=False, default=None, metadata=dgconfig(dg_field=datetime)
    )
    deletedAt: str = field(
        init=False, default=None, metadata=dgconfig(dg_field=datetime)
    )
    updatedAt: str = field(
        init=False, default=None, metadata=dgconfig(dg_field=datetime)
    )

    userEntities: List[Entity] = field(init=False, default_factory=list)
    groupedWith: Optional[List["User"]] = field(init=False, default_factory=list)

    attribute: str = field(default="userId")

    def __post_init__(self):
        self.xid = self.spokenBy
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class TranscriptionProvider(DgraphAttributes):
    transcriber: str = field(
        default="", repr=False, metadata=dgconfig(field_name="xid")
    )
    name: str = field(init=False, default=None)

    attribute: str = field(default="segmentProvider")

    def __post_init__(self):
        self.xid = self.transcriber
        self.name = self.transcriber
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Source(DgraphAttributes):
    recordingId: str = field(
        default=None, repr=False, metadata=dgconfig(field_name="xid")
    )
    type: str = field(default="recording")
    attribute: str = field(default="sourceId")

    def __post_init__(self):
        self.xid = self.recordingId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class SummaryGroup(DgraphAttributes):
    groupId: str = field(default="", repr=False, metadata=dgconfig(field_name="xid"))
    hasKeywords: List[Keyphrase] = field(default_factory=list)
    hasEntities: List[Entity] = field(default_factory=list)
    hasUser: List[User] = field(default_factory=list)

    attribute: str = field(default="groupId")

    def __post_init__(self):
        self.xid = self.groupId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class TranscriptionSegment(DgraphAttributes):
    id: str = field(default="", repr=False, metadata=dgconfig(field_name="xid"))
    originalText: str = field(
        default="", repr=False, metadata=dgconfig(field_name="text")
    )
    languageCode: str = field(
        default="", repr=False, metadata=dgconfig(field_name="language")
    )
    startTime: str = field(default=None, metadata=dgconfig(dg_field=datetime))
    endTime: str = field(default=None, metadata=dgconfig(dg_field=datetime))
    duration: int = field(default=None)

    analyzedText: str = field(default="")
    embedding_vector_uri: str = field(default="")
    embedding_model: str = field(default="")
    embedding_vector_group_uri: str = field(default="")
    groupId: str = field(default=None)
    highlight: bool = field(default=False)

    authoredBy: User = field(default=None)
    providedBy: TranscriptionProvider = field(default=None)
    hasSource: Source = field(default=None)
    hasKeywords: List[Keyphrase] = field(default_factory=list)
    hasEntities: List[Entity] = field(default_factory=list)
    belongsTo: SummaryGroup = field(default=None)

    attribute: str = field(default="segmentId")
    text: str = field(default=None)
    language: str = field(default=None)

    def __post_init__(self):
        self.xid = self.id
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()
        self.text = self.originalText
        self.language = self.languageCode


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(init=False)
class Marker(DgraphAttributes):
    xid: str
    type: str
    description: str
    isSuggested: bool

    createdBy: User
    createdAt: str = field(metadata=dgconfig(dg_field=datetime))
    attribute: str = field(default="markerId")

    def __post_init__(self):
        self.xid = str(uuid.UUID(self.xid))
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Mind(DgraphAttributes):
    mindId: str = field(default="", repr=False, metadata=dgconfig(field_name="xid"))
    name: str = field(default=None, init=False)
    type: str = field(default=None, init=False)

    attribute: str = field(default="mindId")

    def __post_init__(self):
        self.xid = self.mindId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ContextSession(DgraphAttributes):
    instanceId: str = field(
        default=None, repr=False, metadata=dgconfig(field_name="xid")
    )
    startTime: str = field(default=None, metadata=dgconfig(dg_field=datetime))
    attribute: str = field(default="instanceId")

    associatedMind: Mind = field(default=None)
    hasSegment: List[TranscriptionSegment] = field(default_factory=list)
    hasMarker: List[Marker] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.xid = self.instanceId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Context(DgraphAttributes):
    contextId: str = field(repr=False, default="", metadata=dgconfig(field_name="xid"))
    attribute: str = field(default="contextId")

    hasMeeting: List[ContextSession] = field(default_factory=list)
    hasMember: List[User] = field(default_factory=list)

    def __post_init__(self):
        self.xid = self.contextId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Customer(DgraphAttributes):
    customerId: str = field(default="", repr=False, metadata=dgconfig(field_name="xid"))
    attribute: str = field(default="customerId")

    hasUser: List[User] = field(default_factory=list)

    def __post_init__(self):
        self.xid = self.customerId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Workspace(DgraphAttributes):
    workspaceId: str = field(
        default="", repr=False, metadata=dgconfig(field_name="xid")
    )
    attribute: str = field(default="workspaceId")
    name: str = field(default="")

    belongsTo: Customer = field(default=None)
    hasMember: List[User] = field(default_factory=list)

    def __post_init__(self):
        self.xid = self.workspaceId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Channel(DgraphAttributes):
    channelId: str = field(default="", repr=False, metadata=dgconfig(field_name="xid"))
    attribute: str = field(default="channelId")
    name: str = field(default="")

    belongsTo: Workspace = field(default=None)
    hasContext: Context = field(default=None)
    hasMember: List[User] = field(default_factory=list)

    def __post_init__(self):
        self.xid = self.channelId
        self.uid = "_:" + self.xid
        self.dgraphType = self.get_class_name()


@dataclass_json
@dataclass
class IndexRules:
    xid: str = field(metadata=dgconfig(index_type=["exact"], directive=["@upsert"]))
    attribute: str = field(metadata=dgconfig(index_type=["hash"]))
    name: str = field(metadata=dgconfig(index_type=["term"]))
    email: str = field(metadata=dgconfig(index_type=["exact"]))
    value: str = field(metadata=dgconfig(index_type=["term, fulltext"]))
    originalForm: str = field(metadata=dgconfig(index_type=["term, fulltext"]))
    label: str = field(metadata=dgconfig(index_type=["exact"]))
    text: str = field(metadata=dgconfig(index_type=["fulltext"]))
    embedding_vector_uri: str = field(metadata=dgconfig())
    embedding_vector_group_uri: str = field(metadata=dgconfig())
    description: str = field(metadata=dgconfig(index_type=["term, fulltext"]))
    type: str = field(metadata=dgconfig(index_type=["term"]))
    startTime: datetime = field(metadata=dgconfig(index_type=["month"]))
    updatedAt: datetime = field(metadata=dgconfig(index_type=["month"]))
    deleted: bool = field(metadata=dgconfig(index=True, index_type=["bool"]))
    deletedAt: datetime = field(metadata=dgconfig(index_type=["month"]))
    createdAt: datetime = field(metadata=dgconfig(index_type=["month"]))
    highlight: bool = field(metadata=dgconfig(index=True, index_type=["bool"]))

    # UID type relations
    associatedMind: Mind = field(metadata=dgconfig(directive=["@reverse"]))
    hasMeeting: List[ContextSession] = field(metadata=dgconfig(directive=["@reverse"]))
    hasSegment: List[TranscriptionSegment] = field(
        metadata=dgconfig(directive=["@reverse"])
    )
    hasKeywords: List[Keyphrase] = field(metadata=dgconfig(directive=["@reverse"]))
    hasEntities: List[Entity] = field(metadata=dgconfig(directive=["@reverse"]))
    authoredBy: User = field(metadata=dgconfig(directive=["@reverse"]))
    providedBy: TranscriptionProvider = field(metadata=dgconfig())
    hasSource: Source = field(metadata=dgconfig())
    belongsTo: Union[SummaryGroup, Workspace, Customer] = field(
        metadata=dgconfig(directive=["@reverse"])
    )
    hasMember: List[User] = field(metadata=dgconfig(directive=["@reverse"]))
    hasContext: Context = field(metadata=dgconfig(directive=["@reverse"]))
    hasMarker: List[Marker] = field(metadata=dgconfig(directive=["@reverse"]))
    hasUser: List[User] = field(metadata=dgconfig(directive=["@reverse"]))
    createdBy: User = field(metadata=dgconfig(directive=["@reverse"]))
    groupedWith: List[User] = field(metadata=dgconfig())
    userEntities: List[Entity] = field(metadata=dgconfig())
