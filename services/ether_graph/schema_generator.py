from dataclasses import dataclass, field, fields, is_dataclass
from dataclasses_json import dataclass_json
from typing import List, get_type_hints
import typing
import datetime
import json

from graph_definitions import (
    Context,
    ContextSession,
    TranscriptionSegment,
    TranscriptionProvider,
    Keyphrase,
    Entity,
    Mind,
    User,
    Source,
    SummaryGroup,
    Customer,
    Workspace,
    Channel,
)

dgraph_types = {str: "string", bool: "bool", int: "int", datetime.datetime: "dateTime"}

ignore_attrs = [
    "uid",
    "dgraphType",
    "instanceId",
    "contextId",
    "mindId",
    "spokenBy",
    "id",
    "recordingId",
    "transcriber",
    "customerId",
    "workspaceId",
    "channelId",
]


def make_object_container():
    return [f.type for f in fields(ObjectContainer)]


@dataclass
class ObjectContainer:
    context: Context
    context_session: ContextSession
    transcription_segment: TranscriptionSegment
    transcription_provider: TranscriptionProvider
    keyphrase: Keyphrase
    entity: Entity
    mind: Mind
    user: User
    source: Source
    summary_group: SummaryGroup
    customer: Customer
    workspace: Workspace
    channel: Channel


@dataclass_json
@dataclass
class SchemaGenerator:
    version: str
    dgraph_version: str = field(default="1.1.0")
    objects: List = field(default_factory=make_object_container)
    schema_string: str = field(default="")

    def __post_init__(self):
        self.schema_string = self.form_schema_string()

    def get_type_annotations(self, class_object):
        if is_dataclass(class_object):
            result = []
            for f in fields(class_object):
                if f.name not in ignore_attrs:
                    type_name, dest_node = self._format_types(f.type)
                    result.append((f.name, type_name, dest_node))
                else:
                    continue

            return result
        else:
            raise TypeError(
                f"get_type_annotations() should be called on dataclass instances; {class_object}"
            )

    def _format_types(self, attr_type):
        if attr_type in dgraph_types.keys():
            type_name = dgraph_types[attr_type]
            dest_node = None

            return type_name, dest_node

        # Check if destination node is an instance of typing.Typing
        elif type(attr_type) is typing._GenericAlias:

            dest_class_instance = attr_type.__args__[0]
            if is_dataclass(dest_class_instance):
                # Handle direct types and aliases from typing.Typing
                type_name = attr_type._name
                dest_node = attr_type.__args__[0].__name__

            else:
                # Handle ForwardRefs in typing
                type_name = dest_class_instance._name
                forward_node_class = dest_class_instance.__args__[0]
                dest_node = forward_node_class.__forward_arg__

            # print(attr_type, type_name, dest_node)
            return type_name, dest_node

        elif is_dataclass(attr_type):
            type_name = None
            dest_node = attr_type.__name__

            return type_name, dest_node
        else:
            return None, None

    def _get_type_name(self, attr_type: typing._GenericAlias):
        if isinstance(attr_type.__args__, typing.Tuple):
            type_name = attr_type.__args__[0]._name
        else:
            type_name = attr_type._name

        return type_name

    def _get_type_args(self, attr_type: typing._GenericAlias):
        if isinstance(attr_type.__args__, typing.Tuple):
            fn_args = attr_type.__args__[0].__args__[0]
        else:
            fn_args = attr_type.__args__[0]

        return fn_args

    def form_schema_string(self):
        cls_repr_list = []
        for c in self.objects:
            repr_str_iter = []

            anns = self.get_type_annotations(c)
            for attr, type_name, dest_node in anns:

                if dest_node is not None and type_name is not None:
                    class_schema_string = f"\t{attr:10}: [{dest_node}]"
                elif dest_node is not None:
                    class_schema_string = f"\t{attr:10}: {dest_node}"
                else:
                    class_schema_string = f"\t{attr:10}: {type_name}"

                repr_str_iter.append(class_schema_string)

            cls_repr_str = "\n".join(repr_str_iter)
            cls_repr = f"type {c.__name__} {{\n{cls_repr_str}\n}}"
            cls_repr_list.append(cls_repr)

        version_str = (
            f"# EtherGraph Schema v{self.version} \n"
            f"# Generated schema for Dgraph v{self.dgraph_version} \n\n"
        )
        schema_string = "\n\n".join(cls_repr_list)
        schema_string = version_str + schema_string
        return schema_string

    def __repr__(self):
        return f"{self.__class__.__name__}({self.schema_string})"

    def __str__(self):
        return f"{self.schema_string}"

    def to_file(self, file_format: str = "schema"):
        f_name = f"schema_v{self.version}.{file_format}"
        with open(f_name, "w") as f_:
            f_.write(self.schema_string)


if __name__ == "__main__":
    s = SchemaGenerator(version="0.1")
    s.to_file()
