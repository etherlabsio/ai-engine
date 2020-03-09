from dataclasses import dataclass, field, fields
from dataclasses_json import dataclass_json
from typing import List, Dict, Any, TextIO, ContextManager, Mapping
from io import StringIO, BytesIO

from meta_config import MetaConfig, TypeDefContainer, IndexContainer

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
    IndexRules,
)


class IOHandler(ContextManager):
    """
    Custom Context manager for IO operations. More methods can be added for custom use-cases.
    """

    def __init__(self, file_path, mode="r"):
        self.__path = file_path
        self.__mode = mode
        self.__file_object = None

    def __enter__(self) -> TextIO:
        if self.__file_object is not None:
            raise RuntimeError(
                f"{self.__class__.__name__} is already open. It is not re-entrant"
            )

        self.__file_object = open(self.__path, self.__mode)
        return self.__file_object

    def __exit__(self, type, val, tb):
        self.__file_object.close()


class Store:
    """
    A base storage class, providing some default behaviors that all other
    storage systems can inherit or override, as necessary.

    # The public methods shouldn't be overridden by subclasses unless required.
    # Ideally, subclasses should add/override private methods as per the sepcs of the specific storage api
    """

    def write(self, content, name="schema"):
        if isinstance(content, BytesIO) or isinstance(content, StringIO):
            content = content.getvalue()
        return self._write(content, name)

    def _write(self, content, name: str, mode="w"):
        return content


class FileStore(Store):
    def _write(self, content, name, mode="w"):
        with IOHandler(name, mode) as f:
            f.write(content)


@dataclass(init=False)
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

    @staticmethod
    def make_object_container():
        return [f.type for f in fields(ObjectContainer)]

    @staticmethod
    def make_index_container():
        return [IndexRules]


@dataclass_json
@dataclass
class TypeGenerator:
    typedef_container: TypeDefContainer = field(default=None)
    typedef_string: str = ""
    class_objects: List = field(default_factory=ObjectContainer.make_object_container)

    def __post_init__(self):
        self.typedef_config = TypeDefContainer()

    def generate(self, io_handler=None, name=None):
        return self._generate(io_handler=io_handler, name=name)

    def as_object(self):
        return self.typedef_container

    def as_string(self):
        return self.typedef_string

    def _generate(self, io_handler=None, name=None):
        field_object = [(fields(cls), cls) for cls in self.class_objects]
        self.typedef_container = self.typedef_config.make_typedef_container(
            field_object
        )
        self.typedef_string = self.form_typedef_string()

        if io_handler is not None:
            io_handler.write(content=self.as_string(), name=name)

        return self

    def form_typedef_string(self):
        schema_predicates_iter = []

        for cls, edges in self.typedef_container.items():

            class_predicates_iter = [
                f"\t{edge.name:10}: {edge.link_name}" for edge in edges
            ]
            class_predicates_str = "\n".join(class_predicates_iter)

            class_def_str = f"type {cls} {{\n{class_predicates_str}\n}}"
            schema_predicates_iter.append(class_def_str)

        typedef_str = "\n\n".join(schema_predicates_iter)
        return typedef_str

    def __repr__(self):
        return f"{self.__class__.__name__}({self.typedef_string})"

    def __str__(self):
        return f"{self.typedef_string}"


@dataclass_json
@dataclass
class IndexGenerator(TypeGenerator):
    index_object: List = field(default_factory=ObjectContainer.make_index_container)
    index_container: IndexContainer = field(default=None)
    index_string: str = field(default="\n\n# Defining indices\n\n")

    def __post_init__(self):
        self.ic = IndexContainer()

    def _generate(self, io_handler=None, name=None):
        index_field_object = [f for cls in self.index_object for f in fields(cls)]
        self.index_container = self.ic.make_predicate_container(index_field_object)
        self.index_string = self.form_index_string()

        if io_handler is not None:
            io_handler.write(content=self.as_string(), name=name)

        return self

    def as_object(self):
        return self.index_container

    def as_string(self):
        return self.index_string

    def form_index_string(self):
        # Iterator to store the indices for every relations and nodes defined
        relations_index_iter = []

        for predicate, meta in self.index_container.items():
            # Iterator to store each predicate's index rule
            predicate_index_iter = []

            predicate_type = meta.dg_field
            index_type = meta.index_type
            directive = meta.directive

            _, formatted_relation = MetaConfig.format_types(
                predicate_type, indexing=True
            )
            relation_index_str = MetaConfig._format_index_string(index_type, directive)

            index_str = f"{predicate}: {formatted_relation} {relation_index_str} ."
            predicate_index_iter.append(index_str)

            predicate_index_str = "\n".join(predicate_index_iter)
            relations_index_iter.append(predicate_index_str)

        index_string = self.index_string + "\n".join(relations_index_iter)
        return index_string

    def __repr__(self):
        return f"{self.__class__.__name__}({self.index_string})"

    def __str__(self):
        return f"{self.index_string}"


@dataclass
class SchemaGenerator(IndexGenerator):
    version: str = "0.1"
    dgraph_version: str = field(default="1.1.0")

    typedef_gen: TypeGenerator = field(default_factory=TypeGenerator)
    index_gen: IndexGenerator = field(default_factory=IndexGenerator)
    schema_string: StringIO = field(default_factory=StringIO)

    def __post_init__(self):
        self.version_str = (
            f"# EtherGraph Schema v{self.version} \n"
            f"# Generated schema for Dgraph v{self.dgraph_version} \n\n"
        )

    def _generate(self, io_handler: Any = None, name=None):
        typedef_string = self.typedef_gen.generate().as_string()
        index_string = self.index_gen.generate().as_string()

        self.schema_string.write(
            self.version_str + typedef_string + index_string + "\n"
        )
        if io_handler is not None:
            f_name = f"schema_v{self.version}.schema" if name is None else name
            io_handler.write(content=self.schema_string, name=f_name)

        return self.schema_string.getvalue()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.schema_string.getvalue()})"

    def __str__(self):
        return f"{self.schema_string.getvalue()}"


if __name__ == "__main__":
    ls = FileStore()
    s = SchemaGenerator(version="0.1")
    schema = s.generate(ls)

    t = TypeGenerator()
    ts = t.generate()
    ts.as_string()
