from dataclasses import dataclass, field, fields
from dataclasses_json import dataclass_json
from typing import List, Dict

from meta_config import MetaConfig

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
    objects: List = field(default_factory=ObjectContainer.make_object_container)
    typedef_string: str = field(default="")

    def __post_init__(self):
        self.typedef_string = self.form_typedef_string()

    def form_typedef_string(self):
        schema_predicates_iter = []
        field_object = [(fields(cls), cls) for cls in self.objects]

        for f, cls in field_object:
            anns = MetaConfig.get_type_annotations(class_fields=f)

            class_predicates_iter = [
                f"\t{attr:10}: {type_name}" for attr, type_name in anns
            ]
            class_predicates_str = "\n".join(class_predicates_iter)

            class_def_str = f"type {cls.__name__} {{\n{class_predicates_str}\n}}"
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
    index_string: str = field(default="\n\n# Defining indices\n\n")

    def __post_init__(self):
        index_field_object = [f for cls in self.index_object for f in fields(cls)]
        self.predicate_container = MetaConfig.make_predicate_container(
            index_field_object
        )

        self.index_string = self.form_index_string()

    def form_index_string(self):
        # Iterator to store the indices for every relations and nodes defined
        relations_index_iter = []

        for predicate, meta in self.predicate_container.items():
            # Iterator to store each predicate's index rule
            predicate_index_iter = []

            relation = meta.get("dg_field")
            index_type = meta.get("index_type")
            directive = meta.get("directive")

            formatted_relation = MetaConfig._format_types(relation, indexing=True)
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
    schema_string: str = ""

    def __post_init__(self):
        version_str = (
            f"# EtherGraph Schema v{self.version} \n"
            f"# Generated schema for Dgraph v{self.dgraph_version} \n\n"
        )

        self.schema_string = (
            version_str + self.typedef_gen.typedef_string + self.index_gen.index_string
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.schema_string})"

    def __str__(self):
        return f"{self.schema_string}"

    def to_file(self, file_format: str = "schema"):
        f_name = f"schema_v{self.version}.{file_format}"
        with open(f_name, "w") as f_:
            f_.write(self.schema_string)


if __name__ == "__main__":
    s = SchemaGenerator()
    s.to_file()
