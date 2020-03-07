from dataclasses_json import config
from dataclasses import is_dataclass, Field
from typing import List, Dict, Any, Generic
from datetime import datetime
import typing
from copy import deepcopy


class MetaConfig:
    @staticmethod
    def get_dataclass_metadata(field_name: str = None) -> Dict[str, dict]:
        dataclass_meta = config(field_name=field_name)

        return dataclass_meta

    @staticmethod
    def infer_dgraph_type(dtype):
        inferred_type = dtype
        field_types = {
            str: "string",
            bool: "bool",
            int: "int",
            datetime: "dateTime",
            "default": "uid",
        }

        if isinstance(dtype, str):
            inferred_type = dtype

        if dtype in field_types.keys():
            inferred_type = field_types[dtype]

        return inferred_type

    @staticmethod
    def get_type_annotations(class_fields: List[Field]):
        result = []

        for f in class_fields:
            class_name = f.name
            dgraph_meta = dict(f.metadata).setdefault("dgraph", {})
            ignore_schema = dgraph_meta.get("ignore_schema")
            dtype = dgraph_meta.get("dg_field")

            if dtype is None:
                dtype = f.type

            if not ignore_schema:
                type_name = MetaConfig._format_types(dtype)

                result.append((class_name, type_name))

        return result

    @staticmethod
    def make_predicate_container(class_fields: List[Field]):
        predicate_container = {}

        for f in class_fields:
            class_name = f.name
            dgraph_meta = dict(f.metadata).setdefault("dgraph", {})
            predicate_meta = deepcopy(dgraph_meta)
            dtype = predicate_meta.get("dg_field")
            index = predicate_meta.get("index")

            if dtype is None:
                dtype = f.type
                predicate_meta["dg_field"] = dtype

            if index:
                predicate_container.update({class_name: predicate_meta})

        return predicate_container

    @staticmethod
    def _format_types(relation, indexing: bool = False):
        # Check if destination node is an instance of typing.Typing
        if type(relation) is typing._GenericAlias:
            dest_class_instance = relation.__args__[0]
            dest_predicate_name = MetaConfig._get_predicate_class_name(
                relation, predicate=dest_class_instance, indexing=indexing
            )
            type_name = [dest_predicate_name]

        elif is_dataclass(relation):
            dest_node = MetaConfig._get_predicate_class_name(
                relation, predicate=None, indexing=indexing
            )
            type_name = dest_node

        else:
            type_name = MetaConfig.infer_dgraph_type(relation)

        type_name = MetaConfig._format_str(type_name)
        return type_name

    @staticmethod
    def _get_predicate_class_name(relation, predicate, indexing: bool = False) -> str:
        if predicate is None:
            dest_node = relation.__name__

        elif is_dataclass(predicate):
            # Handle direct types and aliases from typing.Typing
            dest_node = relation.__args__[0].__name__
        else:
            # Handle ForwardRefs in typing
            forward_node_class = predicate.__args__[0]
            dest_node = forward_node_class.__forward_arg__

        if indexing:
            dest_node = "uid"

        dest_node = MetaConfig._format_str(dest_node)
        return dest_node

    @staticmethod
    def _format_str(st: Any) -> str:
        st = str(st).replace("'", "")
        return st

    @staticmethod
    def _format_index_string(
        index_type: List[str], directive: List[str], index_directive="@index"
    ):
        if directive is None or len(directive) == 0:
            directive_string = ""
        else:
            directive_string = " ".join(directive)

        if index_type is None or len(index_type) == 0:
            index_type_string = ""
        else:
            index_type_string = ", ".join(index_type)

        if len(index_type_string) > 0:
            index_string = f"{index_directive}({index_type_string}) {directive_string}"
        else:
            index_string = f"{directive_string}"

        return index_string


def dgconfig(
    ignore_schema: bool = False,
    index: bool = True,
    index_type: List = None,
    dg_field: str = None,
    directive: List = None,
    field_name: str = None,
) -> Dict[str, dict]:

    dgraph_meta = {}
    dataclass_meta = MetaConfig.get_dataclass_metadata(field_name=field_name)
    dataclass_metadata = dataclass_meta.setdefault("dataclasses_json", {})
    dgraph_metadata = dgraph_meta.setdefault("dgraph", {})

    if dataclass_metadata.get("letter_case") is not None:
        ignore_schema = True

    if ignore_schema:
        index = False

    inferred_dg_field = MetaConfig.infer_dgraph_type(dg_field)

    dgraph_metadata["ignore_schema"] = ignore_schema
    dgraph_metadata["index"] = index
    dgraph_metadata["index_type"] = index_type
    dgraph_metadata["dg_field"] = inferred_dg_field
    dgraph_metadata["directive"] = directive

    dgraph_meta.update(dataclass_meta)
    return dgraph_meta
