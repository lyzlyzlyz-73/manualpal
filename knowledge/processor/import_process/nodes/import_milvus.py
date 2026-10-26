import json
import logging
import os
from dataclasses import dataclass
from logging import exception
from pathlib import Path

from typing import Tuple, List, Dict, Optional, Sequence, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType, MilvusClient, CollectionSchema

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.query_process.exceptions import ValidationError
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ScalarFieldSpec:
    """标量字段规范"""
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None

class _MilvusSchemaBuilder:

    @staticmethod
    def build(client:MilvusClient,dim:int):
        logger.info("开始构建Milvus schema")
        schema=client.create_schema(enable_dynamic_field=True)

        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )

        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )





class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        validated_chunks,dim,config=self._validate_get_inputs(state)

        milvus_client=StorageClients.get_milvus_client()
        if not milvus_client:
            return state
        collection =getattr(config,'chunks_collection')


        self._ensure_has_collection(milvus_client,collection,dim)

        inserter = _MilvusInserter(client=milvus_client,collection_name=collection)
        final_chunks = inserter.insert(chunks=validated_chunks)

        state['chunks']=final_chunks


        return state

    def _validate_get_inputs(self, state):
        self.log_step("step1", "校验输入参数")
        config=self.config
        chunks=state.get("chunks", [])
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks is required")

        validated_chunks=[]
        for chunk in chunks:
            if chunk['sparse_vector'] or chunk['dense_vector']:
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的chunk缺少向量")

        if not validated_chunks:
            raise ValidationError("chunks 都无效")

        # dim 必须是稠密向量长度(1024)；稀疏向量的大小是"非零token数",每个chunk都不同(实测75/111/77...)
        dim=len(validated_chunks[0]['dense_vector'])
        self.logger.info(f"{len(chunks)}块中导入有效块{len(validated_chunks)}向量维度为:{dim}")
        return validated_chunks,dim,config

    def _ensure_has_collection(self, milvus_client:MilvusClient, collection:str, dim:int,drop_collection:bool=False):
        self.log_step("step2", "确保集合存在")
        if drop_collection and milvus_client.has_collection(collection):
            self.logger.info(f"集合{collection}存在，已经删除")
            milvus_client.drop_collection(collection)
        if milvus_client.has_collection(collection):
            return

        schema=_MilvusSchemaBuilder.build(milvus_client,dim)
        index=_MilvusIndexBuilder.build(milvus_client,collection)


        milvus_client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index
        )


@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name:str
    datatype: DataType
    max_length: Optional[int] = None

_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="body", datatype=DataType.VARCHAR, max_length=65535),  # 要和切片数据的字段名一致(原写 content 导致 DataNotMatchException)
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)

class _MilvusSchemaBuilder():

    @staticmethod
    def build(client:MilvusClient,dim:int)->CollectionSchema:
        schema=client.create_schema(enable_dynamic_field=True)

        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        for field in _SCALAR_FIELDS:
            keyword:Dict[str:Any]={
                "field_name":field.field_name,
                "datatype":field.datatype
            }
            if field.max_length is not None:
                keyword["max_length"]=field.max_length
            schema.add_field(**keyword)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        logger.info("schema已构建")
        return schema

@dataclass(frozen=True)
class IndexScalarSpec:
    filed_name:str
    index_name:str
    index_type:str
    metric_type:str

SequenceIndexScalar:Sequence[IndexScalarSpec] = (
    IndexScalarSpec(filed_name="sparse_vector",index_name="sparse_vector_index",index_type="SPARSE_INVERTED_INDEX",metric_type="IP"),
    IndexScalarSpec(filed_name="dense_vector",index_name="dense_vector_index",index_type="AUTOINDEX",metric_type="COSINE"),
)


class _MilvusIndexBuilder():

    @staticmethod
    def build(client:MilvusClient,collection:str):
        index_params=client.prepare_index_params(collection_name=collection)

        for index_param in SequenceIndexScalar:
            keyword:Dict[str, Any] = {
                "field_name": index_param.filed_name,
                "index_name": index_param.index_name,
                "index_type": index_param.index_type,
                "metric_type": index_param.metric_type
            }
            index_params.add_index(**keyword)
        logger.info("索引参数已添加到集合")
        return index_params


class _MilvusInserter:

    def __init__(self,client:MilvusClient,collection_name: str):
        self._client=client
        self._collection_name=collection_name

    def insert(self,chunks: List[Dict[str,Any]])->List[Dict[str,Any]]:

        inserted_result=self._client.insert(
            collection_name=self._collection_name,
            data=chunks
        )
        inserted_count = inserted_result.get('insert_count')
        ids = inserted_result.get('ids')
        logger.info(f"插入结果: {inserted_count} 记录, ids: {ids}")
        logger.info(f"插入结果: {inserted_count} 记录, ids: {ids}")

        # 2. 回填id
        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入{inserted_count}记录,并且回填chunk_id到chunk中")
        return chunks

    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, id in zip(chunks, ids):
            chunk["chunk_id"] = id

def _cli_main() -> None:
    setup_logging()

    temp_dir = Path(r"G:\LLL\LLL\2.资料\pdf文档\test_runimportgraph\万用表RS-12的使用\auto")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

    state: ImportGraphState = {
        "chunks": content.get("chunks", [])
    }

    import_milvus = ImportMilvusNode()
    result_state = import_milvus.process(state)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result_state, fh, ensure_ascii=False, indent=4)

    logger.info(f"备份临时文件{output_path}成功")


if __name__ == "__main__":
    _cli_main()