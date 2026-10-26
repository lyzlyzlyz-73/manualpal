import json
import os

from typing import Tuple, List, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.query_process.exceptions import ValidationError
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients




class ItemNameRecognitionNode(BaseNode):
    name="item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title, item_name_chunks_k, item_name_chunks_size, chunks = self._validate_state(state)

        recognition_context = self._prepare_item_recognition_context(item_name_chunks_k, item_name_chunks_size, chunks)

        item_name = self._recognition_name(file_title, recognition_context)

        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        self._insert_milvus(dense_vector, sparse_vector, item_name, file_title, self.config.item_name_collection)

        self._fill_item_name(chunks, state, item_name)

        self._backup_chunks(state, chunks)
        return state

    def _validate_state(self, state) -> Tuple[str, int, int, list]:
        file_title = state.get("file_title", "")
        chunks = state.get("chunks", [])
        item_name_chunks_k = self.config.item_name_chunk_k
        item_name_chunks_size = self.config.item_name_chunk_size
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError("chunks is required")
        if not file_title:
            raise StateFieldError("file_title is required")
        if not item_name_chunks_k or item_name_chunks_k <= 0:
            raise ValidationError("item_name_chunk_k is required and must be greater than 0")
        if not item_name_chunks_size or item_name_chunks_size <= 0:
            raise ValidationError("item_name_chunk_size is required and must be greater than 0")
        return file_title, item_name_chunks_k, item_name_chunks_size, chunks

    def _prepare_item_recognition_context(self, item_name_chunks_k, item_name_chunks_size, chunks):
        total = 0
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunks_k]):
            if not isinstance(chunk, dict):
                continue
            context = f"【切片】-{index+1}-{chunk['body']}"
            if len(context) + total > item_name_chunks_size:
                break
            if len(context) + total > item_name_chunks_size and not final_context:
                raise ValidationError(f"{self.name},请检查item_name_chunks_size和max_content_length是否冲突")
            final_context.append(context)
            total += len(context)

        return "\n".join(final_context)

    def _recognition_name(self, file_title, recognition_context):
        try:
            llm_client = AIClients.get_llm_openai(response_format=False)
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=recognition_context)
            response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            response = response.content.strip()
            if not response or response == "UNKNOWN":
                self.logger.info(f"{self.name}, 获取名称失败，降级采用文件名")
                return file_title
            self.logger.info(f"{self.name}, 获取名称成功:{response}")
            return json.loads(response).get("item_name")
        except Exception as e:
            self.logger.error(f"{self.name}, 获取名称失败，降级采用文件名，{e}")
            return file_title

    def _embedding_item_name(self, item_name) -> Tuple[List[float], Dict[int, float]]:
        try:
            emb3_client = AIClients.get_bge_m3_client()
            vector_result = emb3_client.encode_documents([item_name])  # 向量化结果

            dense_vector = vector_result["dense"][0].tolist()
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))
            return dense_vector, sparse_vector

        except ConnectionError as e:
            self.logger.error(f"{self.name}, BGE-M3客户端获取失败{e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名{item_name}向量化失败{e}")
            return None, None

    def _insert_milvus(self, dense_vector: List[float], sparse_vector: Dict[int, float], item_name: str,
                       file_title: str, item_name_collection):
        if not dense_vector or not sparse_vector:
            self.logger.error(f"{file_title}对应的商品名{item_name}, 向量不完整")
            return

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"{self.name}, Milvus客户端获取失败{e}")
            return

        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(milvus_client, item_name_collection)
                self.logger.info(f"{self.name}, Milvus集合创建成功")
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"{self.name}, Milvus集合插入成功，结果:{result['ids'][0]}")

        except Exception as e:
            self.logger.error(f"{self.name}, Milvus集合创建失败{e}")
            return

    def _create_item_name_collection(self, milvus_client, item_name_collection):
        schema = milvus_client.create_schema()

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_name="dense_vector_index",
                               index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(collection_name=item_name_collection, schema=schema, index_params=index_params)

        self.logger.info(f"{self.name}, {item_name_collection}创建成功并构建了索引")

    def _fill_item_name(self, chunks, state: ImportGraphState, item_name):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name

    def _backup_chunks(self, state, chunks):
        """
        将回填item_name的切片列表结果备份到json文件，给下个节点单元测试使用。
        :param state:
        :param chunks:
        :return:
        """
        # 备份到 md 所在目录（与 _backup.md 同级）；没有 md_path 时回退到 file_dir
        md_path = state.get("md_path", "")
        local_dir = os.path.dirname(md_path) if md_path else state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 md_path/file_dir，跳过备份")
            return  # ★ 空就跳过，别去建目录

        try:
            os.makedirs(local_dir, exist_ok=True)  # ★ 移进 try：建目录失败也能被接住
            output_path = os.path.join(local_dir, "chunks_item_name.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
            self.logger.info(f"已备份到: {output_path}")  # ★ 补个成功日志
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"G:\LLL\LLL\2.资料\pdf文档\test_runimportgraph\万用表RS-12的使用\auto\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表RS-12的使用",
        "file_dir":r"G:\LLL\LLL\2.资料\pdf文档\test_runimportgraph\万用表RS-12的使用\auto",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")



