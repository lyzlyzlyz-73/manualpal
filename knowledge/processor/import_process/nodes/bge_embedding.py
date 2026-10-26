import json
import os
from logging import exception
from pathlib import Path

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

class BgeEmbeddingNode(BaseNode):
    name="bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 验证输入
        state_chunks, config = self._validate_input(state)
        embedding_batch_size = config.embedding_batch_size

        len_chunks = len(state_chunks)
        final_chunks = []
        for i in range(0, len_chunks, embedding_batch_size):
            part_chunks = state_chunks[i:i + embedding_batch_size]
            embedding_content_part_chunks = self._embeding_content_part_chunks(part_chunks, i, len_chunks)
            final_chunks.extend(embedding_content_part_chunks)
        state["chunks"] = final_chunks
        return state

    def _validate_input(self, state: ImportGraphState):
        self.log_step("step1", "校验输入参数")
        chunks = state.get("chunks", [])
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks is required")
        config = self.config
        return chunks, config

    def _embeding_content_part_chunks(self, part_chunks: List[Dict], i: int, total_len_chunks: int) -> List[Dict]:
        embedding_contents: List[str] = []
        for _, chunk in enumerate(part_chunks):
            item_name = chunk.get("item_name")
            content = chunk.get("body")
            embedding_content = f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)

        try:
            emb3_client = AIClients.get_bge_m3_client()
            vector_result = emb3_client.encode_documents(documents=embedding_contents)
            if not vector_result:
                return part_chunks

        except Exception as e:
            self.logger.error(f"{self.name}, 获取emb3嵌入模型客户端失败{e}")
            return part_chunks

        for index, chunk in enumerate(part_chunks):
            dense_vector = vector_result["dense"][index].tolist()
            start, end = vector_result["sparse"].indptr[index:index + 2]
            token_id = vector_result["sparse"].indices[start:end].tolist()
            weight = vector_result["sparse"].data[start:end].tolist()
            sparse_vector = dict(zip(token_id, weight))
            part_chunks[index]["dense_vector"] = dense_vector
            part_chunks[index]["sparse_vector"] = sparse_vector
            self.logger.info(f"已经将{index + 1 + i}-{i + len(part_chunks)}/{total_len_chunks}批次数据嵌入向量")
        return part_chunks

if __name__ == '__main__':
    setup_logging()

    base_temp_dir = Path(r"G:\LLL\LLL\2.资料\pdf文档\test_runimportgraph\万用表RS-12的使用\auto")

    input_path = base_temp_dir / "chunks_item_name.json"  # ← 改：必须用带 item_name 的那份
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")  # ← 改：原来只 print，会继续往下崩

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态
    state = {
        "chunks": content
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingNode()  # ← 注意类名，见下
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测试完成! 结果已成功备份至:\n{output_path}")

