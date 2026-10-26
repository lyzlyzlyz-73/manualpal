"""向量检索节点

对用户查询进行向量化，在 Milvus 中执行混合搜索（稠密 + 稀疏），返回相关切片。
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, item_names_filter


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_state(state)

        # 2. 获取嵌入模型
        try:
            embedding_molde = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return state

        # 3. 获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return state

        # 4. 对问题嵌入
        try:
            embed_query = generate_bge_m3_hybrid_vectors(model=embedding_molde, embedding_documents=[validated_query])
        except Exception as e:
            self.logger.error(f"问题{validated_query}嵌入失败")
            return state

        # 5. 构建过滤表达式以及表达式参数
        filter_expr, filter_expr_param = item_names_filter(validate_item_names)

        try:
            # 6. 创建混合请求
            hybrid_search_request = create_hybrid_search_requests(
                dense_vector=embed_query['dense'][0],
                sparse_vector=embed_query['sparse'][0],
                expr=filter_expr,
                expr_params=filter_expr_param  # 参数名是 expr_params（原 filter_expr_params 会 TypeError）
            )

            # 7. 执行混合搜索请求
            hybrid_search_reps = execute_hybrid_search_query(milvus_client=milvus_client,
                                                             collection_name=self.config.chunks_collection,
                                                             search_requests=hybrid_search_request,
                                                             output_fields=['chunk_id', 'body', 'item_name',
                                                                            'title'])  # content -> body（集合里实际字段是 body）

            # 8. 获取搜索结果
            if not hybrid_search_reps or not hybrid_search_reps[0]:
                return state

            # 9. 更新state
            state['embedding_chunks'] = hybrid_search_reps[0]

            # 10. 返回
            return {
                "embedding_chunks": state['embedding_chunks']
            }
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return state

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """
        校验输入参数
        Args:
            state:

        Returns:

        """
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        # 3. 返回
        return rewritten_query, item_names


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS PRO RS-12 数字万用表"]  # 必须与 kb_item_names_v1 里的标准名精确一致（生产中由 item_name_confirm 节点对齐产出）
    }

    vector_search = VectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))


# ================================================================== #
#                    节点速查（输入 / 处理 / 输出）                    #
# ------------------------------------------------------------------ #
# 【拿了什么数据】state 里的
#   - rewritten_query: 改写后的用户问题（由 item_name_confirm 节点产出）
#   - item_names:      已对齐的标准商品名列表（同上）
#   两者缺失或类型不对 → 抛 StateFieldError，本节点不产出任何字段
#
# 【用了什么模型 / 服务】
#   - BGE-M3（AIClients.get_bge_m3_client()）：把 query 编码成 稠密 + 稀疏 双向量
#   - Milvus（StorageClients.get_milvus_client()）：集合 config.chunks_collection
#     过滤条件：item_name in [...item_names]
#     融合方式：WeightedRanker(0.5, 0.5) + norm_score（稠密 COSINE / 稀疏 IP）
#
# 【做了什么事情】
#   校验 → query 向量化 → 构造 item_name 过滤表达式 → Milvus 混合检索
#   → 取第 0 个查询的命中列表
#
# 【更新了什么字段】
#   state['embedding_chunks'] = hybrid_search_reps[0]
#   函数返回 {"embedding_chunks": [...]}
#
#   结构示例（真实产物，body 已截断）—— 实测命中 5 条:
#   [
#     {
#       "chunk_id": 469746418373863607,       # Milvus 主键
#       "distance": 0.7604104280471802,       # 混合检索融合相似度（越大越相关）
#       "entity": {
#         "chunk_id": 469746418373863607,
#         "body": "## 直流电流测量\n\n注意：在10A情况下测量时间不能超过30秒...\n
#                  ![数字万用表测量直流电流的接线示意图](http://host:post/xxxx/.../xxx.jpg)",
#         "item_name": "RS PRO RS-12 数字万用表",
#         "title": "## 直流电流测量"
#       }
#     },
#     ... 共 5 条，按 distance 倒序
#   ]
#
# 【不更新字段时的返回】
#   - 嵌入模型获取失败 / Milvus 客户端获取失败 / query 嵌入异常 / 检索异常 / 检索 0 条
#     → return state（原样返回传入的 state，embedding_chunks 保持默认 []，不抛错不中断）
# ================================================================== #
