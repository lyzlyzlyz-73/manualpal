import json
import logging
from typing import List, Tuple, Union, Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients#get_llm_openai,get_bge_m3_client
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.storage_clients import StorageClients#get_milvus_client
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query
)
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyDeSearchNode(BaseNode):
    """HyDE 检索节点

    流程: 参数校验 → LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
    """
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """执行 HyDE 检索

        Args:
            state: 需包含 rewritten_query 和 item_names

        Returns:
            {"hyde_embedding_chunks": [...]} 搜索结果列表
        """
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        # 3. 获取嵌入模型 & milvus 客户端
        embedding_model = AIClients.get_bge_m3_client()
        milvus_client = StorageClients.get_milvus_client()
        if not embedding_model or not milvus_client:
            return state

        # 4. 假设性文档嵌入(注入问题+假设性文档)
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_bge_m3_hybrid_vectors(
            embedding_model,
            embedding_documents=[embedding_document]
        )

        if not embedding_result:
            return state

        # 5. 获取 item_name 的过滤表达式
        item_name_filtered_expr = self._item_name_filte_expr(validate_item_names)

        # 6. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filtered_expr
        )

        # 7. 执行混合搜索请求
        reps = execute_hybrid_search_query(
            milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id", "body", "item_name", 'title']
        )

        if not reps or not reps[0]:
            return state

        # 8. 只更新 hyde_embedding_chunks
        return {"hyde_embedding_chunks": reps[0]}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        return rewritten_query, item_names

    def _item_name_filte_expr(self, validate_item_names: List[str]) -> str:
        """构建商品名过滤表达式"""
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        return f" item_name in [{quoted}]"


    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:
        # 1. 获取 LLM 客户端
        llm_client = AIClients.get_llm_openai(False)

        # 2. 判断
        if llm_client is None:
            return ""

        # 3. 获取系统提示词以及用户提示词
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names="、".join(validate_item_names),
            rewritten_query=validated_query
        )
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"

        try:
            # 4. 获取 AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, 'content', '').strip()

            # 6. 判断是否存在
            if not llm_response_content:
                return ""

            return llm_response_content

        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""

# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS PRO RS-12 数字万用表"],  # 必须与 kb_chunks_v1 里的标准名逐字一致，否则过滤 0 条
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        body = entity.get("body", "")
        print(f"      内容: {body[:80]}...")

    print("-" * 60)
    print("测试完成")


# ================================================================== #
#                    节点速查（输入 / 处理 / 输出）                    #
# ------------------------------------------------------------------ #
# 【拿了什么数据】state 里的
#   - rewritten_query: 改写后的用户问题
#   - item_names:      已对齐的标准商品名列表（过滤用）
#   两者缺失或类型不对 → 抛 StateFieldError，本节点不产出任何字段
#
# 【用了什么模型 / 服务】
#   - LLM（AIClients.get_llm_openai(False)，即 DashScope 上的 qwen-flash）
#     用 HYDE_USER_PROMPT_TEMPLATE 生成 200~300 字的"假设性文档"
#   - BGE-M3（AIClients.get_bge_m3_client()）：把 "原问题\n假设文档" 编码成 稠密 + 稀疏 双向量
#   - Milvus（StorageClients.get_milvus_client()）：集合 config.chunks_collection
#     过滤条件：item_name in [...item_names]
#
# 【做了什么事情】
#   校验 → LLM 生成假设文档 → 与原问题拼接后向量化 → Milvus 混合检索
#   （HyDE 的本质：让"问题"先变成"像文档的文本"，再去和切片比相似度）
#
# 【更新了什么字段】
#   state['hyde_embedding_chunks'] = reps[0]
#   函数返回 {"hyde_embedding_chunks": [...]}
#   每条结构与向量检索节点完全相同（同一批切片的另一种召回路径）
#
#   结构示例（真实产物，body 已截断）—— 实测命中 5 条:
#   [
#     {
#       "chunk_id": 469746418373863607,       # Milvus 主键
#       "distance": 0.8099275827407837,       # 混合检索融合相似度
#       "entity": {
#         "chunk_id": 469746418373863607,
#         "body": "## 直流电流测量\n\n注意：在10A情况下测量时间不能超过30秒...\n
#                  ![数字万用表测量直流电流的接线示意图]()",
#         "item_name": "RS PRO RS-12 数字万用表",
#         "title": "## 直流电流测量"
#       }
#     },
#     ... 共 5 条，按 distance 倒序
#   ]
#
# 【不更新字段时的返回】
#   - LLM 生成失败 → 假设文档为空串，自动退化成"纯原问题"照常检索（降级，不算失败）
#   - 嵌入模型/Milvus 客户端为空、嵌入结果为空、检索 0 条 → return state
#     （不写 hyde_embedding_chunks，保持默认 []）
# ================================================================== #