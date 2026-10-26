from typing import Dict, Any, List, Tuple

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, setup_logging


class RrfNode(BaseNode):
    """RRF 融合排序节点

    流程: 收集多路检索结果 → 带权重 RRF 融合 → 按得分降序返回
    """
    name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 RRF 融合

        Args:
            state: 需包含 embedding_chunks 和 hyde_embedding_chunks

        Returns:
            更新后的 state，包含 rrf_chunks
        """
        # 1. 各路搜索的结果
        # 1.1 获取向量检索路的结果
        vector_search_chunks = state.get('embedding_chunks') or []
        # 1.2 获取 hyde 向量检索路的结果
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []

        # 2. 为不同路的搜索结果设置不同的权重
        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result": (self._normalize_input(hyde_search_chunks), 1.0),
        }

        # 3. 构建 rrf_inputs
        rrf_inputs = list(search_source.values())

        # 4. 利用 RRF 的计算公式去获取到所有路查询到的所有 chunk 对应的 score
        rrf_merge_results: List[Tuple[Dict[str, Any], float]] = self._rrf_merge(
            rrf_inputs, self._rrf_k, self._top_k
        )

        # 5. 获取 rrf_chunks（只取文档，不要分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新 state
        state['rrf_chunks'] = rrf_chunks

        # 8. 返回 state
        return state

    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        统一处理各路检索到的结果

        Args:
            rrf_input: 各路不同数据结构的检索结果

        Returns:
            统一处理后的标准数据结构的检索结果
        """
        diff_path_result = []

        # 1. 遍历各路搜索结果
        if not rrf_input:
            return []

        # 2. 遍历该路的所有结果
        for doc in rrf_input:
            # 2.1 判断是否有效
            if not isinstance(doc, dict):
                continue

            # 2.2 获取 entity
            entity = doc.get('entity')
            if not entity:
                continue

            diff_path_result.append(entity)

        return diff_path_result

    def _rrf_merge(self, rrf_inputs, _rrf_k, _top_k) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用 RRF 公式计算每一个文档的总得分

        Args:
            rrf_inputs: 各路的搜索结果 + 权重
            _rrf_k: 平滑参数，通常取 60
            _top_k: 合并完之后返回的文档数

        Returns:
            合并以及排序后的文档列表
        """
        chunk_scores = {}  # 存放所有 chunk 的 RRF 计算后的分数值
        chunk_data = {}    # 存放所有 chunk 的文档数据

        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue

                # RRF 公式: score += weight / (k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (_rrf_k + i)

                # 使用 setdefault 保留首次遇到的文档版本
                chunk_data.setdefault(chunk_id, doc)

        # 按得分降序排序动态截取前 top_k 条
        sorted_results = sorted(
            [(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()],
            key=lambda x: x[1],#根据score进行排序
            reverse=True
        )

        return sorted_results[:_top_k] if _top_k else sorted_results


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")


# ================================================================== #
#                    节点速查（输入 / 处理 / 输出）                    #
# ------------------------------------------------------------------ #
# 【拿了什么数据】state 里的
#   - embedding_chunks:      向量检索路的原始命中（vector_search 节点产出）
#   - hyde_embedding_chunks: HyDE 检索路的原始命中（hyde_search 节点产出）
#   取值为 None 时用 `or []` 兜底；两路都缺失也不报错，只是融合结果为空
#
# 【用了什么模型 / 服务】
#   - 不需要模型、也不访问任何服务：纯算法融合（RRF = Reciprocal Rank Fusion）
#   - 只用到两个配置项（config.py，可用 .env 覆盖）：
#       RRF_K（默认 60）            平滑参数 k
#       RRF_MAX_RESULTS（默认 10）  融合后保留条数 top_k
#   - 两路权重目前都写死 1.0（要调某一路话语权，改 search_source 里的数值）
#
# 【做了什么事情】
#   1. _normalize_input：把两路的三层命中 {chunk_id, distance, entity} 拍平成 entity
#      （丢掉 distance —— RRF 只看名次，不看各路分值）
#   2. _rrf_merge：对每一路按名次累加分数
#        score += weight / (k + rank)      # enumerate 从 1 开始，rank 即名次
#      同一 chunk_id 跨路出现则分数累加（多路共识 → 排名被抬高）；
#      chunk_data.setdefault 保证同一 chunk 只保留"首次遇到"的那份正文
#   3. 按总分倒序排序，再截取前 top_k 条
#
# 【更新了什么字段】
#   state['rrf_chunks'] = 融合后的文档列表（已按分数倒序）
#   注意：这是"纯文档列表"，不含 rrf_score，排序信息只体现在列表顺序上；
#   分数只出现在日志里（"RRF 融合完成"/"分数范围" 两条 INFO，需 setup_logging 才可见）
#
#   结构示例（真实产物，body 已截断）—— 两路各 5 条 → 融合后 5 条:
#   [
#     {
#       "chunk_id": 469746418373863607,       # 来自 _normalize_input 拍平后的 entity
#       "body": "## 直流电流测量\n\n注意：在10A情况下测量时间不能超过30秒...\n
#                ![数字万用表测量直流电流的接线示意图]()",
#       "item_name": "RS PRO RS-12 数字万用表",
#       "title": "## 直流电流测量"
#     },
#     {
#       "chunk_id": 469746418373863605,
#       "body": "## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量...",
#       "item_name": "RS PRO RS-12 数字万用表",
#       "title": "## 直流电压测量"
#     },
#     ...
#   ]
#   对比上游：embedding_chunks/hyde_embedding_chunks 是"带 distance 的三层命中"，
#   rrf_chunks 是"拍平后的纯文档"，distance 已被丢弃、改由列表顺序表达排名
#
# 【不更新字段 / 无结果时的返回】
#   - 本节点没有 try/except 兜底：
#       两路都为空 → 融合结果为空列表，state['rrf_chunks'] 会被写成 []（不是"不更新"）
#       内部异常 → 向上冒泡，节点被标记失败
#   - 与 vector / hyde 不同：那两个节点失败时是"原样返回 state、不写字段"，
#     本节点无论有没有结果，都会写 state['rrf_chunks']
# ================================================================== #