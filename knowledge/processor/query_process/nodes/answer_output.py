from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.task_util import set_task_result
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.mongo_history_util import save_chat_message


class AnswerOutputNode(BaseNode):
    """答案输出节点

    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件
    """
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行答案生成

        Args:
            state: 需包含 task_id、session_id、reranked_docs 等

        Returns:
            更新后的 state，包含 answer 和 prompt
        """
        task_id = state.get("task_id")
        is_stream = state.get("is_stream")

        # 1. 已有答案 → 直接返回
        if state.get("answer"):
            self._push_existing_answer(state)

        # 2. 构建提示词 → 调用 LLM 生成答案
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(state, prompt)

        # 3. 写入历史记录
        self._write_history(state)

        # 4. 流式模式发送结束事件
        #    全流程只在这里发一次 FINAL（BaseNode 只推 progress，不推 FINAL）
        if is_stream:
            push_sse_event(task_id, SSEEvent.FINAL,
                           {"answer": state.get("answer", "")})
        return state

    # ================================================================== #
    #                    已有答案推送                                       #
    # ================================================================== #

    def _push_existing_answer(self, state: QueryGraphState):
        """非流式模式：存入任务结果；流式模式：让 FINAL 统一推送。"""
        if not state.get("is_stream"):
            set_task_result(state["task_id"], "answer", state["answer"])

    # ================================================================== #
    #                    提示词构建                                         #
    # ================================================================== #

    def _build_prompt(self, state: QueryGraphState) -> str:
        """根据检索结果、历史对话组装 LLM 提示词"""
        char_budget = self.config.max_context_chars

        # 1. 获取问题和商品名
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 5. 组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=", ".join(item_names),
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            # rerank 节点产出的文本字段是 body（chunks 里没有 content 字段）
            content = doc.get("body", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        # Mongo 按时间倒序返回，拼【历史对话】时按时间正序，避免模型读到的对话是倒着的
        for message in reversed(chat_history):
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"

            if used_chars + len(formatted_line) > char_budget:
                # 预算不够就停（这里必须 break：return 不带值会让调用处解包失败）
                break

            formatted_lines.append(formatted_line)
            used_chars += len(formatted_line) + 1

        return "\n".join(formatted_lines), char_budget - used_chars

    # ================================================================== #
    #                    LLM 生成                                          #
    # ================================================================== #

    def _generate_answer(self, state, prompt):
        """调用 LLM 生成答案（流式/非流式）"""
        self.log_step("generate", "生成答案")
        llm_client = AIClients.get_llm_openai(False)
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")

        task_id = state["task_id"]

        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(llm_client, prompt)
            set_task_result(task_id, "answer", state["answer"])

    def _invoke_generate(self, llm_client, prompt: str) -> str:
        """非流式生成"""
        self.log_step("generate", "生成答案")

        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    def _stream_generate(self, llm_client, prompt, task_id):
        """流式生成，逐 chunk 推送 delta 事件"""
        accumulated_answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id, "delta", {"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return accumulated_answer

    # ================================================================== #
    #                    历史记录                                           #
    # ================================================================== #

    def _write_history(self, state: QueryGraphState):
        """将用户问题和助手回答写入 MongoDB 历史记录"""
        session_id = state["session_id"]
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_names = state.get("item_names") or []

        try:
            # 1. 写用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            # 2. 写助手回复
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                )
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")



if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    load_dotenv()

    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("开始测试: 答案生成节点 (AnswerOutputNode)")
    print("=" * 60)

    # 构造模拟状态
    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": False,
        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",
        "item_names": ["RS-12数字万用表"],
        "reranked_docs": [
            {
                "body": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            },
            {
                "body": "测量直流电压时需注意正负极性，红表笔接正极，黑表笔接负极。",
                "source": "web",
                "url": "https://example.com/guide",
                "title": "电压测量指南",
                "score": 0.8756
            }
        ],
        "history": [
            {"role": "user", "text": "万用表是什么？"},
            {"role": "assistant", "text": "万用表是一种多功能电子测量仪器..."}
        ],
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  item_names: {mock_state['item_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")
    print("-" * 60)

    # 执行答案生成
    node = AnswerOutputNode()
    result = node.process(mock_state)

    # 打印结果
    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)

    print("\n测试完成")