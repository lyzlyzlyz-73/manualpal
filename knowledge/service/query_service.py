"""查询服务

把 Web 层的查询逻辑从路由里抽出来：任务/会话 ID、任务提交、图执行、结果与历史读取。
与 file_import_service.py 同一层，路由里通过 Depends(get_query_service) 注入。
"""

import logging
import uuid
from typing import Any, Dict, List

from knowledge.processor.query_process.main_graph import query_app
from knowledge.processor.query_process.state import create_default_state
from knowledge.utils.mongo_history_util import clear_history, get_recent_messages
from knowledge.utils.sse_util import create_sse_queue, push_sse_event, SSEEvent
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)

logger = logging.getLogger(__name__)


class QueryService:
    """查询服务：任务编排 + 会话历史"""

    # ==================== ID 生成 ====================

    @staticmethod
    def generate_session_id() -> str:
        """会话ID（前端没传时兜底生成）"""
        return uuid.uuid4().hex

    @staticmethod
    def generate_task_id() -> str:
        """任务ID：与导入任务保持一致，取 8 位短码"""
        return uuid.uuid4().hex[:8]

    # ==================== 任务提交与执行 ====================

    def submit_query(self, task_id: str, is_stream: bool) -> None:
        """提交查询任务

        流式模式必须在这里（图启动之前）建好 SSE 队列，
        否则节点先跑、事件先推，队列还不存在 → 事件全部丢失。
        """
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    def run_query_graph(self, task_id: str, session_id: str, user_query: str, is_stream: bool) -> Dict[str, Any]:
        """执行 LangGraph 查询流程（同步函数，由后台任务或线程池调用）

        Args:
            task_id: 任务ID
            session_id: 会话ID
            user_query: 用户原始问题
            is_stream: 是否流式输出

        Returns:
            图执行后的最终状态（失败时返回空字典）
        """
        # 1. 组装初始状态（用 state.py 的工厂，保证所有字段都有默认值）
        init_state = create_default_state(
            original_query=user_query,
            session_id=session_id,
            task_id=task_id,
            is_stream=is_stream,
        )

        try:
            # 2. 执行图
            result = query_app.invoke(init_state)
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            return result
        except Exception as e:
            # 3. 失败兜底：写状态/错误，并给前端补一个 final，避免 SSE 一直挂着
            logger.error(f"查询流程执行失败: {e}", exc_info=True)
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(e))
            if is_stream:
                push_sse_event(
                    task_id,
                    SSEEvent.FINAL,
                    {"answer": f"抱歉，本次处理失败：{e}", "status": TASK_STATUS_FAILED},
                )
            return {}
        finally:
            # 4. 流式模式补最后一条 progress：让前端把"进行中"的节点收尾
            if is_stream:
                push_sse_event(
                    task_id,
                    SSEEvent.PROGRESS,
                    {
                        "status": get_task_status(task_id),
                        "done_list": get_done_task_list(task_id),
                        "running_list": get_running_task_list(task_id),
                    },
                )

    # ==================== 结果读取 ====================

    @staticmethod
    def get_answer(task_id: str) -> str:
        """取任务答案（非流式接口用）"""
        return get_task_result(task_id, "answer", "")

    # ==================== 历史会话 ====================

    @staticmethod
    def get_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """读取会话历史

        MongoDB 按时间倒序返回，前端按数组顺序渲染，所以这里反转成正序。
        """
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(record.get("_id", "")),
                "session_id": record.get("session_id", ""),
                "role": record.get("role", ""),
                "text": record.get("text", ""),
                "rewritten_query": record.get("rewritten_query", ""),
                "item_names": record.get("item_names", []),
                "ts": record.get("ts"),
            }
            for record in reversed(records)
        ]

    @staticmethod
    def clear_history(session_id: str) -> int:
        """清空会话历史，返回删除条数"""
        return clear_history(session_id)
