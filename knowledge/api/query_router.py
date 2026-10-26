"""查询流程 API 路由（端口 8001）

与 import_router 相同的组织方式：get_router(app) 注册接口，get_app() 组装应用。

接口:
    GET    /chat                  聊天页面
    POST   /query                 提交查询（流式返回 task_id / 非流式直接返回答案）
    GET    /stream/{task_id}      SSE 流式输出（progress / delta / final）
    GET    /status/{task_id}      任务状态与结果（轮询兜底）
    GET    /history/{session_id}  历史对话
    DELETE /history/{session_id}  清空历史对话
"""

import logging
import os

from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import sse_generator
from knowledge.utils.task_util import get_task_info, get_task_result

logger = logging.getLogger(__name__)


def get_router(app: FastAPI) -> None:
    @app.get("/chat")
    async def chat_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query")
    async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            service: QueryService = Depends(get_query_service),
    ):
        # 1. 解析参数（前端没传 session_id 时新建会话）
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()

        # 2. 提交任务：置为处理中；流式模式下在这里建 SSE 队列（必须早于图启动）
        service.submit_query(task_id, request.is_stream)

        # 3. 流式：丢后台执行，立刻返回 task_id
        if request.is_stream:
            background_tasks.add_task(
                service.run_query_graph, task_id, session_id, request.query, True
            )
            return StreamSubmitResponse(
                message="查询任务已提交", session_id=session_id, task_id=task_id
            )

        # 4. 非流式：放到线程池里同步跑完
        #    （图里的 MCP 节点用了 asyncio.run，不能在事件循环线程里直接执行）
        await run_in_threadpool(
            service.run_query_graph, task_id, session_id, request.query, False
        )
        return QueryResponse(
            message="处理完成",
            session_id=session_id,
            answer=service.get_answer(task_id),
            task_id=task_id,
            error=get_task_result(task_id, "error", ""),
        )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        """SSE 实时推送：progress（节点进度）/ delta（逐字答案）/ final（完整答案）"""
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，保证逐字下发
            },
        )

    @app.get("/status/{task_id}")
    async def get_status(task_id: str):
        """轮询兜底：任务状态 + 已完成/进行中节点 + 结果（含失败原因）"""
        task_info = get_task_info(task_id)
        task_info["answer"] = get_task_result(task_id, "answer", "")
        task_info["error"] = get_task_result(task_id, "error", "")
        task_info["image_urls"] = get_task_result(task_id, "image_urls", [])
        return task_info

    @app.get("/history/{session_id}")
    async def history(
            session_id: str,
            limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        # pymongo 是阻塞调用，放线程池
        items = await run_in_threadpool(service.get_history, session_id, limit)
        return {"session_id": session_id, "items": items}

    @app.delete("/history/{session_id}")
    async def delete_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        deleted = await run_in_threadpool(service.clear_history, session_id)
        return {"message": "历史记录已清空", "session_id": session_id, "deleted_count": deleted}


def get_app() -> FastAPI:
    app = FastAPI(title="ManualPal · Query Service", description="手册通 · 知识库查询服务")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # 与 import_router 一致：origins="*" 时不能开 credentials
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 前端静态路径挂载（/front/chat.html 也可直接访问）
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    get_router(app)
    return app


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run(app=get_app(), host="0.0.0.0", port=8001)
