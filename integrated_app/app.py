"""ManualPal（手册通）· 集成版 Web 应用

目的：
    把「文件导入」与「知识库问答」整合到**同一个进程、同一个端口、同一个页面**，
    原 knowledge/ 工程保持不动，方便随时回顾/对照。

设计要点：
    - 直接复用 knowledge/ 里已有的服务与工具（ImportFileService / QueryService / sse_util / task_util），
      不复制业务逻辑，避免两套实现漂移。
    - 两个流程共用一个 /status 接口：因为同进程，task_util 的内存任务表是共享的，
      上传任务和问答任务都能用同一个 /status 查询（这是拆成两个服务时做不到的）。
    - 上传走「后台任务 + 轮询」，问答支持「流式(SSE) / 非流式」两种。

启动：
    python -m integrated_app.app          # 默认 http://127.0.0.1:8010
"""

import logging
import os

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from knowledge.processor.query_process.base import setup_logging
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.service.file_import_service import ImportFileService
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import sse_generator
from knowledge.utils.task_util import get_task_info, get_task_result

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="ManualPal（手册通）", description="上传 + 问答 一体化控制台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # origins="*" 时不能开 credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# 两个服务都是无状态的（真正的状态在 task_util / Mongo 里），进程内各留一个实例即可
import_service = ImportFileService()
query_service = QueryService()


# ==================================================================== #
#                              页面                                    #
# ==================================================================== #

@app.get("/", include_in_schema=False)
async def index():
    """集成页面"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ==================================================================== #
#                            文件导入                                   #
# ==================================================================== #

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """上传文件并在后台跑导入图

    返回后前端用 /status/{task_id} 轮询进度（导入流程耗时较长，不适合同步等待）。
    """
    # 1. 落本地 + MinIO（同步 IO，放线程池避免阻塞事件循环）
    task_id, file_dir, import_file_path = await run_in_threadpool(import_service.upload_file, file)

    # 2. 后台跑导入流程（同步函数，Starlette 会放到线程池执行）
    background_tasks.add_task(import_service.run_import_graph, task_id, file_dir, import_file_path)

    return {
        "message": "文件已上传，开始解析",
        "task_id": task_id,
        "filename": file.filename,
    }


# ==================================================================== #
#                            知识库问答                                 #
# ==================================================================== #

@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """提交提问

    is_stream=True  : 立刻返回 task_id，答案通过 /stream/{task_id} 以 SSE 推送
    is_stream=False : 同步跑完，直接返回答案
    """
    # 1. 会话/任务 ID
    session_id = request.session_id or query_service.generate_session_id()
    task_id = query_service.generate_task_id()

    # 2. 置为处理中；流式模式在这里建 SSE 队列（必须早于图启动，否则事件会丢）
    query_service.submit_query(task_id, request.is_stream)

    # 3. 流式：后台执行 + 立即返回
    if request.is_stream:
        background_tasks.add_task(
            query_service.run_query_graph, task_id, session_id, request.query, True
        )
        return StreamSubmitResponse(message="查询任务已提交", session_id=session_id, task_id=task_id)

    # 4. 非流式：线程池里跑完（图内 MCP 节点用 asyncio.run，不能在事件循环线程执行）
    await run_in_threadpool(
        query_service.run_query_graph, task_id, session_id, request.query, False
    )
    return QueryResponse(
        message="处理完成",
        session_id=session_id,
        answer=query_service.get_answer(task_id),
        task_id=task_id,
        error=get_task_result(task_id, "error", ""),
    )


@app.get("/stream/{task_id}")
async def stream(task_id: str, request: Request):
    """SSE：progress（节点进度）/ delta（逐字答案）/ final（完整答案）"""
    return StreamingResponse(
        sse_generator(task_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 反代时关闭缓冲，保证逐字下发
        },
    )


# ==================================================================== #
#                       状态 / 历史（两个流程共用）                      #
# ==================================================================== #

@app.get("/status/{task_id}")
async def status(task_id: str):
    """任务状态：上传任务与问答任务共用

    返回字段是两种任务需求的并集：
        status/done_list/running_list/durations —— 进度
        answer/error/image_urls                 —— 问答结果
    """
    info = get_task_info(task_id)
    info["answer"] = get_task_result(task_id, "answer", "")
    info["error"] = get_task_result(task_id, "error", "")
    info["image_urls"] = get_task_result(task_id, "image_urls", [])
    return info


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """会话历史（正序返回，前端直接渲染）"""
    items = await run_in_threadpool(query_service.get_history, session_id, limit)
    return {"session_id": session_id, "items": items}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """清空会话历史"""
    deleted = await run_in_threadpool(query_service.clear_history, session_id)
    return {"message": "历史记录已清空", "session_id": session_id, "deleted_count": deleted}


# 静态资源（如需放图片/字体）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run(app=app, host="0.0.0.0", port=8010)
