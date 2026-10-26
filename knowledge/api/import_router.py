import os.path

from fastapi import FastAPI, UploadFile, Depends, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging

from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.utils.task_util import get_task_info


def get_router(app: FastAPI):
    @app.get("/import")
    def import_page():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    @app.post("/upload", response_model=UploadResponse)  # 注意拼写：response_model（原 reponse_model 会 TypeError 导致服务起不来）
    async def upload_file(
            background_tasks: BackgroundTasks,
            service=Depends(get_import_file_service),
            file: UploadFile = File(...)):
        # file_dir文减价，import_file_path上传后完整路径
        task_id, file_dir, import_file_path = service.upload_file(file)
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)
        return UploadResponse(task_id=task_id, message="文件上传处理已经完成")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)


def get_app():
    app = FastAPI(title="ManualPal · Import Service", description="手册通 · 文件导入服务")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # 如果为True，则3ge不能为"*"
        allow_methods=["*"],
        allow_headers=["*"],
    )
    front_page_dir = get_front_page_dir()
    # 前端静态路径挂载
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    get_router(app)

    return app


if __name__ == "__main__":
    import uvicorn
    setup_logging()
    uvicorn.run(app=get_app(), host="0.0.0.0", port=8000)
