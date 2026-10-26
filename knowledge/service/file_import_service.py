import os.path
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple

from aiofiles.os import makedirs
from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import kb_import_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_running_task, add_done_task, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

logger = logging.getLogger(__name__)


class ImportFileService:

    def upload_file(self, file:UploadFile) -> Tuple[str, str, Path]:
        task_id = self._generate_task_id()
        data_path = self._get_data_path(get_local_base_dir())
        fire_dir = os.path.join(data_path, task_id)
        os.makedirs(fire_dir, exist_ok=True)  # exist_ok=True表示如果目录存在则不会报错

        add_running_task(task_id,"upload_file")
        import_file_path = self._upload_file_to_local(file, fire_dir)

        self._upload_file_to_minio(import_file_path, file.filename)
        add_done_task(task_id,"upload_file")
        return task_id, fire_dir, import_file_path

    def run_import_graph(self, task_id, file_dir, import_file_path):
        init_state = {"task_id": task_id, "file_dir": file_dir, "import_file_path": import_file_path}
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        try:
            for event in kb_import_graph.stream(init_state):
                for name_node, state in event.items():
                    print(name_node)
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"{task_id}导入文件处理失败")

    def _generate_task_id(self)->str:
        task_id = uuid.uuid4().hex[:8]
        return task_id

    def _get_data_path(self, file_dir)->str:
        data_path = os.path.join(file_dir, datetime.now().strftime("%Y%m%d"))
        return data_path

    def _upload_file_to_local(self, file, fire_dir):
        os.makedirs(fire_dir, exist_ok=True)
        try:
            import_file_path=os.path.join(fire_dir, file.filename)
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f) #带缓冲区的写效率更高

        except Exception as e:
            raise FileProcessingError(f"文件上传失败:{e}")

        return import_file_path

    def _upload_file_to_minio(self, import_file_path, file_name):
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.info(f"客户端获取失败:{e}")
            return

        bucket_name=os.getenv("MINIO_BUCKET_NAME")
        object_name=f"origin_files/{datetime.now().strftime('%Y%m%d')}/{file_name}"

        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"文件上传minio失败:{e}")
