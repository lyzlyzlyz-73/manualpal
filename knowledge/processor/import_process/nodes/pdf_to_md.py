import json
import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.query_process.exceptions import ValidationError


class PdfToMdNode(BaseNode):
    name="pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        pdf_to_md
        将pdf转换为md
        :param state: {
        "import_file_path":import_file_path,
        "file_dir":file_dir,
        "is_md_read_enabled":False,
        "is_pdf_read_enabled":True
    }
        :return:
        """
        #1.用函数进行验证,并得到path对象
        import_file_path_obj, file_dir_path_obj = self._validate_state_inputs_path(state)

        #2.执行MinerU，转换为md
        processed_code=self._excute_maneru(import_file_path_obj, file_dir_path_obj)
        if processed_code == 0:
            self.logger.info(f"MinerU转换成功,返回码: {processed_code},{self.name}")
        else:
            self.logger.error(f"MinerU转换失败,返回码: {processed_code},{self.name}")
            raise PdfConversionError(f"MinerU转换失败,返回码: {processed_code},{self.name}")
        #3.获取md路径
        md_path=self._get_md_paths(file_dir_path_obj, import_file_path_obj)

        #4.更行md路径并返回
        state["md_path"]=md_path


        return state

    def _validate_state_inputs_path(self, state:ImportGraphState)->Tuple[Path,Path]:

        import_file_path=state.get("import_file_path","")
        file_dir_path=state.get("file_dir","")

        if not import_file_path:
            raise ValidationError("输入文件路径为空",self.name)
        if not file_dir_path:
            raise ValidationError("输出目录为空",self.name)

        file_dir_path_obj=Path(file_dir_path)
        import_file_path_obj=Path(import_file_path)

        if not import_file_path_obj.exists():
            raise FileProcessingError(f"输入文件路径不存在: {import_file_path_obj}",self.name)
        if not file_dir_path_obj.exists():
            file_dir_path_obj=import_file_path_obj.parent
        return import_file_path_obj,file_dir_path_obj

    def _excute_maneru(self, import_file_path_obj, file_dir_path_obj)->int:
        proc = subprocess.Popen(
            args=["mineru", "-p", import_file_path_obj,
                  "-o", file_dir_path_obj, "--source","local", "--backend", "pipeline"],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲, 实时输出
        )

        for line in proc.stdout:
            print(line.rstrip())

        processed_code = proc.wait()

        print(processed_code)  # 0成功非0失败
        return processed_code

    def _get_md_paths(self, file_dir_path_obj, import_file_path_obj)->str:
        file_name=import_file_path_obj.stem
        return str(file_dir_path_obj/file_name/"auto"/f"{file_name}.md")

if __name__ == "__main__":
    import_file_path = "G:\\LLL\LLL\\2.资料\\pdf文档\\doc\\万用表RS-12的使用.pdf"
    file_dir = "G:\\LLL\\LLL\\2.资料\\pdf文档\\test_runimportgraph"
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir,
        "is_md_read_enabled": False,
        "is_pdf_read_enabled": False
    }
    setup_logging()
    nb = 1  # 类()是构造对象init,对象()是调用对象call
    pdf_to_md_node = PdfToMdNode()
    processed_state = pdf_to_md_node(state)
    print(json.dumps(processed_state, indent=4, ensure_ascii=False))

