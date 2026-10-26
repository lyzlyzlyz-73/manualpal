import json

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from pathlib import Path
class EntryNode(BaseNode):
    name="entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        file_dir=state.get('file_dir','')
        import_file_path=state.get('import_file_path','')
        path = Path(import_file_path)

        self.log_step("1", "检验输入路径")
        nb = 1  # 若state['import_file_path']不存在，会抛出异常，程序直接崩溃,get更保险
        if not import_file_path:
            raise ValueError(f"import_file_path为空{self.name}")

        self.log_step("2", "检验文件是否存在")
        if not path.exists():
            raise ValueError(f"文件不存在{self.name}")

        self.log_step("3", "检验文件类型")
        sufix=Path(import_file_path).suffix.lower()
        state["file_title"]=Path(import_file_path).stem
        if sufix==".pdf":
            state['is_pdf_read_enabled']=True
            state['pdf_path']=import_file_path
        elif sufix==".md":
            state['is_md_read_enabled'] = True
            state['md_path']=import_file_path
        else:
            raise ValidationError(f"不支持文件类型{import_file_path},{sufix},{self.name}")

        return state



if __name__ == "__main__":
    import_file_path="G:\\LLL\LLL\\2.资料\\pdf文档\\doc\\万用表RS-12的使用.pdf"
    file_dir="G:\\LLL\\LLL\\2.资料\\pdf文档\\test_runimportgraph"
    state={
        "import_file_path":import_file_path,
        "file_dir":file_dir,
        "is_md_read_enabled":False,
        "is_pdf_read_enabled":False
    }
    setup_logging()
    nb=1#类()是构造对象init,对象()是调用对象call
    entry_node = EntryNode()
    processed_state=entry_node(state)
    print(json.dumps(processed_state, indent=4, ensure_ascii=False))
    



