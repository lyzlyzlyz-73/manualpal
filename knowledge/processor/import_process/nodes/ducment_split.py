import json
import os
import re
from typing import List, TypedDict, Any, final

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"
    """
    1,获取输入参数并且校验:md_content,file_title,max_content_length,min_content_length
    """
    """
    2,
    """

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 验证输入的参数，得到config内的边界
        md_content, file_title, max_content_length, min_content_length = self._validate_get_input(state)
        # 根据标题进行切分，切分得到一个一个的secton
        sections: List[dict[str, Any]] = self._split_by_title(md_content, file_title)
        # 长度且，大切小，小合并
        final_sections=self._split_long_merge_short(sections, max_content_length, min_content_length)
        #chunks
        final_chunks=self._assable_chunks(final_sections)
        #日志记录
        self._log_summary(md_content,final_chunks,max_content_length)
        #备份切分结果
        self._backup_chunks(state, final_chunks)
        state["chunks"]=final_chunks
        return state

    def _validate_get_input(self, state):
        self.log_step("step1", "校验输入参数")
        file_title = state.get("file_title")
        md_content = state.get("md_content")
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if not md_content:
            raise ValidationError(f"md_文档不存在", self.name)
        if not file_title:
            raise ValidationError(f"文件标题不存在", self.name)
        if min_content_length < 0:
            raise ValidationError(f"min_content_length不能小于0", self.name)
        if max_content_length < 0:
            raise ValidationError(f"max_content_length不能小于0", self.name)
        if max_content_length < min_content_length:
            raise ValidationError(f"max_content_length不能小于min_content_length", self.name)
        return md_content, file_title, max_content_length, min_content_length

    def _split_by_title(self, md_content: str, file_title: str) -> List[dict[str, Any]]:
        self.log_step("step2", "按标题切分 section")
        """
        标题切分:\n获取文档的行集合，便利没事行，是否在代码围栏中，如果不在则正则表达判断是否为标题，构建sections
        sections[{
            "title":"",
            "parent_title":"",
            "file_name":"",
            "body":""
        }
        """
        # 变量区
        sections: List[dict[str, Any]] = []
        body_lines = []  # 沙箱，存储当前section的body
        title = ""
        is_node = False
        hierarchy = [""] * 7
        current_level = 0
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)$")

        # 获取所有行集合
        lines = md_content.split("\n")

        def _flush():
            """
            进行封装为section操作
            :return:
            """
            body = "\n".join(body_lines)
            if body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title = file_title
                sections.append({
                    "title": title if title else file_title,
                    "parent_title": parent_title,
                    "file_title": file_title,
                    "body": body
                })

        # 便利所有行，不是标题就装箱，是标题就创建section
        for line in lines:
            #非代码区域中遇到标题就刷新
            if line.startswith("```") or line.startswith("~~~"):
                is_node = not is_node
            match = heading_re.match(line)
            if match and not is_node:
                _flush()
                title = line
                current_level = len(match.group(1))
                hierarchy[current_level] = title
                for i in range(current_level + 1, 7, 1):
                    hierarchy[i] = ""
                body_lines = []
            else:
                body_lines.append(line)

        _flush()

        return sections

    def _split_long_merge_short(self, sections: List[dict[str, Any]], max_content_length: int = 1000,
                                min_content_length: int = 100) -> List[dict[str, Any]]:
        self.log_step("step3", f"长块切分+短块合并（输入 {len(sections)} 个 section）")
        current_sections=[]
        for section in sections:
            sub_section=self._split_long(section,max_content_length)
            current_sections.extend(sub_section)#{},{}   [{},{}]解包并
        current_sections=self._merge_short_section(current_sections,max_content_length,min_content_length)
        return current_sections

    def _split_long(self, section, max_content_length)->list[dict[str, Any]]:

        sub_section:List[dict[str, Any]] = []
        # 1. 获取section对象属性
        title=section.get("title")
        parent_title=section.get("parent_title")
        file_title=section.get("file_title")
        body=section.get("body")
        # 2. 判断表格
        if "<table>" in body:
            self.logger.info(f"表格降维处理")
            body=MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 3. 对标题校验长度，超过50截断
        if len(title)>50:
            title=title[:50]

        # 4. 拼接标题前缀
        title_prefix=f"{title}\n\n"

        # 5. 计算标题前缀长度 + 内容长度
        content_length=len(title_prefix)+len(body)

        # 6. 判断是否需要切分
        if content_length <= max_content_length:
            return [section]

        # 7. 计算body可用长度，判断长度是否小于等于0
        body_available_length = max_content_length - len(title_prefix)
        if body_available_length <= 0:
            self.logger.error(f"{self.name},请检查max_content_length配置，确保其大于标题长度")
            return [section]

        # 8. 需要继续切分
        splitter_false=RecursiveCharacterTextSplitter(
            chunk_size=body_available_length,#用"正文可用长度"，不能写len(body)，否则等于不切分
            chunk_overlap=0,
            keep_separator=False,
            separators=["\n\n","\n","。","!","？","；",".","!","?",";"," ",""]
        )
        texts:List[str]=splitter_false.split_text(body)
        self.log_step("step3", f"长块切分 -> {len(texts)} 片")
        #    判断切分数量结果，要么返回唯一原元素列表，要么返回多个新元素类别
        for index,text in enumerate(texts):
            sub_section.append({
                "title":title,
                "parent_title":parent_title,
                "file_title":file_title,
                "body":f"{title}\n\n{text}",#
                "part":index+1
            })
        # 9. 返回

        return sub_section

    def _merge_short_section(self, current_sections:List[dict[str,Any]], max_content_length:int, min_content_length:int=100)->List[dict[str,Any]]:
        final_sections=[]
        current_section=current_sections[0]

        for next_section  in current_sections[1:]:
            is_same_parent=current_section["parent_title"]==next_section["parent_title"]
            merged_length=len(current_section["body"])+len(next_section["body"])+2
            #三个条件都满足才合并：下一段够小 + 同父标题 + 合并后不超上限
            if len(next_section["body"]) <= min_content_length and is_same_parent and merged_length <= max_content_length:
                current_section["title"]=next_section["parent_title"]
                current_section["body"]=current_section["body"].rstrip()+"\n\n"+next_section["body"].lstrip()
            else:
                final_sections.append(current_section)
                current_section=next_section
        final_sections.append(current_section)
        self.log_step("step3", f"短块合并：{len(current_sections)} -> {len(final_sections)} 个")
        return final_sections

    def _assable_chunks(self, final_sections:List[dict[str,Any]]):
        self.log_step("step4", "组装最终 chunks")
        final_chunks=[]
        for section in final_sections:
            final_chunks.append(
                {
                    "title":section["title"],
                    "parent_title":section["parent_title"],
                    "file_title":section["file_title"],
                    "body":section["title"]+"\n\n"+section["body"]
                }
            )
        return final_chunks

    def _log_summary(self, md_content:str, final_chunks:List[dict[str,Any]], max_content_length:int):
        self.log_step("step5","输出统计")

        lines_count=md_content.count("\n")+1
        self.logger.info(f"总行数:{lines_count}")
        self.logger.info(f"最大长度:{max_content_length}")
        self.logger.info(f"最终章节数:{len(final_chunks)}")

        if final_chunks:
            for i,sec in enumerate(final_chunks[:5]):
                title=sec.get("title","")[:30]
                self.logger.info(f"章节:{i+1} {title}")
            if len(final_chunks)>5:
                self.logger.info(f"... {len(final_chunks)-5}个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step(step_name="step6", message="备份切片")

        # 备份到 md 所在目录（与 _backup.md 同级）；没有 md_path 时回退到 file_dir
        md_path = state.get("md_path", "")
        local_dir = os.path.dirname(md_path) if md_path else state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 md_path/file_dir, 跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)  # json.dump() 具有写操作能力   注意:  不是json.dumps()
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

from pathlib import Path

if __name__ == '__main__':
    setup_logging()
    node = DocumentSplitNode()

    file_dir = r"G:\LLL\LLL\2.资料\pdf文档\test_runimportgraph\万用表RS-12的使用\auto"
    file_path = Path(file_dir) / "万用表RS-12的使用_backup.md"      # ← 只写一次长路径

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        "file_dir": file_dir,
    }

    node.process(state)
