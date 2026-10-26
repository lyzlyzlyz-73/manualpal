import base64
import json
import re
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from threading import Lock
from typing import Tuple, List, Dict, Deque
from collections import deque

from multipart import file_path
from win32gui import set_logger

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client import storage_clients
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    heading:str
    pre_text:str
    post_text:str


@dataclass
class ImageInfo:
    name: str
    path: str
    context: ImageContext


# -------1.读取md并且获得images路径-----------
class MdFileHandler:
    """
    验证state.mdpath是否存在，不存在报错stateFieldError
    将md_path_obj是否存在，不存在报错FileProcessError
    state，obj都存在了，with读取md文件，返回tupple
    """
    def __init__(self,logger:Logger,node_name:str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self,state:ImportGraphState)->Tuple[str,Path,Path]:
        md_path=state.get("md_path","")
        if not md_path:
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=str)
        md_path_obj=Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(node_name=self.node_name)

        with open(md_path_obj,'r',encoding='utf-8') as f:
            md_content=f.read()
        return md_content, md_path_obj, md_path_obj.parent/"images"

    def backup(self,md_content:str,md_path_obj:Path):
        self.logger.info(f"备份md文件")
        backup_path=md_path_obj.with_name(md_path_obj.stem+"_backup.md")
        with open(backup_path,'w',encoding='utf-8') as f:
            f.write(md_content)

# -------2.图片上下文截取----------1
class ImageScanner:
    def __init__(self,logger:Logger,node_name:str):
        self.logger = logger
        self.node_name = node_name


    def _find_context(self,md_content:str, img_name:str, max_chars:int=200) -> ImageContext | None:
        """
        根据md_content，img_name，max_chars，返回ImageContext
        :param md_content:
        :param img_name:
        :param max_chars:
        :return:
        """
        #识别md中的特殊格式的图片
        # - 定义正则表达式
        pattern= re.compile(
            r"!\[.*?\]\(.*?"+re.escape(img_name)+r".*?\)"
        )
        md_lines=md_content.split("\n")

        for line_idx,line in enumerate(md_lines):
            if not pattern.search(line):
                continue
            #匹配则向上查找最近标题和标题与图片之间的内容
            prev_title,prev_boundary=self._find_heading_above(
                md_lines,#找标题行数
                line_idx
            )
            pre_content=md_lines[prev_boundary+1:line_idx]
            img_pre=self._extract_limited_context(pre_content,direction="front",max_chars=max_chars)

            next_boundary=self._find_heading_blow(
                md_lines,
                line_idx)
            post_content=md_lines[line_idx+1:next_boundary]
            img_post=self._extract_limited_context(lines=post_content,max_chars=max_chars,direction="end")
            return ImageContext(pre_text=img_pre,heading=prev_title,post_text=img_post)
        return None


            #向下查找到下一个标题之间的内容作为下文


    def scan_img_dir(self,md_content:str,imagers_dir:Path,image_extensions:set[str],context_length:int=200)->List[ImageInfo]:
        """
        根据md_context，imagers_dir，image_extensions，context_length，返回image_info_list
        :param md_context:
        :param imagers_dir:
        :param image_extensions: {".jpg",".jpeg",".png",".gif",".bmp"}图片合规格式
        :param context_length:图片上下文长度
        :return:List[ImageInfo]
        """
        image_info_list:List[ImageInfo]=[]
        for image_path in imagers_dir.iterdir():# 遍历图片目录，不合规的图片就过
            if image_path.suffix.lower() not in image_extensions:
                continue
            if not image_path.is_file():#不是文件就过
                continue
            cxt=self._find_context(md_content=md_content,img_name=image_path.name,max_chars=context_length)
            if cxt is None:#对于返回None的值跳过
                continue
            #image_name=image_path.name
            image_info_list.append(ImageInfo(name=image_path.name,path=str(image_path),context=cxt))
        return image_info_list

    @staticmethod
    def _find_heading_above(md_lines:List[str], line_idx:int)->Tuple[str,int]:
        for i in range(line_idx,-1,-1):
            if re.match(r"^#{1,6}\s+",md_lines[i]):
                return md_lines[i].strip(),i
        return "",-1
    @staticmethod
    def _find_heading_blow(md_lines:List[str], from_idx:int)->int:
        for i in range(from_idx+1,len(md_lines),1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)

    def _extract_limited_context(self, lines:List[str], direction:str,max_chars:int=200)->str:
        current_paragraph:List[str]=[]
        paragraphs:List[str]=[]

        for line in lines:
            is_blank_line=not line.strip()
            is_other_image=re.match(r"^!\[.*?\]\(.*?\)$",line.strip())
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph=[]
                continue
            current_paragraph.append(line)


        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction=="front":
            paragraphs.reverse()

        total=0
        selected:List[str]=[]
        for para in paragraphs:
            if (total+len(para)>max_chars) and selected:#如果字数够了并且列表里存在段落，那么不需要便利段落了
                break
            selected.append(para)
            total+=len(para)

        if direction == "front":
            selected.reverse()

        return "\n\n".join(selected)


# -------3.图片生成概述----------
class VLMSummarizer:
    def __init__(self, logger:Logger, node_name:str):
        self.logger = logger
        self.node_name = node_name
        self._stamps_deque: Deque[float] = deque()
        self.lock = Lock()

    def summarizer_all(self, imageinfo_list:List[ImageInfo], document_name:str, vl_model:str, requests_per_minute:int)->Dict[str,str]:
        """
        :param imageinfo_list: 图片信息列表
        :param document_name: name,不带扩展名
        :param vl_model: 视觉模型
        :param requests_per_minute:调用模型的频率
        :return:
        """
        summaries:Dict[str,str]={}

        try:
            openai_client=AIClients.get_openai()
        except Exception as e:
            self.logger.error(f"获取openai客户端失败:{e}")
            for image_info in imageinfo_list:
                summaries[image_info.name]="默认图片摘要"
            return summaries

        for image_info in imageinfo_list:
            self._enforce_rate_limit(self._stamps_deque, requests_per_minute, 60)
            try:
                summaries[image_info.name]=self.summarizer_one(image_info,openai_client,vl_model)
            except Exception as e:
                # 单张失败不中断整体：降级兜底，其余图片继续处理
                self.logger.error(f"图片摘要失败:{image_info.name} - {e}")
                summaries[image_info.name]="图片"
            



        return summaries

    @staticmethod
    def summarizer_one(image_info: ImageInfo, openai_client, vl_model):
        with open(image_info.path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        image_info_context_property = [p for p in (image_info.context.heading, image_info.context.pre_text,
                                                   image_info.context.post_text) if p]
        finally_context = "\n".join(image_info_context_property)
        # 调用 VLM
        response = openai_client.chat.completions.create(
            model=vl_model,  # 用配置传入的模型（原硬编码会忽略.env里的VL_MODEL）
            # model="qwen3-vl-plus", #备选视觉模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",  # 告诉 API：这是一段文字
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题： \"{image_info.name}\"\n"
                                f"  2. 图片上下文： {finally_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            )
                        },
                        {
                            "type": "image_url",  # 告诉 API：这是一张图片
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        summary = response.choices[0].message.content
        return summary

    def _enforce_rate_limit(self, stamps_deque: Deque[float], requests_per_minute: int, window: int = 60):
        """
        Enforce rate limit for API calls.
        :param stamps_deque: 时间戳队列
        :param requests_per_minute: 每 window 时间次数
        :param window: 设置时间窗口的时间
        :return:
        """
        with self.lock:
            now = time.time()
            # 时间滑动窗口：移除窗口外（已过期）的时间戳
            while stamps_deque and now - stamps_deque[0] > window:
                stamps_deque.popleft()

            if len(stamps_deque) >= requests_per_minute:
                sleep_time = window - (now - stamps_deque[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # 睡醒后重新取当前时间，并再清理一次过期的记录
                now = time.time()
                while stamps_deque and now - stamps_deque[0] > window:
                    stamps_deque.popleft()

            stamps_deque.append(now)

#
class ImageUploader:
    def __init__(self, logger:Logger, node_name:str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, md_content:str, imageinfo_list:List[ImageInfo], summaries:Dict[str,str], minio_bucket:str,
                           minio_endpoint:str,document_name:str):
        """
        上传图片到minio，并替换md内容中的图片路径为minio url
        :param images_dir:
        :param md_content:
        :param imageinfo_list:
        :param summaries:
        :param minio_bucket:
        :param minio_endpoint:
        :return:
        """
        remote_urls:Dict[str,str]=self._upload_all(imageinfo_list,minio_bucket,minio_endpoint,document_name)
        new_md_content=self._replace_in_md(md_content,imageinfo_list,remote_urls,summaries)
        return new_md_content

    def _upload_all(self, imageinfo_list:List[ImageInfo], minio_bucket:str, minio_endpoint:str,document_name:str)->Dict[str,str]:
        """
        将本地图片上传到minio，返回endpoint+bucket+name，失败则返回本地路径
        :param imageinfo_list:
        :param minio_bucket: 桶
        :param minio_endpoint: 域名端口
        :return:
        """
        remote_urls:Dict[str,str]={}
        try:
            minio_client=StorageClients.get_minio_client()
        except Exception as e:
            self.logger.error(f"获取minio客户端失败:{e}")
            for image_info in imageinfo_list:
                remote_urls[image_info.name]=image_info.path
            return remote_urls

        for image_info in imageinfo_list:
            try:
                minio_client.fput_object(
                    bucket_name=minio_bucket,
                    object_name=f"{document_name}/{image_info.name}",
                    file_path=image_info.path
                )
                remote_url=f"{minio_endpoint}/{minio_bucket}/{document_name}/{image_info.name}"
                remote_urls[image_info.name]=remote_url
            except Exception as e:
                # 上传失败降级为本地路径，不影响整体流程
                self.logger.error(f"图片上传MinIO失败:{image_info.name} - {e}")
                remote_urls[image_info.name]=image_info.path
        return remote_urls



    def _replace_in_md(self, md_content:str, imageinfo_list:List[ImageInfo], remote_urls:Dict[str,str], summaries:Dict[str,str]):
        """替换图片的摘要和路径，返回新的md内容"""
        pattern=r'!\[(.*?)\]\((.*?)\)'
        def replacer(match):
            original_url=match.group(2).strip()
            '''original_summary=match.group(1).strip()'''
            image_url_md=Path(original_url).name
            for image_name, summary in summaries.items():
                if image_name==image_url_md:
                    return f"![{summary}]({remote_urls[image_name]})"
            return match.group(0)
        return re.sub(pattern,replacer, md_content)


class MdImgNode(BaseNode):
    name="md_img_node"
    """
    识别图片，保留图片标题，上下文->
    根据得到的类交给vlm生成概述
    将图片url替换为miniourl
    
    """
    def __init__(self,):
        super().__init__()
        self.md_file_handler=MdFileHandler(logger=self.logger,node_name=self.name)
        self.image_scanner=ImageScanner(logger=self.logger,node_name=self.name)
        self.vlm_summarizer=VLMSummarizer(logger=self.logger,node_name=self.name)
        self.image_uploader=ImageUploader(logger=self.logger,node_name=self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        md_content,md_path_obj,images_dir=self.md_file_handler.read_md(state)

        if not images_dir.exists():
            state['md_content']=md_content
            return state
        #识别图片，得到名称，路径，上文，下文，上标题
        imageinfo_list:List[ImageInfo]=self.image_scanner.scan_img_dir(md_content,images_dir,
                                         image_extensions=self.config.image_extensions,#{.jpg,.jpeg,.png,gif,bmp},
                                         context_length=self.config.img_content_length
                                         )
        #根据得到的类交给vlm生成概述
        summaries=self.vlm_summarizer.summarizer_all(imageinfo_list=imageinfo_list,document_name=md_path_obj.stem,
                                                 vl_model=self.config.vl_model,
                                                 requests_per_minute=self.config.requests_per_minute)
        #将md中的图片摘要替换，图片路径替换为miniourl，返回md
        new_md_content = self.image_uploader.upload_and_replace(
            md_content=md_content,
            imageinfo_list=imageinfo_list,
            summaries=summaries,
            minio_bucket=self.config.minio_bucket,      # minio存储桶
            minio_endpoint=self.config.get_minio_base_url(),  # minio地址
            document_name=md_path_obj.stem
        )
        # 替换结果写回state（原来漏了这步，new_md_content 不会被保存）
        state["md_content"]=new_md_content
        #将新的md备份
        self.md_file_handler.backup(new_md_content,md_path_obj)
        return state

if __name__ == '__main__':
    node=MdImgNode()
    setup_logging()
    state={
        "md_path":r"G:\LLL\LLL\2.资料\pdf文档\test_runimportgraph\万用表RS-12的使用\auto\万用表RS-12的使用.md"
    }
    node(state)
    print(json.dumps(state, indent=4, ensure_ascii=False))



