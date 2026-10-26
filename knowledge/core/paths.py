import os

# 必须用 __file__（当前文件路径）；__name__ 是模块名字符串，dirname 得到 ""，路径会随 cwd 乱跑
KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT,"temp_data")

FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")

def get_local_base_dir() -> str:
    return LOCAL_BASE_DIR

def get_front_page_dir():
    return FRONT_PAGE_DIR