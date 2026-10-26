import os
import subprocess
from pathlib import Path

# === 关键: 加载环境变量（模拟 PyCharm 运行 pdf_to_md.py 时的环境） ===
# 方法1: 直接读取 .env 文件并设置环境变量
env_file = Path(__file__).parent.parent.parent / '.env'
if env_file.exists():
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# 方法2: 显式设置关键环境变量（确保 MinerU 使用 ModelScope 离线模式）
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("✅ 环境变量已配置")
print(f"  MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
print(f"  MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
print(f"  HF_HOME = {os.environ.get('HF_HOME')}")
print()
PDF_PATH = r"G:\LLL\LLL\2.资料\pdf文档\doc\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"
OUT_DIR = r"G:\LLL\LLL\2.资料\pdf文档\test_runimportgraph"
proc = subprocess.Popen(
    args=["mineru", "-p", PDF_PATH,
          "-o", OUT_DIR,"--resource=local" ,"--backend", "pipeline"],
    stdout=subprocess.PIPE,     # 捕获标准输出
    stderr=subprocess.STDOUT,   # 合并错误到标准输出
    text=True,
    encoding="utf-8",
    errors="replace",           # 遇到乱码时替换
    bufsize=1                   # 行缓冲, 实时输出
)

for line in proc.stdout:
    print(line.rstrip())

return_code = proc.wait()
print(return_code)#0成功非0失败
