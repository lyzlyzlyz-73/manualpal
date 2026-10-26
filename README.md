# ManualPal（手册通）

> 设备手册知识库系统：把 PDF/MD 设备手册一键入库，再用「多路召回 + 重排」的 RAG 流程对手册内容进行智能问答。

上传一份设备手册，系统会自动完成 **PDF 解析 → 图片提取 → 文档切分 → 向量化 → 入库**；提问时走 **名称确认 → 多路并行检索（向量 / HyDE / 联网）→ RRF 融合 → 重排 → 生成答案** 的完整链路，支持 SSE 逐字流式输出。

---

## 功能特性

- **一体化控制台**：文件导入与知识库问答共用一个页面、一个端口、一个进程
- **双 LangGraph 工作流**：导入图与查询图各自独立编排，节点职责单一、可插拔
- **PDF 深度解析**：基于 [MinerU](https://github.com/opendatalab/MinerU) 的 `pipeline` 后端，本地解析，公式/表格/图片一网打尽
- **多路并行召回**：Milvus 向量检索、HyDE 假设文档检索、MCP 联网搜索三路并发
- **RRF 融合 + 重排**：多路结果用倒数排序融合，再经重排模型精选
- **任务进度可查**：导入/问答均为后台任务，前端轮询 `/status/{task_id}` 实时展示节点进度
- **流式问答**：SSE 推送节点进度与逐字答案
- **会话历史**：MongoDB 持久化，支持查看与一键清空

## 架构与工作流

```
┌─────────────────────── 导入流程（LangGraph）───────────────────────┐
│                                                                    │
│  entry_node ──router──> pdf_to_md_node(MinerU) ──> md_img_node     │
│        │                                       (图片上传 MinIO)     │
│        └─(md 直读)──────────────────────────────> document_split   │
│                                                  node(切分)        │
│                                                        │           │
│                                          item_name_recognition     │
│                                          (LLM 识别商品名称)         │
│                                                        │           │
│                                          bge_embedding(向量化)     │
│                                                        │           │
│                                          import_milvus(入库)       │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────── 查询流程（LangGraph）───────────────────────┐
│                                                                    │
│  item_name_confirm ──(已有答案)─────────────────────> answer_output │
│        │(无答案)                                                    │
│   multi_search ──┬── search_embedding      (Milvus 向量检索)       │
│    (并行分发)    ├── search_embedding_hyde (HyDE 改写检索)          │
│                  └── web_search_mcp        (MCP 联网搜索)           │
│        │                                                           │
│      join ──> rrf(融合) ──> rerank(重排) ──> answer_output(生成)    │
└────────────────────────────────────────────────────────────────────┘
```

## 技术栈

| 层 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn（SSE 流式） |
| 流程编排 | LangGraph |
| 文档解析 | MinerU `pipeline` 后端（GPU 加速） |
| 向量数据库 | Milvus（pymilvus） |
| 对象存储 | MinIO（手册原文 / 图片） |
| 会话与任务状态 | MongoDB |
| 向量模型 | BGE（sentence-transformers） |
| 大模型接入 | OpenAI 兼容接口（langchain-openai / openai-agents） |

## 目录结构

```
manualpal/
├── integrated_app/            # 集成版 Web 应用（推荐入口）
│   ├── app.py                 #   FastAPI：上传/问答/流式/状态/历史
│   └── static/index.html      #   一体化控制台页面
├── knowledge/                 # 知识库核心工程
│   ├── api/                   #   旧版独立路由（import/query_router）
│   ├── core/                  #   路径与依赖注入
│   ├── processor/
│   │   ├── import_process/    #   导入工作流（节点 + 状态 + 主图）
│   │   └── query_process/     #   查询工作流（节点 + 状态 + 主图）
│   ├── prompt/                #   提示词模板
│   ├── schema/                #   请求/响应/任务数据模型
│   ├── service/               #   文件导入服务 / 问答服务
│   ├── utils/                 #   embedding/milvus/mongo/sse/task 工具
│   ├── front/                 #   旧版分离页面（chat.html / import.html）
│   ├── temp_data/             #   运行时上传目录（不入库）
│   ├── test/                  #   分步测试脚本
│   └── requirements.txt
└── environment_rebuild        # 环境重建笔记
```

## 快速开始

### 环境要求

- Windows / Linux，Python **3.11**
- NVIDIA GPU（MinerU 解析与 BGE 向量化建议使用；纯 CPU 也能跑，速度较慢）
- 需提前准备并启动的外部服务：

| 服务 | 用途 |
|---|---|
| MinIO | 存储手册原文与图片 |
| Milvus | 向量检索 |
| MongoDB | 会话历史与任务状态 |
| OpenAI 兼容大模型服务 | 名称识别 / 答案生成 / 重排等（支持本地 lmdeploy 或云端 API） |

### 安装

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux/macOS

# 2. 安装项目依赖
pip install -r knowledge/requirements.txt

# 3. ★收尾★ 用官方 CUDA 源安装 GPU 版 PyTorch
#    （必须放在最后一步，避免 mineru 依赖解析时覆盖为 CPU 版）
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 配置

模型服务地址、MinIO/Milvus/Mongo 连接信息等通过环境变量（`.env`）或客户端配置注入。
`.env` 已被 `.gitignore` 排除，**请勿把密钥提交到仓库**；可参考 `knowledge/utils/client/` 下的客户端封装了解所需配置项。

### 启动

```bash
python -m integrated_app.app
```

浏览器访问 **http://127.0.0.1:8010** 即可使用一体化控制台（上传 + 问答）。
绑定了 `0.0.0.0`，同一局域网内其他设备可通过本机 IP 访问。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 一体化控制台页面 |
| POST | `/upload` | 上传手册文件，后台执行导入图 |
| POST | `/query` | 提交问题（`is_stream=true` 返回 task_id 走 SSE） |
| GET | `/stream/{task_id}` | SSE：节点进度 / 逐字答案 / 最终结果 |
| GET | `/status/{task_id}` | 导入与问答共用的任务状态查询 |
| GET | `/history/{session_id}` | 会话历史（正序） |
| DELETE | `/history/{session_id}` | 清空会话历史 |

## 常见问题（FAQ）

**Q: 项目目录改名/移动后，虚拟环境报 `uv trampoline failed to canonicalize script path`？**

venv 里的 `*.exe` 启动器内嵌了创建时的绝对路径，目录一改名即失效。三种处理方式：

1. 临时方案：所有命令改用模块方式调用，如 `python -m pip`、`python -m uvicorn`
2. 修复单个启动器：`python -m pip install --force-reinstall --no-deps <包名>==<原版本>`
3. 终极方案：删除 `.venv` 按上文「安装」步骤重建

**Q: MinerU 转换失败？**

先确认 `mineru --version` 可用（同上 FAQ），再确认 `--source local` 对应的模型文件已通过 `mineru-models-download` 下载完毕。

**Q: 页面能打开但上传/问答报错？**

Web 服务本身不依赖外部组件，页面正常 ≠ 功能可用。请逐一确认 MinIO、Milvus、MongoDB 与大模型服务是否在线。

## 说明

- 本项目为**学习用途**的 RAG 实践项目：导入/查询两条 LangGraph 流程均为手写节点，便于逐个替换与实验（`knowledge/test/` 下有分步测试脚本）。
- `knowledge/api/` + `knowledge/front/` 是拆分为两个服务的旧版实现，保留用于对照学习；日常使用请以 `integrated_app` 为准。
