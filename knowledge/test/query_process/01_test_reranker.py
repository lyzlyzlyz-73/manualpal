"""测试 BGE-reranker-large（交叉编码器）的相关性打分

与 rerank 节点里 AIClients.get_bge_m3_rerank_client() 用的是同一个模型、同一套调用方式。
"""

from FlagEmbedding import FlagReranker

# 1. 加载模型（本地模型路径，与 .env 里的 BGE_RERANKER_LARGE 一致）
reranker = FlagReranker(
    # model_name_or_path="BAAI/bge-reranker-large",
    model_name_or_path=r"G:\LLL\LLL\5.本地模型\ai_models\modelscope_cache\models\BAAI\bge-reranker-large",
    device="cuda:0",  # GPU 加速
    # device="cpu",
    use_fp16=True,  # 半精度推理
    # use_fp16=False,
)

# 2. 构造 (query, doc) 对
#    交叉编码器必须"问题和文档拼成一条"送进去，所以入参是 pairs，而不是两条独立的文本
pairs = [
    ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
    ["什么是万用表？", "今天天气很好"],
]

# 3. 计算相关性得分
#    3.1 原始输出是 logits：可正可负、没有上下界，只能同一批内比较大小
raw_scores = reranker.compute_score(pairs)

#    3.2 归一化输出走 sigmoid → 落在 0~1，越大越相关
#        项目 rerank 节点用的就是这一种：compute_score(pairs, 归一化normalize=True)
norm_scores = reranker.compute_score(pairs, normalize=True)

# 4. 打印结果
print("=" * 60)
print(f"原始 logits: {raw_scores}")
print(f"归一化分数 : {[round(float(s), 4) for s in norm_scores]}")
print("-" * 60)
for (query, doc), score in zip(pairs, norm_scores):
    print(f"相关度 {float(score):.4f} | 问题: {query} | 文档: {doc}")
print("=" * 60)
print("预期: 第一条远高于第二条（高分 = 高相关）")
print("说明: logits 可正可负、无上下界；normalize=True 走 sigmoid 落到 0~1，"
      "项目 rerank 节点（含 rerank_gap_abs=0.15 断崖截断）依赖的就是归一化后的分数")
