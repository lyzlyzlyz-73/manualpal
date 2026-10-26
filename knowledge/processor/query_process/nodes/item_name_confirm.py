import json
import re
import time
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages, update_message_item_names

# 商品名规范化时忽略的字符（空格、连字符、括号、常见标点）
_NAME_NOISE_PATTERN = re.compile(r"[\s\-_/·.()（）【】\[\]、，,。:：;；]+")
# 用户原话里的"型号"片段：字母+数字（hak180 / RS-12 / LA2608），或数字+字母（211H）
_MODEL_TOKEN_PATTERN = re.compile(r"[A-Za-z]{1,}[\s-]?\d{1,}|\d{1,}[A-Za-z]{1,}")


def normalize_item_name(name: str) -> str:
    """商品名规范化：小写 + 去掉空格/标点，用于 exact / alias 比对"""
    return _NAME_NOISE_PATTERN.sub("", str(name or "")).lower()


def extract_model_tokens(text: str) -> List[str]:
    """抽取文本中的型号片段（去重保序）"""
    tokens = [match.group(0).strip() for match in _MODEL_TOKEN_PATTERN.finditer(str(text or ""))]
    return [token for token in dict.fromkeys(tokens) if token]


class ItemNameAligner:
    """
    商品名对齐器
    主要职责：
    1. 查询向量数据库
    2. 评分对齐
    3. 分数差异过滤
    """
    def __init__(self, logger, name, config):
        self.logger = logger
        self.name = name
        self.config = config
        # 标准商品名缓存（exact/alias 匹配用），避免每次请求都全量查 Milvus
        self._known_item_names: List[str] = []
        self._known_loaded_at: float = 0.0

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """执行匹配、对齐、过滤三步流程

        Returns:
            (confirmed, options, basis)
            basis 记录每条结果的匹配依据，例如
                [{"extracted": "hak180", "matched": "HAK 180 烫金机",
                  "match_type": "alias", "score": 0.99}]
        """
        # 1. 规范化 exact/alias 匹配 + 向量检索
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)

        # 2. 按"匹配类型 + 分数"对齐
        confirmed, options, basis = self._item_name_score_align(search_result)

        # 3. 分数差异过滤（仅当 confirmed 有多个时）
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)
            basis = [item for item in basis if item.get("matched") in confirmed]

        return confirmed, options, basis

    def _load_known_item_names(self) -> List[str]:
        """加载知识库里的标准商品名（5 分钟缓存）

        用于 exact / alias 匹配：不依赖向量分数，避免 "hak180 → HAK 180" 这类
        空格/大小写/写法差异被向量模型判成低分。
        """
        now = time.time()
        if self._known_item_names and (now - self._known_loaded_at) < 300:
            return self._known_item_names

        try:
            milvus_client = StorageClients.get_milvus_client()
            rows = milvus_client.query(
                collection_name=self.config.item_name_collection or "kb_item_names_v1",
                filter='item_name != ""',
                output_fields=["item_name"],
                limit=2000,
            )
            self._known_item_names = sorted({
                str(row.get("item_name", "")).strip()
                for row in rows
                if row.get("item_name")
            })
            self._known_loaded_at = now
        except Exception as e:
            self.logger.error(f"{self.name}, 加载标准商品名失败，exact/alias 匹配降级: {e}")

        return self._known_item_names

    def _exact_alias_matches(self, extracted_name: str) -> List[Dict[str, Any]]:
        """规范化后的 exact / alias 匹配

        - exact: 规范化后与标准名完全相等（"RS PRO RS-12 数字万用表"）
        - alias: 一方包含另一方，且较短一方含数字、长度 >= 4
                 （"hak180" ⊆ "HAK 180 烫金机"、"RS-12数字万用表" ⊆ "RS PRO RS-12 数字万用表"）
          限定"含数字 + 长度"是为了避免 "万用表" 这类泛词误命中。
        """
        normalized = normalize_item_name(extracted_name)
        if not normalized:
            return []

        matches: List[Dict[str, Any]] = []
        for known_name in self._load_known_item_names():
            known_normalized = normalize_item_name(known_name)
            if not known_normalized:
                continue

            if normalized == known_normalized:
                matches.append({"item_name": known_name, "score": 1.0, "match_type": "exact"})
                continue

            shorter, longer = (
                (normalized, known_normalized)
                if len(normalized) <= len(known_normalized)
                else (known_normalized, normalized)
            )
            if len(shorter) >= 4 and any(ch.isdigit() for ch in shorter) and shorter in longer:
                matches.append({"item_name": known_name, "score": 0.99, "match_type": "alias"})

        return matches

    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """商品名匹配：先做规范化 exact/alias，再用向量检索补充候选"""
        search_results: List[Dict[str, Any]] = []

        if not item_names:                       # 空列表提前返回（generate_bge_m3_hybrid_vectors 遇空列表会抛 ValueError）
            return search_results

        # 1. 先做 exact/alias 匹配（本地比对，不依赖模型和向量库）
        for extract_item_name in item_names:
            search_results.append({
                "extracted_name": extract_item_name,
                "matches": self._exact_alias_matches(extract_item_name),
            })

        # 2. 向量检索补充候选（模型/库不可用时，只保留第 1 步的结果）
        try:
            milvus_client = StorageClients.get_milvus_client()
            embedding_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"{self.name}, 依赖服务不可用，仅使用 exact/alias 匹配: {e}")
            return search_results

        try:
            hybrid_embedding_result = generate_bge_m3_hybrid_vectors(embedding_model, item_names)
        except (ValueError, RuntimeError) as e:
            self.logger.error(f"{self.name}, 商品名向量化失败，仅使用 exact/alias 匹配: {e}")
            return search_results

        for index, extract_item_name in enumerate(item_names):
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result['dense'][index],
                sparse_vector=hybrid_embedding_result['sparse'][index],
            )

            try:
                hybrid_search_result = execute_hybrid_search_query(
                    milvus_client,
                    collection_name=self.config.item_name_collection or "kb_item_names_v1",
                    search_requests=hybrid_search_requests,
                    ranker_weights=(self.config.item_name_dense_weight, self.config.item_name_sparse_weight),
                    norm_score=True,
                    output_fields=["item_name"]
                )
            except Exception as e:
                # 单个商品名检索失败不影响其它商品名；第 1 步的 exact/alias 结果保留
                self.logger.error(f"{self.name}, 商品名[{extract_item_name}]检索失败: {e}")
                continue

            vector_matches = [
                {"item_name": hit["entity"]["item_name"], "score": hit["distance"], "match_type": "vector"}
                for hit in (hybrid_search_result[0] if hybrid_search_result else [])
            ]

            # 3. 合并：exact/alias 优先，同名的向量结果不再重复
            decisive = search_results[index]["matches"]
            decisive_names = {match["item_name"] for match in decisive}
            search_results[index]["matches"] = decisive + [
                match for match in vector_matches if match["item_name"] not in decisive_names
            ]

        return search_results

    def _item_name_score_align(
            self, search_results: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """按"匹配类型 + 分数"决定自动确认还是给候选

        规则（步骤5）：
        1. exact / alias 命中（规范化比对）：唯一 → 自动确认；多个 → 给候选让用户选
        2. 纯向量匹配：即使分数 >= high_confidence 也只给候选，不再自动确认
           （向量分数只能说明"像"，不足以替用户认定是哪个商品）
        """
        confirmed: List[str] = []
        options: List[str] = []
        basis: List[Dict[str, Any]] = []
        max_options = self.config.item_name_max_options
        mid_confidence = self.config.item_name_mid_confidence

        for item_name_search_result in search_results:
            extracted_name = item_name_search_result.get("extracted_name")
            matches = sorted(
                item_name_search_result.get("matches") or [],
                key=lambda match: match.get("score", 0),
                reverse=True,
            )

            # 1. exact / alias：可以自动确认
            decisive = [match for match in matches if match.get("match_type") in ("exact", "alias")]
            if decisive:
                names: List[str] = []
                for match in decisive:
                    if match["item_name"] not in names:
                        names.append(match["item_name"])

                if len(names) == 1:
                    if names[0] not in confirmed:
                        confirmed.append(names[0])
                    basis.append({
                        "extracted": extracted_name,
                        "matched": names[0],
                        "match_type": decisive[0].get("match_type"),
                        "score": decisive[0].get("score", 1.0),
                    })
                else:
                    # 一个提取名命中多个标准名 → 有歧义，交给用户确认
                    for name in names[:max_options]:
                        if name not in options and name not in confirmed:
                            options.append(name)
                    basis.append({
                        "extracted": extracted_name,
                        "matched": names,
                        "match_type": "ambiguous",
                        "score": decisive[0].get("score", 1.0),
                    })
                continue

            # 2. 同品类 / 部分型号：分数可能不高，但名字存在包含关系 → 作为候选提示
            #    （"万用表" ⊆ "RS PRO RS-12 数字万用表"、"hak18" ⊆ "HAK 180 烫金机"）
            normalized_extracted = normalize_item_name(extracted_name)
            containment_candidates: List[str] = []
            if len(normalized_extracted) >= 2:
                for match in matches:
                    name = match.get("item_name")
                    if not name or match.get("score", 0) < self.config.item_name_low_confidence:
                        continue
                    if normalized_extracted in normalize_item_name(name):
                        containment_candidates.append(name)

            if containment_candidates:
                for name in containment_candidates[:max_options]:
                    if name not in options and name not in confirmed:
                        options.append(name)
                basis.append({
                    "extracted": extracted_name,
                    "matched": containment_candidates[:max_options],
                    "match_type": "vector",
                    "score": max(
                        (match.get("score", 0.0) for match in matches
                         if match.get("item_name") in containment_candidates),
                        default=0.0,
                    ),
                })
                continue

            # 3. 纯向量匹配：只给候选
            added_candidates: List[str] = []
            for match in matches:
                name = match.get("item_name")
                if not name or match.get("score", 0) < mid_confidence:
                    continue
                if name not in options and name not in confirmed:
                    options.append(name)
                    added_candidates.append(name)
                if len(options) >= max_options:
                    break

            if added_candidates:
                basis.append({
                    "extracted": extracted_name,
                    "matched": added_candidates,
                    "match_type": "vector",
                    "score": max(
                        (match.get("score", 0.0) for match in matches
                         if match.get("item_name") in added_candidates),
                        default=0.0,
                    ),
                })

        return confirmed, options[:max_options], basis

    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]):
        """分数差异过滤，剔除误判"""
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get('matches')
            for m in matches:
                score = m.get('score')
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)
        max_item_name_score = sorted_item_name_score[0][1]

        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]



class ItemNameExtractor:
    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name
    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """LLM 根据用户原始问题提取商品名"""
        result: Dict[str, Any] = {"item_names": [], "rewritten_query": original_query}

        llm_client = AIClients.get_llm_openai(response_format=False)  # 用公开方法(走单例缓存)，别用 _create_xxx 私有方法
        if llm_client is None:
            return result

        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text if history_text else "暂无上下文",
            query=original_query
        )
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            llm_content = llm_response.content.strip()
        except Exception as e:
            # LLM/网络不可用时降级：result 已含默认值（item_names=[]、rewritten_query=original_query）
            self.logger.error(f"{self.node_name}, LLM调用失败，降级使用原始问题: {e}")
            return result

        if not llm_content.strip():
            return result

        try:
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names")
        except Exception as e:
            self.logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        # 防"历史污染"：用户原话里写了型号，但 LLM 给出的商品名与原话没有任何词面重叠
        # （典型：带着 LA2608 的历史问 hak180，模型把问题改写成了 LA2608）→ 以用户原话为准
        return self._guard_history_contamination(result, original_query)

    @staticmethod
    def _has_lexical_overlap(item_names: List[str], original_query: str) -> bool:
        """商品名（或其型号片段）是否在用户原话里出现过"""
        query_normalized = normalize_item_name(original_query)
        for name in item_names or []:
            name_normalized = normalize_item_name(name)
            if name_normalized and name_normalized in query_normalized:
                return True
            for token in extract_model_tokens(name):
                token_normalized = normalize_item_name(token)
                if token_normalized and token_normalized in query_normalized:
                    return True
        return False

    def _guard_history_contamination(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """阻止 LLM 用历史里的商品替换用户新提到的型号"""
        item_names = result.get("item_names") or []
        model_tokens = extract_model_tokens(original_query)
        if not model_tokens or not item_names:
            return result

        if self._has_lexical_overlap(item_names, original_query):
            return result

        corrected = [token for token in dict.fromkeys(model_tokens) if token]
        self.logger.warning(
            f"{self.node_name}, 疑似历史污染：原话含型号 {corrected}，LLM 却给出 {item_names}，已按原话纠正"
        )
        result["item_names"] = corrected
        result["rewritten_query"] = original_query
        return result

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            raw_item_names = parsed_llm_result.get('item_names')
            if not isinstance(raw_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in raw_item_names if raw_item.strip()]

            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")



class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点

    流程: 获取历史 → LLM提取商品名 → 向量匹配 → 评分对齐 → 更新状态 → 历史回填
    """
    name = "item_name_confirm_node"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor(self.logger, self.name)
        self._item_name_aligner = ItemNameAligner(self.logger, self.name, self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取用户的原始问题
        original_query = state.get("original_query")
        session_id = state.get('session_id')

        # 2. 获取历史对话
        #    Mongo 按时间倒序返回（最新在前），拼提示词必须按时间正序，
        #    否则"是的 / 这个 / 它"这类指代消解会接不上上一轮，提取不到商品名
        chat_history = get_recent_messages(session_id, limit=10)
        history_text = ""
        for msg in reversed(chat_history):
            role = msg.get("role")
            content = msg.get("text", "")
            history_text += f"{role}: {content}\n"

        # 3. 调用 LLM 提取商品名
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)
        item_names = clean_llm_result.get('item_names')
        rewritten_query = clean_llm_result.get('rewritten_query')
        print(f"  商品名: {item_names}")
        print(f"  重写查询: {rewritten_query}")
        # 4. 匹配（exact/alias → 向量补充）并对齐
        if item_names:
            confirmed, options, match_basis = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options, match_basis = [], [], []

        # 5. 决策分支，更新 state（含匹配依据 item_match）
        self._decide(state, item_names, confirmed, options, rewritten_query, match_basis)

        # 6. 历史回填
        if confirmed:
            ids_to_update = [
                str(msg["_id"]) for msg in chat_history if not msg.get("item_names")
            ]
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as e:
                    self.logger.warning(f"回填历史 item_names 失败: {e}")

        # 将历史对话写入 state，供下游使用
        state["history"] = chat_history

        return state

    def _decide(self, state: QueryGraphState, item_names: List[str],
                confirmed: List[str], options: List[str], rewritten_query: str,
                match_basis: List[Dict[str, Any]] = None):
        """根据对齐结果更新 state，并把匹配依据写入 state['item_match']"""
        match_basis = match_basis or []

        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
            top_basis = match_basis[0] if match_basis else {}
            state['item_match'] = {
                "extracted": item_names,
                "confirmed": confirmed,
                "match_type": top_basis.get("match_type", "exact"),
                "score": top_basis.get("score", 1.0),
                "candidates": options,
            }
            self.logger.info(
                f"{self.name}, 商品名确认: {confirmed} | 依据={state['item_match']['match_type']} "
                f"score={state['item_match']['score']} | 提取={item_names}"
            )
        elif options:
            has_ambiguity = any(item.get("match_type") == "ambiguous" for item in match_basis)
            best_score = max((float(item.get("score") or 0.0) for item in match_basis), default=0.0)
            state['item_match'] = {
                "extracted": item_names,
                "confirmed": [],
                "match_type": "ambiguous" if has_ambiguity else "vector",
                "score": best_score,
                "candidates": options,
            }
            state['answer'] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
            self.logger.info(
                f"{self.name}, 商品名待确认（不自动认定）: 候选={options} | 依据="
                f"{state['item_match']['match_type']} score={best_score:.4f} | 提取={item_names}"
            )
        else:
            state['item_match'] = {
                "extracted": item_names,
                "confirmed": [],
                "match_type": "none",
                "score": 0.0,
                "candidates": [],
            }
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            self.logger.info(f"{self.name}, 商品名未匹配: 提取={item_names}")


if __name__ == "__main__":
    test_state: QueryGraphState = {
        "original_query": "无线网关用户手册 中包含操作环境嘛？"
    }
    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result = node_item_name_confirm.process(test_state)
    print(f"确认商品: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")