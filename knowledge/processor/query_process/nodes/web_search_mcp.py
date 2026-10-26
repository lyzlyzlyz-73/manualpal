import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from agents.mcp import MCPServerSse, MCPServerStreamableHttp
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    """MCP 网络搜索节点。

    负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
    MCP 形式调用第三方的各种通用的搜索工具。
    """

    name = "mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """执行网络搜索。

        Args:
            state: 查询图状态，需包含 rewritten_query 和 item_names。

        Returns:
            更新后的状态，包含 web_search_docs 搜索结果。
        """
        # 1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 创建mcp_client 并且让客户端执行工具 bailian_web_search
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            return state

        # 3. 只更新 state 的 web_search_docs
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数。

        Args:
            state: 查询图状态。

        Returns:
            校验后的 rewritten_query 和 item_names。

        Raises:
            StateFieldError: 参数校验失败时抛出。
        """
        # 1. 获取 state 的 rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取 state 的 item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_query):
        """
        使用 async with 上下文管理器自动管理连接生命周期：
        - 进入 with 块时自动建立连接（connect）
        - 退出 with 块时自动关闭连接（cleanup），即使抛异常也会执行

        Args:
            validated_query: 校验后的查询文本。

        Returns:
            搜索结果列表，每项包含 title、url、snippet。
        """
        # 1. 定义 MCP 客户端（Streamable HTTP 模式，自动管理连接）
        async with MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {self.config.dashscope_api_key}"},
                "timeout": 300,
                "terminate_on_close": True,
            },
            max_retry_attempts=2,
            cache_tools_list=True,
            client_session_timeout_seconds=60,  # 默认 5 秒不够：首次连接是冷启动，联网搜索本身也慢
        ) as client:
            # 2. 调用工具
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_query, "count": 3}
            )

            print(execute_tool_result)
            # 3. 解析结果
            if not execute_tool_result or not execute_tool_result.content or not execute_tool_result.content[0]:
                return []
            text_json = json.loads(execute_tool_result.content[0].text)
            if not text_json:
                return []
            pages = text_json.get("pages")
            if not pages:
                return []
            web_search_docs = []
            for page in pages:
                snippet = page.get("snippet")
                title = page.get("title")
                url = page.get("url")
                web_search_docs.append(
                    {
                        "snippet": snippet,
                        "title": title,
                        "url": url
                    }
                )
            return web_search_docs


#pip install mcp==1.12.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

if __name__ == '__main__':
    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    mcp_search = McpSearchNode()
    result = mcp_search.process(state)

    for r in result.get('web_search_docs'):
        print(json.dumps(r, ensure_ascii=False, indent=2))


# ================================================================== #
#                    节点速查（输入 / 处理 / 输出）                    #
# ------------------------------------------------------------------ #
# 【拿了什么数据】state 里的
#   - rewritten_query: 送给搜索工具的查询词
#   - item_names:      只做参数校验，不参与过滤（联网搜索不受商品名限制）
#
# 【用了什么服务 / 工具】
#   - DashScope MCP（Streamable HTTP）：config.mcp_dashscope_base_url
#     鉴权：headers 里带 Bearer config.dashscope_api_key
#     客户端：openai-agents 的 MCPServerStreamableHttp（连接/断开由 async with 自动管理）
#   - 工具：bailian_web_search，参数 {"query": rewritten_query, "count": 3}
#   - 依赖版本：mcp==1.12.0（2.x 会先发 server/discover 协议协商，DashScope 返回 500）
#
# 【做了什么事情】
#   校验 → asyncio.run 连接 MCP → 调用 bailian_web_search
#   → 解析 content[0].text（JSON 字符串）→ 取 pages 映射成 {title, url, snippet}
#
# 【更新了什么字段】
#   函数返回 {"web_search_docs": [{"title": ..., "url": ..., "snippet": ...}, ...]}
#   注意：这里保存的是"网页摘要"，没有 chunk_id / item_name
#
#   结构示例（真实产物，count=3，snippet 已截断）:
#   [
#     {
#       "title": "小马智行-W",
#       "url": "http://stock.finance.sina.com.cn/hkstock/quotes/02026.html",
#       "snippet": "名称: 小马智行-W, 股票代码: 02026.HK, 市场: 港股, 币种: 港币, 价格: 39.320, 涨跌幅: -8.217%..."
#     },
#     {
#       "title": "BBA和特斯拉全被干翻!小米SU7拿下20万元以上轿车销量第一",
#       "url": "https://www.163.com/dy/article/L94Q99UO0511CPVM.html",
#       "snippet": "小米SU7拿下20万元以上轿车销量第一 快科技10月13日消息,小米汽车官方今日宣布..."
#     },
#     { "title": "小米汽车:2026 年 9 月,SU7 位列 20 万以上轿车销量第一", "url": "https://stock.10jqka.com.cn/...", "snippet": "..." }
#   ]
#   （MCP 原始返回每条还带 hostname / hostlogo，这里只映射了 title / url / snippet）
#
# 【不更新字段时的返回】
#   - 解析不出 pages / 内容为空 → 内部返回 []，process 里 if not mcp_result: return state
#     → 不写 web_search_docs（保持默认 []）
#   - 连接或调用抛异常（超时、鉴权、网络）→ 异常向上冒泡，节点被标记失败
#
# 备注：第 104 行的 print(execute_tool_result) 是调试用的原始返回，正式跑可删
# ================================================================== #