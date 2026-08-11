"""互联网搜索工具。"""

from ddgs import DDGS
from langchain_core.tools import tool

from ..security import log_tool_event, sanitize_external_content


STOP_HINT = "已有搜索结果，请基于当前信息总结，不要继续重复搜索。"


@tool
def web_search(query: str) -> str:
    """搜索互联网信息。当用户需要最新资料、新闻、论文或外部知识时调用。输入搜索关键词，返回标题、URL 和摘要。"""
    query = query.strip()
    if not query:
        log_tool_event("web_search", False)
        return "搜索失败：搜索关键词不能为空。"

    try:
        results = list(DDGS().text(query, max_results=5))
        if not results:
            log_tool_event("web_search", False)
            return f"未找到与“{query}”相关的搜索结果。"

        formatted = []
        for index, item in enumerate(results, start=1):
            formatted.append(
                f"结果 {index}\n"
                f"标题：{item.get('title', '无标题')}\n"
                f"URL：{item.get('href', item.get('url', '无链接'))}\n"
                f"摘要：{item.get('body', '无摘要')}"
            )
        log_tool_event("web_search", True)
        return sanitize_external_content("\n\n".join(formatted)) + f"\n\n{STOP_HINT}"
    except Exception as exc:  # 第三方搜索可能因网络或限流失败
        log_tool_event("web_search", False)
        return f"搜索失败：{type(exc).__name__}: {exc}"
