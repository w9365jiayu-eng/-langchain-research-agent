"""网页正文抓取工具。"""

import re

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from ..security import log_tool_event, sanitize_external_content


MAX_CONTENT_LENGTH = 5000
REQUEST_TIMEOUT = 15


@tool
def fetch_webpage(url: str) -> str:
    """读取网页详细内容。当已有网页 URL、需要查看其正文和细节时调用；输入一个 HTTP/HTTPS URL，返回清洗后的正文文本。"""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        log_tool_event("fetch_webpage", False)
        return "网页访问失败：URL 必须以 http:// 或 https:// 开头。"

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        if not text:
            log_tool_event("fetch_webpage", False)
            return "网页访问失败：页面为空或未提取到正文。"

        text = sanitize_external_content(text)
        log_tool_event("fetch_webpage", True)
        if len(text) > MAX_CONTENT_LENGTH:
            return text[:MAX_CONTENT_LENGTH] + "\n\n[内容已截断，最多返回 5000 字符]"
        return text
    except requests.Timeout:
        log_tool_event("fetch_webpage", False)
        return f"网页访问失败：请求超时（{REQUEST_TIMEOUT} 秒）。"
    except requests.HTTPError as exc:
        log_tool_event("fetch_webpage", False)
        status = exc.response.status_code if exc.response is not None else "未知"
        return f"网页访问失败：HTTP 状态码 {status}。"
    except requests.RequestException as exc:
        log_tool_event("fetch_webpage", False)
        return f"网页访问失败：{type(exc).__name__}: {exc}"
    except Exception as exc:
        log_tool_event("fetch_webpage", False)
        return f"网页解析失败：{type(exc).__name__}: {exc}"
