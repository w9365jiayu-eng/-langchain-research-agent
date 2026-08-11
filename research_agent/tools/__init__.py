"""集中注册 Agent 可调用的工具。"""

from .search_tool import web_search
from .web_scraper import fetch_webpage
from .pdf_parser import parse_pdf


TOOLS = [web_search, fetch_webpage, parse_pdf]

__all__ = ["TOOLS", "web_search", "fetch_webpage", "parse_pdf"]
