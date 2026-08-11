"""PDF 文档解析工具。"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool

from ..security import log_tool_event, sanitize_external_content

MAX_CONTENT_LENGTH = 5000


@tool
def parse_pdf(file_path: str) -> str:
    """读取本地 PDF 文档。当用户提供 PDF 路径，需要读取论文、报告或文档内容时调用。"""
    raw_path = file_path.strip().strip("\"'")
    path = Path(raw_path).expanduser()
    project_root = Path(__file__).resolve().parent.parent.parent
    try:
        resolved_path = path.resolve()
    except OSError:
        log_tool_event("parse_pdf", False)
        return "PDF解析失败：文件路径无效"
    if (
        ".." in path.parts
        or not resolved_path.is_relative_to(project_root)
        or resolved_path.suffix.lower() != ".pdf"
        or any(part.startswith(".") for part in resolved_path.relative_to(project_root).parts)
    ):
        log_tool_event("parse_pdf", False)
        return "PDF解析失败：仅允许读取项目目录内明确指定的非隐藏PDF文件"
    if not resolved_path.is_file():
        log_tool_event("parse_pdf", False)
        return "PDF解析失败：文件不存在"

    try:
        pages = PyPDFLoader(str(resolved_path)).load()
        if not pages:
            log_tool_event("parse_pdf", False)
            return "PDF解析失败：PDF中没有可读取的页面"
        title = next(
            (str(page.metadata.get("title", "")).strip() for page in pages if page.metadata.get("title")),
            resolved_path.stem,
        )
        text = sanitize_external_content(
            "\n\n".join(page.page_content.strip() for page in pages)
        )
        if not text:
            log_tool_event("parse_pdf", False)
            return "PDF解析失败：未提取到文本内容"
        header = f"PDF标题：{title}\n页数：{len(pages)}\n主要文本内容：\n"
        available_length = max(0, MAX_CONTENT_LENGTH - len(header))
        if len(text) > available_length:
            text = text[:available_length] + "\n[内容已截断，最多返回5000字符]"
        log_tool_event("parse_pdf", True)
        return sanitize_external_content(header + text)[:MAX_CONTENT_LENGTH]
    except Exception as exc:
        log_tool_event("parse_pdf", False)
        return f"PDF解析失败：{type(exc).__name__}: {exc}"
