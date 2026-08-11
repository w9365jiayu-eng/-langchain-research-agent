"""Research Agent 外层输入、外部内容和输出安全防护。"""

import logging
from pathlib import Path
import re


MAX_INPUT_CHARS = 5000

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOGGER = logging.getLogger("research_agent.security")
if not _LOGGER.handlers:
    handler = logging.FileHandler(_LOG_DIR / "security.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_INJECTION_PATTERNS = (
    r"忽略(?:之前|以上|所有).{0,12}(?:指令|规则|要求)",
    r"忽略.{0,8}(?:系统|开发者|原始).{0,8}(?:指令|规则|要求)",
    r"忽略(?:指令|规则|要求)",
    r"ignore\s+(?:all\s+|previous\s+)?instructions?",
    r"你现在是.{0,40}",
    r"pretend\s+you\s+are",
    r"\bDAN\b",
    r"\bno\s+(?:restriction|limit)s?\b",
    r"forget\s+everything",
    r"system\s+prompt",
    r"系统提示词",
)
_SENSITIVE_TARGET = re.compile(
    r"(?:api[ _-]?key|api密钥|password|密码|数据库连接串|"
    r"internal\s+document|内部文档|(?:api|access|系统)[ _-]?token|secret|\.env)",
    flags=re.IGNORECASE,
)
_SECRET_REQUEST = re.compile(
    r"(?:告诉我|输出|显示|读取|泄露|给我|获取|提供|查看|打印|"
    r"reveal|show|read|give\s+me|dump)",
    flags=re.IGNORECASE,
)
_SYSTEM_SCOPE = re.compile(
    r"(?:当前系统|系统里|你的|真实|内部|环境变量|\.env|current\s+system|internal)",
    flags=re.IGNORECASE,
)
_EDUCATIONAL_CONTEXT = re.compile(
    r"(?:是什么|什么是|有什么作用|如何防止|如何保护|怎么防止|连接池|"
    r"what\s+is|how\s+to\s+(?:prevent|protect))",
    flags=re.IGNORECASE,
)


def sanitize_external_content(text: str) -> str:
    """移除外部文档中的HTML注释和危险控制字符，不改变正常语义。"""
    without_comments = _HTML_COMMENT.sub("", text or "")
    return _CONTROL_CHARS.sub("", without_comments).strip()


def log_tool_event(tool_name: str, success: bool) -> None:
    """记录工具名称和成功状态，不记录输入、输出或敏感值。"""
    _LOGGER.info("tool=%s success=%s", tool_name, success)


class AgentSecurityGuard:
    """位于Agent外层、不可由LLM选择跳过的安全守卫。"""

    def check_input(self, user_input: str) -> dict:
        """检测直接/间接注入、敏感信息请求和输入长度风险。"""
        text = user_input or ""
        risks: list[str] = []
        injection = any(
            re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            for pattern in _INJECTION_PATTERNS
        )
        if injection:
            risks.append("检测到Prompt Injection模式")
        if _HTML_COMMENT.search(text):
            risks.append("检测到HTML注释形式的潜在间接注入")
        if len(text) > MAX_INPUT_CHARS:
            risks.append("输入过长")

        sensitive_request = bool(
            _SENSITIVE_TARGET.search(text)
            and _SECRET_REQUEST.search(text)
            and not _EDUCATIONAL_CONTEXT.search(text)
        )
        if sensitive_request:
            risks.append("检测到获取真实系统敏感信息的请求")

        if injection or sensitive_request or _HTML_COMMENT.search(text) or len(text) > MAX_INPUT_CHARS:
            risk_level = "high"
        elif risks:
            risk_level = "medium"
        else:
            risk_level = "low"
        safe = risk_level != "high"
        details = "；".join(risks) if risks else "未检测到明显风险"
        _LOGGER.info("input_risk=%s blocked=%s reasons=%s", risk_level, not safe, details)
        return {
            "safe": safe,
            "risk_level": risk_level,
            "details": details,
            "risks": risks,
        }

    def sanitize_input(self, user_input: str) -> str:
        """仅移除HTML注释、危险控制字符和首尾空白。"""
        return sanitize_external_content(user_input)

    def check_output(self, agent_output: str) -> dict:
        """检测并脱敏API Key、密码和数据库连接串。"""
        filtered = agent_output or ""
        risks: list[str] = []

        database_pattern = re.compile(
            r"(?:mongodb(?:\+srv)?|mysql|postgres(?:ql)?|redis)://[^\s\"'<>]+",
            flags=re.IGNORECASE,
        )
        if database_pattern.search(filtered):
            filtered = database_pattern.sub("[已脱敏: 数据库连接串]", filtered)
            risks.append("数据库连接串")

        api_key_pattern = re.compile(
            r"(?:sk-[A-Za-z0-9_-]{16,}|api_[A-Za-z0-9_-]{12,})",
            flags=re.IGNORECASE,
        )
        if api_key_pattern.search(filtered):
            filtered = api_key_pattern.sub("[已脱敏: API Key]", filtered)
            risks.append("API Key")

        password_pattern = re.compile(
            r"\bpassword\s*[:=]\s*[^\s,;]+",
            flags=re.IGNORECASE,
        )
        if password_pattern.search(filtered):
            filtered = password_pattern.sub("[已脱敏: 密码]", filtered)
            risks.append("密码")

        _LOGGER.info("output_safe=%s risk_types=%s", not risks, ",".join(risks) or "none")
        return {"safe": not risks, "filtered": filtered, "risks": risks}
