"""Research Agent 输出与历史记忆的统一有效性校验。"""


INVALID_ANSWER_MARKERS = (
    "agent stopped due to iteration limit",
    "time limit",
    "agent调用失败",
    "未能完成推理",
)


def contains_failure_marker(text: str) -> bool:
    """判断文本是否包含已知的 Agent 执行失败标记。"""
    normalized = (text or "").strip().lower()
    return any(marker in normalized for marker in INVALID_ANSWER_MARKERS)


def is_valid_answer(answer: str) -> bool:
    """仅当答案非空且不包含执行失败标记时返回 True。"""
    return bool(answer and answer.strip()) and not contains_failure_marker(answer)
