"""ReflectionEvaluator 的Pydantic JSON解析规则测试。"""

from research_agent.reflection import ReflectionEvaluator


def test_low_score_requires_revision() -> None:
    """低于8分时必须要求修订。"""
    result = ReflectionEvaluator._parse_json(
        '{"score": 6, "needs_revision": false, '
        '"substantive": false, "issues": [], "improvement": "补充场景"}'
    )
    assert result.needs_revision is True


def test_substantive_issue_requires_revision_even_with_high_score() -> None:
    """评分达到8分但存在实质问题时仍必须修订。"""
    result = ReflectionEvaluator._parse_json(
        '{"score": 8, "needs_revision": false, '
        '"substantive": true, "issues": ["存在误导性结论"], "improvement": "修正事实"}'
    )
    assert result.needs_revision is True


def test_high_score_without_issues_passes() -> None:
    """高分且没有实质问题时应通过。"""
    result = ReflectionEvaluator._parse_json(
        '{"score": 9, "needs_revision": false, '
        '"substantive": false, "issues": [], "improvement": "无"}'
    )
    assert result.needs_revision is False


def test_non_substantive_suggestion_does_not_require_revision() -> None:
    """高分回答只有案例、细节或表达建议时不应消耗Revision轮次。"""
    result = ReflectionEvaluator._parse_json(
        '{"score": 8, "needs_revision": true, "substantive": false, '
        '"issues": ["可以增加案例", "可以优化表达"], '
        '"improvement": "后续可按需补充细节"}'
    )

    assert result.issues
    assert result.substantive is False
    assert result.needs_revision is False
