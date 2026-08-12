"""Continuous Evaluation 评分、报告与 baseline 的隔离测试。"""

import json

from evaluation.report import (
    _bar,
    compare_with_baseline,
    update_baseline,
    write_health_report,
)
from evaluation.run_evaluation import parse_pytest_report
from evaluation.scorer import AgentScore


def test_agent_score_uses_required_weights() -> None:
    """综合分必须使用功能50%、安全30%、效率20%的权重。"""
    result = AgentScore().calculate(
        {
            "total": 43,
            "passed": 43,
            "security_total": 10,
            "security_passed": 10,
        },
        {"total": 5, "passed": 5, "avg_steps": 0.0, "avg_time": 0.0},
    )

    assert result.functional_score == 100.0
    assert result.security_score == 100.0
    assert result.efficiency_score == 100.0
    assert result.overall_score == 100.0


def test_regression_requires_more_than_five_point_drop() -> None:
    """只有综合分比 baseline 低超过5分时才标记退化。"""
    assert not compare_with_baseline(
        {"overall_score": 80.0}, {"overall_score": 85.0}
    )["degradation_detected"]
    comparison = compare_with_baseline(
        {"overall_score": 79.9}, {"overall_score": 85.0}
    )
    assert comparison["degradation_detected"]
    assert comparison["message"] == "Performance degradation detected"


def test_parse_pytest_report_counts_security_tests(tmp_path) -> None:
    """JUnit统计应独立计算名称或类名包含security的测试。"""
    report = tmp_path / "pytest.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite tests="3" failures="1">
  <testcase classname="tests.test_memory" name="test_memory" />
  <testcase classname="tests.test_security" name="test_safe" />
  <testcase classname="tests.test_security" name="test_block"><failure /></testcase>
</testsuite></testsuites>""",
        encoding="utf-8",
    )

    summary = parse_pytest_report(report)

    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["security_total"] == 2
    assert summary["security_passed"] == 1


def test_health_report_and_baseline_are_written(tmp_path) -> None:
    """健康报告和未退化 baseline 都应写入可读取的JSON。"""
    report_path = tmp_path / "evaluation_results.json"
    baseline_path = tmp_path / "baseline.json"
    scores = {
        "functional_score": 100.0,
        "security_score": 100.0,
        "efficiency_score": 80.0,
        "overall_score": 96.0,
    }
    comparison = compare_with_baseline(scores, {"overall_score": 85.0})

    write_health_report(
        report_path,
        {"total": 43, "passed": 43},
        {"summary": {"avg_steps": 2.0, "avg_time": 10.0}},
        scores,
        comparison,
    )
    update_baseline(
        baseline_path,
        scores,
        {"avg_steps": 2.0, "avg_time": 10.0},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert report["scores"]["overall_score"] == 96.0
    assert report["status"] == "healthy"
    assert baseline["overall_score"] == 96.0
    assert baseline["avg_steps"] == 2.0


def test_dashboard_bar_supports_non_unicode_console(monkeypatch) -> None:
    """Windows GBK等终端无法显示块字符时必须降级为ASCII。"""
    class AsciiStdout:
        encoding = "ascii"

    monkeypatch.setattr("evaluation.report.sys.stdout", AsciiStdout())
    assert _bar(70.0) == "#######---"
