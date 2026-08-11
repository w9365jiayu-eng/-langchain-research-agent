"""AgentEvaluator判定、轨迹和健壮性测试。"""

import json
from pathlib import Path

from evaluation.evaluator import AgentEvaluator, AgentTrace, TestCase as EvalTestCase, _EvaluationRuntime
from evaluation import eval_agent


REPORT_PATH = Path("tests/data/evaluation_test_results.json")


def test_trace_counts_only_react_tool_actions() -> None:
    """steps只统计intermediate_steps中的Agent Action。"""
    action1 = type("Action", (), {"tool": "web_search"})()
    action2 = type("Action", (), {"tool": "fetch_webpage"})()
    tools, steps = _EvaluationRuntime._trace_steps(
        {"intermediate_steps": [(action1, "结果1"), (action2, "结果2")]}
    )
    assert tools == ["web_search", "fetch_webpage"]
    assert steps == 2


def test_run_test_checks_tools_keywords_steps_and_security() -> None:
    """单测试应同时检查工具、关键词、步数和阻断状态。"""
    evaluator = AgentEvaluator(
        runner=lambda question: AgentTrace(
            output="RAG研究结论",
            tools_called=["web_search"],
            steps=1,
        )
    )
    result = evaluator.run_test(
        EvalTestCase(
            name="通过场景",
            input="研究RAG",
            expected_tools=["web_search"],
            expected_keywords=["RAG"],
            max_steps=2,
        )
    )
    assert result.passed is True


def test_runner_exception_becomes_failed_result() -> None:
    """LLM/Tool异常必须转换为失败结果而不是终止suite。"""
    def failing_runner(question):
        raise TimeoutError("模拟超时")

    result = AgentEvaluator(runner=failing_runner).run_test(
        EvalTestCase(name="异常场景", input="问题")
    )
    assert result.passed is False
    assert "Agent运行异常: TimeoutError" in result.details


def test_suite_continues_and_writes_json_report() -> None:
    """某项失败后suite继续，并生成不含秘密的结构化报告。"""
    def runner(question):
        if question == "失败":
            raise RuntimeError("测试异常")
        return AgentTrace(output="安全输出", tools_called=[], steps=0)

    try:
        evaluator = AgentEvaluator(runner=runner, report_path=str(REPORT_PATH))
        summary = evaluator.run_suite(
            [
                EvalTestCase(name="成功", input="成功"),
                EvalTestCase(name="失败", input="失败"),
            ]
        )
        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 0.5
        assert len(payload["results"]) == 2
        assert "sk-" not in REPORT_PATH.read_text(encoding="utf-8")
    finally:
        REPORT_PATH.unlink(missing_ok=True)


def test_eval_agent_exits_zero_only_when_all_tests_pass(monkeypatch) -> None:
    """CI评估必须全部通过才返回0，任何失败都返回1。"""
    class FakeEvaluator:
        def __init__(self, summary):
            self.summary = summary

        def run_suite(self, test_cases):
            return self.summary

    for summary, expected_code in [
        ({"passed": 5, "total": 5}, 0),
        ({"passed": 4, "total": 5}, 1),
    ]:
        monkeypatch.setattr(
            eval_agent,
            "AgentEvaluator",
            lambda summary=summary: FakeEvaluator(summary),
        )
        try:
            eval_agent.main()
        except SystemExit as exc:
            assert exc.code == expected_code
        else:
            raise AssertionError("eval_agent.main()必须通过SystemExit返回CI退出码")
