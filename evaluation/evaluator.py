"""独立于正式运行时的Research Agent自动评估框架。"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Callable, List

from research_agent.agent import create_research_agent
from app import MAX_REFLECTIONS, _build_research_prompt
from research_agent.config import get_settings
from research_agent.memory import VectorMemory
from research_agent.reflection import ReflectionEvaluator
from research_agent.security import AgentSecurityGuard
from research_agent.validation import is_valid_answer


EVAL_MEMORY_DIR = "./eval_agent_memory"


@dataclass
class TestCase:
    """描述一个Agent能力评估场景及其通过条件。"""

    name: str
    input: str
    expected_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    not_expected_keywords: List[str] = field(default_factory=list)
    max_steps: int = 10
    max_time_seconds: float = 120.0
    expect_blocked_by_security: bool = False


@dataclass
class TestResult:
    """记录单个评估场景的观测值与判定结果。"""

    name: str
    passed: bool
    steps_used: int
    time_seconds: float
    tools_called: List[str]
    output: str
    details: str = ""
    blocked_by_security: bool = False


@dataclass
class AgentTrace:
    """一次Agent评估执行的安全输出与ReAct轨迹。"""

    output: str
    tools_called: list[str]
    steps: int
    blocked_by_security: bool = False
    research_success: bool = True


class _EvaluationRuntime:
    """持有一次评估suite共享的独立Memory和Agent组件。"""

    def __init__(self) -> None:
        get_settings()  # 在加载Embedding前快速校验评估所需LLM配置。
        self.guard = AgentSecurityGuard()
        self.memory = VectorMemory(EVAL_MEMORY_DIR)
        self._reset_and_seed_memory()
        self.executor = create_research_agent(memory=self.memory)
        self.planner = (self.executor.metadata or {}).get("planner")
        self.reflection = ReflectionEvaluator()

    def _reset_and_seed_memory(self) -> None:
        """仅清空eval collection并写入固定RAG种子，不触碰正式Memory。"""
        if self.memory.collection is None:
            return
        existing = self.memory.collection.get()
        ids = existing.get("ids") or []
        if ids:
            self.memory.collection.delete(ids=ids)
        self.memory.save(
            "什么是RAG？",
            "RAG是Retrieval-Augmented Generation，一种检索增强生成技术。",
            metadata={"source": "evaluation_seed"},
        )

    @staticmethod
    def _trace_steps(result: dict) -> tuple[list[str], int]:
        """从LangChain intermediate_steps中提取工具名称和Action次数。"""
        intermediate_steps = result.get("intermediate_steps") or []
        tools = [action.tool for action, _ in intermediate_steps]
        return tools, len(intermediate_steps)

    def run(self, question: str) -> AgentTrace:
        """执行Security、Memory、Planning、ReAct、Reflection和Output Guard。"""
        input_check = self.guard.check_input(question)
        if input_check["risk_level"] == "high":
            return AgentTrace(
                output=f"请求已被安全层阻止：{input_check['details']}",
                tools_called=[],
                steps=0,
                blocked_by_security=True,
                research_success=False,
            )
        safe_question = self.guard.sanitize_input(question)
        memory_context = self.memory.search(safe_question)
        if memory_context == "Memory暂不可用":
            memory_context = ""
        plan = self.planner.generate_plan(safe_question, memory_context)
        prompt = _build_research_prompt(safe_question, plan, memory_context)

        all_tools: list[str] = []
        result = self.executor.invoke({"input": prompt})
        tools, steps = self._trace_steps(result)
        all_tools.extend(tools)
        answer = str(result.get("output", "")).strip()

        if not is_valid_answer(answer):
            retry_prompt = f"""上一次研究未正常完成。请直接回答：{safe_question}
最多调用1次工具，然后必须输出Final Answer。"""
            result = self.executor.invoke({"input": retry_prompt})
            retry_tools, retry_steps = self._trace_steps(result)
            all_tools.extend(retry_tools)
            steps += retry_steps
            answer = str(result.get("output", "")).strip()
        if not is_valid_answer(answer):
            return AgentTrace(
                output="Research Agent未能正常完成任务",
                tools_called=all_tools,
                steps=steps,
                research_success=False,
            )

        final_answer = answer
        for reflection_round in range(1, MAX_REFLECTIONS + 1):
            evaluation = self.reflection.evaluate(safe_question, final_answer)
            if not evaluation.needs_revision:
                break
            if reflection_round >= MAX_REFLECTIONS:
                break
            revised = self.reflection.revise(safe_question, final_answer, evaluation)
            if is_valid_answer(revised):
                final_answer = revised

        output_check = self.guard.check_output(final_answer)
        return AgentTrace(
            output=output_check["filtered"],
            tools_called=all_tools,
            steps=steps,
            research_success=True,
        )


_DEFAULT_RUNTIME: _EvaluationRuntime | None = None
_PRECHECK_GUARD = AgentSecurityGuard()


def run_agent_with_trace(question: str) -> AgentTrace:
    """使用独立eval Memory运行Agent，并返回最终答案和LangChain工具轨迹。"""
    global _DEFAULT_RUNTIME
    input_check = _PRECHECK_GUARD.check_input(question)
    if input_check["risk_level"] == "high":
        return AgentTrace(
            output=f"请求已被安全层阻止：{input_check['details']}",
            tools_called=[],
            steps=0,
            blocked_by_security=True,
            research_success=False,
        )
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = _EvaluationRuntime()
    return _DEFAULT_RUNTIME.run(question)


class AgentEvaluator:
    """运行、观察、判定并汇总Research Agent评估用例。"""

    def __init__(
        self,
        runner: Callable[[str], AgentTrace] = run_agent_with_trace,
        report_path: str = "evaluation_results.json",
    ) -> None:
        """注入可替换runner，便于隔离测试Evaluator自身。"""
        self.runner = runner
        self.report_path = Path(report_path)

    def run_test(self, test_case: TestCase) -> TestResult:
        """运行单个用例，检查工具、关键词、步数、耗时和安全阻断。"""
        started = time.perf_counter()
        try:
            trace = self.runner(test_case.input)
            elapsed = time.perf_counter() - started
            failures: list[str] = []
            for tool_name in test_case.expected_tools:
                if tool_name not in trace.tools_called:
                    failures.append(f"期望调用{tool_name}但未调用")
            for tool_name in test_case.forbidden_tools:
                if tool_name in trace.tools_called:
                    failures.append(f"调用了禁止工具{tool_name}")

            normalized_output = trace.output.lower()
            for keyword in test_case.expected_keywords:
                if keyword.lower() not in normalized_output:
                    failures.append(f"输出缺少关键词：{keyword}")
            for keyword in test_case.not_expected_keywords:
                if keyword.lower() in normalized_output:
                    failures.append(f"输出包含禁止关键词：{keyword}")
            if trace.steps > test_case.max_steps:
                failures.append(f"工具步数{trace.steps}超过上限{test_case.max_steps}")
            if elapsed > test_case.max_time_seconds:
                failures.append(
                    f"耗时{elapsed:.2f}s超过上限{test_case.max_time_seconds:.2f}s"
                )
            if trace.blocked_by_security != test_case.expect_blocked_by_security:
                failures.append("Security Guard阻断状态不符合预期")
            if not test_case.expect_blocked_by_security and not trace.research_success:
                failures.append("Research Agent未生成有效答案")

            return TestResult(
                name=test_case.name,
                passed=not failures,
                steps_used=trace.steps,
                time_seconds=elapsed,
                tools_called=trace.tools_called,
                output=trace.output,
                details="；".join(failures),
                blocked_by_security=trace.blocked_by_security,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return TestResult(
                name=test_case.name,
                passed=False,
                steps_used=0,
                time_seconds=elapsed,
                tools_called=[],
                output="",
                details=f"Agent运行异常: {type(exc).__name__}: {exc}",
            )

    def run_suite(self, test_cases: list[TestCase]) -> dict:
        """运行全部用例，单例失败不中断，并打印/保存汇总报告。"""
        results: list[TestResult] = []
        for test_case in test_cases:
            result = self.run_test(test_case)
            results.append(result)
            icon = "✅" if result.passed else "❌"
            try:
                icon.encode(sys.stdout.encoding or "utf-8")
            except UnicodeEncodeError:
                icon = "[PASS]" if result.passed else "[FAIL]"
            print(
                f"{icon} {result.name} "
                f"({result.time_seconds:.1f}s, {result.steps_used}步)"
            )
            if result.details:
                print(f"   → {result.details}")

        total = len(results)
        passed = sum(result.passed for result in results)
        avg_time = sum(result.time_seconds for result in results) / total if total else 0.0
        avg_steps = sum(result.steps_used for result in results) / total if total else 0.0
        summary = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_time": avg_time,
            "avg_steps": avg_steps,
        }
        print("=" * 50)
        print(f"测试结果: {passed}/{total} 通过 ({summary['pass_rate']:.0%})")
        print(f"平均耗时: {avg_time:.1f}s")
        print(f"平均步数: {avg_steps:.1f}")
        print("=" * 50)

        payload = {
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        self.report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
