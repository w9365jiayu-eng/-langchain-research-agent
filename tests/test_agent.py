"""Research Agent 端到端工具调用测试。运行前需配置真实 LLM API。"""

from pathlib import Path
from typing import Iterable

from research_agent.agent import create_research_agent


SAMPLE_PDF = Path("tests/data/sample.pdf")
TEST_CASES = [
    ("Test 1 - 网页研究", "研究一下RAG最近一年有哪些重要发展趋势，搜索资料并总结", {"web_search"}),
    ("Test 2 - 网页抓取", "请打开这个网页并总结内容：https://www.langchain.com/", {"fetch_webpage"}),
    ("Test 3 - PDF解析", f"请读取这个PDF并总结内容：{SAMPLE_PDF.as_posix()}", {"parse_pdf"}),
    (
        "Test 4 - 多工具组合",
        f"搜索RAG最新研究，并结合PDF资料 {SAMPLE_PDF.as_posix()} 总结趋势",
        {"web_search", "parse_pdf"},
    ),
]


def print_intermediate_steps(steps: Iterable[tuple]) -> set[str]:
    """打印 Agent Action、工具输入与 Observation，并返回调用过的工具名。"""
    called_tools: set[str] = set()
    print("\n--- Agent Actions 与工具调用结果 ---")
    for index, (action, observation) in enumerate(steps, start=1):
        called_tools.add(action.tool)
        print(f"Step {index}")
        print(f"Agent Action: {action.tool}")
        print(f"Action Input: {action.tool_input}")
        print(f"Tool调用结果: {observation}")
    return called_tools


def run_test(agent_executor, name: str, question: str, expected: set[str]) -> bool:
    """执行一个完整 Agent 回合并验证预期工具是否被实际调用。"""
    print("\n=====================")
    print(name)
    print("=====================")
    print(f"用户输入: {question}\n")
    try:
        result = agent_executor.invoke({"input": question})
        called = print_intermediate_steps(result.get("intermediate_steps", []))
        print(f"\nFinal Answer:\n{result.get('output', 'Agent未返回答案')}")
        missing = expected - called
        if missing:
            print(f"\n[FAIL] 未调用预期工具: {', '.join(sorted(missing))}")
            return False
        print(f"\n[PASS] 已调用预期工具: {', '.join(sorted(expected))}")
        return True
    except Exception as exc:
        print(f"[FAIL] Agent调用异常：{type(exc).__name__}: {exc}")
        return False


def main() -> None:
    """依次运行四个端到端场景并打印汇总结果。"""
    if not SAMPLE_PDF.is_file():
        print(f"测试PDF不存在：{SAMPLE_PDF}。请先恢复仓库中的测试文件。")
        raise SystemExit(1)
    try:
        agent_executor = create_research_agent()
    except Exception as exc:
        print(f"Agent初始化失败：{exc}")
        print("请先复制 .env.example 为 .env，并配置真实的 LLM API。")
        raise SystemExit(1)

    results = [run_test(agent_executor, name, question, expected) for name, question, expected in TEST_CASES]
    passed = sum(results)
    print(f"\n测试汇总：{passed}/{len(results)} 通过")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
