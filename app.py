"""Memory + Planning + ReAct + Reflection Agent 的唯一交互入口。"""

from typing import Any 

from research_agent.agent import create_research_agent
from research_agent.reflection import ReflectionEvaluator
from research_agent.security import AgentSecurityGuard
from research_agent.validation import is_valid_answer


MAX_REFLECTIONS = 2
MAX_RESEARCH_RETRIES = 1

BANNER = """============================
Research Agent
Memory + Planning + ReAct + Reflection
输入 exit 退出
============================"""


def _invoke_answer(executor: Any, prompt: str) -> str:
    """调用现有AgentExecutor并安全提取文本答案。"""
    result = executor.invoke({"input": prompt})
    return str(result.get("output", "")).strip()


def _build_research_prompt(
    question: str,
    plan: str,
    memory_context: str,
) -> str:
    """构造包含问题、适度计划和受限Memory的首次Research输入。"""
    prompt = f"""用户问题：

{question}

研究计划：

{plan}

请执行以上研究计划。工具只用于补充必要证据，已有足够信息后直接输出Final Answer。"""
    if memory_context:
        prompt += f"\n\n已有Memory研究记录：\n\n{memory_context}"
    return prompt


def _research_with_retry(
    executor: Any,
    question: str,
    plan: str,
    memory_context: str,
) -> str:
    """执行Research；失败时最多进行一次限制为单工具调用的受控重试。"""
    prompt = _build_research_prompt(question, plan, memory_context)
    try:
        answer = _invoke_answer(executor, prompt)
    except Exception as exc:
        print(f"Agent调用失败:\n{type(exc).__name__}: {exc}")
        answer = ""

    if is_valid_answer(answer):
        return answer

    print("Research初次执行未正常完成，正在进行一次受控重试。")
    for _ in range(MAX_RESEARCH_RETRIES):
        retry_prompt = f"""上一次研究执行未正常完成。

原问题：
{question}

研究计划：
{plan}

已有Memory：
{memory_context or '无'}

请基于当前Memory和Planning直接回答核心问题。
不要继续扩展搜索。
最多调用1次工具，然后必须输出Final Answer。"""
        try:
            answer = _invoke_answer(executor, retry_prompt)
        except Exception as exc:
            print(f"Research重试失败：{type(exc).__name__}: {exc}")
            answer = ""
        if is_valid_answer(answer):
            return answer

    return ""


def _reflect_and_revise(
    reflection: ReflectionEvaluator | None, 
    question: str,
    answer: str,
) -> str:
    """只对有效初稿进行最多两轮评价，并使用普通LLM无工具修订。"""
    if reflection is None:
        print("\n[Reflection]\n状态: Reflection暂不可用，保留当前有效回答。")
        return answer

    final_answer = answer
    for reflection_round in range(1, MAX_REFLECTIONS + 1):
        print("\n[Reflection]")
        try:
            evaluation = reflection.evaluate(question, final_answer)
        except Exception as exc:
            print(f"状态: Reflection评估失败，保留当前回答。\n{type(exc).__name__}: {exc}")
            break

        print(f"评分: {evaluation.score}/10")
        print(f"问题: {'；'.join(evaluation.issues) if evaluation.issues else '无'}")
        if not evaluation.needs_revision:
            print("状态: 评估通过。")
            break

        if reflection_round >= MAX_REFLECTIONS:
            print("状态: 已达到Reflection轮次上限，输出当前最佳版本。")
            break

        print(f"状态: 未通过，正在进行第{reflection_round}轮无工具修订...")
        try:
            revised = reflection.revise(question, final_answer, evaluation)
            if is_valid_answer(revised):
                final_answer = revised
            else:
                print("Revision返回无效答案，保留上一版回答。")
                break
        except Exception as exc:
            print(f"Revision失败，保留上一版回答：{type(exc).__name__}: {exc}")
            break
    return final_answer


def save_valid_answer(memory: Any, question: str, final_answer: str) -> bool:
    """只把最终有效答案写入Memory，失败或停止信息永不保存。"""
    if memory is None or not is_valid_answer(final_answer):
        return False
    result = memory.save(
        question,
        final_answer,
        metadata={"source": "research"},
    )
    return isinstance(result, str) and result.startswith("mem_")


def check_and_sanitize_user_input(
    guard: AgentSecurityGuard,
    question: str,
) -> str | None:
    """在任何Memory/LLM/Tool调用前阻止高风险输入并清洗中风险输入。"""
    security_result = guard.check_input(question)
    if not security_result["safe"]:
        print("\n[Security]")
        print("检测到高风险输入，本次请求已阻止。")
        print(f"风险等级：{security_result['risk_level']}")
        print(f"风险详情：\n{security_result['details']}")
        return None
    if security_result["risk_level"] == "medium":
        print("\n[Security]")
        print("检测到潜在风险，已进行输入清洗。")
    return guard.sanitize_input(question)


def process_question(
    executor: Any,
    memory: Any,
    planner: Any,
    reflection: ReflectionEvaluator | None,
    question: str,
    guard: AgentSecurityGuard | None = None,
) -> str | None:
    """严格执行Memory、Planning、Research、Reflection和最终保存流程。"""
    memory_context = ""
    print("\n[Memory]")
    try:
        if memory is not None:
            memory_context = memory.search(question)
        if memory_context and memory_context != "Memory暂不可用":
            print("已召回相关历史研究：")
            print(memory_context)
        else:
            print("未召回相关历史研究。")
            memory_context = ""
    except Exception as exc:
        print(f"Memory检索失败，本轮将继续执行：{type(exc).__name__}: {exc}")

    print("\n[Planning]")
    try:
        if planner is None:
            raise RuntimeError("Planner暂不可用")
        plan = planner.generate_plan(question, memory_context)
        print(plan)
    except Exception as exc:
        print(f"Planner生成失败，本轮由ReAct Agent直接处理：{type(exc).__name__}: {exc}")
        plan = "直接回答核心问题，不扩展用户未要求的研究范围。"

    print("\n[Research]")
    answer = _research_with_retry(executor, question, plan, memory_context)
    if not is_valid_answer(answer):
        print("[Research] Research Agent未能正常完成任务，本轮结束。")
        return None

    research_answer = answer.strip()
    if not is_valid_answer(research_answer):
        print("[Research] Research Agent未能正常完成任务，本轮结束。")
        return None
    print(research_answer)

    final_answer = research_answer
    final_answer = _reflect_and_revise(reflection, question, research_answer)
    if not is_valid_answer(final_answer):
        print("最终答案无效，本轮不保存Memory。")
        return None

    safe_final_answer = final_answer
    if guard is not None:
        output_check = guard.check_output(final_answer)
        safe_final_answer = output_check["filtered"]
        if output_check["risks"]:
            print("\n[Security]")
            print("输出中检测到潜在敏感信息，已进行脱敏。")
    if not is_valid_answer(safe_final_answer):
        print("输出安全检查后答案无效，本轮不保存Memory。")
        return None

    print(f"\n最终研究报告：\n\n{safe_final_answer}")
    try:
        if memory is not None and not save_valid_answer(memory, question, safe_final_answer):
            print("Memory保存失败，但最终研究报告已正常返回。")
    except Exception as exc:
        print(f"Memory保存失败，但最终研究报告已正常返回：{type(exc).__name__}: {exc}")
    return safe_final_answer


def process_secured_question(
    guard: AgentSecurityGuard,
    executor: Any,
    memory: Any,
    planner: Any,
    reflection: ReflectionEvaluator | None,
    question: str,
) -> str | None:
    """为测试和复用提供不可绕过的输入Guard到输出Guard完整调用。"""
    safe_question = check_and_sanitize_user_input(guard, question)
    if safe_question is None:
        return None
    return process_question(
        executor,
        memory,
        planner,
        reflection,
        safe_question,
        guard=guard,
    )


def main() -> None:
    """启动唯一CLI入口；单轮失败不会结束后续交互。"""
    print(BANNER)
    executor = memory = planner = reflection = None
    guard = AgentSecurityGuard()
    while True:
        try:
            question = input("\n用户：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出 Research Agent。")
            break
        if question.lower() in {"exit", "quit", "退出"}:
            print("已退出 Research Agent。")
            break
        if not question:
            print("输入不能为空，请重新输入。")
            continue

        safe_question = check_and_sanitize_user_input(guard, question)
        if safe_question is None:
            continue

        if executor is None:
            try:
                executor = create_research_agent()
                metadata = executor.metadata or {}
                memory = metadata.get("vector_memory")
                planner = metadata.get("planner")
            except Exception as exc:
                print(f"Agent初始化失败：\n{type(exc).__name__}: {exc}")
                continue
            try:
                reflection = ReflectionEvaluator()
            except Exception as exc:
                print(f"Reflection初始化失败，将跳过自我评估：{type(exc).__name__}: {exc}")

        final_answer = process_question(
            executor,
            memory,
            planner,
            reflection,
            safe_question,
            guard=guard,
        )
        if final_answer is None:
            continue


if __name__ == "__main__":
    main()
