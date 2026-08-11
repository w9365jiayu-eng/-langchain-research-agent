"""带研究规划、向量记忆和 ReAct 工具调用的交互式入口。"""

from research_agent.agent import create_research_agent


BANNER = """==========================
Research Agent Planning Chat

输入 exit 退出
=========================="""


def main() -> None:
    """运行 Memory检索、规划、ReAct执行和记忆保存的交互循环。"""
    print(BANNER)
    try:
        executor = create_research_agent()
        metadata = executor.metadata or {}
        memory = metadata.get("vector_memory")
        planner = metadata.get("planner")
    except Exception as exc:
        print(f"Agent初始化失败：{exc}")
        return

    while True:
        try:
            question = input("\n用户：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出 Research Agent Planning Chat。")
            break

        if question.lower() in {"exit", "quit", "退出"}:
            print("已退出 Research Agent Planning Chat。")
            break
        if not question:
            print("输入不能为空，请重新输入。")
            continue

        memory_context = ""
        if memory:
            memory_context = memory.search(question)
            if memory_context == "Memory暂不可用":
                print("Memory暂不可用，将继续生成研究计划。")
                memory_context = ""
            elif memory_context:
                print("Memory检索：已召回相关历史研究。")

        try:
            if planner is None:
                raise RuntimeError("Planner未初始化")
            plan = planner.generate_plan(question, memory_context)
            print(f"\n研究计划：\n{plan}")
            agent_input = f"""用户问题：

{question}

研究计划：

{plan}

请执行以上研究计划。
已有足够信息后直接输出最终研究报告。"""
            if memory_context:
                agent_input += f"\n\n已有Memory研究记录：\n\n{memory_context}"
        except Exception as exc:
            print(f"Planner生成失败，将由ReAct Agent直接处理：{exc}")
            plan = "Planner暂不可用，由ReAct Agent直接规划并执行。"
            agent_input = question

        try:
            result = executor.invoke({"input": agent_input})
            answer = result.get("output", "Agent未返回答案。")
            print(f"\n最终答案：\n{answer}")
        except Exception as exc:
            print(f"Agent调用失败，请重试：{type(exc).__name__}: {exc}")
            continue

        if memory:
            save_result = memory.save(question, answer, {"plan": plan})
            if save_result == "Memory暂不可用":
                print("Memory暂不可用，本轮答案未保存，但不影响回答。")


if __name__ == "__main__":
    main()
