"""用于手动验证 Research Agent 长期向量记忆的命令行入口。"""

from research_agent.agent import create_research_agent


BANNER = """====================
Research Agent Memory Chat
输入 exit 退出
===================="""


def main() -> None:
    """启动带 VectorMemory 检索与保存能力的交互式 Agent。"""
    print(BANNER)
    try:
        executor = create_research_agent()
        memory = (executor.metadata or {}).get("vector_memory")
    except Exception as exc:
        print(f"Agent初始化失败。请检查API配置：{exc}")
        return

    while True:
        try:
            question = input("\n用户：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出 Research Agent Memory Chat。")
            break

        if question.lower() in {"exit", "quit", "退出"}:
            print("已退出 Research Agent Memory Chat。")
            break
        if not question:
            print("输入不能为空，请重新输入。")
            continue

        try:
            memory_text = memory.search(question) if memory else "Memory暂不可用"
            if memory_text and memory_text != "Memory暂不可用":
                agent_input = (
                    f"当前问题：\n{question}\n\n"
                    f"相关历史研究记录：\n{memory_text}\n\n"
                    "请结合历史信息回答。"
                )
                print("Memory检索：已找到相关历史研究记录。")
            else:
                agent_input = question
                print("Memory检索：暂无相关历史记录。")

            result = executor.invoke({"input": agent_input})
            answer = result.get("output", "Agent未返回答案。")
            print(f"\n最终答案：\n{answer}")

            if memory:
                save_result = memory.save(question, answer)
                if save_result.startswith("mem_"):
                    print("Memory保存成功。")
                else:
                    print("Memory保存失败，但答案正常返回。")
            else:
                print("Memory保存失败，但答案正常返回。")
        except Exception as exc:
            print(f"Agent调用失败，请重试。错误信息：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
