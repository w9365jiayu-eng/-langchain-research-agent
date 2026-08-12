"""VectorMemory 语义检索与 Agent 多轮记忆集成测试。"""

import os
import shutil

from research_agent.agent import create_research_agent
from research_agent.memory import MEMORY_DISTANCE_THRESHOLD, VectorMemory


TEST_MEMORY_DIR = "./test_agent_memory"


class FakeCollection:
    """用固定距离模拟 Chroma 查询，避免测试依赖真实 embedding 波动。"""

    def __init__(self, distance: float) -> None:
        self.distance = distance

    def count(self) -> int:
        return 1

    def query(self, **kwargs):
        assert "distances" in kwargs["include"]
        return {
            "documents": [["问题:\nRAG和Fine-tuning哪个好？\n\n回答:\n两者适用于不同场景。"]],
            "metadatas": [[{
                "question": "RAG和Fine-tuning哪个好？",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }]],
            "distances": [[self.distance]],
        }


def _memory_with_fake_distance(distance: float) -> VectorMemory:
    """创建不加载真实模型的 VectorMemory 测试实例。"""
    memory = VectorMemory.__new__(VectorMemory)
    memory.collection = FakeCollection(distance)
    memory.error = None
    return memory


def test_graphrag_question_does_not_recall_unrelated_comparison() -> None:
    """超过距离阈值的 RAG/Fine-tuning 历史不得被 GraphRAG 问题召回。"""
    memory = _memory_with_fake_distance(MEMORY_DISTANCE_THRESHOLD + 0.01)

    result = memory.search("GraphRAG解决了传统RAG的什么问题？")

    assert result == "未召回相关历史研究"


def test_rag_question_recalls_related_rag_memory() -> None:
    """阈值内的 RAG 历史应继续正常召回。"""
    memory = _memory_with_fake_distance(MEMORY_DISTANCE_THRESHOLD - 0.01)

    result = memory.search("RAG是什么？")

    assert "RAG和Fine-tuning哪个好" in result


def test_semantic_search() :
    """验证中英文语义相近问题能够召回已保存的 RAG 记忆。"""
    print("\n=====================\nTest 1 - 语义记忆检索\n=====================")
    if os.path.exists(TEST_MEMORY_DIR):
        shutil.rmtree(TEST_MEMORY_DIR, ignore_errors=True)

    memory = VectorMemory(persist_dir=TEST_MEMORY_DIR)
    saved = memory.save("什么是RAG", "RAG是检索增强生成技术")
    if saved == "Memory暂不可用":
        print(f"[FAIL] Memory暂不可用：{memory.error}")
        memory = None
        return False
    result = memory.search("什么是Retrieval Augmented Generation")
    print(f"Memory检索结果：\n{result}")
    passed = "RAG是检索增强生成技术" in result
    print("[PASS] 语义召回成功" if passed else "[FAIL] 未召回预期记忆")
    memory = None
    assert passed


def test_agent_multi_turn() :
    """执行两轮真实 Agent 调用，验证第二轮能获得第一轮的向量记忆。"""
    print("\n=====================\nTest 2 - Agent多轮记忆\n=====================")
    try:
        executor = create_research_agent()
        memory = (executor.metadata or {}).get("vector_memory")
        if memory is None or memory.search("状态检查") == "Memory暂不可用":
            print(f"[FAIL] Memory暂不可用：{getattr(memory, 'error', '未初始化')}")
            return False
        first_question = "帮我研究RAG最新发展趋势"
        first_result = executor.invoke({"input": first_question})
        first_answer = first_result.get("output", "")
        saved = memory.save(first_question, first_answer, {"test": "multi_turn"})
        print(f"第一轮回答：\n{first_answer}\nMemory保存：{saved}")
        second_question = (
            "RAG有哪些优化方法？"
            "请结合之前关于RAG基础原理的研究，"
            "分析当前RAG系统的问题以及对应优化方向。"
        )
        recalled = memory.search(second_question)
        print(f"\n第二轮Memory检索：\n{recalled}")
        if not recalled or recalled == "Memory暂不可用":
            print("[FAIL] 第二轮未检索到历史记忆")
            return False
        contextual_input = (
            f"当前问题：\n{second_question}\n\n相关历史研究记录：\n{recalled}\n\n"
            "请结合历史信息回答。"
        )
        second_result = executor.invoke({"input": contextual_input})
        second_answer = second_result.get("output", "")
        memory.save(second_question, second_answer, {"test": "multi_turn"})
        print(f"\n第二轮Agent回答：\n{second_answer}")
        print("[PASS] Agent已检索历史Memory并完成回答")
        return True
    except Exception as exc:
        print(f"[FAIL] Agent多轮测试异常：{type(exc).__name__}: {exc}")
        print("请确认 .env 中配置了可用的 LLM API，且网络可访问。")
        return False


def main() -> None:
    """运行全部 Memory 测试，并通过退出码表示测试结果。"""
    results = [test_semantic_search(), test_agent_multi_turn()]
    print(f"\n测试汇总：{sum(results)}/{len(results)} 通过")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
