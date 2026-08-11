"""Memory、Planning、Research和Reflection集成质量门控测试。"""

from types import SimpleNamespace

from app import _reflect_and_revise, process_question, save_valid_answer
from research_agent.memory import MAX_MEMORY_CONTEXT_CHARS, VectorMemory
from research_agent.planner import ResearchPlanner, classify_question_complexity
from research_agent.reflection import ReflectionResult


class FakeCollection:
    """用于隔离测试VectorMemory查询逻辑的最小collection。"""

    def __init__(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]

    def count(self) -> int:
        return len(self.documents)

    def query(self, **kwargs):
        return {
            "documents": [self.documents],
            "metadatas": [self.metadatas],
            "distances": [[0.1 for _ in self.documents]],
        }


class FakeMemory:
    """记录保存调用的Memory替身。"""

    def __init__(self) -> None:
        self.saved: list[tuple] = []

    def search(self, query: str) -> str:
        return ""

    def save(self, question: str, answer: str, metadata=None) -> str:
        self.saved.append((question, answer, metadata))
        return "mem_test"


class FakePlanner:
    """返回固定短计划。"""

    def generate_plan(self, question: str, memory_context: str = "") -> str:
        return "1. 回答核心比较\n2. 总结适用条件"


def _memory_with_documents(
    documents: list[str], metadatas: list[dict] | None = None
) -> VectorMemory:
    memory = object.__new__(VectorMemory)
    memory.collection = FakeCollection(documents, metadatas)
    memory.error = None
    return memory


def test_failed_answer_cannot_be_saved() -> None:
    """iteration/time limit失败字符串不得写入Memory。"""
    memory = FakeMemory()
    assert not save_valid_answer(
        memory, "问题", "Agent stopped due to iteration limit or time limit."
    )
    assert memory.saved == []


def test_valid_final_answer_is_saved() -> None:
    """正常最终答案应保存一次。"""
    memory = FakeMemory()
    assert save_valid_answer(memory, "问题", "这是完整且有效的研究结论。")
    assert memory.saved[0][1] == "这是完整且有效的研究结论。"


def test_memory_search_filters_failed_records() -> None:
    """查询层必须过滤尚未清理的失败记录。"""
    memory = _memory_with_documents([
        "问题:\n失败\n\n回答:\nAgent stopped due to iteration limit or time limit.",
        "问题:\n正常\n\n回答:\n有效研究结果",
    ])
    result = memory.search("RAG")
    assert "Agent stopped" not in result
    assert "有效研究结果" in result


def test_memory_search_deduplicates_and_limits_context() -> None:
    """完全重复文档只返回一次，整体上下文不超过限制。"""
    document = "问题:\nRAG\n\n回答:\n" + "有效内容" * 1000
    memory = _memory_with_documents([document, document, "问题:\n其他\n\n回答:\n补充"])
    result = memory.search("RAG")
    assert result.count("问题:\nRAG") == 1
    assert len(result) <= MAX_MEMORY_CONTEXT_CHARS


def test_simple_plan_has_at_most_three_steps() -> None:
    """简单二选一计划不得超过3步，且每步必须包含真实文本。"""
    steps = [
        "对比RAG与Fine-tuning的核心机制",
        "分析两者各自适用场景",
        "给出选型建议",
        "不应保留的额外步骤",
    ]
    limited = ResearchPlanner._format_plan("RAG和Fine-tuning哪个好？", steps)
    assert classify_question_complexity("RAG和Fine-tuning哪个好？") == "simple"
    assert len(limited.splitlines()) <= 3
    assert all(line.split(". ", 1)[1].strip() not in {"", "--"} for line in limited.splitlines())
    assert "核心机制" in limited and "适用场景" in limited


def test_complex_plan_has_at_most_five_steps() -> None:
    """复杂多场景问题的计划不得超过5步。"""
    steps = [f"复杂研究步骤{i}" for i in range(1, 8)]
    question = "请对比RAG和Fine-tuning在医疗、法律、客服三个场景的优劣，并分析成本"
    limited = ResearchPlanner._format_plan(question, steps)
    assert classify_question_complexity(question) == "complex"
    assert 3 <= len(limited.splitlines()) <= 5


def test_planner_json_preserves_complete_step_text() -> None:
    """结构化JSON计划应保留正文，不得退化为横线占位符。"""
    planner = object.__new__(ResearchPlanner)
    planner.llm = SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(
            content=(
                '{"complexity":"simple","plan":['
                '"对比RAG与Fine-tuning的核心机制",'
                '"分析两者各自适用场景",'
                '"给出选型建议"]}'
            )
        )
    )
    plan = planner.generate_plan("RAG和Fine-tuning哪个好？")
    assert len(plan.splitlines()) <= 3
    assert "--" not in plan
    assert "核心机制" in plan and "适用场景" in plan and "选型建议" in plan


def test_memory_keeps_latest_answer_for_same_question() -> None:
    """同一问题存在多个版本时只召回timestamp最新的一条。"""
    documents = [
        "问题:\nRAG和Fine-tuning哪个好？\n\n回答:\n旧答案",
        "问题:\nRAG和Fine-tuning哪个好？\n\n回答:\n新答案",
        "问题:\n其他问题\n\n回答:\n其他内容",
    ]
    metadatas = [
        {"question": "RAG和Fine-tuning哪个好？", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"question": "RAG和Fine-tuning哪个好？", "timestamp": "2026-02-01T00:00:00+00:00"},
        {"question": "其他问题", "timestamp": "2026-01-15T00:00:00+00:00"},
    ]
    result = _memory_with_documents(documents, metadatas).search("RAG Fine-tuning")
    assert "新答案" in result
    assert "旧答案" not in result


def test_failed_research_skips_reflection_and_memory() -> None:
    """Research初次及受控重试均失败时，不得进入Reflection或保存。"""
    executor = SimpleNamespace(
        invoke=lambda payload: {"output": "Agent stopped due to iteration limit or time limit."}
    )
    memory = FakeMemory()
    reflection = SimpleNamespace(evaluate=lambda *args: (_ for _ in ()).throw(AssertionError()))
    result = process_question(executor, memory, FakePlanner(), reflection, "研究RAG")
    assert result is None
    assert memory.saved == []


def test_reflection_receives_exact_research_answer() -> None:
    """Reflection收到的必须严格等于本轮executor输出，不能是Memory内容。"""
    executor = SimpleNamespace(invoke=lambda payload: {"output": "这是Research初稿"})
    memory = FakeMemory()

    class CapturingReflection:
        received_answer = None

        def evaluate(self, question, answer):
            self.received_answer = answer
            return ReflectionResult(
                score=9, needs_revision=False, issues=[], improvement="无"
            )

    reflection = CapturingReflection()
    result = process_question(executor, memory, FakePlanner(), reflection, "研究问题")
    assert reflection.received_answer == "这是Research初稿"
    assert result == "这是Research初稿"


def test_revision_answer_is_the_only_saved_answer() -> None:
    """低分初稿经普通LLM修订并通过后，只保存修订版本。"""
    executor = SimpleNamespace(invoke=lambda payload: {"output": "初稿：比较不够完整。"})
    memory = FakeMemory()

    class FakeReflection:
        evaluations = 0

        def evaluate(self, question, answer):
            self.evaluations += 1
            if self.evaluations == 1:
                return ReflectionResult(
                    score=8,
                    needs_revision=True,
                    issues=["缺少适用场景"],
                    improvement="补充适用条件",
                )
            return ReflectionResult(
                score=9, needs_revision=False, issues=[], improvement="无"
            )

        def revise(self, question, answer, evaluation):
            return "修订后答案：已补充RAG与Fine-tuning的适用场景。"

    result = process_question(
        executor, memory, FakePlanner(), FakeReflection(), "RAG和Fine-tuning哪个好？"
    )
    assert result.startswith("修订后答案")
    assert len(memory.saved) == 1
    assert memory.saved[0][1] == result
    assert memory.saved[0][1] != "初稿：比较不够完整。"


def test_revision_rejects_new_unsupported_numbers() -> None:
    """无工具Revision不得加入原答案中不存在的调用量、阈值或百分比。"""
    from research_agent.reflection import ReflectionEvaluator

    evaluator = object.__new__(ReflectionEvaluator)
    evaluator.llm = SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(
            content=(
                "RAG推理成本可能更高。月调用量低于10万次时RAG更优，"
                "超过100万次时Fine-tuning反超，预计节省20%。"
            )
        )
    )
    original = "RAG推理成本可能更高，但没有具体数据。"
    evaluation = ReflectionResult(
        score=6,
        needs_revision=True,
        issues=["成本结论缺少依据"],
        improvement="删除无依据的阈值",
    )
    revised = evaluator.revise("RAG和Fine-tuning哪个好？", original, evaluation)
    assert revised == original
    assert "10万次" not in revised
    assert "100万次" not in revised
    assert "20%" not in revised


def test_reflection_stops_after_second_evaluation() -> None:
    """第二轮仍需修改时应停止，不得进行第二次Revision。"""
    class AlwaysNeedsRevision:
        evaluations = 0
        revisions = 0

        def evaluate(self, question, answer):
            self.evaluations += 1
            return ReflectionResult(
                score=8,
                needs_revision=True,
                issues=["仍有实质问题"],
                improvement="继续优化",
            )

        def revise(self, question, answer, evaluation):
            self.revisions += 1
            return answer + " 已进行一次修订。"

    reflection = AlwaysNeedsRevision()
    result = _reflect_and_revise(reflection, "问题", "有效初稿。")
    assert result.endswith("已进行一次修订。")
    assert reflection.evaluations == 2
    assert reflection.revisions == 1
