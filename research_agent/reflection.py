"""Research Agent 的结构化 Self-Reflection 与无工具修订模块。"""

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import get_settings


REFLECTION_PROMPT = ChatPromptTemplate.from_template(
    """你是一名严格、客观的研究报告质量评估助手。

用户问题：
{question}

待评估回答：
{answer}

请评估准确性、完整性、清晰度和有用性。
只输出一个合法JSON对象，不要输出Markdown代码块或额外解释：
{{
  "score": 8,
  "needs_revision": true,
  "substantive": true,
  "issues": ["具体且实质性的问题"],
  "improvement": "可执行的改进建议"
}}

规则：
- score必须是0到10之间的整数。
- substantive issues包括：事实错误、核心问题遗漏、推理逻辑错误、结论不准确。存在其中任一问题时substantive为true。
- non-substantive issues包括：可以增加案例、可以补充细节、表达优化、格式优化。只有这些普通建议时substantive为false。
- score低于8或substantive为true时，needs_revision必须为true。
- score达到8且substantive为false时，needs_revision必须为false，即使存在普通补充建议。
- issues可以记录实质问题或普通建议，但不能仅因issues非空就要求修订。"""
)

REVISION_PROMPT = ChatPromptTemplate.from_template(
    """你是研究报告编辑器，不是新的研究Agent。

原始问题：
{question}

当前答案：
{answer}

Reflection反馈：
{feedback}

你只能修改当前答案中已有内容。
禁止新增未经现有答案支持的新事实、新数字、新统计、新案例或新来源。
禁止新增未经当前答案支持的成本阈值、百分比或调用量。
如果某个结论缺乏证据，请删除或改成谨慎的定性表达，而不是创造新数字。
请只根据Reflection反馈修改当前答案。
不要重新搜索。
不要调用工具。
不要改变原问题范围。
保留原答案中正确内容。
修复指出的问题。
只输出修改后的完整研究报告。"""
)


class ReflectionResult(BaseModel):
    """Reflection LLM 的结构化评价结果。"""

    score: int = Field(ge=0, le=10)
    needs_revision: bool
    substantive: bool = False
    issues: list[str]
    improvement: str


class ReflectionEvaluator:
    """使用普通 LLM 完成结构化评估和无工具答案修订。"""

    def __init__(self) -> None:
        """从 config.py 读取配置并初始化评估/修订模型。"""
        settings = get_settings()
        self.llm = ChatOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
            temperature=0,
            timeout=60,
            max_retries=2,
        )

    @staticmethod
    def _parse_json(content: str) -> ReflectionResult:
        """从模型文本中提取JSON并使用Pydantic严格校验。"""
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("Reflection响应中没有JSON对象")
        payload = json.loads(match.group(0))
        result = ReflectionResult.model_validate(payload)
        normalized_issues = [
            issue.strip()
            for issue in result.issues
            if issue.strip().lower() not in {"无", "没有", "none", "n/a"}
        ]
        result.issues = normalized_issues
        result.needs_revision = result.score < 8 or result.substantive
        return result

    def evaluate(self, question: str, answer: str) -> ReflectionResult:
        """评估真实研究初稿并返回结构化修订决策。"""
        if not question.strip() or not answer.strip():
            raise ValueError("Reflection评估的问题和答案不能为空")
        try:
            response = self.llm.invoke(
                REFLECTION_PROMPT.format_messages(question=question, answer=answer)
            )
            content = response.content if isinstance(response.content, str) else str(response.content)
            return self._parse_json(content)
        except Exception as exc:
            raise RuntimeError(f"Reflection评估失败：{type(exc).__name__}: {exc}") from exc

    def revise(
        self,
        question: str,
        answer: str,
        evaluation: ReflectionResult,
    ) -> str:
        """使用普通LLM按反馈编辑答案，不调用ReAct Agent或任何工具。"""
        feedback = (
            f"问题：{'；'.join(evaluation.issues) or '无'}\n"
            f"改进建议：{evaluation.improvement}"
        )
        try:
            response = self.llm.invoke(
                REVISION_PROMPT.format_messages(
                    question=question,
                    answer=answer,
                    feedback=feedback,
                )
            )
            revised = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
            if not revised:
                raise ValueError("Revision返回了空答案")
            number_pattern = r"\d+(?:\.\d+)?(?:万|亿)?(?:次|元|%|％)?"
            original_numbers = set(re.findall(number_pattern, answer))
            revised_numbers = set(re.findall(number_pattern, revised))
            if revised_numbers - original_numbers:
                return answer
            return revised
        except Exception as exc:
            raise RuntimeError(f"Revision生成失败：{type(exc).__name__}: {exc}") from exc
