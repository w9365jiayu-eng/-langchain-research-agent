"""Research Agent 的结构化任务规划模块。"""

import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, field_validator

from .config import get_settings


PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """你是一名专业研究规划助手。

研究目标：
{question}

代码判定的问题复杂度：
{complexity}

已有研究记忆：
{memory_context}

请生成与问题复杂度匹配的计划：
- simple：1-3个必要步骤。
- complex：3-5个步骤，最多5步。

simple问题禁止主动扩展ROI模型、Excel工具、大量行业场景、大型决策手册或额外系统实现，除非用户明确要求。

只输出合法JSON，不要输出Markdown或额外说明：
{{
  "complexity": "{complexity}",
  "plan": ["完整步骤文本", "完整步骤文本"]
}}

plan中的每一项必须是清晰、非空、可执行的完整研究步骤，不得使用“-”或“--”作为占位符。"""
)


class PlanResult(BaseModel):
    """Planner LLM 返回的结构化计划。"""

    complexity: Literal["simple", "complex"]
    plan: list[str]

    @field_validator("plan")
    @classmethod
    def validate_plan_steps(cls, steps: list[str]) -> list[str]:
        """过滤空白和横线占位符，并要求至少存在一个真实步骤。"""
        cleaned: list[str] = []
        for step in steps:
            text = step.strip()
            if text and text not in {"-", "--", "—", "——"}:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("计划不包含有效步骤文本")
        return cleaned


def classify_question_complexity(question: str) -> str:
    """根据用户明确要求的场景、维度和交付范围区分简单或复杂问题。"""
    normalized = question.strip()
    complex_markers = (
        "医疗", "法律", "客服", "三个场景", "多个场景", "多个公开案例",
        "企业落地建议", "部署风险", "成本与", "多行业", "多维度",
        "系统性研究", "深入研究",
    )
    return "complex" if any(marker in normalized for marker in complex_markers) else "simple"


class ResearchPlanner:
    """使用现有OpenAI-compatible LLM生成Pydantic校验后的适度计划。"""

    def __init__(self) -> None:
        """读取现有模型配置并初始化规划模型。"""
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
    def _extract_result(content: str) -> PlanResult:
        """提取JSON并通过Pydantic校验，不再解析Markdown编号。"""
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Planner响应中没有JSON对象")
        return PlanResult.model_validate(json.loads(content[start : end + 1]))

    @staticmethod
    def _format_plan(question: str, steps: list[str]) -> str:
        """按代码判定的复杂度限制步骤数，并格式化完整步骤文本。"""
        complexity = classify_question_complexity(question)
        max_steps = 5 if complexity == "complex" else 3
        limited = [step.strip() for step in steps if step.strip() and step.strip() != "--"][:max_steps]
        if complexity == "complex":
            fallbacks = ("核对关键比较维度", "综合证据形成结论", "总结成本、风险与建议")
            for fallback in fallbacks:
                if len(limited) >= 3:
                    break
                limited.append(fallback)
        if not limited:
            limited = ["直接分析并回答用户的核心问题"]
        return "\n".join(f"{index}. {step}" for index, step in enumerate(limited, start=1))

    def generate_plan(self, question: str, memory_context: str = "") -> str:
        """生成simple 1-3步或complex 3-5步的结构化研究计划。"""
        if not question.strip():
            raise ValueError("研究问题不能为空")
        complexity = classify_question_complexity(question)
        try:
            response = self.llm.invoke(
                PLANNER_PROMPT.format_messages(
                    question=question,
                    complexity=complexity,
                    memory_context=memory_context or "暂无相关历史研究记忆",
                )
            )
            content = response.content if isinstance(response.content, str) else str(response.content)
            result = self._extract_result(content)
            return self._format_plan(question, result.plan)
        except Exception as exc:
            raise RuntimeError(f"Planner生成失败：{type(exc).__name__}: {exc}") from exc
