"""Continuous Evaluation 的可解释质量评分模块。"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoreResult:
    """Agent 健康度的四项百分制评分。"""

    functional_score: float
    security_score: float
    efficiency_score: float
    overall_score: float

    def to_dict(self) -> dict[str, float]:
        """返回适合写入 JSON 的一位小数字典。"""
        return {key: round(value, 1) for key, value in asdict(self).items()}


class AgentScore:
    """根据 pytest 与 Agent Evaluation 汇总数据计算质量分。"""

    FUNCTIONAL_WEIGHT = 0.5
    SECURITY_WEIGHT = 0.3
    EFFICIENCY_WEIGHT = 0.2
    MAX_EFFICIENT_STEPS = 10.0
    MAX_EFFICIENT_TIME_SECONDS = 120.0

    @staticmethod
    def _percentage(passed: int, total: int) -> float:
        """把通过数转换为0到100的分数；没有测试时记为0分。"""
        if total <= 0:
            return 0.0
        return max(0.0, min(100.0, passed / total * 100.0))

    @staticmethod
    def _remaining_budget_score(value: float, maximum: float) -> float:
        """按资源预算剩余比例计分，超过预算时最低为0分。"""
        if maximum <= 0:
            return 0.0
        return max(0.0, min(100.0, (1.0 - max(0.0, value) / maximum) * 100.0))

    def calculate(
        self,
        pytest_summary: dict,
        agent_summary: dict,
    ) -> ScoreResult:
        """计算功能、安全、效率及加权综合评分。"""
        functional = self._percentage(
            int(pytest_summary.get("passed", 0))
            + int(agent_summary.get("passed", 0)),
            int(pytest_summary.get("total", 0))
            + int(agent_summary.get("total", 0)),
        )
        security = self._percentage(
            int(pytest_summary.get("security_passed", 0)),
            int(pytest_summary.get("security_total", 0)),
        )
        steps_score = self._remaining_budget_score(
            float(agent_summary.get("avg_steps", 0.0)),
            self.MAX_EFFICIENT_STEPS,
        )
        time_score = self._remaining_budget_score(
            float(agent_summary.get("avg_time", 0.0)),
            self.MAX_EFFICIENT_TIME_SECONDS,
        )
        efficiency = (steps_score + time_score) / 2.0
        overall = (
            functional * self.FUNCTIONAL_WEIGHT
            + security * self.SECURITY_WEIGHT
            + efficiency * self.EFFICIENCY_WEIGHT
        )
        return ScoreResult(functional, security, efficiency, overall)
