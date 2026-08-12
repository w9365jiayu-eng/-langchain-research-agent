"""Agent Health Dashboard、baseline 对比与 JSON 报告输出。"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


DEGRADATION_TOLERANCE = 5.0


def _bar(score: float) -> str:
    """生成固定10格的百分制文本进度条。"""
    filled = max(0, min(10, round(score / 10)))
    unicode_bar = "█" * filled + "░" * (10 - filled)
    try:
        unicode_bar.encode(sys.stdout.encoding or "utf-8")
        return unicode_bar
    except UnicodeEncodeError:
        return "#" * filled + "-" * (10 - filled)


def compare_with_baseline(current: dict, baseline: dict) -> dict:
    """当前综合分低于 baseline 超过5分时判定为退化。"""
    previous = float(baseline.get("overall_score", 0.0))
    current_overall = float(current.get("overall_score", 0.0))
    degraded = current_overall < previous - DEGRADATION_TOLERANCE
    return {
        "previous": round(previous, 1),
        "current": round(current_overall, 1),
        "tolerance": DEGRADATION_TOLERANCE,
        "degradation_detected": degraded,
        "message": (
            "Performance degradation detected"
            if degraded
            else "No performance degradation"
        ),
    }


def print_dashboard(scores: dict, comparison: dict) -> None:
    """向终端打印简洁的 Agent 健康面板。"""
    unicode_status = (
        "⚠️ Regression Detected"
        if comparison["degradation_detected"]
        else "✅ Healthy"
    )
    try:
        unicode_status.encode(sys.stdout.encoding or "utf-8")
        status = unicode_status
    except UnicodeEncodeError:
        status = "[REGRESSION] Regression Detected" if comparison["degradation_detected"] else "[OK] Healthy"
    print("\n" + "=" * 48)
    print("Agent Health Dashboard")
    print("=" * 48)
    print(f"Functional Test: {_bar(scores['functional_score'])} {scores['functional_score']:.1f}%")
    print(f"Security:        {_bar(scores['security_score'])} {scores['security_score']:.1f}%")
    print(f"Efficiency:      {_bar(scores['efficiency_score'])} {scores['efficiency_score']:.1f}%")
    print(f"Overall Score:   {scores['overall_score']:.1f}/100")
    print(f"Previous:        {comparison['previous']:.1f}")
    print(f"Current:         {comparison['current']:.1f}")
    print(f"Status:          {status}")
    print(comparison["message"])
    print("=" * 48)


def write_health_report(
    report_path: Path,
    pytest_summary: dict,
    agent_evaluation: dict,
    scores: dict,
    comparison: dict,
) -> dict:
    """把完整持续评估结果保存为 evaluation_results.json。"""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pytest": pytest_summary,
        "agent_evaluation": agent_evaluation,
        "scores": scores,
        "baseline_comparison": comparison,
        "status": "regression" if comparison["degradation_detected"] else "healthy",
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def update_baseline(
    baseline_path: Path,
    scores: dict,
    agent_summary: dict,
) -> None:
    """在未退化时用当前健康结果更新 baseline。"""
    baseline = {
        **scores,
        "avg_steps": round(float(agent_summary.get("avg_steps", 0.0)), 2),
        "avg_time": round(float(agent_summary.get("avg_time", 0.0)), 2),
    }
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
