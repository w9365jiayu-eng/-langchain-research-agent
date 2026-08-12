"""统一执行 pytest、Agent Evaluation、评分、报告和 baseline 对比。"""

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.report import (  # noqa: E402
    compare_with_baseline,
    print_dashboard,
    update_baseline,
    write_health_report,
)
from evaluation.scorer import AgentScore  # noqa: E402


EVALUATION_DIR = PROJECT_ROOT / "evaluation"
PYTEST_REPORT = EVALUATION_DIR / "pytest_results.xml"
BASELINE_PATH = EVALUATION_DIR / "baseline.json"
RESULT_PATH = PROJECT_ROOT / "evaluation_results.json"


def _run(command: list[str]) -> int:
    """在项目根目录执行子进程并把实时输出转发到当前终端。"""
    print(f"\n$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def parse_pytest_report(report_path: Path) -> dict:
    """从 pytest JUnit XML 统计整体及 security 相关测试通过率。"""
    if not report_path.is_file():
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "pass_rate": 0.0,
            "security_total": 0,
            "security_passed": 0,
            "security_pass_rate": 0.0,
        }

    root = ET.parse(report_path).getroot()
    testcases = root.findall(".//testcase")
    failed = sum(case.find("failure") is not None for case in testcases)
    errors = sum(case.find("error") is not None for case in testcases)
    skipped = sum(case.find("skipped") is not None for case in testcases)
    total = len(testcases)
    passed = max(0, total - failed - errors - skipped)

    security_cases = []
    for case in testcases:
        identity = " ".join(
            str(case.attrib.get(key, "")) for key in ("classname", "name", "file")
        ).lower()
        if "security" in identity:
            security_cases.append(case)
    security_failed = sum(
        case.find("failure") is not None
        or case.find("error") is not None
        or case.find("skipped") is not None
        for case in security_cases
    )
    security_total = len(security_cases)
    security_passed = security_total - security_failed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "pass_rate": passed / total if total else 0.0,
        "security_total": security_total,
        "security_passed": security_passed,
        "security_pass_rate": security_passed / security_total if security_total else 0.0,
    }


def _load_json(path: Path, fallback: dict) -> dict:
    """读取 JSON；缺失或格式错误时使用明确的降级值。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def main() -> None:
    """运行完整 Continuous Evaluation，并返回适合 CI 的退出码。"""
    pytest_code = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--junitxml",
            str(PYTEST_REPORT),
        ]
    )
    pytest_summary = parse_pytest_report(PYTEST_REPORT)

    if RESULT_PATH.exists():
        RESULT_PATH.unlink()
    agent_code = _run([sys.executable, "-m", "evaluation.eval_agent"])
    agent_evaluation = _load_json(
        RESULT_PATH,
        {
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 1,
                "pass_rate": 0.0,
                "avg_time": 120.0,
                "avg_steps": 10.0,
            },
            "results": [],
        },
    )
    agent_summary = agent_evaluation.get("summary", {})
    scores = AgentScore().calculate(pytest_summary, agent_summary).to_dict()
    baseline = _load_json(BASELINE_PATH, {"overall_score": 0.0})
    comparison = compare_with_baseline(scores, baseline)

    write_health_report(
        RESULT_PATH,
        pytest_summary,
        agent_evaluation,
        scores,
        comparison,
    )
    print_dashboard(scores, comparison)
    print(f"评估报告：{RESULT_PATH}")

    evaluation_succeeded = pytest_code == 0 and agent_code == 0
    if evaluation_succeeded and not comparison["degradation_detected"]:
        update_baseline(BASELINE_PATH, scores, agent_summary)
        print(f"Baseline已更新：{BASELINE_PATH}")
    elif not evaluation_succeeded:
        print("测试或Agent Evaluation未全部通过，Baseline保持不变。")
    else:
        print("检测到性能退化，Baseline保持不变。")

    failed = not evaluation_succeeded or comparison["degradation_detected"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
