"""应用配置：从环境变量读取 OpenAI-compatible 模型参数。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """研究助理运行所需的模型配置。"""

    api_key: str
    base_url: str
    model: str


def get_settings() -> Settings:
    """读取并校验环境变量，缺少配置时给出友好提示。"""
    values = {
        "LLM_API_KEY": os.getenv("LLM_API_KEY", "").strip(),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "").strip(),
        "LLM_MODEL": os.getenv("LLM_MODEL", "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            f"缺少环境变量：{', '.join(missing)}。请复制 .env.example 为 .env 并填写配置。"
        )

    return Settings(
        api_key=values["LLM_API_KEY"],
        base_url=values["LLM_BASE_URL"].rstrip("/"),
        model=values["LLM_MODEL"],
    )
