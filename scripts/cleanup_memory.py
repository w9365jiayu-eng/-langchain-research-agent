"""一次性清理 Chroma 长期记忆中的 Agent 失败记录。"""

from research_agent.memory import VectorMemory


def main() -> None:
    """扫描默认 Memory collection，仅删除包含失败标记的记录。"""
    memory = VectorMemory()
    if memory.collection is None:
        print(f"Memory暂不可用：{memory.error}")
        return

    stats = memory.cleanup_failed_memories()
    print(f"扫描记忆数量：{stats['scanned']}")
    print(f"删除失败记忆数量：{stats['deleted']}")
    print(f"剩余记忆数量：{stats['remaining']}")


if __name__ == "__main__":
    main()
