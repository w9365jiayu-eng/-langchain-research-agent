"""基于 ChromaDB 的持久化语义记忆模块。"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from .validation import contains_failure_marker, is_valid_answer

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except Exception as exc:
    chromadb = None
    SentenceTransformerEmbeddingFunction = None
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


MAX_MEMORY_CHARS_PER_ITEM = 2000
MAX_MEMORY_CONTEXT_CHARS = 4000
MEMORY_DISTANCE_THRESHOLD = 0.35


class VectorMemory:
    """使用 Chroma 和 SentenceTransformer 保存、检索长期语义记忆。"""

    def __init__(self, persist_dir: str = "./agent_memory") -> None:
        """初始化持久化 Chroma collection；失败时进入安全降级状态。"""
        self.persist_dir = Path(persist_dir)
        self.client = None
        self.collection = None
        self.error: str | None = None
        try:
            if _IMPORT_ERROR is not None:
                raise _IMPORT_ERROR
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-small-zh-v1.5"
            )
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name="agent_memory", embedding_function=embedding_function
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """把额外元数据转换为 Chroma 支持的标量类型。"""
        safe: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is not None:
                safe[str(key)] = value if isinstance(value, (str, int, float, bool)) else str(value)
        return safe

    @staticmethod
    def _truncate_document(document: str) -> str:
        """将单条记忆压缩到固定字符上限，并保留问题与回答前文。"""
        if len(document) <= MAX_MEMORY_CHARS_PER_ITEM:
            return document
        separator = "\n\n回答:\n"
        if separator in document:
            question_part, answer = document.split(separator, 1)
            question = question_part.removeprefix("问题:\n").strip()[:300]
            header = f"问题:\n{question}\n\n回答摘要/前2000字符:\n"
            available = max(0, MAX_MEMORY_CHARS_PER_ITEM - len(header))
            return header + answer[:available]
        return document[:MAX_MEMORY_CHARS_PER_ITEM]

    def save(self, question: str, answer: str, metadata: dict | None = None) -> str:
        """仅持久化有效的问题、回答和可选元数据，并返回记忆 ID。"""
        if self.collection is None:
            return "Memory暂不可用"
        if not question.strip() or not is_valid_answer(answer):
            return "Memory保存失败：问题或回答无效"
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            memory_id = f"mem_{time.time_ns()}"
            stored_metadata = self._safe_metadata(metadata or {})
            stored_metadata.update({"question": question, "timestamp": timestamp})
            self.collection.add(
                ids=[memory_id],
                documents=[f"问题:\n{question}\n\n回答:\n{answer}"],
                metadatas=[stored_metadata],
            )
            return memory_id
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return "Memory暂不可用"

    def search(self, query: str, top_k: int = 2) -> str:
        """检索、过滤、去重并截断相关记忆，总长度不超过4000字符。"""
        if self.collection is None:
            return "Memory暂不可用"
        if not query.strip() or top_k <= 0:
            return ""
        try:
            count = self.collection.count()
            if count == 0:
                return ""
            fetch_count = count
            result = self.collection.query(
                query_texts=[query],
                n_results=fetch_count,
                include=["documents", "metadatas", "distances"],
            )
            nested_documents = result.get("documents") or []
            candidates = nested_documents[0] if nested_documents else []
            nested_metadatas = result.get("metadatas") or []
            candidate_metadatas = nested_metadatas[0] if nested_metadatas else []
            nested_distances = result.get("distances") or []
            candidate_distances = nested_distances[0] if nested_distances else []
            latest_by_question: dict[str, tuple[str, int, str, float]] = {}
            for index, document in enumerate(candidates):
                if not document or contains_failure_marker(document):
                    continue
                if index >= len(candidate_distances):
                    continue
                distance = candidate_distances[index]
                if distance is None:
                    continue
                metadata = candidate_metadatas[index] if index < len(candidate_metadatas) else {}
                metadata = metadata or {}
                question_key = str(metadata.get("question", "")).strip()
                if not question_key:
                    question_key = document.split("\n\n回答:\n", 1)[0].removeprefix("问题:\n").strip()
                timestamp = str(metadata.get("timestamp", ""))
                current = latest_by_question.get(question_key)
                if current is None or timestamp > current[0]:
                    latest_by_question[question_key] = (
                        timestamp,
                        index,
                        document,
                        float(distance),
                    )

            deduplicated_candidates = [
                item[2]
                for item in sorted(latest_by_question.values(), key=lambda item: item[1])
                if item[3] <= MEMORY_DISTANCE_THRESHOLD
            ]
            unique_documents: list[str] = []
            seen: set[str] = set()
            for document in deduplicated_candidates:
                if not document or document in seen or contains_failure_marker(document):
                    continue
                seen.add(document)
                unique_documents.append(self._truncate_document(document))
                if len(unique_documents) >= top_k:
                    break

            if not unique_documents:
                return "未召回相关历史研究"

            context = ""
            separator = "\n\n--- 相关记忆 ---\n\n"
            for document in unique_documents:
                addition = document if not context else separator + document
                remaining = MAX_MEMORY_CONTEXT_CHARS - len(context)
                if remaining <= 0:
                    break
                context += addition[:remaining]
            return context
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return "Memory暂不可用"

    def cleanup_failed_memories(self) -> dict[str, int]:
        """删除包含失败标记的历史记录，并返回扫描、删除和剩余数量。"""
        if self.collection is None:
            return {"scanned": 0, "deleted": 0, "remaining": 0}
        try:
            result = self.collection.get(include=["documents"])
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            invalid_ids = [
                memory_id
                for memory_id, document in zip(ids, documents)
                if document and contains_failure_marker(document)
            ]
            if invalid_ids:
                self.collection.delete(ids=invalid_ids)
            return {
                "scanned": len(ids),
                "deleted": len(invalid_ids),
                "remaining": self.collection.count(),
            }
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return {"scanned": 0, "deleted": 0, "remaining": 0}

    def get_stats(self) -> dict[str, int]:
        """返回当前 collection 中保存的记忆数量。"""
        if self.collection is None:
            return {"total_memories": 0}
        try:
            return {"total_memories": self.collection.count()}
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return {"total_memories": 0}
