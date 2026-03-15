from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.services.llm_service import chat_json
from app.services.retriever import retrieve
from app.utils.prompt_builder import build_chat_messages


@dataclass
class RAGConfig:
    default_top_k: int = 3


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()

    def chat(
        self,
        question: str,
        history: Sequence[Dict[str, Any]] | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question 不能为空")

        k = top_k or self.config.default_top_k
        references = retrieve(question=question, top_k=k)
        messages = build_chat_messages(question=question, references=references, history=history)
        llm_result = chat_json(messages=messages)
        parsed = self._normalize_result(llm_result.data)

        result: Dict[str, Any] = {
            "question": question,
            "answer": parsed["answer"],
            "steps": parsed["steps"],
            "used_knowledge": parsed["used_knowledge"],
            "related_questions": parsed["related_questions"],
            "references": references,
        }

        if llm_result.reasoning_content:
            result["reasoning_content"] = llm_result.reasoning_content

        return result

    @staticmethod
    def _normalize_str_list(value: Any, default: List[str] | None = None) -> List[str]:
        if default is None:
            default = []
        if value is None:
            return default
        if isinstance(value, list):
            output: List[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    output.append(text)
            return output or default
        text = str(value).strip()
        return [text] if text else default

    def _normalize_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        answer = str(data.get("answer", "")).strip()
        if not answer:
            answer = "参考知识不足以生成稳定回答。"

        steps = self._normalize_str_list(data.get("steps"))
        used_knowledge = self._normalize_str_list(data.get("used_knowledge"))
        related_questions = self._normalize_str_list(data.get("related_questions"))

        return {
            "answer": answer,
            "steps": steps,
            "used_knowledge": used_knowledge,
            "related_questions": related_questions[:2],
        }


_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def chat_with_rag(
    question: str,
    history: Sequence[Dict[str, Any]] | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    return get_rag_pipeline().chat(question=question, history=history, top_k=top_k)
