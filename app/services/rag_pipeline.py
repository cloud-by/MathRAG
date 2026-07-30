from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.core.config import settings
from app.modules.knowledge.search_service import (
    KnowledgeSearchService,
    build_knowledge_search_service,
)
from app.services.agentic_rag import QueryPlanner, get_query_planner
from app.services.llm_service import chat_json
from app.utils.prompt_builder import build_chat_messages


@dataclass
class RAGConfig:
    default_top_k: int = settings.TOP_K


class RAGPipeline:
    def __init__(
        self,
        *,
        knowledge_search: KnowledgeSearchService,
        planner: QueryPlanner | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._knowledge_search = knowledge_search
        self._planner = planner or get_query_planner()
        self.config = config or RAGConfig()

    async def chat(
        self,
        question: str,
        history: Sequence[Dict[str, Any]] | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question 不能为空")

        k = self.config.default_top_k if top_k is None else top_k
        if type(k) is not int or not 1 <= k <= 10:
            raise ValueError("top_k 必须是 1 到 10 的整数")

        plan = await asyncio.to_thread(
            self._planner.create_plan,
            question=question,
            history=history,
        )
        queries = self._normalize_queries(question, plan.retrieval_queries)
        hits = await self._knowledge_search.search(queries, top_k=k)
        references = [
            hit.to_reference(rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]

        messages = build_chat_messages(
            question=question,
            references=references,
            history=history,
        )
        llm_result = await asyncio.to_thread(chat_json, messages=messages)
        parsed = self._normalize_result(llm_result.data, references, question)

        result: Dict[str, Any] = {
            "question": question,
            "answer": parsed["answer"],
            "steps": parsed["steps"],
            "used_knowledge": parsed["used_knowledge"],
            "related_questions": parsed["related_questions"],
            "references": references,
            "agentic_plan": {
                "strategy": plan.strategy,
                "retrieval_queries": plan.retrieval_queries,
            },
        }
        return result

    @staticmethod
    def _normalize_queries(
        question: str,
        retrieval_queries: Sequence[str],
    ) -> List[str]:
        queries = [question]
        seen = {question}
        for query in retrieval_queries:
            text = " ".join(str(query).split()).strip()
            if text and text not in seen:
                queries.append(text)
                seen.add(text)
            if len(queries) >= 4:
                break
        return queries

    @staticmethod
    def _normalize_str_list(value: Any, default: List[str] | None = None) -> List[str]:
        if default is None:
            default = []

        if value is None:
            return default

        if isinstance(value, list):
            output: List[str] = []
            seen = set()
            for item in value:
                text = str(item).strip()
                if not text:
                    continue
                if text not in seen:
                    output.append(text)
                    seen.add(text)
            return output or default

        text = str(value).strip()
        return [text] if text else default

    @staticmethod
    def _normalize_reference_item(ref: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rank": int(ref.get("rank", 0) or 0),
            "score": float(ref.get("score", 0.0) or 0.0),
            "index": ref.get("index"),
            "chunk_id": str(ref.get("chunk_id", "")).strip(),
            "source_id": str(ref.get("source_id", "")).strip(),
            "category": str(ref.get("category", "")).strip(),
            "title": str(ref.get("title", "")).strip(),
            "keywords": RAGPipeline._normalize_str_list(ref.get("keywords", [])),
            "content": str(ref.get("content", "")).strip(),
            "example": str(ref.get("example", "")).strip(),
            "steps": RAGPipeline._normalize_str_list(ref.get("steps", [])),
            "difficulty": str(ref.get("difficulty", "")).strip(),
            "answer_context": str(ref.get("answer_context", "")).strip(),
            "retrieval_text": str(ref.get("retrieval_text", "")).strip(),
            "source_line": ref.get("source_line"),
            "metadata": ref.get("metadata", {}) if isinstance(ref.get("metadata", {}), dict) else {},
        }

    def _normalize_references(self, references: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for ref in references:
            if not isinstance(ref, dict):
                continue
            normalized.append(self._normalize_reference_item(ref))
        return normalized

    @staticmethod
    def _build_fallback_steps(references: Sequence[Dict[str, Any]]) -> List[str]:
        if not references:
            return [
                "当前没有检索到足够的参考知识。",
                "请补充更具体的问题条件，或扩大知识库后再尝试回答。",
            ]

        first = references[0]
        title = str(first.get("title", "")).strip()

        steps: List[str] = []
        if title:
            steps.append(f"先结合检索到的知识点《{title}》理解当前问题。")
        else:
            steps.append("先结合检索到的参考知识理解当前问题。")

        steps.append("优先参考最相关的知识内容、例子和步骤说明。")

        return steps[:2]

    @staticmethod
    def _normalize_used_knowledge(
        used_knowledge: List[str],
        references: Sequence[Dict[str, Any]],
    ) -> List[str]:
        reference_titles = [
            str(ref.get("title", "")).strip()
            for ref in references
            if str(ref.get("title", "")).strip()
        ]

        if not reference_titles:
            return used_knowledge

        title_set = set(reference_titles)
        filtered: List[str] = []
        seen = set()

        for item in used_knowledge:
            text = str(item).strip()
            if not text:
                continue

            if text in title_set and text not in seen:
                filtered.append(text)
                seen.add(text)

        if filtered:
            return filtered

        return reference_titles[: min(2, len(reference_titles))]

    @staticmethod
    def _normalize_related_questions(
        related_questions: List[str],
        question: str,
    ) -> List[str]:
        cleaned: List[str] = []
        seen = set()

        for item in related_questions:
            text = str(item).strip()
            if not text:
                continue
            if text not in seen:
                cleaned.append(text)
                seen.add(text)

        if cleaned:
            return cleaned[:2]

        question = question.strip()
        return [
            f"{question} 的关键知识点是什么？",
            f"{question} 还有没有其它解法或理解方式？",
        ]

    def _normalize_result(
        self,
        data: Dict[str, Any],
        references: Sequence[Dict[str, Any]],
        question: str,
    ) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}

        answer = str(data.get("answer", "")).strip()
        if not answer:
            answer = "参考知识不足以生成稳定回答。"

        steps = self._normalize_str_list(data.get("steps"))
        if not steps:
            steps = self._build_fallback_steps(references)

        used_knowledge = self._normalize_str_list(data.get("used_knowledge"))
        used_knowledge = self._normalize_used_knowledge(used_knowledge, references)

        related_questions = self._normalize_str_list(data.get("related_questions"))
        related_questions = self._normalize_related_questions(related_questions, question)

        return {
            "answer": answer,
            "steps": steps[:6],
            "used_knowledge": used_knowledge[:2],
            "related_questions": related_questions[:2],
        }


_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            knowledge_search=build_knowledge_search_service(),
        )
    return _rag_pipeline


async def chat_with_rag(
    question: str,
    history: Sequence[Dict[str, Any]] | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    return await get_rag_pipeline().chat(
        question=question,
        history=history,
        top_k=top_k,
    )
