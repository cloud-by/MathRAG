from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.services.llm_service import chat_json


PLANNER_SYSTEM_PROMPT = """你是一个负责 RAG 检索规划的智能体。
你的任务是把用户问题转成更容易命中知识库的检索子问题。

要求：
1. 输出必须是一个 json 对象。
2. retrieval_queries 必须是字符串数组，1~4 条。
3. 每条检索子问题要尽量包含核心概念、公式名、关键条件或同义表达。
4. strategy 用一句话解释检索策略。
5. 不要输出 markdown，不要输出代码块。

输出格式：
{
  "strategy": "...",
  "retrieval_queries": ["...", "..."]
}
"""


@dataclass
class RetrievalPlan:
    strategy: str
    retrieval_queries: List[str]


class QueryPlanner:
    """使用 LLM 将原问题拆分为多查询检索计划。"""

    def create_plan(
        self,
        question: str,
        history: Sequence[Dict[str, Any]] | None = None,
    ) -> RetrievalPlan:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question 不能为空")

        history_text = self._format_history(history)
        user_prompt = (
            "请为这个问题生成检索计划。\n\n"
            f"【用户问题】\n{question}\n\n"
            f"【最近对话】\n{history_text}\n"
        )

        try:
            response = chat_json(
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            data = response.data if isinstance(response.data, dict) else {}
            strategy = str(data.get("strategy", "")).strip() or "直接按原问题检索"
            queries = self._normalize_queries(data.get("retrieval_queries"), fallback=question)
            return RetrievalPlan(strategy=strategy, retrieval_queries=queries)
        except Exception:
            # 规划失败时回退到单查询，保证主流程可用。
            return RetrievalPlan(strategy="规划失败，回退为原问题检索", retrieval_queries=[question])

    @staticmethod
    def _format_history(history: Sequence[Dict[str, Any]] | None) -> str:
        if not history:
            return "无"

        lines: List[str] = []
        for item in history[-4:]:
            role = str(item.get("role", "")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")

        return "\n".join(lines) if lines else "无"

    @staticmethod
    def _normalize_queries(value: Any, fallback: str) -> List[str]:
        if isinstance(value, list):
            output: List[str] = []
            seen = set()
            for item in value:
                text = str(item).strip()
                if not text or text in seen:
                    continue
                output.append(text)
                seen.add(text)
                if len(output) >= 4:
                    break
            if output:
                return output

        text = str(value or "").strip()
        if text:
            return [text]

        return [fallback]


_query_planner: QueryPlanner | None = None


def get_query_planner() -> QueryPlanner:
    global _query_planner
    if _query_planner is None:
        _query_planner = QueryPlanner()
    return _query_planner