from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.llm import ChatMessage, LlmClient
from shared.tools import get_order, get_ticket

from question1.agent import (
    ACTIONS,
    DEFAULT_ARTICLE,
    SYSTEM_PROMPT,
    Triage,
    build_user_message,
    parse_resolution,
    policy_index,
    triage,
)


@dataclass
class TicketResolution:
    action: str
    kb_article: str


class SupportPipeline:
    """
    Same product as Question 1: triage -> retrieve -> compose -> audit.

    Each node uses the cheapest mechanism that can do its job:
      triage   code    structured order facts + stated intent -> query terms
      retrieve code    BM25 over policy articles, adaptive cut-off (1-2 hits)
      compose  LLM     the only model call; sparse context, JSON-only answer
      audit    code    schema + grounding check; one corrective LLM retry,
                       then fail safe to ESCALATE_TIER2

    Why not four LLM calls: the minimal JSON answer costs ~40 output tokens
    (~600 pts) per call, so four calls would spend ~2,400 pts on output
    alone, i.e. the whole 2,500 budget, before any input.

    Why the >200k-token help center is never sent: it is five policy
    articles plus thousands of "internal routing note" blocks that state
    they do not change customer policy. Retrieval indexes the policy
    articles (data/kb) once, as an offline indexing job would.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.index = policy_index()

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])

        tri = self._triage(ticket, order)
        retrieved = self._retrieve(tri)
        draft = self._compose(ticket_id, ticket, order, retrieved)
        return self._audit(ticket_id, ticket, order, retrieved, draft)

    # ---- node 1: triage (code) ------------------------------------------- #
    def _triage(self, ticket: dict, order: dict) -> Triage:
        return triage(ticket, order)

    # ---- node 2: retrieve (code) ----------------------------------------- #
    def _retrieve(self, tri: Triage) -> list[str]:
        return self.index.search(tri.query_terms)

    # ---- node 3: compose (the one LLM call) ------------------------------ #
    def _compose(
        self,
        ticket_id: str,
        ticket: dict,
        order: dict,
        retrieved: list[str],
        feedback: str | None = None,
    ) -> str:
        user = build_user_message(ticket_id, ticket, order, retrieved)
        if feedback:
            user += f"\nPrevious answer rejected: {feedback}."
        messages = [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]
        label = "compose_retry" if feedback else "compose"
        return self.llm.chat(label, messages, cached_prefix=f"SYSTEM:\n{SYSTEM_PROMPT}")

    # ---- node 4: audit (code, LLM only on failure) ----------------------- #
    @staticmethod
    def _validate(draft: str, retrieved: list[str]):
        try:
            res = parse_resolution(draft)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return None, f"unparseable JSON ({exc.__class__.__name__})"
        if res.action not in ACTIONS:
            return None, "action must be one of " + "|".join(ACTIONS)
        allowed = (*retrieved, DEFAULT_ARTICLE)
        if res.kb_article not in allowed:
            return None, "kb_article must be one of " + "|".join(dict.fromkeys(allowed))
        return res, None

    def _audit(
        self,
        ticket_id: str,
        ticket: dict,
        order: dict,
        retrieved: list[str],
        draft: str,
    ) -> TicketResolution:
        res, problem = self._validate(draft, retrieved)
        if res is None:
            retry = self._compose(ticket_id, ticket, order, retrieved, feedback=problem)
            res, problem = self._validate(retry, retrieved)
        if res is None:
            return TicketResolution("ESCALATE_TIER2", DEFAULT_ARTICLE)
        return TicketResolution(res.action, res.kb_article)
