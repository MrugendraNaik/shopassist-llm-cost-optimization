from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


ACTIONS = ("REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2")

# House convention: when no specific policy authorizes an automated action, the
# ticket is escalated citing the general returns policy. Stated once in the
# static system prompt (~8 tokens) instead of retrieving that article for every
# ticket (~34 tokens), and always accepted by the grounding check.
DEFAULT_ARTICLE = "returns-within-30-days"

# =========================================================================== #
# Context engineering helpers (shared with question2)
# =========================================================================== #

# Static instruction block. Written to be token-dense but unambiguous:
#  * a closed action set and a closed citation set (the POLICIES given),
#  * an explicit closed-world convention for the sparse snapshot,
#  * an explicit safe default, so the model never needs to "reason out loud".
# Deliberately free of any wording that invites long-form output.
SYSTEM_PROMPT = (
    "ShopAssist Tier-1 support. Pick action REFUND|REPLACE|SEND_ARTICLE|ESCALATE_TIER2 "
    "and the POLICIES id supporting it; if none applies, ESCALATE_TIER2 with "
    f"{DEFAULT_ARTICLE}. Unlisted flags are false.\n"
    "JSON only: action, kb_article, customer_reply"
)

# Snapshot fields the decision never depends on. Dropped before the LLM sees them.
_SNAPSHOT_DROP = {"order_id", "sku", "title"}


def compact_snapshot(order: dict) -> str:
    """
    Sparse, closed-world serialization of the order record.

    * Identifier fields (order_id, sku, title) carry no decision signal; the
      customer turn already names the product for the reply.
    * False booleans are omitted; SYSTEM_PROMPT states "flags not listed are
      false". Booleans are the majority of the record, so this halves it.
    * Compact separators: `indent=2` JSON spends ~1 token per newline/indent.
    """
    sparse = {
        k: v
        for k, v in order.items()
        if k not in _SNAPSHOT_DROP and v is not False and v is not None
    }
    return json.dumps(sparse, separators=(",", ":"))


def customer_text(ticket: dict) -> str:
    """Customer turns only; agent turns are acknowledgement boilerplate."""
    return " ".join(m["text"] for m in ticket["messages"] if m["from"] == "customer")


# ---- triage (code) --------------------------------------------------------- #

@dataclass
class Triage:
    intent: str
    query_terms: list[str] = field(default_factory=list)


def triage(ticket: dict, order: dict) -> Triage:
    """
    Deterministic triage from structured order facts plus the intent stated
    in the customer turns. Query terms are chosen per *situation* so the
    governing policy ranks first instead of every policy that mentions
    "delivery" or "refund".
    """
    text = customer_text(ticket).lower()
    if "replace" in text:
        intent = "replacement"
    elif "refund" in text or "cancel" in text:
        intent = "refund"
    else:
        intent = "other"

    ptype = order.get("product_type", "")
    terms: list[str] = [ptype]
    if ptype == "subscription":
        terms += ["subscription", "cancelled", "prorated", "refund"]
    if order.get("license_revealed"):
        terms += ["digital", "license", "revealed", "refundable"]
    if order.get("customized"):
        terms += ["custom", "personalized", "merchandise"]
    if order.get("damage_reported"):
        terms += ["damaged", "replacement", "catalog"]
    if ptype == "physical" and not order.get("damage_reported"):
        terms += ["physical", "refunded", "resalable", "returns"]
    if intent != "other":
        terms.append(intent)
    return Triage(intent, _tokens(" ".join(terms)))


# ---- retrieve (code) ------------------------------------------------------- #

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "for", "is", "are", "be", "may", "or", "and",
    "in", "on", "after", "within", "has", "been", "not", "follow", "item", "items",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def compact_article(article_id: str) -> str:
    """One line per article: no markdown title (repeats the id), no emphasis."""
    body = load_article(article_id)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
    return f"{article_id}: " + " ".join(lines).replace("**", "")


class PolicyIndex:
    """
    BM25 over policy articles with an adaptive cut-off.

    Instead of a fixed top-k, keep only articles scoring within GAP of the best
    hit: when retrieval is confident the model sees one policy; when two are
    close (e.g. damaged *custom* item: damage vs. custom policy) it sees both
    and makes the call. If nothing matches, fall back to the full policy set.
    """

    K1, B = 1.5, 0.75
    GAP = 0.6  # keep docs with score >= GAP * best
    MAX_K = 3

    def __init__(self) -> None:
        self.ids = article_ids()
        self.docs = {
            aid: _tokens(aid.replace("-", " ") + " " + load_article(aid)) for aid in self.ids
        }
        self.df = Counter(t for toks in self.docs.values() for t in set(toks))
        self.avgdl = sum(len(d) for d in self.docs.values()) / len(self.docs)

    def scores(self, terms: list[str]) -> dict[str, float]:
        n = len(self.ids)
        out = {}
        for aid, toks in self.docs.items():
            tf = Counter(toks)
            s = 0.0
            for t in set(terms):
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (self.K1 + 1) / (
                        tf[t] + self.K1 * (1 - self.B + self.B * len(toks) / self.avgdl)
                    )
            out[aid] = s
        return out

    def search(self, terms: list[str]) -> list[str]:
        sc = self.scores(terms)
        ranked = sorted(self.ids, key=lambda a: (-sc[a], a))
        best = sc[ranked[0]]
        if best <= 0:
            return list(self.ids)
        return [a for a in ranked[: self.MAX_K] if sc[a] >= self.GAP * best]


@lru_cache
def policy_index() -> PolicyIndex:
    return PolicyIndex()


def build_user_message(ticket_id: str, ticket: dict, order: dict, policies: list[str]) -> str:
    return "\n".join(
        [
            "POLICIES:",
            *(compact_article(a) for a in policies),
            f"Customer: {customer_text(ticket)}",
            f"ORDER_SNAPSHOT:{compact_snapshot(order)}END_SNAPSHOT",
        ]
    )


# =========================================================================== #
# Question 1 agent
# =========================================================================== #


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot (cost-optimized). Exactly one LLM call per ticket.

    Cost model: output tokens cost 5x input, and the simulator's answer is
    already minimal JSON (~40 tokens = ~600 pts), so after removing the
    verbose-reasoning trigger, the remaining lever is input. The prompt is
    assembled from four compressions:
      1. static, dense instruction block (no uuid, no few-shot)
      2. adaptive retrieval: only the policies the order facts point at
      3. sparse closed-world order snapshot
      4. customer turns only
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self.index = policy_index()

    def build_messages(self, ticket_id: str) -> tuple[list[ChatMessage], list[str]]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        policies = self.index.search(triage(ticket, order).query_terms)
        user = build_user_message(ticket_id, ticket, order, policies)
        return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)], policies

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages, policies = self.build_messages(ticket_id)
        completion = self.llm.chat(
            "handle_ticket", messages, cached_prefix=f"SYSTEM:\n{SYSTEM_PROMPT}"
        )
        res = parse_resolution(completion)
        if res.action not in ACTIONS or res.kb_article not in (*policies, DEFAULT_ARTICLE):
            # Ungrounded or malformed answer: never auto-act on it.
            return TicketResolution("ESCALATE_TIER2", DEFAULT_ARTICLE, "", completion)
        return res
