from __future__ import annotations

import pytest

from eval.reference import TICKET_EXPECTATIONS
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]
ALL = PUBLIC + ["T-1091"]


@pytest.mark.parametrize("tid", ALL)
def test_q1_correct_and_under_target(tid):
    copilot = SupportCopilot()
    got = copilot.handle_ticket(tid)
    exp = TICKET_EXPECTATIONS[tid]
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)
    assert len(copilot.session.calls) == 1
    if tid in PUBLIC:
        assert copilot.session.total_points() <= 3000


@pytest.mark.parametrize("tid", ALL)
def test_q2_correct_and_under_target(tid):
    pipe = SupportPipeline()
    got = pipe.handle_ticket(tid)
    exp = TICKET_EXPECTATIONS[tid]
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)
    if tid in PUBLIC:
        assert pipe.session.total_points() <= 2500


def test_q2_audit_retries_then_fails_safe(monkeypatch):
    pipe = SupportPipeline()
    calls = []

    def bad_compose(*args, **kwargs):
        calls.append(kwargs.get("feedback"))
        return '{"action": "REFUND", "kb_article": "made-up-article"}'

    monkeypatch.setattr(pipe, "_compose", bad_compose)
    got = pipe.handle_ticket("T-1001")
    assert len(calls) == 2 and calls[1] is not None  # one corrective retry
    assert got.action == "ESCALATE_TIER2"


def test_retrieval_recall_over_fact_space():
    """
    Every combination of order facts x stated intent must put the governing
    policy in front of the model (or be the escalation default, which the
    system prompt names). The public tickets cover 5 of these 1,152 cases.
    """
    import itertools

    from eval.reference import decide_order
    from question1.agent import DEFAULT_ARTICLE, policy_index, triage

    idx = policy_index()
    msgs = ["I want a refund", "Please send a replacement", "Help with my order"]
    misses = []
    for pt, opened, lic, cust, dmg, days, sub, msg in itertools.product(
        ["physical", "digital", "custom", "subscription"],
        [False, True], [False, True], [False, True], [False, True],
        [2, 10, 40], [5, 38], msgs,
    ):
        order = dict(product_type=pt, opened=opened, license_revealed=lic,
                     customized=cust, damage_reported=dmg, delivered_days_ago=days)
        if pt == "subscription":
            order["subscription_age_days"] = sub
        ticket = {"messages": [{"from": "customer", "text": msg}]}
        got = idx.search(triage(ticket, order).query_terms)
        exp = decide_order(order).kb_article
        if exp not in (*got, DEFAULT_ARTICLE):
            misses.append((order, msg, got, exp))
    assert not misses, misses[:3]
