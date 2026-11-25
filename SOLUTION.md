# ShopAssist cost optimization: solution write-up

## Executive summary

| Metric | Q1 baseline | Q1 optimized | Q2 baseline | Q2 optimized |
|---|---:|---:|---:|---:|
| Points, 4 public tickets | 35,007 | **4,398** | 292,800 | **4,398** |
| Reduction | | **−87.4%** | | **−98.5%** |
| Points per ticket (target) | ~8,750 | **1,005–1,170** (≤ 3,000) | ~73,200 | **1,005–1,170** (≤ 2,500) |
| LLM calls per ticket | 1 | 1 | 4 | **1** |
| Input tokens per ticket (avg) | ~515 | **~167** | ~16,900 | **~167** |
| Output tokens per ticket (avg) | ~490 | **~40** | ~1,500 | **~40** |
| Wall-clock, 4 tickets (local simulator) | 0.04 s | 0.01 s | 192 s | **0.01 s** |
| Graded fields correct (4 public + holdout T-1091) | 5 / 5 | **5 / 5** | 5 / 5 | **5 / 5** |
| Retrieval recall over all 1,152 order situations | n/a | **1,152 / 1,152** | n/a | **1,152 / 1,152** |

Points formula: `uncached_input × 3 + cached_input × 0.3 + output × 15`.

`action` and `kb_article` are unchanged on T-1001–T-1004, and the holdout T-1091 (subscription cancelled on day 38) resolves correctly in both tracks. Neither agent imports `eval/reference.py`; it is used only in tests. Baseline totals vary by about ±0.5% between runs, because the baseline's random uuids tokenize differently and change its verbose output length; the figures above are from one measured run.

On speed: the Q2 baseline spends about 3 minutes building and tokenizing the 200k-token help center before the first ticket; the optimized pipeline never builds it. Against a real API, latency would also fall sharply, since generation time scales with output tokens: Q2 goes from about 1,500 to about 40 generated tokens per ticket (≈ 37× fewer) and from 4 sequential calls to 1. The simulator's own wall-clock for Q1 is too small to be meaningful.

### Per-ticket results (optimized, both tracks)

| Ticket | Graded output | Input tok | Output tok | Points |
|---|---|---:|---:|---:|
| T-1001 | REFUND / returns-within-30-days | 160 | 46 | 1,170 |
| T-1002 | SEND_ARTICLE / digital-license-no-refund | 167 | 39 | 1,086 |
| T-1003 | ESCALATE_TIER2 / custom-items-no-replace | 184 | 39 | 1,137 |
| T-1004 | REPLACE / damage-replacement-7-days | 155 | 36 | 1,005 |
| T-1091 (holdout) | SEND_ARTICLE / subscription-cooling-off | 156 | 38 | 1,038 |

Both tracks end up with the same single model call per ticket, built by the same context layer (in `question1/agent.py`, imported by `question2/agent.py`), so the per-ticket numbers match.

## What was wrong with the baseline

Line numbers refer to the untouched originals kept in `baseline/`.

| # | Problem | Where | Cost impact | Fix |
|---|---|---|---|---|
| 1 | Prompts that invite long output ("think step by step", "full reasoning", "restate", "double-check") | `question1_agent.py` l.71; `question2_agent.py` l.49, l.116 and compose prompt | ~490 output tokens per call ≈ 7,300 pts, 82% of Q1 cost | Constrained, JSON-only instruction block (§2a) |
| 2 | Random uuid as the first line of the prompt | `question1_agent.py` l.65; `question2_agent.py` l.48 | Every prefix unique, so no cache hit is ever possible | Static, byte-identical system prefix (§2b) |
| 3 | Few-shot block plus all five articles on every call | `question1_agent.py` l.35, l.69 | ~200 input tokens per ticket | Adaptive retrieval sends 1.7 policies on average (§2d) |
| 4 | 120k + 80k characters of the 200k-token help center sent to triage and audit | `question2_agent.py` l.51, l.122 | ~8,400 + ~6,700 input tokens per ticket | Index the five policy articles; never send the blob (§3) |
| 5 | Four LLM calls per ticket | `question2_agent.py` | 4 × the output floor; ~2,760 pts of output alone even if made minimal | One LLM call; triage, retrieve and audit in code (§3) |
| 6 | Context snowballing: triage passed twice, audit re-sends everything | `question2_agent.py` l.97, l.122 | ~1,000+ duplicated tokens per ticket | Only compose sees model context, once (§3) |
| 7 | Pretty-printed order snapshot full of false flags and ids | both, `json.dumps(order, indent=2)` | 80 tokens for 3 useful facts | Sparse closed-world snapshot, 14 tokens (§2c) |
| 8 | Help-center blob built and tokenized at pipeline start-up | `question2_agent.py` `__init__` | ~3 minutes before the first ticket | Not built at all (§3) |

## How the solution evolved

| Version | Main changes | Q1 total (4 tickets) | Q1 reduction | Q2 total (4 tickets) | Q2 reduction |
|---|---|---:|---:|---:|---:|
| Baseline | Original code | 35,007 | | 292,800 | |
| v1 | Verbose triggers, few-shot and uuid removed; compact JSON; Q2 down to one LLM call with BM25 top-3 retrieval and a code audit | 6,666 | −81.0% | 5,850 | −98.0% |
| v2 | Sparse closed-world snapshot, adaptive-k retrieval, situation-based query terms, customer turns only, default-citation rule (found by the recall sweep) | 4,446 | −87.3% | 4,446 | −98.5% |
| v3 (final) | Tighter instruction wording, retrieval cut-off sweep confirming 0.6 | **4,398** | **−87.4%** | **4,398** | **−98.5%** |

## Files changed

| File | Status | What it does |
|---|---|---|
| `question1/agent.py` | modified | Q1 agent plus the shared context layer: `SYSTEM_PROMPT`, `compact_snapshot()`, `triage()`, `PolicyIndex` (BM25, adaptive k), grounding guard |
| `question2/agent.py` | modified | Q2 pipeline: code triage and retrieve, one LLM compose, code audit with one corrective retry and fail-safe escalation |
| `tests/test_optimized.py` | new | Correctness and per-ticket targets on all tickets in both tracks, single-call property, audit retry path, 1,152-case recall sweep |
| `baseline/` | new | Untouched copies of the original agents, used only by `demo.py` for live before/after numbers |
| `demo.py` | new | Colored before/after terminal report for the demo |
| `SOLUTION.md` | new | This write-up |

`shared/`, `eval/`, `data/` and the provided tests are unchanged.

---

## 1. Start from the cost model, not the prompt

Output is priced at 15 and input at 3, so **one output token costs as much as five input tokens**. The baseline spent 82% of its Q1 points on output: the prompt said *"think step by step, write full reasoning, quote any article"*, and the model produced about 490 tokens of trace that the product parses and then discards.

Once that trigger is removed, the answer is a fixed JSON object of about 40 tokens (about 600 points). That gives a **hard floor**:

```
output floor, Q1 public set = 160 output tokens × 15 = 2,400 pts → 93.1% is the ceiling with zero input
```

Everything after step 1 is therefore about input, and every input token removed is worth 3 points. The final prompt is about 160 tokens, down from 526, so output is now 55% of the remaining cost. That is the expected end state of a well-optimized classification call.

## 2. The prompt, token by token

The complete prompt for T-1001 (160 tokens):

```
SYSTEM:
ShopAssist Tier-1 support. Pick action REFUND|REPLACE|SEND_ARTICLE|ESCALATE_TIER2 and the POLICIES
id supporting it; if none applies, ESCALATE_TIER2 with returns-within-30-days. Unlisted flags are false.
JSON only: action, kb_article, customer_reply

USER:
POLICIES:
returns-within-30-days: Physical items in resalable condition may be refunded within 30 days of delivery. ...
Customer: Hi, I'd like a refund for the wireless headphones. They arrived 10 days ago and I never opened the box.
ORDER_SNAPSHOT:{"product_type":"physical","delivered_days_ago":10}END_SNAPSHOT
```

Each piece comes from a specific technique:

**(a) Constrained decision instead of open reasoning (output side).** The model gets a closed action set, a closed citation set and an explicit default, so it never has to reason out loud. It just fills slots. The instruction block contains none of the phrasings that invite long output ("step by step", "restate", "double-check", "quote the…").

**(b) Static prefix first, variables last.** The baseline began with `Run id {uuid4()}`, which makes every prompt prefix unique and so blocks any cache hit. The system block is now byte-identical across tickets and passed as `cached_prefix`. It is too short to reach the 1024-token cache threshold; see §5 for why padding it to qualify would *raise* cost.

**(c) Sparse, closed-world snapshot.** Most of the order record is `false` flags plus identifiers that never affect a decision. The serializer:
- drops `order_id`, `sku` and `title` (the customer turn already names the product, for the reply);
- drops every `false`/`null` field, with a single system-prompt rule, *"Unlisted flags are false"* (5 tokens), that makes the omission lossless. This is a closed-world assumption, the same idea as sparse feature encoding;
- uses compact JSON separators (`indent=2` spends about one token per line on newlines and indentation).

The T-1001 snapshot goes from 80 tokens (baseline `indent=2`) to 14.

**(d) Adaptive-k retrieval.** BM25 over the policy articles, but instead of a fixed top-k the index keeps only hits scoring within 60% of the best one. When the order facts point at one policy, the model sees one policy. When two policies really compete, as with T-1003's *damaged custom* mug (the damage-replacement and custom-items policies), both survive the cut-off and the model makes the call. On average 1.7 policies are sent instead of 5 (Q1) or about 8,000 tokens of help center (Q2).

**(e) Query construction from structured facts, not prose.** Triage turns the order flags plus the customer's stated intent into retrieval terms per *situation*. A damaged item queries damage/replacement terms; an undamaged physical item queries returns terms. This avoids a classic BM25 failure on short documents, where every policy that mentions "delivery" or "refund" scores about the same.

**(f) A default citation instead of a default document.** See §4. One sentence in the system prompt replaces sending a 34-token article on every ticket.

**(g) Customer turns only.** Agent turns in these threads are acknowledgement boilerplate ("Thanks for reaching out — checking your order now.").

**(h) Grounding guard.** The output is accepted only if `action` is in the allowed set and `kb_article` is either one of the policies actually shown to the model or the stated default. Otherwise Q1 fails safe to `ESCALATE_TIER2`; Q2's audit first retries once, telling the model exactly what was wrong.

### Q1 per-call breakdown: T-1001

| | Uncached input | Cached input | Output | Points |
|---|---:|---:|---:|---:|
| Baseline `handle_ticket` | 526 × 3 = 1,578 | 0 | 492 × 15 = 7,380 | **8,958** |
| v1 (verbosity removed) | 366 × 3 = 1,098 | 0 | 46 × 15 = 690 | 1,788 |
| **Final** | 160 × 3 = 480 | 0 | 46 × 15 = 690 | **1,170** |

## 3. Q2: one LLM call in a four-node pipeline

The node structure stays; each node uses the cheapest mechanism that can do its job.

| Node | Mechanism | Why |
|---|---|---|
| triage | code | The facts are already structured JSON; asking a model to restate them adds only cost. |
| retrieve | code: BM25 + adaptive k | A search problem over a fixed corpus; an index is cheaper and deterministic. |
| compose | **LLM (the only call)** | The one step that needs language understanding plus a drafted customer reply. |
| audit | code; LLM retry only on failure | Schema and grounding checks are exact in code. One corrective retry, then fail safe to Tier 2. |

**Why not four lean LLM calls?** Each call's minimal JSON answer costs about 540–690 points no matter how short the prompt is. Four calls on T-1001 would cost about 2,760 points of output alone, already over the 2,500 budget. Fewer calls is not a style choice here; the arithmetic requires it.

**The 200k-token help center.** It is the five policy articles followed by thousands of "internal routing note" blocks, which state that they do not change customer policy. The baseline sent 120k characters of it to triage and 80k to audit. The pipeline indexes the policy articles (`data/kb`) once, as an offline indexing job would, and never sends the blob. A side benefit: the pipeline no longer builds the blob at startup, which took minutes to tokenize.

**Context snowballing removed.** The baseline's compose step received the triage output twice ("TRIAGE AGAIN"), and audit re-sent everything plus the help-center slice. Only compose sees model context now, and only once.

### Q2 per-call breakdown: T-1004 (physical fulfillment, REPLACE)

| Node | Baseline in / out → pts | Final |
|---|---|---|
| triage | 8,440 / 493 → 32,715 | code, 0 |
| retrieve | 603 / 36 → 2,349 | code, 0 (1 policy retrieved) |
| compose | 1,176 / 488 → 10,848 | 155 × 3 + 36 × 15 = **1,005** |
| audit | 6,697 / 492 → 27,471 | code, 0 (passed) |
| **Total** | **73,383** | **1,005** (−98.6%) |

### Q2 per-call breakdown: T-1003 (escalation, damaged custom item)

| Node | Baseline in / out → pts | Final |
|---|---|---|
| triage | 8,437 / 495 → 32,736 | code, 0 |
| retrieve | 605 / 39 → 2,400 | code, 0 (2 competing policies retrieved) |
| compose | 1,181 / 480 → 10,743 | 184 × 3 + 39 × 15 = **1,137** |
| audit | 6,692 / 491 → 27,441 | code, 0 (passed) |
| **Total** | **73,320** | **1,137** (−98.4%) |

Denial path, for reference: T-1002 (SEND_ARTICLE / digital-license-no-refund) = 1,086 points.

## 4. Generalization check: retrieval recall over the whole fact space

Five sample tickets say little about whether aggressive pruning is safe. `tests/test_optimized.py::test_retrieval_recall_over_fact_space` therefore enumerates **1,152 combinations** of product type × opened × license revealed × customized × damaged × delivery age (2/10/40 days) × subscription age (5/38 days) × stated intent (refund / replacement / other), and asserts that the governing policy is in front of the model for every one.

The first run of this test found **216 misses** that the public tickets never hit. Each was the "nothing specific applies" case (for example, a damaged item reported on day 10), where the correct output is `ESCALATE_TIER2` citing the general returns policy. Retrieval was correctly *not* surfacing that article, because it isn't about damage, so the model had no way to know the house convention. The fix is to state the convention once in the static prompt (≈ 8 tokens) and accept that id in the grounding check. The alternative, always retrieving the returns article, costs about 34 tokens on every ticket. **Recall is now 1,152 / 1,152** with an average of 1.7 policies per prompt.

## 5. Things evaluated and rejected

- **Padding the prefix to 1,024 tokens to unlock cached pricing.** The static prefix is 71 tokens. Padding it to 1,024 and paying the cached rate costs 1,024 × 0.3 = 307 points, plus 89 × 3 = 267 for the user part, for a total of 574 input points, against 480 today. That is worse even under this simulator, which charges the cached rate from the very first call; a real provider also charges a cache write and expires idle entries. Caching pays off when the *useful* static context is already large: a few thousand tokens of policies, tools or few-shot examples shared across calls. Then the right move is to put that block first and keep it byte-stable, which the prompt layout already does.
- **Hard-coding the decision table in Python.** That would cost 0 points on these tickets, but it means re-implementing the grader's rules, it breaks silently whenever policy text changes, and it skips drafting the customer reply. The model stays the decision-maker, grounded in retrieved policy text, while code does only what code is exact at: structuring, search and validation.
- **A tighter retrieval cut-off.** Raising the cut-off from 0.6 to 0.7 or more would send one policy in more cases, but the recall sweep drops to 1,008 / 1,152 (144 misses, all subscriptions that also carry `license_revealed`, where the digital-license policy crowds out the subscription policy that actually governs). The saving would be about 75 points on the public set, so 0.6 stays.
- **Replacing the customer's message with a code-extracted intent label.** That saves about 25 tokens, but the model would then draft the reply blind to what the customer actually wrote. Not worth 75 points.

## 6. Production notes (beyond what the simulator measures)

- **Output is now the main cost (55%)**, and on a real API it can be cut further. Use structured outputs / a JSON-schema response format plus a `max_tokens` cap of about 60, or have the model return only `{action, kb_article}` and render `customer_reply` from per-action templates in code. That drops output from about 40 to about 15 tokens, about −375 points per ticket, which would take Q1 to about 91.7% and Q2 to about 99%. The simulator always emits the reply, so this isn't claimed in the numbers above.
- **Less chain of thought is a real trade-off on hard tickets.** The mitigations are explicit, structured facts, narrowed policy choice, and a grounding audit that escalates when the answer is unsupported. If accuracy dropped on messier traffic, the better pattern is a cascade, with a reasoning call only for low-confidence or audit-failed tickets, rather than paying for reasoning on every ticket.
- **Volume levers** if traffic grows: batch APIs (typically about 50% off for non-urgent tickets) and a semantic cache keyed on the decision-relevant fact signature, so repeat situations skip the model for the decision.

## Limitations (stated openly)

- **The escalation default citation is inferred, not documented.** The help-center articles do not say which article to cite when escalating because no policy applies. The convention (cite `returns-within-30-days`) was inferred from the expected outcomes and encoded as a house rule. It affects none of the public tickets; it only matters for unseen edge cases such as a damaged item reported after day 7.
- **Triage query terms are hand-picked for this five-article corpus.** For a real help center with hundreds of articles, the term lists would be replaced by learned retrieval (embeddings or BM25 with field boosts) evaluated with the same recall-sweep methodology.
- **All numbers come from the provided simulator.** It decides from the order snapshot, so prompt changes affect only cost there. Against a real model, the compact prompt and the grounding audit would need re-validation on a labeled ticket set.

## Demo

`python demo.py` prints a colored before/after report: billing rule, per-ticket savings bars, the T-1001 per-call breakdown, the actual prompts side by side with annotations, the Q2 node-by-node table for T-1004 and T-1003, correctness on all five tickets, and the 1,152-case recall check. The untouched original agents live in `baseline/` so the "before" numbers are measured live. `--quick` uses recorded Q2 baseline numbers to skip the 2–3 minute help-center build; `--section q1|prompts|q2|recall|summary` runs one part; `--slow` adds pauses for screen recording.

## How to run

```bash
pip install -r requirements.txt
python question1/run.py          # total_points: 4398
python question2/run.py          # total_points: 4398
pytest -q tests/test_optimized.py   # 11 passed, under a second
pytest -q                        # 16 passed, 2 failed by design (see below), ~3 min
python demo.py --quick           # before/after report
```

## Tests

```
pytest -q   →  16 passed, 2 failed
FAILED tests/test_question1.py::test_baseline_cost_is_high   (assert 4398 > 15000)
FAILED tests/test_question2.py::test_baseline_cost_is_high   (assert 1170 > 70000)
```

The two failures are the provided guards asserting that the *unoptimized* baseline is expensive, so they fail once cost goes down. All correctness tests pass, including `test_holdout_subscription` and `test_help_center_exceeds_context`. The new `tests/test_optimized.py` covers correctness and per-ticket targets on all tickets in both tracks, the single-call property, the audit retry/fail-safe path, and the 1,152-case retrieval recall sweep.
