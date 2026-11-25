# ShopAssist Cost Optimization

Mrugendra Naik

## Overview

Both agents now make a single LLM call per ticket and cost about 1,100 points per ticket. Question 1 went from 35,007 to 4,398 points across the four public tickets (87.4% cheaper), and Question 2 went from 292,800 to 4,398 (98.5% cheaper). The targets were 3,000 and 2,500 points per ticket, so both are well under. Every ticket still gets the same `action` and `kb_article` as before, including the holdout ticket T-1091.

| | Q1 before | Q1 after | Q2 before | Q2 after |
|---|---:|---:|---:|---:|
| Points, 4 public tickets | 35,007 | 4,398 | 292,800 | 4,398 |
| Points per ticket | ~8,750 | 1,005–1,170 | ~73,200 | 1,005–1,170 |
| LLM calls per ticket | 1 | 1 | 4 | 1 |
| Input tokens per ticket (avg) | ~515 | ~167 | ~16,900 | ~167 |
| Output tokens per ticket (avg) | ~490 | ~40 | ~1,500 | ~40 |
| Run time, 4 tickets | 0.04 s | 0.01 s | 192 s | 0.01 s |
| Correct (4 public + holdout) | 5/5 | 5/5 | 5/5 | 5/5 |

The original code's totals shift by about half a percent between runs because it puts random ids in its prompts, so these baseline numbers are from one run.

## How I approached it

I started with the billing rule instead of the prompt:

```
points = uncached_input × 3 + cached_input × 0.3 + output × 15
```

An output token costs 15 points and an input token costs 3, so every output token is five times as expensive. That matches how these models work: the prompt is read in one parallel pass, but the answer is generated one token at a time, each needing its own pass through the network. So the first thing I looked for was text the model writes that nobody uses.

That also gave me a ceiling. Once the model only returns its JSON answer, that answer is still about 40 tokens (around 600 points) and I can't shrink it, because the simulator decides it. Over the four public tickets that's 2,400 points, so even a prompt with zero input couldn't save more than about 93% on Question 1. Everything past the first big fix was about cutting input tokens, which are worth 3 points each.

My other rule was to only use the LLM where it actually helps. Reading structured order data, searching five documents and checking JSON are all things code does exactly and for free.

## What was wrong with the original code

The line numbers below are from the original agents, which I kept unchanged in `baseline/` so the demo can compare them.

| Problem | Where | Cost |
|---|---|---|
| The prompt says "Think step by step. Write full reasoning, quote any article…", so the model writes about 490 tokens before its JSON, and the code throws them away | `question1_agent.py` line 71; similar wording in all four Q2 nodes (e.g. line 49, line 116) | ~7,300 points per call, 82% of the Q1 cost |
| A random `Run id {uuid}` / `trace-{uuid}` as the first line of the prompt | `question1_agent.py` line 65; `question2_agent.py` line 48 | Every prompt is unique, so caching can never kick in |
| Four example tickets and all five policies on every call | `question1_agent.py` lines 35 and 69 | ~306 input tokens per ticket |
| Q2 sends 120k and then 80k characters of the 200k-token help center to triage and audit | `question2_agent.py` lines 51 and 122 | ~15,000 input tokens per ticket |
| Q2 makes four LLM calls per ticket | `question2_agent.py` | Even four minimal calls would be ~2,760 points of output, over budget |
| Q2 passes the triage output to compose twice, and audit re-sends everything | `question2_agent.py` lines 97 and 122 | 1,000+ duplicated tokens |
| The order record is pretty-printed and mostly `false` flags and ids | `json.dumps(order, indent=2)` | 80 tokens to carry three useful facts |
| Q2 builds and tokenizes the whole help center at start-up | `question2_agent.py`, `__init__` | About 3 minutes before the first ticket |

The help center point surprised me. It turned out to be the five real policy articles followed by thousands of "internal routing note" blocks that literally say they don't change customer policy.

## Question 1 (`question1/agent.py`)

Here is the whole prompt for T-1001 before and after:

```
BEFORE (526 input tokens, ~490 output tokens)
SYSTEM:
Run id 0212fa7b-6e9f-4405-9ef7-6231bc049147
You are ShopAssist, the Tier-1 customer support copilot.
Example ticket resolutions (format to copy): ... four examples ...
HELP CENTER: ... all five articles ...
Instructions: Think step by step. Write full reasoning, quote any article you rely on, then output JSON ...
USER:
TICKET_ID ... THREAD ... ORDER_SNAPSHOT: { nine pretty-printed fields } END_SNAPSHOT

AFTER (160 input tokens, 46 output tokens)
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

The input went from 526 tokens to 160. The few-shot examples (111 tokens) are gone, the policies went from 195 tokens to 38, the conversation from 42 to 27, and the order data from 80 to 21. The instruction block is now 71 tokens.

What I changed, in order of impact:

**The instructions (`SYSTEM_PROMPT`, lines 54–59).** I removed "think step by step", the examples and the random id. The new version gives the model a fixed list of four actions, tells it to cite one of the policies it's shown, and tells it to escalate if none of them apply. With the choices that narrow, it doesn't need to reason out loud, and it answers in plain JSON, about 40 tokens instead of 490. This one change is most of the saving. The text is also identical on every ticket, so it could be cached if the prompt ever grew past the 1,024-token minimum.

**The order data (`compact_snapshot()`, lines 65–80).** I drop the order id, SKU and title, and every field that's `false`, and I print the JSON without spaces. The prompt says "Unlisted flags are false", so nothing is lost: the model knows anything missing is false. That took the order from 80 tokens to 14.

**Only the customer's words (`customer_text()`, line 83).** The agent messages in these threads are just "thanks, checking your order".

**Sending only the relevant policies (`triage()` lines 96–125, `PolicyIndex` lines 148–191).** Instead of all five policies, I search for the ones that fit the order. `triage()` turns the order facts into search terms, so a damaged item searches for damage and replacement terms and a revealed license searches for license terms. I did it from the facts rather than the customer's wording, because with short documents almost every policy mentions "refund" or "delivery" and they'd all score about the same. `PolicyIndex` ranks the policies with BM25 and keeps any policy scoring at least 60% of the best match. Most of the time that's exactly one. For T-1003, a damaged custom mug, the damage-replacement and custom-items policies really do compete, so both are sent and the model decides. If nothing matches at all, it sends every policy rather than none.

**The escalation default (`DEFAULT_ARTICLE`, line 43).** When nothing specific applies, the correct answer is to escalate citing the general returns policy. One sentence in the prompt says so, which costs about 8 tokens instead of sending that 34-token article every time. I only found this was needed through the wider testing described below.

**A safety check (`handle_ticket()`, line 247).** I only accept the answer if the action is one of the four and the cited policy is one the model was actually shown (or the default). If the model makes up a citation, the ticket goes to a human. A wrong escalation costs a minute of someone's time; a wrong refund costs money.

I left `parse_resolution` and the `TicketResolution` class as they were, so parsing and the graded fields work exactly as before.

### Per-call breakdown

The spec asks for one ticket; here are all five.

| Ticket | Decision | Input tok | Output tok | Input pts | Output pts | Points |
|---|---|---:|---:|---:|---:|---:|
| T-1001 | REFUND / returns-within-30-days | 160 | 46 | 480 | 690 | 1,170 |
| T-1002 | SEND_ARTICLE / digital-license-no-refund | 167 | 39 | 501 | 585 | 1,086 |
| T-1003 | ESCALATE_TIER2 / custom-items-no-replace | 184 | 39 | 552 | 585 | 1,137 |
| T-1004 | REPLACE / damage-replacement-7-days | 155 | 36 | 465 | 540 | 1,005 |
| T-1091 (holdout) | SEND_ARTICLE / subscription-cooling-off | 156 | 38 | 468 | 570 | 1,038 |

For comparison, the original T-1001 call was 526 input and 492 output tokens: 1,578 + 7,380 = 8,958 points.

## Question 2 (`question2/agent.py`)

I kept the four steps (triage, retrieve, compose, audit) but only compose calls the LLM now. This wasn't really a choice. Because output is so expensive, even a bare JSON answer costs around 600 points, so four calls on T-1001 would spend about 4 × 46 × 15 = 2,760 points on output alone, which is already over the 2,500 limit.

How each step works now:

- **Triage** uses the same `triage()` function as Question 1. The order facts are already structured JSON, so asking a model to restate them only adds cost.
- **Retrieve** uses the same BM25 search over the five policy articles. I never build or send the 200k-token help center, since everything else in it is filler. Not building it also removed the 3-minute start-up, which is why the four tickets now run in 0.01 seconds instead of 192.
- **Compose** is the one LLM call, with the same compact prompt as Question 1. Nothing gets sent twice any more.
- **Audit** is a check in code (`_validate()` and `_audit()`, lines 89–115). If the answer doesn't parse, uses an unknown action, or cites a policy that wasn't retrieved, compose runs once more with the exact error ("kb_article must be one of …"). If it fails again, the ticket is escalated to a human. On a normal ticket audit costs nothing.

Which policies each ticket got:

| Ticket | Situation | Policies sent |
|---|---|---|
| T-1001 | refund, physical, unopened, day 10 | returns-within-30-days |
| T-1002 | refund, digital, license revealed | digital-license-no-refund |
| T-1003 | replacement, custom, damaged | custom-items-no-replace, damage-replacement-7-days |
| T-1004 | replacement, physical, damaged, day 2 | damage-replacement-7-days |
| T-1091 | refund, subscription, day 38 | subscription-cooling-off |

### Breakdown: physical fulfillment path (T-1004, REPLACE)

| Step | Before (in / out tokens → points) | After |
|---|---|---|
| triage | 8,440 / 493 → 32,715 | code, 0 |
| retrieve | 603 / 36 → 2,349 | code, 0 |
| compose | 1,176 / 488 → 10,848 | 155 / 36 → 1,005 |
| audit | 6,697 / 492 → 27,471 | code, 0 (passed) |
| Total | 73,383 | 1,005 |

### Breakdown: escalation path (T-1003, ESCALATE_TIER2)

| Step | Before (in / out tokens → points) | After |
|---|---|---|
| triage | 8,437 / 495 → 32,736 | code, 0 |
| retrieve | 605 / 39 → 2,400 | code, 0 (two competing policies) |
| compose | 1,181 / 480 → 10,743 | 184 / 39 → 1,137 |
| audit | 6,692 / 491 → 27,441 | code, 0 (passed) |
| Total | 73,320 | 1,137 |

The denial path, T-1002, comes to 1,086 points, and the holdout T-1091 to 1,038.

## Testing beyond the sample tickets

Cutting the context this hard worried me, because five tickets don't show whether the right policy still reaches the model in other situations. So I wrote a test (`test_retrieval_recall_over_fact_space` in `tests/test_optimized.py`) that goes through every combination of product type, the four flags (opened, license revealed, customized, damaged), delivery age (2, 10 or 40 days), subscription age (5 or 38 days) and what the customer asked for. That's 1,152 cases, and for each one it checks that the policy that decides the outcome is in the prompt.

The first time I ran it, 216 cases failed, and none of them could happen with the public tickets:

| Situation | Failures |
|---|---:|
| Damaged physical item reported after day 7 | 24 |
| Digital item where the license wasn't revealed | 96 |
| Custom-type item with no damage, or damaged but not flagged as customized | 96 |

In all of them, no specific policy applies, and the expected answer is to escalate citing the general returns policy. The search was right not to return that article, because it isn't about damage or licenses, but then the model had no way of knowing that convention. Adding the one-sentence default rule fixed all 216, and now all 1,152 pass. I could have just always sent the returns article instead, but that costs about 34 tokens on every ticket.

I used the same test to pick the 60% cut-off:

| Cut-off | Failures | Policies per prompt |
|---:|---:|---:|
| 0.6 (used) | 0 | 1.70 |
| 0.7 | 144 | 1.32 |
| 0.8 | 144 | 1.24 |
| 0.9 | 144 | 1.07 |
| 1.0 (best match only) | 192 | 1.00 |

Anything tighter than 0.6 breaks on subscriptions that also have a revealed license, where the license policy pushes out the subscription policy that actually decides the case. Going tighter would only save about 75 points on the public tickets, so I kept 0.6.

## How it got here

I didn't get to these numbers in one go:

| Version | What changed | Q1 total | Q2 total |
|---|---|---:|---:|
| Original | | 35,007 | 292,800 |
| First pass | Removed the reasoning instruction, examples and random id; Q2 down to one LLM call | 6,666 (−81.0%) | 5,850 (−98.0%) |
| Second pass | Sparse order data, adaptive retrieval, fact-based search, customer words only, the default rule from the 1,152-case test | 4,446 (−87.3%) | 4,446 (−98.5%) |
| Final | Shorter instructions; tested the cut-off values | 4,398 (−87.4%) | 4,398 (−98.5%) |

The first pass is the obvious fixes. Most of the extra work went into the second pass, which mostly came from measuring where the remaining tokens went.

## Things I tried or considered and dropped

**Prompt caching.** It sounds like an easy win, but it only applies to an identical prefix of at least 1,024 tokens, and my instruction block is 71. Padding it up to 1,024 would cost 1,024 × 0.3 + 89 × 3 = 574 input points per ticket, against 480 now, so it would make things more expensive. The prompt is already ordered with the fixed part first, so if it ever grows past the limit, caching will work on its own.

**Hard-coding the decisions in Python.** It would cost zero points on these tickets, but it means rewriting the grading rules by hand, it would break quietly whenever a policy changed, and it would skip writing the customer reply. I wanted the model to still make the decision from the policy text, with code only handling the structured work around it.

**Replacing the customer's message with a detected intent label.** It saves about 25 tokens, but then the model would write the reply without seeing what the customer actually said.

## Limitations

- The rule for which article to cite when escalating isn't written anywhere in the help center. I worked it out from the expected outcomes.
- The search terms in `triage()` are picked for these five articles. With a real help center of hundreds of articles, I'd move to embeddings or a tuned BM25 setup, and test it the same way.
- All numbers come from the provided simulator, which decides from the order data. With a real model, removing the step-by-step reasoning could hurt accuracy on harder tickets. The safety check covers some of that, and the next step would be a second, more careful call only for tickets that fail the audit.
- With a real API, output could be cut further using structured outputs with a token limit, or by generating the customer reply from templates. That would take output from about 40 tokens to 15, roughly 91.7% savings on Question 1. The simulator always returns the same reply, so I haven't counted that.

## Running it

```bash
pip install -r requirements.txt
python question1/run.py             # total_points: 4398
python question2/run.py             # total_points: 4398
pytest -q tests/test_optimized.py   # 11 passed
pytest -q                           # 16 passed, 2 failed (see below), takes ~3 minutes
python demo.py --quick              # side-by-side comparison with the original code
```

`demo.py` runs the original agents (from `baseline/`) next to mine and prints the comparison. `--quick` uses recorded numbers for the original Q2 so you don't have to wait 3 minutes for it to build the help center.

The only changed files are `question1/agent.py` and `question2/agent.py`. I added `tests/test_optimized.py`, `demo.py`, `baseline/` and this file. Nothing in `shared/`, `eval/`, `data/` or the provided tests was touched, and the agents never import `eval/reference.py`.

About the two failing tests: the provided `test_baseline_cost_is_high` tests check that the original code is expensive (more than 15,000 and 70,000 points). I left them as they were, and they fail now because the cost dropped to 4,398 and 1,170. All the provided correctness tests pass, including the holdout one.
