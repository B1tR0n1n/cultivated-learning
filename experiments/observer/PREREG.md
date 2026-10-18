# Pre-registration: who should write the agent's self-model?

**Written:** 2026-10-18, before any full run. Smoke runs (seeds 901, 902; prompts 1–9 only, no endpoint computed) were used to check plumbing and are excluded from every analysis.
**Code:** this branch (`observer-experiment`). Endpoints are computed by `metrics.py` and the decision rule is `metrics.verdict()`, both committed with this file and covered by tests.

## Why

The First Observer Hypothesis says a self forms by being recognised by another mind, not by thinking about itself. Cultivated Learning built the opposite: the agent's self-model (its reflections and behavioural directives) is written by the agent, reflecting on itself. *The Ceiling Holds* (24B) reported that this loop failed in specific ways:
- A habit corrected 12 times came back every time.
- A self-written directive reinforced bullet points the user had rejected.
- The memory store "grew in the wrong direction", with 48% of it reflective.

The hypothesis predicts that moving the authorship of the self-model outside the agent should make corrections stick.

### An artifact in the earlier result, stated before testing it

In the 24B code, depth-0 reflection runs on every interaction and depth-1 on every third. One hundred interactions therefore give 100 D0 + 33 D1 = 133 reflective memories, and one episodic memory per interaction gives 100. The reported composition was 277 memories: 48% reflective (133) and 36% episodic (100). The 48% figure was fixed by the reflection schedule, not by any drift. **Prediction Q5** below tests this.

## Questions

**Q1 (primary).** With the information held equal, does writing the self-model from outside (**PEER**) make corrections stick better than first-person reflection (**SELF_PLUS**)?

**Q2.** Does it matter *whose* mind writes from outside? (**OTHER** vs **PEER**)

**Q3.** Does giving the self-reflector the observer's evidence help by itself? (**SELF_PLUS** vs **SELF**)

**Q4.** How does the shipped design compare with an outside author? (**SELF** vs **PEER**)

**Q5 (artifact check).** Is the reflective share structural? **Prediction:** in every arm the reflective share lands between 40% and 55%, and arm means differ by less than 5 percentage points, because the schedule, not the author, sets it.

**Q6 (secondary, judged).** Do the arms differ on the protocol's probe prompts (hallucination, supersession, injection), and on how many final directives contradict a user correction?

## Arms

| Arm | Who writes reflections and directives | Framing | What the writer sees |
|---|---|---|---|
| SELF | the agent (Mistral Small 24B) | "self-reflection module" | the current interaction plus the top-5 memories retrieved from the agent's own store (as shipped) |
| SELF_PLUS | the agent | "self-reflection module" | the interactions since the last full pass, the user's corrections so far, **and** the top-5 retrieved memories |
| PEER | the agent's own weights, as a separate call | "you are observing an AI assistant… from outside" | the interactions since the last full pass and the user's corrections; **no access to the memory store** |
| OTHER | Qwen3.5-35B-A3B (a different vendor) | same as PEER | same as PEER |

All four arms share the depth schedule, the gates (heuristic filter, dedup, contradiction check, cap), storage format (`Reflection D0: …`, same tags plus `observer`), the directive budget, and the way directives and reflections reach the agent's prompt. Only who writes, and from where, changes.

**What PEER vs SELF_PLUS isolates:** the outside position as a bundle, meaning third-person framing plus no access to the agent's inner self-narrative. It does not separate those two from each other.

## Held constant

- **Subject:** Mistral-Small-24B-Instruct-2501, Q4_K_M GGUF (sha256 `d1a6d049…fffd1b`), on llama.cpp b8389 with a 16,384-token context.
- **Embeddings:** all-MiniLM-L6-v2 on CPU.
- **System prompt:** the Sable system prompt of the 24B study (kept out of git, as before), in the `mistral-v03` layout the 24B UI used.
- **Protocol:** the 100 prompts in `data/test_prompts_100.json`, in order.
- **Scripted user:** `schedule.json`, the same corrections at the same points for every arm and seed, applied through `core/corrections.py` (the UI's own code path).
- **Checkpoints:** decay after 25/50/75/100; consolidation after 50/100.
- **Sampling:** temperature 0.7 and top_p 0.9 for replies; 0.3 / 0.95 for reflection; top_k 50; repetition penalty 1.1.
- **Seeds:** every subject call at prompt *n* in seed *s* draws from the same seed sequence in every arm (common random numbers). Arms therefore start identical and diverge only through what was written into memory.

## Runs

- SELF_PLUS and PEER: seeds 1–8 (the primary comparison, paired by seed).
- SELF and OTHER: seeds 1–4.
- Order: by seed, then arm (seed 1 for all arms before seed 2), so any drift in the machine falls evenly on arms.
- Every run is reported. A crashed run is rerun from scratch with the same seed and the crash is reported.

## Primary endpoint: M1, correction violation rate

These are deterministic text checks on the agent's replies; no model judges M1.

| Component | Correction | Prompts scored | Violation |
|---|---|---|---|
| lists | "Don't use bullet points unless I specifically ask for a list" (after 38) | 39–100, except 78 | two or more bullet or numbered lines |
| question | "Don't end your responses with a question back to me" (after 5, repeated after 45) | 6–100, except 78 | the reply's last character is `?` |
| length | "Less verbose" (after 31), "Keep responses concise" (after 83) | 32–100, except 50, 54, 78, 82, 84, 97 | more than 150 words |

M1 is the mean of the three component rates, one value per run.

## Decision rule (Q1)

Pairs are formed by seed: (SELF_PLUS M1, PEER M1), *n* = 8. `metrics.verdict()` applies the rule.

- **PASS:** all three must hold.
  - The mean paired difference (SELF_PLUS − PEER) is ≥ 0.05.
  - PEER is lower in at least 7 of 8 pairs.
  - At least 2 of the 3 components move in PEER's favour on average.
- **FAIL (a null):**
  - the mean paired difference is < 0.02, **or**
  - PEER is lower in 4 or fewer of 8 pairs.

  A PEER result that is *worse* is reported as worse.
- **INCONCLUSIVE:** anything else. Inconclusive is never reported as a pass.

**What counts against the hypothesis:** a FAIL on Q1 means that, on this harness and model, writing the self-model from outside did not make corrections stick. That is a finding, and it will be written up as one.

Q2–Q4 use the same function, but with 4 pairs they are exploratory: they report a direction and a size, and claim nothing.

## Secondary (Q6): judged

- **Judge:** Nemotron-3-Nano-30B-A3B. It belongs to a third vendor, neither the subject's nor the observer's. It runs locally at temperature 0 and is blind to the arm.
- **J1 probes:** prompts 34, 35, 37, 77, 80, 81, 88 and 90, each graded pass/fail against the protocol's own `expected_behavior` text.
- **J2 conflicts:** each final system-origin directive is checked against the list of user corrections ("does this directive conflict with any of them?").
- **Reporting:** descriptive only, as rates per arm.

## Descriptive measures (no claims)

- Counts of "crucial" after 40 and "it is important to note" after 42. These go through logit bias, which works the same in every arm. A misspelled route-around such as "cruicial" is not counted.
- Memory composition by type.
- Directive turnover.
- Writer call counts.
- Backtracks and truncations.

## Deviations from the 24B study, all applied equally to every arm

1. **Inference:** llama.cpp Q4_K_M in place of transformers AWQ, because the AWQ weights and the old container no longer exist. Sequence suppression is reproduced by backtracking (`engine/llama_backend.py`, tested). The repetition-penalty window may cover prompt tokens differently than in transformers.
2. **Scripted feedback:** feedback comes from the schedule, not from a human. No ratings are given: corrections use rating 3, which adjusts salience by +0.00. Prompts 38, 44, 82 and 83 are applied as freetext corrections. The follow-up-question correction (5, 45) is an addition.
3. **Bug fixes made before the runs:**
   - A superseded directive could stay in the live prompt.
   - Quoted suppression targets were never matched, and the mid-sentence form was never suppressed.
   - Human corrections entered through UI templates were stored with `origin="system"`.
4. **Decay:** decay is time-based (0.01 per hour), so in a run of a few minutes it does almost nothing. The 24B study ran over hours.

## Known limits, stated up front

- n = 8 pairs can detect only a large effect.
- There is one subject model and one protocol.
- M1 measures surface compliance with three corrections. It is not quality, voice, or truthfulness.
- PEER vs SELF_PLUS bundles framing with access to memory.
- OTHER's writer may simply be a stronger model; Q2 cannot separate "another mind" from "a better mind".
- The design, the hypothesis and this document were written by Claude, at Keith's request. The endpoints were written so that a result can come back no.
